"""
Turns retrieved chunks + a question into an actual written answer.

Like embeddings.py, this is a swappable-provider setup: LLM_PROVIDER in
.env decides whether we use Anthropic's Claude (paid, higher quality) or
Google's Gemini (free tier available, no credit card needed).

The critical design choice here: we do NOT ask the LLM to "remember" or
reproduce exact quotes itself. LLMs can subtly misquote text. Instead:

1. The LLM only writes a SHORT, synthesized answer using the retrieved
   chunks as its only source of truth (this is the "generalized answer"
   part of what you asked for).
2. Separately, WE (plain Python, not the LLM) display the raw, unedited
   chunk text as the "source excerpt" -- guaranteed character-for-character
   identical to the original document. This is the "exact lines from the
   document" part.

This split is what makes the system trustworthy: the answer is readable,
the proof is exact.

Every actual network call below goes through call_with_retry (see
backend/common/resilience.py) so that a rate limit or a brief server
hiccup gets retried automatically instead of crashing the program.
"""

import os
from abc import ABC, abstractmethod

from backend.common.resilience import call_with_retry

SYSTEM_PROMPT = """You are an enterprise knowledge assistant. You answer questions using ONLY the provided context below -- never your own general knowledge.

Rules:
- Base your answer strictly on the context. Do not add facts that aren't there.
- Keep the answer short and to the point (2-4 sentences), written naturally -- not a copy-paste of the context.
- If the context does not contain the answer, say clearly: "I couldn't find this in the provided documents." Do not guess.
- Do not fabricate names, numbers, or dates that are not present in the context."""


def _build_context(matches: list[dict]) -> str:
    blocks = []
    for i, m in enumerate(matches, start=1):
        blocks.append(
            f"[Context {i} -- source: {m['source_filename']}, page {m['page_number']}]\n{m['text']}"
        )
    return "\n\n".join(blocks)


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, question: str, matches: list[dict]) -> str:
        raise NotImplementedError

    @abstractmethod
    def complete(self, prompt: str) -> str:
        """A plain, general-purpose completion -- no RAG grounding rules
        attached. Used for smaller utility tasks like rewriting a
        follow-up question into a standalone one."""
        raise NotImplementedError


class AnthropicLLM(LLMProvider):
    """Claude via the Anthropic API. Paid (needs API credits)."""

    def __init__(self, model: str = "claude-sonnet-5"):
        from anthropic import Anthropic

        self.client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model

    def generate(self, question: str, matches: list[dict]) -> str:
        if not matches:
            return "I couldn't find this in the provided documents."

        context = _build_context(matches)

        def _call():
            response = self.client.messages.create(
                model=self.model,
                max_tokens=400,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}],
            )
            return response.content[0].text

        return call_with_retry(_call)

    def complete(self, prompt: str) -> str:
        def _call():
            response = self.client.messages.create(
                model=self.model,
                max_tokens=150,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text

        return call_with_retry(_call)


class GeminiLLM(LLMProvider):
    """
    Google Gemini via Google AI Studio. Free tier available -- no credit
    card, no expiration, rate-limited (fine for development/learning).
    Get a key at https://aistudio.google.com/apikey

    Model defaults to the Flash-Lite family, which historically gets the
    most generous free-tier daily quota of the Gemini lineup. You can
    override this via GEMINI_MODEL in .env if you want to try others.
    """

    def __init__(self, model: str | None = None):
        from google import genai

        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")

    def generate(self, question: str, matches: list[dict]) -> str:
        if not matches:
            return "I couldn't find this in the provided documents."

        context = _build_context(matches)
        prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {question}"

        def _call():
            response = self.client.models.generate_content(model=self.model, contents=prompt)
            return response.text

        return call_with_retry(_call)

    def complete(self, prompt: str) -> str:
        def _call():
            response = self.client.models.generate_content(model=self.model, contents=prompt)
            return response.text

        return call_with_retry(_call)


def get_llm() -> LLMProvider:
    """Reads LLM_PROVIDER from environment and returns the right one."""
    provider = os.environ.get("LLM_PROVIDER", "gemini").lower()

    if provider == "anthropic":
        return AnthropicLLM()
    elif provider == "gemini":
        return GeminiLLM()
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}")


# Kept for backwards compatibility with earlier code -- routes through
# whichever provider is configured.
def generate_grounded_answer(question: str, matches: list[dict]) -> str:
    return get_llm().generate(question, matches)
