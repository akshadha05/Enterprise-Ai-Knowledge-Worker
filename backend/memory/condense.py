"""
Rewrites a follow-up question into a standalone one, using recent
conversation history.

Why this needs to be its own step: our vector search (embeddings.py)
has NO memory of its own -- it only ever sees the exact text you hand it.
If a user asks "what was the F1 score?" right after discussing ChurnWise,
searching for literally "what was the F1 score?" will likely miss,
because the document doesn't repeat "F1 score" next to enough surrounding
context in a way that would score high on its own.

So before searching, we ask the LLM to do a small, cheap rewrite:
"what was the F1 score?" + (recent history mentions ChurnWise)
    -> "What was the F1 score of the ChurnWise project?"
THIS rewritten version is what actually gets embedded and searched.
"""

from .conversation_memory import ConversationMemory

CONDENSE_PROMPT_TEMPLATE = """Given this recent conversation:

{history}

And this follow-up question from the user: "{question}"

Rewrite the follow-up question as a fully standalone question that includes any necessary context from the conversation (e.g. replace "it", "her", "that project" with the actual subject). If the follow-up question is already standalone and needs no context, return it unchanged.

Output ONLY the rewritten question, nothing else -- no preamble, no explanation."""


def condense_question(question: str, memory: ConversationMemory, llm) -> str:
    if memory.is_empty():
        # No history yet -- nothing to resolve against, so use as-is.
        return question

    prompt = CONDENSE_PROMPT_TEMPLATE.format(
        history=memory.format_recent(max_turns=4),
        question=question,
    )

    rewritten = llm.complete(prompt).strip()
    return rewritten if rewritten else question
