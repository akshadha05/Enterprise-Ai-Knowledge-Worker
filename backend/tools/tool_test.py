"""
Standalone test of tool-calling -- separate from the RAG pipeline for now.

Why separate: we want to prove "the LLM correctly decides when to call a
tool, with the right arguments" in isolation, before combining it with
document search and memory. That combination is Phase 7 (the
orchestrator) -- next step after this one.

Usage:
    python -m backend.tools.tool_test
"""

import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from backend.common.resilience import call_with_retry

from .actions import AVAILABLE_TOOLS

load_dotenv()

# Defaults to Flash-Lite, which historically gets the most generous free
# daily quota. Override via GEMINI_MODEL in .env if you want to try others.
MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")

SYSTEM_INSTRUCTION = """You are an office assistant for a company. You can send emails, create tickets, and schedule meetings using the tools available to you.

Use a tool whenever the user's request requires taking one of these actions. If the user is just asking a question or chatting, respond normally without calling a tool."""


def main():
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    print("Tool-calling test. Try things like:")
    print('  "send an email to priya@company.com about tomorrow\'s meeting being moved to 3pm"')
    print('  "create a high priority ticket: login page is throwing a 500 error"')
    print('  "schedule a meeting with the design team for Aug 10th at 2pm about the new logo"')
    print("(or 'quit' to exit)\n")

    while True:
        user_input = input("> ").strip()
        if user_input.lower() in ("quit", "exit"):
            break
        if not user_input:
            continue

        try:
            response = call_with_retry(
                client.models.generate_content,
                model=MODEL,
                contents=user_input,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    tools=AVAILABLE_TOOLS,  # <- the SDK handles calling these automatically
                ),
            )
            print(f"\nAssistant: {response.text}\n")
        except RuntimeError as e:
            # All retries exhausted -- fail gracefully instead of a raw traceback.
            print(f"\n[Sorry, couldn't complete that right now: {e}]\n")


if __name__ == "__main__":
    main()
