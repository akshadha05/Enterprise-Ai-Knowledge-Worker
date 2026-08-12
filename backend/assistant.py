"""
THE ORCHESTRATOR (Phase 7): one assistant with access to everything --
document search, conversation memory, AND real actions (email, tickets,
meetings) -- deciding per message what's actually needed.

This is the payoff of everything built so far. Compare this file to
backend/rag/ask.py (documents + memory, no actions) and
backend/tools/tool_test.py (actions only, no documents): this file
merges both into one assistant, using Gemini's `Chat` object, which:

  1. Automatically remembers the conversation across turns (no need for
     our own ConversationMemory/condense_question -- the chat session
     itself carries history, and Gemini resolves follow-ups using it).
  2. Automatically decides, per message, whether to call
     search_knowledge_base, send_email, create_ticket, schedule_meeting,
     some combination of them in sequence, or none at all.

This is what "AI agent" / "orchestration" actually means in practice:
one model, a toolbox, and a running conversation -- not a hand-coded
if/else deciding what to do.

Usage:
    python -m backend.assistant
"""

import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from backend.common.resilience import call_with_retry
from backend.rag.embeddings import get_embedder
from backend.rag.vector_store import VectorStore
from backend.tools.actions import create_ticket, list_recent_actions, schedule_meeting, send_email
from backend.tools.search_tool import make_search_tool

load_dotenv()

MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")
PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "./backend/data/vector_store")

SYSTEM_INSTRUCTION = """You are an enterprise AI knowledge worker -- a digital employee for this company.

You have access to these capabilities:
- search_knowledge_base: look up information in the company's internal documents
- send_email, create_ticket, schedule_meeting: take real actions
- list_recent_actions: look up tickets/emails/meetings that were already created in this session

Rules:
- If a request could be answered using internal documents (facts about people, projects, policies, etc.), call search_knowledge_base BEFORE answering. Never answer such questions from your own general knowledge.
- If search_knowledge_base finds nothing relevant, tell the user you couldn't find it in the documents. Do not guess.
- A single request may need MULTIPLE steps -- e.g. look something up, then use what you found to send an email or create a ticket. Do this in sequence as needed.
- If the user just wants to chat or asks something general, respond normally without forcing a tool call.
- Keep answers concise and natural."""


def main():
    embedder = get_embedder()
    store = VectorStore(embedder=embedder, persist_dir=PERSIST_DIR)
    search_knowledge_base = make_search_tool(store, embedder)

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    chat = client.chats.create(
        model=MODEL,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            tools=[search_knowledge_base, send_email, create_ticket, schedule_meeting, list_recent_actions],
        ),
    )

    print(f"Enterprise AI Knowledge Worker ready. {store.count()} document chunks loaded.")
    print("Ask a question, request an action, or combine both. ('quit' to exit)\n")

    while True:
        user_input = input("> ").strip()
        if user_input.lower() in ("quit", "exit"):
            break
        if not user_input:
            continue

        try:
            response = call_with_retry(chat.send_message, user_input)
            print(f"\nAssistant: {response.text}\n")
        except RuntimeError as e:
            print(f"\n[Sorry, couldn't complete that right now: {e}]\n")


if __name__ == "__main__":
    main()
