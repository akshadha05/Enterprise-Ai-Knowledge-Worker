"""
The full, real pipeline: question -> resolve follow-ups using memory ->
retrieve -> grounded answer -> exact sources.

This is what a real user-facing "ask the knowledge base" feature looks
like, now with conversational memory (Phase 5).

Usage:
    python -m backend.rag.ask
"""

import os

from dotenv import load_dotenv

from ..memory.condense import condense_question
from ..memory.conversation_memory import ConversationMemory
from .embeddings import get_embedder
from .highlight import find_relevant_lines
from .llm import get_llm
from .vector_store import VectorStore

load_dotenv()

PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "./backend/data/vector_store")
RELEVANCE_THRESHOLD = 0.3
TOP_K = 3
MAX_SOURCES_SHOWN = 2       # don't clutter the output with every match
LINES_PER_SOURCE = 2        # how many specific lines to show per source chunk


def main():
    embedder = get_embedder()
    llm = get_llm()
    store = VectorStore(embedder=embedder, persist_dir=PERSIST_DIR)
    memory = ConversationMemory()

    print(f"Knowledge base loaded: {store.count()} chunks available.\n")
    print("Ask a question about your document (or 'quit' to exit):\n")

    while True:
        question = input("> ").strip()
        if question.lower() in ("quit", "exit"):
            break
        if not question:
            continue

        try:
            # Resolve follow-up questions ("what was the F1 score?") into
            # standalone ones using recent conversation history.
            search_question = condense_question(question, memory, llm)
            if search_question != question:
                print(f"  (interpreted as: \"{search_question}\")")

            all_matches = store.query(search_question, top_k=TOP_K)
            # Only keep chunks that actually clear the relevance bar --
            # this is the anti-hallucination filter in action.
            relevant_matches = [m for m in all_matches if m["similarity"] >= RELEVANCE_THRESHOLD]

            answer = llm.generate(search_question, relevant_matches)

            print(f"\nAnswer: {answer}\n")

            if relevant_matches:
                print("Exact lines this was based on:")
                for m in relevant_matches[:MAX_SOURCES_SHOWN]:
                    lines = find_relevant_lines(search_question, m["text"], embedder, top_n=LINES_PER_SOURCE)
                    print(f"  --- {m['source_filename']}, page {m['page_number']} ---")
                    for line in lines:
                        print(f"  \"{line}\"")
                    print()

            memory.add_turn(question, answer)
        except RuntimeError as e:
            # All retries exhausted (e.g. rate limit) -- fail gracefully,
            # keep the session alive so the user can try again.
            print(f"\n[Sorry, couldn't complete that right now: {e}]")

        print()


if __name__ == "__main__":
    main()
