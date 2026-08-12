"""
Quick manual test of retrieval -- BEFORE we add LLM answer generation.

This lets you ask a question and see, with your own eyes, exactly which
stored chunk(s) get matched and how similar they are. This is a crucial
debugging step: if retrieval finds the wrong chunk, no amount of clever
prompting later will produce a correct answer -- garbage in, garbage out.

Usage:
    python -m backend.rag.retrieve_test
"""

import os

from dotenv import load_dotenv

from .embeddings import get_embedder
from .vector_store import VectorStore

load_dotenv()

PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "./backend/data/vector_store")

# Below this similarity score, we treat it as "not actually relevant" --
# this is the anti-hallucination threshold we discussed earlier.
RELEVANCE_THRESHOLD = 0.3


def main():
    embedder = get_embedder()
    store = VectorStore(embedder=embedder, persist_dir=PERSIST_DIR)

    print(f"Vector store loaded: {store.count()} chunks available.\n")
    print("Type a question about your document (or 'quit' to exit):\n")

    while True:
        question = input("> ").strip()
        if question.lower() in ("quit", "exit"):
            break
        if not question:
            continue

        matches = store.query(question, top_k=3)

        print()
        if not matches or matches[0]["similarity"] < RELEVANCE_THRESHOLD:
            print("  [No sufficiently relevant chunk found -- would answer 'not in the documents']\n")
            continue

        for i, m in enumerate(matches, start=1):
            flag = "OK" if m["similarity"] >= RELEVANCE_THRESHOLD else "below threshold, ignored"
            print(f"  Match {i} | similarity={m['similarity']:.3f} ({flag})")
            print(f"  Source: {m['source_filename']} (page {m['page_number']})")
            print(f"  Text: {m['text'][:300]}{'...' if len(m['text']) > 300 else ''}")
            print()


if __name__ == "__main__":
    main()
