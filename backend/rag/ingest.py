"""
Ingestion pipeline: run this whenever you add new documents.

Flow: raw file -> exact text -> chunks -> vectors -> stored in Chroma.

Usage:
    python -m backend.rag.ingest
(reads every file in backend/data/raw_documents/)
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from .chunker import chunk_document
from .document_loader import load_document
from .embeddings import get_embedder
from .vector_store import VectorStore

load_dotenv()

RAW_DOCS_DIR = Path(__file__).parent.parent / "data" / "raw_documents"
PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "./backend/data/vector_store")


def ingest_all() -> None:
    embedder = get_embedder()
    store = VectorStore(embedder=embedder, persist_dir=PERSIST_DIR)

    files = [f for f in RAW_DOCS_DIR.iterdir() if f.is_file() and not f.name.startswith(".")]

    if not files:
        print(f"No files found in {RAW_DOCS_DIR}. Add a document and re-run.")
        return

    succeeded, skipped = [], []

    for file_path in files:
        print(f"Loading {file_path.name} ...")
        try:
            doc = load_document(file_path)

            if not doc.pages:
                # Common cause: a scanned/image-only PDF with no extractable
                # text (would need OCR, which we don't do here). Skip it
                # cleanly instead of silently storing zero useful chunks.
                print(f"  -> No extractable text found (possibly a scanned/image PDF). Skipping.")
                skipped.append(file_path.name)
                continue

            chunks = chunk_document(doc)
            print(f"  -> split into {len(chunks)} chunks")

            store.add_chunks(chunks)
            print(f"  -> embedded and stored")
            succeeded.append(file_path.name)

        except Exception as e:
            # One bad file (corrupted, unsupported format, etc.) should
            # never stop the rest of the batch from being ingested.
            print(f"  -> FAILED to process this file: {e}")
            print(f"  -> Skipping it and continuing with the rest.")
            skipped.append(file_path.name)

    print(f"\nDone. Vector store now has {store.count()} chunks total.")
    print(f"Successfully ingested: {succeeded if succeeded else 'none'}")
    if skipped:
        print(f"Skipped (see reasons above): {skipped}")


if __name__ == "__main__":
    ingest_all()
