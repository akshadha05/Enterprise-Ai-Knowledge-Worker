"""
Vector store wrapper around ChromaDB.

Chroma stores each chunk's vector (from embeddings.py) alongside its
original text and metadata (filename, page number). Later, given a new
vector (from a user's question), it can instantly find the chunks whose
vectors are closest -- this is the "search by meaning" step.
"""

import chromadb

from .chunker import Chunk
from .embeddings import EmbeddingProvider


class VectorStore:
    def __init__(self, embedder: EmbeddingProvider, persist_dir: str, collection_name: str = "knowledge_base"):
        self.embedder = embedder
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # cosine similarity = standard for semantic search
        )

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """
        Each chunk's own .classification field (set during chunk_document())
        determines who's allowed to see it later -- "public" (the default)
        means everyone. See query()'s allowed_access_levels parameter and
        backend/common/access_control.py for how this gets enforced.
        """
        if not chunks:
            return

        vectors = self.embedder.embed([c.text for c in chunks])

        self.collection.upsert(
            ids=[c.id for c in chunks],
            embeddings=vectors,
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "source_filename": c.source_filename,
                    "page_number": c.page_number,
                    "access_level": c.classification,
                }
                for c in chunks
            ],
        )

    def query(self, question: str, top_k: int = 5, allowed_access_levels: list[str] | None = None) -> list[dict]:
        """
        Returns the top_k most similar chunks to the question, each with:
        - text
        - source_filename, page_number (for citation)
        - similarity (0-1, higher = more relevant)

        allowed_access_levels: if given, ONLY chunks tagged with one of
        these access levels are eligible to be returned -- this is the
        actual enforcement point for role-based access control. If None,
        no filtering happens (all chunks are eligible), which matches the
        old behavior for callers that haven't been updated yet (like the
        terminal scripts).
        """
        query_vector = self.embedder.embed([question])[0]

        where_filter = None
        if allowed_access_levels is not None:
            where_filter = {"access_level": {"$in": allowed_access_levels}}

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=where_filter,
        )

        matches = []
        for i in range(len(results["ids"][0])):
            distance = results["distances"][0][i]
            similarity = 1 - distance  # cosine distance -> similarity
            matches.append(
                {
                    "text": results["documents"][0][i],
                    "source_filename": results["metadatas"][0][i]["source_filename"],
                    "page_number": results["metadatas"][0][i]["page_number"],
                    "similarity": similarity,
                }
            )
        return matches

    def count(self) -> int:
        return self.collection.count()

    def list_sources(self) -> list[dict]:
        """
        Returns each distinct source document currently in the store,
        with how many chunks it contributed and its access level. Used
        by the /documents endpoint so the frontend can show what's
        actually loaded.
        """
        if self.collection.count() == 0:
            return []

        all_records = self.collection.get(include=["metadatas"])
        info: dict[str, dict] = {}
        for meta in all_records["metadatas"]:
            filename = meta["source_filename"]
            if filename not in info:
                info[filename] = {"filename": filename, "chunks": 0, "access_level": meta.get("access_level", "public")}
            info[filename]["chunks"] += 1

        return sorted(info.values(), key=lambda d: d["filename"])
