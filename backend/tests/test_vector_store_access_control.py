"""
Integration test: proves that access-level filtering is actually
ENFORCED by the vector store, not just configured. Uses a fake but
deterministic embedder (no network) and a temporary Chroma directory
(no impact on your real data).

    pytest backend/tests/test_vector_store_access_control.py -v
"""

from backend.rag.chunker import Chunk
from backend.rag.vector_store import VectorStore


class FakeEmbedder:
    """Deterministic stand-in: vectors based on which keywords appear."""

    TOPIC_WORDS = ["salary", "vacation", "project"]

    def embed(self, texts):
        vectors = []
        for text in texts:
            lower = text.lower()
            vectors.append([1.0 if word in lower else 0.1 for word in self.TOPIC_WORDS])
        return vectors


def _make_store(tmp_path):
    return VectorStore(embedder=FakeEmbedder(), persist_dir=str(tmp_path))


def test_employee_cannot_see_hr_classified_chunks(tmp_path):
    store = _make_store(tmp_path)

    public_chunk = Chunk(
        text="The project uses Python and React.",
        source_filename="handbook.txt",
        page_number=1,
        chunk_index=0,
        classification="public",
    )
    hr_chunk = Chunk(
        text="Salary bands are confidential and range by level.",
        source_filename="hr_policy.txt",
        page_number=1,
        chunk_index=0,
        classification="hr",
    )
    store.add_chunks([public_chunk, hr_chunk])

    # Employee role -- only "public" allowed
    results = store.query("salary information", top_k=5, allowed_access_levels=["public"])
    filenames = [r["source_filename"] for r in results]

    assert "hr_policy.txt" not in filenames


def test_hr_role_can_see_hr_classified_chunks(tmp_path):
    store = _make_store(tmp_path)

    hr_chunk = Chunk(
        text="Salary bands are confidential and range by level.",
        source_filename="hr_policy.txt",
        page_number=1,
        chunk_index=0,
        classification="hr",
    )
    store.add_chunks([hr_chunk])

    results = store.query("salary information", top_k=5, allowed_access_levels=["public", "hr"])
    filenames = [r["source_filename"] for r in results]

    assert "hr_policy.txt" in filenames


def test_no_filter_returns_everything(tmp_path):
    # allowed_access_levels=None (the terminal scripts' default) should
    # apply NO restriction -- a deliberate choice for trusted local use,
    # not an accidental bypass. Confirm both classifications come back.
    store = _make_store(tmp_path)

    public_chunk = Chunk(
        text="Project uses Python.", source_filename="public.txt",
        page_number=1, chunk_index=0, classification="public",
    )
    hr_chunk = Chunk(
        text="Salary bands are confidential.", source_filename="hr_policy.txt",
        page_number=1, chunk_index=0, classification="hr",
    )
    store.add_chunks([public_chunk, hr_chunk])

    results = store.query("salary project", top_k=5, allowed_access_levels=None)
    filenames = {r["source_filename"] for r in results}

    assert filenames == {"public.txt", "hr_policy.txt"}


def test_classification_defaults_to_public():
    chunk = Chunk(text="text", source_filename="f.txt", page_number=1, chunk_index=0)
    assert chunk.classification == "public"
