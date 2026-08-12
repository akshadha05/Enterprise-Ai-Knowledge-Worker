"""
Unit tests for highlight.py (finding the specific relevant lines within
a matched chunk). Uses a FAKE embedder with predictable, controllable
output instead of a real one -- so these tests run instantly, with no
API calls, and the "correct answer" is known in advance.

    pytest backend/tests/test_highlight.py -v
"""

from backend.rag.highlight import _cosine_similarity, find_relevant_lines


class FakeEmbedder:
    """
    A stand-in embedder for testing. Instead of real semantic meaning,
    it builds a simple vector based on which of a fixed set of "topic
    words" appear in the text. This lets tests assert exactly which line
    SHOULD be considered most similar to a given question, deterministically.
    """

    TOPIC_WORDS = ["python", "javascript", "database", "meeting", "vacation"]

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            lower = text.lower()
            vectors.append([1.0 if word in lower else 0.0 for word in self.TOPIC_WORDS])
        return vectors


def test_cosine_similarity_identical_vectors_is_one():
    v = [1.0, 0.0, 1.0]
    assert abs(_cosine_similarity(v, v) - 1.0) < 1e-9


def test_cosine_similarity_orthogonal_vectors_is_zero():
    assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_find_relevant_lines_picks_the_matching_topic():
    chunk_text = (
        "The candidate has strong Python skills.\n\n"
        "They also scheduled a meeting for next week.\n\n"
        "Vacation policy allows 18 days per year."
    )
    embedder = FakeEmbedder()

    result = find_relevant_lines("does she know python?", chunk_text, embedder, top_n=1)

    assert len(result) == 1
    assert "Python" in result[0]


def test_find_relevant_lines_falls_back_when_no_split_possible():
    # No blank-line breaks at all -- nothing to zoom into.
    chunk_text = "Just one single dense line with no paragraph breaks anywhere in it."
    embedder = FakeEmbedder()

    result = find_relevant_lines("anything", chunk_text, embedder, top_n=2)

    assert result == [chunk_text]


def test_find_relevant_lines_respects_top_n():
    chunk_text = "Python here.\n\nJavaScript here.\n\nDatabase here.\n\nMeeting here."
    embedder = FakeEmbedder()

    result = find_relevant_lines("python and javascript skills", chunk_text, embedder, top_n=2)

    assert len(result) <= 2
