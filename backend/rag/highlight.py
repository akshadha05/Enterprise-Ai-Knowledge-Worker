"""
Zooms into a matched chunk to find the SPECIFIC lines that actually
support the answer, instead of showing the whole chunk.

How it works: the chunk itself is already the right "neighborhood" (we
found it via chunk-level semantic search). Now we split that chunk into
individual lines/bullets and do the SAME kind of semantic search again,
one level smaller, to rank which lines are most relevant to the question.

Important: we are NOT asking the LLM to pick or quote lines. We compute
similarity ourselves and slice the original string directly -- so what's
shown is guaranteed to be an exact, unedited substring of the source
document, just like the full-chunk citations were.
"""

from .embeddings import EmbeddingProvider


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    return dot / (norm_a * norm_b + 1e-9)


def find_relevant_lines(
    question: str,
    chunk_text: str,
    embedder: EmbeddingProvider,
    top_n: int = 2,
    min_similarity: float = 0.25,
) -> list[str]:
    """
    Returns the top_n lines within chunk_text most relevant to the question.
    Falls back to the whole chunk if it can't be usefully split (e.g. it's
    already just one dense line with no natural breaks).
    """
    lines = [l.strip() for l in chunk_text.split("\n\n") if l.strip()]

    if len(lines) <= 1:
        return [chunk_text]  # nothing finer to zoom into

    question_vector = embedder.embed([question])[0]
    line_vectors = embedder.embed(lines)

    scored = [
        (line, _cosine_similarity(question_vector, vec))
        for line, vec in zip(lines, line_vectors)
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)

    top_lines = [line for line, score in scored[:top_n] if score >= min_similarity]

    return top_lines if top_lines else [scored[0][0]]
