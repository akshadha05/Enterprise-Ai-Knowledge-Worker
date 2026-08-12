"""
Finds the specific line(s) within a matched chunk that actually answer
the question -- instead of showing the whole chunk as "the source."

How it works: a chunk is really just several original lines/bullet points
glued together (see chunker.py). We already know WHICH chunk is relevant
(from the main vector search). This module does a second, smaller search
INSIDE that chunk: embed each individual line, compare each to the
question, and keep only the highest-scoring 1-3 lines.

Same guarantee as before: these lines are copied verbatim from the
extracted document text. No LLM touches them, so they cannot be
misquoted -- we're just being more selective about which exact lines
we show.
"""

import math


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def find_supporting_lines(question: str, chunk_text: str, embedder, top_n: int = 2) -> list[str]:
    """
    Returns the top_n lines within chunk_text most relevant to the question,
    in their ORIGINAL order (not sorted by score) so they still read naturally.
    """
    # Chunks are built by joining original lines/paragraphs with "\n\n"
    # (see chunker.py) -- splitting on that recovers those original units.
    lines = [l.strip() for l in chunk_text.split("\n\n") if l.strip()]

    if not lines:
        return []
    if len(lines) <= top_n:
        return lines

    line_vectors = embedder.embed(lines)
    question_vector = embedder.embed([question])[0]

    scored = [
        (_cosine_similarity(question_vector, vec), line)
        for vec, line in zip(line_vectors, lines)
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    top_lines = {line for _, line in scored[:top_n]}

    # Return in original document order, not score order -- reads better
    return [line for line in lines if line in top_lines]
