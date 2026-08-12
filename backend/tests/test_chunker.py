"""
Unit tests for the chunker. These test the SPLITTING LOGIC only -- no
embeddings, no API calls, no network. Run these anytime with:

    pytest backend/tests/test_chunker.py -v
"""

from backend.rag.chunker import (
    _split_into_paragraphs,
    _split_oversized_paragraph,
    chunk_document,
)
from backend.rag.document_loader import LoadedDocument, PageText


def test_splits_on_blank_lines_when_present():
    text = "First paragraph here.\n\nSecond paragraph here.\n\nThird one."
    result = _split_into_paragraphs(text)
    assert result == ["First paragraph here.", "Second paragraph here.", "Third one."]


def test_falls_back_to_single_newlines_when_no_blank_lines():
    # This is the exact bug we hit with the resume PDF: no blank lines
    # at all, so we must fall back to single-newline splitting.
    text = "Line one\nLine two\nLine three"
    result = _split_into_paragraphs(text)
    assert result == ["Line one", "Line two", "Line three"]


def test_oversized_paragraph_gets_split_on_word_boundaries():
    # A single "paragraph" way bigger than our target chunk size.
    long_text = " ".join([f"word{i}" for i in range(500)])  # ~3500 chars
    pieces = _split_oversized_paragraph(long_text)

    assert len(pieces) > 1
    # Every piece should end on a whole word -- never mid-word.
    for piece in pieces:
        assert not piece.endswith(" ")
        words = piece.split()
        assert all(w.startswith("word") for w in words)


def test_short_paragraph_is_not_split():
    short_text = "This is a short paragraph that fits in one chunk easily."
    pieces = _split_oversized_paragraph(short_text)
    assert pieces == [short_text]


def test_chunk_document_preserves_page_and_source_metadata():
    doc = LoadedDocument(
        filename="test.pdf",
        pages=[
            PageText(text="Page one content here.\n\nMore page one content.", page_number=1),
            PageText(text="Page two content here.", page_number=2),
        ],
    )
    chunks = chunk_document(doc)

    assert len(chunks) >= 2
    assert all(c.source_filename == "test.pdf" for c in chunks)
    page_numbers = {c.page_number for c in chunks}
    assert page_numbers == {1, 2}


def test_chunk_ids_are_unique():
    doc = LoadedDocument(
        filename="test.pdf",
        pages=[PageText(text="A.\n\nB.\n\nC.\n\nD.", page_number=1)],
    )
    chunks = chunk_document(doc)
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))  # no duplicates
