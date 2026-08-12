"""
Chunker: splits document text into small, overlapping, retrievable pieces.

Design choices made here (and why):

1. We split on paragraph breaks first, THEN group paragraphs up to a target
   size -- instead of blindly cutting every N characters. This avoids
   slicing a sentence or idea in half.

2. Overlap: consecutive chunks share a bit of text. If an important detail
   sits right at a boundary, it still shows up whole in at least one chunk.

3. Every chunk keeps metadata (source filename, page number) attached.
   This travels all the way to the final answer as a citation.
"""

from dataclasses import dataclass, field

from .document_loader import LoadedDocument

TARGET_CHUNK_CHARS = 1500     # roughly 300-400 words
OVERLAP_CHARS = 200           # roughly 10-15% overlap


@dataclass
class Chunk:
    text: str
    source_filename: str
    page_number: int
    chunk_index: int
    classification: str = "public"  # "public" or "confidential" -- controls who can retrieve this chunk
    metadata: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.source_filename}::p{self.page_number}::c{self.chunk_index}"


def _split_oversized_paragraph(text: str) -> list[str]:
    """
    Fallback for a single 'paragraph' that's way too long (e.g. a resume or
    dense PDF page with no blank lines to split on). Breaks it into
    TARGET_CHUNK_CHARS-ish pieces on WORD boundaries (never mid-word),
    with overlap, so we never end up with one giant unsearchable blob.
    """
    if len(text) <= TARGET_CHUNK_CHARS:
        return [text]

    words = text.split()
    pieces = []
    current = []
    current_len = 0

    for word in words:
        current.append(word)
        current_len += len(word) + 1
        if current_len >= TARGET_CHUNK_CHARS:
            pieces.append(" ".join(current))
            # keep the last few words as overlap into the next piece
            overlap_word_count = max(1, OVERLAP_CHARS // 8)  # ~8 chars/word estimate
            current = current[-overlap_word_count:]
            current_len = sum(len(w) + 1 for w in current)

    if current:
        pieces.append(" ".join(current))

    return pieces


def _split_into_paragraphs(text: str) -> list[str]:
    """
    Tries blank-line paragraph breaks first (the ideal case). If that
    produces basically nothing (one giant blob -- common in resumes and
    tightly-formatted PDFs), falls back to single-newline splitting.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # If blank-line splitting didn't actually break anything up, try single newlines
    if len(paragraphs) <= 1:
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    return paragraphs


def chunk_document(doc: LoadedDocument, classification: str = "public") -> list[Chunk]:
    chunks: list[Chunk] = []
    chunk_index = 0

    for page in doc.pages:
        raw_paragraphs = _split_into_paragraphs(page.text)

        # Any paragraph that's still too long on its own (e.g. a resume
        # section extracted as one dense block) gets further split.
        paragraphs = []
        for p in raw_paragraphs:
            paragraphs.extend(_split_oversized_paragraph(p))

        current_text = ""
        for para in paragraphs:
            # If adding this paragraph would blow past our target size,
            # close out the current chunk first.
            if current_text and len(current_text) + len(para) > TARGET_CHUNK_CHARS:
                chunks.append(
                    Chunk(
                        text=current_text.strip(),
                        source_filename=doc.filename,
                        page_number=page.page_number,
                        chunk_index=chunk_index,
                        classification=classification,
                    )
                )
                chunk_index += 1
                # Start the next chunk with the tail of the previous one (the overlap)
                current_text = current_text[-OVERLAP_CHARS:] + "\n\n" + para
            else:
                current_text = (current_text + "\n\n" + para) if current_text else para

        # Don't forget whatever's left over at the end of the page
        if current_text.strip():
            chunks.append(
                Chunk(
                    text=current_text.strip(),
                    source_filename=doc.filename,
                    page_number=page.page_number,
                    chunk_index=chunk_index,
                    classification=classification,
                )
            )
            chunk_index += 1

    return chunks
