import logging
from dataclasses import dataclass
from typing import List

from services.pdf_processor import PageText

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    chunk_id: int         # 1-indexed
    page_number: int      # page where this chunk starts
    text: str
    word_count: int


def _words(text: str) -> List[str]:
    return text.split()


def chunk_pages(
    pages: List[PageText],
    min_words: int = 300,
    max_words: int = 500,
) -> List[TextChunk]:
    """
    Merge page texts and split into chunks of min_words–max_words.

    Strategy:
    - Accumulate words across pages, tracking which page each word came from.
    - Emit a chunk when accumulated words reach max_words, or on natural
      paragraph breaks within min_words–max_words range.
    - Final partial chunk is emitted if it meets min_words, otherwise merged
      into the previous chunk.
    """
    chunks: List[TextChunk] = []
    buffer_words: List[str] = []
    buffer_page: int = 1  # page where buffer started

    def flush(words: List[str], page: int) -> None:
        if not words:
            return
        text = " ".join(words)
        chunks.append(
            TextChunk(
                chunk_id=len(chunks) + 1,
                page_number=page,
                text=text,
                word_count=len(words),
            )
        )

    for page in pages:
        if not page.text:
            continue

        page_words = _words(page.text)

        for word in page_words:
            if not buffer_words:
                buffer_page = page.page_number
            buffer_words.append(word)

            if len(buffer_words) >= max_words:
                flush(buffer_words, buffer_page)
                buffer_words = []

    # Handle remaining buffer
    if buffer_words:
        if len(buffer_words) >= min_words or not chunks:
            flush(buffer_words, buffer_page)
        else:
            # Merge short tail into the last chunk
            if chunks:
                last = chunks[-1]
                merged_text = last.text + " " + " ".join(buffer_words)
                chunks[-1] = TextChunk(
                    chunk_id=last.chunk_id,
                    page_number=last.page_number,
                    text=merged_text,
                    word_count=last.word_count + len(buffer_words),
                )
            else:
                flush(buffer_words, buffer_page)

    logger.info("Created %d chunks from %d pages", len(chunks), len(pages))
    return chunks
