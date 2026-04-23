import io
import logging
from dataclasses import dataclass
from typing import List, Optional

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


@dataclass
class PageText:
    page_number: int  # 1-indexed
    text: str
    used_ocr: bool
    word_count: int


def _extract_text_fitz(page: fitz.Page) -> str:
    """Extract text from a single fitz page."""
    return page.get_text("text").strip()


def _ocr_page(page: fitz.Page) -> str:
    """Render page to image and run OCR via pytesseract."""
    try:
        import pytesseract
        from PIL import Image

        # Render at 300 DPI for good OCR quality
        mat = fitz.Matrix(300 / 72, 300 / 72)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
        img_bytes = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_bytes))
        text = pytesseract.image_to_string(image, config="--psm 6")
        return text.strip()
    except ImportError:
        logger.error("pytesseract or Pillow not installed — OCR unavailable")
        return ""
    except Exception as exc:
        logger.warning("OCR failed for page: %s", exc)
        return ""


def extract_pages(pdf_bytes: bytes, ocr_threshold: int = 20) -> List[PageText]:
    """
    Extract text from every page of a PDF.

    Falls back to OCR when a page yields fewer than `ocr_threshold` characters
    (indicating a scanned/image-only page).
    """
    results: List[PageText] = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    for page_index in range(len(doc)):
        page = doc[page_index]
        raw_text = _extract_text_fitz(page)
        used_ocr = False

        if len(raw_text) < ocr_threshold:
            logger.info("Page %d has sparse text (%d chars) — attempting OCR", page_index + 1, len(raw_text))
            ocr_text = _ocr_page(page)
            if len(ocr_text) > len(raw_text):
                raw_text = ocr_text
                used_ocr = True

        # Normalize whitespace
        clean = " ".join(raw_text.split())
        word_count = len(clean.split()) if clean else 0

        results.append(
            PageText(
                page_number=page_index + 1,
                text=clean,
                used_ocr=used_ocr,
                word_count=word_count,
            )
        )

    doc.close()
    logger.info(
        "Extracted text from %d pages (%d OCR pages)",
        len(results),
        sum(1 for p in results if p.used_ocr),
    )
    return results
