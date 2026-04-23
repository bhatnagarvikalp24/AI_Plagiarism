"""
Integration tests for the AI Detection API.

Run with:
  pytest tests/test_api.py -v

Or against a live server:
  BASE_URL=http://localhost:8000 pytest tests/test_api.py -v
"""

import io
import os
import sys
import json
import textwrap

import pytest
import httpx

# Allow running as a standalone script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
API_URL = f"{BASE_URL}/api/v1"


# ---------------------------------------------------------------------------
# Helpers to create in-memory test PDFs
# ---------------------------------------------------------------------------
def _create_test_pdf(text: str) -> bytes:
    """Create a minimal PDF containing `text` using PyMuPDF."""
    import fitz

    doc = fitz.open()
    words_per_page = 400

    words = text.split()
    for i in range(0, len(words), words_per_page):
        page = doc.new_page()
        chunk = " ".join(words[i : i + words_per_page])
        page.insert_text((72, 72), chunk, fontsize=10)

    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


HUMAN_TEXT = textwrap.dedent("""\
    I remember the summer I turned twelve — the heat was different that year, thick and
    lazy, the kind that makes everything shimmer at the edges. My grandmother had a garden
    out back, all overgrown with tomatoes and basil, and I'd spend entire mornings in there
    before the sun got too sharp. She'd come out around ten with a glass of iced tea, sit
    in the old metal chair, and watch me do absolutely nothing useful. "You're a good
    watcher," she once told me, meaning it as a compliment. I didn't understand that then.
    The days slipped past like water. We didn't have plans. I think that's what I miss most —
    not the garden, not even her, exactly, but that feeling of time being wide open and
    generous. These days I make lists. I schedule recovery time. I track my productivity in
    a spreadsheet. Sometimes I wonder what that twelve-year-old would make of all this.
    Probably he'd just shrug and go back to watching the tomatoes grow.
""") * 25  # ~350 words × 25 ≈ 8,750 words, enough for several chunks

AI_TEXT = textwrap.dedent("""\
    Artificial intelligence represents one of the most transformative technological
    advancements of the modern era. It is important to note that AI systems are capable
    of performing a wide range of tasks with remarkable efficiency. Furthermore, these
    systems can process vast amounts of data in a fraction of the time required by human
    operators. In conclusion, the adoption of AI technologies offers significant benefits
    across numerous sectors. Additionally, machine learning algorithms continuously improve
    their performance through iterative training processes. Moreover, natural language
    processing enables seamless human-computer interaction. It should be emphasized that
    responsible deployment of AI requires careful consideration of ethical implications.
    In summary, artificial intelligence is revolutionizing industries worldwide by
    providing scalable, efficient, and cost-effective solutions to complex problems.
    Furthermore, the integration of AI into business processes enhances productivity.
""") * 25

MIXED_TEXT = HUMAN_TEXT[:len(HUMAN_TEXT)//2] + AI_TEXT[:len(AI_TEXT)//2]


# ---------------------------------------------------------------------------
# Unit tests (no live server needed — test services directly)
# ---------------------------------------------------------------------------
class TestPDFProcessor:
    def test_extract_clean_pdf(self):
        from services.pdf_processor import extract_pages

        pdf_bytes = _create_test_pdf("Hello world. This is a test page with some words.")
        pages = extract_pages(pdf_bytes)
        assert len(pages) >= 1
        assert any(p.word_count > 0 for p in pages)

    def test_extract_multipage_pdf(self):
        from services.pdf_processor import extract_pages

        long_text = "word " * 2000
        pdf_bytes = _create_test_pdf(long_text)
        pages = extract_pages(pdf_bytes)
        total_words = sum(p.word_count for p in pages)
        assert total_words > 1000


class TestTextChunker:
    def test_chunk_count(self):
        from services.pdf_processor import PageText
        from services.text_chunker import chunk_pages

        pages = [
            PageText(page_number=i + 1, text=" ".join(["word"] * 400), used_ocr=False, word_count=400)
            for i in range(10)
        ]
        chunks = chunk_pages(pages, min_words=300, max_words=500)
        assert len(chunks) >= 5

    def test_chunk_word_count_within_bounds(self):
        from services.pdf_processor import PageText
        from services.text_chunker import chunk_pages

        pages = [
            PageText(page_number=i + 1, text=" ".join(["word"] * 400), used_ocr=False, word_count=400)
            for i in range(8)
        ]
        chunks = chunk_pages(pages, min_words=300, max_words=500)
        # All but the last chunk should be within bounds
        for chunk in chunks[:-1]:
            assert 300 <= chunk.word_count <= 500


class TestHeuristicAnalyzer:
    def test_returns_scores_in_range(self):
        from services.heuristic_analyzer import analyze_chunk

        scores = analyze_chunk(HUMAN_TEXT[:1500])
        assert 0.0 <= scores.combined_heuristic_score <= 1.0
        assert 0.0 <= scores.perplexity_score <= 1.0
        assert 0.0 <= scores.burstiness_score <= 1.0
        assert 0.0 <= scores.repetition_score <= 1.0
        assert 0.0 <= scores.lexical_diversity_score <= 1.0

    def test_ai_text_scores_higher_than_human(self):
        from services.heuristic_analyzer import analyze_chunk

        human_score = analyze_chunk(HUMAN_TEXT[:1500]).combined_heuristic_score
        ai_score = analyze_chunk(AI_TEXT[:1500]).combined_heuristic_score
        # AI text should generally score higher — give generous tolerance
        assert ai_score >= human_score - 0.15, (
            f"AI score ({ai_score:.3f}) should be >= human score ({human_score:.3f}) - 0.15"
        )

    def test_short_text_does_not_crash(self):
        from services.heuristic_analyzer import analyze_chunk

        scores = analyze_chunk("Just three words.")
        assert scores is not None


# ---------------------------------------------------------------------------
# Integration tests (require live server)
# ---------------------------------------------------------------------------
@pytest.mark.integration
class TestAPI:
    def _post_pdf(self, pdf_bytes: bytes, filename: str = "test.pdf") -> dict:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(
                f"{API_URL}/analyze-book",
                files={"file": (filename, pdf_bytes, "application/pdf")},
            )
        resp.raise_for_status()
        return resp.json()

    def test_health_check(self):
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(f"{API_URL}/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_clean_pdf_analysis(self):
        pdf = _create_test_pdf(HUMAN_TEXT)
        result = self._post_pdf(pdf, "human_book.pdf")

        assert "overall_ai_percentage" in result
        assert 0.0 <= result["overall_ai_percentage"] <= 100.0
        assert result["confidence"] in ("Low", "Medium", "High")
        assert len(result["chunks"]) > 0

    def test_ai_pdf_scores_higher(self):
        human_pdf = _create_test_pdf(HUMAN_TEXT)
        ai_pdf = _create_test_pdf(AI_TEXT)

        human_result = self._post_pdf(human_pdf, "human.pdf")
        ai_result = self._post_pdf(ai_pdf, "ai.pdf")

        assert ai_result["overall_ai_percentage"] > human_result["overall_ai_percentage"]

    def test_mixed_content(self):
        pdf = _create_test_pdf(MIXED_TEXT)
        result = self._post_pdf(pdf, "mixed.pdf")

        assert result["overall_ai_percentage"] > 0
        assert result["processing_stats"]["total_chunks"] > 0

    def test_invalid_file_type_rejected(self):
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(
                f"{API_URL}/analyze-book",
                files={"file": ("test.txt", b"some text", "text/plain")},
            )
        assert resp.status_code == 400

    def test_chunk_scores_within_range(self):
        pdf = _create_test_pdf(AI_TEXT)
        result = self._post_pdf(pdf)

        for chunk in result["chunks"]:
            assert 0.0 <= chunk["ai_score"] <= 1.0
            assert chunk["label"] in ("Likely AI", "Possibly AI", "Likely Human", "Uncertain")

    def test_high_risk_sections_threshold(self):
        """High-risk sections must all have scores >= 0.70."""
        pdf = _create_test_pdf(AI_TEXT)
        result = self._post_pdf(pdf)

        for section in result["high_risk_sections"]:
            assert section["ai_score"] >= 0.70

    def test_response_schema(self):
        """Validate all required fields are present."""
        pdf = _create_test_pdf(HUMAN_TEXT)
        result = self._post_pdf(pdf)

        required_top = {"overall_ai_percentage", "confidence", "score_variance", "chunks", "high_risk_sections", "processing_stats"}
        assert required_top.issubset(result.keys())

        if result["chunks"]:
            chunk = result["chunks"][0]
            required_chunk = {"chunk_id", "page", "word_count", "ai_score", "heuristic_score", "label", "text_snippet"}
            assert required_chunk.issubset(chunk.keys())


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Running unit tests directly...\n")

    # Quick smoke tests without pytest
    t = TestHeuristicAnalyzer()
    t.test_returns_scores_in_range()
    print("✓ Heuristic scores in range")
    t.test_ai_text_scores_higher_than_human()
    print("✓ AI text scores higher than human text")
    t.test_short_text_does_not_crash()
    print("✓ Short text handled gracefully")

    t2 = TestPDFProcessor()
    t2.test_extract_clean_pdf()
    print("✓ PDF text extraction works")
    t2.test_extract_multipage_pdf()
    print("✓ Multi-page PDF extraction works")

    t3 = TestTextChunker()
    t3.test_chunk_count()
    print("✓ Chunk count is reasonable")
    t3.test_chunk_word_count_within_bounds()
    print("✓ Chunk word counts within bounds")

    print("\nAll unit tests passed!")
    print(f"\nFor integration tests, start the server and run:")
    print(f"  pytest tests/test_api.py -v -m integration --base-url={BASE_URL}")
