"""
Generates three sample PDFs for manual testing:
  - sample_human.pdf   : clearly human-written prose
  - sample_ai.pdf      : clearly AI-generated prose
  - sample_mixed.pdf   : 50/50 mix
  - sample_scanned.pdf : image-only (tests OCR fallback)

Run with:
  python tests/generate_sample_pdf.py
"""

import io
import os
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


HUMAN_PARAGRAPHS = [
    textwrap.dedent("""\
        I remember the first time I saw the ocean. I was seven, maybe eight, and my uncle had
        driven us down through the night in his old pickup. My sister slept the whole way.
        When I finally saw that expanse of grey-green water stretching out to nowhere I just
        stood there with my shoes filling up with sand and didn't say anything for a long time.
        My uncle put his hand on my shoulder and said, "Well?" and I still didn't have words.
        I think that was the moment I understood that some things don't need explaining.
    """),
    textwrap.dedent("""\
        The machine shop smelled like metal shavings and old oil, a smell I've associated with
        competence ever since. My grandfather worked there for thirty-one years. He had these
        thick hands, the knuckles permanently darkened no matter how hard he scrubbed, and he
        could hold a tolerance of two thousandths of an inch by feel alone. I used to watch him
        with his calipers and think: this is what it means to know something. Not to know about
        it, but to know it in your hands.
    """),
    textwrap.dedent("""\
        We had a dog named Pickles who was afraid of absolutely nothing except the vacuum cleaner.
        The UPS man, the smoke detector, thunder, fireworks — all fine. But the moment my mother
        dragged out that Hoover he would sprint full-speed into the bathroom and sit behind the
        toilet until it was put away again. We never figured out why. Some things about animals
        you just accept. He died on a Tuesday in November and we all cried more than we expected to.
    """),
] * 40  # repeat to get ~10,000 words

AI_PARAGRAPHS = [
    textwrap.dedent("""\
        Artificial intelligence represents a transformative paradigm shift in modern technology.
        It is important to note that machine learning algorithms enable systems to learn from data
        and improve their performance over time. Furthermore, natural language processing has
        advanced significantly in recent years. In conclusion, these technologies offer tremendous
        potential for enhancing productivity and efficiency across diverse sectors of the economy.
    """),
    textwrap.dedent("""\
        The integration of AI into business processes offers numerous advantages. First and
        foremost, automated systems can process information at unprecedented speeds. Additionally,
        AI-powered tools reduce the likelihood of human error. Moreover, these systems operate
        continuously without fatigue. It should be emphasized that the implementation of AI
        requires careful strategic planning and stakeholder alignment to maximize return on
        investment.
    """),
    textwrap.dedent("""\
        In today's rapidly evolving technological landscape, organizations must adapt to remain
        competitive. Leveraging cutting-edge artificial intelligence solutions enables enterprises
        to unlock new value streams. Furthermore, data-driven decision-making enhances operational
        agility. It is worth noting that successful digital transformation initiatives require
        executive sponsorship and cross-functional collaboration. In summary, AI adoption is no
        longer optional for organizations seeking sustainable growth.
    """),
] * 40


def _make_pdf(paragraphs: list, output_path: str, title: str) -> None:
    import fitz

    doc = fitz.open()
    all_text = "\n\n".join(paragraphs)
    words = all_text.split()
    words_per_page = 350

    for i in range(0, len(words), words_per_page):
        page = doc.new_page(width=595, height=842)  # A4
        chunk = " ".join(words[i : i + words_per_page])
        # Title on first page
        if i == 0:
            page.insert_text((50, 50), title, fontsize=16)
            page.insert_text((50, 80), chunk, fontsize=10, linesize=14)
        else:
            page.insert_text((50, 50), chunk, fontsize=10, linesize=14)

    doc.save(output_path)
    doc.close()
    size_kb = os.path.getsize(output_path) / 1024
    print(f"  Created {output_path} ({size_kb:.0f} KB, ~{len(words)} words)")


def _make_scanned_pdf(output_path: str) -> None:
    """Create a PDF with text rendered as an image (simulates scanned book)."""
    try:
        import fitz
        from PIL import Image, ImageDraw, ImageFont

        text_lines = [
            "This is a scanned page. The text here is an image,",
            "not selectable text. The OCR engine should extract it.",
            "",
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
            "Sed do eiusmod tempor incididunt ut labore et dolore magna.",
            "Ut enim ad minim veniam, quis nostrud exercitation ullamco.",
        ]

        doc = fitz.open()
        for _ in range(3):
            # Create an image with text
            img = Image.new("RGB", (595, 842), color="white")
            draw = ImageDraw.Draw(img)
            y = 50
            for line in text_lines:
                draw.text((50, y), line, fill="black")
                y += 25

            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)

            page = doc.new_page(width=595, height=842)
            rect = fitz.Rect(0, 0, 595, 842)
            page.insert_image(rect, stream=buf.read())

        doc.save(output_path)
        doc.close()
        size_kb = os.path.getsize(output_path) / 1024
        print(f"  Created {output_path} ({size_kb:.0f} KB) — image-only/scanned")
    except Exception as exc:
        print(f"  Skipped scanned PDF ({exc})")


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "sample_pdfs")
    os.makedirs(out_dir, exist_ok=True)

    print("Generating sample PDFs...")

    _make_pdf(HUMAN_PARAGRAPHS, os.path.join(out_dir, "sample_human.pdf"), "Human-Written Book Sample")
    _make_pdf(AI_PARAGRAPHS, os.path.join(out_dir, "sample_ai.pdf"), "AI-Generated Book Sample")
    _make_pdf(
        HUMAN_PARAGRAPHS[:20] + AI_PARAGRAPHS[:20],
        os.path.join(out_dir, "sample_mixed.pdf"),
        "Mixed Content Book Sample",
    )
    _make_scanned_pdf(os.path.join(out_dir, "sample_scanned.pdf"))

    print("\nDone. PDFs saved to:", out_dir)
    print("\nTo test the API:")
    print(f'  curl -X POST http://localhost:8000/api/v1/analyze-book \\')
    print(f'       -F "file=@{out_dir}/sample_ai.pdf"')
