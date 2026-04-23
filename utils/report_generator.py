"""
Export the analysis result as a formatted PDF report using ReportLab.
Returns raw bytes of the PDF.
"""

import io
import logging
from datetime import datetime
from typing import Optional

from models.schemas import AnalysisResponse, ConfidenceLevel

logger = logging.getLogger(__name__)


def _confidence_colour(level: ConfidenceLevel):
    """Return an RGB tuple for the confidence badge."""
    from reportlab.lib import colors
    mapping = {
        ConfidenceLevel.HIGH: colors.green,
        ConfidenceLevel.MEDIUM: colors.orange,
        ConfidenceLevel.LOW: colors.red,
    }
    return mapping.get(level, colors.gray)


def generate_pdf_report(result: AnalysisResponse, filename: str = "upload.pdf") -> bytes:
    """Generate a PDF analysis report and return its bytes."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            HRFlowable,
        )
    except ImportError:
        logger.error("reportlab not installed — PDF export unavailable")
        raise RuntimeError("reportlab is required for PDF export. Install it with: pip install reportlab")

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title2", parent=styles["Title"], fontSize=18, spaceAfter=10)
    heading_style = ParagraphStyle("Heading2", parent=styles["Heading2"], fontSize=13, spaceBefore=12, spaceAfter=6)
    normal = styles["Normal"]
    small = ParagraphStyle("Small", parent=normal, fontSize=8, textColor=colors.gray)

    story = []

    # --- Header ---
    story.append(Paragraph("AI Generated Content Detection Report", title_style))
    story.append(Paragraph(f"File: <b>{filename}</b>", normal))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", small))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.black, spaceAfter=12))

    # --- Summary ---
    story.append(Paragraph("Summary", heading_style))

    pct = result.overall_ai_percentage
    pct_colour = colors.red if pct >= 70 else (colors.orange if pct >= 40 else colors.green)

    summary_data = [
        ["Metric", "Value"],
        ["Overall AI Likelihood", f"{pct:.1f}%"],
        ["Confidence Level", result.confidence.value],
        ["Score Variance", f"{result.score_variance:.4f}"],
        ["Total Pages", str(result.processing_stats.total_pages)],
        ["Total Chunks Analysed", str(result.processing_stats.total_chunks)],
        ["Total Words", str(result.processing_stats.total_words)],
        ["OCR Pages", str(result.processing_stats.ocr_pages)],
        ["Processing Time", f"{result.processing_stats.processing_time_seconds}s"],
        ["LLM Provider", result.processing_stats.llm_provider_used or "Heuristic Only"],
    ]

    summary_table = Table(summary_data, colWidths=[8 * cm, 8 * cm])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f4f4")]),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 0.5 * cm))

    # --- High-Risk Sections ---
    if result.high_risk_sections:
        story.append(Paragraph(f"High-Risk Sections ({len(result.high_risk_sections)})", heading_style))
        hr_data = [["Page", "Chunk", "AI Score", "Snippet"]]
        for sec in result.high_risk_sections[:20]:  # cap at 20 rows
            hr_data.append([
                str(sec.page),
                str(sec.chunk_id),
                f"{sec.ai_score:.2f}",
                Paragraph(sec.text_snippet[:120] + "...", small),
            ])

        hr_table = Table(hr_data, colWidths=[2 * cm, 2 * cm, 2.5 * cm, 9.5 * cm])
        hr_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#c0392b")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fdecea")]),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(hr_table)
        story.append(Spacer(1, 0.5 * cm))

    # --- Chunk breakdown ---
    story.append(Paragraph("Chunk-Level Breakdown", heading_style))
    chunk_data = [["Chunk", "Page", "AI Score", "LLM Score", "Heuristic", "Label"]]
    for chunk in result.chunks:
        chunk_data.append([
            str(chunk.chunk_id),
            str(chunk.page),
            f"{chunk.ai_score:.2f}",
            f"{chunk.llm_score:.2f}" if chunk.llm_score is not None else "N/A",
            f"{chunk.heuristic_score:.2f}",
            chunk.label.value,
        ])

    chunk_table = Table(chunk_data, colWidths=[2 * cm, 2 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm, 4 * cm])
    chunk_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2980b9")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eaf4fb")]),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(chunk_table)

    doc.build(story)
    return buf.getvalue()
