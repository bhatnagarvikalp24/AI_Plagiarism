import logging
import os

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

from config import get_settings
from models.schemas import AnalysisResponse, ErrorResponse
from services.detection_engine import run_detection
from services.pdf_processor import extract_pages
from services.text_chunker import chunk_pages
from utils.file_utils import cleanup_file, save_upload_to_disk, validate_pdf_upload
from utils.report_generator import generate_pdf_report

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/analyze-book",
    response_model=AnalysisResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file"},
        413: {"model": ErrorResponse, "description": "File too large"},
        422: {"model": ErrorResponse, "description": "Processing error"},
        500: {"model": ErrorResponse, "description": "Internal error"},
    },
    summary="Analyse a PDF book for AI-generated content",
    description=(
        "Upload a PDF (up to 50 MB). The system extracts text page-by-page, "
        "splits it into 300–500 word chunks, runs heuristic + LLM analysis on each "
        "chunk, and returns an overall AI-likelihood percentage with a chunk-level "
        "breakdown and high-risk sections."
    ),
)
async def analyze_book(file: UploadFile = File(...)):
    settings = get_settings()
    validate_pdf_upload(file, settings.max_file_size_mb)

    pdf_path = None
    try:
        pdf_path, pdf_bytes = await save_upload_to_disk(file, settings.upload_dir)

        # 1. Extract pages
        pages = extract_pages(pdf_bytes)
        if not pages or all(p.word_count == 0 for p in pages):
            raise HTTPException(
                status_code=422,
                detail="Could not extract any text from the PDF. "
                       "Ensure the file is not password-protected or corrupted.",
            )

        # 2. Chunk text
        chunks = chunk_pages(pages, settings.chunk_min_words, settings.chunk_max_words)
        if not chunks:
            raise HTTPException(
                status_code=422,
                detail="The document appears to be too short for analysis (no valid chunks).",
            )

        # 3. Run detection
        result = await run_detection(chunks, pages)
        return result

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error during book analysis")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if pdf_path:
            cleanup_file(pdf_path)


@router.post(
    "/analyze-book/report",
    summary="Analyse and return a downloadable PDF report",
    response_class=Response,
    responses={
        200: {"content": {"application/pdf": {}}, "description": "PDF report"},
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def analyze_book_report(file: UploadFile = File(...)):
    """Same as /analyze-book but returns a formatted PDF report."""
    settings = get_settings()
    validate_pdf_upload(file, settings.max_file_size_mb)

    original_filename = file.filename or "upload.pdf"
    pdf_path = None
    try:
        pdf_path, pdf_bytes = await save_upload_to_disk(file, settings.upload_dir)
        pages = extract_pages(pdf_bytes)
        chunks = chunk_pages(pages, settings.chunk_min_words, settings.chunk_max_words)

        if not chunks:
            raise HTTPException(status_code=422, detail="No valid text chunks found in the PDF.")

        result = await run_detection(chunks, pages)
        report_bytes = generate_pdf_report(result, filename=original_filename)

        return Response(
            content=report_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="ai_detection_report.pdf"'},
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Report generation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if pdf_path:
            cleanup_file(pdf_path)


@router.get("/health", summary="Health check")
async def health():
    settings = get_settings()
    return {
        "status": "ok",
        "llm_provider": settings.llm_provider,
        "openai_configured": bool(settings.openai_api_key),
    }
