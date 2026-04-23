"""
Main orchestration layer.

For each chunk:
  1. Run heuristic analysis (CPU-bound, sync → run in thread pool)
  2. Run LLM analysis     (async HTTP)
  3. Combine scores: final = 0.6 * llm + 0.4 * heuristic
     (falls back to heuristic-only if LLM is unavailable)

Then aggregate across chunks for the overall report.
"""

import asyncio
import logging
import statistics
import time
from typing import List, Optional, Tuple

import httpx

from config import get_settings
from models.schemas import (
    AnalysisResponse,
    ChunkLabel,
    ChunkResult,
    ConfidenceLevel,
    HighRiskSection,
    ProcessingStats,
)
from services import heuristic_analyzer, llm_analyzer
from services.pdf_processor import PageText
from services.text_chunker import TextChunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------
def _label(score: float) -> ChunkLabel:
    if score >= 0.70:
        return ChunkLabel.LIKELY_AI
    if score >= 0.50:
        return ChunkLabel.POSSIBLY_AI
    if score >= 0.30:
        return ChunkLabel.UNCERTAIN
    return ChunkLabel.LIKELY_HUMAN


def _confidence(variance: float) -> ConfidenceLevel:
    settings = get_settings()
    if variance <= settings.high_confidence_variance_threshold:
        return ConfidenceLevel.HIGH
    if variance <= settings.medium_confidence_variance_threshold:
        return ConfidenceLevel.MEDIUM
    return ConfidenceLevel.LOW


# ---------------------------------------------------------------------------
# Per-chunk processing
# ---------------------------------------------------------------------------
async def _process_chunk(
    chunk: TextChunk,
    http_client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    loop: asyncio.AbstractEventLoop,
) -> Tuple[ChunkResult, float]:
    """Analyse one chunk and return (ChunkResult, ai_score)."""
    settings = get_settings()

    # Heuristic analysis in thread pool (avoids blocking the event loop)
    h_scores = await loop.run_in_executor(
        None, heuristic_analyzer.analyze_chunk, chunk.text
    )

    # LLM analysis with concurrency cap
    llm_score: Optional[float] = None
    async with semaphore:
        llm_score = await llm_analyzer.score_chunk(chunk.text, http_client)

    # Combine scores
    if llm_score is not None:
        ai_score = round(
            settings.llm_weight * llm_score
            + settings.heuristic_weight * h_scores.combined_heuristic_score,
            4,
        )
    else:
        ai_score = h_scores.combined_heuristic_score

    snippet = chunk.text[:200].replace("\n", " ")

    result = ChunkResult(
        chunk_id=chunk.chunk_id,
        page=chunk.page_number,
        word_count=chunk.word_count,
        ai_score=ai_score,
        llm_score=llm_score,
        heuristic_score=h_scores.combined_heuristic_score,
        heuristic_details=h_scores,
        label=_label(ai_score),
        text_snippet=snippet,
    )
    return result, ai_score


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------
async def run_detection(
    chunks: List[TextChunk],
    pages: List[PageText],
) -> AnalysisResponse:
    settings = get_settings()
    start_time = time.perf_counter()

    semaphore = asyncio.Semaphore(settings.llm_concurrency_limit)
    loop = asyncio.get_event_loop()

    async with httpx.AsyncClient() as http_client:
        tasks = [
            _process_chunk(chunk, http_client, semaphore, loop)
            for chunk in chunks
        ]
        chunk_outputs = await asyncio.gather(*tasks)

    chunk_results: List[ChunkResult] = []
    scores: List[float] = []

    for result, score in chunk_outputs:
        chunk_results.append(result)
        scores.append(score)

    # --- Aggregation ---
    overall_percentage = round(statistics.mean(scores) * 100, 2) if scores else 0.0
    variance = round(statistics.variance(scores), 6) if len(scores) > 1 else 0.0
    confidence = _confidence(variance)

    # High-risk sections
    high_risk: List[HighRiskSection] = [
        HighRiskSection(
            page=r.page,
            chunk_id=r.chunk_id,
            ai_score=r.ai_score,
            text_snippet=r.text_snippet,
        )
        for r in chunk_results
        if r.ai_score >= settings.high_risk_threshold
    ]
    # Sort by score descending
    high_risk.sort(key=lambda x: x.ai_score, reverse=True)

    # Stats
    ocr_pages = sum(1 for p in pages if p.used_ocr)
    llm_calls = sum(1 for r in chunk_results if r.llm_score is not None)
    provider_used = settings.llm_provider if llm_calls > 0 else None

    stats = ProcessingStats(
        total_pages=len(pages),
        total_chunks=len(chunk_results),
        total_words=sum(c.word_count for c in chunk_results),
        ocr_pages=ocr_pages,
        processing_time_seconds=round(time.perf_counter() - start_time, 2),
        llm_provider_used=provider_used,
        llm_calls_made=llm_calls,
    )

    logger.info(
        "Detection complete: %.1f%% AI | confidence=%s | chunks=%d | high_risk=%d",
        overall_percentage,
        confidence.value,
        len(chunk_results),
        len(high_risk),
    )

    return AnalysisResponse(
        overall_ai_percentage=overall_percentage,
        confidence=confidence,
        score_variance=variance,
        chunks=chunk_results,
        high_risk_sections=high_risk,
        processing_stats=stats,
    )
