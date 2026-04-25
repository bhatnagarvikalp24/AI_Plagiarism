"""
Main orchestration layer — 4-signal ensemble detector.

Signal weights (redistribute when a signal is unavailable):
  1. GPT-4o LLM judge    — 0.45  (best calibrated for modern AI text)
  2. GPT-2 perplexity    — 0.25  (linguistic predictability signal)
  3. AI vocab signature  — 0.15  (GPT-4/Claude phrase patterns)
  4. Heuristics          — 0.15  (burstiness, repetition, diversity)

Note: Pre-trained RoBERTa classifiers were evaluated and excluded —
they score modern GPT-4/Claude text as ≥99% human (trained on GPT-2-era
data only). See services/classifier.py for rationale.
"""

import asyncio
import logging
import statistics
import time
from typing import List, Optional, Tuple

import httpx

from config import get_settings
from models.schemas import (
    AnalysisResponse, ChunkLabel, ChunkResult,
    ConfidenceLevel, HighRiskSection, ProcessingStats,
)
from services import heuristic_analyzer, llm_analyzer
from services.classifier import ai_vocab_score, gpt2_perplexity_score
from services.pdf_processor import PageText
from services.text_chunker import TextChunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _label(score: float) -> ChunkLabel:
    if score >= 0.70: return ChunkLabel.LIKELY_AI
    if score >= 0.50: return ChunkLabel.POSSIBLY_AI
    if score >= 0.30: return ChunkLabel.UNCERTAIN
    return ChunkLabel.LIKELY_HUMAN


def _confidence(variance: float) -> ConfidenceLevel:
    s = get_settings()
    if variance <= s.high_confidence_variance_threshold:   return ConfidenceLevel.HIGH
    if variance <= s.medium_confidence_variance_threshold: return ConfidenceLevel.MEDIUM
    return ConfidenceLevel.LOW


def _ensemble(
    llm:       Optional[float],
    gpt2:      Optional[float],
    vocab:     float,
    heuristic: float,
) -> float:
    """
    Weighted average of available signals.
    Missing signals have weight redistributed proportionally to the rest.
    """
    s = get_settings()
    pool = {}

    if llm  is not None: pool["llm"]  = (llm,  s.weight_llm)
    if gpt2 is not None: pool["gpt2"] = (gpt2, s.weight_gpt2)
    pool["vocab"]     = (vocab,     s.weight_vocab)
    pool["heuristic"] = (heuristic, s.weight_heuristic)

    total_w = sum(w for _, w in pool.values())
    score   = sum(v * w for v, w in pool.values()) / total_w
    return round(score, 4)


# ---------------------------------------------------------------------------
# Per-chunk processing
# ---------------------------------------------------------------------------
async def _process_chunk(
    chunk:       TextChunk,
    http_client: httpx.AsyncClient,
    semaphore:   asyncio.Semaphore,
    loop:        asyncio.AbstractEventLoop,
) -> Tuple[ChunkResult, float]:
    settings = get_settings()
    low_conf = chunk.word_count < settings.min_score_words

    # 1. Heuristics (CPU — thread pool)
    h_scores = await loop.run_in_executor(None, heuristic_analyzer.analyze_chunk, chunk.text)

    # 2. AI vocab score (instant, regex-based)
    vocab = ai_vocab_score(chunk.text)

    # 3. GPT-2 perplexity (CPU — thread pool, optional)
    gpt2 = None
    if settings.use_local_models and not low_conf:
        gpt2 = await loop.run_in_executor(None, gpt2_perplexity_score, chunk.text)
        if gpt2 is not None:
            h_scores = h_scores.model_copy(update={"perplexity_score": gpt2})

    # 4. LLM judge (async HTTP, rate-limited)
    llm: Optional[float] = None
    if not low_conf:
        async with semaphore:
            llm = await llm_analyzer.score_chunk(chunk.text, http_client)

    # 5. Ensemble
    ai_score = _ensemble(llm, gpt2, vocab, h_scores.combined_heuristic_score)

    result = ChunkResult(
        chunk_id          = chunk.chunk_id,
        page              = chunk.page_number,
        word_count        = chunk.word_count,
        ai_score          = ai_score,
        gpt2_score        = gpt2,
        vocab_score       = round(vocab, 4),
        llm_score         = llm,
        heuristic_score   = h_scores.combined_heuristic_score,
        heuristic_details = h_scores,
        label             = _label(ai_score),
        text_snippet      = chunk.text[:200].replace("\n", " "),
        low_confidence    = low_conf,
    )
    return result, ai_score


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------
async def run_detection(
    chunks: List[TextChunk],
    pages:  List[PageText],
) -> AnalysisResponse:
    settings   = get_settings()
    start_time = time.perf_counter()

    semaphore = asyncio.Semaphore(settings.llm_concurrency_limit)
    loop      = asyncio.get_event_loop()

    async with httpx.AsyncClient() as http_client:
        tasks         = [_process_chunk(c, http_client, semaphore, loop) for c in chunks]
        chunk_outputs = await asyncio.gather(*tasks)

    chunk_results: List[ChunkResult] = []
    scores:        List[float]       = []

    for result, score in chunk_outputs:
        chunk_results.append(result)
        if not result.low_confidence:
            scores.append(score)

    if not scores:
        scores = [r.ai_score for r in chunk_results]

    overall_pct = round(statistics.mean(scores) * 100, 2)
    variance    = round(statistics.variance(scores), 6) if len(scores) > 1 else 0.0
    confidence  = _confidence(variance)

    high_risk = sorted(
        [
            HighRiskSection(
                page         = r.page,
                chunk_id     = r.chunk_id,
                ai_score     = r.ai_score,
                text_snippet = r.text_snippet,
            )
            for r in chunk_results
            if r.ai_score >= settings.high_risk_threshold
        ],
        key=lambda x: x.ai_score,
        reverse=True,
    )

    ocr_pages   = sum(1 for p in pages if p.used_ocr)
    llm_calls   = sum(1 for r in chunk_results if r.llm_score  is not None)
    gpt2_used   = any(r.gpt2_score  is not None for r in chunk_results)

    stats = ProcessingStats(
        total_pages             = len(pages),
        total_chunks            = len(chunk_results),
        total_words             = sum(c.word_count for c in chunk_results),
        ocr_pages               = ocr_pages,
        processing_time_seconds = round(time.perf_counter() - start_time, 2),
        llm_provider_used       = settings.llm_provider if llm_calls > 0 else None,
        llm_calls_made          = llm_calls,
        gpt2_used               = gpt2_used,
        vocab_signal_used       = True,
    )

    logger.info(
        "Detection done: %.1f%% AI | %s confidence | chunks=%d | high_risk=%d | "
        "gpt2=%s llm_calls=%d",
        overall_pct, confidence.value, len(chunk_results), len(high_risk),
        gpt2_used, llm_calls,
    )

    return AnalysisResponse(
        overall_ai_percentage = overall_pct,
        confidence            = confidence,
        score_variance        = variance,
        chunks                = chunk_results,
        high_risk_sections    = high_risk,
        processing_stats      = stats,
    )
