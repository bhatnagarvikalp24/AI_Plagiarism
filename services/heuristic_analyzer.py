"""
Heuristic-based AI text detection — three statistical signals.

The Brown-corpus perplexity proxy has been removed; real GPT-2 perplexity
is now handled by services/classifier.py.  The remaining three signals are:

1. Burstiness   — human writing has high sentence-length variance;
                  AI text is suspiciously uniform.
2. Repetition   — AI text re-uses trigrams more than human text.
3. Lexical div. — AI text favours common vocabulary (lower TTR).

Each signal is normalised to [0, 1] where 1 = more AI-like.
Combined score = weighted average of the three signals.
"""

import logging
import math
import re
import string
from collections import Counter
from typing import List

from models.schemas import HeuristicScores

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_PUNCT_TABLE   = str.maketrans("", "", string.punctuation)


def _tokenize_words(text: str) -> List[str]:
    return text.lower().translate(_PUNCT_TABLE).split()


def _tokenize_sentences(text: str) -> List[str]:
    return [s for s in _SENT_SPLIT_RE.split(text.strip()) if s.strip()]


# ---------------------------------------------------------------------------
# Signal 1 — Burstiness (sentence-length variance)
# ---------------------------------------------------------------------------
def _burstiness_score(sentences: List[str]) -> float:
    """
    Low coefficient-of-variation → uniform lengths → AI-like → high score.
    Range [0, 1].
    """
    if len(sentences) < 4:
        return 0.5

    lengths = [len(s.split()) for s in sentences]
    mean    = sum(lengths) / len(lengths)
    if mean == 0:
        return 0.5

    variance = sum((l - mean) ** 2 for l in lengths) / len(lengths)
    cv       = math.sqrt(variance) / mean   # coefficient of variation

    # Human CV ~0.5–1.2 | AI CV ~0.1–0.4
    lo, hi   = 0.05, 1.0
    clamped  = max(lo, min(hi, cv))
    score    = 1.0 - (clamped - lo) / (hi - lo)
    return round(score, 4)


# ---------------------------------------------------------------------------
# Signal 2 — Repetition (trigram density)
# ---------------------------------------------------------------------------
def _repetition_score(words: List[str]) -> float:
    """
    High ratio of repeated trigrams → AI-like → high score.
    Range [0, 1].
    """
    if len(words) < 6:
        return 0.0

    trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
    total    = len(trigrams)
    repeated = total - len(set(trigrams))
    ratio    = repeated / total

    # Human ~0.03–0.10 | AI ~0.10–0.30
    lo, hi   = 0.02, 0.30
    clamped  = max(lo, min(hi, ratio))
    score    = (clamped - lo) / (hi - lo)
    return round(score, 4)


# ---------------------------------------------------------------------------
# Signal 3 — Lexical diversity (inverse TTR)
# ---------------------------------------------------------------------------
def _lexical_diversity_score(words: List[str]) -> float:
    """
    Low TTR (type-token ratio) → limited vocabulary → AI-like → high score.
    Range [0, 1].
    """
    if not words:
        return 0.5

    ttr = len(set(words)) / len(words)

    # Human TTR ~0.4–0.7 | AI TTR ~0.2–0.4
    lo, hi   = 0.15, 0.75
    clamped  = max(lo, min(hi, ttr))
    score    = 1.0 - (clamped - lo) / (hi - lo)
    return round(score, 4)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------
_WEIGHTS = {"burstiness": 0.40, "repetition": 0.30, "diversity": 0.30}


def analyze_chunk(text: str) -> HeuristicScores:
    """Compute heuristic AI-detection scores for a text chunk."""
    words     = _tokenize_words(text)
    sentences = _tokenize_sentences(text)

    b_score = _burstiness_score(sentences)
    r_score = _repetition_score(words)
    d_score = _lexical_diversity_score(words)

    combined = (
        _WEIGHTS["burstiness"] * b_score
        + _WEIGHTS["repetition"] * r_score
        + _WEIGHTS["diversity"]  * d_score
    )

    # perplexity_score field is now filled by classifier.py (gpt2_perplexity_score)
    # We set it to 0.5 (neutral) here so the schema stays valid.
    return HeuristicScores(
        perplexity_score=0.5,
        burstiness_score=b_score,
        repetition_score=r_score,
        lexical_diversity_score=d_score,
        combined_heuristic_score=round(combined, 4),
    )
