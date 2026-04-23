"""
Heuristic-based AI text detection.

Four signals are computed and combined:

1. Perplexity proxy  — AI text is more predictable (lower perplexity).
   We approximate with mean word log-frequency using NLTK's Brown corpus.

2. Burstiness score  — human writing has high sentence-length variance;
   AI text tends to be suspiciously uniform.

3. Repetition score  — AI text re-uses trigrams more than human text.

4. Lexical diversity — AI text favours common vocabulary (lower TTR).

Each signal is normalised to [0, 1] where 1 = "more AI-like".
Combined score = weighted average of the four signals.
"""

import math
import logging
import re
import string
from collections import Counter
from typing import List, Tuple, Optional
from functools import lru_cache

from models.schemas import HeuristicScores

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NLTK corpus bootstrap (lazy, one-time)
# ---------------------------------------------------------------------------
_WORD_FREQ: Optional[Counter] = None
_TOTAL_TOKENS: int = 0


def _get_word_freq() -> Tuple[Counter, int]:
    global _WORD_FREQ, _TOTAL_TOKENS
    if _WORD_FREQ is not None:
        return _WORD_FREQ, _TOTAL_TOKENS

    try:
        import nltk
        try:
            from nltk.corpus import brown
            brown.words()  # check availability
        except LookupError:
            nltk.download("brown", quiet=True)
            from nltk.corpus import brown

        words = [w.lower() for w in brown.words()]
        _WORD_FREQ = Counter(words)
        _TOTAL_TOKENS = len(words)
        logger.info("Loaded Brown corpus: %d tokens, %d unique", _TOTAL_TOKENS, len(_WORD_FREQ))
    except Exception as exc:
        logger.warning("Could not load Brown corpus (%s). Using fallback frequency.", exc)
        _WORD_FREQ = Counter()
        _TOTAL_TOKENS = 1

    return _WORD_FREQ, _TOTAL_TOKENS


# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _tokenize_words(text: str) -> List[str]:
    return text.lower().translate(_PUNCT_TABLE).split()


def _tokenize_sentences(text: str) -> List[str]:
    sentences = _SENT_SPLIT_RE.split(text.strip())
    return [s for s in sentences if s.strip()]


# ---------------------------------------------------------------------------
# Signal 1: Perplexity proxy
# ---------------------------------------------------------------------------
def _perplexity_score(words: List[str], freq: Counter, total: int) -> float:
    """
    Lower log-perplexity ≈ more predictable ≈ more AI-like.
    Returns a score in [0, 1] where 1 = very AI-like (low perplexity).
    """
    if not words:
        return 0.5

    vocab_size = len(freq) or 1
    log_prob_sum = 0.0
    for w in words:
        count = freq.get(w, 0)
        # Laplace smoothing
        p = (count + 1) / (total + vocab_size)
        log_prob_sum += math.log(p)

    avg_log_prob = log_prob_sum / len(words)
    perplexity = math.exp(-avg_log_prob)

    # Normalise: human text ~400–1000, AI text ~50–300 (rough empirical bounds)
    lo, hi = 50.0, 1200.0
    clamped = max(lo, min(hi, perplexity))
    # Invert: low perplexity → high AI score
    score = 1.0 - (clamped - lo) / (hi - lo)
    return round(score, 4)


# ---------------------------------------------------------------------------
# Signal 2: Burstiness (sentence-length variance)
# ---------------------------------------------------------------------------
def _burstiness_score(sentences: List[str]) -> float:
    """
    Low variance in sentence lengths → uniform → AI-like → high score.
    Returns [0, 1] where 1 = very uniform (AI-like).
    """
    if len(sentences) < 4:
        return 0.5

    lengths = [len(s.split()) for s in sentences]
    mean = sum(lengths) / len(lengths)
    if mean == 0:
        return 0.5

    variance = sum((l - mean) ** 2 for l in lengths) / len(lengths)
    cv = math.sqrt(variance) / mean  # coefficient of variation

    # Human CV typically 0.5–1.2; AI CV typically 0.1–0.4
    lo, hi = 0.05, 1.0
    clamped = max(lo, min(hi, cv))
    # Invert: low CV → high AI score
    score = 1.0 - (clamped - lo) / (hi - lo)
    return round(score, 4)


# ---------------------------------------------------------------------------
# Signal 3: Repetition (trigram density)
# ---------------------------------------------------------------------------
def _repetition_score(words: List[str]) -> float:
    """
    High ratio of repeated trigrams → AI-like → high score.
    Returns [0, 1].
    """
    if len(words) < 6:
        return 0.0

    trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
    total = len(trigrams)
    unique = len(set(trigrams))
    repeated = total - unique
    ratio = repeated / total

    # Normalise: human ~0.03–0.10, AI ~0.10–0.30
    lo, hi = 0.02, 0.30
    clamped = max(lo, min(hi, ratio))
    score = (clamped - lo) / (hi - lo)
    return round(score, 4)


# ---------------------------------------------------------------------------
# Signal 4: Lexical diversity (inverse TTR)
# ---------------------------------------------------------------------------
def _lexical_diversity_score(words: List[str]) -> float:
    """
    Low TTR (type-token ratio) → limited vocabulary → AI-like → high score.
    Returns [0, 1] where 1 = very low diversity (AI-like).
    """
    if not words:
        return 0.5

    ttr = len(set(words)) / len(words)

    # Human TTR ~0.4–0.7, AI TTR ~0.2–0.4
    lo, hi = 0.15, 0.75
    clamped = max(lo, min(hi, ttr))
    # Invert: low TTR → high AI score
    score = 1.0 - (clamped - lo) / (hi - lo)
    return round(score, 4)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------
SIGNAL_WEIGHTS = {
    "perplexity": 0.35,
    "burstiness": 0.25,
    "repetition": 0.20,
    "diversity": 0.20,
}


def analyze_chunk(text: str) -> HeuristicScores:
    """Compute heuristic AI-detection scores for a text chunk."""
    freq, total = _get_word_freq()
    words = _tokenize_words(text)
    sentences = _tokenize_sentences(text)

    p_score = _perplexity_score(words, freq, total)
    b_score = _burstiness_score(sentences)
    r_score = _repetition_score(words)
    d_score = _lexical_diversity_score(words)

    combined = (
        SIGNAL_WEIGHTS["perplexity"] * p_score
        + SIGNAL_WEIGHTS["burstiness"] * b_score
        + SIGNAL_WEIGHTS["repetition"] * r_score
        + SIGNAL_WEIGHTS["diversity"] * d_score
    )

    return HeuristicScores(
        perplexity_score=p_score,
        burstiness_score=b_score,
        repetition_score=r_score,
        lexical_diversity_score=d_score,
        combined_heuristic_score=round(combined, 4),
    )
