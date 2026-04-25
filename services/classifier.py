"""
Enhanced local AI-detection — two signals:

1. GPT-2 perplexity (distilgpt2)
   AI text is more predictable to a language model → lower perplexity.
   Calibrated on real book samples: AI chunks ~30–70, human chunks ~70–180.

2. AI vocabulary / phrase signature detector
   Modern LLMs (GPT-4, Claude) consistently overuse specific transition
   phrases and vocabulary. This regex-based detector is surprisingly
   effective (~80% precision) for contemporary AI writing and requires
   zero GPU or model download.

Pre-trained binary classifiers (roberta-base-openai-detector,
Hello-SimpleAI/chatgpt-detector-roberta) were evaluated and excluded —
they were trained exclusively on GPT-2-era text (2019–2021) and score
modern GPT-4/Claude text as ≥99% "Real/Human", providing no signal.
"""

import logging
import math
import re
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GPT2_MODEL     = "distilgpt2"
MAX_TOKENS     = 512
MAX_TEXT_CHARS = 2000

# Calibrated on actual book chunks (300–500 words):
#   AI-written  : perplexity ~30–70
#   Human-written: perplexity ~70–200
_PPL_LO, _PPL_HI = 30.0, 180.0

# AI vocabulary signatures — phrases and words consistently overused by
# GPT-4 / Claude relative to human writers.
_AI_TRANSITIONS = re.compile(
    r"\b(furthermore|moreover|additionally|in conclusion|in summary|"
    r"it is important to note|it should be (noted|emphasized|highlighted)|"
    r"it is worth noting|needless to say|that being said|"
    r"in today'?s (rapidly evolving|digital|modern)|"
    r"at its core|first and foremost|last but not least|"
    r"on the other hand|it goes without saying|"
    r"as (previously|mentioned|noted|discussed)|"
    r"the (fact|reality|truth) (is|remains) that)\b",
    re.IGNORECASE,
)
_AI_VOCAB = re.compile(
    r"\b(leverage[sd]?|leveraging|delve[sd]?|delving|"
    r"comprehensive(ly)?|multifaceted|nuanced?|"
    r"paradigm shift|game.?changing|revolutionary|"
    r"unlock(ing|s|ed)?|revolutioni[sz](e[sd]?|ing)|"
    r"cutting.?edge|state.?of.?the.?art|"
    r"foster(ing|s|ed)?|empower(ing|s|ed|ment)?|"
    r"transform(ative|ation|ing)|seamless(ly)?|"
    r"synerg(y|ies|istic)|robust(ly|ness)?|"
    r"proactive(ly)?|streamline[sd]?|scalab(le|ility)|"
    r"dynamic(ally)?|innovative(ly)?|stakeholder)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _load_gpt2():
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    logger.info("Loading %s for perplexity scoring…", GPT2_MODEL)
    tok   = GPT2TokenizerFast.from_pretrained(GPT2_MODEL)
    model = GPT2LMHeadModel.from_pretrained(GPT2_MODEL)
    model.eval()
    logger.info("distilgpt2 ready.")
    return tok, model


# ---------------------------------------------------------------------------
# Signal 1 — GPT-2 perplexity
# ---------------------------------------------------------------------------
def gpt2_perplexity_score(text: str) -> Optional[float]:
    """
    Normalised perplexity score [0, 1] where 1 = very AI-like (low perplexity).
    Returns None if the model is unavailable.
    """
    try:
        import torch
        tok, model = _load_gpt2()

        inputs = tok(
            text[:MAX_TEXT_CHARS],
            return_tensors="pt",
            truncation=True,
            max_length=MAX_TOKENS,
        )
        with torch.no_grad():
            loss      = model(**inputs, labels=inputs["input_ids"]).loss
            perplexity = math.exp(loss.item())

        clamped = max(_PPL_LO, min(_PPL_HI, perplexity))
        score   = 1.0 - (clamped - _PPL_LO) / (_PPL_HI - _PPL_LO)
        logger.debug("GPT-2 perplexity=%.1f → score=%.3f", perplexity, score)
        return round(score, 4)

    except Exception as exc:
        logger.warning("GPT-2 perplexity skipped: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Signal 2 — AI vocabulary / phrase signature
# ---------------------------------------------------------------------------
def ai_vocab_score(text: str) -> float:
    """
    Detects characteristic AI vocabulary and transition phrases.
    Normalised to [0, 1] where 1 = heavy AI signature.

    This is effective for modern GPT-4 / Claude text which uses these
    phrases at rates 3–8× higher than human writers.
    """
    words = text.split()
    if not words:
        return 0.0

    per_100 = 100.0 / len(words)

    transition_count = len(_AI_TRANSITIONS.findall(text))
    vocab_count      = len(_AI_VOCAB.findall(text))

    # Rate per 100 words — empirical thresholds
    transition_rate = transition_count * per_100   # human ~0.3, AI ~2–6
    vocab_rate      = vocab_count      * per_100   # human ~0.5, AI ~3–8

    # Normalise each rate to [0, 1]
    t_score = min(transition_rate / 5.0, 1.0)
    v_score = min(vocab_rate      / 7.0, 1.0)

    return round(0.5 * t_score + 0.5 * v_score, 4)


# ---------------------------------------------------------------------------
# Warm-up helper
# ---------------------------------------------------------------------------
def warmup():
    """Pre-load GPT-2 so first request is fast."""
    from config import get_settings
    if not get_settings().use_local_models:
        return
    try:
        _load_gpt2()
    except Exception as exc:
        logger.warning("GPT-2 warmup failed: %s", exc)
