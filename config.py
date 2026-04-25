import os
from typing import Optional
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # --- LLM Provider ---
    llm_provider: str = "openai"          # "openai" | "ollama"

    # --- OpenAI ---
    openai_api_key:    Optional[str] = None
    openai_model:      str = "gpt-4o"
    openai_max_tokens: int = 200
    openai_temperature: float = 0.0

    # --- Ollama ---
    ollama_base_url: str = "http://localhost:11434"
    ollama_model:    str = "mistral"

    # --- Local models ---
    # GPT-2 perplexity is unreliable for book/formal text (distilgpt2 was
    # trained on web text and assigns lower perplexity to narrative prose
    # than to AI-generated academic text, giving the wrong direction).
    # Disabled by default; vocab_signature is the reliable local signal.
    use_local_models: bool = False
    gpt2_model:       str  = "distilgpt2"

    # --- Ensemble weights (redistribute when a signal is unavailable) ---
    # LLM (GPT-4o) is the most reliable signal for modern AI text.
    # AI vocab signature is the best local signal (no model needed).
    weight_llm:       float = 0.55
    weight_gpt2:      float = 0.15   # only active when use_local_models=True
    weight_vocab:     float = 0.25
    weight_heuristic: float = 0.20   # burstiness + repetition + diversity

    # --- Chunking ---
    chunk_min_words: int = 300
    chunk_max_words: int = 500
    min_score_words: int = 150   # chunks below this get low_confidence=True

    # --- Confidence thresholds (variance of chunk scores) ---
    high_confidence_variance_threshold:   float = 0.03
    medium_confidence_variance_threshold: float = 0.08

    # --- High-risk threshold ---
    high_risk_threshold: float = 0.70

    # --- File upload ---
    max_file_size_mb: int  = 50
    upload_dir:       str  = "/tmp/ai_detection_uploads"

    # --- Async concurrency ---
    llm_concurrency_limit: int = 2

    class Config:
        env_file            = ".env"
        env_file_encoding   = "utf-8"
        extra               = "ignore"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
