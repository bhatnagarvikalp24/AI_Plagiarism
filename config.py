import os
from typing import Optional
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # --- LLM Provider ---
    llm_provider: str = "openai"  # "openai" | "ollama"

    # --- OpenAI ---
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-3.5-turbo"
    openai_max_tokens: int = 200
    openai_temperature: float = 0.0

    # --- Ollama ---
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "mistral"

    # --- Detection Weights ---
    llm_weight: float = 0.6
    heuristic_weight: float = 0.4

    # --- Chunking ---
    chunk_min_words: int = 300
    chunk_max_words: int = 500

    # --- Confidence Thresholds (variance of chunk scores) ---
    high_confidence_variance_threshold: float = 0.03
    medium_confidence_variance_threshold: float = 0.08

    # --- High-Risk Score Threshold ---
    high_risk_threshold: float = 0.70

    # --- File Upload ---
    max_file_size_mb: int = 50
    upload_dir: str = "/tmp/ai_detection_uploads"

    # --- Async Concurrency ---
    llm_concurrency_limit: int = 5  # max parallel LLM calls

    # --- Heuristic Tuning ---
    # AI text usually has lower perplexity; we normalize against these reference values
    max_perplexity_ref: float = 1000.0
    min_perplexity_ref: float = 50.0

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
