from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class ConfidenceLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class ChunkLabel(str, Enum):
    LIKELY_AI = "Likely AI"
    POSSIBLY_AI = "Possibly AI"
    LIKELY_HUMAN = "Likely Human"
    UNCERTAIN = "Uncertain"


class HeuristicScores(BaseModel):
    perplexity_score: float = Field(..., ge=0.0, le=1.0, description="Normalized perplexity (higher = more AI-like)")
    burstiness_score: float = Field(..., ge=0.0, le=1.0, description="Sentence length uniformity (higher = more AI-like)")
    repetition_score: float = Field(..., ge=0.0, le=1.0, description="Repeated n-gram density")
    lexical_diversity_score: float = Field(..., ge=0.0, le=1.0, description="Inverse of lexical diversity")
    combined_heuristic_score: float = Field(..., ge=0.0, le=1.0)


class ChunkResult(BaseModel):
    chunk_id: int
    page: int
    word_count: int
    ai_score: float = Field(..., ge=0.0, le=1.0)
    llm_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    heuristic_score: float = Field(..., ge=0.0, le=1.0)
    heuristic_details: HeuristicScores
    label: ChunkLabel
    text_snippet: str = Field(..., description="First 200 characters of chunk")


class HighRiskSection(BaseModel):
    page: int
    chunk_id: int
    ai_score: float
    text_snippet: str


class ProcessingStats(BaseModel):
    total_pages: int
    total_chunks: int
    total_words: int
    ocr_pages: int
    processing_time_seconds: float
    llm_provider_used: Optional[str]
    llm_calls_made: int


class AnalysisResponse(BaseModel):
    overall_ai_percentage: float = Field(..., ge=0.0, le=100.0)
    confidence: ConfidenceLevel
    score_variance: float
    chunks: List[ChunkResult]
    high_risk_sections: List[HighRiskSection]
    processing_stats: ProcessingStats


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
