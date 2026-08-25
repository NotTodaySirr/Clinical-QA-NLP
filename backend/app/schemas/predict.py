from pydantic import BaseModel, Field
from typing import List, Optional


class ScoreItem(BaseModel):
    label: str = Field(..., description="Class verdict label ('Yes', 'No', 'Maybe')")
    score: float = Field(..., description="Normalized confidence probability (0.0 to 1.0)")


class RetrievedCandidateItem(BaseModel):
    rank: int = Field(..., description="1-indexed retrieval ranking")
    pubid: Optional[int] = Field(None, description="PubMed article ID if available")
    question: Optional[str] = Field(None, description="Original question in KB if available")
    context: str = Field(..., description="Retrieved medical abstract snippet")
    similarity_score: float = Field(..., description="Cosine similarity score (0.0 to 1.0)")


class PredictionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=2,
        max_length=2000,
        description="Clinical question to evaluate",
        examples=["Does hydroxychloroquine improve survival in hospitalized COVID-19 patients?"],
    )
    context: Optional[str] = Field(
        None,
        max_length=15000,
        description="Optional custom medical abstract context. If omitted, vector retriever queries the knowledge base.",
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of candidate medical abstracts to retrieve",
    )


class PredictionResponse(BaseModel):
    task: str = Field(default="Clinical Question Answering", description="NLP Pipeline Task Name")
    question: str = Field(..., description="Input clinical question")
    prediction: str = Field(..., description="Primary clinical decision ('Yes', 'No', 'Maybe')")
    confidence: float = Field(..., description="Highest class probability (0.0 to 1.0)")
    scores: List[ScoreItem] = Field(..., description="Probability distribution across all 3 classes")
    retrieved_context: str = Field(..., description="Medical evidence abstract fed to the clinical reader")
    candidates: List[RetrievedCandidateItem] = Field(
        default_factory=list,
        description="Top-k retrieved candidate abstracts with cosine similarity scores",
    )
    mode: str = Field(..., description="'retriever_reader' (ODQA) or 'direct_reading' (Custom Context)")
    retrieval_time_ms: float = Field(..., description="Retriever execution time in milliseconds")
    inference_time_ms: float = Field(..., description="PubMedBERT reader execution time in milliseconds")
    total_time_ms: float = Field(..., description="End-to-end processing time in milliseconds")
    device: str = Field(..., description="Compute device ('cuda' or 'cpu')")
    model_name: str = Field(default="PubMedBERT Fine-Tuned (3-class)", description="Reader model architecture")


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    model_loaded: bool
    retriever_loaded: bool
    num_indexed_contexts: int
    device: str
