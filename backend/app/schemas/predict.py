from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Input text to be processed by the model")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Optional inference parameters")

class ScoreItem(BaseModel):
    label: str
    score: float

class PredictionResponse(BaseModel):
    task: str = Field(..., description="NLP task name (e.g. classification, ner, summarization)")
    prediction: Any = Field(..., description="Primary prediction outcome")
    confidence: Optional[float] = Field(None, description="Overall confidence score (0.0 to 1.0)")
    scores: Optional[List[ScoreItem]] = Field(default_factory=list, description="Distribution of class/token scores")
    execution_time_ms: float = Field(..., description="Model inference runtime in milliseconds")
    model_version: str = Field(default="baseline-v0.1.0", description="Model architecture or checkpoint version")

class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    model_loaded: bool
