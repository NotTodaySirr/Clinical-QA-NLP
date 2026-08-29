from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    PROJECT_NAME: str = "Clinical Question Answering NLP API"
    API_V1_STR: str = "/api/v1"
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*",
    ]
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True

    # Model & Retriever Artifact Paths
    PUBMEDBERT_MODEL_PATH: str = str(BASE_DIR / "AI" / "saved_model" / "pubmedbert_finetuned")
    FAISS_INDEX_PATH: str = str(BASE_DIR / "AI" / "saved_model" / "faiss_medical.index")
    CONTEXTS_PATH: str = str(BASE_DIR / "AI" / "saved_model" / "contexts.npy")
    METADATA_PATH: str = str(BASE_DIR / "AI" / "saved_model" / "contexts_meta.npy")
    EMBEDDER_MODEL: str = "pritamdeka/S-PubMedBert-MS-MARCO"
    
    # Classification labels (Index 0: No, Index 1: Yes, Index 2: Maybe)
    ID2LABEL: dict = {0: "No", 1: "Yes", 2: "Maybe"}
    MAX_SEQUENCE_LENGTH: int = 512

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
