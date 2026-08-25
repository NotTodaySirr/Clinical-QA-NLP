import sys
import io
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Ensure UTF-8 stdout encoding on Windows
if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Ensure backend directory is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.core.config import settings
from app.api.routes import router as api_router
from app.services.inference import model_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm and load PubMedBERT & FAISS index on server startup
    model_service.load_model()
    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="FastAPI Backend for Open-Domain Clinical Question Answering (PubMedBERT + FAISS)",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routes
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/")
def root():
    return {
        "message": f"Welcome to {settings.PROJECT_NAME}",
        "docs_url": "/docs",
        "api_v1": settings.API_V1_STR,
    }


if __name__ == "__main__":
    print("=" * 65)
    print(f"[SERVER] Starting {settings.PROJECT_NAME}")
    print(f"[SERVER] Local URL          : http://127.0.0.1:{settings.PORT}")
    print(f"[SERVER] API Documentation  : http://127.0.0.1:{settings.PORT}/docs")
    print(f"[SERVER] Health Check        : http://127.0.0.1:{settings.PORT}/api/v1/health")
    print("=" * 65)
    uvicorn.run("app.main:app", host="127.0.0.1", port=settings.PORT, reload=False)
