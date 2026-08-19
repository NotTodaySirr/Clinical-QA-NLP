from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.routes import router as api_router
from app.services.inference import model_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm and load NLP model on server startup
    model_service.load_model()
    yield
    # Cleanup / GPU memory release if necessary
    pass

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="FastAPI Backend for Transformer-based NLP Web Application",
    version="0.1.0",
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
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
