from fastapi import APIRouter, HTTPException, status
from app.schemas.predict import PredictionRequest, PredictionResponse, HealthResponse
from app.services.inference import model_service
from app.core.config import settings

router = APIRouter()

@router.get("/health", response_model=HealthResponse, summary="Health Check")
async def health_check():
    return HealthResponse(
        status="healthy",
        app_name=settings.PROJECT_NAME,
        version="0.1.0",
        model_loaded=model_service.is_ready
    )

@router.post("/predict", response_model=PredictionResponse, summary="Run NLP Inference")
async def run_prediction(request: PredictionRequest):
    try:
        response = model_service.predict(
            text=request.text,
            parameters=request.parameters
        )
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(e)}"
        )
