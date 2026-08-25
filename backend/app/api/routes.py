from fastapi import APIRouter, HTTPException, status
from app.schemas.predict import PredictionRequest, PredictionResponse, HealthResponse
from app.services.inference import model_service
from app.core.config import settings

router = APIRouter()


@router.get("/health", response_model=HealthResponse, summary="Clinical QA Health Check")
async def health_check():
    return HealthResponse(
        status="healthy" if model_service.is_ready else "initializing",
        app_name=settings.PROJECT_NAME,
        version="1.0.0",
        model_loaded=model_service.is_ready and model_service.model is not None,
        retriever_loaded=model_service.faiss_index is not None,
        num_indexed_contexts=len(model_service.contexts),
        device=model_service.device,
    )


@router.post("/predict", response_model=PredictionResponse, summary="Execute Clinical QA Reasoning")
async def run_prediction(request: PredictionRequest):
    if not model_service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is still initializing or not ready.",
        )
    try:
        response = model_service.predict(
            question=request.question,
            context=request.context,
            top_k=request.top_k,
        )
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Clinical inference failed: {str(e)}",
        )
