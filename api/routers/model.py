"""Model metadata endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import get_service
from api.inference import GenPyInferenceService
from api.schemas import ModelResponse

router = APIRouter(tags=["model"])
SERVICE_DEPENDENCY = Depends(get_service)


@router.get("/model", response_model=ModelResponse)
def model_info(service: GenPyInferenceService = SERVICE_DEPENDENCY) -> ModelResponse:
    """Return metadata for the loaded model."""

    return service.model_info()
