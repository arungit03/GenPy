"""Health endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import get_service
from api.inference import GenPyInferenceService
from api.schemas import HealthResponse

router = APIRouter(tags=["health"])
SERVICE_DEPENDENCY = Depends(get_service)


@router.get("/health", response_model=HealthResponse)
def health(service: GenPyInferenceService = SERVICE_DEPENDENCY) -> HealthResponse:
    """Return readiness and selected device."""

    return service.health()
