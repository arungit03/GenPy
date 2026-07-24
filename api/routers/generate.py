"""Prompt generation endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_service
from api.inference import GenPyInferenceService
from api.schemas import GenerationRequest, GenerationResponse
from genpy_llm.generation import GenerationError

LOGGER = logging.getLogger("genpy_api")
router = APIRouter(tags=["generation"])
SERVICE_DEPENDENCY = Depends(get_service)


@router.post("/generate", response_model=GenerationResponse)
def generate(
    request: GenerationRequest,
    service: GenPyInferenceService = SERVICE_DEPENDENCY,
) -> GenerationResponse:
    """Generate text from a prompt."""

    try:
        return service.generate(request)
    except GenerationError as exc:
        LOGGER.warning("api_generate_bad_request error=%s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
