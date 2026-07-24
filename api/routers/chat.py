"""Chat generation endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_service
from api.inference import GenPyInferenceService
from api.schemas import ChatRequest, GenerationResponse
from genpy_llm.generation import GenerationError

LOGGER = logging.getLogger("genpy_api")
router = APIRouter(tags=["chat"])
SERVICE_DEPENDENCY = Depends(get_service)


@router.post("/chat", response_model=GenerationResponse)
def chat(
    request: ChatRequest,
    service: GenPyInferenceService = SERVICE_DEPENDENCY,
) -> GenerationResponse:
    """Generate text from chat-style messages."""

    try:
        return service.chat(request)
    except GenerationError as exc:
        LOGGER.warning("api_chat_bad_request error=%s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
