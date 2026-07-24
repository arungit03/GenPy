"""FastAPI dependencies for the GenPy offline API."""

from __future__ import annotations

from typing import cast

from fastapi import HTTPException, Request

from api.inference import GenPyInferenceService


def get_service(request: Request) -> GenPyInferenceService:
    """Return the process-local inference service."""

    service = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    return cast(GenPyInferenceService, service)
