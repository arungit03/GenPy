"""FastAPI application factory for GenPy offline serving."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from api.config import DEFAULT_API_CONFIG_PATH, APIConfig, load_api_config
from api.inference import GenPyInferenceService
from api.routers import chat, generate, health, model

LOGGER = logging.getLogger("genpy_api")


def create_app(
    config_path: Path | str = DEFAULT_API_CONFIG_PATH,
    *,
    config: APIConfig | None = None,
    service: GenPyInferenceService | None = None,
) -> FastAPI:
    """Create the GenPy offline FastAPI app."""

    api_config = config or load_api_config(config_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        LOGGER.info("api_lifespan_startup")
        app.state.service = service or GenPyInferenceService.from_config(api_config)
        yield
        LOGGER.info("api_lifespan_shutdown")

    app = FastAPI(
        title="GenPy Offline API",
        description="Local-only FastAPI server for GenPy checkpoints, LoRA, and quantization.",
        version="0.11.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.include_router(health.router)
    app.include_router(model.router)
    app.include_router(generate.router)
    app.include_router(chat.router)
    if service is not None:
        app.state.service = service
    return app


app = create_app()


__all__ = ["app", "create_app"]
