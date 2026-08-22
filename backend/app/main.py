"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import health, query, stt
from app.config import get_settings
from app.exceptions import register_exception_handlers
from app.logging_config import setup_logging

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Application factory."""
    settings = get_settings()
    setup_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        logger.info(
            "Starting %s (env=%s, debug=%s)",
            settings.app_name,
            settings.app_env,
            settings.debug,
        )
        yield

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="ThinkZen — multilingual voice-native RAG for HH Goa 2026 Task 2",
        debug=settings.debug,
        lifespan=lifespan,
    )

    # Enable CORS for local development and voice interactions
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)

    # Register API Routers
    app.include_router(health.router)
    app.include_router(query.router)
    app.include_router(stt.router)

    # Mount static frontend files if directory exists
    frontend_static_dir = Path(__file__).resolve().parents[2] / "frontend" / "static"
    if frontend_static_dir.exists():
        app.mount("/", StaticFiles(directory=str(frontend_static_dir), html=True), name="static")

    return app


app = create_app()

