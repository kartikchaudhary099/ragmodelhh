"""Health check endpoints."""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter

from app import __version__

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check() -> dict:
    """Return service health status for smoke tests and monitoring."""
    logger.debug("Health check requested")
    return {
        "status": "ok",
        "service": "ThinkZen",
        "version": __version__,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
