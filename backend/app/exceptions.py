"""Application-level exception types and handlers."""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class ThinkZenError(Exception):
    """Base exception for ThinkZen application errors."""

    def __init__(self, message: str, status_code: int = 500) -> None:
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class ConfigurationError(ThinkZenError):
    """Raised when required configuration is missing or invalid."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=500)


def register_exception_handlers(app: FastAPI) -> None:
    """Register global exception handlers on the FastAPI app."""

    @app.exception_handler(ThinkZenError)
    async def thinkzen_rag_error_handler(
        _request: Request, exc: ThinkZenError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.message, "type": exc.__class__.__name__},
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(
        _request: Request, exc: Exception
    ) -> JSONResponse:
        from app.config import get_settings

        settings = get_settings()
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "type": "InternalError",
                "detail": str(exc) if settings.debug else None,
            },
        )
