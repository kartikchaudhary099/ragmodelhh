"""Smoke tests for the ThinkZen backend foundation."""

from fastapi.testclient import TestClient


def test_health_endpoint_returns_ok(client: TestClient) -> None:
    """Health endpoint should return 200 with status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "ThinkZen"
    assert "version" in body
    assert "timestamp" in body


def test_app_factory_creates_instance() -> None:
    """Application factory should produce a configured FastAPI app."""
    from app.main import create_app

    app = create_app()
    assert app.title == "ThinkZen"
