"""Test that configuration loads from defaults without .env."""

from app.config import Settings, get_settings


def test_settings_defaults() -> None:
    """Settings should have sensible defaults when no .env is present."""
    settings = Settings()
    assert settings.app_name == "ThinkZen"
    assert settings.backend_port == 8000
    assert settings.log_level == "INFO"


def test_get_settings_is_cached() -> None:
    """get_settings should return the same cached instance."""
    get_settings.cache_clear()
    first = get_settings()
    second = get_settings()
    assert first is second
