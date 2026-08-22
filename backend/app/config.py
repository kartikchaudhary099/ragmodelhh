"""Application configuration via environment variables."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root: ThinkZen/ (two levels above backend/app/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Centralized settings loaded from environment and optional .env file."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = Field(default="ThinkZen", alias="APP_NAME")
    app_env: str = Field(default="development", alias="APP_ENV")
    debug: bool = Field(default=False, alias="DEBUG")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    backend_host: str = Field(default="0.0.0.0", alias="BACKEND_HOST")
    backend_port: int = Field(default=8000, alias="BACKEND_PORT")

    data_raw_dir: Path = Field(
        default=PROJECT_ROOT / "data" / "raw", alias="DATA_RAW_DIR"
    )
    data_processed_dir: Path = Field(
        default=PROJECT_ROOT / "data" / "processed", alias="DATA_PROCESSED_DIR"
    )
    data_samples_dir: Path = Field(
        default=PROJECT_ROOT / "data" / "samples", alias="DATA_SAMPLES_DIR"
    )

    # ------------------------------------------------------------------
    # Corpus selection (demo vs official MSMARCO-XI)
    # ------------------------------------------------------------------
    # Which corpus the query pipeline seeds on first request:
    #   "demo"     -> data/samples/demo_docs.json  (DEFAULT; explicitly-labelled fallback)
    #   "official" -> validated ai4bharat/MSMARCO-XI subset under data/official/
    # Set THINKZEN_CORPUS=official (after building the artifact) to serve official data.
    # If "official" is selected but the artifact is missing/invalid, ingestion RAISES —
    # it never silently falls back to demo data while claiming MSMARCO-XI.
    corpus_mode: str = Field(default="demo", alias="THINKZEN_CORPUS")
    data_official_dir: Path = Field(
        default=PROJECT_ROOT / "data" / "official", alias="DATA_OFFICIAL_DIR"
    )
    official_sample_path: Path = Field(
        default=PROJECT_ROOT / "data" / "official" / "msmarco_xi_sample.json",
        alias="OFFICIAL_SAMPLE_PATH",
    )
    official_provenance_path: Path = Field(
        default=PROJECT_ROOT / "data" / "official" / "provenance.json",
        alias="OFFICIAL_PROVENANCE_PATH",
    )


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
