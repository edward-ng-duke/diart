"""Environment-driven settings for the diart server.

Settings are loaded from environment variables prefixed with ``DIART_``.
This module performs no disk I/O at import time; it only reads environment
variables via :class:`pydantic_settings.BaseSettings`.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the diart server, sourced from env vars."""

    model_config = SettingsConfigDict(case_sensitive=False, extra="ignore")

    port: int = Field(default=7007, validation_alias="DIART_PORT")
    host: str = Field(default="0.0.0.0", validation_alias="DIART_HOST")
    device: str = Field(default="cpu", validation_alias="DIART_DEVICE")
    models_dir: Path = Field(
        default=Path("/app/models"), validation_alias="DIART_MODELS_DIR"
    )
    max_concurrent_sessions: int = Field(
        default=8, validation_alias="DIART_MAX_CONCURRENT_SESSIONS"
    )
    max_frame_bytes: int = Field(
        default=10485760, validation_alias="DIART_MAX_FRAME_BYTES"
    )
    log_level: str = Field(default="INFO", validation_alias="DIART_LOG_LEVEL")


settings = Settings()
