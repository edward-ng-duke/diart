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
    voiceprints_dir: Path = Field(
        default=Path("voiceprints"), validation_alias="DIART_VOICEPRINTS_DIR"
    )
    asr_base_url: str = Field(
        default="http://10.0.0.32:6001/v1", validation_alias="DIART_ASR_BASE_URL"
    )
    asr_model: str = Field(
        default="models/Qwen3-ASR-1.7B", validation_alias="DIART_ASR_MODEL"
    )
    asr_api_key: str = Field(default="EMPTY", validation_alias="DIART_ASR_API_KEY")
    asr_timeout: float = Field(default=60, validation_alias="DIART_ASR_TIMEOUT")
    # Tuned for the mean-centered + remapped [0,1] voiceprint score (see
    # docs/bugs/2026-06-30-voiceprint-score-compression.md). Measured on the
    # real embed->match chain (data/2news.mp3 clips A1/A2 same, B different):
    #   wespeaker:        same=0.753 diff=0.462
    #   pyannote/embedding: same=0.712 diff=0.563
    # 0.60 sits above both different-speaker scores and below both
    # same-speaker scores, separating them for both working embedding models.
    match_threshold: float = Field(
        default=0.6, validation_alias="DIART_MATCH_THRESHOLD"
    )
    silence_gap: float = Field(default=0.7, validation_alias="DIART_SILENCE_GAP")
    max_turn_seconds: float = Field(
        default=20.0, validation_alias="DIART_MAX_TURN_SECONDS"
    )
    ring_buffer_seconds: float = Field(
        default=120.0, validation_alias="DIART_RING_BUFFER_SECONDS"
    )
    max_asr_concurrency: int = Field(
        default=2, validation_alias="DIART_MAX_ASR_CONCURRENCY"
    )


settings = Settings()
