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
    # Lowered 20.0 -> 12.0 as defense-in-depth for BUG1 (long-monologue latency):
    # even with partial ASR disabled, worst-case final transcript latency drops
    # from ~20s to ~12s. See docs/bugs/2026-07-02-turn-asr-quality.md.
    max_turn_seconds: float = Field(
        default=12.0, validation_alias="DIART_MAX_TURN_SECONDS"
    )
    ring_buffer_seconds: float = Field(
        default=120.0, validation_alias="DIART_RING_BUFFER_SECONDS"
    )
    max_asr_concurrency: int = Field(
        default=2, validation_alias="DIART_MAX_ASR_CONCURRENCY"
    )
    # BUG1: run partial ASR on an open turn every this-many seconds of newly
    # accumulated open-turn audio, emitting a dimmed partial transcript that the
    # frontend later replaces with the finalized turn (keyed by turn_id). 0
    # disables the feature entirely (feature flag).
    partial_asr_interval: float = Field(
        default=5.0, validation_alias="DIART_PARTIAL_ASR_INTERVAL"
    )
    # BUG2: overlap (multi-speaker) fraction of a turn above which the mixed
    # audio is left INTACT and merely flagged (overlapped=True); at or below it
    # the overlapped samples are excised (zeroed) before ASR.
    overlap_max_ratio: float = Field(
        default=0.5, validation_alias="DIART_OVERLAP_MAX_RATIO"
    )
    # BUG3: a sub-0.25s finalized fragment is absorbed into the previous
    # same-speaker turn if that turn's end is within this many seconds of the
    # fragment's start (instead of being silently dropped).
    absorb_gap: float = Field(
        default=1.0, validation_alias="DIART_ABSORB_GAP"
    )


settings = Settings()
