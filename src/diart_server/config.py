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
    # Tuned for the mean-centered + remapped [0,1] voiceprint score with the
    # extended cohort (14 x 4s 2news clips at 28-95s + 5 held-out AISHELL-3
    # speakers; scripts/compute_cohort_mean.py). Recalibrated 0.60 -> 0.82 for
    # the per-turn identity fix: on the cross-validation pool (5 AISHELL
    # speakers incl. the near-twin pair SSB0261/SSB0590 + the two 2news
    # speakers), turn-level impostor scores max ~0.79 while asserted-genuine
    # scores min ~0.84 (see docs/bugs/2026-07-06-voiceprint-false-accept.md
    # §3.3/§4.1e). The old 0.60 (2news-only cohort) could not reject
    # same-gender impostors (they score 0.72-0.80). Cross-channel genuine may
    # fall below 0.82 -> shown as unknown, never as a wrong name.
    match_threshold: float = Field(
        default=0.82, validation_alias="DIART_MATCH_THRESHOLD"
    )
    # Minimum turn/partial duration (seconds) for per-turn voiceprint
    # identification. Shorter slices give unreliable utterance embeddings
    # (genuine/impostor score ranges overlap below ~2s); such turns emit
    # name=None (no centroid fallback, no previous-turn fallback - both were
    # the false-accept bug). See 2026-07-06-voiceprint-false-accept.md §4.1b.
    min_identify_seconds: float = Field(
        default=2.0, validation_alias="DIART_MIN_IDENTIFY_SECONDS"
    )
    # Floor (seconds of PURE non-overlapped speech) for DISPLAY-ONLY candidate
    # identification. Turns/partials with pure speech in
    # [min_candidate_seconds, min_identify_seconds) still run the embed+match
    # chain, but the best match is emitted only as cand_name/cand_score (the
    # demo renders a dimmed "≈name score?") and never stamps ``name`` nor
    # updates ``_last_match`` — sub-2s utterance embeddings are unreliable
    # (2026-07-06 §4.1b), yet the measured score is useful for observability
    # and future threshold recalibration. Set >= min_identify_seconds to
    # disable candidate-only runs.
    min_candidate_seconds: float = Field(
        default=0.6, validation_alias="DIART_MIN_CANDIDATE_SECONDS"
    )
    # Staleness bound for the per-label identity echo (2026-08-20).
    # ``_last_match[label]`` is a display hint: the last TURN-VERIFIED identity
    # for a diart label. It is rewritten by every turn with at least
    # ``min_identify_seconds`` of pure speech, so it self-corrects -- EXCEPT
    # when the label's new occupant only ever speaks in shorter bursts, which
    # never write it. Then the previous occupant's name echoes on the
    # prediction stream forever, and diart really does hand two people the same
    # label whenever a silence longer than its 5s window separates them
    # (docs/bugs/2026-08-20-diart-label-merge-identity-leak.md).
    #
    # Deliberately a COUNT of non-confirming turns, not a similarity threshold.
    # Turn-to-turn embedding comparison was measured on CROSS-SEGMENT pairs of
    # the same speaker (the live case, not adjacent slices of one utterance)
    # and the genuine/impostor distributions OVERLAP at every turn length:
    # genuine min 0.569-0.587 vs impostor max 0.813-0.839, so any threshold
    # with usable detection costs 21-34% false contests
    # (scripts/verify_label_continuity.py prints the table). A count reuses the
    # already-calibrated confirm/reject decision instead of inventing a second,
    # uncalibratable one: after this many consecutive short turns that fail to
    # re-confirm the cached name, the echo is dropped. It bounds the leak; it
    # does not pretend to identify the new speaker.
    identity_echo_max_misses: int = Field(
        default=3, ge=1, validation_alias="DIART_IDENTITY_ECHO_MAX_MISSES"
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
