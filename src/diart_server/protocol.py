"""Pydantic v2 message schemas for the diart_server WebSocket protocol.

These models define the wire format for all server -> client JSON messages
(``ReadyMessage``, ``PredictionMessage``, ``FinalMessage``, ``ErrorMessage``)
and the one client -> server JSON control message (``EndControl``).

The :class:`QueryParams` model validates the per-session query string
parameters that are passed when opening the WebSocket. ``parse_query``
wraps :class:`pydantic.ValidationError` as
:class:`diart_server.errors.BadAudioError` so the error boundary can map
it directly to a wire-format error message.

This module is pure: it does not import FastAPI/Starlette and performs no
I/O. Routes/sessions consume these schemas in later tasks.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from diart_server.errors import BadAudioError, ErrorCode


# ---------------------------------------------------------------------------
# Server -> client messages
# ---------------------------------------------------------------------------


class ReadyMessage(BaseModel):
    """Sent once after the session is fully initialized."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["ready"] = "ready"
    session_id: str
    sample_rate: int
    pipeline: Literal["diarization", "vad"]
    models: dict[str, str]
    config: dict[str, Any]


class PredictionMessage(BaseModel):
    """Streamed once per inference window with detected speaker info.

    ``name``/``score`` are a display hint echoing the last TURN-VERIFIED
    identity for this diart label: identity is computed per transcript turn
    from the turn's own audio (see ``TranscriptMessage``), never from the
    cluster centroid (centroid matching false-accepted impostors; see
    docs/bugs/2026-07-06-voiceprint-false-accept.md). ``None`` until the first
    identified turn for the label. ``embedding`` still carries the running
    cluster centroid for API compatibility.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["prediction"] = "prediction"
    start: float
    duration: float
    speaker: str
    embedding: list[float]
    name: str | None = None
    score: float | None = None
    samples_count: int


class FinalMessage(BaseModel):
    """Terminal summary message sent once when the stream ends."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["final"] = "final"
    rttm: str
    duration: float
    num_speakers: int
    stats: dict[str, Any]


class ErrorMessage(BaseModel):
    """Wire-format error message; ``code`` serializes as the enum value."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    type: Literal["error"] = "error"
    code: ErrorCode
    message: str


# ---------------------------------------------------------------------------
# Client -> server control messages
# ---------------------------------------------------------------------------


class EndControl(BaseModel):
    """Client signals end-of-stream. Distinguished from binary audio frames
    by being JSON with ``type == "end"``."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["end"] = "end"


class TranscriptMessage(BaseModel):
    """Per-utterance ASR result, sent when a speaker turn is finalized.

    ``name``/``score`` come from re-embedding THIS turn's own PURE
    (non-overlapped) speech and matching it against enrolled voiceprints. The
    identification gate is on pure non-overlapped seconds: turns with less
    than ``settings.min_identify_seconds`` of pure speech carry ``name=None``
    (too little audio for a reliable utterance embedding), and majority-overlap
    turns (mixed two-speaker audio) are never identified
    (docs/bugs/2026-08-05-identity-continuity-overlap-purity.md).
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["transcript"] = "transcript"
    speaker: str
    name: str | None = None
    score: float | None = None
    start: float
    end: float
    text: str
    lang: str = ""
    # Additive fields (BUG1/BUG2). All optional with back-compat defaults so
    # existing messages/tests stay valid under the extra="forbid" config.
    #
    # BUG1: a stable id shared by an open turn's partial transcripts and its
    # final transcript. The frontend keys timeline entries by turn_id and
    # REPLACES a partial with the final (empty -> legacy prepend behaviour).
    turn_id: str = ""
    # BUG1: True while this is a partial (in-progress) transcript of an open
    # turn; the final transcript for the same turn_id carries partial=False.
    partial: bool = False
    # BUG2: True when the turn's audio is mostly multi-speaker (majority
    # overlap left intact and flagged rather than excised).
    overlapped: bool = False
    # BUG2: fraction of the turn [start,end] that overlapped another speaker.
    overlap_ratio: float = 0.0
    # Display-only best-match candidate from THIS turn's own pure speech,
    # emitted whenever the embed+match chain ran (pure seconds >=
    # settings.min_candidate_seconds) — including runs too short or too
    # low-scoring to stamp a confirmed ``name``. NEVER a verified identity:
    # clients must render it as uncertain (the demo shows a dimmed
    # "≈name score?"). None when identification did not run at all
    # (majority overlap, no cohort, no store, or pure speech below the floor).
    cand_name: str | None = None
    cand_score: float | None = None
    # 2026-08-20: True on the turn where the server DROPPED this diart label's
    # stale identity echo, after settings.identity_echo_max_misses consecutive
    # short turns failed to re-confirm the cached name. diart hands two people
    # the same label whenever a silence longer than its 5s window separates
    # them; this bounds how long the first occupant's name can keep appearing
    # on the prediction stream over the second one's audio. It is a staleness
    # signal, NOT a claim about who is speaking: clients must drop any name
    # they cached for this label, and must not infer a new one.
    # docs/bugs/2026-08-20-diart-label-merge-identity-leak.md
    contested: bool = False


class VoiceprintInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    n_clips: int


class EnrollResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["enrolled"] = "enrolled"
    name: str
    embedding_model: str
    n_clips: int


# ---------------------------------------------------------------------------
# Query parameters
# ---------------------------------------------------------------------------


class QueryParams(BaseModel):
    """Per-session configuration parsed from the WebSocket query string.

    Defaults and ranges come from STANDALONE_SERVER.md §4.2. All fields
    are optional in the request; missing fields fall back to defaults.
    """

    model_config = ConfigDict(extra="forbid")

    sample_rate: int = Field(default=16000, ge=8000, le=48000)
    pipeline: Literal["diarization", "vad"] = "diarization"
    segmentation_model: str = Field(default="pyannote/segmentation-3.0")
    embedding_model: str = Field(default="pyannote/embedding")
    # --- diart clustering knobs (measured 2026-08-20; read before tuning) ---
    # These three are diart's README hyper-parameters for AMI, which were
    # measured with pyannote.audio<3.1 and pyannote/segmentation v1 (SOFT
    # per-speaker probabilities). We run pyannote/segmentation-3.0, a POWERSET
    # model that diart adapts via PowersetAdapter -> Powerset.to_multilabel()
    # with soft=False, so the segmentation this pipeline sees is strictly
    # BINARY (measured: the only two values ever emitted are 0.0 and 1.0).
    # Consequences, all measured at HEAD 1fb7baa and pinned by
    # tests/diart_server/test_diart_param_invariants.py:
    #
    # * tau_active is INERT. `max(seg) >= 0.507` <=> `max == 1`, so a single
    #   17 ms frame marks a speaker active; and with latency == step == 0.5,
    #   DelayedAggregation.num_overlapping_windows == 1, so Binarize also sees
    #   raw 0/1 and is a no-op for ANY threshold in (0, 1).
    # * rho_update=0.006 means 0.6% of a 5s window = 30 ms of speech is enough
    #   to spawn or update a centroid. Sweeping it over 0.006/0.05/0.10
    #   produced BIT-IDENTICAL diarizations on every probe scenario.
    # * delta_new=1.057 is SATURATED: the cross-speaker cosine distance never
    #   reaches it (measured max 1.047 in the wespeaker space, 1.065 in
    #   pyannote/embedding), so "this matches nobody -> new speaker" never
    #   fires and new labels appear only via the one-to-one assignment
    #   constraint. Its known cost is merging two speakers separated by a
    #   silence longer than `duration`; that is handled at the TURN level
    #   (settings.identity_echo_max_misses), NOT here. Lowering it was
    #   measured and is WORSE: 0.80 -> 4 labels and 0.75 -> 5 labels on
    #   data/2news.mp3, which really has 2. Re-run
    #   scripts/eval/diart_cluster_sweep.py before changing any of these.
    tau_active: float = Field(default=0.507, ge=0.0, le=1.0)
    rho_update: float = Field(default=0.006, ge=0.0, le=1.0)
    delta_new: float = Field(default=1.057, ge=0.0, le=2.0)
    latency: float = Field(default=0.5, ge=0.5, le=5.0)
    max_speakers: int = Field(default=10, ge=2, le=20)
    duration: float = Field(default=5.0, gt=0.0)
    step: float = Field(default=0.5, ge=0.1, le=1.0)


def parse_query(qp: dict) -> QueryParams:
    """Validate raw query-string params and wrap errors as ``BadAudioError``.

    Starlette's ``request.query_params`` is a ``MultiDict``; callers should
    pass ``dict(query_params)`` since all our fields are single-valued.
    """
    try:
        return QueryParams.model_validate(qp)
    except ValidationError as exc:
        raise BadAudioError(str(exc)) from exc


__all__ = [
    "ReadyMessage",
    "PredictionMessage",
    "FinalMessage",
    "ErrorMessage",
    "EndControl",
    "QueryParams",
    "parse_query",
    "TranscriptMessage",
    "VoiceprintInfo",
    "EnrollResponse",
]
