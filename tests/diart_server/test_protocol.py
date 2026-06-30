"""Tests for diart_server.protocol message schemas and QueryParams."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from diart_server.errors import BadAudioError, ErrorCode
from diart_server.protocol import (
    EndControl,
    ErrorMessage,
    FinalMessage,
    PredictionMessage,
    QueryParams,
    ReadyMessage,
    parse_query,
)


# ---------------------------------------------------------------------------
# Round-trip serialization for each message type
# ---------------------------------------------------------------------------


def test_ready_message_roundtrip():
    payload = {
        "type": "ready",
        "session_id": "abc-123",
        "sample_rate": 16000,
        "pipeline": "diarization",
        "models": {
            "segmentation": "pyannote/segmentation-3.0",
            "embedding": "pyannote/embedding",
        },
        "config": {"tau_active": 0.507, "max_speakers": 10},
    }
    msg = ReadyMessage.model_validate(payload)
    dumped = msg.model_dump()
    assert dumped == payload
    # JSON round-trip preserves shape
    assert json.loads(msg.model_dump_json()) == payload


def test_prediction_message_roundtrip():
    payload = {
        "type": "prediction",
        "start": 1.23,
        "duration": 0.50,
        "speaker": "speaker_0",
        "embedding": [0.123, 0.456, -0.789],
        "samples_count": 6,
    }
    msg = PredictionMessage.model_validate(payload)
    dumped = msg.model_dump()
    # name/score are optional fields added in Task 3; they default to None
    expected = {**payload, "name": None, "score": None}
    assert dumped == expected
    assert json.loads(msg.model_dump_json()) == expected


def test_final_message_roundtrip():
    payload = {
        "type": "final",
        "rttm": "SPEAKER stream 1 0.000 1.500 <NA> <NA> speaker_0 <NA> <NA>",
        "duration": 35.6,
        "num_speakers": 3,
        "stats": {"total_predictions": 42, "session_seconds": 35.6},
    }
    msg = FinalMessage.model_validate(payload)
    dumped = msg.model_dump()
    assert dumped == payload
    assert json.loads(msg.model_dump_json()) == payload


def test_error_message_roundtrip():
    payload = {
        "type": "error",
        "code": "BAD_AUDIO",
        "message": "invalid sample_rate",
    }
    msg = ErrorMessage.model_validate(payload)
    # code serializes as the string value, not "ErrorCode.BAD_AUDIO"
    dumped = msg.model_dump()
    assert dumped == payload
    json_payload = json.loads(msg.model_dump_json())
    assert json_payload == payload
    assert json_payload["code"] == "BAD_AUDIO"


def test_error_message_accepts_enum_input():
    msg = ErrorMessage(code=ErrorCode.RATE_LIMIT_EXCEEDED, message="slow down")
    dumped = msg.model_dump()
    assert dumped == {
        "type": "error",
        "code": "RATE_LIMIT_EXCEEDED",
        "message": "slow down",
    }


def test_end_control_roundtrip():
    payload = {"type": "end"}
    msg = EndControl.model_validate(payload)
    assert msg.model_dump() == payload
    assert json.loads(msg.model_dump_json()) == payload


# ---------------------------------------------------------------------------
# QueryParams: defaults & validation
# ---------------------------------------------------------------------------


def test_query_params_defaults_match_spec():
    qp = QueryParams.model_validate({})
    assert qp.sample_rate == 16000
    assert qp.pipeline == "diarization"
    assert qp.segmentation_model == "pyannote/segmentation-3.0"
    assert qp.embedding_model == "pyannote/embedding"
    assert qp.tau_active == 0.507
    assert qp.rho_update == 0.006
    assert qp.delta_new == 1.057
    assert qp.latency == 0.5
    assert qp.max_speakers == 10
    assert qp.duration == 5.0
    assert qp.step == 0.5
    # 11 explicit fields besides `type`-style discriminator (not present
    # here); spec calls them out as 12 once `pipeline` is included.
    # Make sure no field was forgotten.
    assert len(qp.model_dump()) == 11


def test_query_params_accepts_valid_sample_rate():
    qp = QueryParams.model_validate({"sample_rate": 22050})
    assert qp.sample_rate == 22050


def test_query_params_accepts_pipeline_vad():
    qp = QueryParams.model_validate({"pipeline": "vad"})
    assert qp.pipeline == "vad"


def test_query_params_rejects_unknown_pipeline():
    with pytest.raises(ValidationError):
        QueryParams.model_validate({"pipeline": "transcription"})


# ---------------------------------------------------------------------------
# parse_query: wraps ValidationError as BadAudioError
# ---------------------------------------------------------------------------


def test_parse_query_returns_query_params_on_valid_input():
    qp = parse_query({"sample_rate": 22050, "latency": 1.0})
    assert isinstance(qp, QueryParams)
    assert qp.sample_rate == 22050
    assert qp.latency == 1.0


def test_parse_query_rejects_low_sample_rate():
    with pytest.raises(BadAudioError):
        parse_query({"sample_rate": 4000})


def test_parse_query_rejects_high_sample_rate():
    with pytest.raises(BadAudioError):
        parse_query({"sample_rate": 96000})


def test_parse_query_rejects_high_latency():
    with pytest.raises(BadAudioError):
        parse_query({"latency": 6.0})


def test_parse_query_rejects_low_latency():
    with pytest.raises(BadAudioError):
        parse_query({"latency": 0.1})


def test_parse_query_rejects_max_speakers_out_of_range():
    with pytest.raises(BadAudioError):
        parse_query({"max_speakers": 1})
    with pytest.raises(BadAudioError):
        parse_query({"max_speakers": 21})


# ---------------------------------------------------------------------------
# EndControl distinguishes from other JSON message types
# ---------------------------------------------------------------------------


def test_end_control_rejects_other_types():
    # A correct EndControl payload accepts {"type":"end"} and rejects
    # any other discriminator value, so it can be safely used to tell
    # client control messages apart from server prediction frames.
    EndControl.model_validate({"type": "end"})
    with pytest.raises(ValidationError):
        EndControl.model_validate({"type": "prediction"})
    with pytest.raises(ValidationError):
        EndControl.model_validate({"type": "final"})


def test_end_control_rejects_extra_fields():
    # Strict schema: a binary audio frame would never look like
    # {"type":"end", ...} with extra keys, but extras are forbidden so
    # any malformed control message is rejected explicitly.
    with pytest.raises(ValidationError):
        EndControl.model_validate({"type": "end", "extra": 1})
