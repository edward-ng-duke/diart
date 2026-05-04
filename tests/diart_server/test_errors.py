"""Tests for diart_server.errors."""

from __future__ import annotations

import pytest

from diart_server.errors import (
    BadAudioError,
    DiartServerError,
    ErrorCode,
    InferenceCrashError,
    ModelNotFoundError,
    RateLimitExceededError,
    ResampleFailedError,
    SessionCapacityFullError,
)


# (exc_cls, expected_code, expected_code_value)
SUBCLASS_CASES = [
    (BadAudioError, ErrorCode.BAD_AUDIO, "BAD_AUDIO"),
    (ModelNotFoundError, ErrorCode.MODEL_NOT_FOUND, "MODEL_NOT_FOUND"),
    (ResampleFailedError, ErrorCode.RESAMPLE_FAILED, "RESAMPLE_FAILED"),
    (InferenceCrashError, ErrorCode.INFERENCE_CRASH, "INFERENCE_CRASH"),
    (RateLimitExceededError, ErrorCode.RATE_LIMIT_EXCEEDED, "RATE_LIMIT_EXCEEDED"),
    (
        SessionCapacityFullError,
        ErrorCode.SESSION_CAPACITY_FULL,
        "SESSION_CAPACITY_FULL",
    ),
]


@pytest.mark.parametrize("exc_cls,expected_code,expected_code_value", SUBCLASS_CASES)
def test_subclass_is_diart_server_error(exc_cls, expected_code, expected_code_value):
    e = exc_cls("boom")
    assert isinstance(e, DiartServerError)
    assert isinstance(e, Exception)


@pytest.mark.parametrize("exc_cls,expected_code,expected_code_value", SUBCLASS_CASES)
def test_subclass_has_correct_code(exc_cls, expected_code, expected_code_value):
    e = exc_cls("boom")
    assert e.code == expected_code
    # Class-level attribute matches instance attribute.
    assert exc_cls.code == expected_code


@pytest.mark.parametrize("exc_cls,expected_code,expected_code_value", SUBCLASS_CASES)
def test_subclass_to_message_shape(exc_cls, expected_code, expected_code_value):
    msg = "something went wrong"
    e = exc_cls(msg)
    assert e.to_message() == {
        "type": "error",
        "code": expected_code_value,
        "message": msg,
    }


def test_base_class_defaults_to_internal_error():
    e = DiartServerError("kaboom")
    assert e.code == ErrorCode.INTERNAL_ERROR
    assert e.to_message() == {
        "type": "error",
        "code": "INTERNAL_ERROR",
        "message": "kaboom",
    }


def test_error_code_values_are_stable_strings():
    # StrEnum members compare equal to their string values; this guards
    # against accidental rename of the wire-format code strings.
    assert ErrorCode.BAD_AUDIO == "BAD_AUDIO"
    assert ErrorCode.MODEL_NOT_FOUND == "MODEL_NOT_FOUND"
    assert ErrorCode.RESAMPLE_FAILED == "RESAMPLE_FAILED"
    assert ErrorCode.INFERENCE_CRASH == "INFERENCE_CRASH"
    assert ErrorCode.RATE_LIMIT_EXCEEDED == "RATE_LIMIT_EXCEEDED"
    assert ErrorCode.SESSION_CAPACITY_FULL == "SESSION_CAPACITY_FULL"
    assert ErrorCode.INTERNAL_ERROR == "INTERNAL_ERROR"
