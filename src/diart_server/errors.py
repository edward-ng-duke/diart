"""Error codes and exception hierarchy for diart_server.

This module defines a stable set of error codes used to map server-side
exceptions to WebSocket error messages. Subsequent components (audio
ingestion, inference, session/rate-limit management) raise these
exceptions; a single boundary handler converts them to wire-format
messages via :meth:`DiartServerError.to_message`.
"""

from __future__ import annotations

from enum import Enum


class ErrorCode(str, Enum):
    """Stable error codes sent to clients in error messages."""

    BAD_AUDIO = "BAD_AUDIO"
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    RESAMPLE_FAILED = "RESAMPLE_FAILED"
    INFERENCE_CRASH = "INFERENCE_CRASH"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    SESSION_CAPACITY_FULL = "SESSION_CAPACITY_FULL"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class DiartServerError(Exception):
    """Base class for all diart_server exceptions.

    Subclasses override :attr:`code` with a more specific
    :class:`ErrorCode`. The base class defaults to
    :attr:`ErrorCode.INTERNAL_ERROR` so that any unclassified subclass
    or bare instance still serializes to a valid error message.
    """

    code: ErrorCode = ErrorCode.INTERNAL_ERROR

    def to_message(self) -> dict:
        """Return the wire-format error message for this exception."""
        return {
            "type": "error",
            "code": self.code.value,
            "message": str(self),
        }


class BadAudioError(DiartServerError):
    code = ErrorCode.BAD_AUDIO


class ModelNotFoundError(DiartServerError):
    code = ErrorCode.MODEL_NOT_FOUND


class ResampleFailedError(DiartServerError):
    code = ErrorCode.RESAMPLE_FAILED


class InferenceCrashError(DiartServerError):
    code = ErrorCode.INFERENCE_CRASH


class RateLimitExceededError(DiartServerError):
    code = ErrorCode.RATE_LIMIT_EXCEEDED


class SessionCapacityFullError(DiartServerError):
    code = ErrorCode.SESSION_CAPACITY_FULL


__all__ = [
    "ErrorCode",
    "DiartServerError",
    "BadAudioError",
    "ModelNotFoundError",
    "ResampleFailedError",
    "InferenceCrashError",
    "RateLimitExceededError",
    "SessionCapacityFullError",
]
