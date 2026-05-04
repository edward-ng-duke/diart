"""Failure mode test suite for the diart_server WebSocket route (T23).

Covers each documented failure row of STANDALONE_SERVER.md §9 against
the real FastAPI application driven through Starlette's synchronous
:class:`fastapi.testclient.TestClient`. Heavy ML weights are avoided
exactly as in T19/T21/T22 by monkeypatching
:class:`diart_server.models_registry.ModelsRegistry` to hand back the
:class:`DummySegmentationModel` / :class:`DummyEmbeddingModel` fixtures
from ``tests/conftest.py``.

Each test pins one row of the failure matrix:

  1. Empty binary frame -> ``BAD_AUDIO`` *and connection stays open*
     (route swallows the decoder error and ``continue``s the loop).
  2. Frame larger than ``settings.max_frame_bytes`` (10 MB by default)
     -> ``RATE_LIMIT_EXCEEDED`` + close 1009.
  3. Query ``?sample_rate=4000`` outside the [8000, 48000] Pydantic range
     -> ``BAD_AUDIO`` + close 1008 (rejected by ``QueryParams``).
  4. Unknown segmentation model -> ``MODEL_NOT_FOUND`` + close 1008
     (already covered for ``embedding_model`` in T22; this test pins
     the segmentation arm of the same code path).
  5. 9th concurrent connection when capacity is bounded ->
     ``SESSION_CAPACITY_FULL`` + close 1013. We monkeypatch the manager
     down to capacity = 2 so the test does not need to keep 8
     long-lived sockets open.
  6. Inference thread crash -> the daemon thread emits an
     ``INFERENCE_CRASH`` wire message via
     ``asyncio.run_coroutine_threadsafe`` and the route eventually
     closes when the client disconnects.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

# Reuse the dummy model classes defined in tests/conftest.py.
TESTS_DIR = Path(__file__).resolve().parent.parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from conftest import DummyEmbeddingModel, DummySegmentationModel  # noqa: E402

from diart.models import EmbeddingModel, SegmentationModel  # noqa: E402

from diart_server import app as app_module  # noqa: E402
from diart_server.errors import ModelNotFoundError  # noqa: E402
from diart_server.models_registry import ModelsRegistry  # noqa: E402


@pytest.fixture
def patched_app(monkeypatch):
    """Replace registry loaders with dummies before the lifespan runs.

    Same shape as T19/T21/T22 fixtures: patching at the class level means
    any ``ModelsRegistry`` constructed during the FastAPI lifespan --
    including ``app.state.registry`` -- transparently returns dummy
    models. ``warm_up`` is stubbed to a no-op so the lifespan flips
    ``app.state.ready=True`` without trying to pin pyannote weights.
    """
    seg_model = SegmentationModel(DummySegmentationModel)
    emb_model = EmbeddingModel(DummyEmbeddingModel)

    def fake_get_seg(self, name):  # noqa: ARG001
        return seg_model

    def fake_get_emb(self, name):  # noqa: ARG001
        return emb_model

    def fake_warm_up(self, device, default_seg, default_emb):  # noqa: ARG001
        return None

    monkeypatch.setattr(ModelsRegistry, "warm_up", fake_warm_up)
    monkeypatch.setattr(ModelsRegistry, "get_segmentation", fake_get_seg)
    monkeypatch.setattr(ModelsRegistry, "get_embedding", fake_get_emb)
    return app_module.app


def _drain_messages(ws, max_wait_seconds: float) -> list[dict]:
    """Best-effort drain of pending JSON frames from a TestClient websocket.

    Same trick as T19/T21/T22: peek at ``WebSocketTestSession._send_queue``
    with a short timeout. Stops on ``Empty`` or ``websocket.close`` envelope.
    """
    out: list[dict] = []
    deadline = time.monotonic() + max_wait_seconds
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        try:
            frame = ws._send_queue.get(timeout=min(remaining, 0.5))
        except Exception:
            break
        if frame.get("type") == "websocket.close":
            break
        text = frame.get("text")
        if text is None:
            continue
        try:
            out.append(json.loads(text))
        except (TypeError, ValueError):
            continue
    return out


# ---------------------------------------------------------------------------
# 1. Empty frame -> BAD_AUDIO; connection stays open.
# ---------------------------------------------------------------------------


def test_empty_frame_emits_bad_audio_and_keeps_connection_open(patched_app):
    """An empty binary frame must not tear down the session.

    ``audio_decoder.to_pcm_16k_mono(b"", sr)`` raises ``BadAudioError``;
    the route's binary-frame branch sends the wire-format error message
    and ``continue``s the loop without closing. Sending ``{"type":"end"}``
    afterwards must still produce a clean ``final`` summary.
    """
    with TestClient(patched_app) as client:
        with client.websocket_connect(
            "/v1/diarize/stream?sample_rate=16000"
        ) as ws:
            ready = ws.receive_json()
            assert ready["type"] == "ready"

            # Empty frame -> BAD_AUDIO, but the loop must keep going.
            ws.send_bytes(b"")
            err = ws.receive_json()
            assert err["type"] == "error"
            assert err["code"] == "BAD_AUDIO"

            # Connection is still alive: send the graceful end control
            # frame and drain whatever the route flushes before close.
            ws.send_text(json.dumps({"type": "end"}))
            drained = _drain_messages(ws, max_wait_seconds=3.0)

    finals = [m for m in drained if m.get("type") == "final"]
    assert finals, (
        "expected a final summary after end-of-stream control; "
        f"drained={drained}"
    )


# ---------------------------------------------------------------------------
# 2. Frame larger than settings.max_frame_bytes -> RATE_LIMIT_EXCEEDED + 1009.
# ---------------------------------------------------------------------------


def test_oversized_frame_returns_rate_limit_exceeded(patched_app):
    """A binary frame > 10 MB must close the socket with 1009."""
    from diart_server.config import settings

    too_big = b"\x00" * (settings.max_frame_bytes + 1)

    error_msg: dict | None = None
    with TestClient(patched_app) as client:
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect(
                "/v1/diarize/stream?sample_rate=16000"
            ) as ws:
                ready = ws.receive_json()
                assert ready["type"] == "ready"

                ws.send_bytes(too_big)
                error_msg = ws.receive_json()
                # Next receive raises: route closed with code 1009.
                ws.receive_json()

    assert error_msg is not None
    assert error_msg["type"] == "error"
    assert error_msg["code"] == "RATE_LIMIT_EXCEEDED"
    assert exc_info.value.code == 1009


# ---------------------------------------------------------------------------
# 3. ?sample_rate=4000 -> BAD_AUDIO + 1008.
# ---------------------------------------------------------------------------


def test_sample_rate_below_minimum_rejects_with_bad_audio(patched_app):
    """``?sample_rate=4000`` is rejected by Pydantic -> BAD_AUDIO + 1008.

    ``QueryParams.sample_rate`` is constrained to [8000, 48000]; values
    outside the range trip ``ValidationError`` which the route wraps as
    a ``BAD_AUDIO`` wire message and closes with code 1008 (per
    STANDALONE_SERVER.md §6).
    """
    error_msg: dict | None = None
    with TestClient(patched_app) as client:
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect(
                "/v1/diarize/stream?sample_rate=4000"
            ) as ws:
                error_msg = ws.receive_json()
                ws.receive_json()  # next receive raises

    assert error_msg is not None
    assert error_msg["type"] == "error"
    assert error_msg["code"] == "BAD_AUDIO"
    assert exc_info.value.code == 1008


# ---------------------------------------------------------------------------
# 4. Unknown segmentation model -> MODEL_NOT_FOUND + 1008.
# ---------------------------------------------------------------------------


def test_unknown_segmentation_model_returns_model_not_found(
    patched_app, monkeypatch
):
    """Re-pin the MODEL_NOT_FOUND row using the segmentation arm.

    T22 already covers unknown ``embedding_model``; this test pins the
    same code path for ``segmentation_model`` so both registry calls in
    :class:`DiarizationSession.__init__` are guarded.
    """

    def raising_get_seg(self, name):  # noqa: ARG001
        raise ModelNotFoundError(f"unknown: {name}")

    monkeypatch.setattr(ModelsRegistry, "get_segmentation", raising_get_seg)

    error_msg: dict | None = None
    with TestClient(patched_app) as client:
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect(
                "/v1/diarize/stream?segmentation_model=does/not/exist"
            ) as ws:
                error_msg = ws.receive_json()
                ws.receive_json()  # next receive raises

    assert error_msg is not None
    assert error_msg["type"] == "error"
    assert error_msg["code"] == "MODEL_NOT_FOUND"
    assert "does/not/exist" in error_msg["message"]
    assert exc_info.value.code == 1008


# ---------------------------------------------------------------------------
# 5. (cap+1)th concurrent connection -> SESSION_CAPACITY_FULL + 1013.
# ---------------------------------------------------------------------------


def test_capacity_full_returns_session_capacity_full(patched_app, monkeypatch):
    """When the manager is at capacity, the next connect must reject.

    To avoid keeping 8 long-lived sockets open we shrink the manager's
    ``_max`` to 2 for this test only. Two connections fill the capacity,
    a third must be rejected with ``SESSION_CAPACITY_FULL`` and close
    1013 (per STANDALONE_SERVER.md §6).

    Capacity is tracked by ``session_manager.manager``, a module-level
    singleton constructed at import time from
    ``settings.max_concurrent_sessions``. We patch the manager's ``_max``
    in place rather than rebuilding the singleton because
    ``ws_routes.py`` imported the existing object by reference.
    """
    from diart_server import session_manager

    monkeypatch.setattr(session_manager.manager, "_max", 2)

    with TestClient(patched_app) as client:
        # Open 2 connections to fill capacity. We use raw __enter__ so we
        # can hold them open while attempting the 3rd connection.
        ws1_cm = client.websocket_connect("/v1/diarize/stream?sample_rate=16000")
        ws2_cm = client.websocket_connect("/v1/diarize/stream?sample_rate=16000")
        ws1 = ws1_cm.__enter__()
        ws2 = ws2_cm.__enter__()
        try:
            assert ws1.receive_json()["type"] == "ready"
            assert ws2.receive_json()["type"] == "ready"

            error_msg: dict | None = None
            with pytest.raises(WebSocketDisconnect) as exc_info:
                with client.websocket_connect(
                    "/v1/diarize/stream?sample_rate=16000"
                ) as ws3:
                    error_msg = ws3.receive_json()
                    ws3.receive_json()  # next receive raises (1013)

            assert error_msg is not None
            assert error_msg["type"] == "error"
            assert error_msg["code"] == "SESSION_CAPACITY_FULL"
            assert exc_info.value.code == 1013
        finally:
            # Cleanly tear down the two held-open sessions so the
            # TestClient lifespan can shut down without orphan threads.
            ws1_cm.__exit__(None, None, None)
            ws2_cm.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# 6. Inference thread crash -> INFERENCE_CRASH wire message.
# ---------------------------------------------------------------------------


def test_inference_crash_emits_inference_crash_wire_message(
    patched_app, monkeypatch
):
    """If the inference thread raises, the route must emit INFERENCE_CRASH.

    ``DiarizationSession._run`` wraps ``self.inference()`` and converts
    any exception into the documented ``INFERENCE_CRASH`` error message
    via ``asyncio.run_coroutine_threadsafe``. Patching
    :meth:`StreamingInference.__call__` is the cheapest way to force the
    crash deterministically.

    The test does not push any frames; the thread raises immediately on
    first iteration. We then drain the websocket queue looking for the
    INFERENCE_CRASH frame, then send ``{"type":"end"}`` so the route's
    ``finally`` block can run cleanly.
    """
    from diart.inference import StreamingInference

    def crash(self):  # noqa: ARG001
        # Small sleep so the route has flipped to the receive loop
        # before the thread raises; otherwise the run_coroutine_threadsafe
        # call may race with WebSocket setup.
        time.sleep(0.1)
        raise RuntimeError("simulated inference crash")

    monkeypatch.setattr(StreamingInference, "__call__", crash)

    crash_msgs: list[dict] = []
    with TestClient(patched_app) as client:
        with client.websocket_connect(
            "/v1/diarize/stream?sample_rate=16000"
        ) as ws:
            ready = ws.receive_json()
            assert ready["type"] == "ready"

            # Wait for the daemon thread to crash and the wire message
            # to land in the send queue.
            drained = _drain_messages(ws, max_wait_seconds=3.0)
            crash_msgs = [
                m
                for m in drained
                if m.get("type") == "error"
                and m.get("code") == "INFERENCE_CRASH"
            ]

            # Cleanly tell the route to drain so the finally block can
            # send the final summary without exceptions.
            try:
                ws.send_text(json.dumps({"type": "end"}))
                _drain_messages(ws, max_wait_seconds=2.0)
            except Exception:
                # If the socket has already been torn down by the close
                # path, that's still acceptable for this test.
                pass

    assert crash_msgs, (
        "expected at least one INFERENCE_CRASH wire message after the "
        f"inference thread raised; drained={drained}"
    )
    assert crash_msgs[0]["message"]
