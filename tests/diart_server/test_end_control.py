"""End-control text frame + multi-model query tests for the WS route (T22).

T17/T18 wired the WebSocket route around ``websocket.iter_bytes()`` which
silently dropped text frames. T22 replaces that loop with explicit
``await websocket.receive()`` so the route can additionally honour a
graceful end-of-stream control message (``{"type":"end"}``) without
relying on the client tearing down the TCP connection. This file pins
that behaviour:

  * ``test_end_control_triggers_final`` — a binary frame followed by a
    text ``{"type":"end"}`` frame must produce a ``final`` summary
    before the socket closes.
  * ``test_known_embedding_model_is_echoed_in_ready`` — a non-default
    ``?embedding_model=...`` must be threaded through to
    ``ModelsRegistry.get_embedding`` and reflected in the ``ready``
    payload's ``models.embedding`` field.
  * ``test_unknown_embedding_model_returns_model_not_found`` — when the
    registry raises :class:`ModelNotFoundError`, the route must send an
    ``error`` frame with code ``MODEL_NOT_FOUND`` and close with code
    1008.
  * ``test_malformed_text_frame_does_not_terminate`` — random text /
    invalid JSON must be silently ignored so a buggy client cannot tear
    down an in-flight session by sending stray text.

Heavy ML weights are avoided exactly as in T19/T21 by monkeypatching
:class:`ModelsRegistry` to hand back the
:class:`DummySegmentationModel` / :class:`DummyEmbeddingModel` from
``tests/conftest.py``. The dummy embedding model produces randomly-
sized output which can trip diart's online clustering check, but the
end-control path is upstream of inference so the test is robust to
that.
"""

from __future__ import annotations

import json
import sys
import time
import wave
from pathlib import Path

import numpy as np
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


SAMPLE_WAV = TESTS_DIR / "data" / "audio" / "sample.wav"


def _load_sample_pcm() -> tuple[np.ndarray, int]:
    """Read ``sample.wav`` and return ``(float32 mono PCM, sample_rate)``."""
    with wave.open(str(SAMPLE_WAV), "rb") as w:
        sr = w.getframerate()
        n_channels = w.getnchannels()
        n_frames = w.getnframes()
        sampwidth = w.getsampwidth()
        raw = w.readframes(n_frames)

    assert sampwidth == 2, "expected int16 sample.wav"
    pcm_int16 = np.frombuffer(raw, dtype=np.int16)
    if n_channels == 2:
        pcm_int16 = pcm_int16.reshape(-1, 2).mean(axis=1).astype(np.int16)
    pcm = pcm_int16.astype(np.float32) / 32768.0
    return pcm, sr


@pytest.fixture
def patched_app(monkeypatch):
    """Replace the registry's loaders with dummies before lifespan runs.

    Same shape as T19's fixture: patching at the class level means any
    ``ModelsRegistry`` constructed during the FastAPI lifespan -- including
    ``app.state.registry`` -- transparently returns dummy models. We also
    stub ``warm_up`` to a no-op so the lifespan flips
    ``app.state.ready=True`` without trying to pin pyannote weights to a
    GPU/CPU.

    Individual tests override ``get_embedding`` further (e.g. to raise
    :class:`ModelNotFoundError`) by re-applying ``monkeypatch.setattr``
    after this fixture has run.
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

    Same trick as T19/T21: peek at ``WebSocketTestSession._send_queue``
    with a short timeout and stop on ``Empty`` or a ``websocket.close``
    envelope. We avoid blocking ``receive_json`` because it has no
    built-in deadline and the route may not emit any predictions at all
    before sending ``final`` (the dummy embedding model often crashes
    diart's clusterer mid-run).
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
# Test 1: end-control text frame triggers a graceful drain + final summary.
# ---------------------------------------------------------------------------


def test_end_control_triggers_final(patched_app):
    """Sending ``{"type":"end"}`` after a binary frame must produce a final.

    The route's new ``receive()``-based loop should:
      1. accept the binary frame and forward it to the session,
      2. parse the subsequent text frame as JSON, recognise the ``end``
         control, and break out of the loop,
      3. let the ``finally:`` block drain the session and send a
         ``final`` summary before closing the socket.

    We don't assert on prediction counts because the dummy embedding
    model frequently crashes diart's clusterer; we only require the
    end-of-stream contract.
    """
    pcm, sr = _load_sample_pcm()
    # Cap at 1 s of audio: enough to fill the diart pipeline's input
    # buffer briefly without bloating the test budget. The end-control
    # path is independent of how much audio we sent.
    frame = pcm[: sr].astype(np.float32).tobytes()

    ready_msg: dict | None = None
    final_msg: dict | None = None
    other_msgs: list[dict] = []

    with TestClient(patched_app) as client:
        with client.websocket_connect(
            f"/v1/diarize/stream?sample_rate={sr}"
        ) as ws:
            ready_msg = ws.receive_json()
            assert ready_msg["type"] == "ready"

            # Push one second of audio as a single binary frame.
            ws.send_bytes(frame)
            # Brief pause: lets the inference thread pick up the frame
            # before we ask the server to wrap up. Without this the
            # ``final`` summary is emitted instantly with zero observed
            # predictions, which is fine for the contract but makes the
            # test less useful as a smoke check.
            time.sleep(0.2)

            # The contract under test: a JSON text frame with
            # ``type=="end"`` must drive the route into its drain path.
            ws.send_text(json.dumps({"type": "end"}))

            # The server's ``finally:`` block runs ``session.close()``
            # and ``send_json(final)`` before closing. Drain whatever
            # frames land on the queue and pull out the final.
            drained = _drain_messages(ws, max_wait_seconds=10.0)
            for msg in drained:
                if msg.get("type") == "final":
                    final_msg = msg
                else:
                    other_msgs.append(msg)

        # In case the close envelope sandwiched a frame we missed inside
        # the context, do one more short drain after exit.
        if final_msg is None:
            for msg in _drain_messages(ws, max_wait_seconds=2.0):
                if msg.get("type") == "final":
                    final_msg = msg
                    break

    assert ready_msg is not None and ready_msg["type"] == "ready"
    assert final_msg is not None, (
        "expected a final summary after sending {'type':'end'}; "
        f"other frames={other_msgs}"
    )
    assert final_msg["type"] == "final"
    assert "rttm" in final_msg and isinstance(final_msg["rttm"], str)
    assert "duration" in final_msg
    assert "num_speakers" in final_msg
    assert "stats" in final_msg and "predictions_count" in final_msg["stats"]


# ---------------------------------------------------------------------------
# Test 2: a known non-default embedding model is threaded through to ready.
# ---------------------------------------------------------------------------


def test_known_embedding_model_is_echoed_in_ready(patched_app, monkeypatch):
    """Multi-model query support: ``?embedding_model=...`` reaches ready.

    We patch ``ModelsRegistry.get_embedding`` to record which name was
    requested, then assert both that the registry was invoked with the
    query-supplied name and that the ``ready`` message echoes the same
    string in ``models.embedding``. This proves the route is not silently
    falling back to the default model.
    """
    requested: list[str] = []
    emb_model = EmbeddingModel(DummyEmbeddingModel)

    def recording_get_emb(self, name):  # noqa: ARG001
        requested.append(name)
        return emb_model

    monkeypatch.setattr(ModelsRegistry, "get_embedding", recording_get_emb)

    custom_emb = "speechbrain/spkrec-ecapa-voxceleb"
    with TestClient(patched_app) as client:
        with client.websocket_connect(
            f"/v1/diarize/stream?sample_rate=16000&embedding_model={custom_emb}"
        ) as ws:
            ready_msg = ws.receive_json()

    assert ready_msg["type"] == "ready"
    assert ready_msg["models"]["embedding"] == custom_emb, (
        f"ready did not echo embedding_model: {ready_msg!r}"
    )
    # ``warm_up`` is also a registry call but it's a no-op stub here, so
    # the only ``get_embedding`` call we expect is from the session
    # constructor with the query-supplied name.
    assert custom_emb in requested, (
        f"registry.get_embedding was never called with {custom_emb!r}; "
        f"saw {requested!r}"
    )


# ---------------------------------------------------------------------------
# Test 3: an unknown embedding model surfaces MODEL_NOT_FOUND + close 1008.
# ---------------------------------------------------------------------------


def test_unknown_embedding_model_returns_model_not_found(
    patched_app, monkeypatch
):
    """An unknown ``embedding_model`` query must close 1008 with the code.

    The DiarizationSession constructor calls
    ``registry.get_embedding(params.embedding_model)``. We override the
    registry to raise :class:`ModelNotFoundError`, which the route must
    catch, send a wire-format error frame for, and close the socket
    with code 1008 (per STANDALONE_SERVER.md §6).
    """

    def raising_get_emb(self, name):  # noqa: ARG001
        raise ModelNotFoundError(f"unknown: {name}")

    monkeypatch.setattr(ModelsRegistry, "get_embedding", raising_get_emb)

    error_msg: dict | None = None
    disconnect_code: int | None = None
    with TestClient(patched_app) as client:
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect(
                "/v1/diarize/stream?embedding_model=does/not/exist"
            ) as ws:
                error_msg = ws.receive_json()
                # The next receive should raise: the route closed the
                # socket with code 1008 right after sending the error.
                ws.receive_json()
        disconnect_code = exc_info.value.code

    assert error_msg is not None, "expected an error frame before disconnect"
    assert error_msg["type"] == "error"
    assert error_msg["code"] == "MODEL_NOT_FOUND"
    # ``message`` should at least mention the offending name to help
    # client-side debugging; the registry stub formats it explicitly.
    assert "does/not/exist" in error_msg["message"]
    assert disconnect_code == 1008, (
        f"expected close code 1008, got {disconnect_code!r}"
    )


# ---------------------------------------------------------------------------
# Test 4: malformed / unrecognised text frames must NOT terminate the session.
# ---------------------------------------------------------------------------


def test_malformed_text_frame_does_not_terminate(patched_app):
    """A bogus text frame must be silently ignored (no early ``final``).

    The route's text-frame branch parses JSON and only acts on
    ``{"type":"end"}``. Garbage text must be discarded so that a buggy
    or hostile client cannot prematurely close someone else's stream.
    We verify by sending non-JSON, then a JSON object with an unknown
    type, then *finally* the real end-control; only the latter should
    trigger a ``final`` summary.
    """
    ready_msg: dict | None = None
    final_msg: dict | None = None
    pre_end_msgs: list[dict] = []

    with TestClient(patched_app) as client:
        with client.websocket_connect(
            "/v1/diarize/stream?sample_rate=16000"
        ) as ws:
            ready_msg = ws.receive_json()
            assert ready_msg["type"] == "ready"

            # Garbage frames the route must ignore.
            ws.send_text("not json at all")
            ws.send_text(json.dumps({"type": "wat", "payload": 42}))
            ws.send_text(json.dumps(["not", "an", "object"]))

            # Brief settle: if the route mistakenly treated any of the
            # above as end-of-stream we'd see a ``final`` show up here.
            pre_drain = _drain_messages(ws, max_wait_seconds=0.5)
            for msg in pre_drain:
                pre_end_msgs.append(msg)

            # Real end control: this one *must* terminate the session.
            ws.send_text(json.dumps({"type": "end"}))

            for msg in _drain_messages(ws, max_wait_seconds=10.0):
                if msg.get("type") == "final":
                    final_msg = msg

    # No ``final`` should have arrived before the real end-control.
    early_finals = [m for m in pre_end_msgs if m.get("type") == "final"]
    assert not early_finals, (
        f"route ended the session on a malformed text frame: {early_finals!r}"
    )
    assert final_msg is not None, (
        "expected a final summary after the real end-control frame; "
        f"pre-end frames={pre_end_msgs!r}"
    )
    assert final_msg["type"] == "final"
