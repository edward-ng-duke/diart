"""4-connection concurrent load test for the diart_server WebSocket route (T21).

Spawns four threads that each open an independent WebSocket against the
same ``TestClient``-driven FastAPI app, push the same ``sample.wav``
audio payload, and walk through the ``ready`` -> stream -> ``final``
lifecycle. The test asserts:

  * all four connections complete without raising,
  * each receives a ``ready`` frame with a unique ``session_id``,
  * each receives a ``final`` summary before the socket closes,
  * the server never returns ``SESSION_CAPACITY_FULL`` (default cap is 8,
    we open 4),
  * any errors emitted are the documented dummy-induced ones, not
    surprises.

Heavy ML weights are avoided exactly as in T19's single-connection test
by monkeypatching :class:`ModelsRegistry` to hand back the
:class:`DummySegmentationModel` / :class:`DummyEmbeddingModel` from
``tests/conftest.py``. The dummy embedding model picks a random output
dim per call which makes diart's clusterer crash mid-run, so we do not
require ``>= 3`` predictions per session -- we only require the lifecycle
completes cleanly. Independence is asserted via ``session_id`` uniqueness
(each session's clusterer is local; ``speaker_0`` in one session has no
relation to ``speaker_0`` in another) rather than by trying to detect
cross-talk in the typically 0-2 predictions the dummy emits per session.

Caveat: Starlette's :class:`fastapi.testclient.TestClient` runs the ASGI
app on a single portal thread with its own asyncio loop, so the four
"concurrent" WebSocket connections may be serialized by that portal in
practice. The test still validates the multi-session contract (capacity
accounting, per-session isolation, deterministic teardown); a true wall-
clock parallelism benchmark belongs in T30.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import wave
from pathlib import Path
from queue import Empty, Queue

import numpy as np
import pytest
from fastapi.testclient import TestClient

# Reuse the dummy model classes defined in tests/conftest.py.
TESTS_DIR = Path(__file__).resolve().parent.parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from conftest import DummyEmbeddingModel, DummySegmentationModel  # noqa: E402

from diart.models import EmbeddingModel, SegmentationModel  # noqa: E402

from diart_server import app as app_module  # noqa: E402
from diart_server.models_registry import ModelsRegistry  # noqa: E402


SAMPLE_WAV = TESTS_DIR / "data" / "audio" / "sample.wav"
FRAME_SECONDS = 0.5
# 10 s per session keeps the wall-clock budget reasonable while still
# pushing enough audio (>5 s window) to exercise the inference loop.
MAX_FRAMES = 20
NUM_CONNECTIONS = 4


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

    Identical to T19's fixture: we patch on the class so any
    ``ModelsRegistry`` constructed during the FastAPI lifespan -- including
    ``app.state.registry`` -- transparently returns dummy models. We also
    stub ``warm_up`` to a no-op so the lifespan flips
    ``app.state.ready=True`` without trying to pin pyannote weights to a
    GPU.

    The two dummy models are instantiated once and shared across all four
    sessions; per-session isolation is provided by diart's pipeline
    constructing fresh clusterers, accumulators, and inference threads.
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

    Same trick as T19's ``_drain_messages``: peek at
    ``WebSocketTestSession._send_queue`` with a short timeout and stop on
    ``Empty`` or a ``websocket.close`` envelope. We avoid blocking
    ``receive_json`` because it has no built-in deadline.
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


def _run_one_connection(
    client: TestClient,
    sr: int,
    pcm: np.ndarray,
    results: Queue,
    conn_id: int,
) -> None:
    """Worker target: drive a single WebSocket through its full lifecycle.

    On any exception we still publish a result entry so the main thread's
    join loop can detect partial completion rather than hang. The result
    dict carries enough information for the assertions to localize which
    of the four connections misbehaved.
    """
    out: dict = {"conn_id": conn_id}
    try:
        with client.websocket_connect(
            f"/v1/diarize/stream?sample_rate={sr}"
        ) as ws:
            ready_msg = ws.receive_json()
            out["ready"] = ready_msg

            frame_samples = int(sr * FRAME_SECONDS)
            n_frames = min(len(pcm) // frame_samples, MAX_FRAMES)
            for i in range(n_frames):
                start = i * frame_samples
                chunk = pcm[start : start + frame_samples]
                ws.send_bytes(chunk.tobytes())
                # Same micro-pause as T19: lets the inference thread drain
                # so MeetingAudioSource doesn't overflow its bounded queue.
                time.sleep(0.05)

            # Drain whatever inference frames are already on the queue
            # before we ask the server to close. With dummies this is
            # typically a handful of predictions or a single
            # INFERENCE_CRASH. We don't gate on count.
            out["mid"] = _drain_messages(ws, max_wait_seconds=3.0)

            # Politely tell the server to disconnect *while still inside
            # the TestClient context manager*. ``ws.close()`` only puts a
            # ``websocket.disconnect`` envelope on the receive queue; it
            # does NOT trigger the portal's ``_notify_close`` event (that
            # only fires from ``__exit__``). So the server sees the
            # disconnect, runs its ``finally:`` block to ``session.close()``
            # and ``send_json(final)``, and the final frame lands in our
            # send queue while the portal is still alive. If we instead
            # leaned on context exit alone, the portal's task-group cancel
            # races the route's final-send and under concurrent load the
            # cancel sometimes wins -- producing 0 finals on some
            # connections.
            ws.close(code=1000)
            out["tail"] = _drain_messages(ws, max_wait_seconds=5.0)
    except Exception as e:  # noqa: BLE001
        out["error"] = repr(e)
    finally:
        results.put(out)


def test_4_concurrent_connections(patched_app):
    """Open 4 WS connections in parallel; assert each completes cleanly.

    The test relaxes the original "each receives >= 3 predictions" goal
    because the dummy embedding model trips diart's online-clustering
    dim check after a handful of frames (see T14/T19 notes). The
    invariants we *do* enforce are the ones the multi-session contract
    actually guarantees:

      * each session is uniquely identified,
      * the route's finally: block always sends a final summary,
      * the capacity guard is never triggered with N=4 < cap=8,
      * any errors are the documented dummy-induced ones.
    """
    pcm, sr = _load_sample_pcm()

    results: Queue = Queue()
    with TestClient(patched_app) as client:
        threads = [
            threading.Thread(
                target=_run_one_connection,
                args=(client, sr, pcm, results, i),
                name=f"ws-conn-{i}",
                daemon=True,
            )
            for i in range(NUM_CONNECTIONS)
        ]

        t_start = time.monotonic()
        for t in threads:
            t.start()
        for t in threads:
            # 60 s per-thread budget; the suite-level expectation is 90 s
            # total. Threads share the TestClient portal so this is an
            # upper bound, not a per-connection wall-clock minimum.
            t.join(timeout=60.0)
        elapsed = time.monotonic() - t_start

        # Any thread still alive after join means a deadlock somewhere
        # (likely the route blocked on session.close()). Surface that
        # before we start asserting on results.
        for t in threads:
            assert not t.is_alive(), f"{t.name} did not finish in 60s"

    collected: list[dict] = []
    while True:
        try:
            collected.append(results.get_nowait())
        except Empty:
            break

    assert len(collected) == NUM_CONNECTIONS, (
        f"expected {NUM_CONNECTIONS} thread results, got {len(collected)}: "
        f"{collected}"
    )

    # No worker raised. If one did, fail loudly with the repr so we can
    # tell e.g. a WebSocketDisconnect from a TimeoutError.
    for c in collected:
        assert "error" not in c, (
            f"conn {c['conn_id']} raised: {c.get('error')!r}; partial="
            f"ready={c.get('ready')} mid={c.get('mid')} tail={c.get('tail')}"
        )

    # Every connection got a ready frame with a session_id and the right
    # query-echoed sample rate.
    session_ids: list[str] = []
    for c in collected:
        ready = c.get("ready")
        assert ready is not None, f"conn {c['conn_id']} missing ready"
        assert ready["type"] == "ready"
        assert ready["sample_rate"] == sr
        assert ready["pipeline"] == "diarization"
        sid = ready.get("session_id")
        assert isinstance(sid, str) and sid, (
            f"conn {c['conn_id']} ready missing session_id: {ready}"
        )
        session_ids.append(sid)

    # Independence at the protocol level: every session must have its
    # own id. With unique ids we can be confident the speaker labels
    # *within* a session (e.g. ``speaker_0``) refer to that session's
    # local clusterer state and don't leak across sessions.
    assert len(set(session_ids)) == NUM_CONNECTIONS, (
        f"expected {NUM_CONNECTIONS} unique session_ids, got {session_ids}"
    )

    # Capacity: NUM_CONNECTIONS=4 < cap=8 (DIART_MAX_CONCURRENT_SESSIONS
    # default), so the manager must never reject. Scan every drained
    # frame for the documented capacity-full code.
    for c in collected:
        for batch_key in ("mid", "tail"):
            for msg in c.get(batch_key, []) or []:
                if msg.get("type") == "error":
                    assert msg.get("code") != "SESSION_CAPACITY_FULL", (
                        f"conn {c['conn_id']} hit capacity error: {msg}"
                    )

    # Every connection must have received a final summary. The route's
    # finally: block sends it best-effort before the close, so it lands
    # in either the mid or tail drained batch (typically tail).
    for c in collected:
        all_msgs = (c.get("mid") or []) + (c.get("tail") or [])
        finals = [m for m in all_msgs if m.get("type") == "final"]
        assert finals, (
            f"conn {c['conn_id']} got no final message; "
            f"mid={c.get('mid')} tail={c.get('tail')}"
        )
        final = finals[-1]
        assert "rttm" in final and isinstance(final["rttm"], str)
        assert "duration" in final
        assert "num_speakers" in final
        assert "stats" in final and "predictions_count" in final["stats"]

    # Any error frames must use the small set of codes we know dummies
    # can legitimately produce. A surprise code -- e.g.
    # ``BACKPRESSURE_OVERFLOW`` -- would mean concurrent traffic broke
    # something the single-connection T19 test couldn't see, and is
    # exactly the kind of regression this load test exists to catch.
    allowed_error_codes = {
        "INFERENCE_CRASH",
        "BAD_AUDIO",
        "RESAMPLE_FAILED",
    }
    for c in collected:
        all_msgs = (c.get("mid") or []) + (c.get("tail") or [])
        for msg in all_msgs:
            if msg.get("type") == "error":
                assert msg.get("code") in allowed_error_codes, (
                    f"conn {c['conn_id']} unexpected error: {msg}"
                )

    # Sanity: the whole show should be quick under dummies. The PLAN
    # budget is 90 s; if we ever blow past 75 s something is queueing
    # badly inside the TestClient portal and we want to know.
    assert elapsed < 75.0, f"4-connection test took {elapsed:.1f}s (>75s)"
