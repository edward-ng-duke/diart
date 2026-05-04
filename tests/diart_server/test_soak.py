"""Manual soak test for the diart_server WebSocket route (T29).

This test is **skipped by default** because it deliberately runs for
many minutes per invocation -- 1 hour by default, configurable via the
``SOAK_MINUTES`` environment variable. CI does not have the wall-clock
budget for this. To run locally::

    # 5-minute version, useful for smoke-testing the soak harness
    SOAK_MINUTES=5 pytest tests/diart_server/test_soak.py -v -s -p no:cacheprovider \\
        --override-ini='addopts=' --no-header -k test_soak_4_connections \\
        --co  # to discover, drop --co to actually run

    # full 1-hour soak (the default)
    SOAK_MINUTES=60 pytest tests/diart_server/test_soak.py::test_soak_4_connections \\
        -v -s --runxfail

(or simply unskip the ``@pytest.mark.skip`` decorator below.)

What it asserts
---------------
The soak opens 4 concurrent WebSocket connections, each looping
``sample.wav`` at real-time pacing for the configured duration, and
checks three things at the end:

* **No memory leak.** Process RSS growth must be < 200 MB. We sample
  RSS once before opening connections and once after they all close.
* **No thread leak.** The number of live Python threads after the soak
  must be within +4 of the baseline (one slack thread per connection
  for any TestClient portal task that hasn't been GC'd yet).
* **Bounded queue overflow.** The fraction of frames the server
  reported as ``BACKPRESSURE_OVERFLOW`` must be < 1 % of frames pushed
  per connection. Anything more means the inference path can't keep up
  with real-time at this concurrency, which is the failure mode the
  soak exists to surface.

Patterning
----------
This file deliberately reuses the dummy-models monkeypatch pattern from
T19 / T21 (see ``test_e2e_websocket.py``, ``test_load_4_concurrent.py``)
so a developer can run the soak without HF credentials. With dummy
models the inference loop occasionally aborts mid-session due to the
random embedding dim; the soak loop catches that and re-opens the
connection so the duration target is still reached.

For a *real* soak against pyannote weights, drop the monkeypatch and
run with ``DIART_SEG_MODEL`` / ``DIART_EMB_MODEL`` pointing at locally
cached models. In that mode the embedding-dim crash goes away and the
overflow-rate assertion becomes the dominant signal.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import threading
import time
import wave
from pathlib import Path
from queue import Empty, Queue

import numpy as np
import psutil
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
NUM_CONNECTIONS = 4

# Operator override: the test honours SOAK_MINUTES so the same body can
# back a quick 5-min smoke or the full 1-hour run.
SOAK_MINUTES = int(os.environ.get("SOAK_MINUTES", "60"))

# Memory-leak budget: process RSS may legitimately grow during warmup
# (model state, lazy thread stacks). 200 MB is generous for dummies and
# tight for real pyannote weights without GPU pinning.
MAX_RSS_GROWTH_BYTES = 200 * 1024 * 1024

# Thread-leak budget: per-connection slack for any TestClient portal
# task that hasn't been collected yet by the time we sample.
MAX_THREAD_GROWTH = NUM_CONNECTIONS

# Per-connection backpressure-overflow tolerance: < 1 % of frames
# pushed.
MAX_OVERFLOW_RATIO = 0.01


def _load_sample_pcm() -> tuple[np.ndarray, int]:
    """Read sample.wav and return ``(float32 mono PCM, sample_rate)``."""
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
    """Replace registry's loaders with dummies before the lifespan runs.

    Mirrors the T19 / T21 fixture so the soak can run without HF auth.
    To soak against real weights, delete this fixture and the test's
    ``patched_app`` parameter -- the route will load whatever models
    ``DIART_SEG_MODEL`` / ``DIART_EMB_MODEL`` resolve to.
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
    """Best-effort drain of pending JSON frames from a TestClient WS.

    Same trick as T19 / T21: peek at the underlying ``_send_queue`` with
    a bounded timeout so a quiet server doesn't block the soak loop.
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


def _soak_one_connection(
    client: TestClient,
    sr: int,
    pcm: np.ndarray,
    deadline: float,
    results: Queue,
    conn_id: int,
) -> None:
    """Worker target: loop the wav at real-time until ``deadline``.

    Real-time pacing means we sleep ``FRAME_SECONDS`` between sends so
    one second of audio takes one second of wall-clock to transmit. If
    the server emits ``INFERENCE_CRASH`` (dummy-induced dim mismatch)
    we close the WS and immediately re-open: the soak's job is to keep
    pressure on the server for the full duration, not to wedge on the
    first dummy crash.
    """
    frame_samples = int(sr * FRAME_SECONDS)
    n_frames_per_loop = len(pcm) // frame_samples

    frames_pushed = 0
    overflow_count = 0
    crashes = 0
    sessions_opened = 0

    try:
        while time.monotonic() < deadline:
            with client.websocket_connect(
                f"/v1/diarize/stream?sample_rate={sr}"
            ) as ws:
                sessions_opened += 1
                ready = ws.receive_json()
                if ready.get("type") != "ready":
                    # Should not happen with a healthy server; bail
                    # this session and try again.
                    continue

                aborted_early = False
                # Inner loop: stream frames at real-time. Reset the
                # per-loop cursor each session so we always start from
                # the beginning of the wav -- simpler than threading a
                # global cursor through the worker state.
                i = 0
                while time.monotonic() < deadline and not aborted_early:
                    start = (i % n_frames_per_loop) * frame_samples
                    chunk = pcm[start : start + frame_samples]
                    try:
                        ws.send_bytes(chunk.tobytes())
                    except Exception:  # noqa: BLE001
                        # Server closed under us (e.g. inference crash).
                        aborted_early = True
                        break
                    frames_pushed += 1
                    i += 1
                    # Real-time pacing. We deliberately sleep the full
                    # frame interval rather than the smaller 0.05 s
                    # used in T19 / T21 -- the soak is about sustaining
                    # 1x realtime throughput, not maximizing it.
                    time.sleep(FRAME_SECONDS)

                    # Periodically peek at server frames so we can
                    # account for ``BACKPRESSURE_OVERFLOW`` errors and
                    # detect early aborts before the close handshake.
                    if i % 4 == 0:
                        drained = _drain_messages(ws, max_wait_seconds=0.0)
                        for msg in drained:
                            if msg.get("type") == "error":
                                code = msg.get("code")
                                if code == "BACKPRESSURE_OVERFLOW":
                                    overflow_count += 1
                                elif code == "INFERENCE_CRASH":
                                    crashes += 1
                                    aborted_early = True

                # Politely close so the route's finally: block runs.
                try:
                    ws.close(code=1000)
                except Exception:  # noqa: BLE001
                    pass
                # One last drain to pick up the final summary and any
                # tail errors. We don't fail the soak on missing finals
                # -- the load test (T21) already covers that path.
                tail = _drain_messages(ws, max_wait_seconds=2.0)
                for msg in tail:
                    if msg.get("type") == "error":
                        code = msg.get("code")
                        if code == "BACKPRESSURE_OVERFLOW":
                            overflow_count += 1
                        elif code == "INFERENCE_CRASH":
                            crashes += 1
    except Exception as e:  # noqa: BLE001
        results.put(
            {
                "conn_id": conn_id,
                "error": repr(e),
                "frames_pushed": frames_pushed,
                "overflow_count": overflow_count,
                "crashes": crashes,
                "sessions_opened": sessions_opened,
            }
        )
        return

    results.put(
        {
            "conn_id": conn_id,
            "frames_pushed": frames_pushed,
            "overflow_count": overflow_count,
            "crashes": crashes,
            "sessions_opened": sessions_opened,
        }
    )


@pytest.mark.skip(
    reason="manual soak test - run with --run-soak or unskip locally"
)
def test_soak_4_connections(patched_app):
    """4 connections * SOAK_MINUTES * realtime = the soak.

    The default is 60 minutes; set ``SOAK_MINUTES=5`` for a quick
    smoke. The test asserts no RSS leak, no thread leak, and per-conn
    overflow rate < 1 %.
    """
    pcm, sr = _load_sample_pcm()

    # Force a GC pass before the baseline so warmup objects from
    # earlier tests don't bias the RSS-growth measurement.
    gc.collect()
    proc = psutil.Process(os.getpid())
    rss_before = proc.memory_info().rss
    threads_before = threading.active_count()

    results: Queue = Queue()
    deadline = time.monotonic() + SOAK_MINUTES * 60

    with TestClient(patched_app) as client:
        threads = [
            threading.Thread(
                target=_soak_one_connection,
                args=(client, sr, pcm, deadline, results, i),
                name=f"soak-conn-{i}",
                daemon=True,
            )
            for i in range(NUM_CONNECTIONS)
        ]
        for t in threads:
            t.start()
        # Generous join slack: the workers self-terminate at ``deadline``
        # but a session in progress may still be flushing its tail.
        join_budget = SOAK_MINUTES * 60 + 30
        for t in threads:
            t.join(timeout=join_budget)
        for t in threads:
            assert not t.is_alive(), (
                f"{t.name} did not finish within {join_budget}s"
            )

    # Sample RSS / threads after the client context has exited so any
    # background portal threads have had a chance to wind down.
    gc.collect()
    rss_after = proc.memory_info().rss
    threads_after = threading.active_count()

    collected: list[dict] = []
    while True:
        try:
            collected.append(results.get_nowait())
        except Empty:
            break

    assert len(collected) == NUM_CONNECTIONS, (
        f"expected {NUM_CONNECTIONS} thread results, got {len(collected)}"
    )

    # No worker raised; if one did, surface the partial counters with
    # the repr so we can localize which connection misbehaved.
    for c in collected:
        assert "error" not in c, (
            f"conn {c['conn_id']} raised: {c.get('error')!r}; partial={c}"
        )

    # 1) Memory leak guard.
    rss_growth = rss_after - rss_before
    assert rss_growth < MAX_RSS_GROWTH_BYTES, (
        f"RSS grew by {rss_growth / 1e6:.1f} MB during soak "
        f"(budget {MAX_RSS_GROWTH_BYTES / 1e6:.0f} MB). "
        f"before={rss_before / 1e6:.1f} MB, after={rss_after / 1e6:.1f} MB"
    )

    # 2) Thread leak guard.
    thread_growth = threads_after - threads_before
    assert thread_growth <= MAX_THREAD_GROWTH, (
        f"thread count grew by {thread_growth} during soak "
        f"(budget {MAX_THREAD_GROWTH}). "
        f"before={threads_before}, after={threads_after}"
    )

    # 3) Per-connection overflow guard.
    for c in collected:
        pushed = max(c["frames_pushed"], 1)
        ratio = c["overflow_count"] / pushed
        assert ratio < MAX_OVERFLOW_RATIO, (
            f"conn {c['conn_id']} overflow rate {ratio:.4f} "
            f"({c['overflow_count']} / {pushed}) exceeds budget "
            f"{MAX_OVERFLOW_RATIO}"
        )

    # Print a concise summary so the operator running the soak by hand
    # sees throughput / sessions / crashes at a glance.
    print(
        f"\n[soak] minutes={SOAK_MINUTES} conns={NUM_CONNECTIONS} "
        f"rss_growth={rss_growth / 1e6:.1f}MB "
        f"thread_growth={thread_growth} "
        f"per-conn={[{'pushed': c['frames_pushed'], 'sessions': c['sessions_opened'], 'crashes': c['crashes'], 'overflow': c['overflow_count']} for c in collected]}"
    )
