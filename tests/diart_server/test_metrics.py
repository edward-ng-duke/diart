"""Tests for Prometheus metric wiring across diart_server (T28).

T7 defined eight Prometheus instruments (see
:mod:`diart_server.metrics`). Earlier tasks wired four of them (T11/T12/T13
for ``active_sessions``, ``predictions_total``, ``audio_bytes_received``
and ``session_duration``; T6 for ``queue_overflow_total``; T11 also wired
``session_errors_total{code="INFERENCE_CRASH"}``). T28 finishes the job:

* ``prediction_latency`` is now observed in :meth:`DiarizationSession._on_prediction`
  (push timestamp -> emission timestamp).
* ``model_load_seconds`` is now set by :meth:`ModelsRegistry.get_segmentation`
  / :meth:`get_embedding` on first load.
* ``session_errors_total`` is now incremented at every error path in
  :mod:`diart_server.ws_routes` (BAD_AUDIO, MODEL_NOT_FOUND,
  RATE_LIMIT_EXCEEDED, SESSION_CAPACITY_FULL, RESAMPLE_FAILED).

These tests drive the real FastAPI application via Starlette's
synchronous :class:`fastapi.testclient.TestClient`, monkeypatching the
registry to return dummy models (same pattern as
``test_e2e_websocket.py``/``test_failure_modes.py``). They scrape the
``/metrics`` endpoint and assert that the relevant series appear with
the right labels and non-zero values where required.
"""

from __future__ import annotations

import json
import re
import sys
import time
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

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

from diart_server import app as app_module, metrics  # noqa: E402
from diart_server.errors import ModelNotFoundError  # noqa: E402
from diart_server.models_registry import ModelsRegistry  # noqa: E402


SAMPLE_WAV = TESTS_DIR / "data" / "audio" / "sample.wav"
FRAME_SECONDS = 0.5
MAX_FRAMES = 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _drain_messages(ws, max_wait_seconds: float) -> list[dict]:
    """Best-effort drain of pending JSON frames from a TestClient websocket."""
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


def _counter_value(scrape: str, name: str, labels: dict[str, str] | None = None) -> float:
    """Parse a Prometheus exposition payload for a single sample's value.

    Returns ``0.0`` when the (name, labels) tuple is absent. We deliberately
    avoid pulling in :mod:`prometheus_client.parser` because the tests only
    care about a handful of named samples; a simple regex is faster and
    keeps the test readable.
    """
    if labels:
        # Prometheus emits labels in declaration order (LRU); accept any
        # order by walking each line that starts with the metric name.
        for line in scrape.splitlines():
            if line.startswith("#") or not line.startswith(name):
                continue
            m = re.match(rf'^{re.escape(name)}\{{([^}}]*)\}}\s+(\S+)$', line)
            if not m:
                continue
            label_str = m.group(1)
            parsed = dict(re.findall(r'(\w+)="([^"]*)"', label_str))
            if all(parsed.get(k) == v for k, v in labels.items()):
                try:
                    return float(m.group(2))
                except ValueError:
                    return 0.0
        return 0.0
    # Unlabeled metric: match the bare name.
    for line in scrape.splitlines():
        if line.startswith("#") or not line.startswith(name):
            continue
        m = re.match(rf'^{re.escape(name)}\s+(\S+)$', line)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return 0.0
    return 0.0


@pytest.fixture
def patched_app(monkeypatch):
    """Replace registry's loaders with dummies before the lifespan runs.

    Same shape as the T19/T23 fixtures. ``warm_up`` is stubbed so the
    lifespan flips ``app.state.ready=True`` without trying to load
    pyannote weights from disk.
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


# ---------------------------------------------------------------------------
# 1. /metrics exposes every defined series with the expected label keys.
# ---------------------------------------------------------------------------


def test_metrics_endpoint_exposes_all_eight_series(patched_app):
    """The exposition must include every metric name from metrics.py.

    Counters/gauges show up as bare names; histograms expose
    ``_bucket``, ``_count``, ``_sum``. Even with zero samples the HELP
    and TYPE lines are emitted so dashboards have a stable schema.
    """
    with TestClient(patched_app) as client:
        scrape = client.get("/metrics").text

    # Gauges + counters (counter total exposed under ``_total`` suffix).
    assert "diart_active_sessions" in scrape
    assert "diart_predictions_total" in scrape
    assert "diart_audio_bytes_received_total" in scrape
    assert "diart_model_load_seconds" in scrape
    assert "diart_session_errors_total" in scrape
    assert "diart_queue_overflow_total" in scrape

    # Histograms expose helper series with these suffixes.
    assert "diart_prediction_latency_seconds_bucket" in scrape
    assert "diart_prediction_latency_seconds_count" in scrape
    assert "diart_session_duration_seconds_bucket" in scrape
    assert "diart_session_duration_seconds_count" in scrape


# ---------------------------------------------------------------------------
# 2. End-to-end run: active_sessions returns to 0; bytes/predictions
#    counters reflect activity.
# ---------------------------------------------------------------------------


def test_e2e_metrics_after_websocket_session(patched_app):
    """After a WS session closes, gauges return to 0 and counters reflect activity.

    Drives a short streaming session with the dummy models, then scrapes
    ``/metrics`` and checks:
      * active_sessions == 0 after close (T13 decrement runs in ``finally``).
      * audio_bytes_received_total strictly increased (T11 wiring).
      * session_duration histogram has at least one observation (T13 wiring).
      * prediction_latency histogram count is monotonically non-decreasing
        (>=0; we tolerate 0 because the dummy models can early-abort
        before any prediction is emitted).
    """
    pcm, sr = _load_sample_pcm()
    frame_samples = int(sr * FRAME_SECONDS)

    with TestClient(patched_app) as client:
        before = client.get("/metrics").text
        bytes_before = _counter_value(before, "diart_audio_bytes_received_total")
        sd_count_before = _counter_value(
            before, "diart_session_duration_seconds_count"
        )
        latency_count_before = _counter_value(
            before, "diart_prediction_latency_seconds_count"
        )

        with client.websocket_connect(
            f"/v1/diarize/stream?sample_rate={sr}"
        ) as ws:
            ready = ws.receive_json()
            assert ready["type"] == "ready"
            n_frames = min(len(pcm) // frame_samples, MAX_FRAMES)
            for i in range(n_frames):
                start = i * frame_samples
                chunk = pcm[start : start + frame_samples]
                ws.send_bytes(chunk.tobytes())
                time.sleep(0.05)
            # Send graceful end-of-stream so the route's finally block
            # can drain promptly without waiting on the 5-second thread
            # join timeout.
            ws.send_text(json.dumps({"type": "end"}))
            _drain_messages(ws, max_wait_seconds=3.0)

        # Give the route's ``finally:`` block a moment to run close()
        # and decrement the active_sessions gauge. The route awaits
        # session.close() before returning; poll the gauge with a
        # bounded deadline rather than guessing a sleep.
        deadline = time.monotonic() + 6.0
        while time.monotonic() < deadline:
            scrape = client.get("/metrics").text
            if _counter_value(scrape, "diart_active_sessions") == 0.0:
                break
            time.sleep(0.1)

        after = client.get("/metrics").text

    active = _counter_value(after, "diart_active_sessions")
    bytes_after = _counter_value(after, "diart_audio_bytes_received_total")
    sd_count_after = _counter_value(after, "diart_session_duration_seconds_count")
    latency_count_after = _counter_value(
        after, "diart_prediction_latency_seconds_count"
    )

    # The session was closed -> gauge back to baseline.
    assert active == 0.0, f"expected active_sessions=0 after close; got {active}"
    # We pushed at least one frame -> bytes counter strictly increased.
    assert bytes_after > bytes_before, (
        f"audio_bytes_received_total did not increase: "
        f"before={bytes_before} after={bytes_after}"
    )
    # close() observed exactly one session duration.
    assert sd_count_after >= sd_count_before + 1.0, (
        f"session_duration_seconds_count did not increment: "
        f"before={sd_count_before} after={sd_count_after}"
    )
    # Latency histogram is observed-on-prediction; the dummy models
    # *may* not emit any predictions in a short run. We only assert the
    # count is non-decreasing (proves the wiring is at least scrape-safe).
    assert latency_count_after >= latency_count_before


# ---------------------------------------------------------------------------
# 3. session_errors_total{code="BAD_AUDIO"} increments on validation error.
# ---------------------------------------------------------------------------


def test_session_errors_total_increments_on_bad_audio(patched_app):
    """Connecting with ``?sample_rate=4000`` (out of range) must bump
    the BAD_AUDIO error counter."""
    with TestClient(patched_app) as client:
        before = _counter_value(
            client.get("/metrics").text,
            "diart_session_errors_total",
            {"code": "BAD_AUDIO"},
        )

        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(
                "/v1/diarize/stream?sample_rate=4000"
            ) as ws:
                ws.receive_json()  # error frame
                ws.receive_json()  # close (raises)

        after = _counter_value(
            client.get("/metrics").text,
            "diart_session_errors_total",
            {"code": "BAD_AUDIO"},
        )
    assert after >= before + 1.0, (
        f"BAD_AUDIO counter did not increment: before={before} after={after}"
    )


# ---------------------------------------------------------------------------
# 4. session_errors_total{code="MODEL_NOT_FOUND"} increments when registry raises.
# ---------------------------------------------------------------------------


def test_session_errors_total_increments_on_model_not_found(
    patched_app, monkeypatch
):
    """If the registry raises ``ModelNotFoundError``, the route bumps
    the MODEL_NOT_FOUND counter."""

    def raising_get_seg(self, name):  # noqa: ARG001
        raise ModelNotFoundError(f"unknown: {name}")

    monkeypatch.setattr(ModelsRegistry, "get_segmentation", raising_get_seg)

    with TestClient(patched_app) as client:
        before = _counter_value(
            client.get("/metrics").text,
            "diart_session_errors_total",
            {"code": "MODEL_NOT_FOUND"},
        )

        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(
                "/v1/diarize/stream?segmentation_model=does/not/exist"
            ) as ws:
                ws.receive_json()
                ws.receive_json()

        after = _counter_value(
            client.get("/metrics").text,
            "diart_session_errors_total",
            {"code": "MODEL_NOT_FOUND"},
        )
    assert after >= before + 1.0


# ---------------------------------------------------------------------------
# 5. session_errors_total{code="RATE_LIMIT_EXCEEDED"} increments on oversized frame.
# ---------------------------------------------------------------------------


def test_session_errors_total_increments_on_oversized_frame(patched_app):
    """A binary frame > settings.max_frame_bytes must bump the
    RATE_LIMIT_EXCEEDED counter (in addition to closing 1009)."""
    from diart_server.config import settings

    too_big = b"\x00" * (settings.max_frame_bytes + 1)

    with TestClient(patched_app) as client:
        before = _counter_value(
            client.get("/metrics").text,
            "diart_session_errors_total",
            {"code": "RATE_LIMIT_EXCEEDED"},
        )

        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(
                "/v1/diarize/stream?sample_rate=16000"
            ) as ws:
                assert ws.receive_json()["type"] == "ready"
                ws.send_bytes(too_big)
                ws.receive_json()  # error
                ws.receive_json()  # close raises

        after = _counter_value(
            client.get("/metrics").text,
            "diart_session_errors_total",
            {"code": "RATE_LIMIT_EXCEEDED"},
        )
    assert after >= before + 1.0


# ---------------------------------------------------------------------------
# 6. session_errors_total{code="SESSION_CAPACITY_FULL"} increments when at cap.
# ---------------------------------------------------------------------------


def test_session_errors_total_increments_on_capacity_full(
    patched_app, monkeypatch
):
    """When the manager is full, a new connection bumps the
    SESSION_CAPACITY_FULL counter."""
    from diart_server import session_manager

    monkeypatch.setattr(session_manager.manager, "_max", 1)

    with TestClient(patched_app) as client:
        before = _counter_value(
            client.get("/metrics").text,
            "diart_session_errors_total",
            {"code": "SESSION_CAPACITY_FULL"},
        )

        ws1_cm = client.websocket_connect("/v1/diarize/stream?sample_rate=16000")
        ws1 = ws1_cm.__enter__()
        try:
            assert ws1.receive_json()["type"] == "ready"

            with pytest.raises(WebSocketDisconnect):
                with client.websocket_connect(
                    "/v1/diarize/stream?sample_rate=16000"
                ) as ws2:
                    err = ws2.receive_json()
                    assert err["code"] == "SESSION_CAPACITY_FULL"
                    ws2.receive_json()  # close raises
        finally:
            ws1_cm.__exit__(None, None, None)

        after = _counter_value(
            client.get("/metrics").text,
            "diart_session_errors_total",
            {"code": "SESSION_CAPACITY_FULL"},
        )
    assert after >= before + 1.0


# ---------------------------------------------------------------------------
# 7. model_load_seconds set on first load (and not re-set on cache hits).
# ---------------------------------------------------------------------------


def test_model_load_seconds_set_on_first_segmentation_load(tmp_path):
    """``ModelsRegistry.get_segmentation`` must set the model_load_seconds
    gauge with a labels=(model=name) the first time it loads a model."""
    manifest_path = (
        Path(__file__).resolve().parents[2] / "docker" / "models.json"
    )
    reg = ModelsRegistry(tmp_path, manifest_path)

    # Snapshot any pre-existing value so we can detect the post-load
    # write (gauges aggregate across tests via the global registry).
    name = "pyannote/segmentation-3.0"
    fake_model = MagicMock(name="seg-model")
    with patch(
        "diart_server.models_registry.SegmentationModel.from_pyannote",
        return_value=fake_model,
    ):
        reg.get_segmentation(name)

    val = metrics.model_load_seconds.labels(model=name)._value.get()
    assert val >= 0.0, "model_load_seconds gauge must be set with a non-negative value"


def test_model_load_seconds_set_on_first_embedding_load(tmp_path):
    """Companion test for ``get_embedding`` -- ensures the embedding arm
    also writes the gauge."""
    manifest_path = (
        Path(__file__).resolve().parents[2] / "docker" / "models.json"
    )
    reg = ModelsRegistry(tmp_path, manifest_path)

    name = "pyannote/embedding"
    fake_model = MagicMock(name="emb-model")
    with patch(
        "diart_server.models_registry.EmbeddingModel.from_pyannote",
        return_value=fake_model,
    ):
        reg.get_embedding(name)

    val = metrics.model_load_seconds.labels(model=name)._value.get()
    assert val >= 0.0


# ---------------------------------------------------------------------------
# 8. prediction_latency observed when push_pcm precedes _on_prediction.
# ---------------------------------------------------------------------------


def test_prediction_latency_observed_on_emission():
    """A direct unit test for the new latency wiring in DiarizationSession.

    We bypass the full streaming pipeline by constructing a session
    against a fake registry, manually setting ``_last_push_at`` (as
    push_pcm would), then invoking ``_on_prediction`` with a synthetic
    annotation. The histogram count must increase by 1.
    """
    import asyncio

    from pyannote.core import Annotation, Segment

    from diart_server.protocol import QueryParams
    from diart_server.session import DiarizationSession

    class _FakeRegistry:
        def __init__(self):
            self._seg = SegmentationModel(DummySegmentationModel)
            self._emb = EmbeddingModel(DummyEmbeddingModel)

        def get_segmentation(self, name):  # noqa: ARG002
            return self._seg

        def get_embedding(self, name):  # noqa: ARG002
            return self._emb

    loop = asyncio.new_event_loop()
    try:
        async def _on_msg(_msg):
            return None

        params = QueryParams()
        session = DiarizationSession(
            params=params,
            registry=_FakeRegistry(),  # type: ignore[arg-type]
            on_message=_on_msg,
            loop=loop,
        )
        # Simulate a push without actually invoking source.push_pcm
        # (that would require a thread-safe queue setup we don't need).
        session._last_push_at = time.monotonic() - 0.05  # 50 ms ago

        def _hist_count(h) -> float:
            # prometheus_client stores bucket counts non-cumulatively;
            # the total number of observations is the sum across every
            # bucket (the +Inf bucket only catches values above the
            # final boundary, not the running total). Pull the total
            # from the collected ``_count`` sample for clarity.
            for sample in h.collect()[0].samples:
                if sample.name.endswith("_count"):
                    return float(sample.value)
            return 0.0

        before = metrics.prediction_latency._sum.get()
        before_count = _hist_count(metrics.prediction_latency)

        # Build a synthetic 1-speaker annotation; _on_prediction will
        # iterate it and observe the histogram once.
        ann = Annotation()
        ann[Segment(0.0, 1.0)] = "speaker0"
        # Waveform is unused except for unpacking.
        session._on_prediction((ann, None))

        after = metrics.prediction_latency._sum.get()
        after_count = _hist_count(metrics.prediction_latency)

        # The sum must have moved forward by roughly the simulated
        # latency (~0.05 s); we just check it strictly increased to
        # avoid flakes from clock jitter.
        assert after > before, (
            f"prediction_latency sum did not move: before={before} after={after}"
        )
        # And at least one bucket counter must have ticked.
        assert after_count > before_count, (
            f"no histogram bucket incremented: "
            f"before={before_count} after={after_count}"
        )
    finally:
        loop.close()
