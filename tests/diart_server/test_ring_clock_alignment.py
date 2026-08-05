"""Ring/diart clock alignment under queue-overflow frame drops (B2).

Deterministic, no models on the hot path: drives a real
:class:`MeetingAudioSource` wired ``on_frame=ring.append`` (exactly how
:class:`DiarizationSession` builds it) with forced drops (small ``maxsize``,
reader started only after pushing) and asserts the ring ticks diart's
consumed-samples clock EXACTLY — a frame enters the ring iff diart consumes
it. Encodes the acceptance assertions of
docs/bugs/2026-08-05-ring-diart-clock-desync.md §4.3/§5 (pre-fix: the ring
counted every pushed sample, so after 30 dropped frames of 2730 samples
``ring.slice`` returned audio exactly 5.1188 s earlier than the labeled turn).
"""

from __future__ import annotations

import asyncio
import logging
import sys
import threading
import time
from pathlib import Path

import numpy as np

# Ensure the repo's tests/conftest.py is importable so we can reuse the
# DummySegmentationModel / DummyEmbeddingModel classes defined there.
TESTS_DIR = Path(__file__).resolve().parent.parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from conftest import DummyEmbeddingModel, DummySegmentationModel  # noqa: E402
from diart.models import EmbeddingModel, SegmentationModel  # noqa: E402

from diart_server import metrics  # noqa: E402
from diart_server.meeting_audio_source import MeetingAudioSource  # noqa: E402
from diart_server.pcm_ring_buffer import PcmRingBuffer  # noqa: E402
from diart_server.protocol import QueryParams  # noqa: E402
from diart_server.session import DiarizationSession  # noqa: E402

SR = 16000
FRAME = 2730  # ~170.6 ms: live.html ScriptProcessor 8192 @ 48 kHz -> 16 kHz


class FakeRegistry:
    def __init__(self, seg, emb):
        self._seg, self._emb = seg, emb

    def get_segmentation(self, name):  # noqa: ARG002
        return self._seg

    def get_embedding(self, name):  # noqa: ARG002
        return self._emb


def _make_session():
    seg = SegmentationModel(DummySegmentationModel)
    emb = EmbeddingModel(DummyEmbeddingModel)
    params = QueryParams(
        sample_rate=16000, pipeline="diarization",
        segmentation_model="dummy/seg", embedding_model="dummy/emb",
    )
    # These tests never start() the session nor dispatch onto the loop, so an
    # inert (closed) loop placeholder is enough and leaks nothing.
    loop = asyncio.new_event_loop()
    loop.close()

    async def on_message(msg):  # pragma: no cover - never invoked here
        pass

    return DiarizationSession(
        params=params,
        registry=FakeRegistry(seg, emb),
        on_message=on_message,
        loop=loop,
    )


def _drive_forced_drops(
    n_frames: int = 40, maxsize: int = 10
) -> tuple[PcmRingBuffer, list[np.ndarray], int]:
    """Push ``n_frames`` ramp frames into a stalled source (forcing
    ``n_frames - maxsize`` oldest-frame drops), then drain like the inference
    thread would. Returns (ring, consumed frames, drop count)."""
    drops = [0]
    ring = PcmRingBuffer(120.0, sample_rate=SR)
    src = MeetingAudioSource(
        uri="t",
        maxsize=maxsize,
        on_overflow=lambda: drops.__setitem__(0, drops[0] + 1),
        on_frame=ring.append,
    )

    # Ramp audio: sample i (producer order) has value i, so every sample is
    # globally identifiable.
    total = 0
    for _ in range(n_frames):
        src.push_pcm(np.arange(total, total + FRAME, dtype=np.float32))
        total += FRAME

    consumed: list[np.ndarray] = []
    src.stream.subscribe(on_next=lambda t: consumed.append(t.squeeze(0).numpy()))
    th = threading.Thread(target=src.read, daemon=True)
    th.start()
    # Wait for the surviving frames to drain before close() so the sentinel
    # never evicts a queued frame (close drops one on a full queue).
    deadline = time.monotonic() + 5.0
    while len(consumed) < maxsize and time.monotonic() < deadline:
        time.sleep(0.01)
    src.close()
    th.join(timeout=2.0)
    assert not th.is_alive()
    return ring, consumed, drops[0]


def test_ring_clock_equals_consumed_samples_exactly() -> None:
    """With >= 30 forced drops, ring.total_seconds == consumed/SR EXACTLY."""
    ring, consumed, drops = _drive_forced_drops()

    assert drops >= 30
    consumed_samples = sum(a.size for a in consumed)
    # EXACT equality (no tolerance): both sides are multiples of 1/16000.
    # Pre-fix the ring counted every pushed sample and exceeded this by
    # drops * FRAME / SR = 5.1188 s.
    assert ring.total_seconds == consumed_samples / SR


def test_ring_slice_returns_the_consumed_frame_for_consumer_window() -> None:
    """ring.slice over the k-th consumer-timeline window returns sample-exact
    the k-th CONSUMED frame (which, with drops forced, is NOT the k-th pushed
    frame)."""
    ring, consumed, drops = _drive_forced_drops()

    assert drops >= 30
    # With drops forced, the first consumed frame is a later producer frame:
    # its first sample value is the number of dropped-so-far samples, not 0.
    assert consumed[0][0] == drops * FRAME

    for k, frame in enumerate(consumed):
        window = ring.slice(k * FRAME / SR, (k + 1) * FRAME / SR)
        assert window is not None
        assert np.array_equal(window, frame), f"window {k} != consumed frame {k}"


def test_session_push_pcm_no_longer_feeds_ring() -> None:
    """Session-level wiring guard: push_pcm with the reader NOT running leaves
    ring.total_seconds == 0.0; after the source drains, it equals the consumed
    duration (drops excluded)."""
    session = _make_session()

    # 55 frames into the default maxsize=50 queue, reader not running:
    # 5 oldest dropped, nothing consumed yet.
    for i in range(55):
        session.push_pcm(np.full(160, float(i), dtype=np.float32))
    assert session.ring.total_seconds == 0.0
    assert session._overflow_count == 5

    # Drain the source the way the inference thread would (StreamingInference
    # only subscribes in __call__, so read() emits to no other observer).
    consumed: list[int] = []
    session.source.stream.subscribe(on_next=lambda t: consumed.append(t.shape[1]))
    th = threading.Thread(target=session.source.read, daemon=True)
    th.start()
    deadline = time.monotonic() + 5.0
    while len(consumed) < 50 and time.monotonic() < deadline:
        time.sleep(0.01)
    session.source.close()
    th.join(timeout=2.0)
    assert not th.is_alive()

    assert len(consumed) == 50
    assert session.ring.total_seconds == sum(consumed) / SR
    assert session.ring.total_seconds == 50 * 160 / SR  # dropped 5 excluded


def test_overflow_log_rate_limited_counter_not(caplog) -> None:
    """N overflow callbacks within _OVERFLOW_LOG_INTERVAL emit exactly 1
    warning while the Prometheus counter increments N times."""
    session = _make_session()
    before = metrics.queue_overflow_total._value.get()

    with caplog.at_level(logging.WARNING, logger="diart_server.session"):
        for _ in range(7):
            session._on_queue_overflow()

    assert metrics.queue_overflow_total._value.get() - before == 7
    assert session._overflow_count == 7
    warnings = [
        r for r in caplog.records if "audio queue overflow" in r.getMessage()
    ]
    assert len(warnings) == 1
    # The single line carries the drop count at log time (the first drop).
    assert "1 frames dropped" in warnings[0].getMessage()
