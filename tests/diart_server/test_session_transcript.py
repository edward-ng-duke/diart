"""Deterministic test: a finalized turn runs ASR and emits a TranscriptMessage.

Drives ``DiarizationSession._handle_turn`` directly with a populated ring
buffer and a monkeypatched ``AsrClient``, so it exercises the real 0.25s ASR
quality gate without depending on the dummy diarization model's (short,
unstable) turn lengths.
"""
from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

TESTS_DIR = Path(__file__).resolve().parent.parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from conftest import DummyEmbeddingModel, DummySegmentationModel  # noqa: E402
from diart.models import EmbeddingModel, SegmentationModel  # noqa: E402

from diart_server.asr_client import AsrClient  # noqa: E402
from diart_server.protocol import QueryParams  # noqa: E402
from diart_server.session import DiarizationSession  # noqa: E402
from diart_server.turn_aggregator import FinalizedTurn  # noqa: E402


class FakeRegistry:
    def __init__(self, seg, emb):
        self._seg, self._emb = seg, emb

    def get_segmentation(self, name):  # noqa: ARG002
        return self._seg

    def get_embedding(self, name):  # noqa: ARG002
        return self._emb


def _build_dummy_models():
    return SegmentationModel(DummySegmentationModel), EmbeddingModel(DummyEmbeddingModel)


def _make_session(loop, on_message):
    seg, emb = _build_dummy_models()
    params = QueryParams(
        sample_rate=16000, pipeline="diarization",
        segmentation_model="dummy/seg", embedding_model="dummy/emb",
    )
    return DiarizationSession(
        params=params, registry=FakeRegistry(seg, emb), on_message=on_message,
        loop=loop, store=None, asr_client=AsrClient("http://x/v1", "m", "EMPTY", 5.0),
    )


def _run_handle_turn(session, ft, loop):
    fut = asyncio.run_coroutine_threadsafe(session._handle_turn(ft), loop)
    fut.result(timeout=10.0)


def test_handle_turn_emits_transcript(monkeypatch):
    async def fake_transcribe(self, pcm, language=None):  # noqa: ANN001, ARG001
        return ("Chinese", "你好世界")

    monkeypatch.setattr(AsrClient, "transcribe", fake_transcribe)

    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()
    messages, lock = [], threading.Lock()

    async def on_message(msg):
        with lock:
            messages.append(msg)

    session = _make_session(loop, on_message)
    try:
        session._asr_sem = asyncio.Semaphore(2)            # what start() sets
        session.ring.append(np.full(16000, 0.05, dtype=np.float32))  # 1.0s available
        _run_handle_turn(session, FinalizedTurn("speaker0", 0.0, 1.0), loop)  # 1.0s > 0.25s
    finally:
        loop.call_soon_threadsafe(loop.stop)
        t.join(timeout=2.0)
        loop.close()

    with lock:
        transcripts = [m for m in messages if m.get("type") == "transcript"]
    assert len(transcripts) == 1, [m.get("type") for m in messages]
    tr = transcripts[0]
    assert tr["text"] == "你好世界"
    assert tr["lang"] == "Chinese"
    assert tr["speaker"] == "speaker0"
    assert tr["start"] == 0.0 and tr["end"] == 1.0


def test_handle_turn_skips_below_gate(monkeypatch):
    calls = []

    async def fake_transcribe(self, pcm, language=None):  # noqa: ANN001, ARG001
        calls.append(1)
        return ("Chinese", "x")

    monkeypatch.setattr(AsrClient, "transcribe", fake_transcribe)

    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()
    messages, lock = [], threading.Lock()

    async def on_message(msg):
        with lock:
            messages.append(msg)

    session = _make_session(loop, on_message)
    try:
        session._asr_sem = asyncio.Semaphore(2)
        session.ring.append(np.full(16000, 0.05, dtype=np.float32))  # 1.0s available
        _run_handle_turn(session, FinalizedTurn("speaker0", 0.0, 0.1), loop)  # 0.1s < 0.25s
    finally:
        loop.call_soon_threadsafe(loop.stop)
        t.join(timeout=2.0)
        loop.close()

    assert calls == [], "ASR must be skipped for sub-0.25s turns"
    with lock:
        assert not any(m.get("type") == "transcript" for m in messages)


def _spin_loop():
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()
    return loop, t


def _stop_loop(loop, t):
    loop.call_soon_threadsafe(loop.stop)
    t.join(timeout=2.0)
    loop.close()


def test_open_turn_emits_partial_then_final(monkeypatch):
    """BUG1: an open turn emits >=1 partial (partial=True, non-empty turn_id),
    and the final for the SAME turn_id has partial=False."""
    async def fake_transcribe(self, pcm, language=None):  # noqa: ANN001, ARG001
        return ("Chinese", "部分文本")

    monkeypatch.setattr(AsrClient, "transcribe", fake_transcribe)

    loop, t = _spin_loop()
    messages, lock = [], threading.Lock()

    async def on_message(msg):
        with lock:
            messages.append(msg)

    session = _make_session(loop, on_message)
    try:
        session._asr_sem = asyncio.Semaphore(2)
        # 15s of audio available so any slice up to 15s passes the 0.25s gate.
        session.ring.append(np.full(16000 * 15, 0.05, dtype=np.float32))

        # Drive the aggregator + due_partials the way _on_prediction does.
        interval = 5.0
        end = 0.0
        while end < 12.0:
            start = end
            end = round(start + 0.5, 3)
            for ft in session.aggregator.feed("speaker0", start, end, now=end):
                _run_handle_turn(session, ft, loop)
            for pt in session.aggregator.due_partials(interval):
                fut = asyncio.run_coroutine_threadsafe(
                    session._handle_turn(pt, partial=True), loop
                )
                fut.result(timeout=10.0)
        # Finalize via flush.
        for ft in session.aggregator.flush(session.ring.total_seconds):
            _run_handle_turn(session, ft, loop)
    finally:
        _stop_loop(loop, t)

    with lock:
        transcripts = [m for m in messages if m.get("type") == "transcript"]
    partials = [m for m in transcripts if m.get("partial") is True]
    finals = [m for m in transcripts if m.get("partial") is False]
    assert len(partials) >= 1, transcripts
    assert all(p["turn_id"] for p in partials)
    # The final for the partial's turn_id exists with partial False.
    tid = partials[0]["turn_id"]
    assert any(f["turn_id"] == tid and f["partial"] is False for f in finals), \
        [(m.get("turn_id"), m.get("partial")) for m in transcripts]


def test_short_fragment_absorbed_into_previous(monkeypatch):
    """BUG3: a sub-0.25s fragment is absorbed into the previous same-speaker
    turn (re-emitted with the same turn_id and extended end), not dropped."""
    async def fake_transcribe(self, pcm, language=None):  # noqa: ANN001, ARG001
        return ("Chinese", "合并文本")

    monkeypatch.setattr(AsrClient, "transcribe", fake_transcribe)

    loop, t = _spin_loop()
    messages, lock = [], threading.Lock()

    async def on_message(msg):
        with lock:
            messages.append(msg)

    session = _make_session(loop, on_message)
    try:
        session._asr_sem = asyncio.Semaphore(2)
        session.ring.append(np.full(16000 * 2, 0.05, dtype=np.float32))  # 2.0s
        # Normal spk0 turn 0.0-1.0 with a known turn_id.
        first = FinalizedTurn("speaker0", 0.0, 1.0, "TID-X")
        _run_handle_turn(session, first, loop)
        # A 0.15s fragment within absorb_gap of the previous turn's end.
        frag = FinalizedTurn("speaker0", 1.0, 1.15, "TID-FRAG")
        _run_handle_turn(session, frag, loop)
    finally:
        _stop_loop(loop, t)

    with lock:
        transcripts = [m for m in messages if m.get("type") == "transcript"]
    # First transcript (the normal turn).
    assert transcripts[0]["turn_id"] == "TID-X"
    # The fragment did NOT vanish: a replacement carrying TID-X extending to 1.15.
    absorbed = [
        m for m in transcripts if m["turn_id"] == "TID-X" and abs(m["end"] - 1.15) < 1e-6
    ]
    assert absorbed, [(m["turn_id"], m["start"], m["end"]) for m in transcripts]
    # No orphan fragment turn_id emitted.
    assert not any(m["turn_id"] == "TID-FRAG" for m in transcripts)
