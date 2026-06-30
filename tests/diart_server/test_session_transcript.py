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
