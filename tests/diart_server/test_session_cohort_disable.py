"""Cohort-less sessions must disable per-turn identification, loudly.

docs/bugs/2026-08-05-cohort-missing-threshold.md §4.4: settings.match_threshold
(0.82) is calibrated exclusively for the mean-centered score remapped to
[0, 1]; raw same-speaker cosines top out ~0.49 below it, so a session whose
embedding model has no cohort mean can mathematically never match. Such a
session must skip _identify_slice entirely (name=None, score=None, no
fallback, no embedder construction), warn once at construction naming the
model, and report ``voiceprint: disabled_no_cohort`` in the ready config.

Drives ``DiarizationSession._handle_turn`` directly with a populated ring
buffer and a monkeypatched ``AsrClient`` (same fixture style as
test_session_transcript.py) — never the flaky WS-path integration tests.
"""
from __future__ import annotations

import asyncio
import logging
import sys
import threading
from pathlib import Path

import numpy as np

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


class _FakeStore:
    """Minimal store stand-in: only truthiness/vectors() are ever touched
    (and vectors() only inside _identify_slice, which the tests replace)."""

    def vectors(self, model_name):  # noqa: ARG002
        return {}


def _make_session(loop, on_message, store=None):
    seg = SegmentationModel(DummySegmentationModel)
    emb = EmbeddingModel(DummyEmbeddingModel)
    params = QueryParams(
        sample_rate=16000, pipeline="diarization",
        segmentation_model="dummy/seg", embedding_model="dummy/emb",
    )
    return DiarizationSession(
        params=params, registry=FakeRegistry(seg, emb), on_message=on_message,
        loop=loop, store=store, asr_client=AsrClient("http://x/v1", "m", "EMPTY", 5.0),
    )


def _fake_asr(monkeypatch, text="文本"):
    async def fake_transcribe(self, pcm, language=None):  # noqa: ANN001, ARG001
        return ("Chinese", text)

    monkeypatch.setattr(AsrClient, "transcribe", fake_transcribe)


def _spin_loop():
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()
    return loop, t


def _stop_loop(loop, t):
    loop.call_soon_threadsafe(loop.stop)
    t.join(timeout=2.0)
    loop.close()


def _run_handle_turn(session, ft, loop):
    fut = asyncio.run_coroutine_threadsafe(session._handle_turn(ft), loop)
    fut.result(timeout=10.0)


def test_cohortless_session_skips_identify_and_warns(monkeypatch, caplog):
    """dummy/emb has no assets/cohort file: the session must emit the
    transcript with name=None/score=None WITHOUT ever touching the identify
    path, leave _last_match empty, report disabled_no_cohort in the ready
    config, and warn exactly once at construction naming the model."""
    _fake_asr(monkeypatch)
    embed_calls = []
    monkeypatch.setattr(
        "diart_server.session.extract_utterance_embedding",
        lambda *a, **k: embed_calls.append(1) or np.ones(4, dtype=np.float32),
    )

    loop, t = _spin_loop()
    messages, lock = [], threading.Lock()

    async def on_message(msg):
        with lock:
            messages.append(msg)

    with caplog.at_level(logging.WARNING, logger="diart_server.session"):
        session = _make_session(loop, on_message, store=_FakeStore())
    warnings = [
        r for r in caplog.records
        if r.name == "diart_server.session" and r.levelno == logging.WARNING
    ]
    assert len(warnings) == 1, [r.getMessage() for r in warnings]
    assert "dummy/emb" in warnings[0].getMessage()

    assert session.cohort_mean is None
    assert session.ready_message()["config"]["voiceprint"] == "disabled_no_cohort"

    identify_calls = []
    session._identify_slice = (
        lambda pcm: identify_calls.append(1) or ("alice", 0.9)
    )
    try:
        session._asr_sem = asyncio.Semaphore(2)
        session.ring.append(np.full(16000 * 3, 0.05, dtype=np.float32))
        # 2.5s >= min_identify_seconds (2.0): only the missing cohort gates.
        _run_handle_turn(session, FinalizedTurn("speaker0", 0.0, 2.5, "TID-NC"), loop)
    finally:
        _stop_loop(loop, t)

    with lock:
        transcripts = [m for m in messages if m.get("type") == "transcript"]
    assert len(transcripts) == 1, [m.get("type") for m in messages]
    tr = transcripts[0]
    assert tr["name"] is None and tr["score"] is None, tr
    assert tr["text"] == "文本"  # transcript itself never blocked
    assert identify_calls == [], "identify must not run without a cohort mean"
    assert embed_calls == [], "the utterance embedder must never be touched"
    assert session._last_match == {}


def test_cohort_present_session_identifies(monkeypatch):
    """With a cohort mean present (monkeypatched loader) the same drive DOES
    identify: ready config says enabled, _identify_slice runs, and the
    transcript carries the verified name."""
    _fake_asr(monkeypatch)
    unit = np.zeros(8, dtype=np.float32)
    unit[0] = 1.0
    monkeypatch.setattr(
        "diart_server.session._load_cohort_mean", lambda name: unit
    )

    loop, t = _spin_loop()
    messages, lock = [], threading.Lock()

    async def on_message(msg):
        with lock:
            messages.append(msg)

    session = _make_session(loop, on_message, store=_FakeStore())
    assert session.cohort_mean is unit
    assert session.ready_message()["config"]["voiceprint"] == "enabled"

    identify_calls = []

    def fake_identify(pcm):  # noqa: ARG001
        identify_calls.append(1)
        return ("alice", 0.9)

    session._identify_slice = fake_identify
    try:
        session._asr_sem = asyncio.Semaphore(2)
        session.ring.append(np.full(16000 * 3, 0.05, dtype=np.float32))
        _run_handle_turn(session, FinalizedTurn("speaker0", 0.0, 2.5, "TID-EN"), loop)
    finally:
        _stop_loop(loop, t)

    with lock:
        transcripts = [m for m in messages if m.get("type") == "transcript"]
    assert len(transcripts) == 1
    assert transcripts[0]["name"] == "alice" and transcripts[0]["score"] == 0.9
    assert identify_calls == [1]
