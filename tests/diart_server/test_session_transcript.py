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


def _make_session(loop, on_message, store=None):
    seg, emb = _build_dummy_models()
    params = QueryParams(
        sample_rate=16000, pipeline="diarization",
        segmentation_model="dummy/seg", embedding_model="dummy/emb",
    )
    return DiarizationSession(
        params=params, registry=FakeRegistry(seg, emb), on_message=on_message,
        loop=loop, store=store, asr_client=AsrClient("http://x/v1", "m", "EMPTY", 5.0),
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


# ---------------------------------------------------------------------------
# Per-turn voiceprint identity (2026-07-06 false-accept fix): a transcript
# turn's name/score come from the turn's OWN (overlap-excised) slice embedded
# through _identify_slice on EMBED_EXECUTOR - never from the cluster label's
# cached identity. See docs/bugs/2026-07-06-voiceprint-false-accept.md.
# ---------------------------------------------------------------------------


class _FakeStore:
    """Minimal store stand-in: only truthiness/vectors() are ever touched
    (and vectors() only inside _identify_slice, which the tests replace)."""

    def vectors(self, model_name):  # noqa: ARG002
        return {}


def _fake_asr(monkeypatch, text="文本"):
    async def fake_transcribe(self, pcm, language=None):  # noqa: ANN001, ARG001
        return ("Chinese", text)

    monkeypatch.setattr(AsrClient, "transcribe", fake_transcribe)


def _with_cohort(monkeypatch):
    """Give dummy/emb a cohort mean so per-turn identification stays enabled.

    Cohort-less sessions skip identification entirely
    (docs/bugs/2026-08-05-cohort-missing-threshold.md §4.4); these tests
    exercise the identify path itself, so they need a cohort present.
    """
    unit = np.zeros(8, dtype=np.float32)
    unit[0] = 1.0
    monkeypatch.setattr("diart_server.session._load_cohort_mean", lambda name: unit)


def test_turn_identity_from_own_slice_not_cluster_cache(monkeypatch):
    """A >= min_identify_seconds turn is identified from its own excised slice;
    a stale _last_match (the old cluster-label identity) is ignored, and the
    verified identity is written back for the prediction-stream hint."""
    _fake_asr(monkeypatch)
    _with_cohort(monkeypatch)
    loop, t = _spin_loop()
    messages, lock = [], threading.Lock()

    async def on_message(msg):
        with lock:
            messages.append(msg)

    session = _make_session(loop, on_message, store=_FakeStore())
    received = []

    def fake_identify(pcm):
        received.append(np.array(pcm, copy=True))
        return ("alice", 0.9)

    session._identify_slice = fake_identify
    try:
        session._asr_sem = asyncio.Semaphore(2)
        session.ring.append(np.full(16000 * 3, 0.05, dtype=np.float32))
        # Stale cluster-label identity: must NOT leak into the transcript.
        session._last_match["speaker0"] = ("bob", 0.99)
        # Minority overlap 0.5-1.0s (ratio 0.2 <= 0.5) -> excised (zeroed).
        session._overlap_intervals = [(0.5, 1.0)]
        _run_handle_turn(session, FinalizedTurn("speaker0", 0.0, 2.5, "TID-ID"), loop)
    finally:
        _stop_loop(loop, t)

    with lock:
        transcripts = [m for m in messages if m.get("type") == "transcript"]
    assert len(transcripts) == 1
    tr = transcripts[0]
    assert tr["name"] == "alice" and tr["score"] == 0.9, tr
    # Identify received the turn's OWN slice, overlap-excised.
    assert len(received) == 1
    pcm = received[0]
    sr = 16000
    assert pcm.size == int(2.5 * sr)
    assert np.all(pcm[int(0.5 * sr):int(1.0 * sr)] == 0.0), "overlap not excised"
    assert np.all(pcm[: int(0.5 * sr)] != 0.0)
    # Prediction-stream hint now echoes the turn-verified identity.
    assert session._last_match["speaker0"] == ("alice", 0.9)


def test_short_turn_gets_no_name_even_with_cached_match(monkeypatch):
    """A turn below min_identify_seconds emits name=None/score=None - no
    centroid fallback, no previous-turn fallback (both were the bug)."""
    _fake_asr(monkeypatch)
    _with_cohort(monkeypatch)  # so the DURATION gate (not the cohort gate) is tested
    loop, t = _spin_loop()
    messages, lock = [], threading.Lock()

    async def on_message(msg):
        with lock:
            messages.append(msg)

    session = _make_session(loop, on_message, store=_FakeStore())
    calls = []
    session._identify_slice = lambda pcm: calls.append(1) or ("alice", 0.9)
    try:
        session._asr_sem = asyncio.Semaphore(2)
        session.ring.append(np.full(16000 * 2, 0.05, dtype=np.float32))
        session._last_match["speaker0"] = ("bob", 0.99)  # pre-seeded stale id
        # 1.0s: above the 0.25s ASR gate, below min_identify_seconds (2.0).
        _run_handle_turn(session, FinalizedTurn("speaker0", 0.0, 1.0, "TID-S"), loop)
    finally:
        _stop_loop(loop, t)

    with lock:
        transcripts = [m for m in messages if m.get("type") == "transcript"]
    assert len(transcripts) == 1
    assert transcripts[0]["name"] is None and transcripts[0]["score"] is None
    assert calls == [], "identify must not run for sub-min_identify turns"


def test_identify_failure_fails_closed_but_emits_transcript(monkeypatch):
    """An exception inside identification yields (None, None) and the
    transcript is still emitted (identity must never block ASR output)."""
    _fake_asr(monkeypatch)
    _with_cohort(monkeypatch)  # identification must actually RUN (and blow up)
    loop, t = _spin_loop()
    messages, lock = [], threading.Lock()

    async def on_message(msg):
        with lock:
            messages.append(msg)

    session = _make_session(loop, on_message, store=_FakeStore())

    def boom(pcm):  # noqa: ARG001
        raise RuntimeError("embedder exploded")

    session._identify_slice = boom
    try:
        session._asr_sem = asyncio.Semaphore(2)
        session.ring.append(np.full(16000 * 3, 0.05, dtype=np.float32))
        session._last_match["speaker0"] = ("bob", 0.99)
        _run_handle_turn(session, FinalizedTurn("speaker0", 0.0, 2.5, "TID-E"), loop)
    finally:
        _stop_loop(loop, t)

    with lock:
        transcripts = [m for m in messages if m.get("type") == "transcript"]
    assert len(transcripts) == 1, "transcript must still be emitted"
    assert transcripts[0]["text"] == "文本"
    assert transcripts[0]["name"] is None and transcripts[0]["score"] is None


def test_absorbed_fragment_reidentifies_merged_span(monkeypatch):
    """BUG3 absorption re-emits the merged turn through _transcribe_and_emit,
    which must re-identify the MERGED slice (>=2s rule on merged duration)."""
    _fake_asr(monkeypatch)
    _with_cohort(monkeypatch)
    loop, t = _spin_loop()
    messages, lock = [], threading.Lock()

    async def on_message(msg):
        with lock:
            messages.append(msg)

    session = _make_session(loop, on_message, store=_FakeStore())
    received = []

    def fake_identify(pcm):
        received.append(np.array(pcm, copy=True))
        return ("alice", 0.9)

    session._identify_slice = fake_identify
    try:
        session._asr_sem = asyncio.Semaphore(2)
        session.ring.append(np.full(16000 * 3, 0.05, dtype=np.float32))
        # Normal 2.5s turn, then a 0.1s fragment within absorb_gap.
        _run_handle_turn(session, FinalizedTurn("speaker0", 0.0, 2.5, "TID-X"), loop)
        _run_handle_turn(session, FinalizedTurn("speaker0", 2.5, 2.6, "TID-FRAG"), loop)
    finally:
        _stop_loop(loop, t)

    with lock:
        transcripts = [m for m in messages if m.get("type") == "transcript"]
    # Original turn + merged re-emit, both under TID-X.
    assert [m["turn_id"] for m in transcripts] == ["TID-X", "TID-X"]
    merged = transcripts[1]
    assert abs(merged["end"] - 2.6) < 1e-6
    assert merged["name"] == "alice" and merged["score"] == 0.9
    # Identify ran once per emission; the merge re-identified the FULL span.
    assert len(received) == 2
    assert received[0].size == int(2.5 * 16000)
    assert received[1].size == int(2.6 * 16000)


def test_make_embedder_default_device_follows_settings(monkeypatch):
    """Guard for the §3.2 landmine: make_embedder without an explicit device
    must resolve settings.device instead of letting SpeakerEmbedding move the
    SHARED registry model to CPU (which crashed live cuda sessions)."""
    import torch

    from diart_server import embedding_extractor as ee

    class RecordingModel:
        def __init__(self):
            self.to_calls = []

        def eval(self):
            return self

        def to(self, device):
            self.to_calls.append(device)
            return self

    class Reg:
        def __init__(self, model):
            self._m = model

        def get_embedding(self, name):  # noqa: ARG002
            return self._m

    model = RecordingModel()
    # Distinctive device value: proves the default is settings-driven, not
    # SpeakerEmbedding's cpu fallback.
    monkeypatch.setattr(ee.settings, "device", "meta")
    ee.make_embedder("any/model", Reg(model))
    assert model.to_calls == [torch.device("meta")], model.to_calls
