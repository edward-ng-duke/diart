"""B45: identity embeds only the turn's own PURE (non-overlapped) speech.

docs/bugs/2026-08-05-identity-continuity-overlap-purity.md §5.1: the identity
input is a sample-exact concatenation of the non-overlapped sub-slices (never
zeros, never a two-speaker mix), and the identification gate runs on PURE
seconds. Drives _apply_overlap directly (a) and _transcribe_and_emit via
_handle_turn with fakes (b-e), in the style of test_session_transcript.py.
"""
from __future__ import annotations

import asyncio
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

SR = 16000


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
        sample_rate=SR, pipeline="diarization",
        segmentation_model="dummy/seg", embedding_model="dummy/emb",
    )
    return DiarizationSession(
        params=params, registry=FakeRegistry(seg, emb), on_message=on_message,
        loop=loop, store=store, asr_client=AsrClient("http://x/v1", "m", "EMPTY", 5.0),
    )


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


def _fake_asr(monkeypatch, text="文本"):
    async def fake_transcribe(self, pcm, language=None):  # noqa: ANN001, ARG001
        return ("Chinese", text)

    monkeypatch.setattr(AsrClient, "transcribe", fake_transcribe)


def _with_cohort(monkeypatch):
    """Give dummy/emb a cohort mean so per-turn identification stays enabled
    (cohort-less sessions skip identification entirely;
    docs/bugs/2026-08-05-cohort-missing-threshold.md)."""
    unit = np.zeros(8, dtype=np.float32)
    unit[0] = 1.0
    monkeypatch.setattr("diart_server.session._load_cohort_mean", lambda name: unit)


def _ramp(seconds=5.0):
    """Strictly-positive ramp: every sample distinct and nonzero, so zeroing
    and sample-exact concatenation are both detectable."""
    n = int(seconds * SR)
    return (np.arange(1, n + 1, dtype=np.float32)) / n


# ---------------------------------------------------------------------------
# §5.1a: _apply_overlap builds a sample-exact pure concatenation for identity
# while the ASR pcm keeps today's zeroing.
# ---------------------------------------------------------------------------


def test_minority_overlap_identity_is_sample_exact_pure_concatenation():
    loop = asyncio.new_event_loop()
    try:
        session = _make_session(loop, lambda m: None)
    finally:
        loop.close()
    ramp = _ramp(5.0)  # 80000 samples
    ft = FinalizedTurn("speaker0", 0.0, 5.0, "T")
    session._overlap_intervals = [(2.0, 3.0)]  # ratio 0.2 -> minority

    asr, overlapped, ratio, identity = session._apply_overlap(ramp.copy(), ft)

    assert overlapped is False
    assert abs(ratio - 0.2) < 1e-6
    # Identity: 4.0s of PURE speech, sample-exact, no zeros inserted.
    assert identity.size == 64000
    assert np.array_equal(
        identity, np.concatenate([ramp[:32000], ramp[48000:80000]])
    )
    # ASR pcm: zeros exactly in [32000, 48000), the ramp elsewhere.
    assert np.all(asr[32000:48000] == 0.0)
    assert np.array_equal(asr[:32000], ramp[:32000])
    assert np.array_equal(asr[48000:], ramp[48000:])


# ---------------------------------------------------------------------------
# §5.1b-e: _transcribe_and_emit gates identification on PURE duration and
# never embeds mixed audio; below-threshold results still update _last_match.
# ---------------------------------------------------------------------------


def _run_turn_with_spy(monkeypatch, ring_pcm, ft, overlap_intervals,
                       spy_result=("alice", 0.9), last_match=None):
    """Drive one turn through _handle_turn; return (transcripts, calls, session)."""
    _fake_asr(monkeypatch)
    _with_cohort(monkeypatch)
    loop, t = _spin_loop()
    messages, lock = [], threading.Lock()

    async def on_message(msg):
        with lock:
            messages.append(msg)

    session = _make_session(loop, on_message, store=_FakeStore())
    calls = []

    def spy(pcm):
        calls.append(np.array(pcm, copy=True))
        return spy_result

    session._identify_slice = spy
    try:
        session._asr_sem = asyncio.Semaphore(2)
        session.ring.append(ring_pcm)
        if last_match is not None:
            session._last_match = dict(last_match)
        session._overlap_intervals = list(overlap_intervals)
        _run_handle_turn(session, ft, loop)
    finally:
        _stop_loop(loop, t)

    with lock:
        transcripts = [m for m in messages if m.get("type") == "transcript"]
    return transcripts, calls, session


def test_majority_overlap_skips_identify_and_preserves_last_match(monkeypatch):
    """§5.1b: mixed audio is never embedded; the stale hint is not poisoned."""
    transcripts, calls, session = _run_turn_with_spy(
        monkeypatch,
        ring_pcm=np.full(SR * 5, 0.05, dtype=np.float32),
        ft=FinalizedTurn("speaker0", 0.0, 5.0, "TID-MAJ"),
        overlap_intervals=[(0.5, 4.5)],  # ratio 0.8 -> majority
        last_match={"speaker0": ("seed", 0.5)},
    )
    assert calls == [], "identify must not run on majority-overlap (mixed) audio"
    assert session._last_match == {"speaker0": ("seed", 0.5)}
    assert len(transcripts) == 1
    tr = transcripts[0]
    assert tr["name"] is None and tr["score"] is None
    assert tr["overlapped"] is True


def test_pure_duration_below_gate_skips_identify(monkeypatch):
    """§5.1c: 3s turn with only 1.5s pure speech (< min_identify_seconds=2.0)
    must not be identified even though the TOTAL duration passes the gate."""
    transcripts, calls, session = _run_turn_with_spy(
        monkeypatch,
        ring_pcm=np.full(SR * 3, 0.05, dtype=np.float32),
        ft=FinalizedTurn("speaker0", 0.0, 3.0, "TID-PURE"),
        overlap_intervals=[(0.0, 1.5)],  # ratio 0.5 -> minority, pure 1.5s
        last_match={"speaker0": ("seed", 0.5)},
    )
    assert calls == [], "identify must not run below the PURE-seconds gate"
    assert session._last_match == {"speaker0": ("seed", 0.5)}
    assert len(transcripts) == 1
    assert transcripts[0]["name"] is None and transcripts[0]["score"] is None
    assert transcripts[0]["overlapped"] is False


def test_pure_duration_above_gate_identifies_pure_concatenation(monkeypatch):
    """§5.1d: 5s turn with 1s overlap -> 4.0s pure >= 2.0: identify runs
    exactly once on the sample-exact pure concatenation (no zeros)."""
    ramp = _ramp(5.0)
    transcripts, calls, _session = _run_turn_with_spy(
        monkeypatch,
        ring_pcm=ramp,
        ft=FinalizedTurn("speaker0", 0.0, 5.0, "TID-OK"),
        overlap_intervals=[(2.0, 3.0)],  # ratio 0.2 -> minority, pure 4.0s
    )
    assert len(calls) == 1
    assert calls[0].size == 64000
    assert np.array_equal(
        calls[0], np.concatenate([ramp[:32000], ramp[48000:80000]])
    )
    assert len(transcripts) == 1
    assert transcripts[0]["name"] == "alice" and transcripts[0]["score"] == 0.9


def test_below_threshold_result_still_updates_last_match(monkeypatch):
    """§5.1e: when identification RUNS and returns name=None (below match
    threshold), _last_match IS updated (2026-07-06 §4.1d, unchanged)."""
    transcripts, calls, session = _run_turn_with_spy(
        monkeypatch,
        ring_pcm=np.full(SR * 3, 0.05, dtype=np.float32),
        ft=FinalizedTurn("speaker0", 0.0, 2.5, "TID-THR"),
        overlap_intervals=[],  # no overlap: pure == total == 2.5s
        spy_result=(None, 0.5),
    )
    assert len(calls) == 1
    assert session._last_match["speaker0"] == (None, 0.5)
    assert len(transcripts) == 1
    assert transcripts[0]["name"] is None and transcripts[0]["score"] == 0.5
