"""The per-label identity echo must go stale instead of leaking forever.

docs/bugs/2026-08-20-diart-label-merge-identity-leak.md: diart hands two people
the SAME label whenever a silence longer than its 5s window separates them (it
decides "new speaker?" from ~0.5s of new evidence per step, where the score
distributions overlap). ``_last_match[label]`` self-corrects on any turn with
at least ``min_identify_seconds`` of pure speech -- but a new occupant who only
ever speaks in shorter bursts never rewrites it, so the previous occupant's
name echoes on the prediction stream indefinitely.

The bound is a COUNT of consecutive short turns that fail to re-confirm the
cached name, NOT a similarity threshold: measured on cross-segment pairs of the
same speaker, genuine and impostor turn-to-turn scores overlap at every turn
length (genuine min 0.569-0.587 vs impostor max 0.813-0.839), so a threshold
with usable detection costs 21-34% false drops. scripts/verify_label_continuity.py prints that table.

Drives ``_transcribe_and_emit`` via ``_handle_turn`` with fakes, in the style
of test_identity_overlap_purity.py.
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
from diart_server.config import settings  # noqa: E402
from diart_server.protocol import QueryParams  # noqa: E402
from diart_server.session import DiarizationSession  # noqa: E402
from diart_server.turn_aggregator import FinalizedTurn  # noqa: E402

SR = 16000
COHORT = np.eye(1, 8, dtype=np.float32).reshape(-1)  # any non-None cohort mean

# A turn long enough to run the embed+match chain but too short to rewrite
# _last_match -- the exact regime the leak lives in.
SHORT = (settings.min_candidate_seconds + settings.min_identify_seconds) / 2
LONG = settings.min_identify_seconds + 1.0
MISS = (None, 0.40)  # best match below the confirm threshold


class FakeRegistry:
    def __init__(self, seg, emb):
        self._seg, self._emb = seg, emb

    def get_segmentation(self, name):  # noqa: ARG002
        return self._seg

    def get_embedding(self, name):  # noqa: ARG002
        return self._emb


class _FakeStore:
    def vectors(self, model_name):  # noqa: ARG002
        return {}


def _spin_loop():
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()
    return loop, t


def _stop_loop(loop, t):
    loop.call_soon_threadsafe(loop.stop)
    t.join(timeout=2.0)
    loop.close()


@pytest.fixture(autouse=True)
def _fake_asr(monkeypatch):
    async def fake_transcribe(self, pcm, language=None):  # noqa: ANN001, ARG001
        return ("Chinese", "文本")

    monkeypatch.setattr(AsrClient, "transcribe", fake_transcribe)


def _drive(monkeypatch, turns, *, seed_match=("alice", 0.95), cohort=COHORT,
           store=_FakeStore, overlaps=()):
    """Run a SEQUENCE of turns on one label.

    ``turns`` is a list of ``(seconds, (name, score))`` -- the length of the
    turn and what _identify_slice returns for it. Returns
    ``(transcripts, session)``.
    """
    monkeypatch.setattr(
        "diart_server.session._load_cohort_mean", lambda n: cohort  # noqa: ARG005
    )
    messages, lock = [], threading.Lock()

    async def on_message(msg):
        with lock:
            messages.append(msg)

    loop, t = _spin_loop()
    seg = SegmentationModel(DummySegmentationModel)
    emb = EmbeddingModel(DummyEmbeddingModel)
    session = DiarizationSession(
        params=QueryParams(
            sample_rate=SR, pipeline="diarization",
            segmentation_model="dummy/seg", embedding_model="dummy/emb",
        ),
        registry=FakeRegistry(seg, emb), on_message=on_message, loop=loop,
        store=store() if store is not None else None,
        asr_client=AsrClient("http://x/v1", "m", "EMPTY", 5.0),
    )
    results = iter(list(r for _, r in turns))
    session._identify_slice = lambda pcm: next(results)  # noqa: ARG005
    try:
        session._asr_sem = asyncio.Semaphore(2)
        total = sum(sec for sec, _ in turns) + len(turns) + 2.0
        session.ring.append(np.full(int(SR * total), 0.05, dtype=np.float32))
        session._overlap_intervals = list(overlaps)
        if seed_match is not None:
            session._last_match["speaker0"] = seed_match
        cursor = 0.0
        for i, (seconds, _) in enumerate(turns):
            fut = asyncio.run_coroutine_threadsafe(
                session._handle_turn(
                    FinalizedTurn("speaker0", cursor, cursor + seconds, f"T{i}")
                ),
                loop,
            )
            fut.result(timeout=10.0)
            cursor += seconds + 1.0
    finally:
        _stop_loop(loop, t)
    with lock:
        return [m for m in messages if m.get("type") == "transcript"], session


def test_short_regime_is_the_one_the_leak_lives_in():
    """Guard the fixture: SHORT must run the chain yet never rewrite the echo."""
    assert settings.min_candidate_seconds <= SHORT < settings.min_identify_seconds
    assert settings.identity_echo_max_misses >= 1


# ---------------------------------------------------------------------------
# The bug: an echo that can never be rewritten must still expire.
# ---------------------------------------------------------------------------


def test_consecutive_short_misses_drop_the_stale_echo(monkeypatch):
    k = settings.identity_echo_max_misses
    trs, session = _drive(monkeypatch, [(SHORT, MISS)] * k)

    assert [t["contested"] for t in trs] == [False] * (k - 1) + [True]
    assert "speaker0" not in session._last_match
    assert "speaker0" not in session._echo_misses


def test_one_miss_short_of_the_limit_keeps_the_echo(monkeypatch):
    k = settings.identity_echo_max_misses
    trs, session = _drive(monkeypatch, [(SHORT, MISS)] * (k - 1))

    assert all(t["contested"] is False for t in trs)
    assert session._last_match["speaker0"] == ("alice", 0.95)
    assert session._echo_misses["speaker0"] == k - 1


def test_a_short_turn_that_reconfirms_resets_the_counter(monkeypatch):
    """The genuine speaker's protection: any short turn that DOES re-confirm
    the cached name restarts the count, so a run of unlucky short turns can
    never accumulate across a confirmation."""
    k = settings.identity_echo_max_misses
    hit = ("alice", 0.95)  # >= match_threshold
    trs, session = _drive(
        monkeypatch, [(SHORT, MISS)] * (k - 1) + [(SHORT, hit)] + [(SHORT, MISS)]
    )

    assert all(t["contested"] is False for t in trs)
    assert session._last_match["speaker0"] == ("alice", 0.95)
    assert session._echo_misses["speaker0"] == 1


def test_full_length_turn_rewrites_the_echo_and_resets_the_counter(monkeypatch):
    """A full-length turn is authoritative: it rewrites _last_match outright
    (existing contract) and clears any accumulated misses."""
    k = settings.identity_echo_max_misses
    trs, session = _drive(
        monkeypatch, [(SHORT, MISS)] * (k - 1) + [(LONG, ("bob", 0.93))]
    )

    assert all(t["contested"] is False for t in trs)
    assert session._last_match["speaker0"] == ("bob", 0.93)
    assert "speaker0" not in session._echo_misses


def test_short_turn_confirming_a_different_name_counts_but_does_not_short_circuit(
    monkeypatch,
):
    """A sub-min_identify_seconds confirmation is NOT trusted on its own.

    Sub-2s utterance embeddings false-accept (that is why min_identify_seconds
    exists; docs/bugs/2026-07-06-voiceprint-false-accept.md), so "bob at 0.9 on
    a 1.3s turn" is one unit of evidence against "alice", not proof of "bob".
    It is counted like any other miss and never stamps a name.
    """
    k = settings.identity_echo_max_misses
    trs, session = _drive(monkeypatch, [(SHORT, ("bob", 0.93))] * k)

    assert all(t["name"] is None for t in trs)
    assert trs[-1]["contested"] is True
    assert "speaker0" not in session._last_match


def test_label_without_a_cached_name_never_contests(monkeypatch):
    trs, session = _drive(
        monkeypatch, [(SHORT, MISS)] * (settings.identity_echo_max_misses + 2),
        seed_match=None,
    )

    assert all(t["contested"] is False for t in trs)
    assert session._echo_misses == {}


def test_cached_none_result_is_not_a_name_to_protect(monkeypatch):
    """``(None, score)`` means "this label has no verified identity" -- there is
    nothing stale to drop, so the counter must stay out of it."""
    trs, session = _drive(
        monkeypatch, [(SHORT, MISS)] * settings.identity_echo_max_misses,
        seed_match=(None, 0.5),
    )

    assert all(t["contested"] is False for t in trs)
    assert session._echo_misses == {}


def test_runs_where_identification_never_happened_do_not_count(monkeypatch):
    """Majority overlap skips the embed+match chain entirely
    (docs/bugs/2026-08-05-identity-continuity-overlap-purity.md), so those
    turns carry no evidence and must not age the echo."""
    k = settings.identity_echo_max_misses
    trs, session = _drive(
        monkeypatch, [(4.0, MISS)] * k,
        overlaps=[(0.0, 100.0)],  # every turn is fully overlapped
    )

    assert all(t["overlapped"] is True for t in trs)
    assert all(t["contested"] is False for t in trs)
    assert session._last_match["speaker0"] == ("alice", 0.95)
    assert session._echo_misses == {}


def test_cohortless_session_never_contests(monkeypatch):
    """Without a cohort mean identification is disabled outright
    (docs/bugs/2026-08-05-cohort-missing-threshold.md); nothing ages."""
    trs, session = _drive(
        monkeypatch, [(SHORT, MISS)] * (settings.identity_echo_max_misses + 1),
        cohort=None,
    )

    assert all(t["contested"] is False for t in trs)
    assert session._last_match["speaker0"] == ("alice", 0.95)


def test_dropping_the_echo_does_not_alter_this_turns_own_name(monkeypatch):
    """The echo is a display hint about the LABEL. A turn's own name comes from
    its own audio versus the roster and is untouched by the drop."""
    k = settings.identity_echo_max_misses
    trs, _ = _drive(monkeypatch, [(LONG, ("carol", 0.99))] * 1 + [(SHORT, MISS)] * k)

    assert trs[0]["name"] == "carol"
    assert trs[-1]["contested"] is True
    assert trs[-1]["name"] is None  # short turns never stamp a name
