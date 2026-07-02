"""BUG2: DiarizationSession._apply_overlap excises minority / flags majority.

Pure-function test: constructs a 4.0s slice and injects overlap intervals
directly into the session's tracked list, then calls _apply_overlap.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

TESTS_DIR = Path(__file__).resolve().parent.parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from conftest import DummyEmbeddingModel, DummySegmentationModel  # noqa: E402
from diart.models import EmbeddingModel, SegmentationModel  # noqa: E402

from diart_server.protocol import QueryParams  # noqa: E402
from diart_server.session import DiarizationSession  # noqa: E402
from diart_server.turn_aggregator import FinalizedTurn  # noqa: E402


class _FakeRegistry:
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

    async def _noop(_msg):  # pragma: no cover - never called here
        return None

    import asyncio
    loop = asyncio.new_event_loop()
    try:
        return DiarizationSession(
            params=params, registry=_FakeRegistry(seg, emb),
            on_message=_noop, loop=loop, store=None, asr_client=None,
        )
    finally:
        loop.close()


def test_minority_overlap_is_excised():
    session = _make_session()
    ft = FinalizedTurn("speaker0", 0.0, 4.0, "T")
    pcm = np.full(16000 * 4, 0.05, dtype=np.float32)
    session._overlap_intervals = [(1.0, 1.5)]  # 0.5 / 4.0 = 12.5% minority

    out, overlapped, ratio = session._apply_overlap(pcm, ft)

    assert overlapped is False
    assert abs(ratio - 0.125) < 1e-6
    # Samples in [1.0, 1.5] zeroed; outside unchanged (0.05).
    i0, i1 = 16000 * 1, int(16000 * 1.5)
    assert np.all(out[i0:i1] == 0.0)
    assert np.all(out[:i0] == 0.05)
    assert np.all(out[i1:] == 0.05)


def test_majority_overlap_left_intact_and_flagged():
    session = _make_session()
    ft = FinalizedTurn("speaker0", 0.0, 4.0, "T")
    pcm = np.full(16000 * 4, 0.05, dtype=np.float32)
    original = pcm.copy()
    session._overlap_intervals = [(0.0, 3.0)]  # 3.0 / 4.0 = 75% majority

    out, overlapped, ratio = session._apply_overlap(pcm, ft)

    assert overlapped is True
    assert abs(ratio - 0.75) < 1e-6
    assert np.array_equal(out, original)  # no excision


def test_zero_overlap_unchanged():
    session = _make_session()
    ft = FinalizedTurn("speaker0", 0.0, 4.0, "T")
    pcm = np.full(16000 * 4, 0.05, dtype=np.float32)
    original = pcm.copy()
    session._overlap_intervals = []

    out, overlapped, ratio = session._apply_overlap(pcm, ft)

    assert overlapped is False
    assert ratio == 0.0
    assert np.array_equal(out, original)
