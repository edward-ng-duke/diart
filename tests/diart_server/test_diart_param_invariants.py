"""Pin the diart hyper-parameter facts that QueryParams documents.

docs/bugs/2026-08-20-diart-label-merge-identity-leak.md §2: with
pyannote/segmentation-3.0 (a POWERSET model) diart's ``PowersetAdapter`` calls
``Powerset.to_multilabel()`` with ``soft=False``, so the segmentation this
pipeline sees is strictly BINARY. Three of diart's tuned knobs degenerate as a
result. These are not bugs we are fixing -- they are load-bearing facts behind
"do not tune tau_active / rho_update / delta_new", so they get pinned: if a
future diart or pyannote release makes the segmentation soft again, these tests
fail and the QueryParams commentary must be revisited.

The two model-backed tests skip when models/ is not populated.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from diart.blocks.aggregation import DelayedAggregation
from diart.functional import overlapped_speech_penalty

REPO = Path(__file__).resolve().parents[2]
SEG_CKPT = REPO / "models" / "pyannote" / "segmentation-3.0" / "pytorch_model.bin"
needs_models = pytest.mark.skipif(
    not SEG_CKPT.exists(), reason=f"segmentation-3.0 checkpoint absent ({SEG_CKPT})"
)


@needs_models
def test_powerset_segmentation_reaches_diart_as_binary():
    """The only two values diart's segmentation stage ever emits are 0.0/1.0."""
    from diart.models import SegmentationModel

    model = SegmentationModel.from_pyannote(str(SEG_CKPT))
    with torch.no_grad():
        out = model(torch.randn(2, 1, 5 * 16000))  # (batch, channel, samples)
    values = np.unique(out.cpu().numpy())
    assert set(values.tolist()) <= {0.0, 1.0}, (
        f"segmentation is no longer binary ({values[:8]}); tau_active and the "
        "gamma term are live again -- revisit QueryParams' commentary"
    )


@needs_models
def test_overlap_aware_weights_collapse_to_three_values():
    """gamma=3's low-confidence suppression is inert on a binary segmentation.

    ``weights = seg**gamma * softmax(beta*seg)**gamma`` over {0,1} can only be
    "inactive", "overlapped" or "clean" -- a hard mask with a fixed penalty,
    not the paper's confidence weighting.
    """
    seg = torch.tensor([[[1.0, 0.0, 0.0],    # clean single speaker
                         [1.0, 1.0, 0.0],    # two speakers overlapping
                         [0.0, 0.0, 0.0]]])  # silence
    w = overlapped_speech_penalty(seg, gamma=3, beta=10).numpy()
    distinct = np.unique(np.round(w, 6))
    assert distinct.size == 3, f"expected 3 weight levels, got {distinct}"
    clean, overlapped = float(w[0, 0, 0]), float(w[0, 1, 0])
    assert clean == pytest.approx(0.99973, abs=1e-4)
    assert overlapped == pytest.approx(0.125, abs=1e-4)
    # The overlap penalty is a fixed ~8x, independent of model confidence.
    assert clean / overlapped == pytest.approx(8.0, rel=0.01)


def test_tau_active_is_inert_on_binary_segmentation():
    """``max(seg) >= tau_active`` picks the same speakers for ANY tau in (0,1).

    This is the comparison OnlineSpeakerClustering.identify() makes, so the
    tuned 0.507 selects exactly what 0.01 or 0.99 would -- a single 17 ms
    frame is enough to mark a speaker active.
    """
    rng = np.random.default_rng(20260820)
    seg = (rng.random((293, 3)) > 0.7).astype(np.float32)  # binary, like the real one
    peaks = np.max(seg, axis=0)
    reference = np.where(peaks >= 0.507)[0]
    for tau in (0.01, 0.25, 0.507, 0.75, 0.99):
        assert np.array_equal(np.where(peaks >= tau)[0], reference)


def test_min_latency_disables_aggregation_so_binarize_is_a_noop():
    """latency == step aggregates ONE window, so Binarize sees raw 0/1.

    Only latency > step averages several windows and turns tau_active back
    into a real vote threshold -- at the deployed latency=0.5/step=0.5 it
    cannot be one.
    """
    assert DelayedAggregation(step=0.5, latency=0.5).num_overlapping_windows == 1
    assert DelayedAggregation(step=0.5, latency=2.0).num_overlapping_windows == 4
    assert DelayedAggregation(step=0.5, latency=5.0).num_overlapping_windows == 10


def test_queryparams_defaults_are_the_documented_ones():
    """Guard the three knobs against a drive-by "tuning" commit.

    Changing any of them requires re-running scripts/eval/diart_cluster_sweep.py
    (lowering delta_new was measured: 0.80 -> 4 labels and 0.75 -> 5 labels on
    data/2news.mp3, which has 2 speakers).
    """
    from diart_server.protocol import QueryParams

    p = QueryParams()
    assert (p.tau_active, p.rho_update, p.delta_new) == (0.507, 0.006, 1.057)
    assert (p.duration, p.step, p.latency) == (5.0, 0.5, 0.5)
