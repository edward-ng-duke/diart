"""Unit tests for scripts/eval/offline_sweep.py.

The sweep script is the referee for every later threshold/model decision, so it
has to be correct before its numbers are quoted. These tests pin:

* the DIR@FAR / threshold estimators on hand-computable inputs,
* the ``thr(N) = a + b*log2(N/5)`` fit on an exactly-linear input,
* score-scale parity with ``diart_server.voiceprint_matcher`` (raw = plain
  cosine, centered = the ``(cos+1)/2`` remap) — a silent scale drift here would
  make every printed threshold wrong,
* the T4 stickiness / vote / contested-lock simulation,
* end-to-end behaviour on a synthetic 8-speaker cache, including the leakage
  gate and seeded reproducibility,
* exit code 2 (not a traceback) when the cache is missing or a filter is bogus.

The script is loaded by path: ``scripts/`` is not an importable package.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEP_PATH = REPO_ROOT / "scripts" / "eval" / "offline_sweep.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("offline_sweep", SWEEP_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # @dataclass resolves annotations through sys.modules; register before exec.
    sys.modules["offline_sweep"] = module
    spec.loader.exec_module(module)
    return module


sweep = _load_module()


# --------------------------------------------------------------------------- #
# synthetic cache
# --------------------------------------------------------------------------- #
def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def make_cache(
    directory: Path,
    model: str = "fake-model",
    n_spk: int = 8,
    n_utt: int = 6,
    n_cohort: int = 20,
    dim: int = 128,
    jitter: float = 0.05,
    common_weight: float = 2.0,
    seed: int = 1234,
) -> Path:
    """Write an ``emb_<model>.npz`` of ``n_spk`` well-separated fake speakers.

    The geometry mimics a real embedding space: every vector carries the same
    large non-discriminative ``common`` direction (that is what cohort
    centering exists to remove) plus a per-speaker direction plus jitter.
    Cohort clips carry the common direction and a random non-speaker direction,
    so their mean recovers ``common``. Speakers are separable enough that
    top-1 identification is unambiguous and no impostor comes near threshold.
    """
    rng = np.random.default_rng(seed)
    common = _unit(rng.normal(size=dim))
    speaker = np.stack([_unit(rng.normal(size=dim)) for _ in range(n_spk)])

    def vec(direction: np.ndarray, noise: float) -> np.ndarray:
        # noise is scaled by 1/sqrt(dim) so `jitter` reads as a relative size
        # regardless of dimensionality.
        return (common_weight * common + direction
                + noise * rng.normal(size=dim) / math.sqrt(dim)).astype(np.float32)

    arrays: dict[str, np.ndarray] = {}
    for i in range(n_spk):
        spk = f"s{i:02d}"
        arrays[f"cc_enroll/{spk}"] = vec(speaker[i], jitter)
        for utt in range(n_utt):
            for crop in ("0", "2", "4"):
                arrays[f"cc_test/{spk}/{utt}/{crop}"] = vec(speaker[i], jitter)
    for j in range(n_cohort):
        arrays[f"cohC/c{j:02d}/0"] = vec(_unit(rng.normal(size=dim)), 0.0)

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"emb_{model}.npz"
    np.savez(path, **arrays)
    return path


def _run(cache_dir: Path, *extra: str) -> int:
    return sweep.main(["--cache-dir", str(cache_dir), "--draws", "3",
                       "--n-values", "2,4", *extra])


# --------------------------------------------------------------------------- #
# threshold / DIR estimators
# --------------------------------------------------------------------------- #
def test_threshold_at_far_uses_exactly_the_far_budget():
    # 100 impostor scores 0.00 .. 0.99; a 5% budget allows 5 accepts, so the
    # threshold is the 5th largest score, 0.95, and FAR is exactly 5%.
    imp = np.arange(100) / 100.0
    thr = sweep.threshold_at_far(imp, 0.05)
    assert thr == pytest.approx(0.95)
    assert sweep.far_at_threshold(imp, thr) == pytest.approx(0.05)


def test_threshold_at_far_floors_the_budget():
    # 1% of 150 trials = 1.5 -> only 1 accept allowed -> the largest score.
    imp = np.arange(150) / 150.0
    thr = sweep.threshold_at_far(imp, 0.01)
    assert thr == pytest.approx(149 / 150.0)
    assert sweep.far_at_threshold(imp, thr) == pytest.approx(1 / 150.0)
    assert sweep.far_at_threshold(imp, thr) <= 0.01


def test_threshold_at_far_with_budget_below_one_trial_rejects_everything():
    imp = np.array([0.1, 0.5, 0.9])
    thr = sweep.threshold_at_far(imp, 0.01)  # 0.03 accepts allowed -> none
    assert thr > 0.9
    assert sweep.far_at_threshold(imp, thr) == 0.0


def test_threshold_at_far_rejects_empty_impostor_set():
    with pytest.raises(ValueError):
        sweep.threshold_at_far(np.array([]), 0.01)


def test_dir_counts_only_above_threshold_and_correct_top1():
    best = np.array([0.9, 0.9, 0.4, 0.95])
    arg = np.array([0, 1, 0, 0])       # probe 1 is above threshold but WRONG
    truth = np.array([0, 0, 0, 0])     # probe 2 is right but below threshold
    assert sweep.dir_at_threshold(best, arg, truth, 0.8) == pytest.approx(0.5)
    assert sweep.dir_at_threshold(best, arg, truth, 0.99) == 0.0
    assert sweep.dir_at_threshold(best, arg, truth, 0.0) == pytest.approx(0.75)


def test_fit_recovers_an_exact_log2_line():
    n_values = [5, 10, 20, 50]
    a_true, b_true = 0.7890, 0.0086
    thrs = [a_true + b_true * math.log2(n / 5) for n in n_values]
    a, b, r2 = sweep.fit_log2_slope(n_values, thrs)
    assert a == pytest.approx(a_true, abs=1e-9)
    assert b == pytest.approx(b_true, abs=1e-9)
    assert r2 == pytest.approx(1.0, abs=1e-9)


def test_fit_is_consistent_with_lo_for_n():
    n_values = [5, 10, 20, 50]
    thrs = [sweep.lo_for_n(n, 0.82, 0.0086) for n in n_values]
    a, b, _ = sweep.fit_log2_slope(n_values, thrs)
    assert (a, b) == (pytest.approx(0.82), pytest.approx(0.0086))
    assert sweep.lo_for_n(5, 0.82, 0.0086) == pytest.approx(0.82)
    assert sweep.lo_for_n(10, 0.82, 0.0086) == pytest.approx(0.8286)


def test_fit_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        sweep.fit_log2_slope([5, 10], [0.8])


# --------------------------------------------------------------------------- #
# score-scale parity with the production matcher
# --------------------------------------------------------------------------- #
def test_score_matrix_matches_voiceprint_matcher_on_both_scales(tmp_path):
    from diart_server.voiceprint_matcher import match

    make_cache(tmp_path, model="parity", n_spk=4, n_utt=3)
    bundle = sweep.load_bundle(tmp_path / "emb_parity.npz", "parity", ("2",))
    refs = {name: bundle.enroll[i] for i, name in enumerate(bundle.enroll_spk)}

    raw = sweep.score_matrix(bundle, "2", "raw")
    centered = sweep.score_matrix(bundle, "2", "centered")
    query = bundle.tests["2"][0]

    _, raw_best = match(query, refs, threshold=-1.0)
    assert raw_best == pytest.approx(float(raw[0].max()), abs=1e-5)

    _, cen_best = match(query, refs, threshold=-1.0, mean_embedding=bundle.cohort_mean)
    assert cen_best == pytest.approx(float(centered[0].max()), abs=1e-5)

    # Raw mode never remaps: a flipped query scores NEGATIVE in production and
    # score_matrix agrees. Centered stays inside the remapped [0, 1] band.
    _, flipped = match(-query, refs, threshold=-2.0)
    expected = float((sweep.unit(-query)[None, :] @ sweep.unit(bundle.enroll).T).max())
    assert flipped == pytest.approx(expected, abs=1e-5)
    assert flipped < 0.0
    assert 0.0 <= centered.min() and centered.max() <= 1.0
    assert not np.allclose(raw, centered)


# --------------------------------------------------------------------------- #
# stickiness simulation
# --------------------------------------------------------------------------- #
def test_hi_fire_names_the_cluster_and_rewrites_backwards():
    best = np.array([0.70, 0.70, 0.90, 0.70])
    arg = np.array([3, 3, 3, 3])
    shown, ever = sweep.run_cluster(best, arg, hi=0.86, lo=0.82)
    assert ever is True
    assert shown.tolist() == [3, 3, 3, 3]


def test_two_of_five_vote_fires_without_any_hi_turn():
    best = np.array([0.83, 0.10, 0.10, 0.83, 0.10])
    arg = np.array([7, 1, 2, 7, 4])
    shown, ever = sweep.run_cluster(best, arg, hi=0.86, lo=0.82)
    assert ever is True
    # fires on turn 3 (two turns >= lo agreeing on 7), rewritten backwards
    assert shown.tolist() == [7, 7, 7, 7, 7]


def test_single_above_lo_turn_never_fires():
    best = np.array([0.83, 0.10, 0.10, 0.10, 0.10])
    arg = np.array([7, 1, 2, 3, 4])
    shown, ever = sweep.run_cluster(best, arg, hi=0.86, lo=0.82)
    assert ever is False
    assert shown.tolist() == [-1] * 5


def test_disagreeing_votes_inside_the_window_do_not_fire():
    best = np.array([0.83, 0.83, 0.83])
    arg = np.array([7, 8, 9])
    shown, ever = sweep.run_cluster(best, arg, hi=0.86, lo=0.82)
    assert ever is False
    assert shown.tolist() == [-1, -1, -1]


def test_contested_lock_blanks_the_whole_cluster_but_ever_fired_stays_true():
    best = np.array([0.90, 0.90, 0.90])
    arg = np.array([2, 2, 5])
    shown, ever = sweep.run_cluster(best, arg, hi=0.86, lo=0.82)
    assert ever is True          # a name WAS on screen (V3 semantics)
    assert shown.tolist() == [-1, -1, -1]   # ... and was retracted


def test_vote_fires_mirrors_the_run_cluster_vote_branch():
    best = np.array([[0.83, 0.83, 0.10], [0.83, 0.10, 0.10], [0.83, 0.83, 0.10]])
    arg = np.array([[7, 7, 1], [7, 1, 2], [7, 8, 1]])
    fired, name = sweep.vote_fires(best, arg, lo=0.82)
    assert fired.tolist() == [True, False, False]
    assert name.tolist() == [7, -1, -1]


def test_modal_agreement_counts_the_dominant_top1():
    arg = np.array([[1, 1, 1, 1], [1, 1, 2, 3], [1, 2, 3, 4]])
    assert sweep.modal_agreement(arg).tolist() == [1.0, 0.5, 0.25]


# --------------------------------------------------------------------------- #
# end to end on a synthetic cache
# --------------------------------------------------------------------------- #
def test_end_to_end_on_separable_cache_is_perfect_and_leak_free(tmp_path, capsys):
    make_cache(tmp_path)
    json_path = tmp_path / "out.json"
    assert _run(tmp_path, "--json", str(json_path)) == 0

    out = capsys.readouterr().out
    for title in ("Table 1", "Table 2", "Table 3", "Table 4"):
        assert f"== {title}" in out
    assert "Confidence bounds" in out
    assert "CN-Celeb gives ORDERING and SLOPE only" in out
    assert "N <= 8" in out

    result = json.loads(json_path.read_text(encoding="utf-8"))
    assert result["meta"]["models"] == ["fake-model"]
    assert result["meta"]["n_values"] == [2, 4]
    assert result["meta"]["seed"] == sweep.DEFAULT_SEED

    # Well-separated speakers: identification is perfect on every condition.
    for key, cell in result["dir_at_far"].items():
        assert cell["dir_mean"] == pytest.approx(1.0), key
        assert cell["dir_std"] == pytest.approx(0.0), key

    # ... and nobody outside the gallery ever gets a name.
    for key, cell in result["leakage"].items():
        assert cell["leak_ever"] == 0.0, key
        assert cell["leak_final"] == 0.0, key
        assert cell["per_turn"] == 0.0, key
        assert cell["cover"] == pytest.approx(1.0), key
        assert cell["wrong"] == 0.0, key
    assert "gate" in out and "PASS" in out
    assert "GATE FAIL" not in out

    # k accumulation: identical correct top-1 every turn.
    for key, cell in result["k_accumulation"].items():
        assert cell["top1"] == pytest.approx(1.0), key
        assert cell["per_k"]["5"]["consistency"] == pytest.approx(1.0), key
        assert cell["per_k"]["5"]["vote_ok"] == pytest.approx(1.0), key
        assert cell["per_k"]["5"]["vote_wrong"] == 0.0, key
        assert cell["per_k"]["5"]["imp_vote"] == 0.0, key

    assert set(result["limitations"]) == set(sweep.LIMITATIONS)


def test_leakage_gate_fails_loudly_when_the_threshold_is_dropped(tmp_path, capsys):
    make_cache(tmp_path)
    json_path = tmp_path / "out.json"
    # lo = 0.0 accepts literally everyone -> every impostor cluster gets named.
    assert _run(tmp_path, "--lo-base", "0.0", "--json", str(json_path)) == 0
    out = capsys.readouterr().out
    assert "GATE FAIL" in out
    result = json.loads(json_path.read_text(encoding="utf-8"))
    for key, cell in result["leakage"].items():
        assert cell["leak_ever"] == pytest.approx(1.0), key


def test_results_are_reproducible_and_independent_of_filters(tmp_path):
    make_cache(tmp_path)
    both = tmp_path / "both.json"
    one = tmp_path / "one.json"
    again = tmp_path / "again.json"

    assert sweep.main(["--cache-dir", str(tmp_path), "--draws", "3",
                       "--n-values", "2,4", "--json", str(both)]) == 0
    assert sweep.main(["--cache-dir", str(tmp_path), "--draws", "3",
                       "--n-values", "4", "--crops", "2", "--json", str(one)]) == 0
    assert sweep.main(["--cache-dir", str(tmp_path), "--draws", "3",
                       "--n-values", "2,4", "--json", str(again)]) == 0

    a = json.loads(both.read_text(encoding="utf-8"))
    b = json.loads(one.read_text(encoding="utf-8"))
    c = json.loads(again.read_text(encoding="utf-8"))

    assert a["dir_at_far"] == c["dir_at_far"]          # same seed -> same numbers
    assert a["leakage"] == c["leakage"]
    key = "fake-model|centered|2|4"
    assert a["dir_at_far"][key] == b["dir_at_far"][key]  # N/crop filters don't move it
    assert a["leakage"]["fake-model|n-aware|4"] == b["leakage"]["fake-model|n-aware|4"]

    # A different seed must actually redraw the galleries.
    other = tmp_path / "other.json"
    assert sweep.main(["--cache-dir", str(tmp_path), "--draws", "3",
                       "--n-values", "2,4", "--seed", "1", "--json", str(other)]) == 0
    d = json.loads(other.read_text(encoding="utf-8"))
    assert d["dir_at_far"] != a["dir_at_far"]


def test_env_var_supplies_the_cache_dir(tmp_path, monkeypatch):
    make_cache(tmp_path)
    monkeypatch.setenv(sweep.ENV_CACHE, str(tmp_path))
    assert sweep.main(["--draws", "2", "--n-values", "2"]) == 0


# --------------------------------------------------------------------------- #
# fail-soft: exit code 2, never a traceback
# --------------------------------------------------------------------------- #
def test_unspecified_cache_exits_2(monkeypatch, capsys):
    monkeypatch.delenv(sweep.ENV_CACHE, raising=False)
    assert sweep.main([]) == 2
    err = capsys.readouterr().err
    assert "--cache-dir" in err
    assert sweep.ENV_CACHE in err


def test_missing_cache_directory_exits_2(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv(sweep.ENV_CACHE, raising=False)
    assert sweep.main(["--cache-dir", str(tmp_path / "nope")]) == 2
    assert "--cache-dir" in capsys.readouterr().err


def test_empty_cache_directory_exits_2(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv(sweep.ENV_CACHE, raising=False)
    assert sweep.main(["--cache-dir", str(tmp_path)]) == 2
    assert "emb_*.npz" in capsys.readouterr().err


def test_unknown_model_filter_exits_2(tmp_path, capsys):
    make_cache(tmp_path)
    assert _run(tmp_path, "--models", "does-not-exist") == 2
    assert "unknown model" in capsys.readouterr().err


def test_unknown_crop_filter_exits_2(tmp_path, capsys):
    make_cache(tmp_path)
    assert _run(tmp_path, "--crops", "8") == 2
    assert "unknown crop" in capsys.readouterr().err


def test_gallery_larger_than_the_speaker_pool_exits_2(tmp_path, capsys):
    make_cache(tmp_path, n_spk=8)
    assert sweep.main(["--cache-dir", str(tmp_path), "--draws", "2",
                       "--n-values", "8"]) == 2
    assert "out-of-gallery" in capsys.readouterr().err


def test_cache_without_cohort_keys_exits_2(tmp_path, capsys):
    np.savez(tmp_path / "emb_broken.npz", **{
        "cc_enroll/s00": np.ones(8, dtype=np.float32),
        "cc_test/s00/0/2": np.ones(8, dtype=np.float32),
    })
    assert _run(tmp_path) == 2
    assert "cohC" in capsys.readouterr().err
