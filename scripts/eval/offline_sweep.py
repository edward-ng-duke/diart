#!/usr/bin/env python
"""Offline model / scoring / threshold sweep over CACHED speaker embeddings.

Spec: ``docs/superpowers/plans/2026-08-06-voiceprint-selection-plan.md`` T5 + §6.2.

**Zero forward passes.** Everything is computed with numpy from a directory of
``emb_<model>.npz`` caches that already hold every CN-Celeb1 eval embedding
(196 enrollment vectors, 1563 test utterances x {2s, 4s, full}, 764 cohort
clips) for each candidate model. A full sweep is a few seconds, so changing a
threshold or a model has a re-runnable gate number instead of an opinion.

Expected keys inside each ``emb_<model>.npz``::

    cc_enroll/<spk>              enrollment vector for gallery speaker <spk>
    cc_test/<spk>/<utt>/<crop>   test vector, crop in {0 = full, 2 = 2s, 4 = 4s}
    cohC/<spk>/<idx>             cohort (impostor-mean) clip

Four tables (plan §6.1 / §6.2):

1. **model x scoring** - DIR@FAR=1%, stratified by crop and gallery size N.
2. **threshold x N**    - thr@FAR=1% per N plus the least-squares fit
   ``thr(N) = a + b*log2(N/5)``; ``b`` is the N-aware slope T3 ships.
3. **k accumulation**   - top-1 consistency across k consecutive turns of the
   same speaker, and the hit rate of the 2-of-5 vote T4 ships.
4. **leakage**          - fraction of *unregistered* speakers' clusters that
   ever get an enrolled name stamped on them. Plan §2 gate: **<= 1%**.

Score scales match ``diart_server.voiceprint_matcher`` exactly:

* ``raw``      -> plain cosine of unit vectors, in ``[-1, 1]``, NO remap;
* ``centered`` -> ``unit(unit(v) - cohort_mean)`` on both sides, then the
  production ``(cos + 1) / 2`` remap to ``[0, 1]``.

The two scales are therefore NOT comparable to each other; the shipped
operating point (``lo = 0.82 + 0.0086*log2(N/5)``) lives on the centered scale.

Usage (from repo root)::

    DIART_EVAL_CACHE=/path/to/asnorm_cache python scripts/eval/offline_sweep.py
    python scripts/eval/offline_sweep.py --cache-dir /path/to/asnorm_cache \
        --models eres2netv2-zh,campplus-zh --crops 2 --n-values 5,20 \
        --json /tmp/sweep.json

Exit codes: 0 tables printed, 2 cache missing / unusable / bad filter.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

EXIT_OK = 0
EXIT_NO_CACHE = 2

ENV_CACHE = "DIART_EVAL_CACHE"

DEFAULT_SEED = 20260806
DEFAULT_DRAWS = 30
DEFAULT_FAR = 0.01
DEFAULT_N_VALUES = (5, 10, 20, 50)

# Operating point shipped by T3/T4 (plan §6.1 Q4), on the CENTERED scale.
LO_BASE = 0.82
LO_SLOPE = 0.0086
HI_DELTA = 0.04
N_REF = 5

# Cluster simulation shape (mirrors scratchpad/adv_cluster_check.py).
CLUSTER_TURNS = 12
# 50 x 30 draws = 1500 impostor clusters per row, i.e. 0.07% resolution on a 1%
# gate (the prototype's 25 only resolved 0.13%).
IMPOSTOR_CLUSTERS_PER_DRAW = 50
MIN_TURNS_PER_SPEAKER = 3
VOTE_K = 2
VOTE_WINDOW = 5
K_VALUES = (2, 3, 5)

LEAK_GATE = 0.01

SCORINGS = ("raw", "centered")
# CLI crop name -> the last key field inside the npz.
CROP_KEY = {"2": "2", "4": "4", "full": "0"}
CROP_ORDER = ("2", "4", "full")
CROP_LABEL = {"2": "2s", "4": "4s", "full": "full"}

LIMITATIONS = (
    "CN-Celeb gives ORDERING and SLOPE only, never an absolute operating point: "
    "it is not Edward's mic/room/browser. The threshold intercept must be "
    "calibrated by the 5-minute live test (plan §6.4.1).",
    "Absolute accuracy is optimistic: 3D-Speaker models were trained on ~200k "
    "Chinese speakers and CN-Celeb is very likely inside that training set. "
    "eres2netv2-vs-campplus stays a fair comparison (same vendor), but their "
    "margin over the wespeaker family may be inflated (plan §6.4.2).",
    "The cohort used for centering (cohC/*) is NOT disjoint from the eval "
    "gallery (194 of its 196 speakers are also enrolled speakers). A 764-clip "
    "MEAN is only mildly sensitive to that, but centered numbers are an upper "
    "bound, and this cache cannot support AS-Norm (plan §6.1 Q2 / R3b).",
    "No conclusion below FAR < 1% is supported. Do NOT cite the old probe-set "
    "'FA = 0/1668' -- rule of three only bounds it at FAR <= 0.18%, while the "
    "same threshold leaks 7.0% at N=20 (plan §6.4.3).",
    "The dispersion reported is the std across GALLERY DRAWS, not a "
    "speaker-block CI (deliberately not implemented, plan §6.2). Say 'the "
    "6-8pp gap is consistent', never 'better by 6.3 +- X pp'. Conclusions are "
    "claimed only for N <= 8 (plan §6.4.4).",
    "Threshold columns live on the PRODUCTION score scale of each scoring mode "
    "-- centered = (cos+1)/2 remap, raw = plain cosine. Do not compare a raw "
    "threshold against a centered one.",
)


# --------------------------------------------------------------------------- #
# pure numeric core (unit-tested)
# --------------------------------------------------------------------------- #
def unit(v: np.ndarray) -> np.ndarray:
    """L2-normalize along the last axis, leaving all-zero rows untouched."""
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.where(n > 0, v / np.maximum(n, 1e-12), v)


def threshold_at_far(impostor_best: np.ndarray, far: float) -> float:
    """Lowest observed impostor score whose ``>=`` accept rule keeps FAR <= far.

    With ``n`` impostor trials, ``floor(far * n)`` of them are allowed above the
    threshold, so the threshold is the ``floor(far * n)``-th largest impostor
    score. When the budget is below one trial the threshold sits just above the
    highest impostor score (FAR = 0).
    """
    s = np.sort(np.asarray(impostor_best, dtype=np.float64))[::-1]
    if s.size == 0:
        raise ValueError("threshold_at_far: no impostor trials")
    allowed = int(math.floor(far * s.size))
    if allowed < 1:
        return math.nextafter(float(s[0]), math.inf)
    return float(s[allowed - 1])


def far_at_threshold(impostor_best: np.ndarray, thr: float) -> float:
    """Observed false-accept rate of ``score >= thr`` on the impostor trials."""
    b = np.asarray(impostor_best)
    return float(np.mean(b >= thr)) if b.size else float("nan")


def dir_at_threshold(
    best: np.ndarray, arg: np.ndarray, truth: np.ndarray, thr: float
) -> float:
    """Detection & identification rate: above threshold AND top-1 is correct."""
    best = np.asarray(best)
    if best.size == 0:
        return float("nan")
    return float(np.mean((best >= thr) & (np.asarray(arg) == np.asarray(truth))))


def fit_log2_slope(
    n_values, thresholds, n_ref: int = N_REF
) -> tuple[float, float, float]:
    """Least-squares fit of ``thr(N) = a + b*log2(N/n_ref)``. Returns (a, b, r2)."""
    x = np.log2(np.asarray(n_values, dtype=np.float64) / float(n_ref))
    y = np.asarray(thresholds, dtype=np.float64)
    if x.size != y.size:
        raise ValueError("fit_log2_slope: n_values and thresholds differ in length")
    if x.size < 2:
        return float(y[0]), 0.0, float("nan")
    design = np.stack([np.ones_like(x), x], axis=1)
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    resid = y - (a + b * x)
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return a, b, r2


def lo_for_n(n: int, lo_base: float, lo_slope: float, n_ref: int = N_REF) -> float:
    """N-aware lower threshold ``lo(N) = lo_base + lo_slope*log2(N/n_ref)``."""
    return float(lo_base + lo_slope * math.log2(n / float(n_ref)))


def run_cluster(
    best: np.ndarray,
    arg: np.ndarray,
    hi: float,
    lo: float,
    vote_k: int = VOTE_K,
    vote_win: int = VOTE_WINDOW,
) -> tuple[np.ndarray, bool]:
    """Simulate T4's cluster stickiness over one diart label's turns.

    Fire immediately on ``score >= hi``; otherwise fire when ``vote_k`` of the
    last ``vote_win`` turns clear ``lo`` and all agree on one name. The first
    fire is rewritten backwards over the earlier turns; a later fire on a
    DIFFERENT name trips the contested lock and blanks the whole cluster.

    Returns ``(shown, ever_fired)`` where ``shown[i]`` is the gallery index
    displayed on turn ``i`` (``-1`` = neutral) at the END of the cluster, and
    ``ever_fired`` records whether a name was ever on screen (V3 semantics:
    a name that was shown and later retracted still appeared).
    """
    name = -1
    contested = False
    ever = False
    recent: list[tuple[float, int]] = []
    shown = np.full(len(best), -1, dtype=int)
    for i, (s, a) in enumerate(zip(best, arg)):
        fire = -1
        if s >= hi:
            fire = int(a)
        else:
            recent.append((float(s), int(a)))
            recent = recent[-vote_win:]
            hits = [n for (sc, n) in recent if sc >= lo]
            if len(hits) >= vote_k and len(set(hits)) == 1:
                fire = hits[0]
        if contested:
            continue
        if name < 0 and fire >= 0:
            name = fire
            ever = True
            shown[: i + 1] = name
        elif name >= 0 and fire >= 0 and fire != name:
            contested = True
            name = -1
            shown[:] = -1
        elif name >= 0:
            shown[i] = name
    return shown, ever


def vote_fires(
    best_k: np.ndarray, arg_k: np.ndarray, lo: float, vote_k: int = VOTE_K
) -> tuple[np.ndarray, np.ndarray]:
    """Single-window ``vote_k``-of-k vote over ``(clusters, k)`` score/name grids.

    Mirrors :func:`run_cluster`'s vote branch: at least ``vote_k`` turns clear
    ``lo`` and every turn that cleared ``lo`` names the same speaker.
    Returns ``(fired, name)`` with ``name = -1`` where it did not fire.
    """
    fired = np.zeros(best_k.shape[0], dtype=bool)
    name = np.full(best_k.shape[0], -1, dtype=int)
    above = best_k >= lo
    for i in range(best_k.shape[0]):
        hits = arg_k[i][above[i]]
        if hits.size >= vote_k and bool(np.all(hits == hits[0])):
            fired[i] = True
            name[i] = int(hits[0])
    return fired, name


def modal_agreement(arg_k: np.ndarray) -> np.ndarray:
    """Per cluster, the fraction of its k turns whose top-1 equals the modal top-1."""
    if arg_k.shape[1] == 0:
        return np.zeros(arg_k.shape[0], dtype=np.float64)
    same = arg_k[:, :, None] == arg_k[:, None, :]
    return same.sum(axis=2).max(axis=1) / float(arg_k.shape[1])


# --------------------------------------------------------------------------- #
# cache loading
# --------------------------------------------------------------------------- #
@dataclass
class Bundle:
    model: str
    path: Path
    dim: int
    enroll_spk: list[str]
    enroll: np.ndarray  # (S, D) unit
    cohort_mean: np.ndarray  # (D,) unit
    tests: dict[str, np.ndarray]  # crop -> (M, D) unit
    truth: np.ndarray  # (M,) enrollment index of each test row
    test_keys: list[str]


class CacheError(Exception):
    """Raised for anything that makes a cache directory unusable (-> exit 2)."""


def load_bundle(path: Path, model: str, crops: tuple[str, ...]) -> Bundle:
    data = np.load(path)
    files = list(data.files)

    spk = sorted({k.split("/")[1] for k in files if k.startswith("cc_enroll/")})
    if not spk:
        raise CacheError(f"{path}: no 'cc_enroll/<spk>' keys")
    enroll = unit(np.stack([data[f"cc_enroll/{s}"] for s in spk]))

    coh_keys = sorted(k for k in files if k.startswith("cohC/"))
    if not coh_keys:
        raise CacheError(f"{path}: no 'cohC/<spk>/<idx>' cohort keys")
    cohort_mean = unit(unit(np.stack([data[k] for k in coh_keys])).mean(axis=0))

    index = {s: i for i, s in enumerate(spk)}
    tests: dict[str, np.ndarray] = {}
    ref_keys: list[str] | None = None
    for crop in crops:
        field = CROP_KEY[crop]
        keys = sorted(
            k
            for k in files
            if k.startswith("cc_test/")
            and len(k.split("/")) == 4
            and k.split("/")[3] == field
        )
        if not keys:
            raise CacheError(f"{path}: no 'cc_test/*/*/{field}' keys (crop {crop})")
        stems = [k.rsplit("/", 1)[0] for k in keys]
        if ref_keys is None:
            ref_keys = stems
        elif stems != ref_keys:
            raise CacheError(f"{path}: crop {crop} covers a different utterance set")
        tests[crop] = unit(np.stack([data[k] for k in keys]))

    assert ref_keys is not None
    unknown = {k.split("/")[1] for k in ref_keys} - set(index)
    if unknown:
        raise CacheError(f"{path}: test speakers without enrollment: {sorted(unknown)}")
    truth = np.array([index[k.split("/")[1]] for k in ref_keys], dtype=int)

    return Bundle(
        model=model,
        path=path,
        dim=int(enroll.shape[1]),
        enroll_spk=spk,
        enroll=enroll,
        cohort_mean=cohort_mean,
        tests=tests,
        truth=truth,
        test_keys=ref_keys,
    )


def score_matrix(bundle: Bundle, crop: str, scoring: str) -> np.ndarray:
    """(n_test, n_enroll) scores on the production scale of ``scoring``."""
    tests = bundle.tests[crop]
    enroll = bundle.enroll
    if scoring == "raw":
        # voiceprint_matcher raw mode: plain cosine, NO (c+1)/2 remap.
        return tests @ enroll.T
    mean = bundle.cohort_mean
    return (unit(tests - mean[None, :]) @ unit(enroll - mean[None, :]).T + 1.0) / 2.0


# --------------------------------------------------------------------------- #
# gallery / cluster draw plan (seeded, filter-invariant)
# --------------------------------------------------------------------------- #
@dataclass
class DrawPlan:
    n: int
    galleries: list[np.ndarray]
    in_gallery: list[np.ndarray]
    genuine: list[list[tuple[int, np.ndarray]]]
    impostor: list[list[tuple[int, np.ndarray]]]


def build_plan(
    n: int, draws: int, n_spk: int, by_spk: dict[int, np.ndarray], truth: np.ndarray, seed: int
) -> DrawPlan:
    """Draw ``draws`` galleries of size ``n`` plus their genuine/impostor clusters.

    Seeded from ``[seed, n]`` only, so a plan does not depend on which models,
    crops or other N values were requested: filtering the sweep never moves a
    number. The galleries are drawn first, in one pass, so the DIR/threshold
    tables also stay put when the cluster-simulation constants change.
    """
    rng = np.random.default_rng([seed, n])
    all_spk = np.arange(n_spk)
    galleries = [np.sort(rng.choice(n_spk, size=n, replace=False)) for _ in range(draws)]
    in_gallery = [np.isin(truth, gal) for gal in galleries]
    genuine, impostor = [], []
    for gal in galleries:
        genuine.append([(int(g), _pick_turns(rng, by_spk[int(g)])) for g in gal
                        if by_spk[int(g)].size >= MIN_TURNS_PER_SPEAKER])
        pool = np.setdiff1d(all_spk, gal)
        size = min(IMPOSTOR_CLUSTERS_PER_DRAW, pool.size)
        out = rng.choice(pool, size=size, replace=False) if size else np.empty(0, int)
        impostor.append([(int(o), _pick_turns(rng, by_spk[int(o)])) for o in out
                         if by_spk[int(o)].size >= MIN_TURNS_PER_SPEAKER])
    return DrawPlan(n, galleries, in_gallery, genuine, impostor)


def _pick_turns(rng: np.random.Generator, ids: np.ndarray) -> np.ndarray:
    return rng.choice(ids, size=CLUSTER_TURNS, replace=ids.size < CLUSTER_TURNS)


def _cluster_grids(
    scores: np.ndarray, gal: np.ndarray, clusters: list[tuple[int, np.ndarray]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(best, arg, truth)`` grids of shape (clusters, CLUSTER_TURNS)."""
    sub = scores[:, gal]
    best = sub.max(axis=1)
    arg = gal[sub.argmax(axis=1)]
    picks = np.stack([p for _, p in clusters])
    owners = np.array([s for s, _ in clusters], dtype=int)
    return best[picks], arg[picks], owners


# --------------------------------------------------------------------------- #
# the four sweeps
# --------------------------------------------------------------------------- #
def sweep_dir_and_threshold(
    scores: np.ndarray, truth: np.ndarray, plan: DrawPlan, far: float
) -> dict:
    dirs, thrs = [], []
    for gal, in_gal in zip(plan.galleries, plan.in_gallery):
        sub = scores[:, gal]
        best = sub.max(axis=1)
        arg = gal[sub.argmax(axis=1)]
        thr = threshold_at_far(best[~in_gal], far)
        dirs.append(dir_at_threshold(best[in_gal], arg[in_gal], truth[in_gal], thr))
        thrs.append(thr)
    return {
        "dir_mean": float(np.mean(dirs)),
        "dir_std": float(np.std(dirs)),
        "thr_mean": float(np.mean(thrs)),
        "thr_std": float(np.std(thrs)),
        "draws": len(dirs),
    }


def sweep_k_accumulation(scores: np.ndarray, plan: DrawPlan, lo: float) -> dict:
    gen_best, gen_arg, gen_own, imp_best, imp_arg = [], [], [], [], []
    for gal, gen, imp in zip(plan.galleries, plan.genuine, plan.impostor):
        if gen:
            b, a, o = _cluster_grids(scores, gal, gen)
            gen_best.append(b), gen_arg.append(a), gen_own.append(o)
        if imp:
            b, a, _ = _cluster_grids(scores, gal, imp)
            imp_best.append(b), imp_arg.append(a)
    gb, ga = np.concatenate(gen_best), np.concatenate(gen_arg)
    go = np.concatenate(gen_own)
    ib, ia = np.concatenate(imp_best), np.concatenate(imp_arg)

    out: dict = {
        "lo": lo,
        "top1": float(np.mean(ga == go[:, None])),
        "genuine_clusters": int(gb.shape[0]),
        "impostor_clusters": int(ib.shape[0]),
        "per_k": {},
    }
    for k in K_VALUES:
        fired, name = vote_fires(gb[:, :k], ga[:, :k], lo)
        imp_fired, _ = vote_fires(ib[:, :k], ia[:, :k], lo)
        out["per_k"][k] = {
            "consistency": float(np.mean(modal_agreement(ga[:, :k]))),
            "imp_consistency": float(np.mean(modal_agreement(ia[:, :k]))),
            "vote_ok": float(np.mean(fired & (name == go))),
            "vote_wrong": float(np.mean(fired & (name != go))),
            "imp_vote": float(np.mean(imp_fired)),
        }
    return out


def sweep_leakage(scores: np.ndarray, plan: DrawPlan, lo: float, hi: float) -> dict:
    cover, wrong = [], []
    per_turn, leak_turn, leak_ever, leak_final = [], [], [], []
    genuine_clusters = 0
    for gal, gen, imp in zip(plan.galleries, plan.genuine, plan.impostor):
        if gen:
            gb, ga, go = _cluster_grids(scores, gal, gen)
            genuine_clusters += gb.shape[0]
            for row in range(gb.shape[0]):
                shown, _ = run_cluster(gb[row], ga[row], hi, lo)
                cover.append(float(np.mean(shown == go[row])))
                wrong.append(float(np.mean((shown >= 0) & (shown != go[row]))))
        if imp:
            ib, ia, _ = _cluster_grids(scores, gal, imp)
            per_turn.append(ib >= lo)  # per-turn rule, no vote and no stickiness
            for row in range(ib.shape[0]):
                shown, ever = run_cluster(ib[row], ia[row], hi, lo)
                leak_turn.append(float(np.mean(shown >= 0)))
                leak_ever.append(1.0 if ever else 0.0)
                leak_final.append(1.0 if bool(np.any(shown >= 0)) else 0.0)
    return {
        "lo": lo,
        "hi": hi,
        "cover": float(np.mean(cover)) if cover else float("nan"),
        "wrong": float(np.mean(wrong)) if wrong else float("nan"),
        "per_turn": float(np.mean(np.concatenate(per_turn))) if per_turn else float("nan"),
        "leak_turn": float(np.mean(leak_turn)) if leak_turn else float("nan"),
        "leak_ever": float(np.mean(leak_ever)) if leak_ever else float("nan"),
        "leak_final": float(np.mean(leak_final)) if leak_final else float("nan"),
        "genuine_clusters": genuine_clusters,
        "impostor_clusters": len(leak_ever),
    }


# --------------------------------------------------------------------------- #
# printing
# --------------------------------------------------------------------------- #
def print_table(title: str, headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    print(f"\n{title}")
    print("  ".join(h.rjust(w) if i else h.ljust(w) for i, (h, w) in enumerate(zip(headers, widths))))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(c.rjust(w) if i else c.ljust(w) for i, (c, w) in enumerate(zip(row, widths))))


def pct(x: float) -> str:
    return "n/a" if math.isnan(x) else f"{100.0 * x:.1f}"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="offline_sweep.py",
        description="Offline voiceprint model/threshold sweep over cached embeddings "
                    "(zero forward passes).",
    )
    p.add_argument("--cache-dir", default=None,
                   help=f"directory holding emb_<model>.npz (default: ${ENV_CACHE})")
    p.add_argument("--models", default=None,
                   help="comma-separated model keys to keep (default: all found)")
    p.add_argument("--crops", default=None,
                   help=f"comma-separated crops from {','.join(CROP_ORDER)} (default: all)")
    p.add_argument("--n-values", default=",".join(str(n) for n in DEFAULT_N_VALUES),
                   help="comma-separated gallery sizes")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="gallery draw seed")
    p.add_argument("--draws", type=int, default=DEFAULT_DRAWS, help="galleries per N")
    p.add_argument("--far", type=float, default=DEFAULT_FAR, help="target false-accept rate")
    p.add_argument("--lo-base", type=float, default=LO_BASE,
                   help="operating point: lo at N=%d (centered scale)" % N_REF)
    p.add_argument("--lo-slope", type=float, default=LO_SLOPE,
                   help="operating point: lo gain per gallery doubling")
    p.add_argument("--hi-delta", type=float, default=HI_DELTA,
                   help="operating point: hi = lo + this")
    p.add_argument("--json", dest="json_path", default=None,
                   help="also write the full structured result to this path")
    return p.parse_args(argv)


def _resolve_cache(arg_dir: str | None) -> Path:
    raw = arg_dir or os.environ.get(ENV_CACHE)
    if not raw:
        raise CacheError(
            "embedding cache not specified: pass --cache-dir <dir> or set "
            f"{ENV_CACHE}=<dir>. (缓存不存在，请指定 --cache-dir)"
        )
    path = Path(raw).expanduser()
    if not path.is_dir():
        raise CacheError(
            f"embedding cache directory not found: {path} -- pass --cache-dir <dir> "
            f"or set {ENV_CACHE}=<dir>. (缓存不存在，请指定 --cache-dir)"
        )
    if not sorted(path.glob("emb_*.npz")):
        raise CacheError(
            f"no emb_*.npz caches under {path} -- pass --cache-dir <dir> pointing at "
            f"the asnorm_cache directory. (缓存不存在，请指定 --cache-dir)"
        )
    return path


def _selected(requested: str | None, available: list[str], what: str) -> list[str]:
    if requested is None:
        return available
    picked = [x.strip() for x in requested.split(",") if x.strip()]
    missing = [x for x in picked if x not in available]
    if missing:
        raise CacheError(
            f"unknown {what}: {', '.join(missing)} (available: {', '.join(available)})"
        )
    return [x for x in available if x in picked]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.time()
    try:
        cache_dir = _resolve_cache(args.cache_dir)
        available = sorted(p.name[len("emb_"):-len(".npz")]
                           for p in cache_dir.glob("emb_*.npz"))
        models = _selected(args.models, available, "model")
        crops = tuple(_selected(args.crops, list(CROP_ORDER), "crop"))
        n_values = sorted({int(x) for x in args.n_values.split(",") if x.strip()})
        if not n_values:
            raise CacheError("--n-values selected nothing")
        bundles = [load_bundle(cache_dir / f"emb_{m}.npz", m, crops) for m in models]
    except CacheError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_NO_CACHE
    except ValueError as exc:
        print(f"ERROR: bad argument ({exc})", file=sys.stderr)
        return EXIT_NO_CACHE

    ref = bundles[0]
    for b in bundles[1:]:
        if b.test_keys != ref.test_keys or b.enroll_spk != ref.enroll_spk:
            print(f"ERROR: {b.path} does not cover the same speakers/utterances as "
                  f"{ref.path}; galleries would not be paired", file=sys.stderr)
            return EXIT_NO_CACHE

    n_spk = len(ref.enroll_spk)
    bad = [n for n in n_values if n < 1 or n >= n_spk]
    if bad:
        print(f"ERROR: --n-values {bad} outside 1..{n_spk - 1} (need at least one "
              f"out-of-gallery speaker to measure FAR)", file=sys.stderr)
        return EXIT_NO_CACHE

    by_spk = {i: np.flatnonzero(ref.truth == i) for i in range(n_spk)}
    plans = {n: build_plan(n, args.draws, n_spk, by_spk, ref.truth, args.seed)
             for n in n_values}

    scores = {
        (b.model, scoring, crop): score_matrix(b, crop, scoring)
        for b in bundles for scoring in SCORINGS for crop in crops
    }

    # Table 1 + 2 --------------------------------------------------------- #
    grid: dict = {}
    for b in bundles:
        for scoring in SCORINGS:
            for crop in crops:
                s = scores[(b.model, scoring, crop)]
                for n in n_values:
                    grid[(b.model, scoring, crop, n)] = sweep_dir_and_threshold(
                        s, ref.truth, plans[n], args.far)

    # Tables 3 + 4 run on one representative condition.
    primary_crop = crops[0] if "2" not in crops else "2"
    primary_scoring = "centered"
    kcum: dict = {}
    leak: dict = {}
    for b in bundles:
        s = scores[(b.model, primary_scoring, primary_crop)]
        for n in n_values:
            lo_n = lo_for_n(n, args.lo_base, args.lo_slope)
            kcum[(b.model, n)] = sweep_k_accumulation(s, plans[n], lo_n)
            leak[(b.model, "n-aware", n)] = sweep_leakage(
                s, plans[n], lo_n, lo_n + args.hi_delta)
            leak[(b.model, "fixed", n)] = sweep_leakage(
                s, plans[n], args.lo_base, args.lo_base + args.hi_delta)

    elapsed = time.time() - started
    result = _render(args, cache_dir, bundles, crops, n_values, grid, kcum, leak,
                     primary_scoring, primary_crop, elapsed)

    if args.json_path:
        Path(args.json_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_path).write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nwrote {args.json_path}")
    return EXIT_OK


def _render(args, cache_dir, bundles, crops, n_values, grid, kcum, leak,
            primary_scoring, primary_crop, elapsed) -> dict:
    models = [b.model for b in bundles]
    far_pct = f"{100.0 * args.far:g}%"

    print(f"offline_sweep: cache={cache_dir}  seed={args.seed}  draws={args.draws}  "
          f"FAR={far_pct}")
    for b in bundles:
        print(f"  {b.model:<24} dim={b.dim:<4} enroll={len(b.enroll_spk)}  "
              f"test={len(b.test_keys)}/crop  crops={','.join(CROP_LABEL[c] for c in crops)}")

    # Table 1 ------------------------------------------------------------- #
    rows = []
    for model in models:
        for scoring in SCORINGS:
            for crop in crops:
                cells = [model, scoring, CROP_LABEL[crop]]
                for n in n_values:
                    g = grid[(model, scoring, crop, n)]
                    cells.append(f"{pct(g['dir_mean'])} ±{100 * g['dir_std']:.1f}")
                rows.append(cells)
    print_table(
        f"== Table 1: DIR@FAR={far_pct} (%), mean ±sd over {args.draws} gallery draws",
        ["model", "scoring", "crop"] + [f"N={n}" for n in n_values], rows)

    # Table 2 ------------------------------------------------------------- #
    rows, fits = [], {}
    for model in models:
        for scoring in SCORINGS:
            for crop in crops:
                thrs = [grid[(model, scoring, crop, n)]["thr_mean"] for n in n_values]
                a, b_, r2 = fit_log2_slope(n_values, thrs)
                fits[(model, scoring, crop)] = {"a": a, "b": b_, "r2": r2}
                cells = [model, scoring, CROP_LABEL[crop]]
                cells += [f"{t:.4f}" for t in thrs]
                cells += [f"{a:.4f}", f"{b_:+.4f}",
                          "n/a" if math.isnan(r2) else f"{r2:.3f}"]
                rows.append(cells)
    print_table(
        f"== Table 2: thr@FAR={far_pct} vs gallery size, fitted thr(N) = a + b·log2(N/{N_REF})",
        ["model", "scoring", "crop"] + [f"thr@{n}" for n in n_values] + ["a", "b", "R2"],
        rows)
    print(f"   (b is the N-aware slope T3 ships; shipped default lo(N) = "
          f"{args.lo_base:.4f} {args.lo_slope:+.4f}·log2(N/{N_REF}))")
    print("   raw thresholds are plain cosine in [-1,1]; centered are the "
          "production (cos+1)/2 remap — do not compare across rows.")

    # Table 3 ------------------------------------------------------------- #
    rows = []
    for model in models:
        for n in n_values:
            k = kcum[(model, n)]
            cells = [model, str(n), f"{k['lo']:.4f}", pct(k["top1"])]
            cells += [pct(k["per_k"][j]["consistency"]) for j in K_VALUES]
            cells.append(pct(k["per_k"][5]["imp_consistency"]))
            cells += [pct(k["per_k"][j]["vote_ok"]) for j in K_VALUES]
            cells.append(pct(k["per_k"][5]["vote_wrong"]))
            cells.append(pct(k["per_k"][5]["imp_vote"]))
            rows.append(cells)
    print_table(
        f"== Table 3: k-turn accumulation (%) [{primary_scoring}, {CROP_LABEL[primary_crop]}, "
        f"lo=lo(N), {VOTE_K}-of-k vote]",
        ["model", "N", "lo", "top1"]
        + [f"cons@{k}" for k in K_VALUES] + ["impCons@5"]
        + [f"voteOK@{k}" for k in K_VALUES] + ["voteBad@5", "impVote@5"],
        rows)
    print("   top1      = per-turn top-1 accuracy on genuine turns (no threshold)")
    print("   cons@k    = mean share of a genuine cluster's k turns agreeing on its modal top-1")
    print("   impCons@5 = same for UNREGISTERED speakers — high means their errors are "
          "correlated, so voting is nearly free for them (plan §6.1 Q3)")
    print(f"   voteOK/Bad@k = {VOTE_K}-of-k vote fires the right / a wrong name on a genuine cluster")
    print(f"   impVote@5 = {VOTE_K}-of-5 vote fires ANY name on an unregistered speaker")

    # Table 4 ------------------------------------------------------------- #
    rows, gate_fail, denoms = [], [], []
    for model in models:
        for policy in ("n-aware", "fixed"):
            for n in n_values:
                d = leak[(model, policy, n)]
                denoms.append(d["impostor_clusters"])
                ok = (not math.isnan(d["leak_ever"])) and d["leak_ever"] <= LEAK_GATE
                if not ok:
                    gate_fail.append((model, policy, n, d["leak_ever"]))
                rows.append([model, policy, str(n), f"{d['lo']:.4f}", f"{d['hi']:.4f}",
                             pct(d["cover"]), pct(d["wrong"]), pct(d["per_turn"]),
                             pct(d["leak_turn"]), pct(d["leak_ever"]),
                             pct(d["leak_final"]), "PASS" if ok else "FAIL"])
    print_table(
        f"== Table 4: unregistered-speaker leakage (%) [{primary_scoring}, "
        f"{CROP_LABEL[primary_crop]}, {CLUSTER_TURNS}-turn clusters, sticky hi/lo + "
        f"{VOTE_K}-of-{VOTE_WINDOW} + contested lock]",
        ["model", "policy", "N", "lo", "hi", "cover", "wrong", "perTurn", "leakTurn",
         "leakEver", "leakFinal", f"gate<={100 * LEAK_GATE:g}%"],
        rows)
    print("   cover     = share of a genuine cluster's turns showing the CORRECT name")
    print("   wrong     = share of a genuine cluster's turns showing a WRONG name")
    print("   perTurn   = share of unregistered turns clearing lo under the OLD per-turn "
          "rule (no vote, no stickiness) — the baseline stickiness has to beat")
    print("   leakTurn  = share of unregistered turns left displaying a name")
    print("   leakEver  = share of unregistered clusters that EVER showed a name "
          "(V3 red line — a retracted name was still on screen); this is the gate")
    print("   leakFinal = share still showing one at cluster end (contested lock retracted)")
    print(f"   denominators: {min(denoms)}-{max(denoms)} impostor clusters per row "
          f"({args.draws} draws x {IMPOSTOR_CLUSTERS_PER_DRAW}) — a 1% gate resolves to "
          f"~{100.0 / max(denoms):.2f}pp")
    if gate_fail:
        print(f"   GATE FAIL on {len(gate_fail)} row(s); per plan §6.3 R1 raise lo by "
              f"0.01 and re-run: --lo-base {args.lo_base + 0.01:.4f}")
    else:
        print(f"   all rows within the ≤{100 * LEAK_GATE:g}% leakage gate (plan §2 / §6.3 R1)")

    # Honest confidence bounds -------------------------------------------- #
    print("\n== Confidence bounds — what these numbers CANNOT support (plan §6.4)")
    for i, line in enumerate(LIMITATIONS, 1):
        print(f"  {i}. {line}")
    print(f"\ndone in {elapsed:.2f}s")

    return {
        "meta": {
            "cache_dir": str(cache_dir),
            "seed": args.seed,
            "draws": args.draws,
            "far": args.far,
            "models": models,
            "crops": list(crops),
            "n_values": list(n_values),
            "operating_point": {"lo_base": args.lo_base, "lo_slope": args.lo_slope,
                                "hi_delta": args.hi_delta, "n_ref": N_REF},
            "primary_scoring": primary_scoring,
            "primary_crop": primary_crop,
            "cluster_turns": CLUSTER_TURNS,
            "vote": f"{VOTE_K}-of-{VOTE_WINDOW}",
            "leak_gate": LEAK_GATE,
            "elapsed_seconds": round(elapsed, 3),
            "sources": {b.model: {"path": str(b.path), "dim": b.dim,
                                  "enroll_speakers": len(b.enroll_spk),
                                  "test_utterances": len(b.test_keys)}
                        for b in bundles},
        },
        "dir_at_far": {f"{m}|{s}|{c}|{n}": grid[(m, s, c, n)]
                       for (m, s, c, n) in grid},
        "threshold_fit": {f"{m}|{s}|{c}": fits[(m, s, c)] for (m, s, c) in fits},
        "k_accumulation": {f"{m}|{n}": kcum[(m, n)] for (m, n) in kcum},
        "leakage": {f"{m}|{p}|{n}": leak[(m, p, n)] for (m, p, n) in leak},
        "limitations": list(LIMITATIONS),
    }


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
