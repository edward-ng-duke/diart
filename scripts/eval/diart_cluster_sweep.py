#!/usr/bin/env python
"""Gate + sweep for diart's clustering knobs (tau_active / rho_update / delta_new).

Spec: ``docs/bugs/2026-08-20-diart-label-merge-identity-leak.md``.

**Why this exists.** ``delta_new=1.057`` looks obviously wrong -- it sits above
the entire cross-speaker cosine-distance distribution (measured max 1.047 in
the wespeaker space), so diart's "this matches nobody -> new speaker" branch
never fires and two people separated by a silence longer than ``duration``
collapse into one label. The obvious fix (lower it) is measurably WORSE: on
real audio it shatters the clustering into 4-5 labels where there are 2. This
script is the artifact that keeps that from being re-litigated by opinion.

Scenarios are built from ``data/2news.mp3`` (in-repo, two clearly distinct
voices) so the check needs no external corpus:

    handoff-<N>s   A speaks, N seconds of silence, B speaks   -> merge probe
    alt-<T>s       A/B alternate in T-second turns            -> split probe
    solo-A         A alone with pauses                        -> over-split probe
    2news-real     the untouched file                         -> ground-truth count

Metrics per scenario:
    merge  two true speakers share their majority label      (under-clustering)
    split  one true speaker leaks >20% of their time elsewhere (over-clustering)
    nlbl   number of labels diart produced
    purity fraction of labelled frames sitting on the majority label

Gate (shipped config only, ``--gate``, the default):
    G1  2news-real must yield exactly 2 labels.
    G2  zero split failures.
    Merge failures are REPORTED, not gated: they are the documented residual
    the turn-level echo bound handles (settings.identity_echo_max_misses):
    the labels stay merged, but the first occupant's NAME stops following the
    second one around.

Usage (from repo root)::

    DIART_DEVICE=cuda python scripts/eval/diart_cluster_sweep.py
    python scripts/eval/diart_cluster_sweep.py --sweep --deltas 0.75,0.85,1.057

Exit codes: 0 gate passed / table printed, 1 gate FAILED, 2 audio or models
missing (never a traceback).
"""
from __future__ import annotations

import argparse
import itertools
import sys
import warnings
from pathlib import Path

import numpy as np

EXIT_OK, EXIT_GATE_FAILED, EXIT_NO_ASSETS = 0, 1, 2

REPO = Path(__file__).resolve().parents[2]
AUDIO = REPO / "data" / "2news.mp3"
SR = 16000

# Shipped operating point (diart_server.protocol.QueryParams defaults).
SHIPPED = dict(tau_active=0.507, rho_update=0.006, delta_new=1.057,
               latency=0.5, duration=5.0, step=0.5, max_speakers=10)

# Clean single-voice spans located inside 2news.mp3 by the shipped pipeline.
# Guarded at runtime: the two clips must be far apart in embedding space, so a
# changed audio file fails loudly instead of silently measuring one voice.
SPAN_A, SPAN_B = (19.2, 27.2), (78.5, 86.5)
MIN_SPAN_DISTANCE = 0.60

SPLIT_LEAK = 0.20  # fraction of a speaker's time on non-majority labels


def _die(msg: str) -> int:
    print(f"error: {msg}", file=sys.stderr)
    return EXIT_NO_ASSETS


def load_audio() -> np.ndarray:
    import torchaudio

    wav, sr = torchaudio.load(str(AUDIO))
    mono = wav.mean(0, keepdim=True)
    return (
        torchaudio.functional.resample(mono, sr, SR)
        .numpy()
        .reshape(-1)
        .astype(np.float32)
    )


def build_scenarios(audio, pauses, turns):
    A = audio[int(SPAN_A[0] * SR):int(SPAN_A[1] * SR)]
    B = audio[int(SPAN_B[0] * SR):int(SPAN_B[1] * SR)]
    sil = lambda s: np.zeros(int(s * SR), dtype=np.float32)  # noqa: E731
    out = []
    for pause in pauses:
        a = np.concatenate([A, sil(pause), B])
        t0 = (A.size + int(pause * SR)) / SR
        out.append((f"handoff-{pause:g}s", a,
                    [("A", 0.0, A.size / SR), ("B", t0, a.size / SR)]))
    for turn in turns:
        parts, truth, t = [], [], 0.0
        for k in range(8):
            src = A if k % 2 == 0 else B
            off = ((k // 2) * int(turn * SR)) % max(1, src.size - int(turn * SR))
            parts.append(src[off:off + int(turn * SR)])
            truth.append(("A" if k % 2 == 0 else "B", t, t + turn))
            t += turn
            parts.append(sil(0.6))
            t += 0.6
        out.append((f"alt-{turn:g}s", np.concatenate(parts), truth))
    parts, truth, t = [], [], 0.0
    for k in range(4):
        parts.append(A[k * 2 * SR:(k + 1) * 2 * SR])
        truth.append(("A", t, t + 2.0))
        t += 4.0
        parts.append(sil(2.0))
    out.append(("solo-A", np.concatenate(parts), truth))
    out.append(("2news-real", audio,
                [("A", *SPAN_A), ("B", *SPAN_B)]))
    return out, A, B


def make_models(models_dir: Path, device):
    from diart.models import EmbeddingModel, SegmentationModel

    seg_ckpt = models_dir / "pyannote" / "segmentation-3.0" / "pytorch_model.bin"
    emb_ckpt = models_dir / "pyannote" / "wespeaker" / "pytorch_model.bin"
    for p in (seg_ckpt, emb_ckpt):
        if not p.exists():
            raise FileNotFoundError(p)
    return (SegmentationModel.from_pyannote(str(seg_ckpt)),
            EmbeddingModel.from_pyannote(str(emb_ckpt)))


def diarize(audio, seg, emb, device, **cfg):
    from pyannote.core import Annotation, SlidingWindow, SlidingWindowFeature

    from diart import SpeakerDiarization, SpeakerDiarizationConfig

    params = dict(SHIPPED)
    params.update(cfg)
    pipeline = SpeakerDiarization(SpeakerDiarizationConfig(
        segmentation=seg, embedding=emb, sample_rate=SR, device=device, **params))
    win = int(params["duration"] * SR)
    hop = int(params["step"] * SR)
    ann = Annotation(uri="x")
    for i in range(0, max(1, audio.size - win), hop):
        chunk = SlidingWindowFeature(
            audio[i:i + win].reshape(-1, 1),
            SlidingWindow(start=i / SR, duration=1 / SR, step=1 / SR),
        )
        for emitted, _ in pipeline([chunk]):
            for segment, track, label in emitted.itertracks(yield_label=True):
                ann[segment, track] = label
    return ann.support()


def evaluate(ann, truth):
    from pyannote.core import Segment

    per, covered, total = {}, 0.0, 0.0
    for who, start, end in truth:
        cropped = ann.crop(Segment(start, end))
        durations = {lbl: cropped.label_duration(lbl) for lbl in cropped.labels()}
        bucket = per.setdefault(who, {})
        for lbl, value in durations.items():
            bucket[lbl] = bucket.get(lbl, 0.0) + value
        covered += sum(durations.values())
        total += end - start
    majority, correct, split = {}, 0.0, 0
    for who, durations in per.items():
        if not durations:
            continue
        top = max(durations, key=durations.get)
        majority[who] = top
        correct += durations[top]
        leak = sum(v for lbl, v in durations.items() if lbl != top)
        if leak > SPLIT_LEAK * sum(durations.values()):
            split += 1
    return dict(
        merge=len(set(majority.values())) < len(majority),
        split=split,
        nlbl=len(ann.labels()),
        purity=correct / max(1e-9, covered),
        cover=covered / max(1e-9, total),
    )


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep", action="store_true",
                   help="print the delta_new x rho_update grid instead of gating")
    p.add_argument("--deltas", default="0.70,0.75,0.80,0.85,0.95,1.057")
    p.add_argument("--rhos", default="0.006")
    p.add_argument("--pauses", default="1,3,6,10", help="handoff silence lengths (s)")
    p.add_argument("--turns", default="1.5,3", help="alternating turn lengths (s)")
    p.add_argument("--models-dir", default=str(REPO / "models"))
    p.add_argument("--device", default=None, help="default: $DIART_DEVICE or cpu")
    return p.parse_args(argv)


def main(argv=None) -> int:
    warnings.filterwarnings("ignore")
    args = parse_args(argv)
    if not AUDIO.exists():
        return _die(f"{AUDIO} not found")
    try:
        import torch

        from diart_server.config import settings
    except ImportError as exc:  # pragma: no cover - environment problem
        return _die(f"cannot import the pipeline ({exc})")

    device = torch.device(args.device or settings.device)
    try:
        seg, emb = make_models(Path(args.models_dir), device)
    except FileNotFoundError as exc:
        return _die(f"model checkpoint missing: {exc}")

    audio = load_audio()
    floats = lambda s: tuple(float(x) for x in s.split(",") if x)  # noqa: E731
    scenarios, clip_a, clip_b = build_scenarios(
        audio, floats(args.pauses), floats(args.turns))

    # Guard: the two spans must really be two people in the deployed space.
    from diart.blocks.embedding import SpeakerEmbedding

    embedder = SpeakerEmbedding(emb, device)
    vecs = []
    for clip in (clip_a, clip_b):
        wave = torch.from_numpy(np.ascontiguousarray(clip)).reshape(-1, 1)
        v = embedder(wave).detach().cpu().numpy().reshape(-1)
        vecs.append(v / np.linalg.norm(v))
    span_distance = 1.0 - float(vecs[0] @ vecs[1])
    if span_distance < MIN_SPAN_DISTANCE:
        return _die(
            f"SPAN_A/SPAN_B are only {span_distance:.3f} apart (need "
            f">= {MIN_SPAN_DISTANCE}); data/2news.mp3 changed -- relocate the spans"
        )
    print(f"device={device}  span distance={span_distance:.3f}  "
          f"scenarios={len(scenarios)}")

    grid = (list(itertools.product(floats(args.deltas), floats(args.rhos)))
            if args.sweep else [(SHIPPED["delta_new"], SHIPPED["rho_update"])])

    header = (f"{'delta':>6} {'rho':>6} | "
              + " ".join(f"{n:>13}" for n, _, _ in scenarios)
              + " | merge split purity")
    print(header)
    print("-" * len(header))
    gate_rows = {}
    for delta, rho in grid:
        cells, merges, splits, purities = [], 0, 0, []
        for name, wav, truth in scenarios:
            r = evaluate(diarize(wav, seg, emb, device, delta_new=delta,
                                 rho_update=rho), truth)
            flag = ("M" if r["merge"] else "") + ("S" if r["split"] else "")
            cells.append(f"{r['nlbl']}L {r['purity'] * 100:3.0f}%{flag:>2}")
            merges += r["merge"]
            splits += r["split"]
            purities.append(r["purity"])
            if (delta, rho) == (SHIPPED["delta_new"], SHIPPED["rho_update"]):
                gate_rows[name] = r
        print(f"{delta:>6.3f} {rho:>6.3f} | "
              + " ".join(f"{c:>13}" for c in cells)
              + f" | {merges:>5} {splits:>5} {np.mean(purities) * 100:5.1f}%")

    if args.sweep:
        print("\nM = two speakers merged into one label (under-clustering)")
        print("S = one speaker leaked >20% of their time to another label")
        print("Lowering delta_new trades M for S AND inflates the label count on"
              " real audio -- see the 2news-real column before believing it helps.")
        return EXIT_OK

    real = gate_rows.get("2news-real", {})
    total_split = sum(r["split"] for r in gate_rows.values())
    total_merge = sum(r["merge"] for r in gate_rows.values())
    g1 = real.get("nlbl") == 2
    g2 = total_split == 0
    print()
    print(f"G1 2news-real label count == 2 : {real.get('nlbl')}  "
          f"{'PASS' if g1 else 'FAIL'}")
    print(f"G2 split failures == 0         : {total_split}  "
          f"{'PASS' if g2 else 'FAIL'}")
    print(f"-- merge failures (documented residual; the identity echo they "
          f"would leak is bounded by settings.identity_echo_max_misses): "
          f"{total_merge}")
    if not (g1 and g2):
        print("\nGATE FAILED", file=sys.stderr)
        return EXIT_GATE_FAILED
    print("\nGATE PASSED")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
