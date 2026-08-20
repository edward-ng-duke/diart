"""Evidence for HOW the stale identity echo is bounded (count, not threshold).

Spec: ``docs/bugs/2026-08-20-diart-label-merge-identity-leak.md`` §3.

When diart merges two people onto one label, the obvious fix is to compare each
turn's embedding against the label's previous turn and declare the label
contested below some similarity threshold. **That does not work**, and this
script is the measurement that says so, on the REAL chain the server uses:

    embedding_extractor.extract   (same call as _identify_slice)
        -> voiceprint_matcher.match(turn, {"_ref": other_turn}, cohort)

The trap is the genuine set. Comparing adjacent slices of ONE continuous
utterance looks cleanly separable (gap +0.11 at 2s) -- but that is not the live
case. The live case is CROSS-SEGMENT: the same person, two different turns
minutes apart. This script builds genuine pairs only across DIFFERENT segments
of each speaker, and the separation collapses.

Gate (0 = PASS): no threshold in the swept range may reach
``MAX_FALSE_CONTEST`` false contests together with ``MIN_DETECTION`` detection.
Should one ever do so -- a better embedding model, a matched cohort -- this
script FAILS on purpose: the count-based rule in
``DiarizationSession._age_identity_echo`` was chosen because no such threshold
existed, and that choice should then be revisited.

Usage (from repo root):
    DIART_DEVICE=cuda python scripts/verify_label_continuity.py

Exit codes: 0 PASS, 1 a viable threshold now exists (revisit the design),
2 assets missing (audio, models or cohort).
"""
from __future__ import annotations

import itertools
import sys
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

AUDIO = REPO / "data" / "2news.mp3"
EMB_CKPT = REPO / "models" / "pyannote" / "wespeaker" / "pytorch_model.bin"
EMB_NAME = "pyannote/wespeaker-voxceleb-resnet34-LM"
SR = 16000

# Single-voice spans of data/2news.mp3, >= 2s each, taken from the shipped
# pipeline's own 2-label diarization of the file. Self-validated below: every
# span must sit closer to its own speaker's mean than to the other's.
SPANS = {
    "A": [(0.6, 6.6), (19.2, 27.2), (46.0, 49.0), (69.0, 72.0), (75.5, 78.5)],
    "B": [(10.0, 19.2), (34.5, 38.2), (38.5, 46.0), (50.3, 58.3), (78.5, 86.5)],
}
SLICES = (1.0, 1.5, 2.0, 3.0)
THRESHOLDS = (0.55, 0.60, 0.62, 0.65, 0.68, 0.70, 0.72, 0.75)

MAX_FALSE_CONTEST = 0.05   # dropping 5% of a real speaker's hints is the most
MIN_DETECTION = 0.80       # ...that a rule worth having may cost

EXIT_OK, EXIT_REVISIT, EXIT_NO_ASSETS = 0, 1, 2


def main() -> int:
    warnings.filterwarnings("ignore")
    import torch
    import torchaudio

    from diart.blocks.embedding import SpeakerEmbedding
    from diart.models import EmbeddingModel
    from diart_server.cohort import _load_cohort_mean
    from diart_server.config import settings
    from diart_server.embedding_extractor import extract
    from diart_server.voiceprint_matcher import match as vp_match

    for path in (AUDIO, EMB_CKPT):
        if not path.exists():
            print(f"SKIPPED: {path} not found", file=sys.stderr)
            return EXIT_NO_ASSETS
    cohort = _load_cohort_mean(EMB_NAME)
    if cohort is None:
        print(f"SKIPPED: no cohort mean for {EMB_NAME}", file=sys.stderr)
        return EXIT_NO_ASSETS

    device = torch.device(settings.device)
    embedder = SpeakerEmbedding(EmbeddingModel.from_pyannote(str(EMB_CKPT)), device)
    wav, sr = torchaudio.load(str(AUDIO))
    audio = (
        torchaudio.functional.resample(wav.mean(0, keepdim=True), sr, SR)
        .numpy().reshape(-1).astype(np.float32)
    )
    score = lambda x, y: vp_match(  # noqa: E731
        x, {"_ref": y}, 0.0, mean_embedding=cohort)[1]

    # --- guard: the spans really are two people, correctly grouped ---
    whole = {spk: [extract(audio[int(s * SR):int(e * SR)], embedder)
                   for s, e in spans] for spk, spans in SPANS.items()}
    means = {spk: np.mean(np.stack(v), axis=0) for spk, v in whole.items()}
    for spk, vecs in whole.items():
        other = "B" if spk == "A" else "A"
        for (s, e), v in zip(SPANS[spk], vecs):
            own, alien = score(v, means[spk]), score(v, means[other])
            if own <= alien:
                print(f"error: span {spk} {s}-{e}s scores {own:.3f} to its own "
                      f"speaker but {alien:.3f} to the other -- SPANS is wrong "
                      f"for this audio file", file=sys.stderr)
                return EXIT_NO_ASSETS
    print(f"device={device}  spans validated  "
          f"A={len(SPANS['A'])} B={len(SPANS['B'])}\n")

    print("CROSS-SEGMENT genuine (same speaker, DIFFERENT segments) vs impostor")
    print(f"{'slice':>6} | {'n_gen':>5} {'min':>7} {'p5':>7} {'p50':>7} | "
          f"{'n_imp':>5} {'max':>7} {'p95':>7} | {'gap':>7}")
    dists = {}
    for seconds in SLICES:
        n = int(seconds * SR)
        by_speaker = {}
        for spk, spans in SPANS.items():
            by_speaker[spk] = [
                [extract(audio[int(s * SR):int(e * SR)][i:i + n], embedder)
                 for i in range(0, int((e - s) * SR) - n + 1, n)]
                for s, e in spans
            ]
        genuine = []
        for groups in by_speaker.values():
            for g1, g2 in itertools.combinations(groups, 2):  # different segments
                genuine += [score(x, y) for x in g1 for y in g2]
        impostor = [score(x, y)
                    for g1 in by_speaker["A"] for g2 in by_speaker["B"]
                    for x in g1 for y in g2]
        if not genuine or not impostor:
            continue
        g, i = np.array(genuine), np.array(impostor)
        dists[seconds] = (g, i)
        print(f"{seconds:>5}s | {g.size:>5} {g.min():>7.3f} "
              f"{np.percentile(g, 5):>7.3f} {np.percentile(g, 50):>7.3f} | "
              f"{i.size:>5} {i.max():>7.3f} {np.percentile(i, 95):>7.3f} | "
              f"{g.min() - i.max():>+7.3f}")

    print(f"\nfalse-contest% / detection% per threshold "
          f"(a usable rule needs <= {MAX_FALSE_CONTEST:.0%} and "
          f">= {MIN_DETECTION:.0%})")
    print(f"{'slice':>6} " + " ".join(f"{t:>10.2f}" for t in THRESHOLDS))
    viable = []
    for seconds, (g, i) in dists.items():
        cells = []
        for thr in THRESHOLDS:
            fc, det = float((g < thr).mean()), float((i < thr).mean())
            cells.append(f"{fc * 100:>4.0f}/{det * 100:<5.0f}")
            if fc <= MAX_FALSE_CONTEST and det >= MIN_DETECTION:
                viable.append((seconds, thr, fc, det))
        print(f"{seconds:>5}s " + " ".join(f"{c:>10}" for c in cells))

    print()
    if viable:
        for seconds, thr, fc, det in viable:
            print(f"VIABLE {seconds}s @ thr={thr}: false-contest {fc:.1%}, "
                  f"detection {det:.1%}", file=sys.stderr)
        print("A threshold-based continuity rule is now viable -- revisit "
              "DiarizationSession._age_identity_echo.", file=sys.stderr)
        return EXIT_REVISIT
    print("PASS: no threshold separates cross-segment genuine from impostor at "
          "an acceptable cost, which is why the echo is bounded by a COUNT "
          f"(settings.identity_echo_max_misses={settings.identity_echo_max_misses}) "
          "instead.")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
