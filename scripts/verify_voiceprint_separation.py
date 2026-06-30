"""Verification for the voiceprint score-compression fix.

Reproduces the UTTERANCE enrollment path used by diart_server
(embedding_extractor.extract -> voiceprint_matcher cosine) on three clips cut
from data/2news.mp3, and checks that a same-speaker pair scores clearly higher
than different-speaker pairs.

It measures BOTH:
  * RAW cosine (current production behaviour), and
  * MEAN-CENTERED cosine (the recommended fix: subtract a cohort/global mean
    embedding before cosine).

PASS criterion (mean-centered path, the fix):
    The separation GAP is the bug symptom (raw gap is tiny). The fix must
    open the gap AND keep a usable same-speaker score after remapping the
    centered cosine back to [0,1] via (c+1)/2:

        centered gap (same - diff_mean) > 0.25
        AND remapped same-speaker score ((same+1)/2) > 0.40

The RAW numbers are printed for reference but are NOT the pass gate — they are
expected to FAIL (tiny gap), which is the bug.

Usage (from repo root):
    DIART_DEVICE=cuda python scripts/verify_voiceprint_separation.py

Exit code 0 = PASS, 1 = FAIL. Audio clips are (re)generated under
/tmp/vp_investigation if missing.
"""
from __future__ import annotations

import os
import subprocess
import sys
import wave

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

CLIPS = "/tmp/vp_investigation"
SRC_MP3 = os.path.join(REPO, "data", "2news.mp3")
MODEL_BIN = os.path.join(REPO, "models", "pyannote", "wespeaker", "pytorch_model.bin")

# (name, start_s, dur_s)
SPK_CLIPS = {
    "spkA_1": (1, 5),
    "spkA_2": (20, 4),
    "spkB": (9, 9),
}
COHORT_CLIPS = {
    "coh_0": (2, 3),
    "coh_1": (10, 3),
    "coh_2": (13, 3),
    "coh_3": (16, 3),
    "coh_4": (21, 3),
}

# Pass gate (mean-centered path). The centered cosine is remapped to [0,1]
# via (c+1)/2 to give a usable absolute score; the GAP measures discrimination.
REMAPPED_SAME_MIN = 0.40
GAP_MIN = 0.25


def ensure_clips() -> None:
    os.makedirs(CLIPS, exist_ok=True)
    for name, (ss, dur) in {**SPK_CLIPS, **COHORT_CLIPS}.items():
        out = os.path.join(CLIPS, f"{name}.wav")
        if os.path.exists(out):
            continue
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(ss), "-t", str(dur),
             "-i", SRC_MP3, "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le", out],
            check=True,
        )


def load_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(unit(a), unit(b)))


def main() -> int:
    ensure_clips()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from diart.models import EmbeddingModel
    from diart.blocks.embedding import SpeakerEmbedding

    emb_model = EmbeddingModel.from_pyannote(MODEL_BIN, use_hf_token=False)
    emb_model.to(device).eval()
    embedder = SpeakerEmbedding(emb_model, device)

    def extract(pcm: np.ndarray) -> np.ndarray:
        pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)[: 30 * 16000]
        wav = torch.from_numpy(np.ascontiguousarray(pcm)).reshape(-1, 1)
        v = embedder(wav).detach().cpu().numpy().astype(np.float32).reshape(-1)
        return unit(v)

    A1 = extract(load_wav(f"{CLIPS}/spkA_1.wav"))
    A2 = extract(load_wav(f"{CLIPS}/spkA_2.wav"))
    B = extract(load_wav(f"{CLIPS}/spkB.wav"))

    raw_same = cos(A1, A2)
    raw_diff = (cos(A1, B) + cos(A2, B)) / 2
    print(f"RAW      : same={raw_same:.3f} diff={raw_diff:.3f} "
          f"gap={raw_same - raw_diff:.3f}")

    cohort = [extract(load_wav(f"{CLIPS}/coh_{i}.wav")) for i in range(5)]
    mean_emb = np.mean(np.stack(cohort + [A1, A2, B], axis=0), axis=0)

    def center(v):
        return unit(v - mean_emb)

    cA1, cA2, cB = center(A1), center(A2), center(B)
    c_same = cos(cA1, cA2)
    c_diff = (cos(cA1, cB) + cos(cA2, cB)) / 2
    c_gap = c_same - c_diff
    remap_same = (c_same + 1) / 2
    remap_diff = (c_diff + 1) / 2
    print(f"CENTERED : same={c_same:.3f} diff={c_diff:.3f} gap={c_gap:.3f}")
    print(f"REMAPPED : same={remap_same:.3f} diff={remap_diff:.3f} "
          f"gap={remap_same - remap_diff:.3f}")

    ok = (c_gap > GAP_MIN) and (remap_same > REMAPPED_SAME_MIN)
    print(f"\nPASS criterion: centered gap > {GAP_MIN} AND "
          f"remapped same > {REMAPPED_SAME_MIN}")
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
