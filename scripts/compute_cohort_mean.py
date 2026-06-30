"""Compute and persist a per-embedding-model cohort (impostor) mean.

Mean-centering the voiceprint embeddings against a cohort/impostor mean removes
the large shared, non-discriminative direction that compresses raw cosine scores
in wespeaker/pyannote embedding spaces (see
``docs/bugs/2026-06-30-voiceprint-score-compression.md``). This script computes
that mean *once* per working embedding model and saves it to
``assets/cohort/<model_slug>.npy``, so the running server can load it cheaply.

CRITICAL: the cohort must be an *impostor* set — a handful of DIVERSE audio clips
that are NOT the speakers we want to identify. Using the enrolled targets as the
cohort would subtract each target's own direction and is wrong. Here we cut a few
non-overlapping ~3-5s segments out of ``data/2news.mp3`` (a multi-speaker news
clip), which is enough to capture each model's shared baseline direction. A
larger, more diverse multi-speaker cohort (e.g. dozens of clips spanning many
speakers and recording conditions) would make the mean more robust; this minimal
set is sufficient to recover the discriminative range.

Usage (from repo root):
    DIART_DEVICE=cuda python scripts/compute_cohort_mean.py

Writes:
    assets/cohort/pyannote__embedding.npy
    assets/cohort/pyannote__wespeaker-voxceleb-resnet34-LM.npy
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

SRC_MP3 = os.path.join(REPO, "data", "2news.mp3")
CLIPS_DIR = "/tmp/vp_cohort_clips"
OUT_DIR = os.path.join(REPO, "assets", "cohort")

# Diverse ~4s segments tiled across the whole source clip (step 5s). More and
# more-diverse clips give a cleaner shared-direction estimate and wider
# same-vs-different separation; empirically a dense sweep of data/2news.mp3
# (~21 clips) widens the remapped gap vs. a handful of clips. A larger
# multi-speaker, multi-condition cohort would improve robustness further.
# (name, start_s, dur_s)
COHORT_SEG_DUR = 4
COHORT_SEG_STEP = 5
COHORT_CLIPS = {
    f"coh_{i}": (start, COHORT_SEG_DUR)
    # 0..95s covers data/2news.mp3 (~105s) without running past the end.
    for i, start in enumerate(range(0, 96, COHORT_SEG_STEP))
}

# model hf_name -> (local checkpoint path relative to models/, backend)
MODELS = {
    "pyannote/embedding": ("pyannote/embedding/pytorch_model.bin", "pyannote"),
    "pyannote/wespeaker-voxceleb-resnet34-LM": (
        "pyannote/wespeaker/pytorch_model.bin",
        "pyannote",
    ),
}


def model_slug(model_name: str) -> str:
    """Map an embedding-model hf_name to a filesystem-safe slug.

    Must stay consistent with ``diart_server.session._cohort_slug``.
    """
    return model_name.replace("/", "__")


def ensure_clips() -> None:
    os.makedirs(CLIPS_DIR, exist_ok=True)
    for name, (ss, dur) in COHORT_CLIPS.items():
        out = os.path.join(CLIPS_DIR, f"{name}.wav")
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


def main() -> int:
    ensure_clips()
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    from diart.models import EmbeddingModel
    from diart.blocks.embedding import SpeakerEmbedding

    pcms = [load_wav(os.path.join(CLIPS_DIR, f"{n}.wav")) for n in COHORT_CLIPS]

    for model_name, (rel_path, backend) in MODELS.items():
        bin_path = os.path.join(REPO, "models", rel_path)
        if not os.path.exists(bin_path):
            print(f"SKIP {model_name}: checkpoint missing at {bin_path}")
            continue
        emb_model = EmbeddingModel.from_pyannote(bin_path, use_hf_token=False)
        emb_model.to(device).eval()
        embedder = SpeakerEmbedding(emb_model, device)

        vecs = []
        for pcm in pcms:
            pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)[: 30 * 16000]
            wav = torch.from_numpy(np.ascontiguousarray(pcm)).reshape(-1, 1)
            v = embedder(wav).detach().cpu().numpy().astype(np.float32).reshape(-1)
            vecs.append(unit(v))

        mean = np.mean(np.stack(vecs, axis=0), axis=0)
        mean = unit(mean).astype(np.float32)
        out = os.path.join(OUT_DIR, f"{model_slug(model_name)}.npy")
        np.save(out, mean)
        print(
            f"OK {model_name}: dim={mean.shape[0]} "
            f"cohort_mean_norm(pre-unit)={float(np.linalg.norm(np.mean(np.stack(vecs), axis=0))):.3f} "
            f"-> {out}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
