"""Compute and persist a per-embedding-model cohort (impostor) mean.

Mean-centering the voiceprint embeddings against a cohort/impostor mean removes
the large shared, non-discriminative direction that compresses raw cosine scores
in wespeaker/pyannote embedding spaces (see
``docs/bugs/2026-06-30-voiceprint-score-compression.md``). This script computes
that mean *once* per working embedding model and saves it to
``assets/cohort/<model_slug>.npy``, so the running server can load it cheaply.

CRITICAL: the cohort must be an *impostor* set — DIVERSE audio clips that are
NOT the speakers we want to identify. Using the enrolled targets as the cohort
would subtract each target's own direction and is wrong.

Cohort composition (v2, 2026-07-06 false-accept fix — see
``docs/bugs/2026-07-06-voiceprint-false-accept.md`` §4.1e):
  * 14 x 4s segments of ``data/2news.mp3`` starting at 28..95s step 5s. The
    0-27s region is EXCLUDED: it contains the two news speakers used as
    verification targets (newscaster 0.6-6.6s & 19-24s, reporter 7.8-19s), and
    a cohort containing the verification targets contaminates the mean.
  * ALL clips of five AISHELL-3 speakers (SSB0267/0407/0534/0609/0710) from
    the local HF cache. These are *held out* of the verification pool
    (scripts/verify_speaker_cross_matrix.py uses different speakers) and add
    the same-gender/studio/Mandarin shared direction the 2news-only cohort
    missed — without them, threshold 0.6 could not reject same-gender
    impostors (they scored 0.72-0.80 against a stranger's enrollment).
    Skipped with a warning if the cache is absent.

Usage (from repo root):
    DIART_DEVICE=cuda python scripts/compute_cohort_mean.py

Writes:
    assets/cohort/pyannote__embedding.npy
    assets/cohort/pyannote__wespeaker-voxceleb-resnet34-LM.npy
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys
import wave

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

SRC_MP3 = os.path.join(REPO, "data", "2news.mp3")
CLIPS_DIR = "/tmp/vp_cohort_clips_v2"
OUT_DIR = os.path.join(REPO, "assets", "cohort")

# ~4s 2news segments tiled at step 5s over 28..95s ONLY (0-27s holds the
# verification-target speakers; see module docstring). (name -> (start, dur))
COHORT_SEG_DUR = 4
COHORT_SEG_STEP = 5
COHORT_CLIPS = {
    f"coh_{i}": (start, COHORT_SEG_DUR)
    for i, start in enumerate(range(28, 96, COHORT_SEG_STEP))
}

# Held-out AISHELL-3 cohort speakers (never used as enrollment/probe voices in
# verify_speaker_cross_matrix.py).
AISHELL_COHORT_SPEAKERS = ["SSB0267", "SSB0407", "SSB0534", "SSB0609", "SSB0710"]
AISHELL_GLOB = os.path.join(
    os.path.expanduser("~"),
    ".cache", "huggingface", "hub",
    "datasets--AISHELL--AISHELL-3", "snapshots", "*", "train", "wav",
)


def aishell_cohort_wavs() -> list[str]:
    """All clips of the held-out AISHELL cohort speakers, or [] if absent."""
    roots = sorted(glob.glob(AISHELL_GLOB))
    if not roots:
        print("WARNING: AISHELL-3 HF cache not found; cohort will be 2news-only")
        return []
    wavs: list[str] = []
    for spk in AISHELL_COHORT_SPEAKERS:
        hits = sorted(glob.glob(os.path.join(roots[0], spk, "*.wav")))
        if not hits:
            print(f"WARNING: AISHELL speaker {spk} missing from cache; skipping")
        wavs.extend(hits)
    return wavs

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


def load_wav_16k(path: str) -> np.ndarray:
    """Load any wav (e.g. 44.1 kHz AISHELL clips) as mono float32 16 kHz."""
    import soundfile as sf

    data, sr = sf.read(path, dtype="float32", always_2d=False)
    if getattr(data, "ndim", 1) > 1:
        data = data.mean(axis=1)
    if sr != 16000:
        import librosa

        data = librosa.resample(data, orig_sr=sr, target_sr=16000)
    return np.asarray(data, dtype=np.float32)


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
    aishell = aishell_cohort_wavs()
    pcms += [load_wav_16k(p) for p in aishell]
    print(f"cohort clips: {len(COHORT_CLIPS)} x 2news + {len(aishell)} x AISHELL")

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
