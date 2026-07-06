"""End-to-end verification of the voiceprint mean-centering fix.

Exercises the REAL call chain the running server uses:

    embedding_extractor.extract  (utterance embedding)
        -> voiceprint_matcher.match(query, refs, threshold, mean_embedding=cohort)

on three clips cut from data/2news.mp3 (A1/A2 = same speaker, B = different),
loading the committed wespeaker cohort mean from assets/cohort/. It confirms:

    * different-speaker query B against ref {"A": A1} returns None (BELOW
      threshold), and
    * same-speaker query A2 scores clearly above B (gap >= 0.10) at a usable
      absolute level (>= 0.75).

NOTE (2026-07-06 recalibration, docs/bugs/2026-07-06-voiceprint-false-accept.md
§4.1e): A1 and A2 come from DIFFERENT broadcast segments (studio lead-in vs
the 19-24s tail), i.e. this SAME pair is cross-channel. Under the per-turn
identity calibration (extended cohort + threshold 0.82) cross-channel genuine
deliberately lands *below* the accept threshold -> shown as "unknown", never
as a wrong name. So this script asserts separation/ordering, not
same_score > match_threshold.

Usage (from repo root):
    DIART_DEVICE=cuda python scripts/verify_match_separation.py

Exit code 0 = PASS, 1 = FAIL.
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
MODEL_NAME = "pyannote/wespeaker-voxceleb-resnet34-LM"
MODEL_BIN = os.path.join(REPO, "models", "pyannote", "wespeaker", "pytorch_model.bin")
COHORT_NPY = os.path.join(
    REPO, "assets", "cohort", MODEL_NAME.replace("/", "__") + ".npy"
)

# (name, start_s, dur_s) — same clips the Reporter used.
SPK_CLIPS = {
    "spkA_1": (1, 5),
    "spkA_2": (20, 4),
    "spkB": (9, 9),
}


def ensure_clips() -> None:
    os.makedirs(CLIPS, exist_ok=True)
    for name, (ss, dur) in SPK_CLIPS.items():
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


def main() -> int:
    ensure_clips()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from diart.models import EmbeddingModel
    from diart.blocks.embedding import SpeakerEmbedding
    from diart_server import embedding_extractor
    from diart_server.config import settings
    from diart_server.voiceprint_matcher import match as vp_match

    # Build the embedder exactly like make_embedder/extract use it.
    emb_model = EmbeddingModel.from_pyannote(MODEL_BIN, use_hf_token=False)
    emb_model.to(device).eval()
    embedder = SpeakerEmbedding(emb_model, device)

    A1 = embedding_extractor.extract(load_wav(f"{CLIPS}/spkA_1.wav"), embedder)
    A2 = embedding_extractor.extract(load_wav(f"{CLIPS}/spkA_2.wav"), embedder)
    B = embedding_extractor.extract(load_wav(f"{CLIPS}/spkB.wav"), embedder)

    cohort = np.load(COHORT_NPY).astype(np.float32).reshape(-1)
    thr = settings.match_threshold
    refs = {"A": A1}

    same_name, same_score = vp_match(A2, refs, thr, mean_embedding=cohort)
    diff_name, diff_score = vp_match(B, refs, thr, mean_embedding=cohort)

    print(f"model     : {MODEL_NAME}")
    print(f"threshold : {thr}")
    print(f"SAME  (A2 vs A): name={same_name!r} score={same_score:.3f}")
    print(f"DIFF  (B  vs A): name={diff_name!r} score={diff_score:.3f}")

    ok = (
        diff_name is None
        and same_score >= diff_score + 0.10
        and same_score >= 0.75
    )
    print("PASS criterion: diff_name is None AND same-diff >= 0.10 "
          "AND same >= 0.75")
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
