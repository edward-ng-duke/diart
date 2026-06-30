"""Extract an utterance-level voiceprint from an enrollment clip.

Uses the same diart embedding model as live sessions so enrollment vectors
live in the same space as the streamed centroid embeddings.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from diart.blocks.embedding import SpeakerEmbedding

MAX_ENROLL_SECONDS = 30  # cap clip length fed to the model


def make_embedder(model_name: str, registry, device: Optional[torch.device] = None) -> SpeakerEmbedding:
    emb_model = registry.get_embedding(model_name)
    return SpeakerEmbedding(emb_model, device)


def extract(pcm_16k: np.ndarray, embedder) -> np.ndarray:
    pcm = np.asarray(pcm_16k, dtype=np.float32).reshape(-1)
    pcm = pcm[: MAX_ENROLL_SECONDS * 16000]
    waveform = torch.from_numpy(np.ascontiguousarray(pcm)).reshape(-1, 1)  # (samples, channels)
    vec = embedder(waveform)
    v = vec.detach().cpu().numpy().astype(np.float32).reshape(-1)
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 0 else v
