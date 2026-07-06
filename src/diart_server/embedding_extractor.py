"""Extract an utterance-level voiceprint from an enrollment clip or turn slice.

Uses the same diart embedding model as live sessions so enrollment vectors
and per-turn identity vectors live in the same space
(docs/bugs/2026-07-06-voiceprint-false-accept.md).

``EMBED_EXECUTOR`` is the single process-wide worker every voiceprint forward
pass must run on (enrollment route AND per-turn session identification). One
worker (i) keeps the asyncio loop free, (ii) serializes access to the shared,
registry-cached ``EmbeddingModel`` so no two callers race a ``model.to()``
device move, and (iii) bounds GPU contention (a single wespeaker forward is
~10-50 ms on cuda).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
import torch

from diart.blocks.embedding import SpeakerEmbedding

from diart_server.config import settings

MAX_ENROLL_SECONDS = 30  # cap clip length fed to the model

# Single worker for ALL voiceprint embedding forwards (see module docstring).
EMBED_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vp-embed")


def make_embedder(model_name: str, registry, device: Optional[torch.device] = None) -> SpeakerEmbedding:
    """Build a :class:`SpeakerEmbedding` around the registry-cached model.

    ``device`` defaults to ``settings.device`` (NOT ``None``):
    ``SpeakerEmbedding(model, None)`` would call ``model.to(cpu)`` on the
    SHARED registry model, silently moving a live cuda session's embedding
    weights to CPU and crashing its next inference step
    (docs/bugs/2026-07-06-voiceprint-false-accept.md §3.2). Resolving the
    default from settings makes the internal ``.to()`` a no-op against the
    pipeline's device.
    """
    if device is None:
        device = torch.device(settings.device)
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
