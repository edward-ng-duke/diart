"""HTTP endpoints for enrolling/listing/removing voiceprints."""
from __future__ import annotations

import asyncio
import io
import wave

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from diart_server.config import settings
from diart_server.embedding_extractor import EMBED_EXECUTOR, extract, make_embedder
from diart_server.protocol import EnrollResponse

voiceprint_router = APIRouter()


def _decode_wav_to_float32(raw: bytes) -> tuple[np.ndarray, int]:
    """Decode WAV bytes to a float32 numpy array and return (pcm, sample_rate).

    Handles 16-bit PCM WAV files (the format produced by the test helper and
    most microphone / upload paths). Raises ValueError on malformed input.
    """
    try:
        buf = io.BytesIO(raw)
        with wave.open(buf, "rb") as w:
            sr = w.getframerate()
            n_channels = w.getnchannels()
            sampwidth = w.getsampwidth()
            frames = w.readframes(w.getnframes())
    except Exception as exc:
        raise ValueError(f"cannot parse WAV: {exc}") from exc

    if sampwidth == 2:
        pcm = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sampwidth == 4:
        pcm = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported sample width: {sampwidth} bytes")

    if n_channels > 1:
        pcm = pcm.reshape(-1, n_channels).mean(axis=1)

    return pcm, sr


def _resample_if_needed(pcm: np.ndarray, src_sr: int, target_sr: int = 16000) -> np.ndarray:
    """Resample pcm from src_sr to target_sr if needed."""
    if src_sr == target_sr:
        return pcm
    import torch
    import torchaudio.functional as F
    tensor = torch.from_numpy(np.ascontiguousarray(pcm)).unsqueeze(0)
    tensor = F.resample(tensor, src_sr, target_sr)
    return tensor.squeeze(0).numpy()


@voiceprint_router.post("/v1/voiceprints", response_model=EnrollResponse)
async def enroll(
    request: Request,
    name: str = Form(...),
    embedding_model: str = Form(...),
    file: UploadFile = File(...),
) -> EnrollResponse:
    raw = await file.read()
    try:
        pcm, src_sr = _decode_wav_to_float32(raw)
        pcm = _resample_if_needed(pcm, src_sr)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"bad audio: {exc}")
    if pcm.size < 16000 // 2:  # < 0.5 s at 16 kHz
        raise HTTPException(status_code=400, detail="clip too short (need >=0.5s)")

    registry = getattr(request.app.state, "registry", None)
    loop = asyncio.get_running_loop()
    try:
        # Offload the blocking model load + forward pass off the event loop
        # so a slow/failed enrollment cannot stall live WebSocket sessions.
        # Runs on the single-worker EMBED_EXECUTOR (serialized with per-turn
        # identification) and passes the server device explicitly: building a
        # SpeakerEmbedding without a device would move the SHARED registry
        # model to CPU and crash live cuda sessions (see
        # docs/bugs/2026-07-06-voiceprint-false-accept.md §3.2).
        import torch  # local import keeps module import light

        device = torch.device(settings.device)
        vec = await loop.run_in_executor(
            EMBED_EXECUTOR,
            lambda: extract(pcm, make_embedder(embedding_model, registry, device)),
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=400,
            detail=f"embedding model {embedding_model!r} could not be used: {exc}",
        )
    store = request.app.state.store
    store.enroll(embedding_model, name, vec)
    n = next((i["n_clips"] for i in store.list(embedding_model) if i["name"] == name), 1)
    return EnrollResponse(name=name, embedding_model=embedding_model, n_clips=n)


@voiceprint_router.get("/v1/voiceprints")
async def list_voiceprints(request: Request, embedding_model: str) -> dict:
    return {"voiceprints": request.app.state.store.list(embedding_model)}


@voiceprint_router.delete("/v1/voiceprints/{name}")
async def delete_voiceprint(request: Request, name: str, embedding_model: str) -> dict:
    request.app.state.store.remove(embedding_model, name)
    return {"status": "deleted", "name": name}
