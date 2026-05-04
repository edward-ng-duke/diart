"""Audio decoding and resampling for diart_server.

This module converts raw PCM audio frames received over the wire into
float32 numpy arrays at 16 kHz mono, the sample rate expected by the
diarization pipeline. The current implementation assumes the client
sends mono float32 PCM; stereo handling and compressed codec decoding
(WebM/Opus) are deferred to later iterations.
"""

from __future__ import annotations

import numpy as np
import torch
import torchaudio.functional as F

from diart_server.errors import BadAudioError, ResampleFailedError

TARGET_SR = 16000


def to_pcm_16k_mono(frame_bytes: bytes, src_sr: int) -> np.ndarray:
    """Decode a raw PCM frame to a 16 kHz mono float32 numpy array.

    Parameters
    ----------
    frame_bytes:
        Raw bytes representing a contiguous mono float32 PCM buffer.
    src_sr:
        Source sample rate of ``frame_bytes`` in Hz. If different from
        16000, the buffer is resampled via ``torchaudio.functional.resample``.

    Returns
    -------
    np.ndarray
        A 1-D ``float32`` array of shape ``(samples,)`` at 16 kHz.

    Raises
    ------
    BadAudioError
        If ``frame_bytes`` is empty.
    ResampleFailedError
        If ``torchaudio.functional.resample`` raises (e.g., invalid
        ``src_sr``). The boundary handler in ``ws_routes`` maps this to
        the ``RESAMPLE_FAILED`` wire-format error code.
    """
    pcm = np.frombuffer(frame_bytes, dtype=np.float32)
    if pcm.size == 0:
        raise BadAudioError("empty frame")

    if src_sr != TARGET_SR:
        # np.frombuffer returns a read-only view; copy before handing
        # to torch so the resampler can operate on a writable buffer.
        try:
            tensor = torch.from_numpy(pcm.copy()).unsqueeze(0)  # (1, samples)
            tensor = F.resample(tensor, src_sr, TARGET_SR)
            pcm = tensor.squeeze(0).numpy()
        except Exception as exc:  # noqa: BLE001 — wrap any torchaudio failure
            raise ResampleFailedError(
                f"resample {src_sr} -> {TARGET_SR} failed: {exc}"
            ) from exc

    return pcm.astype(np.float32, copy=False)


__all__ = ["to_pcm_16k_mono", "TARGET_SR"]
