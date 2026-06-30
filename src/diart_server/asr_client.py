"""Async client for the Qwen3-ASR OpenAI-compatible transcription API."""
from __future__ import annotations

import io
import re
import wave

import httpx
import numpy as np


def parse_asr_output(raw: str) -> tuple[str, str]:
    if "<asr_text>" not in raw:
        return "", raw.strip()
    meta, text = raw.split("<asr_text>", 1)
    m = re.match(r"language\s+(\S+)", meta.strip(), re.IGNORECASE)
    lang = m.group(1) if m and m.group(1).lower() != "none" else ""
    return lang, text.strip()


def encode_wav(pcm_16k: np.ndarray, sample_rate: int = 16000) -> bytes:
    pcm = np.asarray(pcm_16k, dtype=np.float32).reshape(-1)
    pcm = np.clip(pcm, -1.0, 1.0)
    int16 = (pcm * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(int16.tobytes())
    return buf.getvalue()


class AsrClient:
    def __init__(self, base_url: str, model: str, api_key: str, timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    async def _post_wav(self, wav_bytes: bytes, language: str | None) -> str:
        data = {"model": self.model}
        if language:
            data["language"] = language
        files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/audio/transcriptions",
                data=data, files=files, headers=headers,
            )
            resp.raise_for_status()
            return resp.json().get("text", "")

    async def transcribe(
        self, pcm_16k: np.ndarray, language: str | None = None
    ) -> tuple[str, str]:
        wav = encode_wav(pcm_16k)
        raw = await self._post_wav(wav, language)
        return parse_asr_output(raw)
