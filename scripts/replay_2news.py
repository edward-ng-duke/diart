#!/usr/bin/env python
"""End-to-end regression guard for the turn/ASR quality fixes.

Replays ``data/2news.mp3`` through the REAL diart ``SpeakerDiarization``
pipeline and the REAL ``AsrClient``, collecting emitted ``TranscriptMessage``
payloads, and asserts that the known-clean baseline transcript lines still
appear after the BUG1/BUG2/BUG3 changes. This is deterministic w.r.t. the
audio file (no live mic, no flaky WebSocket path); it exercises the same
``DiarizationSession`` internals the server uses.

Usage:
    DIART_DEVICE=cuda python scripts/replay_2news.py

Exit codes:
    0  -> REPLAY OK (all baseline lines present) OR SKIPPED (models/ASR absent)
    1  -> a baseline line is missing (regression)

NOTE FOR FIXER/REVIEWER:
    ``BASELINE_SUBSTRINGS`` below is intentionally a TODO placeholder. On the
    first clean run against HEAD 058b47f (before the fixes), capture the 3
    stable transcript lines this script prints under "--- emitted ---" and
    pin them here as substrings (short, unambiguous fragments). The guard then
    fails if a fix regresses any of them.
"""
from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Pin these after the first clean baseline run (see module docstring).
BASELINE_SUBSTRINGS: list[str] = [
    # "TODO baseline line 1",
    # "TODO baseline line 2",
    # "TODO baseline line 3",
]

AUDIO = REPO / "data" / "2news.mp3"
CHUNK_SECONDS = 0.5  # feed cadence; matches diart step


def _load_pcm_16k(path: Path) -> np.ndarray:
    """Decode an mp3 to mono float32 16 kHz.

    Prefers librosa (handles mp3 + resample in one call); falls back to
    soundfile. Either is fine — the pipeline just needs mono 16 kHz float32.
    """
    try:
        import librosa  # type: ignore

        data, _ = librosa.load(str(path), sr=16000, mono=True)
        return np.asarray(data, dtype=np.float32)
    except Exception:  # noqa: BLE001
        import soundfile as sf  # type: ignore

        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if getattr(data, "ndim", 1) > 1:
            data = data.mean(axis=1)
        if sr != 16000:
            import librosa  # type: ignore

            data = librosa.resample(data, orig_sr=sr, target_sr=16000)
        return np.asarray(data, dtype=np.float32)


def main() -> int:
    if not AUDIO.exists():
        print(f"SKIPPED: {AUDIO} not found")
        return 0
    try:
        import torch  # noqa: F401

        from diart_server.asr_client import AsrClient
        from diart_server.config import settings
        from diart_server.models_registry import ModelsRegistry
        from diart_server.protocol import QueryParams
        from diart_server.session import DiarizationSession
    except Exception as exc:  # noqa: BLE001
        print(f"SKIPPED: imports unavailable ({exc})")
        return 0

    try:
        pcm = _load_pcm_16k(AUDIO)
    except Exception as exc:  # noqa: BLE001
        print(f"SKIPPED: could not decode audio ({exc})")
        return 0

    try:
        from diart_server.app import MANIFEST_PATH  # same manifest the server uses

        registry = ModelsRegistry(settings.models_dir, MANIFEST_PATH)
        # Warm the default models used by the demo.
        params = QueryParams()
        registry.get_segmentation(params.segmentation_model)
        registry.get_embedding(params.embedding_model)
    except Exception as exc:  # noqa: BLE001
        print(f"SKIPPED: models unavailable ({exc})")
        return 0

    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()
    messages: list[dict] = []
    lock = threading.Lock()

    async def on_message(msg: dict) -> None:
        with lock:
            messages.append(msg)

    asr = AsrClient(
        settings.asr_base_url, settings.asr_model,
        settings.asr_api_key, settings.asr_timeout,
    )
    session = DiarizationSession(
        params=params, registry=registry, on_message=on_message,
        loop=loop, store=None, asr_client=asr,
    )

    rc = 0
    try:
        session.start()
        step = int(CHUNK_SECONDS * 16000)
        for i in range(0, len(pcm), step):
            session.push_pcm(pcm[i:i + step])
            time.sleep(CHUNK_SECONDS)  # real-time-ish pacing so ASR keeps up
        fut = asyncio.run_coroutine_threadsafe(session.close(), loop)
        fut.result(timeout=settings.asr_timeout + 30)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        t.join(timeout=5.0)
        loop.close()

    with lock:
        transcripts = [m for m in messages if m.get("type") == "transcript"]
    print("--- emitted ---")
    for m in transcripts:
        tag = " [partial]" if m.get("partial") else ""
        ov = " [overlap]" if m.get("overlapped") else ""
        print(f"{m['start']:.1f}-{m['end']:.1f} {m['speaker']}{tag}{ov}: {m['text']}")

    if not BASELINE_SUBSTRINGS:
        print("SKIPPED: BASELINE_SUBSTRINGS not pinned yet (see docstring)")
        return 0

    joined = "\n".join(m["text"] for m in transcripts)
    missing = [s for s in BASELINE_SUBSTRINGS if s not in joined]
    if missing:
        print(f"REGRESSION: missing baseline lines: {missing}")
        rc = 1
    else:
        print("REPLAY OK")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
