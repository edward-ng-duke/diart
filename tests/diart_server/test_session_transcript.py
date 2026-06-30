"""Integration test: per-turn ASR → TranscriptMessage emission.

Mirrors test_session.py's setup (dummy models, background loop, synthetic PCM)
but also wires in a monkeypatched AsrClient so no network is needed, and
asserts that at least one ``transcript`` message with the expected text is
emitted after ``close()``.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

# Make tests/conftest.py importable so the dummy model classes are available.
TESTS_DIR = Path(__file__).resolve().parent.parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from conftest import DummyEmbeddingModel, DummySegmentationModel  # noqa: E402

from diart.models import EmbeddingModel, SegmentationModel  # noqa: E402

from diart_server.asr_client import AsrClient  # noqa: E402
from diart_server.protocol import QueryParams  # noqa: E402
from diart_server.session import DiarizationSession  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers (mirrored from test_session.py)
# ---------------------------------------------------------------------------


class FakeRegistry:
    """Minimal registry stub for DiarizationSession."""

    def __init__(self, seg: SegmentationModel, emb: EmbeddingModel) -> None:
        self._seg = seg
        self._emb = emb

    def get_segmentation(self, name: str) -> SegmentationModel:  # noqa: ARG002
        return self._seg

    def get_embedding(self, name: str) -> EmbeddingModel:  # noqa: ARG002
        return self._emb


def _build_dummy_models() -> tuple[SegmentationModel, EmbeddingModel]:
    return (
        SegmentationModel(DummySegmentationModel),
        EmbeddingModel(DummyEmbeddingModel),
    )


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


def test_finalized_turn_emits_transcript(monkeypatch: pytest.MonkeyPatch) -> None:
    """After close(), at least one transcript message with the patched text is received."""

    # Monkeypatch ASR so no network call is made.
    async def fake_transcribe(self, pcm, language=None):  # noqa: ANN001
        return ("Chinese", "你好世界")

    monkeypatch.setattr(AsrClient, "transcribe", fake_transcribe)

    seg, emb = _build_dummy_models()
    registry = FakeRegistry(seg, emb)

    # Background asyncio loop — same pattern as test_session.py.
    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
    loop_thread.start()

    messages: list[dict] = []
    messages_lock = threading.Lock()

    async def on_message(msg: dict) -> None:
        with messages_lock:
            messages.append(msg)

    params = QueryParams(
        sample_rate=16000,
        pipeline="diarization",
        segmentation_model="dummy/seg",
        embedding_model="dummy/emb",
    )

    asr_client = AsrClient("http://x/v1", "m", "EMPTY", 5.0)

    session = DiarizationSession(
        params=params,
        registry=registry,  # type: ignore[arg-type]
        on_message=on_message,
        loop=loop,
        store=None,
        asr_client=asr_client,
    )

    try:
        session.start()

        # Push synthetic PCM: same pacing as test_session.py.
        rng = np.random.default_rng(seed=42)
        frame_samples = 8000  # 0.5s @ 16kHz

        # First 6 frames × 0.5s pacing → fills the 5s diart window.
        for _ in range(6):
            pcm = rng.standard_normal(frame_samples).astype(np.float32) * 0.1
            session.push_pcm(pcm)
            time.sleep(0.5)

        # Extra frames to generate predictions (step=0.5s).
        for _ in range(8):
            pcm = rng.standard_normal(frame_samples).astype(np.float32) * 0.1
            session.push_pcm(pcm)
            time.sleep(0.2)

        # Allow inference to drain.
        time.sleep(2.0)

        # close() flushes remaining turns and waits for pending ASR.
        future = asyncio.run_coroutine_threadsafe(session.close(), loop)
        final = future.result(timeout=30.0)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=2.0)
        loop.close()

    # final summary must be well-formed.
    assert isinstance(final, dict)
    assert final.get("type") == "final"

    with messages_lock:
        snapshot = list(messages)

    types = [m.get("type") for m in snapshot]

    # At least one prediction must have been emitted (sanity check that PCM
    # was actually processed before asserting transcript).
    prediction_msgs = [m for m in snapshot if m.get("type") == "prediction"]
    assert len(prediction_msgs) >= 1, (
        f"no prediction messages received; types={types!r}"
    )

    # The key assertion: at least one transcript message.
    transcript_msgs = [m for m in snapshot if m.get("type") == "transcript"]
    assert len(transcript_msgs) >= 1, (
        f"no transcript messages received; types={types!r}"
    )

    t = transcript_msgs[0]
    assert t["text"] == "你好世界", f"unexpected text: {t['text']!r}"
    assert t["lang"] == "Chinese", f"unexpected lang: {t['lang']!r}"
