"""Integration test for :class:`diart_server.session.DiarizationSession`.

Uses the dummy segmentation/embedding models from ``tests/conftest.py`` (via
:class:`DummySegmentationModel` / :class:`DummyEmbeddingModel`) wrapped in a
fake registry so we never touch HuggingFace. The asyncio event loop is run
on a dedicated thread so the test thread can synchronously push synthetic
PCM frames while ``on_message`` callbacks fire onto the loop and ``close()``
is awaited via :func:`asyncio.run_coroutine_threadsafe`.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

# Ensure the repo's tests/conftest.py is importable so we can reuse the
# DummySegmentationModel / DummyEmbeddingModel classes defined there.
TESTS_DIR = Path(__file__).resolve().parent.parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from conftest import DummyEmbeddingModel, DummySegmentationModel  # noqa: E402

from diart.models import EmbeddingModel, SegmentationModel  # noqa: E402

from diart_server.protocol import QueryParams  # noqa: E402
from diart_server.session import DiarizationSession  # noqa: E402


class FakeRegistry:
    """Mimics :class:`ModelsRegistry` interface used by ``DiarizationSession``.

    The session only calls ``get_segmentation`` / ``get_embedding``; we
    return pre-built dummy models regardless of the requested name.
    """

    def __init__(
        self, seg: SegmentationModel, emb: EmbeddingModel
    ) -> None:
        self._seg = seg
        self._emb = emb

    def get_segmentation(self, name: str) -> SegmentationModel:  # noqa: ARG002
        return self._seg

    def get_embedding(self, name: str) -> EmbeddingModel:  # noqa: ARG002
        return self._emb


def _build_dummy_models() -> tuple[SegmentationModel, EmbeddingModel]:
    """Construct dummy segmentation/embedding models the same way the
    ``segmentation_model`` / ``embedding_model`` fixtures do."""
    return (
        SegmentationModel(DummySegmentationModel),
        EmbeddingModel(DummyEmbeddingModel),
    )


def test_diarization_session_end_to_end() -> None:
    """Full happy-path: start, push PCM, observe predictions, close."""

    seg, emb = _build_dummy_models()
    registry = FakeRegistry(seg, emb)

    # Run a real asyncio loop on a background thread. The session schedules
    # ``on_message`` coroutines onto this loop via run_coroutine_threadsafe.
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

    session = DiarizationSession(
        params=params,
        registry=registry,  # type: ignore[arg-type]
        on_message=on_message,
        loop=loop,
    )

    try:
        # start() must not raise.
        session.start()

        # Push 6 frames of 0.5s of synthetic PCM (random noise float32 16k).
        # Diart's default duration=5s, step=0.5s — predictions start once the
        # first 5s window is filled, so we push more than 5s of audio plus a
        # cool-down to let inference catch up.
        rng = np.random.default_rng(seed=0)
        frame_samples = 8000  # 0.5s @ 16kHz
        for _ in range(6):
            pcm = rng.standard_normal(frame_samples).astype(np.float32) * 0.1
            session.push_pcm(pcm)
            time.sleep(0.5)

        # Push a few extra frames so the pipeline has at least 5s of buffered
        # audio to actually emit predictions (window=5s, step=0.5s).
        for _ in range(8):
            pcm = rng.standard_normal(frame_samples).astype(np.float32) * 0.1
            session.push_pcm(pcm)
            time.sleep(0.2)

        # Allow inference to drain remaining frames and emit predictions.
        time.sleep(2.0)

        # close() is async — schedule onto the loop and await the result.
        future = asyncio.run_coroutine_threadsafe(session.close(), loop)
        final = future.result(timeout=10.0)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=2.0)
        loop.close()

    # ------------------------------------------------------------------
    # Assertions
    # ------------------------------------------------------------------

    # close() returned a final summary message.
    assert isinstance(final, dict)
    assert final.get("type") == "final"
    assert "rttm" in final and isinstance(final["rttm"], str)
    assert "duration" in final
    assert "num_speakers" in final
    assert "stats" in final and "predictions_count" in final["stats"]

    # At least one speaker was tracked across the run.
    assert (
        len(session._speaker_counts) >= 1
    ), f"_speaker_counts unexpectedly empty: {dict(session._speaker_counts)!r}"

    # At least one prediction message reached on_message.
    with messages_lock:
        snapshot = list(messages)
    prediction_msgs = [m for m in snapshot if m.get("type") == "prediction"]
    assert (
        len(prediction_msgs) >= 1
    ), f"no prediction messages received; got types={[m.get('type') for m in snapshot]!r}"

    # Each prediction has the expected shape; embedding is a list of floats
    # (possibly empty if the centroid hadn't accumulated yet — e.g. the
    # speaker label was clustered after the centroid was already L2-normalized
    # to zero, or _extract_embedding returned None).
    for pmsg in prediction_msgs:
        assert "speaker" in pmsg
        assert "start" in pmsg and "duration" in pmsg
        assert "samples_count" in pmsg
        assert isinstance(pmsg.get("embedding"), list)
        for v in pmsg["embedding"]:
            assert isinstance(v, float)

    # At least one prediction should have a non-empty embedding (the dummy
    # embedding model returns 128-512 dim random vectors which after
    # L2-normalization remain non-zero with overwhelming probability).
    non_empty = [p for p in prediction_msgs if len(p["embedding"]) > 0]
    assert (
        len(non_empty) >= 1
    ), "expected at least one prediction with a non-empty embedding"

    # Sanity-check embedding dimensionality: the dummy embedding model
    # produces vectors in [128, 512]. The exact dim is randomized so we
    # only assert membership in that range.
    sample = non_empty[0]["embedding"]
    assert 128 <= len(sample) <= 512, f"unexpected embedding dim: {len(sample)}"
