"""Per-WebSocket diarization session.

A :class:`DiarizationSession` owns the lifetime of one streaming diart
inference: it builds a :class:`~diart.SpeakerDiarization` pipeline from
:class:`~diart_server.protocol.QueryParams`, wires it to a
:class:`~diart_server.meeting_audio_source.MeetingAudioSource`, runs
:class:`~diart.inference.StreamingInference` on a background thread, and
forwards predictions back to the WebSocket via the ``on_message``
coroutine running on the supplied event loop.

This module is the integration point between the asyncio WebSocket route
and diart's threaded inference loop. The bridge from the inference
thread back to the asyncio loop is performed via
``asyncio.run_coroutine_threadsafe`` (wired in T11/T12).

T10 covers only the constructor and the ``ready_message`` payload. The
runtime behaviour (``start``, ``push_pcm``, ``close``, ``_run``,
``_on_prediction``) is filled in by tasks T11-T13.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

import numpy as np
import torch

from diart import SpeakerDiarization, SpeakerDiarizationConfig
from diart.inference import StreamingInference
from diart.sinks import PredictionAccumulator

from diart_server import metrics
from diart_server.config import settings
from diart_server.meeting_audio_source import MeetingAudioSource
from diart_server.models_registry import ModelsRegistry
from diart_server.pcm_ring_buffer import PcmRingBuffer
from diart_server.protocol import (
    FinalMessage,
    PredictionMessage,
    QueryParams,
    ReadyMessage,
    TranscriptMessage,
)
from diart_server.turn_aggregator import TurnAggregator
from diart_server.voiceprint_matcher import match as vp_match

logger = logging.getLogger(__name__)

# Directory holding per-embedding-model cohort/impostor means used to
# mean-center voiceprint embeddings before cosine matching (see
# scripts/compute_cohort_mean.py and
# docs/bugs/2026-06-30-voiceprint-score-compression.md).
_COHORT_DIR = Path(__file__).resolve().parents[2] / "assets" / "cohort"


def _cohort_slug(model_name: str) -> str:
    """Map an embedding-model name to its cohort-mean filename slug.

    Must stay consistent with ``scripts/compute_cohort_mean.py:model_slug``.
    """
    return model_name.replace("/", "__")


def _load_cohort_mean(model_name: str) -> Optional[np.ndarray]:
    """Load the cohort/impostor mean for ``model_name``, or ``None`` if absent.

    Missing or unreadable cohort files are non-fatal: the matcher falls back to
    raw cosine behaviour when the mean is ``None``.
    """
    path = _COHORT_DIR / f"{_cohort_slug(model_name)}.npy"
    if not path.exists():
        logger.info("no cohort mean for %s at %s; using raw cosine", model_name, path)
        return None
    try:
        mean = np.load(path).astype(np.float32).reshape(-1)
    except Exception:  # noqa: BLE001
        logger.warning("failed to load cohort mean %s; using raw cosine", path)
        return None
    if mean.size == 0:
        return None
    return mean


class DiarizationSession:
    """Owns one streaming diart inference for a single WebSocket client.

    Construction only wires objects together; no thread is started and no
    audio is consumed until :meth:`start` (T11) is invoked.

    Parameters
    ----------
    params:
        Validated per-session parameters from the WebSocket query string.
    registry:
        Shared model registry. Models referenced by ``params`` must
        already be warmed up by the registry (handled at server startup
        in T15) before the pipeline is constructed here.
    on_message:
        Async callback used by the inference thread to push outbound
        JSON messages onto the WebSocket. Always invoked via
        ``asyncio.run_coroutine_threadsafe(on_message(...), loop)``
        from the inference thread (wired in T11/T12).
    loop:
        The asyncio event loop owning the WebSocket. Used as the target
        of ``run_coroutine_threadsafe`` calls from the inference thread.
    """

    def __init__(
        self,
        params: QueryParams,
        registry: ModelsRegistry,
        on_message: Callable[[dict], Awaitable[None]],
        loop: asyncio.AbstractEventLoop,
        store=None,
        asr_client=None,
    ) -> None:
        # T22 will route ``vad`` through a different pipeline. Until
        # then, refuse the request rather than silently treating it as
        # diarization.
        if params.pipeline == "vad":
            raise NotImplementedError(
                "vad pipeline is not implemented yet (see T22)"
            )

        self.id: str = str(uuid.uuid4())
        self.params: QueryParams = params
        self.registry: ModelsRegistry = registry
        self.on_message: Callable[[dict], Awaitable[None]] = on_message
        self.loop: asyncio.AbstractEventLoop = loop

        # Build the diart pipeline configuration from the session params.
        diart_config = SpeakerDiarizationConfig(
            segmentation=registry.get_segmentation(params.segmentation_model),
            embedding=registry.get_embedding(params.embedding_model),
            duration=params.duration,
            step=params.step,
            latency=params.latency,
            tau_active=params.tau_active,
            rho_update=params.rho_update,
            delta_new=params.delta_new,
            max_speakers=params.max_speakers,
            sample_rate=16000,
            device=torch.device(settings.device),
        )
        self.pipeline: SpeakerDiarization = SpeakerDiarization(diart_config)

        # Threaded audio source bridging the websocket producer and the
        # diart consumer. Overflows bump a Prometheus counter so we can
        # observe backpressure events without coupling the source to the
        # metrics module.
        self.source: MeetingAudioSource = MeetingAudioSource(
            uri=self.id,
            on_overflow=lambda: metrics.queue_overflow_total.inc(),
        )

        # Streaming inference orchestrator. ``do_plot`` and
        # ``show_progress`` are server-side concerns we always disable.
        self.inference: StreamingInference = StreamingInference(
            self.pipeline,
            self.source,
            do_plot=False,
            show_progress=False,
        )

        # Sink that accumulates the full RTTM up to the moment of close.
        # T13 reads it to build the final summary message.
        self.accumulator: PredictionAccumulator = PredictionAccumulator(self.id)
        self.inference.attach_observers(self.accumulator)

        # Inference runs on its own thread so the asyncio loop is free
        # to keep handling websocket frames. Started in T11.
        self._thread: threading.Thread = threading.Thread(
            target=self._run, daemon=True
        )

        # Per-speaker prediction counts, populated in T12 from the
        # observer callback. Used in the final summary stats (T13).
        self._speaker_counts: dict[str, int] = defaultdict(int)

        # Wall-clock start time, set by :meth:`start`. Used by T13 to
        # compute the total session duration in the close summary.
        self._started_at: Optional[float] = None

        # Monotonic timestamp of the most recent push_pcm call. Sampled
        # in :meth:`_on_prediction` to populate the prediction_latency
        # histogram (T28). This is an approximation: when multiple
        # frames are pushed between predictions only the last is timed.
        # Good enough for observability; precise frame-level latency
        # tracking would require a per-frame timestamp queue.
        self._last_push_at: Optional[float] = None

        # T11: voiceprint matching, turn aggregation, async ASR
        self.store = store
        self.asr_client = asr_client
        # Cohort/impostor mean for this embedding model, loaded once. When
        # present, voiceprint matching mean-centers embeddings before cosine to
        # remove the shared non-discriminative direction (score-compression fix).
        # ``None`` -> raw-cosine fallback (no crash if the asset is missing).
        self.cohort_mean: Optional[np.ndarray] = _load_cohort_mean(
            params.embedding_model
        )
        self.ring: PcmRingBuffer = PcmRingBuffer(
            settings.ring_buffer_seconds, sample_rate=16000
        )
        self.aggregator: TurnAggregator = TurnAggregator(
            settings.silence_gap, settings.max_turn_seconds
        )
        self._last_match: dict[str, tuple[Optional[str], Optional[float]]] = {}
        self._asr_sem: Optional[asyncio.Semaphore] = None
        self._asr_futures: set = set()

    # ------------------------------------------------------------------
    # Outbound payloads
    # ------------------------------------------------------------------

    def ready_message(self) -> dict:
        """Return the JSON-serializable payload for the ``ready`` event.

        ``sample_rate`` echoes the client-declared value (``params.sample_rate``);
        the internal pipeline always runs at 16 kHz with ``audio_decoder``
        resampling on the fly.
        """
        params = self.params
        return ReadyMessage(
            type="ready",
            session_id=self.id,
            sample_rate=params.sample_rate,
            pipeline=params.pipeline,
            models={
                "segmentation": params.segmentation_model,
                "embedding": params.embedding_model,
            },
            config={
                "tau_active": params.tau_active,
                "rho_update": params.rho_update,
                "delta_new": params.delta_new,
                "latency": params.latency,
                "max_speakers": params.max_speakers,
                "duration": params.duration,
                "step": params.step,
            },
        ).model_dump()

    # ------------------------------------------------------------------
    # Lifecycle (filled in by T11 / T13)
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background inference thread.

        Attaches the per-prediction observer hook, kicks off the
        inference thread, bumps the active-session gauge, and records
        the wall-clock start time used by T13 for the close summary.
        """
        self.inference.attach_hooks(self._on_prediction)
        self._thread.start()
        metrics.active_sessions.inc()
        self._started_at = time.monotonic()
        self._asr_sem = asyncio.Semaphore(settings.max_asr_concurrency)

    def push_pcm(self, pcm: np.ndarray) -> None:
        """Forward a decoded PCM frame to the audio source.

        Also increments the bytes-received counter so backpressure and
        throughput are observable from Prometheus, and stamps
        ``_last_push_at`` so :meth:`_on_prediction` can compute the
        end-to-end latency histogram (T28).
        """
        self.source.push_pcm(pcm)
        self.ring.append(pcm)
        metrics.audio_bytes_received.inc(pcm.nbytes)
        self._last_push_at = time.monotonic()

    async def close(self) -> dict:
        """Stop the inference thread and emit a final summary message.

        The source's queue is drained with a sentinel; the inference thread
        is joined with a 5-second timeout. Metrics gauges are decremented.
        The accumulated RTTM is included in the returned payload, along
        with session duration and per-speaker prediction counts.
        """
        self.source.close()
        # Inside an async def, get_running_loop() is the non-deprecated
        # form (get_event_loop() will raise on 3.14+). The join is
        # offloaded to the default executor so the asyncio loop stays
        # free.
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._thread.join, 5.0)
        if self._started_at is not None:
            metrics.active_sessions.dec()
            metrics.session_duration.observe(time.monotonic() - self._started_at)
            self._started_at = None  # idempotent close

        # Flush still-open turns (run on this loop) and also wait for any ASR
        # tasks already dispatched from _on_prediction. The latter are tracked
        # as the concurrent.futures.Futures returned by run_coroutine_threadsafe,
        # registered at dispatch time so none is missed (the inference thread is
        # already joined above, so no new ones can appear).
        flush_tasks = [
            asyncio.create_task(self._handle_turn(ft))
            for ft in self.aggregator.flush(self.ring.total_seconds)
        ]
        in_flight = list(self._asr_futures)

        async def _await_in_flight() -> None:
            if in_flight:
                await loop.run_in_executor(
                    None,
                    lambda: concurrent.futures.wait(
                        in_flight, timeout=settings.asr_timeout + 5
                    ),
                )

        if flush_tasks or in_flight:
            await asyncio.gather(
                *flush_tasks, _await_in_flight(), return_exceptions=True
            )

        prediction = self.accumulator.get_prediction()
        if prediction is not None and len(prediction.labels()) > 0:
            try:
                duration = prediction.get_timeline().extent().end
            except Exception:
                duration = 0.0
            rttm = prediction.to_rttm()
            num_speakers = len(prediction.labels())
        else:
            duration = 0.0
            rttm = ""
            num_speakers = 0

        return FinalMessage(
            type="final",
            rttm=rttm,
            duration=float(duration),
            num_speakers=num_speakers,
            stats={"predictions_count": sum(self._speaker_counts.values())},
        ).model_dump()

    # ------------------------------------------------------------------
    # Internals (filled in by T11 / T12)
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Inference thread target.

        Runs the blocking :class:`StreamingInference` loop. Any
        exception is converted into an ``error`` wire message and
        forwarded to the asyncio loop via
        :func:`asyncio.run_coroutine_threadsafe`, and bumps the
        ``session_errors_total`` counter under the ``INFERENCE_CRASH``
        label.
        """
        try:
            self.inference()
        except Exception as exc:
            msg = {
                "type": "error",
                "code": "INFERENCE_CRASH",
                "message": str(exc),
            }
            asyncio.run_coroutine_threadsafe(self.on_message(msg), self.loop)
            metrics.session_errors_total.labels(code="INFERENCE_CRASH").inc()

    def _on_prediction(self, ann_wav: Any) -> None:
        """Observer callback for each emitted prediction.

        Called by diart's ``attach_hooks`` mechanism on the inference
        thread. For each speaker turn in the emitted annotation, builds a
        :class:`PredictionMessage` (including the running clustering
        centroid as the embedding) and forwards it to the asyncio loop
        via :func:`asyncio.run_coroutine_threadsafe`.

        Also observes the ``prediction_latency`` histogram once per
        callback as ``now - _last_push_at`` (T28). The measurement is
        approximate because multiple frames may be pushed between
        predictions; it is intended for end-to-end observability rather
        than frame-level accounting.
        """
        if self._last_push_at is not None:
            metrics.prediction_latency.observe(
                time.monotonic() - self._last_push_at
            )
        annotation, _waveform = ann_wav
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            self._speaker_counts[speaker] += 1
            embedding = self._extract_embedding(speaker)
            name, score = None, None
            if self.store is not None and embedding is not None:
                refs = self.store.vectors(self.params.embedding_model)
                name, score = vp_match(
                    embedding,
                    refs,
                    settings.match_threshold,
                    mean_embedding=self.cohort_mean,
                )
                self._last_match[speaker] = (name, score)
            msg = PredictionMessage(
                type="prediction",
                start=turn.start,
                duration=turn.duration,
                speaker=speaker,
                embedding=embedding.tolist() if embedding is not None else [],
                samples_count=self._speaker_counts[speaker],
                name=name,
                score=score,
            ).model_dump()
            asyncio.run_coroutine_threadsafe(self.on_message(msg), self.loop)
            metrics.predictions_total.labels(
                pipeline=self.params.pipeline,
                model=self.params.embedding_model,
            ).inc()
            # Aggregate speaker turn and dispatch ASR for any finalized turns
            now = self.ring.total_seconds
            for ft in self.aggregator.feed(
                speaker, turn.start, turn.start + turn.duration, now
            ):
                fut = asyncio.run_coroutine_threadsafe(
                    self._handle_turn(ft), self.loop
                )
                self._asr_futures.add(fut)
                fut.add_done_callback(self._asr_futures.discard)

    async def _handle_turn(self, ft) -> None:
        """Process a finalized turn: run ASR and emit a TranscriptMessage."""
        if self.asr_client is None or self._asr_sem is None:
            return
        async with self._asr_sem:
            pcm = self.ring.slice(ft.start, ft.end)
            if pcm is None or pcm.size < 16000 // 4:  # < 0.25s -> too short for ASR
                return
            name, score = self._last_match.get(ft.speaker, (None, None))
            try:
                lang, text = await self.asr_client.transcribe(pcm)
            except Exception:  # noqa: BLE001
                lang, text = "", "[ASR失败]"
            msg = TranscriptMessage(
                speaker=ft.speaker,
                name=name,
                score=score,
                start=ft.start,
                end=ft.end,
                text=text,
                lang=lang,
            ).model_dump()
            await self.on_message(msg)

    def _extract_embedding(self, speaker_label: str) -> Optional[np.ndarray]:
        """Return the L2-normalized running centroid for the given speaker.

        Speaker labels emitted by diart are formatted as ``"speaker<idx>"``
        (e.g. ``"speaker0"``) where ``<idx>`` is the global cluster index
        in ``OnlineSpeakerClustering.centers``. The raw entry in
        ``centers`` is an unnormalized accumulating sum of all embeddings
        ever assigned to the cluster (see ``diart.blocks.clustering``);
        we L2-normalize before exposing so downstream voiceprint matchers
        receive a stable unit vector regardless of session length.

        Returns ``None`` when the clustering has not initialized yet
        (no speech seen), the label cannot be parsed, the index is out
        of range, the centroid index has not been added as an active
        center, or the accumulated centroid has zero magnitude.
        """
        # ``SpeakerDiarization`` exposes the running clusterer as
        # ``pipeline.clustering`` (see diart.blocks.diarization). We also
        # accept the underscored name as a fallback in case future diart
        # versions make it private.
        clustering = getattr(self.pipeline, "clustering", None) or getattr(
            self.pipeline, "_clustering", None
        )
        if clustering is None:
            return None
        centers = getattr(clustering, "centers", None)
        if centers is None:
            return None
        # Labels are "speaker<idx>" (see diart.blocks.utils); strip prefix.
        label = speaker_label
        if label.startswith("speaker"):
            label = label[len("speaker"):]
        try:
            idx = int(label)
        except (ValueError, TypeError):
            return None
        if idx < 0 or idx >= len(centers):
            return None
        active = getattr(clustering, "active_centers", None)
        if active is not None and idx not in active:
            return None
        vec = np.asarray(centers[idx], dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm <= 0.0:
            return None
        return vec / norm


__all__ = ["DiarizationSession"]
