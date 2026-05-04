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
import threading
import time
import uuid
from collections import defaultdict
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
from diart_server.protocol import PredictionMessage, QueryParams, ReadyMessage


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

    def push_pcm(self, pcm: np.ndarray) -> None:
        """Forward a decoded PCM frame to the audio source.

        Also increments the bytes-received counter so backpressure and
        throughput are observable from Prometheus.
        """
        self.source.push_pcm(pcm)
        metrics.audio_bytes_received.inc(pcm.nbytes)

    async def close(self) -> dict:
        """Stop inference and return the final summary. Implemented in T13."""
        raise NotImplementedError("T13")

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
        """
        annotation, _waveform = ann_wav
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            self._speaker_counts[speaker] += 1
            embedding = self._extract_embedding(speaker)
            msg = PredictionMessage(
                type="prediction",
                start=turn.start,
                duration=turn.duration,
                speaker=speaker,
                embedding=embedding.tolist() if embedding is not None else [],
                samples_count=self._speaker_counts[speaker],
            ).model_dump()
            asyncio.run_coroutine_threadsafe(self.on_message(msg), self.loop)
            metrics.predictions_total.labels(
                pipeline=self.params.pipeline,
                model=self.params.embedding_model,
            ).inc()

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
