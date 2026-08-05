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
from diart_server.embedding_extractor import (
    EMBED_EXECUTOR,
    extract as extract_utterance_embedding,
    make_embedder,
)
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
from diart_server.turn_aggregator import FinalizedTurn, TurnAggregator
from diart_server.voiceprint_matcher import match as vp_match

logger = logging.getLogger(__name__)

# Minimum seconds between queue-overflow warning lines per session. Drops
# arrive per-frame (~6/s at 170 ms frames), so sustained RTF > 1 would
# otherwise log ~6 lines/s; once per 10 s bounds log volume at 0.1 lines/s
# while still making every degraded session visible
# (docs/bugs/2026-08-05-ring-diart-clock-desync.md §4.2).
_OVERFLOW_LOG_INTERVAL = 10.0

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

        # Rolling PCM buffer sliced by _handle_turn/_transcribe_and_emit.
        # Fed from the CONSUMER side (MeetingAudioSource.read via on_frame),
        # never from push_pcm: a frame enters the ring iff diart consumes it,
        # so ring.total_seconds ticks diart's consumed-samples clock and
        # frames dropped on queue overflow reach NEITHER diart NOR the ring —
        # both timelines share the same splice and stay aligned by
        # construction (docs/bugs/2026-08-05-ring-diart-clock-desync.md).
        self.ring: PcmRingBuffer = PcmRingBuffer(
            settings.ring_buffer_seconds, sample_rate=16000
        )

        # Threaded audio source bridging the websocket producer and the
        # diart consumer. Each consumed frame is appended to the ring before
        # it is emitted to the pipeline (see MeetingAudioSource.read).
        # Overflows bump a Prometheus counter and a rate-limited warning via
        # _on_queue_overflow so backpressure events are observable.
        self._overflow_count: int = 0
        self._last_overflow_log: Optional[float] = None
        self.source: MeetingAudioSource = MeetingAudioSource(
            uri=self.id,
            on_overflow=self._on_queue_overflow,
            on_frame=self.ring.append,
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
        self.aggregator: TurnAggregator = TurnAggregator(
            settings.silence_gap, settings.max_turn_seconds
        )
        # Last TURN-VERIFIED identity per diart label (written only by
        # _transcribe_and_emit after re-embedding a turn's own audio; see
        # docs/bugs/2026-07-06-voiceprint-false-accept.md). Read by
        # _on_prediction as a display hint for the prediction stream so it
        # never contradicts the transcript.
        self._last_match: dict[str, tuple[Optional[str], Optional[float]]] = {}
        # Lazily-built SpeakerEmbedding used for per-turn identification.
        # Created on the EMBED_EXECUTOR thread on first use; constructed with
        # the pipeline's device so its internal .to() never moves the shared
        # registry model off the pipeline's device.
        self._identity_embedder = None
        self._asr_sem: Optional[asyncio.Semaphore] = None
        self._asr_futures: set = set()
        # BUG2: multi-speaker (overlap) intervals in diart's timeline, merged
        # and pruned to the ring window. Written on the inference thread in
        # _on_prediction, read on the asyncio loop in _handle_turn -> guarded.
        self._overlap_intervals: list[tuple[float, float]] = []
        self._overlap_lock: threading.Lock = threading.Lock()
        # BUG3: last transcript-emitting finalized turn per speaker, so a
        # sub-0.25s fragment can be absorbed into the adjacent same-speaker turn
        # instead of being silently dropped. Only touched on the asyncio loop.
        self._last_final: dict[str, Any] = {}

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

        The ring buffer is deliberately NOT fed here: it is appended on the
        consumer side (``MeetingAudioSource.read`` via
        ``on_frame=self.ring.append``) so its clock matches diart's
        consumed-samples timeline even when queue overflow drops frames
        (docs/bugs/2026-08-05-ring-diart-clock-desync.md). The
        bytes-received counter and ``_last_push_at`` stay here — they
        intentionally measure the producer side (T28 latency histogram).
        """
        self.source.push_pcm(pcm)
        metrics.audio_bytes_received.inc(pcm.nbytes)
        self._last_push_at = time.monotonic()

    def _on_queue_overflow(self) -> None:
        """Record one dropped audio frame (source queue overflow).

        Runs on the asyncio loop inside :meth:`push_pcm`, so it must stay
        cheap: a counter bump, a clock read, and — at most once per
        ``_OVERFLOW_LOG_INTERVAL`` seconds — a warning so degraded sessions
        (inference slower than real time) are visible in logs instead of
        only in the ``queue_overflow_total`` Prometheus counter.
        """
        metrics.queue_overflow_total.inc()
        self._overflow_count += 1
        now = time.monotonic()
        if (
            self._last_overflow_log is None
            or now - self._last_overflow_log >= _OVERFLOW_LOG_INTERVAL
        ):
            self._last_overflow_log = now
            logger.warning(
                "session %s: audio queue overflow, %d frames dropped so "
                "far (inference slower than real time)",
                self.id,
                self._overflow_count,
            )

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
        # ring.total_seconds is the consumer clock and the ring is complete
        # here: the join above waited for read() to drain every queued frame
        # (each appended via on_frame) before the sentinel, so the flush "now"
        # sits on the same timeline as the aggregator's turn coordinates.
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
            # Identity is NEVER computed from the cluster centroid here: the
            # centroid is a blend of every voice diart assigned to the label,
            # which false-accepted impostors at 0.8-0.9 (see
            # docs/bugs/2026-07-06-voiceprint-false-accept.md). name/score
            # echo the last turn-verified identity for this label (written by
            # _transcribe_and_emit); None before the first identified turn.
            name, score = self._last_match.get(speaker, (None, None))
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
            # Aggregate speaker turn and dispatch ASR for any finalized turns.
            # ring.total_seconds is the CONSUMER clock (ring fed via on_frame),
            # i.e. the same timeline as turn.start/end — no drop-induced skew.
            now = self.ring.total_seconds
            for ft in self.aggregator.feed(
                speaker, turn.start, turn.start + turn.duration, now
            ):
                self._dispatch_turn(ft, partial=False)

        # BUG2: record multi-speaker (overlap) intervals for this emission so
        # _handle_turn can excise/flag mixed audio. get_overlap() gives exactly
        # the multi-speaker sub-intervals; no manual pairwise math needed.
        try:
            overlap_tl = annotation.get_overlap()
        except Exception:  # noqa: BLE001
            overlap_tl = None
        if overlap_tl is not None:
            new = [(float(seg.start), float(seg.end)) for seg in overlap_tl]
            if new:
                self._record_overlaps(new)

        # BUG1: emit partial ASR for open turns every partial_asr_interval
        # seconds of accumulated open-turn audio (0 disables the feature).
        if settings.partial_asr_interval > 0:
            for pt in self.aggregator.due_partials(settings.partial_asr_interval):
                self._dispatch_turn(pt, partial=True)

    def _dispatch_turn(self, ft, partial: bool = False) -> None:
        """Dispatch ASR for a finalized/partial turn onto the asyncio loop."""
        fut = asyncio.run_coroutine_threadsafe(
            self._handle_turn(ft, partial=partial), self.loop
        )
        self._asr_futures.add(fut)
        fut.add_done_callback(self._asr_futures.discard)

    def _record_overlaps(self, new: list[tuple[float, float]]) -> None:
        """Merge new overlap intervals, prune to the ring window, coalesce."""
        window_start = self.ring.total_seconds - settings.ring_buffer_seconds
        with self._overlap_lock:
            merged = self._overlap_intervals + new
            # Drop intervals fully aged out of the ring window.
            merged = [iv for iv in merged if iv[1] >= window_start]
            merged.sort()
            coalesced: list[tuple[float, float]] = []
            for s, e in merged:
                if coalesced and s <= coalesced[-1][1]:
                    prev_s, prev_e = coalesced[-1]
                    coalesced[-1] = (prev_s, max(prev_e, e))
                else:
                    coalesced.append((s, e))
            self._overlap_intervals = coalesced

    def _apply_overlap(self, pcm: np.ndarray, ft) -> tuple[np.ndarray, bool, float]:
        """Excise (minority) or flag (majority) overlapped audio before ASR.

        Pure function of ``pcm`` (a fresh slice copy — safe to mutate) and the
        currently-tracked overlap intervals. Returns ``(pcm, overlapped,
        overlap_ratio)`` where ``overlap_ratio`` is the fraction of
        ``[ft.start, ft.end]`` that overlapped another speaker:

        - ratio <= overlap_max_ratio (minority): zero the overlapped samples in
          the copied slice (keeps timestamps aligned); overlapped=False.
        - ratio  > overlap_max_ratio (majority): leave audio INTACT (excising
          would starve ASR); overlapped=True.
        - ratio == 0: unchanged; overlapped=False.
        """
        duration = ft.end - ft.start
        if duration <= 0:
            return pcm, False, 0.0
        with self._overlap_lock:
            intervals = list(self._overlap_intervals)
        # Intersect each tracked interval with [ft.start, ft.end].
        subs: list[tuple[float, float]] = []
        total = 0.0
        for s, e in intervals:
            lo = max(s, ft.start)
            hi = min(e, ft.end)
            if hi > lo:
                subs.append((lo, hi))
                total += hi - lo
        ratio = total / duration
        if ratio <= 0.0:
            return pcm, False, 0.0
        if ratio > settings.overlap_max_ratio:
            # Majority overlap: leave intact, just flag.
            return pcm, True, ratio
        # Minority overlap: zero out the overlapped samples (relative offsets).
        sr = self.ring.sample_rate
        n = pcm.size
        for lo, hi in subs:
            i0 = max(0, int(round((lo - ft.start) * sr)))
            i1 = min(n, int(round((hi - ft.start) * sr)))
            if i1 > i0:
                pcm[i0:i1] = 0.0
        return pcm, False, ratio

    async def _handle_turn(self, ft, partial: bool = False) -> None:
        """Process a turn: run ASR and emit a TranscriptMessage.

        ``partial=True`` emits an in-progress transcript for an OPEN turn
        (BUG1); the finalized turn later re-emits with the same ``turn_id`` and
        ``partial=False`` so the frontend replaces it. When a finalized turn's
        slice is below the 0.25s ASR gate, it is absorbed into the previous
        adjacent same-speaker turn instead of being dropped (BUG3).
        """
        if self.asr_client is None or self._asr_sem is None:
            return

        turn_id = getattr(ft, "turn_id", "") or ""
        pcm = self.ring.slice(ft.start, ft.end)
        if pcm is None or pcm.size < 16000 // 4:  # < 0.25s -> too short for ASR
            if partial:
                return  # a partial with too little audio yet: nothing to emit
            # BUG3: try to absorb this sub-0.25s fragment into the previous
            # adjacent same-speaker turn rather than dropping it.
            prev = self._last_final.get(ft.speaker)
            if prev is not None and (ft.start - prev.end) <= settings.absorb_gap:
                merged = FinalizedTurn(
                    speaker=prev.speaker,
                    start=prev.start,
                    end=ft.end,
                    turn_id=prev.turn_id,
                )
                emitted = await self._transcribe_and_emit(merged, partial=False)
                if emitted:
                    self._last_final[ft.speaker] = merged
                return
            return

        ft_norm = FinalizedTurn(
            speaker=ft.speaker, start=ft.start, end=ft.end, turn_id=turn_id
        )
        emitted = await self._transcribe_and_emit(ft_norm, partial=partial)
        if emitted and not partial:
            self._last_final[ft.speaker] = ft_norm

    async def _transcribe_and_emit(self, ft, partial: bool) -> bool:
        """Slice, apply overlap handling, identify, run ASR, emit a transcript.

        Returns True if a transcript was emitted, False if skipped (too short).
        ``ft`` must expose ``speaker/start/end/turn_id``.

        Identity comes from the turn's OWN (overlap-excised) audio, re-embedded
        with the same extractor used for enrollment and matched with the same
        matcher - never from the diart cluster label (the cluster centroid is a
        blend of every voice assigned to the label and false-accepted impostors
        at 0.8-0.9; docs/bugs/2026-07-06-voiceprint-false-accept.md). Turns and
        partials shorter than ``settings.min_identify_seconds`` emit
        ``name=None`` (no centroid fallback, no previous-turn fallback - both
        were the bug). Partials go through the same path, so a growing open
        turn's identity firms up and the final (same turn_id) replaces it.
        """
        assert self._asr_sem is not None
        async with self._asr_sem:
            pcm = self.ring.slice(ft.start, ft.end)
            if pcm is None or pcm.size < 16000 // 4:
                return False
            # BUG2: excise minority overlap / flag majority overlap.
            pcm, overlapped, overlap_ratio = self._apply_overlap(pcm, ft)
            name, score = None, None
            dur = ft.end - ft.start
            if self.store is not None and dur >= settings.min_identify_seconds:
                loop = asyncio.get_running_loop()
                try:
                    name, score = await loop.run_in_executor(
                        EMBED_EXECUTOR, self._identify_slice, pcm
                    )
                except Exception:  # noqa: BLE001
                    # Fail closed: an identification error must never stamp a
                    # name on the turn (and must not block the transcript).
                    logger.warning(
                        "per-turn identification failed for %s [%0.2f-%0.2f]",
                        ft.speaker, ft.start, ft.end, exc_info=True,
                    )
                    name, score = None, None
                else:
                    # Surface the turn-verified identity on the prediction
                    # stream (both partial and final go through here).
                    self._last_match[ft.speaker] = (name, score)
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
                turn_id=getattr(ft, "turn_id", "") or "",
                partial=partial,
                overlapped=overlapped,
                overlap_ratio=overlap_ratio,
            ).model_dump()
            await self.on_message(msg)
            return True

    def _identify_slice(
        self, pcm: np.ndarray
    ) -> tuple[Optional[str], Optional[float]]:
        """Embed a turn's own audio and match it against enrolled voiceprints.

        Runs on the single-worker ``EMBED_EXECUTOR`` thread (never on the
        asyncio loop). The per-session :class:`SpeakerEmbedding` wrapper is
        created lazily on first use with the pipeline's device
        (``settings.device``) so its constructor-time ``model.to()`` is a
        no-op against the shared registry model. Concurrent forwards from the
        inference thread (diart's own embedding step) are safe: eval mode,
        ``no_grad``, no mutable module state.
        """
        if self._identity_embedder is None:
            self._identity_embedder = make_embedder(
                self.params.embedding_model,
                self.registry,
                torch.device(settings.device),
            )
        vec = extract_utterance_embedding(pcm, self._identity_embedder)
        return vp_match(
            vec,
            self.store.vectors(self.params.embedding_model),
            settings.match_threshold,
            mean_embedding=self.cohort_mean,
        )

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
