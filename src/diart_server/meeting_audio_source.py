"""Thread-safe AudioSource for streaming PCM frames into a diart pipeline.

This module exposes :class:`MeetingAudioSource`, a subclass of
:class:`diart.sources.AudioSource` that bridges an arbitrary producer thread
(e.g. a websocket handler decoding PCM frames) and the diart inference loop
which consumes audio via the source's ``stream`` Subject.

The producer pushes numpy float32 PCM frames via :meth:`push_pcm`; the
consumer thread runs :meth:`read`, which blocks on an internal queue and
emits ``torch.Tensor`` chunks of shape ``(1, samples)`` (channels-first,
mono) on ``self.stream``.

Backpressure: if the queue is full, the oldest frame is dropped to make
room for the newest, and an optional ``on_overflow`` callback is invoked
so the caller can record metrics. Overflow accounting is intentionally
kept out of this class to avoid coupling it to a specific metrics
backend.
"""

from __future__ import annotations

import queue
import threading
from typing import Callable, Optional

import numpy as np
import torch

from diart.sources import AudioSource


class MeetingAudioSource(AudioSource):
    """Thread-safe AudioSource backed by a bounded queue.

    Parameters
    ----------
    uri:
        Unique identifier of the source (typically the meeting/session id).
    sample_rate:
        Sample rate of the PCM frames pushed via :meth:`push_pcm`. Defaults
        to 16 kHz, matching the diart pipeline.
    maxsize:
        Maximum number of frames buffered before backpressure kicks in.
        When full, :meth:`push_pcm` drops the oldest frame.
    on_overflow:
        Optional callback invoked (with no arguments) every time a frame
        is dropped due to queue saturation. Kept as a callback so this
        class stays decoupled from any metrics implementation.
    """

    def __init__(
        self,
        uri: str,
        sample_rate: int = 16000,
        maxsize: int = 50,
        on_overflow: Optional[Callable[[], None]] = None,
    ) -> None:
        super().__init__(uri, sample_rate)
        self._queue: "queue.Queue[Optional[np.ndarray]]" = queue.Queue(maxsize=maxsize)
        self._closed = threading.Event()
        self._on_overflow = on_overflow

    def push_pcm(self, pcm: np.ndarray) -> None:
        """Enqueue a PCM frame for the consumer thread.

        On queue overflow the oldest frame is dropped and the new one is
        enqueued in its place. The ``on_overflow`` callback (if any) is
        invoked exactly once per drop event.
        """
        try:
            self._queue.put_nowait(pcm)
            return
        except queue.Full:
            pass

        if self._on_overflow is not None:
            self._on_overflow()

        try:
            self._queue.get_nowait()
        except queue.Empty:
            # Another consumer drained between Full and now; harmless.
            pass

        try:
            self._queue.put_nowait(pcm)
        except queue.Full:
            # If still full (highly unlikely under single-producer usage),
            # silently drop the new frame to preserve liveness rather than
            # block the producer thread.
            pass

    def read(self) -> None:
        """Blocking consumer loop.

        Reads frames from the queue and pushes them onto ``self.stream`` as
        ``torch.Tensor`` of shape ``(1, samples)``. Exits when a sentinel
        ``None`` is received (pushed by :meth:`close`) and signals
        completion via ``self.stream.on_completed()``.
        """
        try:
            while True:
                pcm = self._queue.get()
                if pcm is None:
                    break
                tensor = torch.from_numpy(pcm).unsqueeze(0)
                self.stream.on_next(tensor)
        finally:
            self.stream.on_completed()

    def close(self) -> None:
        """Signal the consumer to stop and unblock :meth:`read`.

        Idempotent: subsequent calls are no-ops. If the queue is full, an
        oldest entry is dropped to make room for the sentinel so close()
        never blocks the caller (e.g. in tests with no consumer attached,
        or when the consumer is currently busy draining).
        """
        if self._closed.is_set():
            return
        self._closed.set()
        while True:
            try:
                self._queue.put_nowait(None)
                return
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    # Drained between checks; loop and retry the put.
                    continue
