"""Timestamped rolling buffer of 16 kHz mono float32 PCM.

Indexed by absolute stream time (cumulative samples / sample_rate), so a
finalized turn's [start, end] in diart's timeline can be sliced back out
of the raw audio to send to ASR.
"""
from __future__ import annotations

import threading

import numpy as np


class PcmRingBuffer:
    def __init__(self, max_seconds: float = 120.0, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate
        self._capacity = max(1, int(round(max_seconds * sample_rate)))
        self._buf = np.zeros(self._capacity, dtype=np.float32)
        self._total = 0  # total samples ever appended
        self._lock = threading.Lock()

    @property
    def total_seconds(self) -> float:
        return self._total / float(self.sample_rate)

    def append(self, pcm: np.ndarray) -> None:
        pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
        with self._lock:
            n = pcm.size
            if n == 0:
                return
            if n >= self._capacity:
                # Only the most recent `capacity` samples survive. Write them
                # at their true ring positions so the absolute-index -> position
                # mapping (s % capacity) that slice() relies on still holds.
                tail = pcm[-self._capacity:]
                start = (self._total + n - self._capacity) % self._capacity
                first = self._capacity - start
                self._buf[start:] = tail[:first]
                if first < self._capacity:
                    self._buf[:start] = tail[first:]
                self._total += n
                return
            start = self._total % self._capacity
            end = start + n
            if end <= self._capacity:
                self._buf[start:end] = pcm
            else:
                first = self._capacity - start
                self._buf[start:] = pcm[:first]
                self._buf[: n - first] = pcm[first:]
            self._total += n

    def slice(self, t0: float, t1: float) -> np.ndarray | None:
        with self._lock:
            s0 = int(round(t0 * self.sample_rate))
            s1 = int(round(t1 * self.sample_rate))
            oldest = max(0, self._total - self._capacity)
            s0 = max(s0, oldest)
            s1 = min(s1, self._total)
            if s1 <= s0:
                return None
            n = s1 - s0
            a = s0 % self._capacity
            end = a + n
            if end <= self._capacity:
                return self._buf[a:end].copy()
            first = self._capacity - a
            return np.concatenate((self._buf[a:], self._buf[: n - first]))
