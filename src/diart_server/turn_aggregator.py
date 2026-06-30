"""Aggregate per-window diart predictions into finalized utterances.

diart re-emits a growing turn each step; this collapses them into one
``FinalizedTurn`` per utterance and decides when an utterance is done.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FinalizedTurn:
    speaker: str
    start: float
    end: float


class _Open:
    __slots__ = ("start", "last_end")

    def __init__(self, start: float, last_end: float) -> None:
        self.start = start
        self.last_end = last_end


class TurnAggregator:
    def __init__(self, silence_gap: float = 0.7, max_turn_seconds: float = 20.0) -> None:
        self.silence_gap = silence_gap
        self.max_turn_seconds = max_turn_seconds
        self._open: dict[str, _Open] = {}

    def feed(self, speaker: str, start: float, end: float, now: float) -> list[FinalizedTurn]:
        finalized: list[FinalizedTurn] = []

        # 1) speaker switch: finalize other speakers whose turn ended before this starts
        for other in list(self._open):
            if other != speaker and self._open[other].last_end <= start:
                t = self._open.pop(other)
                finalized.append(FinalizedTurn(other, t.start, t.last_end))

        # 2) extend or open the current speaker
        #    But first: if the same speaker has an open turn with a silence gap,
        #    finalize it before starting a new one.
        cur = self._open.get(speaker)
        if cur is not None and (now - cur.last_end) > self.silence_gap:
            self._open.pop(speaker)
            finalized.append(FinalizedTurn(speaker, cur.start, cur.last_end))
            cur = None

        if cur is None:
            cur = _Open(start, end)
            self._open[speaker] = cur
        else:
            cur.last_end = max(cur.last_end, end)

        # 3) max-length finalization across all open turns (silence handled above)
        for s in list(self._open):
            t = self._open[s]
            if (t.last_end - t.start) > self.max_turn_seconds:
                self._open.pop(s)
                finalized.append(FinalizedTurn(s, t.start, t.last_end))

        finalized.sort(key=lambda f: f.start)
        return finalized

    def flush(self, now: float | None = None) -> list[FinalizedTurn]:
        out = [FinalizedTurn(s, t.start, t.last_end) for s, t in self._open.items()]
        self._open.clear()
        out.sort(key=lambda f: f.start)
        return out
