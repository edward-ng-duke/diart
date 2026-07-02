"""Aggregate per-window diart predictions into finalized utterances.

diart re-emits a growing turn each step; this collapses them into one
``FinalizedTurn`` per utterance and decides when an utterance is done.

Finalization is silence-driven on diart's OWN timeline (``_diart_now`` = the
max segment end ever seen), which is latency-free relative to the ingest clock
(see docs/bugs/2026-07-02-turn-asr-quality.md, BUG3). A speaker switch alone no
longer chops another speaker's open turn — brief interjections leave the other
speaker's turn open — so turns MAY overlap in time across speakers, which is
correct and desired.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass


@dataclass(frozen=True)
class FinalizedTurn:
    speaker: str
    start: float
    end: float
    turn_id: str = ""


@dataclass(frozen=True)
class PartialTurn:
    """A non-destructive snapshot of an open turn's progress so far (BUG1)."""

    speaker: str
    turn_id: str
    start: float
    end: float


class _Open:
    __slots__ = ("start", "last_end", "turn_id", "last_partial_end")

    def __init__(self, start: float, last_end: float) -> None:
        self.start = start
        self.last_end = last_end
        self.turn_id = uuid.uuid4().hex
        # Diart-timeline end at which the last partial was emitted; init to
        # start so the first partial fires after `interval` of new audio.
        self.last_partial_end = start


class TurnAggregator:
    def __init__(self, silence_gap: float = 0.7, max_turn_seconds: float = 12.0) -> None:
        self.silence_gap = silence_gap
        self.max_turn_seconds = max_turn_seconds
        self._open: dict[str, _Open] = {}
        # diart's own timeline: the max segment end ever fed. Used for all
        # silence decisions so they don't inflate by pipeline latency (BUG3).
        self._diart_now: float = 0.0

    def feed(self, speaker: str, start: float, end: float, now: float) -> list[FinalizedTurn]:
        """Feed one emitted (speaker, [start,end]) track.

        ``now`` (the ingest clock) is retained for signature compatibility but
        is NO LONGER used for gap logic: all silence decisions use
        ``_diart_now`` (= max end ever emitted), a latency-free clock (BUG3).
        """
        finalized: list[FinalizedTurn] = []

        # Silence reference = the current speech wavefront when this segment
        # arrives, on diart's own timeline. That is the incoming segment's
        # START (speech resumes at `start`), clamped up by any later wavefront
        # already seen (`_diart_now`). Using the segment's END here would
        # inflate the apparent gap by the segment duration and re-fragment a
        # dominant speaker whenever another speaker interjects (BUG3).
        ref = max(self._diart_now, start)

        # Unified silence finalization across ALL open speakers (BUG3). A
        # speaker switch alone no longer finalizes anyone; only a real silence
        # gap on diart's own timeline does. This lets a brief interjection by B
        # NOT chop A's open turn.
        for s in list(self._open):
            t = self._open[s]
            if (ref - t.last_end) > self.silence_gap:
                self._open.pop(s)
                finalized.append(FinalizedTurn(s, t.start, t.last_end, t.turn_id))

        self._diart_now = max(self._diart_now, end)

        # Extend or open the current speaker (already finalized above if it was
        # silent past the gap).
        cur = self._open.get(speaker)
        if cur is None:
            cur = _Open(start, end)
            self._open[speaker] = cur
        else:
            cur.last_end = max(cur.last_end, end)

        # Max-length finalization across all open turns (silence handled above).
        for s in list(self._open):
            t = self._open[s]
            if (t.last_end - t.start) > self.max_turn_seconds:
                self._open.pop(s)
                finalized.append(FinalizedTurn(s, t.start, t.last_end, t.turn_id))

        finalized.sort(key=lambda f: f.start)
        return finalized

    def due_partials(self, interval: float) -> list[PartialTurn]:
        """Return partial snapshots for open turns with >= ``interval`` new audio.

        Non-destructive: does NOT mutate ``_open`` membership, only advances
        each yielded turn's ``last_partial_end`` so partials fire every
        ``interval`` seconds of NEW open-turn audio (not every step). Reads the
        aggregator's own open state; no wall-clock needed (BUG1).
        """
        out: list[PartialTurn] = []
        if interval <= 0:
            return out
        for s, t in self._open.items():
            if (t.last_end - t.last_partial_end) >= interval:
                out.append(PartialTurn(s, t.turn_id, t.start, t.last_end))
                t.last_partial_end = t.last_end
        out.sort(key=lambda p: p.start)
        return out

    def flush(self, now: float | None = None) -> list[FinalizedTurn]:
        out = [
            FinalizedTurn(s, t.start, t.last_end, t.turn_id)
            for s, t in self._open.items()
        ]
        self._open.clear()
        out.sort(key=lambda f: f.start)
        return out
