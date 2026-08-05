"""Tests for diart_server.meeting_audio_source."""

from __future__ import annotations

import threading
import time
from typing import List

import numpy as np
import torch

from diart_server.meeting_audio_source import MeetingAudioSource


def test_push_5_frames_received_in_order() -> None:
    """Five pushed frames arrive on the stream in the order they were pushed."""
    src = MeetingAudioSource(uri="t")
    received: List[float] = []
    completed = threading.Event()

    src.stream.subscribe(
        on_next=lambda t: received.append(float(t.squeeze(0)[0].item())),
        on_completed=completed.set,
    )

    th = threading.Thread(target=src.read, daemon=True)
    th.start()

    for i in range(5):
        src.push_pcm(np.full(8, fill_value=float(i), dtype=np.float32))

    src.close()
    th.join(timeout=1.0)

    assert not th.is_alive()
    assert completed.is_set()
    assert received == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_emitted_tensor_has_channel_dimension() -> None:
    """Each emission is a torch.Tensor of shape (1, samples)."""
    src = MeetingAudioSource(uri="t")
    shapes: List[tuple] = []
    dtypes: List[torch.dtype] = []

    src.stream.subscribe(
        on_next=lambda t: (shapes.append(tuple(t.shape)), dtypes.append(t.dtype))
    )

    th = threading.Thread(target=src.read, daemon=True)
    th.start()

    src.push_pcm(np.zeros(160, dtype=np.float32))
    src.push_pcm(np.zeros(320, dtype=np.float32))

    src.close()
    th.join(timeout=1.0)

    assert shapes == [(1, 160), (1, 320)]
    assert all(d == torch.float32 for d in dtypes)


def test_overflow_drops_oldest_and_calls_callback() -> None:
    """When the queue is full, oldest is dropped and on_overflow fires once."""
    overflow_count = [0]

    def on_overflow() -> None:
        overflow_count[0] += 1

    src = MeetingAudioSource(uri="t", maxsize=2, on_overflow=on_overflow)

    src.push_pcm(np.array([1.0], dtype=np.float32))
    src.push_pcm(np.array([2.0], dtype=np.float32))
    src.push_pcm(np.array([3.0], dtype=np.float32))  # triggers overflow

    assert overflow_count[0] == 1

    received: List[float] = []
    src.stream.subscribe(
        on_next=lambda t: received.append(float(t.squeeze(0)[0].item()))
    )

    th = threading.Thread(target=src.read, daemon=True)
    th.start()

    src.close()
    th.join(timeout=1.0)

    # Oldest frame (1.0) should have been dropped; the queue should now
    # contain [2.0, 3.0] in FIFO order.
    assert received == [2.0, 3.0]


def test_overflow_callback_called_once_per_drop() -> None:
    """A multi-frame overflow burst triggers the callback once per drop."""
    overflow_count = [0]

    src = MeetingAudioSource(
        uri="t",
        maxsize=2,
        on_overflow=lambda: overflow_count.__setitem__(0, overflow_count[0] + 1),
    )

    for i in range(5):
        src.push_pcm(np.array([float(i)], dtype=np.float32))

    # First 2 fit; remaining 3 each trigger a drop.
    assert overflow_count[0] == 3

    src.close()


def test_close_exits_read_within_100ms() -> None:
    """close() unblocks read() promptly even with an empty queue."""
    src = MeetingAudioSource(uri="t")
    th = threading.Thread(target=src.read, daemon=True)
    th.start()

    # Give the thread a moment to enter queue.get().
    time.sleep(0.01)

    src.close()
    th.join(timeout=0.1)

    assert not th.is_alive()


def test_close_is_idempotent() -> None:
    """Calling close() multiple times is safe and does not raise."""
    src = MeetingAudioSource(uri="t")
    th = threading.Thread(target=src.read, daemon=True)
    th.start()

    src.close()
    src.close()  # second call must be a no-op
    src.close()

    th.join(timeout=0.5)
    assert not th.is_alive()


def test_no_overflow_callback_when_unset() -> None:
    """Overflow without a callback configured does not raise."""
    src = MeetingAudioSource(uri="t", maxsize=1)
    src.push_pcm(np.array([1.0], dtype=np.float32))
    # This must not raise even though on_overflow is None.
    src.push_pcm(np.array([2.0], dtype=np.float32))
    src.close()


def test_on_frame_called_once_per_consumed_frame_never_for_drops() -> None:
    """on_frame fires once per CONSUMED frame, in consumption order, with the
    exact pushed arrays — and never for frames dropped on overflow."""
    frames_in = [np.full(4, float(i), dtype=np.float32) for i in range(5)]
    seen: List[np.ndarray] = []
    src = MeetingAudioSource(uri="t", maxsize=2, on_frame=seen.append)

    # Push all 5 with no reader running: maxsize=2 -> frames 0..2 dropped.
    for pcm in frames_in:
        src.push_pcm(pcm)

    # on_frame is a consumer-side callback: nothing fires on the producer side.
    assert seen == []

    received: List[float] = []
    src.stream.subscribe(
        on_next=lambda t: received.append(float(t.squeeze(0)[0].item()))
    )
    th = threading.Thread(target=src.read, daemon=True)
    th.start()
    src.close()
    th.join(timeout=1.0)
    assert not th.is_alive()

    # Only the surviving frames (3, 4), consumption order, sample-exact.
    assert len(seen) == 2
    assert np.array_equal(seen[0], frames_in[3])
    assert np.array_equal(seen[1], frames_in[4])
    assert received == [3.0, 4.0]


def test_on_frame_invoked_before_stream_emission() -> None:
    """Each on_frame call happens strictly before the matching stream.on_next,
    so downstream prediction code sees a ring that already holds the frame."""
    events: List[tuple] = []
    src = MeetingAudioSource(
        uri="t", on_frame=lambda pcm: events.append(("frame", float(pcm[0])))
    )
    src.stream.subscribe(
        on_next=lambda t: events.append(("next", float(t.squeeze(0)[0].item())))
    )

    th = threading.Thread(target=src.read, daemon=True)
    th.start()

    for i in range(3):
        src.push_pcm(np.full(4, float(i), dtype=np.float32))

    src.close()
    th.join(timeout=1.0)
    assert not th.is_alive()

    assert events == [
        ("frame", 0.0), ("next", 0.0),
        ("frame", 1.0), ("next", 1.0),
        ("frame", 2.0), ("next", 2.0),
    ]


def test_on_frame_default_none_keeps_behaviour() -> None:
    """on_frame=None (the default) consumes frames normally and never raises."""
    src = MeetingAudioSource(uri="t")
    received: List[float] = []
    src.stream.subscribe(
        on_next=lambda t: received.append(float(t.squeeze(0)[0].item()))
    )

    th = threading.Thread(target=src.read, daemon=True)
    th.start()

    src.push_pcm(np.array([1.0], dtype=np.float32))
    src.push_pcm(np.array([2.0], dtype=np.float32))

    src.close()
    th.join(timeout=1.0)

    assert not th.is_alive()
    assert received == [1.0, 2.0]
