"""BUG1: TurnAggregator.due_partials fires every `interval` of new audio.

Feed a single speaker in 0.5s steps up to 15.0s with interval=5.0 and collect
partials after each feed. Assert we see at least 2 partials before any
finalization, all sharing ONE turn_id.
"""
from diart_server.turn_aggregator import TurnAggregator


def test_due_partials_cadence_and_stable_turn_id():
    agg = TurnAggregator(silence_gap=0.7, max_turn_seconds=100.0)
    interval = 5.0

    partials_seen = []
    finalized_seen = []
    end = 0.0
    while end < 15.0:
        start = end
        end = round(start + 0.5, 3)
        finalized_seen.extend(agg.feed("s0", start, end, now=end))
        partials_seen.extend(agg.due_partials(interval))

    # No finalization for a continuous single-speaker monologue below max_turn.
    assert finalized_seen == []

    # Partials fired at ~5.0, ~10.0, ~15.0 -> at least 2 before any finalization.
    assert len(partials_seen) >= 2, partials_seen
    ends = [p.end for p in partials_seen]
    # First partial's end should be the first step where last_end reaches >= 5.0.
    assert min(ends) >= 5.0 - 1e-9
    assert max(ends) >= 10.0 - 1e-9

    # All partials belong to the SAME open turn.
    assert len({p.turn_id for p in partials_seen}) == 1
    # And that turn_id is non-empty.
    assert all(p.turn_id for p in partials_seen)


def test_due_partials_interval_zero_disabled():
    agg = TurnAggregator()
    agg.feed("s0", 0.0, 6.0, now=6.0)
    assert agg.due_partials(0.0) == []
