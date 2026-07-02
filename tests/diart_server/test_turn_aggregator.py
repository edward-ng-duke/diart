from diart_server.turn_aggregator import TurnAggregator, FinalizedTurn


def _triple(ft: FinalizedTurn) -> tuple[str, float, float]:
    """(speaker, start, end) — ignores the now-present uuid turn_id."""
    return (ft.speaker, ft.start, ft.end)


def test_same_speaker_extends_without_finalizing():
    agg = TurnAggregator(silence_gap=0.7, max_turn_seconds=20)
    assert agg.feed("s0", 0.0, 0.5, now=0.5) == []
    assert agg.feed("s0", 0.5, 1.0, now=1.0) == []
    out = agg.flush(now=1.0)
    assert [_triple(f) for f in out] == [("s0", 0.0, 1.0)]


def test_silence_gap_finalizes():
    agg = TurnAggregator(silence_gap=0.7, max_turn_seconds=20)
    agg.feed("s0", 0.0, 1.0, now=1.0)
    # Under the diart-clock rule the gap is measured from _diart_now (= max end
    # ever = 2.5 after this feed): 2.5 - 1.0 = 1.5 > 0.7 -> finalize old.
    out = agg.feed("s0", 2.0, 2.5, now=2.0)
    assert ("s0", 0.0, 1.0) in [_triple(f) for f in out]


def test_speaker_switch_does_not_finalize_previous():
    """NEW semantics (BUG3): a speaker switch alone no longer finalizes.

    Finalization is silence-driven on diart's own timeline; a brief switch
    (no silence gap) leaves the previous speaker's turn OPEN. This replaces
    the old ``test_speaker_switch_finalizes_previous`` (eager switch finalize).
    """
    agg = TurnAggregator(silence_gap=0.7, max_turn_seconds=20)
    agg.feed("s0", 0.0, 1.0, now=1.0)
    out = agg.feed("s1", 1.0, 1.5, now=1.5)  # switch, but diart_now-1.0 = 0.5 < 0.7
    assert out == []  # nothing finalized on a pure switch

    # A later silence gap DOES finalize s0 via the unified silence rule: after
    # feeding s1 up to 2.5, diart_now=2.5, 2.5 - 1.0 (s0 last_end) = 1.5 > 0.7.
    out2 = agg.feed("s1", 2.0, 2.5, now=2.5)
    assert ("s0", 0.0, 1.0) in [_triple(f) for f in out2]


def test_max_turn_finalizes():
    agg = TurnAggregator(silence_gap=5.0, max_turn_seconds=2.0)
    agg.feed("s0", 0.0, 1.0, now=1.0)
    out = agg.feed("s0", 1.0, 2.5, now=2.5)  # length 2.5 > 2.0
    assert ("s0", 0.0, 2.5) in [_triple(f) for f in out]
