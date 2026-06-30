from diart_server.turn_aggregator import TurnAggregator, FinalizedTurn


def test_same_speaker_extends_without_finalizing():
    agg = TurnAggregator(silence_gap=0.7, max_turn_seconds=20)
    assert agg.feed("s0", 0.0, 0.5, now=0.5) == []
    assert agg.feed("s0", 0.5, 1.0, now=1.0) == []
    out = agg.flush(now=1.0)
    assert out == [FinalizedTurn("s0", 0.0, 1.0)]


def test_silence_gap_finalizes():
    agg = TurnAggregator(silence_gap=0.7, max_turn_seconds=20)
    agg.feed("s0", 0.0, 1.0, now=1.0)
    out = agg.feed("s0", 2.0, 2.5, now=2.0)  # 1.0s gap > 0.7 -> finalize old
    assert FinalizedTurn("s0", 0.0, 1.0) in out


def test_speaker_switch_finalizes_previous():
    agg = TurnAggregator(silence_gap=0.7, max_turn_seconds=20)
    agg.feed("s0", 0.0, 1.0, now=1.0)
    out = agg.feed("s1", 1.0, 1.5, now=1.5)  # s1 starts at/after s0 end
    assert FinalizedTurn("s0", 0.0, 1.0) in out


def test_max_turn_finalizes():
    agg = TurnAggregator(silence_gap=5.0, max_turn_seconds=2.0)
    agg.feed("s0", 0.0, 1.0, now=1.0)
    out = agg.feed("s0", 1.0, 2.5, now=2.5)  # length 2.5 > 2.0
    assert FinalizedTurn("s0", 0.0, 2.5) in out
