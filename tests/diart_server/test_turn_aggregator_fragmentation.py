"""BUG3: a brief interjection no longer chops the dominant speaker's turn.

Reproduces the screenshot pattern: spk0 speaks a long stretch while spk1
interjects once for 0.5s. Under the new silence-driven (diart-clock) rules,
spk0's turn is NOT chopped by spk1's single interjection, so on flush we get
exactly one long spk0 turn plus one short spk1 turn (<= 2 total, vs. the old
eager-switch code's >= 4).
"""
from diart_server.turn_aggregator import TurnAggregator


def test_interjection_does_not_fragment_dominant_speaker():
    agg = TurnAggregator(silence_gap=0.7, max_turn_seconds=100.0)

    all_turns = []
    # spk0 0.0-1.0, spk1 1.0-1.5 (interjection), spk0 1.5-6.0.
    all_turns += agg.feed("s0", 0.0, 0.5, now=0.5)
    all_turns += agg.feed("s0", 0.5, 1.0, now=1.0)
    all_turns += agg.feed("s1", 1.0, 1.5, now=1.5)  # brief interjection
    end = 1.5
    while end < 6.0:
        start = end
        end = round(start + 0.5, 3)
        all_turns += agg.feed("s0", start, end, now=end)

    all_turns += agg.flush(now=6.0)

    assert len(all_turns) <= 2, [(t.speaker, t.start, t.end) for t in all_turns]
    # A spk0 turn spanning >= 5.0s must exist (not fragmented).
    spk0 = [t for t in all_turns if t.speaker == "s0"]
    assert any((t.end - t.start) >= 5.0 for t in spk0), \
        [(t.speaker, t.start, t.end) for t in all_turns]
