from diart_server.config import Settings


def test_new_settings_have_expected_defaults():
    s = Settings()
    assert str(s.asr_base_url) == "http://10.0.0.32:6001/v1"
    assert s.asr_model == "models/Qwen3-ASR-1.7B"
    assert s.asr_api_key == "EMPTY"
    assert s.asr_timeout == 60
    assert s.match_threshold == 0.6
    assert s.silence_gap == 0.7
    # Lowered 20.0 -> 12.0 for BUG1 (long-monologue latency defense-in-depth).
    assert s.max_turn_seconds == 12.0
    assert s.ring_buffer_seconds == 120.0
    assert s.max_asr_concurrency == 2
    assert str(s.voiceprints_dir) == "voiceprints"
    # New knobs for the 2026-07-02 turn/ASR quality fixes.
    assert s.partial_asr_interval == 5.0
    assert s.overlap_max_ratio == 0.5
    assert s.absorb_gap == 1.0
