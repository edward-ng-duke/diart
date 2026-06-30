from diart_server.config import Settings


def test_new_settings_have_expected_defaults():
    s = Settings()
    assert str(s.asr_base_url) == "http://10.0.0.32:6001/v1"
    assert s.asr_model == "models/Qwen3-ASR-1.7B"
    assert s.asr_api_key == "EMPTY"
    assert s.asr_timeout == 60
    assert s.match_threshold == 0.6
    assert s.silence_gap == 0.7
    assert s.max_turn_seconds == 20.0
    assert s.ring_buffer_seconds == 120.0
    assert s.max_asr_concurrency == 2
    assert str(s.voiceprints_dir) == "voiceprints"
