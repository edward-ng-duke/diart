from diart_server.protocol import (
    PredictionMessage,
    TranscriptMessage,
    EnrollResponse,
    VoiceprintInfo,
)


def test_prediction_message_optional_identity_defaults_none():
    m = PredictionMessage(
        start=0.0, duration=1.0, speaker="speaker0",
        embedding=[0.1, 0.2], samples_count=3,
    )
    assert m.name is None and m.score is None


def test_transcript_message_roundtrip():
    m = TranscriptMessage(
        speaker="speaker0", name="张三", score=0.81,
        start=1.0, end=3.0, text="你好世界", lang="Chinese",
    )
    d = m.model_dump()
    assert d["type"] == "transcript" and d["name"] == "张三"


def test_enroll_response_and_info():
    r = EnrollResponse(name="张三", embedding_model="speechbrain/ecapa", n_clips=2)
    assert r.type == "enrolled"
    info = VoiceprintInfo(name="张三", n_clips=2)
    assert info.n_clips == 2
