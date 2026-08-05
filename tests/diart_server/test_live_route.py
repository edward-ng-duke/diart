from fastapi.testclient import TestClient
from diart_server.app import create_app


def test_live_page_served():
    client = TestClient(create_app())
    r = client.get("/demo/live")
    assert r.status_code == 200 and "diarize/stream" in r.text


def test_live_page_enrollment_channel_static_asserts():
    """Static guards for the enroll-channel-mismatch fix
    (docs/bugs/2026-08-05-enroll-channel-mismatch.md §4.3): enrollment must
    ship raw PCM through the exact capture chain the live path uses — no
    MediaRecorder/Opus, one getUserMedia call site, browser DSP off."""
    client = TestClient(create_app())
    r = client.get("/demo/live")
    assert r.status_code == 200
    assert "MediaRecorder" not in r.text
    assert r.text.count("getUserMedia(") == 1
    assert "getUserMedia({audio:MIC})" in r.text
    assert "echoCancellation:false" in r.text
    assert "noiseSuppression:false" in r.text
    assert "autoGainControl:false" in r.text
    assert "const ENROLL_MS = 6000;" in r.text
    assert "if(enrolling) return;" in r.text  # pause-during-enroll guard
    assert "n_clips" in r.text  # multi-clip feedback surfaced in the UI


def test_live_page_identity_continuity_hint_static_asserts():
    """Identity-continuity hint (docs/bugs/2026-08-05-identity-continuity-
    overlap-purity.md §4.3): the page must consume the prediction-stream
    identity echo and render nameless lines with an explicitly-uncertain
    ≈name? hint (display-only; the server stamps nothing it did not verify)."""
    client = TestClient(create_app())
    r = client.get("/demo/live")
    assert r.status_code == 200
    assert "type==='prediction'" in r.text
    assert ".hint{" in r.text
    assert "≈" in r.text
    assert "lastVerified" in r.text


def test_live_page_dropdown_references_voiceprint_ready():
    """Cohort-missing fix (docs/bugs/2026-08-05-cohort-missing-threshold.md
    §4.3): the embedding dropdown must be filtered on /v1/info's
    voiceprint_ready field so cohort-less models are hidden."""
    client = TestClient(create_app())
    r = client.get("/demo/live")
    assert r.status_code == 200
    assert "voiceprint_ready" in r.text
