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


def test_live_page_no_similarity_hint():
    """Unconfirmed lines render a NEUTRAL label — no name, no score, no ≈guess
    (docs/superpowers/plans/2026-08-06-voiceprint-selection-plan.md T1).

    REPLACES ``test_live_page_identity_continuity_hint_static_asserts``, which
    pinned exactly the behaviour this task removes; its four overturned
    assertions were ``"≈" in r.text``, ``"lastVerified" in r.text``,
    ``"m.cand_name" in r.text`` and ``"m.cand_score.toFixed(2)" in r.text``
    (plus the ``.hint{`` style rule that existed only to render them).
    Basis: measured, the ≈name? hint appeared on 100% of unregistered-speaker
    turns and was 0% correct (plan §3 ⑤), which makes it the single structural
    failure source for acceptance criterion V3 — "an unregistered speaker
    talking for 2 minutes must never see a registered name on screen"
    (plan §2). The server is unchanged: cand_name/cand_score are still sent,
    the page just refuses to render them."""
    client = TestClient(create_app())
    r = client.get("/demo/live")
    assert r.status_code == 200
    assert "≈" not in r.text
    assert "lastVerified" not in r.text
    assert "cand_name" not in r.text
    assert "cand_score" not in r.text
    assert "说话人" in r.text  # the neutral label
    # The unconfirmed branch of lineHTML must leak neither a name nor a score:
    # the server still ships `score` with `name=None` (session.py:721).
    line_html = r.text.split("function lineHTML(m){", 1)[1]
    unconfirmed = line_html.split("} else {", 1)[1].split("\n  }", 1)[0]
    code = "\n".join(
        ln for ln in unconfirmed.splitlines() if not ln.strip().startswith("//")
    )
    assert "neutralLabel(m.speaker)" in code
    assert "name" not in code
    assert "score" not in code


def test_live_page_keeps_turn_replacement_and_overlap_badge():
    """T1 is display-only surgery on the identity render
    (docs/superpowers/plans/2026-08-06-voiceprint-selection-plan.md T1): the
    turn_id-keyed partial->final in-place replacement (with its stale-partial
    guard) and the ⑂重叠 badge must survive it untouched."""
    client = TestClient(create_app())
    r = client.get("/demo/live")
    assert r.status_code == 200
    assert "turnLines" in r.text
    assert "turnLines.set(tid, div)" in r.text
    assert "if(m.partial && !div.classList.contains('partial')) return;" in r.text
    assert "⑂重叠" in r.text
    assert "m.overlapped" in r.text


def test_live_page_dropdown_references_voiceprint_ready():
    """Cohort-missing fix (docs/bugs/2026-08-05-cohort-missing-threshold.md
    §4.3): the embedding dropdown must be filtered on /v1/info's
    voiceprint_ready field so cohort-less models are hidden."""
    client = TestClient(create_app())
    r = client.get("/demo/live")
    assert r.status_code == 200
    assert "voiceprint_ready" in r.text
