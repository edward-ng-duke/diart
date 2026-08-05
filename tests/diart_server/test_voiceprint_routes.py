"""Tests for the voiceprint enrollment HTTP endpoints.

Uses FastAPI TestClient (no lifespan) so no real models are needed.
`make_embedder` and `extract` are monkeypatched at module scope so the
router logic can be exercised without GPU / model files.
"""
import io
import wave
import numpy as np
from fastapi.testclient import TestClient


def _pcm_to_wav(pcm, sr=16000):
    int16 = (np.clip(pcm, -1.0, 1.0) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(int16.tobytes())
    return buf.getvalue()


def _wav_bytes(seconds=2.0, sr=16000):
    # A tone, not zeros: all-silence clips are now rejected by the
    # route-level trim (docs/bugs/2026-08-05-enroll-channel-mismatch.md §4.2).
    t = np.arange(int(seconds * sr)) / sr
    return _pcm_to_wav(0.3 * np.sin(2 * np.pi * 440 * t), sr)


def test_enroll_list_delete(monkeypatch, tmp_path):
    # Patch the embedder so no real model is needed.
    import diart_server.voiceprint_routes as vr

    monkeypatch.setattr(vr, "make_embedder", lambda *a, **k: object())
    monkeypatch.setattr(vr, "extract", lambda pcm, emb: np.array([1.0, 0.0], dtype=np.float32))

    from diart_server.voiceprint_store import VoiceprintStore
    from diart_server.app import create_app

    app = create_app()
    app.state.store = VoiceprintStore(tmp_path)
    client = TestClient(app)

    r = client.post(
        "/v1/voiceprints",
        data={"name": "张三", "embedding_model": "speechbrain/ecapa"},
        files={"file": ("a.wav", _wav_bytes(), "audio/wav")},
    )
    assert r.status_code == 200 and r.json()["name"] == "张三"

    r = client.get("/v1/voiceprints", params={"embedding_model": "speechbrain/ecapa"})
    assert r.json()["voiceprints"] == [{"name": "张三", "n_clips": 1}]

    r = client.delete("/v1/voiceprints/张三", params={"embedding_model": "speechbrain/ecapa"})
    assert r.status_code == 200
    assert client.get("/v1/voiceprints", params={"embedding_model": "speechbrain/ecapa"}).json()["voiceprints"] == []


def test_enroll_model_load_failure_returns_400(monkeypatch, tmp_path):
    import diart_server.voiceprint_routes as vr

    def boom(*a, **k):
        raise RuntimeError("cannot load model")

    monkeypatch.setattr(vr, "make_embedder", boom)

    from diart_server.voiceprint_store import VoiceprintStore
    from diart_server.app import create_app

    app = create_app()
    app.state.store = VoiceprintStore(tmp_path)
    client = TestClient(app)
    r = client.post(
        "/v1/voiceprints",
        data={"name": "x", "embedding_model": "speechbrain/spkrec-ecapa-voxceleb"},
        files={"file": ("a.wav", _wav_bytes(), "audio/wav")},
    )
    assert r.status_code == 400
    assert "could not be used" in r.json()["detail"]


def test_enroll_passes_explicit_device_to_embedder(monkeypatch, tmp_path):
    """Guard for the shared-model CPU-move landmine
    (docs/bugs/2026-07-06-voiceprint-false-accept.md §3.2): the enrollment
    route must hand make_embedder an explicit device (settings.device), never
    None (which would move the SHARED registry model to CPU mid-session)."""
    import torch

    import diart_server.voiceprint_routes as vr

    seen = {}

    def fake_make_embedder(name, registry, device=None):  # noqa: ARG001
        seen["device"] = device
        return object()

    monkeypatch.setattr(vr, "make_embedder", fake_make_embedder)
    monkeypatch.setattr(vr, "extract", lambda pcm, emb: np.array([1.0, 0.0], dtype=np.float32))

    from diart_server.voiceprint_store import VoiceprintStore
    from diart_server.app import create_app

    app = create_app()
    app.state.store = VoiceprintStore(tmp_path)
    client = TestClient(app)
    r = client.post(
        "/v1/voiceprints",
        data={"name": "y", "embedding_model": "speechbrain/ecapa"},
        files={"file": ("a.wav", _wav_bytes(), "audio/wav")},
    )
    assert r.status_code == 200
    assert seen["device"] == torch.device(vr.settings.device)


def test_enroll_trims_leading_trailing_silence(monkeypatch, tmp_path):
    """Route-level trim (docs/bugs/2026-08-05-enroll-channel-mismatch.md §4.2):
    leading/trailing silence must be cut before the clip reaches extract —
    silence dilutes the utterance embedding and drags genuine scores below
    the match threshold."""
    import diart_server.voiceprint_routes as vr

    captured = {}

    def fake_extract(pcm, emb):
        captured["pcm"] = pcm
        return np.array([1.0, 0.0], dtype=np.float32)

    monkeypatch.setattr(vr, "make_embedder", lambda *a, **k: object())
    monkeypatch.setattr(vr, "extract", fake_extract)

    from diart_server.voiceprint_store import VoiceprintStore
    from diart_server.app import create_app

    app = create_app()
    app.state.store = VoiceprintStore(tmp_path)
    client = TestClient(app)

    sr = 16000
    t = np.arange(2 * sr) / sr
    pcm = np.concatenate([np.zeros(sr), 0.3 * np.sin(2 * np.pi * 440 * t), np.zeros(sr)])
    r = client.post(
        "/v1/voiceprints",
        data={"name": "修剪", "embedding_model": "speechbrain/ecapa"},
        files={"file": ("a.wav", _pcm_to_wav(pcm, sr), "audio/wav")},
    )
    assert r.status_code == 200
    assert 1.9 * sr <= captured["pcm"].size <= 2.2 * sr


def test_enroll_all_silence_rejected_400(monkeypatch, tmp_path):
    """An all-silence clip must be rejected with HTTP 400 (it used to enroll
    successfully and poison the reference vector); extract is never called."""
    import diart_server.voiceprint_routes as vr

    called = {"extract": False}

    def fake_extract(pcm, emb):
        called["extract"] = True
        return np.array([1.0, 0.0], dtype=np.float32)

    monkeypatch.setattr(vr, "make_embedder", lambda *a, **k: object())
    monkeypatch.setattr(vr, "extract", fake_extract)

    from diart_server.voiceprint_store import VoiceprintStore
    from diart_server.app import create_app

    app = create_app()
    app.state.store = VoiceprintStore(tmp_path)
    client = TestClient(app)
    r = client.post(
        "/v1/voiceprints",
        data={"name": "静音", "embedding_model": "speechbrain/ecapa"},
        files={"file": ("a.wav", _pcm_to_wav(np.zeros(4 * 16000)), "audio/wav")},
    )
    assert r.status_code == 400
    assert "说话时间不足" in r.json()["detail"]
    assert called["extract"] is False
