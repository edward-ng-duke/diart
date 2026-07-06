"""Tests for the voiceprint enrollment HTTP endpoints.

Uses FastAPI TestClient (no lifespan) so no real models are needed.
`make_embedder` and `extract` are monkeypatched at module scope so the
router logic can be exercised without GPU / model files.
"""
import io
import wave
import numpy as np
from fastapi.testclient import TestClient


def _wav_bytes(seconds=1.0, sr=16000):
    int16 = (np.zeros(int(seconds * sr))).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(int16.tobytes())
    return buf.getvalue()


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
