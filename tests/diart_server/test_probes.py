"""Tests for the HTTP probe endpoints in diart_server.ws_routes.

These tests use FastAPI's synchronous TestClient and run the real
application lifespan. The lifespan attempts to warm up models from
``settings.models_dir`` (defaults to ``/app/models`` which is absent
in test environments); the warm-up failure is caught by the lifespan
itself, leaving ``app.state.ready=False``. Tests therefore tolerate
both ``200`` and ``503`` for ``/readyz`` so they pass whether or not
the local environment happens to have a populated models directory.
"""

from __future__ import annotations

from fastapi.testclient import TestClient
from prometheus_client import CONTENT_TYPE_LATEST

from diart_server import metrics
from diart_server.app import app
from diart_server.config import settings


def test_healthz_returns_ok():
    with TestClient(app) as client:
        response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readyz_status_code_is_200_or_503():
    """/readyz returns 200 when warm-up succeeded, 503 otherwise.

    In CI without a models directory the lifespan catches the warm-up
    failure and sets ``app.state.ready=False`` -> 503. If a developer
    runs locally with models present the same endpoint returns 200.
    Either is acceptable; the contract is "valid JSON + {200, 503}".
    """
    with TestClient(app) as client:
        response = client.get("/readyz")
    assert response.status_code in (200, 503)
    body = response.json()
    assert "status" in body
    if response.status_code == 503:
        assert body["status"] == "not_ready"
        assert "reason" in body
    else:
        assert body["status"] == "ready"
        assert "active_sessions" in body


def test_readyz_503_when_not_ready():
    """Forcing ``app.state.ready=False`` after lifespan must yield 503."""
    with TestClient(app) as client:
        # Override after lifespan startup so we exercise the not-ready branch
        # regardless of whether warm-up succeeded.
        app.state.ready = False
        response = client.get("/readyz")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["reason"] == "warm-up incomplete"


def test_readyz_503_when_at_capacity(monkeypatch):
    """When active_sessions >= max_concurrent_sessions, /readyz must 503."""
    with TestClient(app) as client:
        # Ensure ready=True so we reach the capacity check; then pin the
        # gauge above the configured cap.
        app.state.ready = True
        monkeypatch.setattr(
            metrics.active_sessions._value,
            "get",
            lambda: settings.max_concurrent_sessions,
        )
        response = client.get("/readyz")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["reason"] == "at capacity"
    assert body["active_sessions"] == settings.max_concurrent_sessions


def test_readyz_200_when_ready_and_under_capacity(monkeypatch):
    """When ready and below capacity, /readyz returns 200 with counters."""
    with TestClient(app) as client:
        app.state.ready = True
        monkeypatch.setattr(metrics.active_sessions._value, "get", lambda: 0)
        response = client.get("/readyz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["active_sessions"] == 0


def test_v1_info_returns_expected_shape():
    with TestClient(app) as client:
        response = client.get("/v1/info")
    assert response.status_code == 200
    body = response.json()
    assert body["version"] == "0.1.0"
    assert body["pipelines"] == ["diarization", "vad"]
    assert body["device"] == settings.device
    assert body["max_concurrent_sessions"] == settings.max_concurrent_sessions
    assert isinstance(body["models"], dict)
    # Manifest at docker/models.json contains 11 entries.
    assert len(body["models"]) == 11
    # Spot-check a couple of well-known names.
    assert body["models"]["pyannote/segmentation-3.0"] == "segmentation"
    assert body["models"]["pyannote/embedding"] == "embedding"


def test_v1_info_voiceprint_ready_lists_cohort_backed_models():
    """Only embedding models with a cohort .npy are voiceprint-ready
    (docs/bugs/2026-08-05-cohort-missing-threshold.md §4.2): the match
    threshold is calibrated for the mean-centered score, so cohort-less
    models must be advertised as unable to identify. Dev layout ships
    three cohort files (the cnceleb one added with the wespeaker_onnx
    backend, 2026-08-06); the field is sorted by construction."""
    with TestClient(app) as client:
        response = client.get("/v1/info")
    assert response.status_code == 200
    body = response.json()
    assert body["voiceprint_ready"] == [
        "pyannote/embedding",
        "pyannote/wespeaker-voxceleb-resnet34-LM",
        "wespeaker/cnceleb-resnet34-LM",
    ]


def test_metrics_exposes_prometheus_payload():
    with TestClient(app) as client:
        response = client.get("/metrics")
    assert response.status_code == 200
    # Content-type must come from prometheus_client so the exposition
    # version stays in sync with whatever the library negotiates.
    assert response.headers["content-type"] == CONTENT_TYPE_LATEST
    assert response.headers["content-type"].startswith("text/plain")
    # All gauges/counters defined in metrics.py should appear in the
    # exposition (even if value=0) so dashboards have a stable schema.
    text = response.text
    assert "diart_active_sessions" in text
    assert "diart_predictions_total" in text
    assert "diart_audio_bytes_received_total" in text
