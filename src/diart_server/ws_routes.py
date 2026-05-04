"""HTTP probe routes for the diart server.

This module exposes liveness, readiness, info, and Prometheus metrics
endpoints. The WebSocket route lives in this module too (added by T17);
the probes are grouped together under :data:`probes_router` so the
application factory in :mod:`diart_server.app` can mount them
independently.

See STANDALONE_SERVER.md sections 5.7 (probes) and 5.9 (metrics).
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from diart_server import metrics
from diart_server.config import settings

probes_router = APIRouter()


@probes_router.get("/healthz")
async def healthz() -> dict[str, str]:
    """Liveness probe: always returns 200 once the process is up."""
    return {"status": "ok"}


@probes_router.get("/readyz")
async def readyz(request: Request) -> JSONResponse:
    """Readiness probe.

    Returns 503 if warm-up has not completed successfully or if the
    server is at its concurrent-session capacity. Otherwise returns
    200 with the current active-session count.
    """
    app = request.app
    if not getattr(app.state, "ready", False):
        return JSONResponse(
            {"status": "not_ready", "reason": "warm-up incomplete"},
            status_code=503,
        )
    active = metrics.active_sessions._value.get()
    if active >= settings.max_concurrent_sessions:
        return JSONResponse(
            {
                "status": "not_ready",
                "reason": "at capacity",
                "active_sessions": active,
            },
            status_code=503,
        )
    return JSONResponse({"status": "ready", "active_sessions": active})


@probes_router.get("/v1/info")
async def info(request: Request) -> dict:
    """Return server build/runtime information for clients to introspect."""
    registry = request.app.state.registry
    models = {name: spec.kind for name, spec in registry.specs.items()}
    return {
        "version": "0.1.0",
        "pipelines": ["diarization", "vad"],
        "models": models,
        "device": settings.device,
        "max_concurrent_sessions": settings.max_concurrent_sessions,
    }


@probes_router.get("/metrics")
async def metrics_endpoint() -> Response:
    """Expose Prometheus metrics in the default text exposition format."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


__all__ = ["probes_router"]
