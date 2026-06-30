"""FastAPI application entry point for diart_server.

This module exposes a FastAPI app whose lifespan eagerly loads the
default segmentation/embedding models at startup so the first
WebSocket request is not penalised by a cold cache. Routers are
attached by T16 (probes) and T17 (websocket); T20 wires the
SessionManager drain into shutdown so in-flight sessions get a
chance to flush their final summaries.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from diart_server.asr_client import AsrClient
from diart_server.config import settings
from diart_server.models_registry import ModelsRegistry
from diart_server.session_manager import manager
from diart_server.voiceprint_routes import voiceprint_router
from diart_server.voiceprint_store import VoiceprintStore
from diart_server.ws_routes import probes_router, ws_router

logger = logging.getLogger(__name__)

DEFAULT_SEGMENTATION = "pyannote/segmentation-3.0"
DEFAULT_EMBEDDING = "pyannote/embedding"


def _resolve_manifest_path() -> Path:
    """Find ``models.json`` in priority order.

    1. ``$DIART_MODELS_MANIFEST`` (explicit override)
    2. ``settings.models_dir / "models.json"`` (the docker layout: image
       bakes the manifest next to the model files)
    3. ``<repo-root>/docker/models.json`` (the dev layout: src/diart_server
       lives at repo-root/src/diart_server, so parent.parent.parent is the
       repo root). Only valid for editable / source-tree runs.

    The first candidate that exists is returned. If none exist, the docker
    layout candidate is returned anyway so the resulting error message
    points at the conventional location.
    """
    import os
    explicit = os.environ.get("DIART_MODELS_MANIFEST")
    if explicit:
        return Path(explicit)
    candidates = [
        Path(settings.models_dir) / "models.json",
        Path(__file__).resolve().parent.parent.parent / "docker" / "models.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


MANIFEST_PATH = _resolve_manifest_path()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up models at startup; drain live sessions on shutdown."""
    logging.basicConfig(level=settings.log_level)
    logger.info(
        "diart_server starting on %s:%d device=%s models_dir=%s",
        settings.host,
        settings.port,
        settings.device,
        settings.models_dir,
    )
    registry = ModelsRegistry(settings.models_dir, MANIFEST_PATH)
    app.state.registry = registry
    try:
        registry.warm_up(
            torch.device(settings.device),
            default_seg=DEFAULT_SEGMENTATION,
            default_emb=DEFAULT_EMBEDDING,
        )
        logger.info(
            "warm-up complete (defaults: %s + %s)",
            DEFAULT_SEGMENTATION,
            DEFAULT_EMBEDDING,
        )
        app.state.ready = True
    except Exception:
        # Warm-up failure leaves the app importable but /readyz will return 503.
        # Logged for diagnosis. T15 verification (DIART_MODELS_DIR=/tmp/missing)
        # would hit this branch if lifespan ran during import.
        logger.exception("warm-up failed; app will report not ready")
        app.state.ready = False

    # Always wire store + ASR client regardless of warm-up success.
    store = VoiceprintStore(settings.voiceprints_dir)
    store.load()
    app.state.store = store
    app.state.asr_client = AsrClient(
        settings.asr_base_url, settings.asr_model, settings.asr_api_key, settings.asr_timeout
    )

    yield

    logger.info("diart_server shutting down; draining active sessions")
    await manager.close_all(timeout=10.0)


_STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns a fully-wired app instance: routers included, static files
    mounted, root redirect defined, and lifespan attached. Call this once;
    the module-level ``app`` is the canonical singleton.
    """
    application = FastAPI(title="diart-server", version="0.1.0", lifespan=lifespan)
    application.include_router(probes_router)
    application.include_router(ws_router)
    application.include_router(voiceprint_router)

    # /demo/live must be registered BEFORE the /demo StaticFiles mount so the
    # mount does not shadow it.
    _live_html = _STATIC_DIR / "live.html"

    @application.get("/demo/live", response_class=HTMLResponse, include_in_schema=False)
    async def demo_live() -> str:
        """Serve the real-time mic + voiceprint + ASR demo page."""
        return _live_html.read_text(encoding="utf-8")

    # Browser-side demo: upload a file → stream to /v1/diarize/stream → render
    # timeline + segments. Served from /demo/ (static SPA, single index.html).
    # The package's static dir is shipped via setup.cfg's package_data so it
    # works for both editable and installed (docker) layouts.
    if _STATIC_DIR.is_dir():
        application.mount(
            "/demo",
            StaticFiles(directory=str(_STATIC_DIR), html=True),
            name="demo",
        )

    @application.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        """Redirect bare visits to the browser demo."""
        return RedirectResponse(url="/demo/", status_code=307)

    return application


app = create_app()
