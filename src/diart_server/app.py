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

from diart_server.config import settings
from diart_server.models_registry import ModelsRegistry
from diart_server.session_manager import manager
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

    yield

    logger.info("diart_server shutting down; draining active sessions")
    await manager.close_all(timeout=10.0)


app = FastAPI(title="diart-server", version="0.1.0", lifespan=lifespan)
app.include_router(probes_router)
app.include_router(ws_router)
