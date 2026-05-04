"""HTTP probe routes and the WebSocket streaming route for diart_server.

This module exposes liveness, readiness, info, and Prometheus metrics
endpoints, plus the ``/v1/diarize/stream`` WebSocket route. The probes
are grouped under :data:`probes_router` and the WebSocket route under
:data:`ws_router` so the application factory in :mod:`diart_server.app`
can mount them independently.

See STANDALONE_SERVER.md sections 5.7 (probes), 5.8 (streaming), and
5.9 (metrics).
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import ValidationError

from diart_server import audio_decoder, metrics
from diart_server.config import settings
from diart_server.errors import BadAudioError, ModelNotFoundError, ResampleFailedError
from diart_server.protocol import QueryParams
from diart_server.session import DiarizationSession

probes_router = APIRouter()
ws_router = APIRouter()


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


@ws_router.websocket("/v1/diarize/stream")
async def stream(websocket: WebSocket) -> None:
    """Bidirectional WebSocket route for streaming diarization.

    Lifecycle:
      1. Accept the connection.
      2. Validate query parameters; reject with ``BAD_AUDIO`` (1008) on failure.
      3. Capacity check against ``settings.max_concurrent_sessions``;
         reject with ``SESSION_CAPACITY_FULL`` (1013) when full.
      4. Construct a :class:`DiarizationSession`. Pipeline-not-implemented
         errors are surfaced as ``BAD_AUDIO``; missing models as
         ``MODEL_NOT_FOUND``.
      5. Start the session and emit the ``ready`` message.
      6. Iterate inbound binary frames, decoding/resampling each to 16 kHz
         mono PCM and forwarding to the session. Oversized frames close
         the socket with ``RATE_LIMIT_EXCEEDED`` (1009). Decode errors are
         reported but do not terminate the stream.
      7. On loop exit (either client end-of-stream or disconnect), close
         the session and best-effort send the final summary message.

    Text frames (``{"type":"end"}``) are out of scope for T17/T18 and are
    silently filtered by ``websocket.iter_bytes()``; T22 introduces the
    explicit end-of-stream control message.
    """
    await websocket.accept()

    try:
        params = QueryParams.model_validate(dict(websocket.query_params))
    except ValidationError as exc:
        await websocket.send_json(
            {"type": "error", "code": "BAD_AUDIO", "message": str(exc)}
        )
        await websocket.close(code=1008)
        return

    if metrics.active_sessions._value.get() >= settings.max_concurrent_sessions:
        await websocket.send_json(
            {
                "type": "error",
                "code": "SESSION_CAPACITY_FULL",
                "message": "server at capacity",
            }
        )
        await websocket.close(code=1013)
        return

    try:
        session = DiarizationSession(
            params=params,
            registry=websocket.app.state.registry,
            on_message=websocket.send_json,
            loop=asyncio.get_running_loop(),
        )
    except NotImplementedError as exc:
        await websocket.send_json(
            {"type": "error", "code": "BAD_AUDIO", "message": str(exc)}
        )
        await websocket.close(code=1008)
        return
    except ModelNotFoundError as exc:
        await websocket.send_json(
            {"type": "error", "code": "MODEL_NOT_FOUND", "message": str(exc)}
        )
        await websocket.close(code=1008)
        return

    session.start()
    await websocket.send_json(session.ready_message())

    try:
        async for raw in websocket.iter_bytes():
            if len(raw) > settings.max_frame_bytes:
                await websocket.send_json(
                    {
                        "type": "error",
                        "code": "RATE_LIMIT_EXCEEDED",
                        "message": "frame too large",
                    }
                )
                await websocket.close(code=1009)
                break
            try:
                pcm = audio_decoder.to_pcm_16k_mono(raw, params.sample_rate)
            except (BadAudioError, ResampleFailedError) as exc:
                await websocket.send_json(exc.to_message())
                continue
            session.push_pcm(pcm)
    except WebSocketDisconnect:
        pass
    finally:
        try:
            final = await session.close()
            await websocket.send_json(final)
        except Exception:
            # Best effort: client may already be gone, or close() may
            # raise after a torn-down connection. Either way, don't
            # mask the original exit reason.
            pass


__all__ = ["probes_router", "ws_router"]
