"""Session manager for diart_server.

Tracks live :class:`~diart_server.session.DiarizationSession` instances
per process. Replaces the racy "read metric, then maybe accept" pattern
in :mod:`diart_server.ws_routes` with an atomic register/unregister
under an :class:`asyncio.Lock`.

The capacity bound is set at init from
:attr:`diart_server.config.Settings.max_concurrent_sessions`.

This is a process-level singleton: cross-container coordination is
explicitly out of scope (see PLAN.md task T20).
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from diart_server.config import settings

if TYPE_CHECKING:
    from diart_server.session import DiarizationSession

logger = logging.getLogger(__name__)


class SessionManager:
    """Track live diarization sessions with an atomic capacity bound.

    The manager exposes coroutine methods only; all state mutations are
    serialized through ``self._lock`` so the capacity check and the
    insertion are observed atomically by concurrent WebSocket handlers.
    """

    def __init__(self, max_concurrent: int) -> None:
        self._sessions: dict[str, "DiarizationSession"] = {}
        self._lock = asyncio.Lock()
        self._max = max_concurrent

    async def has_capacity(self) -> bool:
        """Return ``True`` iff there is at least one slot free.

        This is a non-binding read used by readiness probes; callers that
        actually want to take a slot must use :meth:`register`, which
        performs the check and the insertion atomically.
        """
        async with self._lock:
            return len(self._sessions) < self._max

    async def register(self, session: "DiarizationSession") -> bool:
        """Atomic capacity-check + register.

        Returns ``True`` if the session was registered, ``False`` if the
        manager is already at capacity (in which case the caller should
        reject the connection with ``SESSION_CAPACITY_FULL``).
        """
        async with self._lock:
            if len(self._sessions) >= self._max:
                return False
            self._sessions[session.id] = session
            return True

    async def unregister(self, session_id: str) -> None:
        """Remove ``session_id`` from the live set if present.

        Idempotent: unregistering an unknown id is a silent no-op so the
        WebSocket handler can call this unconditionally in its
        ``finally`` block.
        """
        async with self._lock:
            self._sessions.pop(session_id, None)

    async def close_all(self, timeout: float = 10.0) -> None:
        """Drain every active session in parallel, with a global timeout.

        Used by the FastAPI lifespan shutdown to give in-flight sessions
        a chance to flush their final messages before the process exits.
        Sessions that fail to close within ``timeout`` seconds are left
        behind; the process will tear them down on exit.
        """
        async with self._lock:
            sessions = list(self._sessions.values())
        if not sessions:
            return
        logger.info(
            "closing %d active session(s) with timeout=%.1fs",
            len(sessions),
            timeout,
        )
        await asyncio.wait(
            [asyncio.create_task(self._close_one(s)) for s in sessions],
            timeout=timeout,
        )

    async def _close_one(self, session: "DiarizationSession") -> None:
        try:
            await session.close()
        except Exception:
            logger.exception("error closing session %s", session.id)
        finally:
            async with self._lock:
                self._sessions.pop(session.id, None)


# Module-level singleton. Cross-container coordination is out of scope.
manager = SessionManager(settings.max_concurrent_sessions)


__all__ = ["SessionManager", "manager"]
