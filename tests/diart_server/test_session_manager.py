"""Tests for :class:`diart_server.session_manager.SessionManager`.

The manager is concurrency-sensitive (atomic capacity check + register)
but does not depend on a real :class:`DiarizationSession`; we stub the
session interface with a minimal object exposing ``id`` and an
awaitable ``close``. This keeps the tests fast and avoids pulling
torch/diart into the unit test path.

Tests are written as plain functions that drive their async work through
:func:`asyncio.run` so they don't depend on ``pytest-asyncio`` (the repo
does not declare it as a test dep).
"""

from __future__ import annotations

import asyncio

from diart_server.session_manager import SessionManager


class _FakeSession:
    """Minimal stand-in for :class:`DiarizationSession`.

    The manager only touches ``session.id`` (read) and ``await
    session.close()``, so this is the entire surface area we need.
    """

    def __init__(self, session_id: str) -> None:
        self.id = session_id
        self.close_calls = 0
        self.close_should_raise: BaseException | None = None

    async def close(self) -> dict:
        self.close_calls += 1
        if self.close_should_raise is not None:
            raise self.close_should_raise
        return {"type": "final"}


def test_has_capacity_initially_true() -> None:
    async def _run() -> None:
        mgr = SessionManager(max_concurrent=2)
        assert await mgr.has_capacity() is True

    asyncio.run(_run())


def test_has_capacity_false_at_limit() -> None:
    async def _run() -> None:
        mgr = SessionManager(max_concurrent=2)
        assert await mgr.register(_FakeSession("a")) is True
        assert await mgr.register(_FakeSession("b")) is True
        assert await mgr.has_capacity() is False

    asyncio.run(_run())


def test_register_at_limit_returns_false() -> None:
    async def _run() -> None:
        mgr = SessionManager(max_concurrent=1)
        assert await mgr.register(_FakeSession("a")) is True
        # Second registration must be rejected without modifying state.
        assert await mgr.register(_FakeSession("b")) is False
        # The first session is still tracked; capacity remains exhausted.
        assert await mgr.has_capacity() is False

    asyncio.run(_run())


def test_unregister_then_register_again() -> None:
    async def _run() -> None:
        mgr = SessionManager(max_concurrent=1)
        s1 = _FakeSession("a")
        assert await mgr.register(s1) is True
        assert await mgr.has_capacity() is False

        await mgr.unregister(s1.id)
        assert await mgr.has_capacity() is True

        s2 = _FakeSession("b")
        assert await mgr.register(s2) is True

    asyncio.run(_run())


def test_unregister_unknown_id_is_noop() -> None:
    async def _run() -> None:
        mgr = SessionManager(max_concurrent=1)
        # Must not raise; nothing to remove.
        await mgr.unregister("never-registered")
        assert await mgr.has_capacity() is True

    asyncio.run(_run())


def test_close_all_drains_all_sessions() -> None:
    async def _run() -> None:
        mgr = SessionManager(max_concurrent=4)
        sessions = [_FakeSession(f"s{i}") for i in range(3)]
        for s in sessions:
            assert await mgr.register(s) is True

        await mgr.close_all(timeout=2.0)

        # Each session's close was invoked exactly once.
        for s in sessions:
            assert s.close_calls == 1

        # After draining, the manager has full capacity again.
        assert await mgr.has_capacity() is True
        # And register-at-capacity arithmetic works against the new empty state.
        assert await mgr.register(_FakeSession("fresh")) is True

    asyncio.run(_run())


def test_close_all_empty_is_noop() -> None:
    async def _run() -> None:
        mgr = SessionManager(max_concurrent=1)
        # Must not raise and must return promptly.
        await asyncio.wait_for(mgr.close_all(timeout=1.0), timeout=2.0)

    asyncio.run(_run())


def test_close_all_swallows_session_close_errors() -> None:
    """A failing ``close()`` must not stop other sessions from draining,
    and the failing session must still be removed from the live set so the
    slot is recovered."""

    async def _run() -> None:
        mgr = SessionManager(max_concurrent=4)
        bad = _FakeSession("bad")
        bad.close_should_raise = RuntimeError("boom")
        good = _FakeSession("good")
        assert await mgr.register(bad) is True
        assert await mgr.register(good) is True

        await mgr.close_all(timeout=2.0)

        assert good.close_calls == 1
        assert bad.close_calls == 1
        # Both slots are released even though one close() raised.
        assert await mgr.has_capacity() is True

    asyncio.run(_run())


def test_concurrent_register_respects_limit() -> None:
    """Many concurrent register() calls must never overshoot the limit."""

    async def _run() -> None:
        mgr = SessionManager(max_concurrent=3)
        sessions = [_FakeSession(f"s{i}") for i in range(20)]
        results = await asyncio.gather(*(mgr.register(s) for s in sessions))
        assert sum(results) == 3
        assert await mgr.has_capacity() is False

    asyncio.run(_run())
