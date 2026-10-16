"""Shared LangGraph checkpointer for the Simon service.

The service boundary rebuilds an agent per request, and an ``InMemorySaver``
built inside ``build_agent()`` is therefore a brand-new, empty memory on every
turn. ``thread_id`` was decorative: nothing was ever recovered between
requests, so "compare the first with the second" could not work — the agent had
no idea what the first was.

``AsyncPostgresSaver`` is the documented production checkpointer for async
graphs (LangGraph persistence docs: ``InMemorySaver`` is dev-only,
``AsyncPostgresSaver`` is production). It is created once per process and handed
to every agent, so thread state survives restarts and is shared safely across
concurrent conversations — which the per-request ``InMemorySaver`` could not do.

Lifecycle: ``setup()`` creates the checkpoint tables and is safe to call on
every boot (idempotent). It is invoked from the FastAPI lifespan in
``server/main.py``; this module holds the singleton and the prune helper.

Connectivity note: the production Postgres host resolves to IPv6 only. From an
IPv4-only network the connection fails at the socket, not at auth — which is
why local development falls back to an in-memory checkpointer with a loud
warning rather than crashing the service. On Render the same DSN connects.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from simon.settings import get_pool_settings

if TYPE_CHECKING:  # pragma: no cover - typing only
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

LOGGER = logging.getLogger("simon.checkpointer")

# Checkpoints grow without bound; each super-step of every conversation writes
# one. A week of retention is generous for a concierge thread while keeping the
# table small enough that the prune stays cheap.
RETENTION_DAYS = 7

_checkpointer: "AsyncPostgresSaver | None" = None
_checkpointer_ctx = None
_fallback_reason: str | None = None


def is_persistent() -> bool:
    return _checkpointer is not None


def fallback_reason() -> str | None:
    return _fallback_reason


async def get_checkpointer():
    """Return the shared checkpointer, creating it on first use.

    Falls back to ``InMemorySaver`` when no DSN is configured or Postgres is
    unreachable, and records why — so the service still runs (degraded) and the
    reason surfaces in logs and in /api/health rather than being silent.

    Lifetime note: ``AsyncPostgresSaver.from_conn_string`` is a
    generator-based async context manager that opens a single connection and
    closes it on exit. A long-lived service needs that connection open for its
    whole lifetime, so the ``async with`` block is entered here and kept open —
    ``close_checkpointer()`` exits it on shutdown.
    """
    global _checkpointer, _checkpointer_ctx, _fallback_reason
    if _checkpointer is not None:
        return _checkpointer

    settings = get_pool_settings()
    raw = settings.postgres_dsn
    dsn = raw.get_secret_value() if hasattr(raw, "get_secret_value") else str(raw or "")

    if not dsn:
        _fallback_reason = "POSTGRES_DSN is not set"
        LOGGER.warning(
            "POSTGRES_DSN is not set — conversation memory is per-process "
            "only and will not survive a restart or be shared across workers."
        )
        return _build_memory()

    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    # Keep the connection scope alive for the process lifetime.
    ctx = AsyncPostgresSaver.from_conn_string(dsn)
    try:
        saver = await ctx.__aenter__()
        await saver.setup()
    except Exception as exc:  # noqa: BLE001 - any failure must degrade, not crash
        # The generator may have opened a connection before failing; unwind it.
        try:
            await ctx.__aexit__(type(exc), exc, exc.__traceback__)
        except Exception:  # noqa: BLE001
            pass
        _fallback_reason = f"postgres checkpointer unavailable: {type(exc).__name__}"
        LOGGER.warning(
            "Postgres checkpointer unavailable (%s); falling back to "
            "per-process memory. Thread state will not survive a restart.",
            type(exc).__name__,
        )
        return _build_memory()

    _checkpointer = saver
    _checkpointer_ctx = ctx
    _fallback_reason = None
    LOGGER.info("AsyncPostgresSaver ready (persistent thread memory)")
    return saver


def _build_memory():
    from langgraph.checkpoint.memory import InMemorySaver

    return InMemorySaver()


async def prune_old_checkpoints() -> int:
    """Drop checkpoints past the retention window using the built-in pruner.

    ``AsyncPostgresSaver.aprune`` is the supported API (hand-written DELETE
    against the checkpoint schema would break whenever LangGraph migrates it).
    Returns the number removed, or -1 when there is no persistent checkpointer.
    Never raises: a failed prune is a log line, not a service outage.
    """
    if _checkpointer is None:
        return -1
    try:
        removed = await _checkpointer.aprune()
        return int(removed or 0)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("checkpoint prune failed: %s", type(exc).__name__)
        return -1


async def close_checkpointer() -> None:
    """Exit the connection-pool context manager on shutdown."""
    global _checkpointer, _checkpointer_ctx
    if _checkpointer_ctx is not None:
        try:
            await _checkpointer_ctx.__aexit__(None, None, None)
        except Exception:  # noqa: BLE001 - shutdown must not raise
            pass
    _checkpointer = None
    _checkpointer_ctx = None
