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

Connectivity note — how this actually failed in production, measured 2026-10-17,
and how it was diagnosed. Three separate faults were stacked, and each one hides
the next:

1. The deployed service defines ``DATABASE_URL`` and does NOT define
   ``POSTGRES_DSN``. Reading one hardcoded name therefore returned an empty
   string, and the "DSN is not set" branch burned the whole chain — the real
   Postgres coordinates were present in the environment the entire time.
   Fixed in ``PoolSettings.resolved_postgres_dsn``.

2. Supabase's DIRECT host (``db.<ref>.supabase.co``) has no A record on ANY plan:
   it resolves to IPv6 only (``2a05:d018:…``). An IPv4 address for it is a paid
   Pro-plan add-on. On an IPv4 network the socket fails before authentication,
   which surfaces as ``OperationalError: Network is unreachable`` and looks like
   a bad password rather than an address family.

3. The pooler (``<ref>.pooler.supabase.com`` / ``aws-0-<region>.pooler.supabase.com``)
   IS IPv4-reachable and is free on every plan, but its plain egress to a
   third-party host on 5432/6543 depends on the platform's outbound rules; a
   service whose HTTP egress works can still have its Postgres egress blocked.
   ``candidate_dsns()`` tries every reachable-looking option so the actual
   blocker is named in ``fallback_reason()`` instead of guessed at.

Order matters when reading a failure: check which variable is actually set,
then the address family, then the port's outbound reachability. Diagnosing from
the bottom up produces confident wrong answers — as it did here twice.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING
from urllib.parse import quote, urlparse

from simon.settings import get_pool_settings

if TYPE_CHECKING:  # pragma: no cover - typing only
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

LOGGER = logging.getLogger("simon.checkpointer")

# Checkpoints grow without bound; each super-step of every conversation writes
# one. A week of retention is generous for a concierge thread while keeping the
# table small enough that the prune stays cheap.
RETENTION_DAYS = 7

# Retries per DSN candidate. A cold Postgres (Render free-tier cold start, a
# pooler still waking) answers in a couple of seconds; three short attempts per
# candidate keeps the whole boot under ~10s while not giving up on a transient
# refusal. This is deliberately NOT a retry of the whole chain — each candidate
# gets its own attempts and the next candidate starts immediately after.
CONNECT_ATTEMPTS = 3
CONNECT_BACKOFF_SECONDS = 2.0

_checkpointer: "AsyncPostgresSaver | None" = None
_checkpointer_ctx = None
_fallback_reason: str | None = None
# Set once every candidate has failed, so a service that already knows it is
# degraded stops paying the connection cost on each request.
_fallback_locked = False


def is_persistent() -> bool:
    return _checkpointer is not None


def fallback_reason() -> str | None:
    return _fallback_reason


def _host_of(dsn: str) -> str:
    try:
        return urlparse(dsn).hostname or "?"
    except Exception:  # noqa: BLE001 - never let diagnostics break boot
        return "?"


def _host_port_of(dsn: str) -> str:
    try:
        u = urlparse(dsn)
        port = f":{u.port}" if u.port else ""
        return f"{u.hostname or '?'}{port}"
    except Exception:  # noqa: BLE001
        return "?"


def candidate_dsns(configured: str, settings) -> list[str]:
    """Every DSN worth trying, in order, with the configured one first.

    The fallback is derived from the Supabase pooler variables that already
    exist in the environment. It exists because a DSN that is merely
    *configured* is not the same as a DSN that is *reachable*: the deployed
    value points at an IPv6-only host, so it opened a socket only on an
    IPv6-capable network and failed everywhere else.

    Returns a de-duplicated list, so a correctly-configured pooler DSN yields
    exactly one candidate and costs nothing extra.
    """
    out: list[str] = []
    seen: set[str] = set()

    def add(dsn: str) -> None:
        dsn = (dsn or "").strip()
        if dsn and dsn not in seen:
            seen.add(dsn)
            out.append(dsn)

    add(configured)

    pooler_host = str(getattr(settings, "supabase_pooler_host", "") or "").strip()
    pooler_port = str(getattr(settings, "supabase_pooler_port", "") or "").strip()
    pooler_user = str(getattr(settings, "supabase_pooler_user", "") or "").strip()
    password = getattr(settings, "supabase_db_password", None)
    password = password.get_secret_value() if hasattr(password, "get_secret_value") else str(password or "")
    database = str(getattr(settings, "supabase_db_name", "") or "").strip() or "postgres"

    # Host + user are enough to form a candidate. A password is deliberately NOT
    # required: trust/peer/iam-authenticated clusters have none, and demanding
    # one silently dropped the fallback in exactly the case it exists for.
    if pooler_host and pooler_user:
        # A host that does not resolve from an IPv4 network is not usable, even
        # though it is the one the environment advertises. Supabase also answers
        # on the region-canonical pooler name, so try that too rather than
        # trusting a hostname nobody has verified.
        hosts = [pooler_host]
        region = str(getattr(settings, "supabase_project_region", "") or "").strip()
        if region:
            canonical = f"aws-0-{region}.pooler.supabase.com"
            if canonical != pooler_host:
                hosts.append(canonical)

        # Session mode (5432) first — AsyncPostgresSaver holds one connection and
        # runs setup DDL through it; transaction mode (6543) multiplexes and can
        # break session-level state.
        ports = [pooler_port or "5432", "5432", "6543"]
        # Dedupe ports PER HOST, not once globally: the first host would
        # otherwise consume 5432 and 6543 and leave the fallback host with
        # nothing to try, which is the whole reason it exists.
        for host in hosts:
            tried_ports: set[str] = set()
            for port in ports:
                port = str(port or "").strip()
                if not port or port in tried_ports:
                    continue
                tried_ports.add(port)
                credentials = quote(pooler_user, safe="")
                if password:
                    credentials += ":" + quote(password, safe="")
                add(
                    "postgresql://"
                    + credentials
                    + "@" + host + ":" + port + "/" + quote(database, safe="")
                )

    return out


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
    global _checkpointer, _checkpointer_ctx, _fallback_reason, _fallback_locked
    if _checkpointer is not None:
        return _checkpointer
    if _fallback_locked:
        # A previous resolve already exhausted every candidate this process;
        # don't pay the connection cost again on every request.
        return _build_memory()

    settings = get_pool_settings()
    # POSTGRES_DSN or DATABASE_URL — see PoolSettings.resolved_postgres_dsn. The
    # deployed service defines only DATABASE_URL, so reading one hardcoded name
    # is what made this look like an environment problem.
    configured = settings.resolved_postgres_dsn

    if not configured:
        _fallback_reason = "POSTGRES_DSN is not set and no pooler DSN could be derived"
        _fallback_locked = True
        LOGGER.warning(
            "%s — conversation memory is per-process only and will not survive "
            "a restart or be shared across workers.", _fallback_reason,
        )
        return _build_memory()

    candidates = candidate_dsns(configured, settings)

    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    failures: list[str] = []

    for dsn in candidates:
        for attempt in range(1, CONNECT_ATTEMPTS + 1):
            ctx = None
            try:
                ctx = AsyncPostgresSaver.from_conn_string(dsn)
                saver = await ctx.__aenter__()
                await saver.setup()
            except Exception as exc:  # noqa: BLE001 - any failure must degrade, not crash
                # The generator may have opened a connection before failing;
                # unwind it so the socket is not leaked into the next attempt.
                if ctx is not None:
                    try:
                        await ctx.__aexit__(type(exc), exc, exc.__traceback__)
                    except Exception:  # noqa: BLE001
                        pass
                failures.append(f"{_host_port_of(dsn)}: {type(exc).__name__}")
                # A refused or unreachable socket is worth one more try (a cold
                # database answers in seconds); an authentication failure is not,
                # because retrying the same credentials wastes the whole boot.
                message = str(exc).lower()
                permanent = any(
                    marker in message
                    for marker in ("password authentication failed", "no such user",
                                   "permission denied", "does not exist")
                )
                if permanent or attempt == CONNECT_ATTEMPTS:
                    break
                await asyncio.sleep(CONNECT_BACKOFF_SECONDS * attempt)
                continue
            else:
                _checkpointer = saver
                _checkpointer_ctx = ctx
                _fallback_reason = None
                if dsn != configured:
                    LOGGER.warning(
                        "configured POSTGRES_DSN (%s) was unreachable; using the "
                        "pooler DSN %s instead",
                        _host_port_of(configured), _host_port_of(dsn),
                    )
                LOGGER.info("AsyncPostgresSaver ready (persistent thread memory)")
                return saver

    _fallback_reason = (
        f"postgres checkpointer unavailable after trying "
        f"{len(candidates)} DSN(s): " + "; ".join(failures[:6])
    )
    _fallback_locked = True
    LOGGER.warning(
        "%s. Falling back to per-process memory; thread state will not survive "
        "a restart.", _fallback_reason,
    )
    return _build_memory()


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
