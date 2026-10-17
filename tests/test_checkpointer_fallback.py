"""Checkpointer DSN fallback tests.

These lock in the 2026-10-17 fix. The production failure was silent and lasted
days: the deployed ``POSTGRES_DSN`` points at Supabase's direct host, which has
no A record (IPv6-only), so on an IPv4 network the socket fails before auth and
the service quietly degrades to per-process memory. Nothing crashed, so nothing
ever looked broken — it just forgot every conversation on restart.

The assertions below are about *recovery*, not just degradation: a configured
DSN being unreachable must not mean memory is lost, because the pooler
coordinates are already in the environment.
"""

from __future__ import annotations

import importlib
from urllib.parse import urlparse

import pytest

from simon.checkpointer import candidate_dsns
from simon.settings import PoolSettings


def _dsn(host: str, port: int | str) -> str:
    return f"postgresql://user:pass@{host}:{port}/postgres"


def test_configured_dsn_alone_yields_one_candidate():
    """Without pooler info there is nothing to fall back to."""
    s = PoolSettings()
    out = candidate_dsns(_dsn("db.example.test", 5432), s)
    assert out == [_dsn("db.example.test", 5432)]


def test_pooler_adds_session_mode_before_transaction_mode():
    """AsyncPostgresSaver holds one connection and runs DDL: session (5432) is
    the correct mode, so it must come first."""
    s = PoolSettings(
        supabase_pooler_host="ref.aws-0-eu-west-1.pooler.supabase.com",
        supabase_pooler_port="5432",
        supabase_pooler_user="postgres.ref",
        supabase_db_password="pw",
        supabase_db_name="postgres",
    )
    out = candidate_dsns(_dsn("db.ref.supabase.co", 5432), s)
    # configured first, then the pooler on the port it advertises
    assert out[0] == _dsn("db.ref.supabase.co", 5432)
    # the pooler candidates carry the POOLER credentials, not the configured ones
    assert "postgresql://postgres.ref:pw@ref.aws-0-eu-west-1.pooler.supabase.com:5432/postgres" in out
    pooler_ports = [
        urlparse(p).port
        for p in out
        if "pooler.supabase.com" in p
    ]
    assert 5432 in pooler_ports
    assert 6543 in pooler_ports
    assert pooler_ports.index(5432) < pooler_ports.index(6543)


def test_region_canonical_host_is_tried_when_configured_host_differs():
    """The deployed pooler hostname may not resolve; the region-canonical form
    does, so it is offered rather than trusting an unverified name."""
    s = PoolSettings(
        supabase_pooler_host="unresolvable.pooler.supabase.com",
        supabase_pooler_user="postgres.ref",
        supabase_db_password="pw",
        supabase_project_region="eu-west-1",
    )
    out = candidate_dsns(_dsn("db.ref.supabase.co", 5432), s)
    assert "postgresql://postgres.ref:pw@aws-0-eu-west-1.pooler.supabase.com:5432/postgres" in out


def test_no_duplicate_candidates():
    s = PoolSettings(
        supabase_pooler_host="ref.pooler.supabase.com",
        supabase_pooler_port="5432",
        supabase_pooler_user="postgres.ref",
        supabase_db_password="pw",
    )
    out = candidate_dsns(_dsn("ref.pooler.supabase.com", 5432), s)
    assert len(out) == len(set(out))


def test_pooler_without_password_still_offered():
    """Trust/peer/IAM clusters have no password. Requiring one silently removed
    the fallback in the exact case it existed for."""
    s = PoolSettings(
        supabase_pooler_host="127.0.0.1",
        supabase_pooler_port="55432",
        supabase_pooler_user="postgres",
        supabase_db_password="",
    )
    out = candidate_dsns(_dsn("db.ref.supabase.co", 5432), s)
    assert "postgresql://postgres@127.0.0.1:55432/postgres" in out


def test_special_characters_in_credentials_are_escaped():
    """A password containing '@' or '/' would otherwise corrupt the DSN."""
    s = PoolSettings(
        supabase_pooler_host="ref.pooler.supabase.com",
        supabase_pooler_user="postgres.ref",
        supabase_db_password="p@ss/word#1",
    )
    out = candidate_dsns(_dsn("db.ref.supabase.co", 5432), s)
    candidate = next(d for d in out if "pooler" in d)
    # '@' inside the password must not terminate the userinfo section
    assert candidate.startswith(
        "postgresql://postgres.ref:p%40ss%2Fword%231@ref.pooler.supabase.com:"
    )


@pytest.mark.asyncio
async def test_unreachable_configured_dsn_degrades_with_a_precise_reason(monkeypatch):
    """Degradation is allowed; silence is not. The reason must name host:port so
    /api/health is actionable instead of reporting a bare class name."""
    mod = importlib.import_module("simon.checkpointer")
    monkeypatch.setattr(
        mod, "get_pool_settings",
        lambda: PoolSettings(postgres_dsn=_dsn("127.0.0.1", 1)),  # port 1: refused
    )
    monkeypatch.setattr(mod, "CONNECT_ATTEMPTS", 1)
    monkeypatch.setattr(mod, "_checkpointer", None)
    monkeypatch.setattr(mod, "_fallback_locked", False)

    saver = await mod.get_checkpointer()

    assert mod.is_persistent() is False
    assert mod.fallback_reason() is not None
    assert "127.0.0.1:1" in mod.fallback_reason()
    # the in-memory fallback is what lets the service keep serving
    assert type(saver).__name__ == "InMemorySaver"


@pytest.mark.asyncio
async def test_failure_locks_so_later_requests_do_not_reconnect(monkeypatch):
    """A service that already knows it is degraded must not pay the connection
    cost again on every visitor message."""
    mod = importlib.import_module("simon.checkpointer")
    monkeypatch.setattr(
        mod, "get_pool_settings",
        lambda: PoolSettings(
            postgres_dsn="postgresql://u:p@127.0.0.1:1/postgres",  # no pooler info
        ),
    )
    monkeypatch.setattr(mod, "CONNECT_ATTEMPTS", 1)
    monkeypatch.setattr(mod, "_checkpointer", None)
    monkeypatch.setattr(mod, "_fallback_locked", False)

    await mod.get_checkpointer()
    assert mod.is_persistent() is False
    assert mod._fallback_locked is True

    # Spy on the real connect path: after the lock it must not be entered again.
    entered = {"n": 0}

    class _Counting(Exception):
        pass

    real_model = getattr(mod, "__dict__", {})
    before = dict(real_model)
    try:
        import langgraph.checkpoint.postgres.aio as aio

        original = aio.AsyncPostgresSaver.from_conn_string

        def counting(dsn):
            entered["n"] += 1
            return original(dsn)

        aio.AsyncPostgresSaver.from_conn_string = counting
        await mod.get_checkpointer()
    finally:
        aio.AsyncPostgresSaver.from_conn_string = original
        for k, v in before.items():
            setattr(mod, k, v)

    assert entered["n"] == 0
