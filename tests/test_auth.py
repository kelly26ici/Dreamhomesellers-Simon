"""Auth tests: the middleware must reject without a secret and pass with one.

These are the tests that make the shared-secret decision enforceable. Without
them the only proof that `/api/aux` is no longer an open LLM proxy is a
deploy and a curl.
"""

from __future__ import annotations

import json

import pytest
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from simon.server import auth
from simon.settings import Settings

HEADER = b"x-simon-secret"


def _request(presented: str | None) -> Request:
    headers = [(HEADER, presented.encode())] if presented else []
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/chat",
            "headers": headers,
            "query_string": b"",
        }
    )


@pytest.fixture
def configured(monkeypatch):
    """Point auth at an explicit Settings, isolated from the real .env."""

    def _configure(**kwargs):
        settings = Settings(_env_file="", **kwargs)
        monkeypatch.setattr(auth, "get_settings", lambda: settings)
        return settings

    return _configure


def test_missing_secret_is_rejected(configured):
    configured(shared_secret="s3cret")
    assert auth.is_authenticated(_request(None)) is False


def test_wrong_secret_is_rejected(configured):
    configured(shared_secret="s3cret")
    assert auth.is_authenticated(_request("wrong")) is False


def test_correct_secret_is_accepted(configured):
    configured(shared_secret="s3cret")
    assert auth.is_authenticated(_request("s3cret")) is True


def test_unconfigured_secret_disables_auth_for_local_runs(configured):
    """No SHARED_SECRET means open, which is only acceptable locally. The
    lifespan warns loudly, so the state is never silent in a deploy log."""
    configured(shared_secret="")
    assert auth.is_authenticated(_request(None)) is True


async def _dispatch(request: Request, call_next):
    """Call the middleware's decision logic without building a whole app."""
    return await auth.shared_secret_dispatch(request, call_next)


async def test_health_stays_open_even_with_auth_enabled(configured):
    """A health check that needs a credential cannot be used by a load
    balancer or by the companion keep-alive loop -- which is exactly the
    traffic that keeps a free Render service from spinning down.

    The exemption is in the MIDDLEWARE, not in is_authenticated: the predicate
    answers "is this credential good", and the middleware decides which paths
    need an answer at all. Asserted that way deliberately, so a change that
    moves the exemption into the predicate has to update this test rather than
    silently widening what is open.
    """
    configured(shared_secret="s3cret")
    seen: list[str] = []

    async def call_next(request):
        seen.append(request.url.path)
        return JSONResponse({"ok": True})

    response = await _dispatch(
        Request(
            {
                "type": "http",
                "method": "GET",
                "path": "/api/health",
                "headers": [],
                "query_string": b"",
            }
        ),
        call_next,
    )

    assert response.status_code == 200
    assert seen == ["/api/health"]  # reached the handler, was not rejected


async def test_non_health_path_is_rejected_without_a_credential(configured):
    """The counterpart: the exemption is narrow, not a blanket bypass."""
    configured(shared_secret="s3cret")

    async def call_next(request):  # pragma: no cover - must not be reached
        raise AssertionError("handler must not run without a credential")

    response = await _dispatch(
        Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/api/chat",
                "headers": [],
                "query_string": b"",
            }
        ),
        call_next,
    )

    assert response.status_code == 401
    assert json.loads(response.body)["header"] == auth.HEADER_NAME


def test_middleware_is_registered_as_a_class_not_a_bare_function():
    """Regression: a bare `async def middleware(request, call_next)` passed to
    `app.add_middleware` is instantiated, never called, and TypeErrors at
    import -- so the check looks installed while every endpoint stays open.

    Starlette requires a class it can instantiate with `(app, ...)`. Asserting
    the type keeps that mistake from coming back.
    """
    from simon.server.main import app

    registered = {m.cls.__name__: m.cls for m in app.user_middleware}
    assert "SharedSecretMiddleware" in registered, (
        f"auth middleware not registered; got {sorted(registered)}"
    )
    assert issubclass(registered["SharedSecretMiddleware"], BaseHTTPMiddleware)
