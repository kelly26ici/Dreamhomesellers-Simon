"""Shared-secret authentication.

Every endpoint is on a public Render host. Without a check on the caller,
`POST /api/aux` is an unauthenticated 8,000-character LLM proxy against the
deployment's paid provider keys, and every other endpoint is reachable too.

The check is a constant-time comparison of a header against a configured
secret. It is deliberately tiny: the shared secret is not a session system,
it is a deployment-to-deployment credential, and the Next.js relay is the
only client that should ever hold it.

Configured by SHARED_SECRET. Empty means authentication is DISABLED, which is
only acceptable for a local run -- the lifespan logs a warning in that case so
a production deploy with no secret is visible in the log instead of silently
open.
"""

from __future__ import annotations

import hmac
import logging

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from simon.settings import get_settings

LOGGER = logging.getLogger("simon.auth")

HEADER_NAME = "x-simon-secret"


def _expected() -> str:
    return get_settings().shared_secret_value


def is_authenticated(request: Request) -> bool:
    """True when the request carries the configured secret.

    Compares with `hmac.compare_digest` so the check does not leak the secret
    through timing, and returns False rather than raising when no secret is
    configured -- the caller decides whether an open endpoint is acceptable.
    """
    expected = _expected()
    if not expected:
        return True  # auth disabled: local development only
    presented = request.headers.get(HEADER_NAME, "")
    return hmac.compare_digest(presented, expected)


class SharedSecretMiddleware(BaseHTTPMiddleware):
    """Rejects unauthenticated requests before they reach a handler.

    Registered as a class, because `app.add_middleware` instantiates what it
    is given: a plain `async def middleware(request, call_next)` is silently
    never wrapped, so the check would look installed while every endpoint
    stayed open. Found by the protocol tests failing to build the app with
    `TypeError: middleware() missing 1 required positional argument`.

    `/api/health` stays open: a health check that needs a secret cannot be
    used by a load balancer or by the companion keep-alive loop, which is
    exactly the traffic that keeps a free Render service from spinning down.
    """

    async def dispatch(self, request: Request, call_next):
        return await shared_secret_dispatch(request, call_next)


async def shared_secret_dispatch(request: Request, call_next):
    """The middleware's decision, callable without instantiating the class.

    Kept as a module function so tests can exercise the policy directly
    instead of building a whole ASGI app, and so the class stays a thin
    registration shim.
    """
    if request.url.path == "/api/health":
        return await call_next(request)
    if not is_authenticated(request):
        return JSONResponse(
            status_code=401,
            content={
                "detail": "missing or invalid shared secret",
                "header": HEADER_NAME,
            },
        )
    return await call_next(request)


def warn_if_disabled() -> None:
    """Called from the lifespan, so the log says what the deploy is doing."""
    if not _expected():
        LOGGER.warning(
            "auth DISABLED: SHARED_SECRET is not set. Every endpoint, including "
            "POST /api/aux, is reachable without a credential."
        )
    else:
        LOGGER.info("auth enabled: shared secret required on all but /api/health")
