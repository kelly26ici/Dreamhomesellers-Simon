"""Keep the companion Render service warm by giving it inbound traffic.

Render spins a free web service down after ~15 minutes with no **inbound**
requests. A background loop inside a service makes *outbound* requests, which
does not count — so a service cannot keep itself warm. Each service pinging the
*other* one's health endpoint is therefore the design: both receive inbound
traffic, and a single visit to either URL makes the pair self-sustaining.

The failure mode this cannot fix: if **both** services are cold at the same
time, neither loop is running to wake the other. One visit to either URL boots
the pair; nothing in-repo can do that. See the cross-ping note in the rebuild
plan.

The website's half of the pair lives in the other repo, at
``src/instrumentation.ts``.

    LUXONN_SITE_URL=https://dreamhomesellers.onrender.com uv run uvicorn simon.server.main:app
"""

from __future__ import annotations

import asyncio
import logging

import httpx

from simon.settings import get_settings

LOGGER = logging.getLogger("simon.keepalive")

INTERVAL_SECONDS = 300
HEALTH_PATH = "/api/health"

# A cold free-tier service can take ~90s to answer its first request, and the
# health check is cheap, so the timeout is deliberately generous: a timeout is
# logged, never raised, and never blocks the next cycle.
TIMEOUT = httpx.Timeout(120.0, connect=15.0)


async def _ping_forever() -> None:
    """One GET per cycle, forever. Every failure is logged and swallowed."""
    settings = get_settings()
    url = f"{settings.luxonn_site_url.rstrip('/')}{HEALTH_PATH}"
    LOGGER.info("keep-alive: pinging %s every %ds", url, INTERVAL_SECONDS)

    async with httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=True) as client:
        while True:
            try:
                response = await client.get(url)
                if response.status_code >= 400:
                    LOGGER.warning(
                        "keep-alive: %s returned HTTP %s", url, response.status_code
                    )
            except httpx.HTTPError:
                LOGGER.warning("keep-alive: ping to %s failed", url, exc_info=True)
            await asyncio.sleep(INTERVAL_SECONDS)


def start() -> asyncio.Task | None:
    """Called from the FastAPI lifespan. Returns the task, or None if disabled.

    Returning None keeps a local ``uv run`` (no LUXONN_SITE_URL) from creating a
    loop that cannot work, and keeps the lifespan shutdown path unconditional.
    """
    if not get_settings().luxonn_site_url:
        LOGGER.info("keep-alive disabled: LUXONN_SITE_URL is not set")
        return None
    return asyncio.create_task(_ping_forever())
