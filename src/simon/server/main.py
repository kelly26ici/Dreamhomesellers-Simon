"""FastAPI service boundary for Simon (ADR-008).

Exposes the existing LangChain agent behind typed HTTP endpoints so the
native LUXONN Next.js UI can consume Simon with zero iframes and a real
streaming boundary. Mirrors the frontend contract `src/lib/simon/types.ts`.

Endpoints:
    POST /api/chat          one concierge turn; non-streaming JSON envelope
    POST /api/chat/stream   same turn as an SSE token stream
    GET  /api/health        liveness for the frontend status indicator

SSE event contract (one JSON object per `data:` line):
    {"type": "token",   "data": "<text delta>"}
    {"type": "done",    "data": {"answer": "<full>", "conversationId": "..."}}
    {"type": "timeout", "data": "<message>"}   # turn exceeded the budget; retry
    {"type": "error",   "data": "<message>"}   # turn failed

Run (a container platform injects PORT and requires a non-loopback bind):
    uv run uvicorn simon.server.main:app --host 0.0.0.0 --port ${PORT:-8787}
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, AsyncIterator, Literal

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from simon.agent import build_agent, build_model, converse
from simon.tools import TOOLS

async def _prune_loop():
    """Periodically drop checkpoints past the retention window.

    Checkpoints accumulate one row per super-step per conversation, so an
    unpruned table grows without bound. Runs in the background; failures are
    logged by the prune helper and never surface to a request.
    """
    from simon.checkpointer import prune_old_checkpoints

    interval = 6 * 60 * 60
    while True:
        await asyncio.sleep(interval)
        removed = await prune_old_checkpoints()
        if removed and removed > 0:
            LOGGER.info("pruned %d stale checkpoint groups", removed)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Warm the checkpointer at boot rather than on the first request.

    Warming at boot means /api/health can report memory truth on the very first
    probe, and a bad POSTGRES_DSN is visible in the deploy log immediately
    instead of after the first visitor message.
    """
    from simon.checkpointer import get_checkpointer

    await get_checkpointer()
    prune_task = asyncio.create_task(_prune_loop())
    try:
        yield
    finally:
        prune_task.cancel()
        from simon.checkpointer import close_checkpointer

        await close_checkpointer()


LOGGER = logging.getLogger("simon.server")

app = FastAPI(title="Simon concierge service", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class SearchFilters(BaseModel):
    """Mirror of the frontend SearchFilters contract (contracts.md §10)."""

    model_config = {"extra": "ignore"}

    listing_kind: str | None = None
    property_type: str | None = None
    min_price: str | None = None
    max_price: str | None = None
    price_currency: str | None = None
    min_bedrooms: int | None = None
    min_bathrooms: int | None = None
    city: str | None = None
    area: str | None = None
    amenities: list[str] | None = None
    sort: str | None = None
    cursor: str | None = None


class ApproximateLocation(BaseModel):
    model_config = {"extra": "ignore"}

    latitude: float
    longitude: float


class PageContext(BaseModel):
    """Typed transient intent (untrusted). Consumed as context only — the
    agent's tools revalidate every fact against Postgres/Qdrant. Never
    persisted as authority."""

    model_config = {"extra": "ignore"}

    contract_version: str | None = None
    sequence: int | None = None
    route: str | None = None
    view_mode: Literal[
        "search", "property_detail", "compare", "curated", "saved",
        "workspace", "other",
    ] | None = None
    active_listing_id: str | None = None
    active_revision_id: str | None = None
    result_position: int | None = None
    result_count: int | None = None
    visible_listing_ids: list[str] = Field(default_factory=list, max_length=64)
    search_filters: SearchFilters | None = None
    compare_selection: list[str] = Field(default_factory=list, max_length=4)
    approximate_location: ApproximateLocation | None = None


class ChatRequest(BaseModel):
    model_config = {"extra": "ignore"}

    message: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = None
    page_context: PageContext | None = None


def context_digest(ctx: PageContext | None) -> str:
    """Render page context into a compact preamble so the agent knows where
    the visitor is and what 'this' refers to (simon-boundary.md: context
    declares what the visitor means; the service retrieves what is true)."""
    if ctx is None:
        return ""

    bits: list[str] = []
    if ctx.view_mode == "property_detail" and ctx.active_listing_id:
        bits.append(
            f"The visitor is viewing the property detail page for listing "
            f"{ctx.active_listing_id}; 'this' most likely refers to it."
        )
    elif ctx.view_mode == "search" and ctx.visible_listing_ids:
        bits.append(
            f"The visitor is on the search grid ({len(ctx.visible_listing_ids)} "
            f"listings visible): {', '.join(ctx.visible_listing_ids[:8])}."
        )
    elif ctx.view_mode == "compare" and ctx.compare_selection:
        bits.append(
            f"The visitor is comparing listings: "
            f"{', '.join(ctx.compare_selection)}."
        )
    elif ctx.view_mode == "curated":
        bits.append("The visitor is browsing the curated 3D gallery collection.")
    elif ctx.route:
        bits.append(f"The visitor is on {ctx.route}.")

    f = ctx.search_filters
    if f is not None:
        constraints = ", ".join(
            f"{k}={v}"
            for k, v in {
                "city": f.city,
                "area": f.area,
                "min_bedrooms": f.min_bedrooms,
                "max_price": f.max_price,
                "property_type": f.property_type,
            }.items()
            if v is not None
        )
        if constraints:
            bits.append(f"Active search filters: {constraints}.")

    return ("\n\n[Visitor context] " + " ".join(bits)) if bits else ""


async def make_agent():
    """One agent per request, but sharing the process-wide checkpointer.

    The agent object itself is cheap to rebuild; the checkpointer is the
    expensive, stateful part and must NOT be per-request — that is what made
    ``thread_id`` decorative and multi-turn questions unanswerable.
    ``simon.checkpointer`` owns the shared saver (AsyncPostgresSaver in
    production, degrading to per-process memory with a logged reason when no
    DSN is reachable).
    """
    from simon.checkpointer import get_checkpointer

    return await build_agent(checkpointer=await get_checkpointer())


async def _probe_model() -> str:
    """Cheap readiness signal: can a model actually be constructed?

    Constructing the chat model performs no API call, but it fails loudly when
    no provider key is configured — which is precisely the state the old static
    {"status":"ok"} hid. Returns the resolved provider name, or a reason string.
    """
    try:
        model = build_model()
    except Exception as exc:  # noqa: BLE001 - report, never 500 the health route
        return f"unconfigured ({type(exc).__name__})"
    for attr in ("model", "model_name"):
        name = getattr(model, attr, None)
        if isinstance(name, str) and name:
            return name
    return type(model).__name__


@app.get("/api/health")
async def health() -> dict[str, Any]:
    """Readiness, not liveness.

    Deliberately reports the things a misconfigured deploy used to hide: which
    provider resolved, whether thread memory is persistent, and whether the
    checkpointer fell back. No credential values, key presence only.
    """
    from simon.checkpointer import fallback_reason, is_persistent

    provider = await _probe_model()
    ready = provider and not provider.startswith("unconfigured")
    return {
        "status": "ok" if ready else "degraded",
        "service": "simon",
        "tools": len(TOOLS),
        "provider": provider,
        "memory": "postgres" if is_persistent() else "process",
        "memory_detail": fallback_reason(),
    }


@app.post("/api/chat")
async def chat(req: ChatRequest) -> dict[str, Any]:
    """One concierge turn, non-streaming. Response shape matches the
    frontend SimonChatResponse: {message: {id, role, content, ...},
    conversationId}."""
    conversation_id = req.conversation_id or str(uuid.uuid4())
    prompt = req.message + context_digest(req.page_context)

    agent = await make_agent()
    try:
        answer = await asyncio.wait_for(
            converse(agent, prompt, thread_id=conversation_id),
            timeout=45,
        )
    except asyncio.TimeoutError as exc:
        raise HTTPException(status_code=504, detail="Simon turn timed out") from exc

    return {
        "message": {
            "id": str(uuid.uuid4()),
            "role": "assistant",
            "content": answer,
            "timestamp": "",
            "suggestedFollowUps": [],
        },
        "conversationId": conversation_id,
    }


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    """Same turn as an SSE stream: token deltas as they are generated, then
    a final done event carrying the complete answer."""

    async def event_gen() -> AsyncIterator[str]:
        conversation_id = req.conversation_id or str(uuid.uuid4())
        prompt = req.message + context_digest(req.page_context)
        agent = await make_agent()
        parts: list[str] = []
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=1024)
        SENTINEL = "__SIMON_STREAM_DONE__"

        async def on_token(tok: str) -> None:
            payload = json.dumps({"type": "token", "data": tok})
            await queue.put(f"data: {payload}\n\n")

        async def run() -> None:
            try:
                await asyncio.wait_for(
                    converse(
                        agent, prompt, thread_id=conversation_id,
                        on_token=on_token,
                    ),
                    timeout=45,
                )
            except asyncio.TimeoutError:
                # Distinguishable from a failed turn: the website treats a
                # timeout as "the agent is thinking, retry" rather than
                # "the agent is down", and silently reporting the generic
                # error made a cold start look like an outage.
                await queue.put(
                    "data: "
                    + json.dumps({"type": "timeout", "data": "Simon turn timed out"})
                    + "\n\n"
                )
            except Exception as exc:  # noqa: BLE001
                await queue.put(
                    "data: " + json.dumps({"type": "error", "data": str(exc)}) + "\n\n"
                )
            finally:
                await queue.put(SENTINEL)

        task = asyncio.create_task(run())
        try:
            while True:
                item = await queue.get()
                if item == SENTINEL:
                    break
                yield item
            await task
            yield "data: " + json.dumps({
                "type": "done",
                "data": {
                    "answer": "".join(parts),
                    "conversationId": conversation_id,
                },
            }) + "\n\n"
        finally:
            task.cancel()

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
