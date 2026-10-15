"""FastAPI service boundary for Simon (ADR-008).

Exposes the existing LangChain agent behind typed HTTP endpoints so the
native LUXONN Next.js UI can consume Simon with zero iframes and a real
streaming boundary. Mirrors the frontend contract `src/lib/simon/types.ts`.

Endpoints:
    POST /api/chat          one concierge turn; non-streaming JSON envelope
    POST /api/chat/stream   same turn as an SSE token stream
    GET  /api/health        liveness for the frontend status indicator

SSE event contract (one JSON object per `data:` line):
    {"type": "token", "data": "<text delta>"}
    {"type": "done",  "data": {"answer": "<full>", "conversationId": "..."}}
    {"type": "error", "data": "<message>"}

Run:
    uv run uvicorn simon.server.main:app --host 127.0.0.1 --port 8787
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, AsyncIterator, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from simon.agent import build_agent, converse
from simon.tools import TOOLS

app = FastAPI(title="Simon concierge service", version="0.1.0")

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


def make_agent():
    """One agent per request: the InMemorySaver checkpointer (ADR-011 debt)
    is not safe to share across concurrent conversations. Replace with the
    persistent Postgres checkpointer before the native product launch."""
    return build_agent()


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "service": "simon", "tools": len(TOOLS)}


@app.post("/api/chat")
async def chat(req: ChatRequest) -> dict[str, Any]:
    """One concierge turn, non-streaming. Response shape matches the
    frontend SimonChatResponse: {message: {id, role, content, ...},
    conversationId}."""
    conversation_id = req.conversation_id or str(uuid.uuid4())
    prompt = req.message + context_digest(req.page_context)

    agent = make_agent()
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
        agent = make_agent()
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
            except Exception as exc:  # noqa: BLE001
                await queue.put(
                    "data: " + json.dumps({"type": "error", "data": str(exc)})
                    + "\n\n"
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
