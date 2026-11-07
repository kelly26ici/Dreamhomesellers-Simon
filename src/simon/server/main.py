"""FastAPI service boundary for Simon (ADR-008).

Exposes the existing LangChain agent behind typed HTTP endpoints so the
native LUXONN Next.js UI can consume Simon with zero iframes and a real
streaming boundary. Mirrors the frontend contract `src/lib/simon/types.ts`.

Endpoints:
    POST /api/chat          one concierge turn; non-streaming JSON envelope
    POST /api/chat/stream   same turn as an SSE token stream
    POST /api/aux           one bounded, tool-less model call (thread titles,
                            summaries); never touches the agent or checkpoints
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
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from simon import keepalive
from simon.agent.graph import build_agent
from simon.agent.middleware.serving_model import serving_model_id
from simon.agent.providers import primary_model
from simon.agent.runtime import converse
from simon.checkpointer import get_checkpointer
from simon.server import auth, protocol
from simon.server.cards import cards_from_tool_result, merge_cards
from simon.server.page_context import PageContext, context_digest
from simon.settings import get_settings
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

    await get_checkpointer()
    # Says, in the deploy log, whether the service is open or authenticated.
    # A production deploy with no SHARED_SECRET is otherwise silent.
    auth.warn_if_disabled()
    prune_task = asyncio.create_task(_prune_loop())
    keepalive_task = keepalive.start()
    try:
        yield
    finally:
        if keepalive_task is not None:
            keepalive_task.cancel()
        prune_task.cancel()
        from simon.checkpointer import close_checkpointer

        await close_checkpointer()


# Agent Streaming Protocol (/threads/{id}/commands + /stream): without it
# no standard chat UI — Agent Chat UI, useStream, the LangGraph SDK — can
# talk to Simon at all. Mounted here so it shares the app's CORS and
# lifespan (the checkpointer it reads runs state.get against the same
# singleton /api/chat writes).

LOGGER = logging.getLogger("simon.server")

app = FastAPI(title="Simon concierge service", version="0.1.0", lifespan=lifespan)
app.include_router(protocol.router)

# CORS is env-driven. The previous hardcoded localhost list was dead in
# production (nothing on Render sends an Origin of localhost:3000) and a
# deploy blocker. Empty means same-origin only, which is the correct default:
# this service is called by the website's server-side relay, not by browsers.
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "x-simon-secret"],
)
app.add_middleware(auth.SharedSecretMiddleware)


class ChatRequest(BaseModel):
    model_config = {"extra": "ignore"}

    message: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = None
    page_context: PageContext | None = None


class AuxRequest(BaseModel):
    """A single, tool-less model call for host-side metadata.

    Thread titles and rolling summaries are metadata, not conversation: they
    must not create checkpoints, appear in history, or share the agent's
    system prompt. Keeping them here also means a dead title model can never
    take down `/api/chat`.
    """

    model_config = {"extra": "ignore"}

    prompt: str = Field(min_length=1, max_length=8000)
    timeout_seconds: float = Field(default=8.0, ge=1.0, le=15.0)


def make_agent():
    """One agent per request, but sharing the process-wide checkpointer.

    The agent object itself is cheap to rebuild; the checkpointer is the
    expensive, stateful part and must NOT be per-request — that is what made
    ``thread_id`` decorative and multi-turn questions unanswerable.

    SYNCHRONOUS, and it reads the saver the lifespan already warmed rather
    than resolving its own. Construction performs no I/O, so awaiting here
    would be wrong: an async def returning a non-awaited coroutine hands the
    route a coroutine instead of a graph, and /api/chat answers 500. That
    exact bug shipped once.
    """
    from simon.checkpointer import shared_checkpointer

    return build_agent(checkpointer=shared_checkpointer())


async def _probe_model() -> str:
    """Cheap readiness signal: can a model actually be constructed?

    Constructing the chat model performs no API call — nothing is probed any
    more — but it fails loudly when no provider key is configured, which is
    precisely the state the old static {"status":"ok"} hid. Returns the resolved
    model id, or a reason string.
    """
    try:
        model = primary_model()
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
    # Which model last ANSWERED, not which one is configured to. The old
    # health route reported the resolved provider, which kept saying
    # "nvidia" while every turn was served by the third model in the chain.
    from simon.agent.middleware import middleware_summary
    from simon.agent.middleware.refusal import refusal_signalled

    return {
        "status": "ok" if ready else "degraded",
        "service": "simon",
        "tools": len(TOOLS),
        "provider": provider,
        "serving_model": serving_model_id(),
        "refusal_signal": refusal_signalled(),
        "middleware": middleware_summary(),
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

    # The non-streaming route returns the same buyer-safe cards as the
    # protocol's `custom` channel. Without this, a stream failure that falls
    # back to /api/chat loses the cards the streamed answer would have shown.
    cards: list[dict[str, Any]] = []
    seen_listing_ids: set[str] = set()

    def on_tool_result(call: Any) -> None:
        if getattr(call, "error", None):
            return
        merge_cards(
            cards,
            cards_from_tool_result(getattr(call, "output", None)),
            seen=seen_listing_ids,
        )

    agent = make_agent()
    try:
        answer = await asyncio.wait_for(
            converse(
                agent,
                prompt,
                thread_id=conversation_id,
                on_tool_result=on_tool_result,
            ),
            timeout=45,
        )
    except TimeoutError as exc:
        # A turn that exhausts its budget is evidence about the PROVIDER, not
        # about the question. The framework's ModelRetry and ModelFallback
        # re-select on the next turn, so the honest action here is only to stop
        # claiming to know who served.
        try:
            from simon.agent.middleware.serving_model import reset_serving_model

            # Framework fallback re-selects on the next turn. This only
            # stops the service claiming to know who served.
            reset_serving_model()
        except Exception:  # noqa: BLE001 - never mask the 504 with a bookkeeping error
            LOGGER.warning("could not cool down the provider after a timeout",
                           exc_info=True)
        raise HTTPException(status_code=504, detail="Simon turn timed out") from exc

    return {
        "message": {
            "id": str(uuid.uuid4()),
            "role": "assistant",
            "content": answer,
            "timestamp": "",
            "listingCards": cards,
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
            # Accumulate here as well as forwarding: the final `done` event
            # promises the complete answer, and it used to join an empty list.
            parts.append(tok)
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
            except TimeoutError:
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


@app.post("/api/aux")
async def aux(req: AuxRequest) -> dict[str, Any]:
    """One bounded model call for host-side metadata.

    The caller (the LUXONN Next app) owns the prompt and parses strict JSON
    out of the answer; Simon owns provider selection. No tools are bound and
    no checkpoint is read or written, so this cannot mutate a conversation.

    `primary_model()` construction performs no API call and no probe, so it is
    safe on the event loop. The invoke itself is capped by the caller's
    timeout so a dead provider fails the metadata call instead of the page.
    """
    try:
        model = primary_model()
    except Exception as exc:  # noqa: BLE001 - report as unavailable
        LOGGER.warning("aux: no model available: %s", exc)
        raise HTTPException(status_code=503, detail="model unavailable") from exc

    try:
        # Bounded at the CALL, not the constructor: the NVIDIA endpoint rejects
        # `request_timeout` as an unknown parameter, and ChatNVIDIA forwards
        # constructor kwargs to the provider. The agent path already bounds its
        # own calls this way; /api/aux does the same rather than relying on a
        # construction-time knob that provably does not survive the trip.
        result = await asyncio.wait_for(
            model.ainvoke(req.prompt), timeout=req.timeout_seconds
        )
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail="aux timed out") from exc
    except Exception as exc:  # noqa: BLE001 - one retry belongs to the caller
        LOGGER.warning("aux: model call failed: %s", exc)
        raise HTTPException(status_code=502, detail="aux failed") from exc

    content = getattr(result, "content", result)
    if isinstance(content, list):
        # Some providers return content blocks instead of a string.
        content = "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    if not isinstance(content, str):
        content = str(content)

    return {"text": content}
