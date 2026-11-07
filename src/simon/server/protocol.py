"""Agent Streaming Protocol endpoints (langchain-ai/agent-protocol).

Simon previously exposed only POST /api/chat and POST /api/chat/stream — its own
bespoke contract. That is enough for a hand-rolled fetch loop, but it is why
nothing off-the-shelf can talk to Simon: LangChain's Agent Chat UI, the
`useStream` hook and the LangGraph SDK all speak the thread protocol
(/threads/:id/commands + /stream + /state). Adding it here lets a real chat UI
attach with no bespoke adapter on the client.

Endpoints, mounted under /threads/{thread_id}:

    POST /commands  {"id", "method", "params"} -> {"type":"success"|"error", ...}
                     run.start          start a run                  -> {"runId"}
                     state.get          current state values          -> {"values"}
                     state.listCheckpoints                            -> {"values"}

    POST /stream    SSE of protocol events:
                     {"type":"event","seq":n,"method":<channel>,"params":{...}}
                     channels: messages / tools / lifecycle / values / checkpoints
                     / custom.

Why it is built this way:

- The run executes as a background task and publishes onto a per-subscriber
  asyncio.Queue, so POST /commands returns as soon as the run is *started*, and
  /stream carries the conversation. Without that split, one 15s agent turn would
  hold the commands request open for 15s and the client could not render
  progress while it waited.

- Replay is buffered per thread and keyed by a monotonic `seq`, resumed from the
  `Last-Event-ID` header. Re-running a billable agent turn to recover a dropped
  socket is not acceptable, so a reconnect reads the buffer instead.

- One active run per thread. Simon's agent is one conversation per thread sharing
  one checkpointer, so two concurrent turns would interleave tool calls through
  the same history.

- tool-call events are the point of the adapter. `converse()` emits a start
  event and a terminal event per tool; the protocol's `tools` channel turns a
  15s silent pause into a visible "searched 30 published listings" card, and
  `tool-finished` only lands when the result actually exists.

- listing cards travel on the protocol's `custom` channel as
  ``{"name": "listingCards", "cards": [...]}``. They are derived from the
  terminal tool output by ``simon.server.cards`` so the browser never has to
  parse a model-facing payload, and each card is published once per turn.
  Field names on the `tools` channel follow ``langgraph_sdk`` exactly
  (``tool_call_id`` / ``tool_name`` / ``input`` / ``output`` / ``message``);
  the UI parses the protocol, not Simon's private spelling.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from simon.agent.graph import build_agent
from simon.agent.runtime import converse
from simon.server.cards import cards_from_tool_result, merge_cards
from simon.server.page_context import context_digest, page_context_from_metadata

LOGGER = logging.getLogger("simon.server.protocol")

router = APIRouter()


# ---------------------------------------------------------------------------
# Wire shapes — field names taken from langgraph_sdk.schema so the server and
# the SDK cannot drift. tests/test_protocol.py asserts the shapes stay aligned.
# ---------------------------------------------------------------------------


class RunStartParams(BaseModel):
    """Params of run.start: assistant identity plus graph input."""

    model_config = {"extra": "ignore"}

    assistant_id: str | None = None
    input: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Command(BaseModel):
    model_config = {"extra": "ignore"}

    id: int | str | None = None
    method: str
    params: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Run bookkeeping
# ---------------------------------------------------------------------------


class _Run:
    """One in-flight (or last) run for a thread."""

    __slots__ = ("run_id", "thread_id", "task", "seq", "buffer")

    def __init__(self, run_id: str, thread_id: str) -> None:
        self.run_id = run_id
        self.thread_id = thread_id
        self.task: asyncio.Task | None = None
        self.seq = 0
        # (seq, channel, data). Bounded: a subscriber that never reconnects
        # must not grow the buffer without limit.
        self.buffer: list[tuple[int, str, dict]] = []


_runs: dict[str, _Run] = {}
_subscribers: dict[str, list[asyncio.Queue]] = {}


def _publish(run: _Run, channel: str, data: dict) -> None:
    seq = run.seq = run.seq + 1
    run.buffer.append((seq, channel, data))
    if len(run.buffer) > 2048:
        del run.buffer[:1024]
    terminal = channel == "lifecycle" and data.get("event") in ("completed", "failed")
    for q in list(_subscribers.get(run.thread_id, [])):
        # put_nowait, and skip a full queue: a subscriber that stopped reading
        # must not stall the agent turn.
        if q.full():
            continue
        q.put_nowait((seq, channel, data))
        if terminal and not q.full():
            # The run is over, so this thread has no more events to give.
            # Close the stream rather than leaving subscribers parked on get().
            q.put_nowait(None)


def _replay_since(thread_id: str, last_event_id: int) -> list[tuple[int, str, dict]]:
    run = _runs.get(thread_id)
    if run is None:
        return []
    return [e for e in run.buffer if e[0] > last_event_id]


def _is_terminal(run: _Run | None) -> bool:
    """Has this thread's run finished (successfully or not)?

    Read from the replay buffer rather than the task object, because the buffer
    is what a late subscriber actually receives: if it ends in a terminal
    lifecycle event, there is nothing further to send.
    """
    if run is None:
        return True  # no run at all: nothing will ever be published
    for _seq, channel, data in reversed(run.buffer):
        if channel == "lifecycle":
            return data.get("event") in ("completed", "failed")
    return False


def _message_text(params: RunStartParams) -> str:
    """Last user message out of a run.start input.

    Accepts the SDK shape (input.messages = [{role, content}]) and the plain
    string shape, because both reach this endpoint in practice.
    """
    raw = (params.input or {}).get("messages")
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list) and raw:
        last = raw[-1]
        if isinstance(last, dict):
            return str(last.get("content") or "")
        return str(last)
    return ""


# ---------------------------------------------------------------------------
# Run execution
# ---------------------------------------------------------------------------


def _tool_call_id(call: Any) -> str:
    """The id the protocol's `tools` channel is keyed by.

    ``ToolCallStream`` names it ``tool_call_id``. The ``id`` fallback keeps a
    hand-built call object (tests, older integrations) addressable instead of
    minting a random id that no client can correlate.
    """
    value = getattr(call, "tool_call_id", None) or getattr(call, "id", None)
    return str(value) if value else uuid.uuid4().hex


def _tool_name(call: Any) -> str:
    return (
        getattr(call, "tool_name", None)
        or getattr(call, "name", None)
        or "tool"
    )


async def _execute_run(run: _Run, params: RunStartParams) -> None:
    """Run one agent turn, publishing lifecycle / message / tool events."""
    from simon.checkpointer import get_checkpointer

    try:
        _publish(
            run,
            "lifecycle",
            {"event": "started", "namespace": [], "graphName": "simon"},
        )
        _publish(
            run,
            "messages",
            {"event": "message-start", "id": f"msg_{run.run_id}", "role": "ai"},
        )

        text = _message_text(params)
        if not text:
            raise ValueError("run.start carried no user message")

        # The Next relay puts the typed page context in run.start metadata.
        # Without this digest the streaming path could not answer "what about
        # this one?" on a listing page, even though /api/chat could. Context is
        # untrusted: a malformed payload degrades to no context.
        prompt = text + context_digest(
            page_context_from_metadata(params.metadata)
        )

        agent = await build_agent(checkpointer=await get_checkpointer())

        cards: list[dict] = []
        seen_listing_ids: set[str] = set()

        def on_token(token: str) -> None:
            _publish(
                run,
                "messages",
                {
                    "event": "content-block-delta",
                    "id": f"msg_{run.run_id}",
                    "index": 0,
                    "delta": {"type": "text-delta", "text": token},
                },
            )

        def on_tool_call(call: Any) -> None:
            # START only: `call.output` is still None here. Publishing a
            # finished event from this callback is what made the UI mark a
            # search complete the instant it began.
            _publish(
                run,
                "tools",
                {
                    "event": "tool-started",
                    "tool_call_id": _tool_call_id(call),
                    "tool_name": _tool_name(call),
                    "input": getattr(call, "input", None) or {},
                },
            )

        def on_tool_result(call: Any) -> None:
            call_id = _tool_call_id(call)
            error = getattr(call, "error", None)
            if error:
                _publish(
                    run,
                    "tools",
                    {"event": "tool-error", "tool_call_id": call_id, "message": str(error)},
                )
                return

            output = getattr(call, "output", None)
            _publish(
                run,
                "tools",
                {"event": "tool-finished", "tool_call_id": call_id, "output": output},
            )

            fresh = merge_cards(
                cards, cards_from_tool_result(output), seen=seen_listing_ids
            )
            if fresh:
                # Only the newly discovered cards: the client merges frames,
                # and a replay must reproduce the same sequence it saw live.
                _publish(run, "custom", {"name": "listingCards", "cards": fresh})

        answer = await converse(
            agent,
            prompt,
            thread_id=run.thread_id,
            on_token=on_token,
            on_tool_call=on_tool_call,
            on_tool_result=on_tool_result,
        )

        _publish(
            run,
            "messages",
            {"event": "message-finish", "id": f"msg_{run.run_id}", "role": "ai"},
        )
        _publish(run, "lifecycle", {"event": "completed", "namespace": []})
        _publish(run, "values", {"messages": [{"role": "ai", "content": answer}]})
    except Exception as exc:  # noqa: BLE001 - a run must fail as an event, not a crash
        LOGGER.warning("run %s failed: %s", run.run_id, type(exc).__name__)
        _publish(
            run,
            "lifecycle",
            {"event": "failed", "namespace": [], "error": f"{type(exc).__name__}: {exc}"},
        )


# ---------------------------------------------------------------------------
# /commands
# ---------------------------------------------------------------------------


@router.post("/threads/{thread_id}/commands")
async def commands(thread_id: str, command: Command) -> dict[str, Any]:
    if command.method == "run.start":
        active = _runs.get(thread_id)
        if active is not None and active.task is not None and not active.task.done():
            # Return the SAME run rather than starting a second one: two turns
            # interleaving through one checkpointer corrupts the history.
            return {"type": "success", "id": command.id, "result": {"runId": active.run_id}}

        run = _Run(run_id=f"run_{uuid.uuid4().hex[:12]}", thread_id=thread_id)
        _runs[thread_id] = run
        params = RunStartParams(**(command.params or {}))
        run.task = asyncio.create_task(_execute_run(run, params))
        return {"type": "success", "id": command.id, "result": {"runId": run.run_id}}

    if command.method in ("state.get", "state.listCheckpoints"):
        try:
            from langchain_core.runnables import RunnableConfig

            from simon.checkpointer import get_checkpointer

            saver = await get_checkpointer()
            cfg = RunnableConfig(configurable={"thread_id": thread_id, "checkpoint_ns": ""})
            tup = await saver.aget_tuple(cfg)
            values = {}
            checkpoint_id = None
            if tup is not None and tup.checkpoint is not None:
                values = tup.checkpoint.get("channel_values", {}) or {}
                checkpoint_id = tup.checkpoint.get("id")
            return {
                "type": "success",
                "id": command.id,
                "result": {"values": values, "checkpointId": checkpoint_id},
            }
        except Exception as exc:  # noqa: BLE001 - report, never 500 the endpoint
            return {
                "type": "error",
                "id": command.id,
                "error": "state_unavailable",
                "message": f"{type(exc).__name__}: {exc}",
            }

    return {
        "type": "error",
        "id": command.id,
        "error": "invalid_argument",
        "message": f"unsupported method {command.method!r}",
    }


# ---------------------------------------------------------------------------
# /stream
# ---------------------------------------------------------------------------


def _sse(event_id: int, payload: dict) -> str:
    return f"id: {event_id}\ndata: {json.dumps(payload)}\n\n"


def _last_event_id(request: Request) -> int:
    raw = request.headers.get("last-event-id") or request.headers.get("lastEventId")
    if not raw:
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


async def _stream_response(thread_id: str, request: Request) -> StreamingResponse:
    run = _runs.get(thread_id)
    resume_from = _last_event_id(request)
    q: asyncio.Queue = asyncio.Queue(maxsize=512)
    _subscribers.setdefault(thread_id, []).append(q)

    async def gen() -> AsyncIterator[str]:
        try:
            # Replay what this subscriber missed before going live, so a
            # reconnect never re-runs the model to recover the transcript.
            for seq, channel, data in _replay_since(thread_id, resume_from):
                yield _sse(seq, _envelope(seq, channel, data))

            # A subscriber that arrives AFTER the run finished gets the whole
            # replay and no new events will ever be published, so waiting on
            # get() would park this connection forever. Close once the run is
            # terminal — this is the case a plain "open the transcript later"
            # client hits.
            if _is_terminal(run):
                return

            while True:
                item = await q.get()
                if item is None:
                    break
                seq, channel, data = item
                yield _sse(seq, _envelope(seq, channel, data))
        finally:
            subs = _subscribers.get(thread_id)
            if subs and q in subs:
                subs.remove(q)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _envelope(seq: int, channel: str, data: dict) -> dict:
    return {
        "type": "event",
        "seq": seq,
        "method": channel,
        "params": {"namespace": [], "timestamp": time.time(), "data": data},
    }


@router.post("/threads/{thread_id}/stream")
async def stream_post(thread_id: str, request: Request) -> StreamingResponse:
    return await _stream_response(thread_id, request)


@router.get("/threads/{thread_id}/stream")
async def stream_get(thread_id: str, request: Request) -> StreamingResponse:
    """GET form for plain EventSource clients, which cannot set headers."""
    return await _stream_response(thread_id, request)
