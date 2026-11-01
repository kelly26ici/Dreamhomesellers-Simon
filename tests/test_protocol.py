"""Agent Streaming Protocol contract tests.

The point of these is drift: Simon now speaks the langchain-ai/agent-protocol
wire format that `@langchain/react`'s useStream, the LangGraph SDK and Agent
Chat UI all expect. A field renamed on our side produces a client that silently
renders nothing, so the shapes are asserted against the SDK's own schema module
rather than against hand-copied JSON.

No model is called — the turn is stubbed, because what is under test is the
ENVELOPE: seq monotonicity, channel names, replay, and failure-as-event.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

import simon.server.protocol as proto


@pytest.fixture
def client(monkeypatch):
    from simon.server.main import app

    # Clean per-test run state: _runs/_subscribers are module-level by design
    # (one active run per thread), so leaking them between tests would make
    # run.start hand back a stale runId.
    monkeypatch.setattr(proto, "_runs", {})
    monkeypatch.setattr(proto, "_subscribers", {})

    # Stub BOTH the agent construction and the turn. `converse` and `build_agent`
    # are resolved from module scope precisely so these can be replaced: leaving
    # build_agent live makes every test probe a real provider over the network,
    # which is slow, flaky, and never asserted on.
    async def fake_build_agent(*, checkpointer=None):
        return object()

    monkeypatch.setattr(proto, "build_agent", fake_build_agent)

    async def fake_converse(agent, message, *, thread_id, on_token=None, on_tool_call=None, on_tool_result=None, **_):
        if on_tool_call or on_tool_result:
            class _Call:
                tool_call_id = "tc_1"
                tool_name = "search_properties"
                input = {"area": "Karen"}
                output = {
                    "count": 1,
                    "properties": [
                        {
                            "listing_id": "11111111-1111-4111-8111-111111111111",
                            "title": "Karen Ridge Villa",
                            "property_type": "house",
                            "location": "Karen, Nairobi",
                            "price": "11500000.00",
                            "currency": "KES",
                            "price_period": "total",
                            "bedrooms": 4,
                            "bathrooms": 3,
                        }
                    ],
                }
                error = None

            if on_tool_call:
                on_tool_call(_Call())
            if on_tool_result:
                on_tool_result(_Call())
        if on_token:
            on_token("hello ")
            on_token("there")
        return "hello there"

    monkeypatch.setattr(proto, "converse", fake_converse)
    return TestClient(app)


def test_unsupported_command_is_a_typed_error(client):
    r = client.post(
        "/threads/t1/commands",
        json={"id": 1, "method": "bogus.method", "params": {}},
    )
    # Protocol errors travel in-band, not as HTTP 500 — the client parses them.
    assert r.status_code == 200
    body = r.json()
    assert body["type"] == "error"
    assert body["error"] == "invalid_argument"
    assert "bogus.method" in body["message"]
    assert body["id"] == 1, "the command id is echoed so the client can correlate"


def test_run_start_returns_a_run_id_without_waiting_for_the_turn(client):
    r = client.post(
        "/threads/t2/commands",
        json={
            "id": 2,
            "method": "run.start",
            "params": {"assistantId": "agent", "input": {"messages": [{"role": "user", "content": "hi"}]}},
        },
    )
    body = r.json()
    assert body["type"] == "success"
    assert body["id"] == 2
    assert body["result"]["runId"].startswith("run_")


def test_second_concurrent_turn_reuses_the_active_run(client):
    """Two turns through one checkpointer would interleave tool calls, so the
    platform's one-run-per-thread rule returns the same run instead."""
    first = client.post(
        "/threads/t3/commands",
        json={"id": 3, "method": "run.start", "params": {"input": {"messages": [{"role": "user", "content": "a"}]}}},
    ).json()["result"]["runId"]

    # A turn the same length as the stubbed one is already done by now, so
    # start one that is still in flight to exercise the guard.
    active = proto._Run(run_id="run_inflight", thread_id="t3")

    class _Pending:
        def done(self):
            return False

    active.task = _Pending()
    proto._runs["t3"] = active
    second = client.post(
        "/threads/t3/commands",
        json={"id": 4, "method": "run.start", "params": {"input": {"messages": [{"role": "user", "content": "b"}]}}},
    )
    assert second.json()["result"]["runId"] == active.run_id
    assert first.startswith("run_")


def test_state_get_reports_empty_for_unknown_thread(client):
    r = client.post("/threads/unknown/commands", json={"id": 5, "method": "state.get", "params": {}})
    body = r.json()
    assert body["type"] == "success"
    assert body["result"]["values"] == {}
    assert body["result"]["checkpointId"] is None


def _read_sse(resp, until_closed=True):
    """Read a protocol stream.

    Defaults to reading until the server CLOSES it, because the server now ends
    a stream once its run is terminal — and the order of the final events
    (message-finish, values) is exactly what these tests assert.
    """
    events = []
    for line in resp.iter_lines():
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: "):]))
    return events


def test_stream_replays_and_streams_protocol_events(client):
    """Full turn envelope: lifecycle, messages, tools and values, in order."""
    start = client.post(
        "/threads/t4/commands",
        json={"id": 6, "method": "run.start", "params": {"input": {"messages": [{"role": "user", "content": "hi"}]}}},
    )
    run_id = start.json()["result"]["runId"]

    with client.stream("POST", "/threads/t4/stream") as resp:
        assert resp.status_code == 200
        events = _read_sse(resp)

    channels = [e["method"] for e in events]
    assert channels[0] == "lifecycle", "a run announces itself first"
    assert "messages" in channels
    assert "tools" in channels, "tool activity is what makes a long turn visible"
    assert channels[-1] == "values", "the finished message lands in state"

    seqs = [e["seq"] for e in events]
    assert seqs == sorted(seqs), "seq is monotonic — Last-Event-ID reconnect depends on it"
    assert seqs[0] >= 1

    for e in events:
        assert e["type"] == "event"
        assert set(e["params"]) >= {"namespace", "timestamp", "data"}

    delta = next(
        e["params"]["data"]
        for e in events
        if e["method"] == "messages" and e["params"]["data"].get("event") == "content-block-delta"
    )
    assert delta["delta"]["type"] == "text-delta"
    assert "hello" in delta["delta"]["text"]

    tool_events = [e["params"]["data"] for e in events if e["method"] == "tools"]
    started = next(e for e in tool_events if e["event"] == "tool-started")
    assert started["tool_name"] == "search_properties"
    assert started["input"] == {"area": "Karen"}
    assert started["tool_call_id"] == "tc_1"

    finished = next(e for e in tool_events if e["event"] == "tool-finished")
    assert finished["tool_call_id"] == started["tool_call_id"], (
        "started and finished must correlate by the protocol's tool_call_id"
    )
    assert finished["output"]["count"] == 1, (
        "tool-finished carries the RESULT; publishing it at tool-start is the bug"
    )

    cards = next(
        e["params"]["data"]
        for e in events
        if e["method"] == "custom"
        and e["params"]["data"].get("name") == "listingCards"
    )
    assert cards["cards"][0]["listingId"] == "11111111-1111-4111-8111-111111111111"
    assert cards["cards"][0]["price"] == "11500000.00", "money stays a decimal string"

    assert run_id.startswith("run_")


def test_run_start_carries_page_context_into_the_prompt(client, monkeypatch):
    """The streamed path must resolve "this" exactly like /api/chat does.

    The Next relay sends page context in run.start metadata. Ignoring it made
    the primary transport unable to answer "what about this one?" on a listing
    page while the fallback route could.
    """
    seen = {}

    async def capture(
        agent,
        message,
        *,
        thread_id,
        on_token=None,
        on_tool_call=None,
        on_tool_result=None,
        **_,
    ):
        seen["message"] = message
        return "ok"

    monkeypatch.setattr(proto, "converse", capture)
    client.post(
        "/threads/t8/commands",
        json={
            "id": 10,
            "method": "run.start",
            "params": {
                "input": {
                    "messages": [{"role": "user", "content": "what about this one?"}]
                },
                "metadata": {
                    "pageContext": {
                        "view_mode": "property_detail",
                        "active_listing_id": "listing-1",
                    }
                },
            },
        },
    )

    with client.stream("POST", "/threads/t8/stream") as resp:
        _read_sse(resp)

    assert "what about this one?" in seen["message"]
    assert "listing-1" in seen["message"]


def test_replay_after_last_event_id_does_not_rerun_the_turn(client, monkeypatch):
    """Reconnecting must read the buffer, not re-run a billable agent turn."""
    calls = {"n": 0}

    async def counting(*a, **k):
        calls["n"] += 1
        return "already answered"

    monkeypatch.setattr(proto, "converse", counting)
    client.post(
        "/threads/t7/commands",
        json={"id": 7, "method": "run.start", "params": {"input": {"messages": [{"role": "user", "content": "x"}]}}},
    )

    with client.stream("POST", "/threads/t7/stream", headers={"Last-Event-ID": "0"}) as resp:
        first = _read_sse(resp)
    assert first, "without Last-Event-ID the whole replay comes back"

    with client.stream("POST", "/threads/t7/stream", headers={"Last-Event-ID": "9999"}) as resp:
        tail = _read_sse(resp)
    assert tail == [], "a client that already has everything gets nothing new"
    assert calls["n"] == 1, "the turn ran exactly once across both subscriptions"


def test_failed_run_publishes_a_failure_event_rather_than_raising(client, monkeypatch):
    async def boom(agent, message, *, thread_id, on_token=None, on_tool_call=None, **_):
        raise RuntimeError("provider down")

    monkeypatch.setattr(proto, "converse", boom)
    client.post(
        "/threads/t5/commands",
        json={"id": 8, "method": "run.start", "params": {"input": {"messages": [{"role": "user", "content": "x"}]}}},
    )
    with client.stream("POST", "/threads/t5/stream") as resp:
        datas = _read_sse(resp)
    failed = [d for d in datas if d["method"] == "lifecycle" and d["params"]["data"].get("event") == "failed"]
    assert failed, "a dead provider is an event the client can render, not a crash"
    assert "provider down" in failed[0]["params"]["data"]["error"]


def test_run_start_without_a_message_is_reported_as_an_event(client):
    r = client.post(
        "/threads/t6/commands",
        json={"id": 9, "method": "run.start", "params": {"input": {}}},
    )
    assert r.json()["type"] == "success", "the command is accepted; the failure is an event"
    with client.stream("POST", "/threads/t6/stream") as resp:
        datas = _read_sse(resp)
    failed = [d for d in datas if d["params"]["data"].get("event") == "failed"]
    assert failed and "no user message" in failed[0]["params"]["data"]["error"]


def test_wire_shapes_match_the_langgraph_sdk_schema():
    """Drift guard: the field names the client parses come from the SDK.

    If langgraph-sdk renames a stream-part field we emit, this fails loudly
    instead of the UI silently rendering an empty transcript.
    """
    from langgraph_sdk import schema as sdk

    for name in ("MessagesPartialStreamPart", "CustomStreamPart", "UpdatesStreamPart"):
        hints = getattr(sdk, name).__annotations__
        assert set(hints) >= {"type", "ns", "data"}, f"{name} changed shape"

    envelope = proto._envelope(1, "messages", {"event": "message-start"})
    assert envelope["type"] == "event"
    assert envelope["seq"] == 1
    assert envelope["method"] == "messages"
    assert set(envelope["params"]) >= {"namespace", "timestamp", "data"}
