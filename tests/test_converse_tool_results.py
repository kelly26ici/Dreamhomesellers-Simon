"""converse() tool-lifecycle tests.

The defect this covers: a tool handle is yielded at tool-START, so reading
``call.output`` from the start callback always produced ``None`` and publishing
"finished" there marked the tool done before it ran. These tests drive a fake
v3 stream whose delta channel closes only after it populates ``output`` — the
same ordering the real ToolCallTransformer uses — and assert the callbacks see
it in that order.
"""

from __future__ import annotations

import pytest

from simon.agent.runtime import converse


class _Deltas:
    """Async channel that closes after writing the terminal result."""

    def __init__(self, call: _Call) -> None:
        self._call = call
        self._delivered = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._delivered:
            raise StopAsyncIteration
        self._delivered = True
        if self._call.error is None:
            self._call.output = {"count": 1, "properties": [{"listing_id": "x"}]}
        raise StopAsyncIteration


class _Call:
    def __init__(self, *, error: str | None = None) -> None:
        self.tool_call_id = "tc_1"
        self.tool_name = "search_properties"
        self.input = {"area": "Karen"}
        self.output = None
        self.error = error
        self.output_deltas = _Deltas(self)


class _Messages:
    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


class _Answer:
    text = "done"


class _Stream:
    def __init__(self, calls: list[_Call]) -> None:
        self.messages = _Messages()
        self._calls = calls

    @property
    def tool_calls(self):
        async def gen():
            for call in self._calls:
                yield call

        return gen()

    async def output(self):
        return {"messages": [_Answer()]}


class _Agent:
    def __init__(self, calls: list[_Call]) -> None:
        self._calls = calls

    async def astream_events(self, *args, **kwargs):
        return _Stream(self._calls)


@pytest.mark.asyncio
async def test_tool_result_fires_only_after_output_exists():
    call = _Call()
    seen: list[tuple[str, object]] = []

    async def on_start(c):
        seen.append(("start", c.output))

    async def on_result(c):
        seen.append(("result", c.output))

    answer = await converse(
        _Agent([call]),
        "find a villa in Karen",
        thread_id="t1",
        on_tool_call=on_start,
        on_tool_result=on_result,
    )

    assert answer == "done"
    assert seen == [
        ("start", None),
        ("result", {"count": 1, "properties": [{"listing_id": "x"}]}),
    ]


@pytest.mark.asyncio
async def test_failed_tool_reaches_the_result_callback_with_its_error():
    call = _Call(error="tool exploded")
    results: list[_Call] = []

    answer = await converse(
        _Agent([call]),
        "find a villa",
        thread_id="t2",
        on_tool_result=results.append,
    )

    assert answer == "done"
    assert len(results) == 1
    assert results[0].error == "tool exploded"
    assert results[0].output is None, "a failed tool has no output to publish"


@pytest.mark.asyncio
async def test_each_tool_call_produces_exactly_one_start_and_one_result():
    calls = [_Call(), _Call()]
    starts: list[_Call] = []
    results: list[_Call] = []

    await converse(
        _Agent(calls),
        "compare two",
        thread_id="t3",
        on_tool_call=starts.append,
        on_tool_result=results.append,
    )

    assert len(starts) == 2
    assert len(results) == 2
