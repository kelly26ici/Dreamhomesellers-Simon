"""One concierge turn, streamed.

Kept deliberately separate from agent assembly: this file knows nothing about
providers or middleware, and takes any compiled agent. That is what lets the
protocol adapter and the JSON route share one turn implementation, and what
lets the tests drive it with a stub agent.
"""

from __future__ import annotations

import asyncio
import inspect


async def converse(agent, message: str, *, thread_id: str, on_token=None,
                   on_tool_call=None, on_tool_result=None):
    """Run one turn and return the final answer text.

    Streams through astream_events version v3: the messages and tool_calls
    projections are drained concurrently so tokens and tool activity both
    reach the callbacks as they happen.

    Two tool callbacks, because a tool has two moments that matter:

    - ``on_tool_call`` fires at tool-START. ``call.input`` is populated and
      ``call.output`` is still None. This is what lets a UI show "searching..."
      while the query runs.
    - ``on_tool_result`` fires at tool-END, after ``call.output`` (or
      ``call.error``) has been populated. Publishing both from the start
      callback -- which is what this used to do -- marks a tool finished the
      instant it begins, so the UI says "searched" during the search and the
      finished event carries ``None`` instead of the result.

    ``converse`` owns the drain of ``call.output_deltas``. A tool handle is
    yielded at tool-started, but its terminal fields are only written when the
    tools channel closes; iterating the delta channel is the await point that
    makes ``on_tool_result`` truthful. Simon's tools return dicts, so the loop
    yields nothing today -- it is the synchronization, not the data, that
    matters here. Callbacks may be sync or async.
    """
    config = {"configurable": {"thread_id": thread_id}}
    stream = await agent.astream_events(
        {"messages": [{"role": "user", "content": message}]},
        config=config,
        version="v3",
    )

    async def drain_text():
        async for message_chunk in stream.messages:
            async for token in message_chunk.text:
                if on_token:
                    result = on_token(token)
                    if inspect.isawaitable(result):
                        await result

    async def drain_tools():
        async for call in stream.tool_calls:
            if on_tool_call:
                result = on_tool_call(call)
                if inspect.isawaitable(result):
                    await result

            # Subscribe to the handle's delta channel and drain it. The
            # channel closes on tool-finished / tool-error, and by then the
            # handle carries the terminal output/error. Without this await the
            # "result" callback below would run against a running tool.
            async for _delta in call.output_deltas:
                pass

            if on_tool_result:
                result = on_tool_result(call)
                if inspect.isawaitable(result):
                    await result

    await asyncio.gather(drain_text(), drain_tools())
    final = await stream.output()
    answer = final["messages"][-1]
    return answer.text if hasattr(answer, "text") else str(answer.content)
