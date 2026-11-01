import uuid

import chainlit as cl

from simon.agent import build_agent, converse


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("agent", await build_agent())
    cl.user_session.set("thread_id", str(uuid.uuid4()))
    await cl.Message(
        content=(
            "I'm **Simon**, the Dreamhomesellers property concierge. "
            "Tell me what you're looking for — a neighbourhood, a budget, "
            "a vibe — and I'll pull the listings that match."
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    agent = cl.user_session.get("agent")
    thread_id = cl.user_session.get("thread_id")
    answer = cl.Message(content="")

    async def on_token(token: str):
        await answer.stream_token(token)

    # One step per tool call, opened at start and closed when the result
    # exists. converse() now owns draining output_deltas, so the callback no
    # longer subscribes to it (a second subscriber raises).
    steps: dict[str, cl.Step] = {}

    def call_id(call) -> str:
        value = getattr(call, "tool_call_id", None) or getattr(call, "id", None)
        return str(value) if value else uuid.uuid4().hex

    async def on_tool_call(call):
        name = (
            getattr(call, "tool_name", None)
            or getattr(call, "name", None)
            or "tool"
        )
        step = cl.Step(name=name, type="tool")
        step.input = str(getattr(call, "input", None) or {})
        await step.send()
        steps[call_id(call)] = step

    async def on_tool_result(call):
        step = steps.pop(call_id(call), None)
        if step is None:
            return
        error = getattr(call, "error", None)
        step.output = f"Error: {error}" if error else "Done."
        await step.update()

    await converse(
        agent,
        message.content,
        thread_id=thread_id,
        on_token=on_token,
        on_tool_call=on_tool_call,
        on_tool_result=on_tool_result,
    )
    await answer.update()
