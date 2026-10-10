import uuid

import chainlit as cl

from simon.agent import build_agent, converse


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("agent", build_agent())
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

    async def on_tool_call(call):
        async with cl.Step(name=call.tool_name, type="tool") as step:
            step.input = str(call.input)
            async for delta in call.output_deltas:
                await step.stream_token(delta)
            step.output = "Done." if not call.error else f"Error: {call.error}"

    await converse(
        agent,
        message.content,
        thread_id=thread_id,
        on_token=on_token,
        on_tool_call=on_tool_call,
    )
    await answer.update()
