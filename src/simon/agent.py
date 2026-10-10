import asyncio
import inspect
import os

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

from simon.settings import get_model_settings, get_observability_settings
from simon.tools import TOOLS

SYSTEM_PROMPT = """I am Simon, the property concierge for Dreamhomesellers in Kenya.

I help customers find, compare, and book viewings for properties, and I help
owners post listings. I search first and answer from what the tools return —
I never invent a property, price, or availability. When a request is vague I
search by meaning; when it has hard numbers I filter with search_properties.
Prices are KES unless a listing says otherwise, and rent figures are monthly
unless the listing carries a different price period.

Before booking a viewing I confirm the property, day, hour, and phone number
with the customer. When I cannot serve a request I say so plainly and offer to
record a lead for the sales team."""


def wire_tracing():
    mine = get_observability_settings()
    if mine.langsmith_tracing and mine.langsmith_api_key.get_secret_value():
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_API_KEY"] = (
            mine.langsmith_api_key.get_secret_value()
        )
        os.environ.setdefault("LANGSMITH_PROJECT", mine.langsmith_project)
    else:
        os.environ.pop("LANGSMITH_TRACING", None)


def build_model():
    mine = get_model_settings()
    if mine.nvidia_api_key.get_secret_value():
        return init_chat_model(
            "nvidia/nemotron-3.5-lightning-30b-a3b",
            model_provider="nvidia",
            api_key=mine.nvidia_api_key.get_secret_value(),
            temperature=0,
        )
    if mine.gemini_api_key.get_secret_value():
        return init_chat_model(
            "google_genai:gemini-3.8-flash",
            api_key=mine.gemini_api_key.get_secret_value(),
            temperature=0,
        )
    if mine.groq_api_key.get_secret_value():
        return init_chat_model(
            "groq:openai/gpt-oss-120b",
            api_key=mine.groq_api_key.get_secret_value(),
            temperature=0,
        )
    if mine.poolside_api_key.get_secret_value():
        return init_chat_model(
            "poolside/laguna-s-2.1",
            model_provider="openai",
            base_url="https://inference.poolside.ai/v1",
            api_key=mine.poolside_api_key.get_secret_value(),
            temperature=0,
            use_responses_api=False,
        )
    raise RuntimeError(
        "No model key found — I need NVIDIA_API_KEY, GEMINI_API_KEY, "
        "GROQ_API_KEY, or POOLSIDE_API_KEY."
    )


def build_agent():
    wire_tracing()
    return create_agent(
        model=build_model(),
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),
    )


async def converse(agent, message: str, *, thread_id: str, on_token=None,
                   on_tool_call=None):
    """Run one turn and return the final answer text.

    Streams through astream_events version v3: the messages and tool_calls
    projections are drained concurrently so tokens and tool activity both
    reach the callbacks as they happen. on_token receives text deltas;
    on_tool_call receives the call object (tool_name, input, output_deltas,
    output, error) and may be sync or async — iterating output_deltas is the
    callback's job, since only one consumer can drain them.
    """
    config = {"configurable": {"thread_id": thread_id}}
    stream = await agent.astream_events(
        {"messages": [{"role": "user", "content": message}]},
        config=config,
        version="v3",
    )

    async def drain_text():
        async for message in stream.messages:
            async for token in message.text:
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

    await asyncio.gather(drain_text(), drain_tools())
    final = await stream.output()
    answer = final["messages"][-1]
    return answer.text if hasattr(answer, "text") else str(answer.content)
