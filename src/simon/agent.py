import asyncio
import inspect
import logging
import os
import time

import httpx

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

from simon.settings import get_model_settings, get_observability_settings
from simon.tools import TOOLS

LOGGER = logging.getLogger("simon.agent")

# Providers that failed a probe or timed out a turn, with the time they may be
# retried. Probing once at boot is not enough: a provider can pass a probe and
# degrade minutes later (NVIDIA did exactly that), and a per-boot decision would
# then hang every turn until the next restart.
_provider_cooldown: dict[str, float] = {}
COOLDOWN_SECONDS = 300.0


SYSTEM_PROMPT = """I am Simon, the property concierge for Dreamhomesellers in Kenya.

I help buyers find and compare residences, and I submit viewing requests for the
sales team to confirm. Listing creation belongs to the owner's authenticated
portal, not to me — if someone asks me to post a property I point them there.
I never invent a property, price, or availability: I search first and answer only
from what the tools return. Descriptions without hard numbers still go through
search_properties using the nearest concrete filters; the by-meaning tool is
unavailable until its index is rebuilt, and I say so plainly rather than
pretending it found nothing.
Prices are KES unless a listing says otherwise, and rent figures are monthly
unless the listing carries a different price period.

Before submitting a viewing I confirm the property, day, hour, and phone number
with the buyer. A submitted viewing request is NOT a confirmed booking — every
one starts in contact verification, and I say that rather than implying it is
secured. I cannot look up whether a viewing already exists: there is no safe way
to find someone's bookings from a chat, so when asked I point the buyer to the
sales team rather than guessing. When I cannot serve a request I say so plainly
and offer to record a lead for the sales team to follow up."""


def wire_tracing():
    mine = get_observability_settings()
    if mine.langsmith_tracing and mine.langsmith_api_key.get_secret_value():
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_API_KEY"] = (
            mine.langsmith_api_key.get_secret_value()
        )
        os.environ.setdefault("LANGSMITH_PROJECT", mine.langsmith_project)
        # No workspace id is set here on purpose: the lsv2_sk service key is
        # workspace-scoped natively, and LANGSMITH_WORKSPACE_ID (when needed
        # for an lsv2_pt personal token) is read straight from .env by the SDK.
    else:
        os.environ.pop("LANGSMITH_TRACING", None)


# Ordered fallback chain. NVIDIA is primary: it is the only provider recorded
# as working for this concierge (simon-boundary.md: build a Kenya-focused eval
# before replacing a working model). Gemini is deliberately ABSENT — its key is
# documented dead (401 ACCESS_TOKEN_TYPE_UNSUPPORTED), and a key-presence-ordered
# chain would otherwise fall back to a provider that cannot answer, silently.
PROVIDER_CHAIN = ("nvidia", "poolside", "groq")

# Per-call model timeout. LangChain takes this at CONSTRUCTION time, not per
# invoke() call, so a slow provider has to be bounded where the model is built.
# Kept under the service's 45s turn budget so a dead provider costs a fraction
# of a turn instead of all of it.
PROBE_TIMEOUT = 15.0


def _http_client() -> httpx.Client:
    """One bounded HTTP client shared by every provider.

    ``request_timeout`` is inert on these integrations — LangChain transfers it
    into model_kwargs and the provider SDK then applies its own 60s socket
    default, which is why a dead NVIDIA endpoint burned 60s despite the setting.
    An explicit httpx.Client is the knob that actually bounds the read.
    """
    return httpx.Client(timeout=httpx.Timeout(PROBE_TIMEOUT))


def _provider_model(provider: str):
    """Build the chat model for one provider, or None when unconfigured."""
    mine = get_model_settings()
    if provider == "nvidia":
        key = mine.nvidia_api_key.get_secret_value()
        if key:
            return init_chat_model(
                "nvidia/nemotron-3.5-lightning-30b-a3b",
                model_provider="nvidia",
                api_key=key,
                temperature=0,
                http_client=_http_client(),
            )
    elif provider == "poolside":
        key = mine.poolside_api_key.get_secret_value()
        if key:
            return init_chat_model(
                "poolside/laguna-s-2.1",
                model_provider="openai",
                base_url="https://inference.poolside.ai/v1",
                api_key=key,
                temperature=0,
                use_responses_api=False,
                http_client=_http_client(),
            )
    elif provider == "groq":
        key = mine.groq_api_key.get_secret_value()
        if key:
            return init_chat_model(
                "groq:openai/gpt-oss-120b",
                api_key=key,
                temperature=0,
                http_client=_http_client(),
            )
    return None


def _probe(model, timeout: float = 12.0) -> bool:
    """Is this model actually answering right now?

    A key-presence chain is not enough: a provider can be configured and dead —
    NVIDIA's endpoint began timing out at 60s while its key was present, so the
    chain selected it and every turn hung until the service's own 45s budget
    expired.

    The timeout is deliberately tight (12s, not 60). A dead provider costs a
    fraction of a turn instead of all of it, and the whole resolve stays well
    inside the turn budget even when the first two providers are down. A ladder
    of three 60s probes would defeat the point.

    This runs per process, not per request: the service builds the agent once at
    boot and the resolved model is a module-level fact, so a provider outage is
    paid for on the next restart rather than on every visitor's message.
    """
    try:
        model.invoke("reply with the single word: ok", timeout=timeout)
        return True
    except Exception:
        return False


def build_model():
    """Resolve the first provider that is configured AND answering.

    Deliberately not key-presence order: a documented-dead key must not be able
    to answer first, and a degraded-but-keyed provider must not be able to hang
    every turn. The resolved provider is reported by /api/health so which one is
    serving is visible rather than inferred.
    """
    now = time.monotonic()
    for provider in PROVIDER_CHAIN:
        if _provider_cooldown.get(provider, 0.0) > now:
            continue
        model = _provider_model(provider)
        if model is None:
            continue
        if _probe(model):
            return model
        _provider_cooldown[provider] = now + COOLDOWN_SECONDS
        LOGGER.warning(
            "provider %s is not answering; cooling it down for %.0fs and trying "
            "the next one", provider, COOLDOWN_SECONDS,
        )
    raise RuntimeError(
        f"No model key found for the configured chain "
        f"({' -> '.join(PROVIDER_CHAIN)}). I need one of: {configured}."
    )


async def build_agent(*, checkpointer=None):
    """Build the concierge agent.

    ``checkpointer`` is the shared, process-wide LangGraph saver. It MUST be
    passed: building an ``InMemorySaver`` here would give every request a fresh
    empty memory, which is exactly the bug that made ``thread_id`` decorative
    and multi-turn questions ("compare the first with the second") impossible.
    """
    wire_tracing()
    if checkpointer is None:
        from simon.checkpointer import get_checkpointer

        checkpointer = await get_checkpointer()
    return create_agent(
        model=build_model(),
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
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


def active_provider_name() -> str | None:
    """Which provider is currently serving, if one has been resolved.

    The service needs this to attribute a failed turn to a provider. Derived
    from the cooldown map rather than a stored "current" pointer, because
    build_model() is called per request and the resolved provider can change
    between requests when one is cooling down.
    """
    for provider in PROVIDER_CHAIN:
        if provider not in _provider_cooldown:
            try:
                if _provider_model(provider) is not None:
                    return provider
            except Exception:  # noqa: BLE001
                continue
    return None


def mark_provider_unavailable(provider: str) -> None:
    """Cool a provider down after a turn it could not serve.

    Called by the service boundary when a turn times out, so the next request
    skips the provider that just failed instead of proving it again at the cost
    of another full turn budget.
    """
    _provider_cooldown[provider] = time.monotonic() + COOLDOWN_SECONDS
    LOGGER.warning("provider %s cooled down for %.0fs after a failed turn",
                   provider, COOLDOWN_SECONDS)


def reset_provider_cooldowns() -> None:
    _provider_cooldown.clear()
