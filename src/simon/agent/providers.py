"""Model provider chain. Order IS the fallback order; nothing else knows it.

Replaces the hand-rolled ladder in the old agent.py: a per-boot probe invoke,
a cooldown map, and module-global side channels. LangChain's
ModelRetryMiddleware and ModelFallbackMiddleware do all of that now, and do it
per call rather than per process, which is what makes a provider that degrades
mid-session recoverable without a restart.

What is deliberately KEPT from the old implementation: the per-call timeout.
The old code smuggled an httpx.Client through model_kwargs to bound a slow
provider, because `request_timeout` is inert on some integrations. That hack
is gone -- ChatNVIDIA serialises model_kwargs into the request body, so a
Client in there raised "Object of type Client is not JSON serializable" on
every call. `request_timeout` is passed to the constructor instead, and
ModelRetryMiddleware bounds a failure rather than a slow socket.

The timeout is therefore enforced by retry policy, where the framework can
see it, not by a client hack that broke serialization.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from langchain.chat_models import init_chat_model

from simon.settings import MODEL_PROVIDER_DEFAULTS, get_settings

LOGGER = logging.getLogger("simon.agent")

# Per-call model timeout. LangChain takes this at CONSTRUCTION time, not per
# invoke(), so a slow provider has to be bounded where the model is built.
# Kept under the service's turn budget so a dead provider costs a fraction of
# a turn instead of all of it.
MODEL_TIMEOUT = 15.0

# (provider key, model id, extra kwargs). One LINE per provider -- adding a
# fallback is a config change, not a code change.
#
# NVIDIA is primary: it is the only provider recorded as working for this
# concierge. Gemini is deliberately ABSENT: its key is documented dead (401
# ACCESS_TOKEN_TYPE_UNSUPPORTED), and a key-presence-ordered chain would
# otherwise fall back to a provider that cannot answer, silently.
_PROVIDER_BUILDERS: dict[str, dict] = {
    "nvidia": {
        "model": "nvidia/nemotron-3.5-lightning-30b-a3b",
        "kwargs": {"model_provider": "nvidia"},
    },
    "poolside": {
        "model": "laguna-s-2.1",
        "kwargs": {
            "model_provider": "openai",
            "base_url": "https://inference.poolside.ai/v1",
            "use_responses_api": False,
        },
    },
    "groq": {
        "model": "groq:openai/gpt-oss-120b",
        "kwargs": {},
    },
}


def configured_providers() -> tuple[str, ...]:
    """Providers that have a key, in fallback order.

    An absent key means unconfigured, never a runtime failure: the chain skips
    it rather than trying it and failing.
    """
    mine = get_settings()
    return tuple(
        provider
        for provider, _default in MODEL_PROVIDER_DEFAULTS
        if mine.api_key_for(provider)
    )


def provider_model_id(provider: str) -> str:
    """The model id for a provider, honouring a per-provider env override.

    `SIMON_<PROVIDER>_MODEL` was documented in .env.example and read by
    nothing for the life of the service. It becomes real here, so swapping a
    dead provider is an env change and not a code change.
    """
    mine = get_settings()
    return mine.model_override_for(provider) or _PROVIDER_BUILDERS[provider]["model"]


def build_provider_model(provider: str):
    """The chat model for one provider, or None when unconfigured."""
    mine = get_settings()
    key = mine.api_key_for(provider)
    builder = _PROVIDER_BUILDERS.get(provider)
    if not key or builder is None:
        return None
    base_url = (
        mine.poolside_base_url
        if provider == "poolside" and mine.poolside_base_url
        else builder["kwargs"].get("base_url")
    )
    kwargs = dict(builder["kwargs"])
    if base_url:
        kwargs["base_url"] = base_url
    return init_chat_model(
        provider_model_id(provider),
        api_key=key,
        temperature=0,
        # request_timeout is the supported knob. It is passed to the model
        # constructor rather than smuggled through an httpx.Client in
        # model_kwargs: ChatNVIDIA serialises model_kwargs into the request
        # body, and an httpx.Client in there raises
        # "Object of type Client is not JSON serializable" on every call.
        request_timeout=MODEL_TIMEOUT,
        **kwargs,
    )


@lru_cache(maxsize=1)
def model_chain() -> tuple:
    """Every configured provider's model, in fallback order.

    Cached once per process. The old code re-resolved -- and re-probed -- the
    chain on every request, so a request could burn up to three paid probe
    invocations before answering a single question. That is the per-request
    cost bug this removes.
    """
    models = tuple(
        model
        for provider in configured_providers()
        if (model := build_provider_model(provider)) is not None
    )
    if not models:
        missing = ", ".join(
            key.upper() + "_API_KEY" for key, _ in MODEL_PROVIDER_DEFAULTS
        )
        raise RuntimeError(
            f"no model provider configured - set one of: {missing}"
        )
    return models


def primary_model():
    """The first configured model, i.e. the one fallback starts from."""
    return model_chain()[0]
