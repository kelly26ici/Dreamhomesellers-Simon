"""Model fallback: fall through the chain, in order, per call.

This replaces a hand-rolled ladder that walked PROVIDER_CHAIN by hand, probed
each provider with a REAL invoke (up to 12s each) on every request, and kept a
cooldown map of module globals. Two consequences of that design are gone:

  * a single question could burn three paid provider calls before being
    answered, because both call sites rebuilt the model per request;
  * a provider that degraded mid-session stayed dead until a restart, because
    the cooldown map only ever recorded failures and never recovery.

ModelFallbackMiddleware is per call, so both are fixed by construction.
"""

from __future__ import annotations

from langchain.agents.middleware import ModelFallbackMiddleware

from simon.agent.providers import model_chain


def build() -> ModelFallbackMiddleware:
    """First configured model is the primary; the rest are the fallbacks."""
    return ModelFallbackMiddleware(*model_chain())
