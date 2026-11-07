"""Agent assembly: one primary model, framework-managed fallbacks.

The whole resilience story is the middleware list. This file's only job is
wiring it, which is the point: the previous version had the fallback order
spread across a probe function, a cooldown map, two module globals, and a
per-request rebuild in the service boundary.
"""

from __future__ import annotations

import logging

from langchain.agents import create_agent

from simon.agent.middleware import build_middleware, middleware_summary
from simon.agent.providers import model_chain, primary_model

LOGGER = logging.getLogger("simon.agent")


def build_agent(*, checkpointer=None, system_prompt: str | None = None):
    """Build the concierge agent.

    ``checkpointer`` MUST be passed. Building an InMemorySaver here would give
    every request a fresh empty memory, which is the bug that made thread_id
    decorative and multi-turn questions ("compare the first with the second")
    impossible.
    """
    from simon.agent.prompt import SYSTEM_PROMPT

    chain = model_chain()
    LOGGER.info(
        "agent: %d model(s) in fallback order: %s",
        len(chain),
        " -> ".join(str(m) for m in chain),
    )
    for entry in middleware_summary():
        LOGGER.info("  middleware %-16s %s", entry["name"], entry["detail"])

    return create_agent(
        model=primary_model(),
        system_prompt=system_prompt or SYSTEM_PROMPT,
        middleware=build_middleware(),
        checkpointer=checkpointer,
        name="simon-concierge",
    )
