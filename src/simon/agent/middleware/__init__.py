"""The middleware stack, in order. This file only assembles; nothing else.

Each middleware is its own module next to this one, because a stack described
in prose is a stack nobody can review, and a stack assembled inline in
build_agent is one nobody can test.

ORDER, outermost first, and why that order:

  1-2  ModelCallLimit, ToolCallLimit. Cheapest guards run first, so a runaway
        turn is stopped before anything expensive happens. This pair is the
        direct fix for the per-request cost bug.
  3-4  ModelRetry, ToolRetry. Transient failures are retried IN PLACE, with
        backoff and jitter, before anything more drastic happens.
  5    ModelFallback. Only after retry has failed, fall through the chain.
        Ordering 4-before-5 is the point: a 429 should not cost a provider
        switch.
  6    ToolError. What the model is TOLD when a tool really fails -- redacted,
        so a Postgres error string never reaches a buyer.
  7-9  PII. Outbound only. On the way out so the concierge can still accept a
        buyer's phone number to schedule a viewing.
  10-11 ContextEditing, Summarization. Context hygiene, oldest-first: drop
        stale tool outputs, then compress the whole history if it is long.
  12   TodoList. Available, dormant unless the model needs it.
  13   ServingModel. Observability. It wraps the model call to see what
        actually served, so it sits innermost.

WHAT IS DELIBERATELY ABSENT:

  HumanInTheLoopMiddleware. `schedule_viewing` books something, and the product
  spec requires explicit confirmation for consequential actions -- but the
  agent already does that conversationally, and framework-level HITL requires
  the client to send `Command(resume=...)`. The website's SSE relay has no
  such path, so enabling it would stall every viewing request forever with no
  way to resume. That needs a protocol change on the website first, which is
  the owner's decision, not a default.

  LLMToolSelectorMiddleware. It spends a model call to pick tools, which costs
  more than it saves with a six-tool registry the model already knows.

  A refusal signal. An earlier revision added an after_model hook that set a
  boolean from a substring match on the answer. It was dropped: it needed a
  dedicated entry-point investigation to test, and "the model may have
  refused" is not something an operator can act on. Observability that answers
  a real question stays; observability that only adds surface does not.
"""

from __future__ import annotations

import logging

from simon.agent.middleware import (
    context_edit,
    fallback,
    limits,
    model_retry,
    pii,
    serving_model,
    summarize,
    todo,
    tool_error,
    tool_retry,
)
from simon.agent.providers import configured_providers, provider_model_id
from simon.settings import get_settings

LOGGER = logging.getLogger("simon.agent")


def build_middleware() -> list:
    """The stack, in the order documented above. Built once per process."""
    stack: list = [
        limits.build_model_limit(),
        limits.build_tool_limit(),
        model_retry.build(),
        tool_retry.build(),
        fallback.build(),
        tool_error.build(),
        *pii.build(),
        context_edit.build(),
        summarize.build(),
        todo.build(),
        serving_model.serving_model_middleware,
    ]
    LOGGER.info(
        "middleware stack (%d): %s",
        len(stack),
        ", ".join(type(m).__name__ for m in stack),
    )
    return stack


def middleware_summary() -> list[dict[str, str]]:
    """One line per entry, for the deploy log and /api/health.

    A stack that cannot be read from outside is a stack nobody can debug.
    """
    mine = get_settings()
    chain = " -> ".join(
        f"{p}({provider_model_id(p)})" for p in configured_providers()
    )
    return [
        {"name": "ModelCallLimit", "detail": f"run_limit={limits.MAX_MODEL_CALLS_PER_TURN}"},
        {"name": "ToolCallLimit", "detail": f"run_limit={limits.MAX_TOOL_CALLS_PER_TURN}"},
        {"name": "ModelRetry", "detail": f"max_retries={mine.model_max_retries}"},
        {"name": "ToolRetry", "detail": f"max_retries={mine.model_max_retries}"},
        {"name": "ModelFallback", "detail": chain},
        {"name": "ToolError", "detail": "redacted: detail logged, buyer gets a category"},
        *[
            {"name": "PII", "detail": f"{t} -> {s} (output only)"}
            for t, s in pii.TYPES
        ],
        {"name": "ContextEditing", "detail": "clears stale tool outputs"},
        {"name": "Summarization", "detail": f"trigger=tokens:{summarize.SUMMARIZE_AT_TOKENS}"},
        {"name": "TodoList", "detail": "framework default, model decides"},
        {"name": "ServingModel", "detail": "records which model answered"},
    ]
