"""Call caps. The direct fix for the per-request cost bug.

The old implementation resolved -- and probed -- the provider chain on every
request, so one question could cost three paid invocations before being
answered. These caps make a runaway turn bounded rather than open-ended.

The numbers are deliberately tight. One model call per turn is the normal case
for this concierge (search, then answer); two covers a follow-up search after
a first answer. A turn that needs more is looping, and a loop on a paid
provider is money.
"""

from __future__ import annotations

from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware

MAX_MODEL_CALLS_PER_TURN = 4
MAX_TOOL_CALLS_PER_TURN = 12


def build_model_limit() -> ModelCallLimitMiddleware:
    return ModelCallLimitMiddleware(run_limit=MAX_MODEL_CALLS_PER_TURN)


def build_tool_limit() -> ToolCallLimitMiddleware:
    return ToolCallLimitMiddleware(run_limit=MAX_TOOL_CALLS_PER_TURN)
