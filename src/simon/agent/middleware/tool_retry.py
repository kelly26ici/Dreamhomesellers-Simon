"""Tool retry: a tool that fails transiently is worth one more try.

Distinct from tool error handling, which decides what the MODEL is told. This
only decides whether the tool runs again. Both are needed: retry fixes a
flaky Supabase call, and the error handler stops a real failure from handing
the buyer a Postgres error string.
"""

from __future__ import annotations

from langchain.agents.middleware import ToolRetryMiddleware

from simon.settings import get_settings


def build() -> ToolRetryMiddleware:
    mine = get_settings()
    return ToolRetryMiddleware(
        max_retries=mine.model_max_retries,
        backoff_factor=mine.model_backoff_factor,
        initial_delay=mine.model_initial_delay,
        max_delay=mine.model_max_delay,
    )
