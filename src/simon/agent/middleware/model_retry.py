"""Model retry: retry a transient failure in place BEFORE the chain moves on.

Ordering matters and is the whole point of having both this and fallback: a
429 or a socket timeout should not immediately cost a provider switch. Retry
in place first -- with backoff and jitter -- and only give up on a persistent
failure.

Retry bounds are tight on purpose. Every retry is a paid provider call, and a
buyer waiting on a turn that is quietly retrying three times is a buyer who
leaves.
"""

from __future__ import annotations

from langchain.agents.middleware import ModelRetryMiddleware

from simon.settings import get_settings


def build() -> ModelRetryMiddleware:
    mine = get_settings()
    return ModelRetryMiddleware(
        max_retries=mine.model_max_retries,
        backoff_factor=mine.model_backoff_factor,
        initial_delay=mine.model_initial_delay,
        max_delay=mine.model_max_delay,
    )
