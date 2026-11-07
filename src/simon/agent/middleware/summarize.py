"""Summarization: compress long threads before they exhaust the context.

This service keeps conversation history for two weeks, so a returning buyer's
thread is long by design and not by accident. Without this, a long thread
either fails the context window or gets truncated arbitrarily.

ABSOLUTE tokens, not a fraction -- that is a discovered constraint, not a
preference. A fractional trigger requires the model to publish a
max_input_tokens profile, and none of the three configured providers do:
constructing with ("fraction", 0.75) raises "Model profile information is
required". 100k sits well under a 128k context while still catching the long
threads.
"""

from __future__ import annotations

from langchain.agents.middleware import SummarizationMiddleware

from simon.agent.providers import primary_model

SUMMARIZE_AT_TOKENS = 100_000
KEEP_LAST_MESSAGES = 20


def build() -> SummarizationMiddleware:
    return SummarizationMiddleware(
        model=primary_model(),
        trigger=("tokens", SUMMARIZE_AT_TOKENS),
        keep=("messages", KEEP_LAST_MESSAGES),
    )
