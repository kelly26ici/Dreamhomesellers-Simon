"""Context editing: drop stale tool outputs from the prompt.

Complementary to summarization, and not the same job. Summarization compresses
the whole history when it gets long; this removes individual tool results that
are old and large -- a full listing dump from ten turns ago is dead weight the
model is still paying to re-read on every call.

Cheapest middleware in the stack: no model call, no network.
"""

from __future__ import annotations

from langchain.agents.middleware import ClearToolUsesEdit, ContextEditingMiddleware


def build() -> ContextEditingMiddleware:
    return ContextEditingMiddleware(edits=[ClearToolUsesEdit()])
