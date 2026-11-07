"""Which model actually served the last turn, and how it is recorded.

Framework fallback is invisible by default: when the primary fails and the
second model answers, nothing says so. Simon spent a production incident
answering from its THIRD provider while every check reported the chain was
healthy -- because the old ``active_provider_name()`` derived the answer from
the cooldown map, i.e. "the first provider that has NOT failed", which is a
different question from "which provider answered".

This records the fact, from the response, on every successful model call.
"""

from __future__ import annotations

from langchain.agents.middleware import wrap_model_call

# Module-level, deliberately: the value has to survive across the agent's
# internal step boundaries, and /api/health has to read it without a
# checkpointer round trip.
_state: dict[str, str] = {}


def record_serving_model(model_id: str) -> None:
    _state["model"] = model_id


def serving_model_id() -> str | None:
    return _state.get("model")


def reset_serving_model() -> None:
    _state.clear()


@wrap_model_call
def serving_model_middleware(request, handler):
    """Record the model id that produced this response."""
    response = handler(request)
    model = getattr(response, "model", None)
    if model is not None:
        record_serving_model(str(model))
    return response
