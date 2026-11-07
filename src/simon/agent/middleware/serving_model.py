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
    """Record the model id that is about to serve this call.

    Reads the id off ModelRequest.model, NOT off the response. ModelResponse
    has exactly two fields -- `result` and `structured_response` -- so there is
    no `.model` to read, and a version that looks for one records nothing
    while appearing to work. Verified by introspection, not assumption: the
    first implementation did exactly that and shipped empty in production.

    Recording the request also means the id is the model that actually SERVED,
    including when fallback substituted a different one -- the framework
    rebuilds the request with the fallback model before this hook sees it.
    """
    record_serving_model(model_id(request.model))
    return handler(request)


def model_id(model) -> str:
    """Best available identifier for a chat model, across provider types."""
    for attr in ("model", "model_name", "id"):
        value = getattr(model, attr, None)
        if isinstance(value, str) and value:
            return value
    return type(model).__name__
