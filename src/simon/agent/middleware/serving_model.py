"""Which model actually served the last turn, and how it is recorded.

Framework fallback is invisible by default: when the primary fails and the
second model answers, nothing says so. Simon spent a production incident
answering from its THIRD provider while every check reported the chain was
healthy -- because the old ``active_provider_name()`` derived the answer from
the cooldown map, i.e. "the first provider that has NOT failed", which is a
different question from "which provider answered".

This records the fact, from the request, on every model call.
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


def model_id(model) -> str:
    """Best available identifier for a chat model, across provider types."""
    for attr in ("model", "model_name", "id"):
        value = getattr(model, attr, None)
        if isinstance(value, str) and value:
            return value
    return type(model).__name__


@wrap_model_call
async def serving_model_middleware(request, handler):
    """Record the model id that is about to serve this call, then run it.

    THREE things here are load-bearing, and each was learned by a failure:

    1. ASYNC. The decorator on a *sync* function implements only
       ``wrap_model_call``; the async path raises "Asynchronous implementation
       of awrap_model_call is not available". Simon streams through
       ``astream_events``, which is async, so a sync hook never runs at all.
       This shipped empty in production exactly once for that reason.

    2. Reads ``ModelRequest.model``, NOT the response. ``ModelResponse`` has
       exactly two fields -- ``result`` and ``structured_response`` -- so there
       is no ``.model`` to read, and a version that looks for one records
       nothing while appearing to work.

    3. Records BEFORE calling the handler, so the id is the model that serves.
       When fallback substitutes a different model, the framework rebuilds the
       request with that model before this hook sees it -- so what gets
       recorded is the answer, not the intention.
    """
    record_serving_model(model_id(request.model))
    return await handler(request)
