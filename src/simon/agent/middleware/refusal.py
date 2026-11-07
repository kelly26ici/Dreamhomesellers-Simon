"""Refusal signal. Heuristic, records only, changes nothing.

A detector that ACTED on a guess would be worse than one that only reports:
deciding a buyer "was refused" and behaving differently would be invisible
wrangling. So this records a signal /api/health can show, and nothing else. An
operator can then tell "buyer is being refused" from "buyer got answers"
without reading transcripts.
"""

from __future__ import annotations

from langchain.agents.middleware import after_model

_MARKERS = (
    "i can't help with",
    "i cannot help with",
    "i'm not able to",
    "i am not able to",
    "i won't be able to",
    "i can't assist with",
)

_state: dict[str, bool] = {}


def refusal_signalled() -> bool:
    return _state.get("refused", False)


def reset() -> None:
    _state.clear()


@after_model
def refusal_signal_middleware(state, runtime) -> None:
    """Set the flag when a turn ended in prose that looks like a refusal."""
    messages = state.get("messages") or []
    if not messages:
        return
    last = messages[-1]
    if getattr(last, "tool_calls", None):
        _state["refused"] = False  # a tool call means the model is working
        return
    text = (getattr(last, "text", "") or "").lower()
    _state["refused"] = any(marker in text for marker in _MARKERS)
