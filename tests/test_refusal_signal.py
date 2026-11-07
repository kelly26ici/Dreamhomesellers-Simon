"""refusal_signal must fire, and must be driven through its real entry point.

The serving_model bug was a test that drove the wrong wrapper: sync tests
passed while the async agent never invoked the hook. This test drives
`after_model(...)` — the method the framework calls — so the same class of
mistake cannot pass here.
"""
from __future__ import annotations

from simon.agent.middleware import refusal as rf


class _Message:
    def __init__(self, text: str, tool_calls=None) -> None:
        self.text = text
        self.tool_calls = tool_calls or []


def _run(state: dict) -> None:
    rf.reset()
    rf.refusal_signal_middleware.after_model(state, None)


def test_hook_fires_on_its_real_entry_point() -> None:
    """The regression guard: the hook must be invocable as the agent invokes it."""
    _run({"messages": [_Message("Here are three homes in Kilimani.")]})
    assert rf.refusal_signalled() is False


def test_flags_a_refusal_phrase() -> None:
    _run({"messages": [_Message("I can't help with that request.")]})
    assert rf.refusal_signalled() is True


def test_a_tool_call_means_working_not_refusing() -> None:
    _run({"messages": [_Message("", tool_calls=[{"name": "search_properties"}])]})
    assert rf.refusal_signalled() is False


def test_empty_state_is_safe() -> None:
    _run({"messages": []})
    assert rf.refusal_signalled() is False


def test_health_exposes_the_signal() -> None:
    """A field nothing reads is decoration."""
    from pathlib import Path

    main = Path(
        r"C:\Users\rexk6\Coding\learning\Dreamhomesellers-Simon"
        r"\src\simon\server\main.py"
    ).read_text(encoding="utf-8")
    assert '"refusal_signal": refusal_signalled()' in main
