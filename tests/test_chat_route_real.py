"""The route must run with the REAL, now-synchronous agent factory.

This exists because of a production outage. Slice 4 moved build_agent out of
main.py and made it synchronous -- construction does no I/O -- but the call
sites still awaited it:

    TypeError: object CompiledStateGraph can't be used in 'await' expression

which returned 500 from /api/chat and 502 from /api/aux. The existing suite
passed throughout, because test_chat_cards.py monkeypatches make_agent with a
stub whose signature happened to match. A stub that matches the code under
test proves nothing about the code.

This test drives the real route with the real factory, and mocks only the
provider boundary -- the one thing that is genuinely not a test.
"""

from __future__ import annotations

import inspect

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatResult

from simon.server.main import app, make_agent


class _StubModel:
    """Answers without a network. Implements only what one turn touches."""

    def bind_tools(self, tools, **kwargs):
        return self

    def with_retry(self, **kwargs):
        return self

    def with_fallbacks(self, fallbacks, **kwargs):
        return self

    def invoke(self, messages, config=None, stop=None, **kwargs):
        return AIMessage(content="Stubbed concierge answer.")

    async def ainvoke(self, messages, config=None, stop=None, **kwargs):
        return self.invoke(messages, config=config, stop=stop, **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[])

    @property
    def _llm_type(self) -> str:
        return "stub"


def test_make_agent_is_synchronous() -> None:
    """The signature itself. If this becomes async again, every await must come back."""
    assert not inspect.iscoroutinefunction(make_agent), (
        "make_agent must stay synchronous: construction performs no I/O. "
        "An async def returning a non-awaited coroutine hands the route a "
        "coroutine instead of a graph, and /api/chat answers 500."
    )


@pytest.mark.filterwarnings("ignore::langchain_core._api.beta_decorator.LangChainBetaWarning")
def test_chat_route_runs_with_the_real_agent_factory(monkeypatch) -> None:
    """End to end through the route; only the model is stubbed.

    The shared secret is sent, because authentication is genuinely on: a test
    that omitted it would be asserting the 401 path, not the route.
    """
    agent = make_agent()
    agent.model = _StubModel()  # type: ignore[assignment]
    monkeypatch.setattr("simon.server.main.make_agent", lambda: agent)

    from simon.server.auth import HEADER_NAME
    from simon.settings import get_settings

    # The route's factory is real, so the credential check is real too. Read
    # the secret from settings rather than hardcoding it: the test asserts the
    # wiring, not the value, and a hardcoded value would rot silently.
    secret = get_settings().shared_secret_value

    client = TestClient(app)
    response = client.post(
        "/api/chat",
        json={"message": "Do you have 3-bedroom homes in Kilimani?"},
        headers={HEADER_NAME: secret} if secret else {},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["message"]["role"] == "assistant"
    assert body["message"]["content"]
