"""The non-streaming route must carry the same cards as the protocol stream.

A stream failure falls back to POST /api/chat. If that route dropped the cards
the model already found, the answer would silently lose its listings at exactly
the moment the service is degraded.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import simon.server.main as main


def _row() -> dict[str, Any]:
    return {
        "listing_id": "listing-1",
        "title": "Karen Ridge Villa",
        "property_type": "villa",
        "location": "Karen, Nairobi",
        "price": "85000000",
        "currency": "KES",
        "price_period": "total",
        "bedrooms": 5,
        "bathrooms": 4,
    }


async def test_chat_returns_listing_cards(monkeypatch) -> None:
    async def fake_make_agent() -> object:
        return object()

    async def fake_converse(
        agent: object,
        message: str,
        *,
        thread_id: str,
        on_token=None,
        on_tool_call=None,
        on_tool_result=None,
    ) -> str:
        assert on_tool_result is not None
        on_tool_result(SimpleNamespace(error=None, output={"properties": [_row()]}))
        return "I found one residence."

    monkeypatch.setattr(main, "make_agent", fake_make_agent)
    monkeypatch.setattr(main, "converse", fake_converse)

    result = await main.chat(main.ChatRequest(message="find a villa"))

    assert result["message"]["content"] == "I found one residence."
    assert result["message"]["listingCards"] == [
        {
            "listingId": "listing-1",
            "title": "Karen Ridge Villa",
            "propertyType": "villa",
            "location": "Karen, Nairobi",
            "price": "85000000",
            "currency": "KES",
            "pricePeriod": "total",
            "bedrooms": 5,
            "bathrooms": 4,
        }
    ]


async def test_chat_drops_cards_from_failed_tools(monkeypatch) -> None:
    async def fake_make_agent() -> object:
        return object()

    async def fake_converse(
        agent: object,
        message: str,
        *,
        thread_id: str,
        on_token=None,
        on_tool_call=None,
        on_tool_result=None,
    ) -> str:
        assert on_tool_result is not None
        on_tool_result(
            SimpleNamespace(error="lookup failed", output={"properties": [_row()]})
        )
        return "The lookup failed."

    monkeypatch.setattr(main, "make_agent", fake_make_agent)
    monkeypatch.setattr(main, "converse", fake_converse)

    result = await main.chat(main.ChatRequest(message="find a villa"))

    assert result["message"]["listingCards"] == []
