"""Card extraction contract tests.

The browser renders what this module returns, so the tests pin the exact
buyer-safe shape and the two rules that matter: money stays a decimal string,
and an error envelope never produces a card.
"""

from __future__ import annotations

from simon.server.cards import cards_from_tool_result, merge_cards

LISTING_ID = "11111111-1111-4111-8111-111111111111"
OTHER_ID = "22222222-2222-4222-8222-222222222222"


def _row(listing_id: str = LISTING_ID, **overrides):
    row = {
        "listing_id": listing_id,
        "title": "Karen Ridge Villa",
        "property_type": "house",
        "location": "Karen, Nairobi",
        "price": "11500000.00",
        "currency": "KES",
        "price_period": "total",
        "bedrooms": 4,
        "bathrooms": 3,
        "description": "A long model-facing description that must not leak.",
        "contact": "Reach the listing agent through the enquiry form.",
    }
    row.update(overrides)
    return row


def test_search_result_becomes_a_small_card():
    cards = cards_from_tool_result({"count": 1, "properties": [_row()]})

    assert cards == [
        {
            "listingId": LISTING_ID,
            "title": "Karen Ridge Villa",
            "propertyType": "house",
            "location": "Karen, Nairobi",
            "price": "11500000.00",
            "currency": "KES",
            "pricePeriod": "total",
            "bedrooms": 4,
            "bathrooms": 3,
        }
    ]
    # Model-facing prose and the contact pointer stay out of the card.
    assert "description" not in cards[0]
    assert "contact" not in cards[0]


def test_describe_result_uses_the_single_property():
    cards = cards_from_tool_result({"property": _row(OTHER_ID)})

    assert [c["listingId"] for c in cards] == [OTHER_ID]


def test_error_envelope_produces_no_card():
    assert cards_from_tool_result({"error": "No published listing with id x"}) == []


def test_rows_without_identity_are_dropped():
    cards = cards_from_tool_result(
        {
            "properties": [
                _row(),
                {"title": "No id"},
                {"listing_id": OTHER_ID},
                "not a row",
            ]
        }
    )

    assert [c["listingId"] for c in cards] == [LISTING_ID]


def test_money_is_never_coerced_to_a_float():
    cards = cards_from_tool_result({"properties": [_row(price="11500000.00")]})

    assert isinstance(cards[0]["price"], str)


def test_merge_publishes_each_listing_once_and_caps_the_turn():
    existing: list[dict] = []
    seen: set[str] = set()

    first = merge_cards(existing, cards_from_tool_result({"properties": [_row()]}), seen=seen)
    second = merge_cards(existing, cards_from_tool_result({"property": _row()}), seen=seen)

    assert len(first) == 1
    assert second == [], "a search-then-describe turn must not duplicate the card"

    many = [
        {"listingId": f"{i:08d}-0000-4000-8000-000000000000", "title": f"L{i}"}
        for i in range(20)
    ]
    fresh = merge_cards([], many, seen=set())

    assert len(fresh) == 12, "one turn cannot render an unbounded card list"
