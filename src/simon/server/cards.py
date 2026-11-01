"""Buyer-safe listing cards extracted from tool results.

Why this exists rather than letting the browser read tool output:

- Tool payloads are written for the MODEL — descriptions, amenities, derived
  figures, and a contact pointer. The page needs a small, stable card shape,
  and shipping the whole payload would put every fact the model saw into the
  transcript and into persisted chat metadata.
- One shape for every tool. ``search_properties`` and ``compare_properties``
  return ``properties: [...]``; ``describe_property`` returns
  ``property: {...}``. A client should not have to know which tool ran to
  render a card.

The card is a REFERENCE, not authority. It carries a listing id and display
facts copied from the same published-only query the tool already ran; the
frontend links to ``/property/{listingId}``, which re-reads the published
revision from Postgres. Nothing here can surface an unpublished listing.

Money stays a decimal string (ADR-001). Formatting is the host's job: the
service reports ``price`` + ``currency`` + ``pricePeriod`` and the Next app
renders them with the same ``formatListingPrice`` used by the property grid.
"""

from __future__ import annotations

from typing import Any

# A turn that renders more than this is unusable in a chat transcript, and the
# tools already cap each individual result set. This bounds the accumulation
# across several tool calls in one turn (search -> compare -> describe).
MAX_CARDS = 12


def cards_from_tool_result(output: Any) -> list[dict[str, Any]]:
    """Extract card references from one tool's terminal output.

    Shape-driven rather than tool-name-driven: any result that carries a
    ``properties`` list or a single ``property`` object contributes cards.
    Error envelopes (``{"error": ...}``) contribute nothing, so a failed
    lookup never renders an empty or invented card.
    """
    if not isinstance(output, dict) or "error" in output:
        return []

    rows: list[Any] = []
    properties = output.get("properties")
    if isinstance(properties, list):
        rows.extend(properties)
    single = output.get("property")
    if isinstance(single, dict):
        rows.append(single)

    cards: list[dict[str, Any]] = []
    for row in rows:
        card = _card(row)
        if card is not None:
            cards.append(card)
    return cards


def merge_cards(
    existing: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
    *,
    seen: set[str],
) -> list[dict[str, Any]]:
    """Append only cards this turn has not published yet, up to ``MAX_CARDS``.

    A turn commonly searches and then describes one of the results, so the
    same listing arrives twice. Re-publishing it would duplicate the card in
    the transcript and in the replayed buffer; the first card wins because it
    is the one the model saw when it answered.
    """
    fresh: list[dict[str, Any]] = []
    for card in incoming:
        if len(existing) >= MAX_CARDS:
            break
        listing_id = card.get("listingId")
        if not isinstance(listing_id, str) or listing_id in seen:
            continue
        seen.add(listing_id)
        existing.append(card)
        fresh.append(card)
    return fresh


def _card(row: Any) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        return None

    listing_id = row.get("listing_id")
    title = row.get("title")
    if not isinstance(listing_id, str) or not listing_id:
        return None
    if not isinstance(title, str) or not title:
        return None

    return {
        "listingId": listing_id,
        "title": title,
        "propertyType": _text(row.get("property_type")),
        "location": _text(row.get("location")),
        "price": _text(row.get("price")),
        "currency": _text(row.get("currency")),
        "pricePeriod": _text(row.get("price_period")),
        "bedrooms": _integer(row.get("bedrooms")),
        "bathrooms": _integer(row.get("bathrooms")),
    }


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _integer(value: Any) -> int | None:
    # bool is an int subclass; a boolean bedroom count would be nonsense.
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value
