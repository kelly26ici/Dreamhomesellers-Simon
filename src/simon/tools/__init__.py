from dataclasses import asdict

from langchain.tools import tool

from simon.catalog import (
    agent_for,
    book_viewing,
    browse,
    fetch_card,
    fetch_cards,
    gallery_for,
    normalize_when,
    post_listing,
    save_lead,
    viewings_for,
)
from simon.meaning import nearest


def card_as_dict(card, score: float | None = None):
    data = asdict(card)
    if score is not None:
        data["match_score"] = round(score, 3)
    return data


@tool(parse_docstring=True)
async def search_properties(
    deal: str | None = None,
    kind: str | None = None,
    min_price: float | None = None,
    max_price: float | None = None,
    bedrooms: int | None = None,
    bathrooms: int | None = None,
    city: str | None = None,
    area: str | None = None,
    amenity: str | None = None,
    page: int = 1,
) -> dict:
    """Filter property listings with exact criteria and return matching cards.

    Args:
        deal: Listing intent — "rent", "sale", or "shortlet".
        kind: Property type — "apartment", "house", "office", "land", etc.
        min_price: Minimum price to accept.
        max_price: Maximum price to accept.
        bedrooms: Minimum number of bedrooms.
        bathrooms: Minimum number of bathrooms.
        city: City name, e.g. "Nairobi".
        area: Neighbourhood or location substring, e.g. "Karen".
        amenity: One amenity the property must have, e.g. "parking".
        page: 1-based page number, 6 results per page.
    """
    cards = await browse(
        deal=deal,
        kind=kind,
        min_price=min_price,
        max_price=max_price,
        bedrooms=bedrooms,
        bathrooms=bathrooms,
        city=city,
        area=area,
        amenity=amenity,
        page=max(page, 1) - 1,
    )
    return {"page": page, "count": len(cards), "properties": [asdict(c) for c in cards]}


@tool(parse_docstring=True)
async def describe_property(property_id: str) -> dict:
    """Pull one listing's full details, its photo gallery, and the listing agent.

    Args:
        property_id: The listing's id from a search result.
    """
    card = await fetch_card(property_id)
    if card is None:
        return {"error": f"No property with id {property_id}."}
    images = await gallery_for(property_id)
    profile = await agent_for(card.agent_id) if card.agent_id else None
    return {
        "property": asdict(card),
        "images": images,
        "agent": asdict(profile) if profile else None,
    }


@tool(parse_docstring=True)
async def compare_properties(property_ids: list[str]) -> dict:
    """Put several listings side by side for a comparison.

    Args:
        property_ids: Two to four listing ids to compare.
    """
    ids = property_ids[:4]
    cards = await fetch_cards(ids)
    if not cards:
        return {"error": "None of those ids match a listing."}
    return {"count": len(cards), "properties": [asdict(c) for c in cards]}


@tool(parse_docstring=True)
async def search_by_meaning(query: str, limit: int = 4) -> dict:
    """Find listings by what the buyer described, not by filters — use this when the request is fuzzy or in natural language.

    Args:
        query: The buyer's own words, e.g. "sunny family home near good schools".
        limit: Maximum results to return, at most 6.
    """
    matches = await nearest(query, limit=min(limit, 6))
    return {
        "count": len(matches),
        "properties": [
            card_as_dict(card, score) for card, score in matches
        ],
    }


@tool(parse_docstring=True)
async def create_listing(
    title: str,
    description: str,
    property_type: str,
    listing_type: str,
    price: float,
    location: str,
    city: str = "Nairobi",
    bedrooms: int | None = None,
    bathrooms: int | None = None,
    square_meters: float | None = None,
    furnished: bool = False,
    amenities: list[str] | None = None,
) -> dict:
    """Post a new property listing to the database.

    Args:
        title: Short headline for the listing.
        description: Full description of the property.
        property_type: One of the property types, e.g. "apartment".
        listing_type: "rent", "sale", or "shortlet".
        price: Asking price or rent amount.
        location: Neighbourhood or area name.
        city: City the property sits in.
        bedrooms: Number of bedrooms.
        bathrooms: Number of bathrooms.
        square_meters: Size of the property in square meters.
        furnished: Whether the property comes furnished.
        amenities: Amenities the property offers, e.g. ["parking", "pool"].
    """
    card = await post_listing(
        title=title,
        description=description,
        property_type=property_type,
        listing_type=listing_type,
        price=price,
        location=location,
        city=city,
        bedrooms=bedrooms,
        bathrooms=bathrooms,
        square_meters=square_meters,
        furnished=furnished,
        amenities=amenities,
    )
    return {"created": True, "property": asdict(card)}


@tool(parse_docstring=True)
async def schedule_viewing(
    property_id: str,
    customer_phone: str,
    day: str,
    hour: str,
    customer_name: str | None = None,
    notes: str | None = None,
) -> dict:
    """Book a property viewing for a customer and confirm it immediately.

    Args:
        property_id: The listing to view.
        customer_phone: Customer phone number in any readable format.
        day: Date of the viewing as YYYY-MM-DD.
        hour: Start time as HH:MM (24-hour, local Nairobi time).
        customer_name: Customer's name if known.
        notes: Anything the agent should know before the visit.
    """
    card = await fetch_card(property_id)
    if card is None:
        return {"error": f"No property with id {property_id}."}
    slot = await book_viewing(
        property_id=property_id,
        customer_phone=customer_phone,
        customer_name=customer_name,
        viewing_date=normalize_when(day, hour),
        notes=notes,
    )
    return {
        "booked": True,
        "viewing": asdict(slot),
        "property_title": card.title,
    }


@tool(parse_docstring=True)
async def check_viewings(customer_phone: str) -> dict:
    """List a customer's upcoming and past property viewings.

    Args:
        customer_phone: The phone number the viewings were booked with.
    """
    slots = await viewings_for(customer_phone)
    titles = {s.property_id for s in slots if s.property_id}
    cards = await fetch_cards([str(t) for t in titles]) if titles else []
    by_id = {c.id: c.title for c in cards}
    return {
        "count": len(slots),
        "viewings": [
            {**asdict(s), "property_title": by_id.get(str(s.property_id))}
            for s in slots
        ],
    }


@tool(parse_docstring=True)
async def record_lead(
    customer_phone: str,
    inquiry_type: str = "general",
    customer_name: str | None = None,
    property_id: str | None = None,
    message: str | None = None,
) -> dict:
    """Save a customer inquiry so the sales team can follow up later.

    Args:
        customer_phone: Customer phone number.
        inquiry_type: Kind of inquiry — "general", "viewing", or "listing".
        customer_name: Customer's name if known.
        property_id: Listing the inquiry is about, if any.
        message: What the customer asked for.
    """
    lead = await save_lead(
        customer_phone=customer_phone,
        inquiry_type=inquiry_type,
        customer_name=customer_name,
        property_id=property_id,
        message=message,
    )
    return {"saved": True, "lead": asdict(lead)}


TOOLS = [
    search_properties,
    describe_property,
    compare_properties,
    search_by_meaning,
    create_listing,
    schedule_viewing,
    check_viewings,
    record_lead,
]
