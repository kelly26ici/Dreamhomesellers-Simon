"""Browse published listings with exact criteria."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from simon.catalog import published_grid
from simon.tools import _base as base

# The marketplace ``listing_kind`` vocabulary on the listings root. "shortlet"
# is a real member (verified present), so the concierge can serve short-stay
# requests the same way as sale/rent.
DEAL_VALUES = ("sale", "rent", "shortlet")


class SearchPropertiesArgs(BaseModel):
    """Exact filter criteria for the published grid.

    Every field is optional and unfiltered when absent. The descriptions are
    the text the model reads, so each one names what the field MEANS, not just
    its type.
    """

    model_config = ConfigDict(extra="forbid")

    deal: str | None = Field(
        default=None,
        description=(
            'Offer kind: "sale", "rent", or "shortlet". Leave unset to browse '
            "all offers."
        ),
    )
    kind: str | None = Field(
        default=None,
        description=(
            "Property type, e.g. 'apartment', 'house', 'land'. Matches the "
            "listing's published property type exactly."
        ),
    )
    min_price: float | None = Field(
        default=None, ge=0, description="Lowest price to accept, in KES."
    )
    max_price: float | None = Field(
        default=None, ge=0, description="Highest price to accept, in KES."
    )
    bedrooms: int | None = Field(
        default=None,
        ge=0,
        description="Minimum number of bedrooms (inclusive).",
    )
    bathrooms: int | None = Field(
        default=None,
        ge=0,
        description="Minimum number of bathrooms (inclusive).",
    )
    city: str | None = Field(
        default=None, description='City the property sits in, e.g. "Nairobi".'
    )
    area: str | None = Field(
        default=None,
        description=(
            "Neighbourhood or area substring, matched loosely, e.g. 'Karen' or "
            "'Kantafu, Kangundo Road'."
        ),
    )
    amenity: str | None = Field(
        default=None,
        description="One amenity the property must have, e.g. 'parking'.",
    )
    page: int = Field(
        default=1,
        ge=1,
        description=f"1-based page number; {base.BROWSE_CAP} results per page.",
    )

    @model_validator(mode="after")
    def _sane(self) -> SearchPropertiesArgs:
        if (
            self.min_price is not None
            and self.max_price is not None
            and self.min_price > self.max_price
        ):
            raise ValueError(
                "min_price cannot be higher than max_price — narrow the "
                "band or widen it."
            )
        if self.deal is not None and self.deal.lower() not in DEAL_VALUES:
            raise ValueError(
                f"deal must be one of {', '.join(DEAL_VALUES)}; got "
                f"{self.deal!r}. Ask which kind of offer the buyer means."
            )
        return self


@base.as_tool
async def search_properties(args: SearchPropertiesArgs) -> dict:
    """Filter property listings with exact criteria and return matching cards.

    Use this whenever the buyer gave measurable constraints (price, bedrooms,
    city, property type). Returns the cards plus the page and the count found
    on that page. Result caps, contact handling, and error shaping live in
    ``simon.tools._base`` so they are identical for every tool.
    """
    try:
        cards = await published_grid(
            deal=args.deal.lower() if args.deal else None,
            kind=args.kind.strip() if args.kind else None,
            min_price=args.min_price,
            max_price=args.max_price,
            bedrooms=args.bedrooms,
            bathrooms=args.bathrooms,
            city=args.city.strip() if args.city else None,
            area=args.area.strip() if args.area else None,
            amenity=args.amenity.strip() if args.amenity else None,
            limit=base.BROWSE_CAP,
            page=args.page - 1,
        )
    except Exception as exc:  # noqa: BLE001 - relay, never kill the turn
        return base.error(f"The listing search could not be completed: {exc}")

    return base.guarded(
        {
            "page": args.page,
            "count": len(cards),
            "properties": [_as_dict(card) for card in cards],
            "contact": base.CONTACT_POINTER,
        },
        where="search_properties",
    )


def _as_dict(card) -> dict:
    """The card as the model should see it.

    ``price`` is already a decimal string (ADR-001) and stays one: this module
    never converts money to a float. ``price_per_sqm`` is the only derived
    figure, computed with Decimal so the quotient is rounded the same way
    everywhere.
    """
    return {
        "listing_id": card.id,
        "title": card.title,
        "description": card.description,
        "property_type": card.property_type,
        "property_subtype": card.property_subtype,
        "listing_kind": card.listing_kind,
        "price": card.price,
        "currency": card.currency,
        "price_period": card.price_period,
        "availability": card.availability,
        "location": card.location,
        "city": card.city,
        "county": card.county,
        "country": card.country,
        "bedrooms": card.bedrooms,
        "bathrooms": card.bathrooms,
        "square_meters": card.square_meters,
        "lot_size_sqm": card.lot_size_sqm,
        "amenities": card.amenities,
        "furnished": card.furnished,
        "price_per_sqm": card.price_per_sqm,
        "published_at": card.published_at,
        "contact": base.CONTACT_POINTER,
    }
