"""Put a handful of published listings side by side."""

from __future__ import annotations

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from simon.catalog import fetch_published_listings
from simon.tools import _base as base


class ComparePropertiesArgs(BaseModel):
    """Two to four listing ids to compare.

    A single id is not a comparison, and more than four cannot be read in a
    chat turn — both are refused here with a readable reason rather than
    silently truncated, because truncation hides from the model that it asked
    for something it did not get.
    """

    model_config = ConfigDict(extra="forbid")

    property_ids: list[str] = Field(
        validation_alias=AliasChoices("property_ids", "listing_ids"),
        min_length=2,
        max_length=base.COMPARE_CAP,
        description=(
            "The property_ids to compare, exactly as they appeared in search "
            "results. Two to four of them."
        ),
    )

    @model_validator(mode="after")
    def _unique(self) -> "ComparePropertiesArgs":
        seen = [i.strip() for i in self.property_ids]
        if len(set(seen)) != len(seen):
            raise ValueError(
                "The same listing id appears more than once — give distinct "
                "listings for a comparison."
            )
        return self


@base.as_tool
async def compare_properties(args: ComparePropertiesArgs) -> dict:
    """Put several published listings side by side for a comparison.

    Ids that no longer resolve to a published listing are silently dropped —
    that is what makes a stale saved id degrade gracefully. Only when NOTHING
    resolves does this return a structured error, mirroring the reference
    implementation's saved-listings behaviour.
    """
    ids = [i.strip() for i in args.property_ids if i.strip()]
    try:
        cards = await fetch_published_listings(ids[: base.COMPARE_CAP])
    except Exception as exc:  # noqa: BLE001 - relay, never kill the turn
        return base.error(f"The comparison could not be completed: {exc}")

    if not cards:
        return base.error(
            "None of those ids match a published listing. They may have been "
            "withdrawn, may not be published yet, or may be property ids "
            "rather than listing ids."
        )

    return base.guarded(
        {
            "count": len(cards),
            "requested": len(ids),
            "dropped": [i for i in ids if i not in {c.id for c in cards}],
            "properties": [_as_dict(card) for card in cards],
            "contact": base.CONTACT_POINTER,
        },
        where="compare_properties",
    )


def _as_dict(card) -> dict:
    """The comparison row: identity, money, size, and location only.

    Descriptions are deliberately excluded — four paragraphs of prose side by
    side is not a comparison, and the model can call describe_property for the
    full text of the one the buyer narrows to. ``best_value_per_sqm`` is the
    one derived figure that makes a comparison decidable.
    """
    return {
        "listing_id": card.id,
        "title": card.title,
        "property_type": card.property_type,
        "listing_kind": card.listing_kind,
        "price": card.price,
        "currency": card.currency,
        "price_period": card.price_period,
        "price_per_sqm": card.price_per_sqm,
        "bedrooms": card.bedrooms,
        "bathrooms": card.bathrooms,
        "square_meters": card.square_meters,
        "lot_size_sqm": card.lot_size_sqm,
        "location": card.location,
        "city": card.city,
        "county": card.county,
        "amenities": card.amenities,
        "furnished": card.furnished,
        "availability": card.availability,
        "published_at": card.published_at,
        "contact": base.CONTACT_POINTER,
    }
