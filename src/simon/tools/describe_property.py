"""Full detail for one published listing, with its buyer-safe media."""

from __future__ import annotations

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from simon.catalog import fetch_published_listing, gallery_for_published
from simon.tools import _base as base

# Rendered when a published listing has no buyer-visible media. This is a
# NORMAL state, not an error: only a minority of the inventory has run
# derivative processing, so most published listings have an empty gallery. (The
# alternative — inventing a placeholder URL — would break ADR-005: nothing but
# the published revision is authority.)
MEDIA_PENDING = (
    "Photos are still being processed for this listing. The listing agent can "
    "share them through the enquiry form on the listing page."
)


class DescribePropertyArgs(BaseModel):
    """The listing to describe.

    The id is a ``listings.listing_id`` — the id that comes back from a search
    result, not a legacy property id and not a revision id. Anything else is
    answered with an honest miss rather than a guess.
    """

    model_config = ConfigDict(extra="forbid")

    property_id: str = Field(
        validation_alias=AliasChoices("property_id", "listing_id"),
        description=(
            "The listing_id of the listing, exactly as it appeared in a search "
            "result (a UUID)."
        ),
    )


@base.as_tool
async def describe_property(args: DescribePropertyArgs) -> dict:
    """Pull one published listing's full details and its buyer-safe media.

    The gallery is a SECOND query stitched by revision id (a revision->media
    embed is impossible for an anonymous caller). Its entries are buyer-safe
    storage paths and are legitimately empty; when they are, the listing is
    reported as media-pending rather than as a failure.

    There is deliberately NO agent block. The marketplace has no ``agents``
    table and an anonymous caller cannot read ``listings.organization_id``, so
    the org display data is unreachable — which is a stronger guarantee than
    redacting a contact after the fact. The buyer is pointed at the enquiry
    form instead.
    """
    listing_id = args.property_id.strip()
    try:
        card = await fetch_published_listing(listing_id)
    except Exception as exc:  # noqa: BLE001 - relay, never kill the turn
        return base.error(f"The listing could not be looked up: {exc}")

    if card is None:
        return base.error(
            f"No published listing with id {listing_id}. It may have been "
            "withdrawn, may not be published yet, or the id may belong to a "
            "property rather than a listing."
        )

    try:
        images = await gallery_for_published(card.revision_id)
    except Exception as exc:  # noqa: BLE001 - relay, never kill the turn
        return base.error(f"The listing's media could not be read: {exc}")

    media = [
        {
            "media_id": image.id,
            "kind": image.kind,
            "is_featured": image.is_featured,
            "paths": image.paths,
            "revision_id": image.revision_id,
        }
        for image in images
    ]
    # Media is a storage-path concept, not a URL concept: surface the derivative
    # paths and say plainly when there are none.
    gallery = [path for image in images for path in image.paths]

    return base.guarded(
        {
            "property": _as_dict(card),
            "images": media,
            "gallery_paths": gallery,
            "media_pending": not gallery,
            "media_note": None if gallery else MEDIA_PENDING,
            "contact": base.CONTACT_POINTER,
        },
        where="describe_property",
    )


def _as_dict(card) -> dict:
    """Reuse the browse card shape so both tools read the same to the model."""
    from simon.tools.search_properties import _as_dict as card_as_dict

    data = card_as_dict(card)
    data["revision_id"] = card.revision_id
    return data
