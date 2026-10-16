"""Simon's data access over the LUXONN marketplace schema.

Every read here is a published-only read: ``listings`` is the query root, the
row is gated on ``lifecycle_state = 'published'``, and content resolves ONLY
through the ``published_revision`` FK embed (ADR-005). The embed alias used is
the one the frontend read model proves works —
``listing_revisions!listings_published_revision_fk!inner`` — because a
many-to-one ``!inner`` embed drops any listing whose pointer target RLS hides
from this caller, which is what makes the unpublished/drifted case unreachable
rather than merely filtered.

Writes never INSERT into a marketplace table. Both write paths go through the
owner-domain SECURITY DEFINER commands, which are the ONLY client write path
(anon INSERT on ``viewing_requests`` and ``handoffs`` is refused by RLS, and the
state/publisher columns are trigger-guarded):

  - ``create_viewing_request`` — a viewing request,
  - ``create_handoff``         — a concierge handoff / lead.

Both return a jsonb RESULT envelope (``{"ok": true, ...}`` or
``{"ok": false, "reason_code", "message"}``) instead of raising, so a rejected
request still writes its audit row and the caller can relay the reason.

There is NO anonymous read path to a listing's agent or to a saved customer's
viewings, and that is deliberate:

  - ``agents`` does not exist in this schema. The nearest buyer-visible thing
    is org display data, which is not joinable from ``listings`` for an anon
    caller (organization_id stays restricted), so an anon caller has no way to
    obtain the org id. ``describe_property`` therefore drops the agent block
    entirely rather than substituting data it cannot fetch — a stronger
    guarantee than redacting it.
  - ``viewing_requests`` has no phone-keyed lookup: the read policies require
    an auth uid or org membership. ``viewings_for`` is gone; the reference code
    delivered out-of-band is the only handle.

Contact details are therefore never read by this module at all, and the
redaction pointer lives in ``simon.tools._base``.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

from simon.pool import get_pool
from simon.records import Lead, PropertyCard, PropertyImage, ViewingRequest

# Africa/Nairobi is the marketplace's operating timezone. normalize_when used
# to build a naive datetime and convert it with the SERVER's local timezone,
# which silently meant whatever TZ the host happened to have.
NAIROBI_TZ = ZoneInfo("Africa/Nairobi")

# The published_revision embed alias, verbatim from the reference read model.
# "!inner" is load-bearing: it turns the embed into a true join so a listing
# without a visible published revision cannot appear in the result at all.
PUBLISHED_REVISION_FK = "listing_revisions!listings_published_revision_fk!inner"

# Exactly the reference implementation's column list (read-model.ts REVISION_
# COLUMNS). price_amount carries a ::text cast: it is numeric(18,2), and
# PostgREST would otherwise serialize it as a JSON float — which would let float
# rounding touch money (ADR-001). Verified live: returns "11500000.00".
REVISION_COLUMNS = ",".join(
    [
        "revision_id",
        "title",
        "description",
        "property_type",
        "property_subtype",
        "availability",
        "price_amount:price_amount::text",
        "price_currency",
        "price_period",
        "bedrooms",
        "bathrooms",
        "square_meters",
        "lot_size_sqm",
        "area",
        "city",
        "county",
        "country",
        "amenities",
        "furnished",
        "published_at",
    ]
)

# Parent + embed, the shape every buyer query uses.
PUBLISHED_LISTING_SELECT = (
    f"listing_id, listing_kind, "
    f"published_revision:{PUBLISHED_REVISION_FK}({REVISION_COLUMNS})"
)

# Buyer-safe media columns — exactly the anon grant, nothing else. In
# particular NOT processing_status, moderation_status, original_path or
# organization_id: RLS already restricts rows to processed+approved media that
# a live published pointer targets.
MEDIA_COLUMNS = (
    "media_id, revision_id, position, kind, is_featured, derivative_paths"
)

# Consent recorded natively in this service. "native" is the only provenance
# the domain commands accept — "inherited" consent does not transfer through an
# import and is rejected with invalid_consent.
_NATIVE_CONSENT = {
    "provenance": "native",
    "purpose": (
        "Contact the buyer about their property enquiry from the "
        "Dreamhomesellers concierge."
    ),
}

# An iso-shaped UUID: both the listing ids in tool output and the revision ids
# read back from the embed are UUIDs, and PostgREST 400s on a malformed filter
# value instead of returning empty.
_UUID_LIKE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def _published_first(q):
    """Apply the mandatory published gate and, accepted by every caller, order.

    Every ``listings`` read below ALSO states ``.eq("lifecycle_state",
    "published")`` inline immediately after its ``from_("listings")``, so the
    gate is readable at each call site rather than only inside this helper. It
    is guarded twice on purpose: RLS (``listing_public_read``) enforces it too,
    but a read that relies on the policy alone is one misplaced client away
    from serving a withdrawn listing, and grep has to be able to see the rule.

    Ordering through the embed has exactly one working form in supabase-py:
    ``order("published_at", desc=True, foreign_table="published_revision")``,
    which serializes as the parameter key ``published_revision.order``. A bare
    ``order("published_at")`` 400s ("column listings.published_at does not
    exist") and a dotted ``order("published_revision.published_at")`` 400s with
    PGRST100. postgrest-js's ``referencedTable=`` kwarg is not accepted here.
    """
    return q.eq("lifecycle_state", "published").order(
        "published_at", desc=True, nullsfirst=False, foreign_table="published_revision"
    )


def as_card(row: dict) -> PropertyCard:
    """Map one raw published row to a card. The single raw->domain boundary.

    The embed is a JSON OBJECT, not an array (many-to-one ``!inner``). A row
    whose embed is null is not a card: the pointer target is invisible to this
    caller, which means the content is not buyer-visible.
    """
    revision = row.get("published_revision")
    if not revision:
        raise ValueError(
            "row has no published_revision embed; content must come only off "
            "the published pointer"
        )
    price = revision.get("price_amount")
    if price is None:
        raise ValueError("published revision carries no price_amount")
    return PropertyCard(
        id=row["listing_id"],
        revision_id=revision["revision_id"],
        title=revision["title"],
        description=revision["description"],
        property_type=revision["property_type"],
        listing_kind=row["listing_kind"],
        price=str(price),
        currency=revision["price_currency"],
        price_period=revision["price_period"],
        location=revision["area"],
        city=revision["city"],
        country=revision["country"],
        bedrooms=revision.get("bedrooms"),
        bathrooms=revision.get("bathrooms"),
        square_meters=revision.get("square_meters"),
        lot_size_sqm=revision.get("lot_size_sqm"),
        property_subtype=revision.get("property_subtype"),
        availability=revision.get("availability"),
        town=revision.get("town"),
        county=revision.get("county"),
        amenities=revision.get("amenities") or [],
        furnished=revision.get("furnished"),
        published_at=revision.get("published_at"),
    )


async def published_grid(
    *,
    deal: str | None = None,
    kind: str | None = None,
    min_price: float | None = None,
    max_price: float | None = None,
    bedrooms: int | None = None,
    bathrooms: int | None = None,
    city: str | None = None,
    area: str | None = None,
    amenity: str | None = None,
    limit: int = 6,
    page: int = 0,
) -> list[PropertyCard]:
    """One page of published listings, newest publication first.

    Filters that live on the revision are applied through the embed alias, so
    the ``published_revision.`` prefix is what makes PostgREST treat them as
    join conditions rather than parent columns. ``deal`` is the marketplace
    ``listing_kind`` vocabulary on the listings root (sale / rent / shortlet).
    """
    pool = await get_pool()
    query = (
        pool.from_("listings")
        .select(PUBLISHED_LISTING_SELECT)
        .eq("lifecycle_state", "published")
    )
    query = _published_first(query)
    if deal:
        query = query.eq("listing_kind", deal)
    if kind:
        query = query.eq("published_revision.property_type", kind)
    if min_price is not None:
        query = query.gte("published_revision.price_amount", min_price)
    if max_price is not None:
        query = query.lte("published_revision.price_amount", max_price)
    if bedrooms is not None:
        query = query.gte("published_revision.bedrooms", bedrooms)
    if bathrooms is not None:
        query = query.gte("published_revision.bathrooms", bathrooms)
    if city:
        query = query.ilike("published_revision.city", f"%{city}%")
    if area:
        # The legacy location column carried the same compound "Kantafu, Kangundo
        # Road" string the marketplace area column holds, so the substring
        # search keeps working unchanged.
        query = query.ilike("published_revision.area", f"%{area}%")
    if amenity:
        query = query.cs("published_revision.amenities", [amenity])
    start = max(page, 0) * limit
    rows = (await query.range(start, start + limit - 1).execute()).data
    return [as_card(row) for row in rows if row.get("published_revision")]


async def fetch_published_listing(listing_id: str) -> PropertyCard | None:
    """One published listing by id, or None.

    The UUID guard is not cosmetic: PostgREST 400s on a malformed filter value
    instead of returning empty, and a legacy property id handed back by the
    vector projection or the page context is exactly that case.
    """
    if not _UUID_LIKE.match(listing_id or ""):
        return None
    pool = await get_pool()
    query = _published_first(
        pool.from_("listings")
        .select(PUBLISHED_LISTING_SELECT)
        .eq("lifecycle_state", "published")
    )
    rows = (
        await query.eq("listing_id", listing_id).limit(1).execute()
    ).data
    row = rows[0] if rows else None
    return as_card(row) if row and row.get("published_revision") else None


async def fetch_published_listings(
    listing_ids: list[str],
) -> list[PropertyCard]:
    """Published listings by explicit ids, in the caller's order.

    Mirrors the reference implementation's by-id map: ids that no longer
    resolve to a published, buyer-visible listing are silently dropped, which
    is what makes a stale saved id degrade gracefully instead of 400ing.
    """
    ids = [i for i in (listing_ids or []) if _UUID_LIKE.match(i or "")]
    if not ids:
        return []
    pool = await get_pool()
    query = _published_first(
        pool.from_("listings")
        .select(PUBLISHED_LISTING_SELECT)
        .eq("lifecycle_state", "published")
    )
    rows = (await query.in_("listing_id", ids).execute()).data
    by_id = {
        row["listing_id"]: as_card(row)
        for row in rows
        if row.get("published_revision")
    }
    return [by_id[i] for i in ids if i in by_id]


async def gallery_for_published(revision_id: str) -> list[PropertyImage]:
    """Buyer-safe media for one revision, in display order.

    Media is a SECOND query stitched by revision_id — a revision->media embed
    is impossible for anon, because listing_media's composite FK would have to
    expose the restricted organization_id.

    An empty result is a normal outcome, not an error: only a minority of the
    inventory has run derivative processing, so most published listings have no
    buyer-visible media at all and the caller renders a media-pending
    placeholder.
    """
    if not _UUID_LIKE.match(revision_id or ""):
        return []
    pool = await get_pool()
    rows = (
        await pool.from_("listing_media")
        .select(MEDIA_COLUMNS)
        .in_("revision_id", [revision_id])
        .order("position", nullsfirst=False)
        .execute()
    ).data
    return [
        PropertyImage(
            id=r["media_id"],
            revision_id=r["revision_id"],
            position=r.get("position") or 0,
            kind=r.get("kind") or "image",
            is_featured=bool(r.get("is_featured")),
            # derivative_paths is legitimately null/[] on an RLS-visible row;
            # the caller treats an empty list as "photos pending".
            paths=r.get("derivative_paths") or [],
        )
        for r in rows
    ]


async def request_viewing(
    *,
    listing_id: str,
    contact_channel: str,
    contact_value: str,
    display_name: str | None = None,
    stated_purpose: str,
    viewing_date: str,
    mode: str = "in_person",
    duration_minutes: int = 30,
    notes: str | None = None,
) -> ViewingRequest:
    """Submit a viewing request through the ``create_viewing_request`` command.

    Every anonymous request is created in ``pending_contact_verification`` — the
    service must never claim a viewing is "booked" or "confirmed". The reference
    code returned is the buyer's only handle: the row is not anon-readable
    afterwards.

    The payload mirrors what the website's ViewingRequestForm sends, including
    the per-turn idempotency key: a Simon retry replays the original request
    instead of creating a duplicate.
    """
    result = await _run_domain_command(
        "create_viewing_request",
        {
            "p_listing_id": listing_id,
            "p_contact": {
                "channel": contact_channel,
                "value": contact_value,
                "display_name": display_name,
            },
            "p_consent": dict(_NATIVE_CONSENT),
            "p_stated_purpose": stated_purpose,
            "p_preferred_slot": {
                "mode": mode,
                "duration_minutes": duration_minutes,
                "viewing_date": viewing_date,
                "alternate_viewing_date": None,
            },
            "p_notes": notes,
            "p_idempotency_key": f"simon:{uuid.uuid4()}",
        },
    )
    return _viewing_request_from(listing_id, viewing_date, duration_minutes, mode, result)


def _viewing_request_from(
    listing_id: str,
    viewing_date: str,
    duration_minutes: int,
    mode: str,
    result: dict,
) -> ViewingRequest:
    """Shape a command result into the record the tool layer reports.

    A non-ok result is raised as ``DomainCommandError`` rather than turned into
    a misleading success record — the tool layer catches it and returns the
    reason as a structured message the model can act on.
    """
    if not result.get("ok"):
        raise DomainCommandError(
            result.get("reason_code") or "unavailable",
            result.get("message") or "The request could not be submitted.",
        )
    return ViewingRequest(
        reference=result.get("reference") or "",
        request_id=result.get("request_id"),
        listing_id=listing_id,
        requested_date=viewing_date,
        duration_minutes=duration_minutes,
        mode=mode,
        state=result.get("request_state") or "pending_contact_verification",
        already_submitted=bool(result.get("idempotent_replay")),
    )


async def record_handoff(
    *,
    contact_channel: str,
    contact_value: str,
    display_name: str | None = None,
    handoff_kind: str = "general",
    listing_id: str | None = None,
    viewing_request_id: str | None = None,
    message: str | None = None,
) -> Lead:
    """Record a concierge handoff through the ``create_handoff`` command.

    ``handoff_kind`` maps 1:1 from the legacy ``inquiry_type`` vocabulary
    ("general", "viewing", "listing"). When a viewing request was just created,
    its request_id carries through for continuity on the owner side.
    """
    result = await _run_domain_command(
        "create_handoff",
        {
            "p_listing_id": listing_id,
            "p_viewing_request_id": viewing_request_id,
            "p_handoff_kind": handoff_kind,
            "p_contact": {
                "channel": contact_channel,
                "value": contact_value,
                "display_name": display_name,
            },
            "p_consent": dict(_NATIVE_CONSENT),
            "p_message": message,
            "p_idempotency_key": f"simon:{uuid.uuid4()}",
        },
    )
    if not result.get("ok"):
        raise DomainCommandError(
            result.get("reason_code") or "unavailable",
            result.get("message") or "The enquiry could not be recorded.",
        )
    return Lead(
        reference=result.get("reference") or "",
        handoff_id=result.get("handoff_id"),
        handoff_kind=handoff_kind,
        state=result.get("handoff_state") or "pending_contact_verification",
        listing_id=listing_id,
        already_recorded=bool(result.get("idempotent_replay")),
    )


class DomainCommandError(RuntimeError):
    """A domain command refused the request, carrying its reason_code."""

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.message = message


async def _run_domain_command(name: str, params: dict) -> dict:
    """Call one SECURITY DEFINER domain command over the same anon client.

    Both commands return a jsonb RESULT object rather than raising, so the
    audit row survives a rejected request and the caller can relay the reason.
    """
    pool = await get_pool()
    response = await pool.rpc(name, params).execute()
    data = response.data
    if isinstance(data, list):
        return data[0] if data and isinstance(data[0], dict) else {}
    return data if isinstance(data, dict) else {}


def normalize_when(day: str, hour: str) -> str:
    """Resolve a "YYYY-MM-DD" + "HH:MM" pair into an instant in Africa/Nairobi.

    ``astimezone`` on a naive datetime silently uses the SERVER's local zone,
    which is why the old implementation produced a different instant depending
    on the host's TZ while its docstring promised Nairobi time. Now the
    timezone is attached before the conversion.

    The domain command rejects any slot at or before now(), so a date/time in
    the past has to fail here with a readable message rather than reach the RPC.
    """
    try:
        naive = datetime.fromisoformat(f"{day.strip()}T{hour.strip()}")
    except ValueError as exc:
        raise ValueError(
            "Use the date as YYYY-MM-DD and the time as HH:MM (24-hour), "
            "e.g. 2026-10-20 and 14:00."
        ) from exc
    if naive.tzinfo is not None:
        raise ValueError("Give the date and time separately, without a timezone.")
    aware = naive.replace(tzinfo=NAIROBI_TZ)
    if aware <= datetime.now(NAIROBI_TZ):
        raise ValueError("The viewing date must be in the future.")
    return aware.isoformat()
