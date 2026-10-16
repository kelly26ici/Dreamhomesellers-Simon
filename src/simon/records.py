from dataclasses import dataclass, field


@dataclass(frozen=True)
class PropertyCard:
    """One published listing offer with its buyer-visible content snapshot.

    Content comes off ``listing_revisions`` through the ``published_revision``
    embed only (ADR-005): the published pointer is the sole public authority,
    so ``id`` is a ``listings.listing_id`` and ``revision_id`` is the revision
    the pointer names. Both are needed downstream — media attaches to the
    revision, not to the listing or the property.

    ``price`` is a decimal STRING, not a float. ``price_amount`` is
    numeric(18,2) and PostgREST would serialize it as a JSON float; the query
    selects it with a ``::text`` cast (ADR-001: money is a decimal string), so
    the value is carried through as text and never re-parsed here.
    """

    id: str
    revision_id: str
    title: str
    description: str
    property_type: str
    listing_kind: str
    price_period: str
    price: str
    currency: str
    location: str
    city: str
    country: str
    bedrooms: int | None = None
    bathrooms: int | None = None
    square_meters: float | None = None
    lot_size_sqm: float | None = None
    property_subtype: str | None = None
    availability: str | None = None
    town: str | None = None
    county: str | None = None
    amenities: list[str] = field(default_factory=list)
    furnished: bool | None = None
    published_at: str | None = None

    @property
    def price_per_sqm(self) -> str | None:
        """Price per square metre as a decimal string, or None when unknowable.

        Stays a string: a float here would reintroduce the rounding error the
        ``::text`` cast was introduced to avoid. The IQ arithmetic uses Decimal
        so the quotient is correctly rounded to two places, matching the
        scale of ``price_amount``.
        """
        from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

        if not self.square_meters or self.square_meters <= 0:
            return None
        try:
            unit = (Decimal(self.price) / Decimal(str(self.square_meters))).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
        except (InvalidOperation, ValueError):
            return None
        return f"{unit:.2f}"


@dataclass(frozen=True)
class PropertyImage:
    """One buyer-visible media row off ``listing_media``.

    Two identity changes from the legacy ``property_images`` shape:

    - media attaches to a ``revision_id``, never to a listing or property;
    - there is no ``url`` column. ``derivative_paths`` holds buyer-safe storage
      paths and is legitimately EMPTY on an RLS-visible row (Phase 3 derivative
      processing has not run for most of the inventory), so callers must render
      a media-pending placeholder rather than treat it as an error.
    """

    id: str
    revision_id: str
    position: int
    kind: str
    is_featured: bool
    paths: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ViewingRequest:
    """A submitted viewing request, as the buyer is told about it.

    The confirmation semantics CHANGED with the domain command: an anonymous
    request always starts in ``pending_contact_verification`` and never passes
    through a "booked/confirmed" state from this service. ``viewing_requests``
    is not anon-readable afterwards either (the read policies need an auth uid
    or org membership), so the reference code is the only handle — it is for
    the human to quote, not for Simon to look up again.
    """

    reference: str
    listing_id: str
    requested_date: str
    duration_minutes: int
    mode: str
    state: str
    request_id: str | None = None
    already_submitted: bool = False


@dataclass(frozen=True)
class Lead:
    """A concierge handoff, as the buyer is told about it.

    Shaped like the legacy ``Lead`` only in role: there is no
    ``customer_phone`` column, no ``customer_email`` column and no ``agent_id``.
    Contact is a jsonb channel/value pair and the owner side is resolved
    server-side; ``handoff_kind`` maps 1:1 from the legacy ``inquiry_type``
    vocabulary ("general", "viewing", "listing").
    """

    reference: str
    handoff_kind: str
    state: str
    handoff_id: str | None = None
    listing_id: str | None = None
    already_recorded: bool = False
