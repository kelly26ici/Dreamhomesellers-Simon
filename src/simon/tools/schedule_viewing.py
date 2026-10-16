"""Submit a viewing request for a published listing.

THE CONFIRMATION SEMANTICS CHANGED, and the concierge must say so.

The old implementation INSERTED with status='confirmed' and told the customer a
viewing was booked. That is no longer true, and it could not be: the marketplace
write path is the ``create_viewing_request`` domain command, which starts
EVERY anonymous request in ``pending_contact_verification``. Contact must be
verified before the request becomes owner-visible, and ``viewing_requests`` is
not readable by an anonymous caller afterwards anyway.

So this tool reports what actually happened: the request was SUBMITTED, with a
reference code the buyer can quote, and the owner side follows up. It never says
"booked", "confirmed", or "scheduled".
"""

from __future__ import annotations

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

from simon.catalog import (
    DomainCommandError,
    fetch_published_listing,
    normalize_when,
    request_viewing,
)
from simon.tools import _base as base

# duration_minutes is bounded by the domain command (15..480); mirror it here so
# a bad value is refused with a reason instead of a generic RPC message.
MIN_DURATION = 15
MAX_DURATION = 480

DEFAULT_PURPOSE = "View the property with the concierge's help."


class ScheduleViewingArgs(BaseModel):
    """A viewing request: which listing, which slot, how to reach the buyer.

    The phone number is a CONTACT CHANNEL — the value the owner follows up on.
    It is never used as a lookup key, because ``viewing_requests`` is
    explicitly "never keyed by phone" in this schema.
    """

    model_config = ConfigDict(extra="forbid")

    property_id: str = Field(
        validation_alias=AliasChoices("property_id", "listing_id"),
        description=(
            "The listing_id of the listing to view, exactly as it appeared in "
            "a search result (a UUID)."
        ),
    )
    customer_phone: str = Field(
        description=(
            "The buyer's phone number, in any readable format. Used only to "
            "reach them about this request."
        ),
    )
    day: str = Field(
        description="Date of the viewing as YYYY-MM-DD, in the future."
    )
    hour: str = Field(
        description=(
            "Start time as HH:MM, 24-hour, interpreted as local Nairobi time."
        ),
    )
    customer_name: str | None = Field(
        default=None, description="The buyer's name if known."
    )
    notes: str | None = Field(
        default=None,
        description="Anything the listing agent should know before the visit.",
    )
    duration_minutes: int = Field(
        default=30,
        ge=MIN_DURATION,
        le=MAX_DURATION,
        description=(
            f"How long the buyer wants the viewing to run, between "
            f"{MIN_DURATION} and {MAX_DURATION} minutes."
        ),
    )
    mode: str = Field(
        default="in_person",
        description='How the viewing should happen: "in_person" or "virtual".',
    )

    @field_validator("customer_phone")
    @classmethod
    def _phone_is_dialable(cls, value: str) -> str:
        stripped = (value or "").strip()
        digits = "".join(ch for ch in stripped if ch.isdigit())
        # A phone the owner cannot call is a broken request; catch it here
        # rather than letting it fail as invalid_contact at the command.
        if len(digits) < 7:
            raise ValueError(
                "That doesn't look like a phone number I can pass on. Ask for "
                "a full number with country or leading zero."
            )
        if len(digits) > 15:
            raise ValueError("That phone number is too long to be valid.")
        return stripped

    @field_validator("mode")
    @classmethod
    def _known_mode(cls, value: str) -> str:
        known = {"in_person", "virtual"}
        if value.strip().lower() not in known:
            raise ValueError(
                'mode must be "in_person" or "virtual" — the buyer is either '
                "coming to the property or meeting remotely."
            )
        return value.strip().lower()

    @model_validator(mode="after")
    def _named_or_named_gap(self) -> "ScheduleViewingArgs":
        # A request with no name is still submittable (the owner contacts the
        # number), but a name that is only whitespace is worse than absent.
        if self.customer_name is not None:
            self.customer_name = self.customer_name.strip() or None
        if self.notes is not None:
            self.notes = self.notes.strip() or None
        return self


@base.as_tool
async def schedule_viewing(args: ScheduleViewingArgs) -> dict:
    """Submit a viewing request and give the buyer the reference code.

    Confirms the listing is PUBLISHED first, then builds the slot as an
    explicit instant in Africa/Nairobi (a naive time would silently take the
    server's timezone, and the command rejects slots at or before now). Returns
    the reference and the real request state.
    """
    listing_id = args.property_id.strip()

    try:
        viewing_date = normalize_when(args.day, args.hour)
    except ValueError as exc:
        return base.error(str(exc))

    try:
        card = await fetch_published_listing(listing_id)
    except Exception as exc:  # noqa: BLE001 - relay, never kill the turn
        return base.error(f"The listing could not be looked up: {exc}")

    if card is None:
        return base.error(
            f"No published listing with id {listing_id}. A viewing can only be "
            "requested on a published listing."
        )

    # The stated purpose the command requires. The buyer's own note is quoted
    # when there is one, so the owner sees what was asked rather than a
    # generic line.
    stated_purpose = (
        f"{args.notes[:300]}"
        if args.notes
        else f"Viewing request for {card.title}."
    )

    try:
        record = await request_viewing(
            listing_id=listing_id,
            contact_channel="phone",
            contact_value=args.customer_phone,
            display_name=args.customer_name,
            stated_purpose=stated_purpose,
            viewing_date=viewing_date,
            mode=args.mode,
            duration_minutes=args.duration_minutes,
            notes=args.notes,
        )
    except DomainCommandError as exc:
        # The command refused with a reason_code; relay it rather than
        # pretending the request went through.
        return base.error(
            f"The viewing request could not be submitted: {exc.message}",
            reason_code=exc.reason_code,
        )
    except Exception as exc:  # noqa: BLE001 - relay, never kill the turn
        return base.error(f"The viewing request could not be submitted: {exc}")

    if record.already_submitted:
        return base.error(
            "That viewing request was already submitted — the reference below "
            "is the original one. Check with the sales team before asking again.",
            reference=record.reference,
            request_state=record.state,
        )

    return base.guarded(
        {
            "submitted": True,
            "reference": record.reference,
            "request_state": record.state,
            "listing_id": record.listing_id,
            "listing_title": card.title,
            "requested_date": record.requested_date,
            "duration_minutes": record.duration_minutes,
            "mode": record.mode,
            "status_note": (
                "Submitted, not yet confirmed. Every request starts in "
                "pending_contact_verification: the sales team verifies the "
                "contact details, then contacts the buyer to confirm the "
                "time. Simon cannot look the request up again afterwards."
            ),
            "contact": base.HANDOFF_POINTER,
        },
        where="schedule_viewing",
    )
