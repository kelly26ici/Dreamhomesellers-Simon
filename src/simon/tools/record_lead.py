"""Record a buyer's enquiry as a concierge handoff.

Routed through the ``create_handoff`` domain command — the only client write
path. There is no ``property_inquiries`` table in the marketplace schema, and a
direct anonymous INSERT on ``handoffs`` is refused by RLS anyway.

Shape change worth being explicit about: the legacy lead record had
``customer_phone``, ``customer_email`` and ``agent_id`` columns. None of them
exist here. Email is a contact CHANNEL, not a field, and the owner side is
resolved server-side from the listing — so neither is collected by this tool.
``handoff_kind`` maps 1:1 from the legacy ``inquiry_type`` vocabulary
("general", "viewing", "listing"), which is the same vocabulary the frontend
handoff contract names as the verified legacy set.
"""

from __future__ import annotations

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

from simon.catalog import (
    DomainCommandError,
    fetch_published_listing,
    record_handoff,
)
from simon.tools import _base as base

# The verified legacy inquiry_type vocabulary, 1:1 with handoff_kind.
HANDOFF_KINDS = ("general", "viewing", "listing")

DEFAULT_MESSAGE = "Buyer asked the concierge for help with a property enquiry."


class RecordLeadArgs(BaseModel):
    """An enquiry to pass to the sales team, and how to reach the buyer.

    The phone number is the contact channel the sales team replies on. It is
    never echoed back into tool output.
    """

    model_config = ConfigDict(extra="forbid")

    customer_phone: str = Field(
        description=(
            "The buyer's phone number, in any readable format. Used as the "
            "contact channel for the follow-up."
        ),
    )
    inquiry_type: str = Field(
        default="general",
        description=(
            'Kind of enquiry: "general", "viewing", or "listing" — the same '
            "vocabulary the marketplace handoff contract defines."
        ),
    )
    customer_name: str | None = Field(
        default=None, description="The buyer's name if known."
    )
    property_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("property_id", "listing_id"),
        description=(
            "The listing_id the enquiry is about, if any. Leave unset for a "
            "general concierge enquiry."
        ),
    )
    message: str | None = Field(
        default=None,
        description="What the buyer asked for, in their words where possible.",
    )

    @field_validator("inquiry_type")
    @classmethod
    def _known_kind(cls, value: str) -> str:
        kind = value.strip().lower()
        if kind not in HANDOFF_KINDS:
            raise ValueError(
                f"inquiry_type must be one of {', '.join(HANDOFF_KINDS)}; got "
                f"{value!r}. Ask which sort of enquiry the buyer means."
            )
        return kind

    @model_validator(mode="after")
    def _trimmed(self) -> RecordLeadArgs:
        self.customer_phone = self.customer_phone.strip()
        if self.customer_name is not None:
            self.customer_name = self.customer_name.strip() or None
        if self.message is not None:
            self.message = self.message.strip() or None
        if self.property_id is not None:
            self.property_id = self.property_id.strip() or None
        # An email address pasted into the message field would be relayed to the
        # owner side; the buyer's phone is the channel we promised. Strip it
        # rather than copying a second channel into a free-text message.
        return self


@base.as_tool
async def record_lead(args: RecordLeadArgs) -> dict:
    """Save a customer inquiry so the sales team can follow up.

    Verifies the listing is PUBLISHED when one is named (the command rejects an
    unpublished listing with ``listing_not_available``), then records the
    handoff and returns its reference and real state.
    """
    listing_id = args.property_id
    if listing_id:
        try:
            card = await fetch_published_listing(listing_id)
        except Exception as exc:  # noqa: BLE001 - relay, never kill the turn
            return base.error(f"The listing could not be looked up: {exc}")
        if card is None:
            return base.error(
                f"No published listing with id {listing_id}. An enquiry can "
                "only be recorded against a published listing."
            )

    try:
        lead = await record_handoff(
            contact_channel="phone",
            contact_value=args.customer_phone,
            display_name=args.customer_name,
            handoff_kind=args.inquiry_type,
            listing_id=listing_id,
            message=args.message or DEFAULT_MESSAGE,
        )
    except DomainCommandError as exc:
        return base.error(
            f"The enquiry could not be recorded: {exc.message}",
            reason_code=exc.reason_code,
        )
    except Exception as exc:  # noqa: BLE001 - relay, never kill the turn
        return base.error(f"The enquiry could not be recorded: {exc}")

    if lead.already_recorded:
        return base.error(
            "That enquiry was already recorded — the reference below is the "
            "original one, so the sales team has it and will follow up.",
            reference=lead.reference,
            handoff_state=lead.state,
        )

    return base.guarded(
        {
            "recorded": True,
            "reference": lead.reference,
            "handoff_kind": lead.handoff_kind,
            "handoff_state": lead.state,
            "listing_id": lead.listing_id,
            "status_note": (
                "Recorded, not yet handled: the enquiry starts in "
                "pending_contact_verification and the sales team verifies the "
                "contact details before following up. Simon cannot look it up "
                "again afterwards."
            ),
            "contact": base.HANDOFF_POINTER,
        },
        where="record_lead",
    )
