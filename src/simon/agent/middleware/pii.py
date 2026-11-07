"""PII redaction, outbound only.

``apply_to_input=False`` is load-bearing, not an oversight. A buyer scheduling
a viewing MUST be able to give a phone number: the tool takes it as an
argument and the sales team needs it. Redacting or blocking input would break
the core feature to satisfy a scanner.

What this protects instead: Simon never echoes an email address, a card
number, or an IP back to whoever is typing. ``assert_no_contact`` in
``tools/_base.py`` stays the tool-layer guard, because it is the one that can
tell OWNER contact data -- which must never reach a buyer -- from a buyer's
own number, which must. A generic detector cannot make that distinction.

Known limitation, stated rather than hidden: phone numbers are not in
LangChain's built-in detector set, and adding a custom one here would redact
the buyer's own number out of their confirmation. That is a product decision,
so it is not silently made here.
"""

from __future__ import annotations

from langchain.agents.middleware import PIIMiddleware

# (pii_type, strategy). One entry per thing Simon must never repeat outward.
TYPES: tuple[tuple[str, str], ...] = (
    ("email", "redact"),
    ("credit_card", "redact"),
    ("ip", "redact"),
)


def build() -> list[PIIMiddleware]:
    """One instance per type. Outbound only, in every case."""
    return [
        PIIMiddleware(
            pii_type,
            strategy=strategy,
            apply_to_input=False,
            apply_to_output=True,
        )
        for pii_type, strategy in TYPES
    ]
