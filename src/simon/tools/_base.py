"""Shared machinery for Simon's tools — one place, not one copy per tool.

Everything here exists because it used to be duplicated (or missing) in eight
tool bodies:

* result caps, so a chat turn cannot be flooded by a broad filter;
* contact redaction, which is a hard rule — a tool must NEVER emit an owner's
  or a buyer's email address or phone number;
* structured errors, returned as ``{"error": ...}`` dicts rather than raised,
  so a failed tool becomes something the model can act on instead of a dead
  turn;
* the LangChain ``@tool`` adapter that takes a Pydantic args model.

The redaction rule is enforced structurally wherever it can be. ``records.py``
carries no contact field at all, because an anon caller cannot reach an owner's
contact through the marketplace schema: there is no ``agents`` table, and
``listings.organization_id`` stays restricted so the org display block is not
joinable either. What is left to guard is the contact a BUYER hands us for a
viewing request or an enquiry — in-bound only, and never echoed back. So the
guard below is a shape test plus a key blocklist, and it runs on every payload
a tool returns.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable
from typing import Any, get_type_hints

from langchain.tools import tool
from pydantic import BaseModel, ValidationError

# Result caps. Browse and compare stay deliberately small: a chat turn that
# renders twenty properties is unusable, and the buyer can always ask for more.
BROWSE_CAP = 6
COMPARE_CAP = 4
SEMANTIC_CAP = 6

# A buyer must be pointed at the listing page to reach the agent. This is the
# ONLY thing a concierge may say about contact — never a raw email or phone.
CONTACT_POINTER = (
    "Reach the listing agent through the enquiry form on the listing page."
)

# Lead routing: a concierge hands the enquiry to the sales team; it does not
# hand out direct contact details.
HANDOFF_POINTER = (
    "Record an enquiry and the sales team will follow up through the contact "
    "details you gave us."
)

# Tool argument names whose VALUES must never be echoed into tool output. They
# are contact channels for the domain commands — in-bound only.
_CONTACT_KEYS = frozenset(
    {
        "phone",
        "phone_number",
        "customer_phone",
        "contact_phone",
        "contact_value",
        "email",
        "email_address",
        "customer_email",
        "contact_email",
        "agent_phone",
        "agent_email",
        "owner_phone",
        "owner_email",
    }
)

_EMAIL_SHAPE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
# A canonical UUID (listing ids, revision ids, media ids). Every id in Simon's
# output is one of these, and a UUID's hyphen-separated digit groups otherwise
# read as a phone-shaped run — so identify them and skip them.
_UUID_SHAPE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
# A dialling string: an optional +country code, then digits possibly separated
# by the usual separators (space, dot, dash, parens). Judged by counting DIGITS
# rather than characters, so "0722 123 456" is caught the same way as
# "0722123456".
_DIALSTRING = re.compile(
    r"(?<![\d.])(?:\+\d{1,3}[\s.-]?)?\d[\d\s().-]{5,}\d(?![\d.])"
)
# Money. Deliberately NARROW: a currency token, OR a decimal point, OR a
# comma-grouped integer. Grouping alone is included because a comma is not a
# phone separator — but a SPACE-grouped run ("0722 123 456") is exactly what a
# phone number looks like, so space grouping is NOT treated as money.
_MONEY = re.compile(
    r"(?:KES|KSh|ksh|Ksh)\s*[\d.,\s]+"          # "KES 12,500,000"
    r"|\d[.,]\d{2}(?!\d)"                        # "11500000.00", "250.50"
    r"|\d{1,3}(?:,\d{3})+(?!\d)"                 # "12,500,000"
)
_YEAR = re.compile(r"^(?:19|20)\d{2}$")

# The minimum number of digits for a run to be treated as a phone number. Card
# fields legitimately carry 1-4 digit figures (bedrooms, floors, position,
# page), and no Kenyan phone number has fewer than 9 — so this keeps ordinary
# payloads clean without weakening the guard.
_MIN_PHONE_DIGITS = 9


def capped(value: int, ceiling: int, *, floor: int = 1) -> int:
    """Clamp a caller-supplied count into the safe range."""
    return max(floor, min(value, ceiling))


def resolve_id(data: Any, *names: str) -> str:
    """Read the first present key among several accepted spellings.

    ``search_properties`` returns cards keyed ``listing_id``, so the model
    naturally echoes ``listing_id`` into the tools that take one listing. Those
    tools historically named the field ``property_id``, which made the
    documented value an unexpected extra under ``extra="forbid"`` — the model
    supplied exactly what it was told and was refused. Accepting both spellings
    here keeps one coherent vocabulary across the whole tool surface.
    """
    if isinstance(data, dict):
        for name in names:
            value = data.get(name)
            if value:
                return str(value)
    return getattr(data, names[0], "") or ""


def error(message: str, **extra: Any) -> dict[str, Any]:
    """A structured tool error.

    Returned, never raised: the model reads it as tool output and can rephrase
    or ask a follow-up, which is strictly better than the turn dying with an
    exception.
    """
    payload: dict[str, Any] = {"error": message}
    payload.update(extra)
    return payload


def _is_dialstring(value: str) -> bool:
    """True when the candidate is a phone number rather than money or a count.

    Discriminating is the whole difficulty here. Prices in KES
    ("KES 12,500,000", "11500000.00"), areas ("250 sqm") and years ("2019") all
    contain digit runs to a shape test, and flagging any of them would make
    every ordinary listing payload an error — which is exactly how a guard gets
    switched off by whoever is reading the error.

    Judged three ways: currency token / decimal point / 3-digit grouping means
    money; 4-digit 19xx-20xx means a year; and too FEW digits means a count,
    since card fields legitimately carry 1-4 digit figures (bedrooms, floors,
    position, page numbers). What survives is a digit-heavy run that can only be
    a dialling string.
    """
    text = value.strip()
    digits = sum(ch.isdigit() for ch in text)
    return (
        digits >= _MIN_PHONE_DIGITS
        and not _YEAR.match(text)
        and not _MONEY.search(text)
    )


def assert_no_contact(payload: Any, *, where: str) -> Any:
    """Reject a payload that carries a contact value (defence in depth).

    Walks dicts and lists. A key named for a contact, or a string shaped like
    an email address or a phone number, raises — so a bug in one tool cannot
    silently ship a buyer's phone number to the model.
    """
    if isinstance(payload, dict):
        for key, value in payload.items():
            if str(key).lower() in _CONTACT_KEYS:
                raise ValueError(
                    f"{where}: tool output must not carry a contact field "
                    f"'{key}'. Point at the enquiry form instead."
                )
            assert_no_contact(value, where=where)
    elif isinstance(payload, (list, tuple)):
        for item in payload:
            assert_no_contact(item, where=where)
    elif isinstance(payload, str):
        if _EMAIL_SHAPE.search(payload):
            raise ValueError(f"{where}: tool output must not contain an email.")
        # Blank out the ids first: a UUID's digit groups are not a phone number,
        # and treating them as one would make every listing payload an error.
        text = _UUID_SHAPE.sub(" ", payload)
        for hit in _DIALSTRING.finditer(text):
            if _is_dialstring(hit.group()):
                raise ValueError(
                    f"{where}: tool output must not contain a phone number."
                )
    return payload


def guarded(result: Any, *, where: str) -> Any:
    """Final gate a tool runs its payload through before returning it."""
    return assert_no_contact(result, where=where)


def _validation_details(exc: ValidationError) -> list[dict[str, str]]:
    """Flatten a Pydantic error into readable (field, problem) pairs.

    A ``model_validator`` raises with an empty ``loc``, so the field column falls
    back to the artifact's title rather than reporting "" — the model needs to
    know WHICH argument to change, and "input" told it nothing.

    Deliberately field names and messages only — never the rejected values,
    which are model-generated and could carry a buyer's personal data.
    """
    details = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", ()) if p != "__root__")
        if not loc:
            title = getattr(getattr(exc, "title", None), "__name__", None)
            loc = title or "arguments"
        details.append({
            "field": loc,
            "problem": str(err.get("msg", "is invalid")),
        })
    return details


def _validation_message(exc: ValidationError) -> str:
    """A corrective sentence, not a stack trace: the model reads this and retries."""
    fields = ", ".join(d["field"] for d in _validation_details(exc)) or "input"
    return (
        f"Those arguments were rejected: {fields}. Fix them to the documented "
        "types and ranges and try again."
    )


def as_tool(handler: Callable) -> Any:
    """Turn an ``async def handler(args: SomeModel) -> dict`` into a tool.

    The handler takes ONE argument: its already-validated Pydantic model. This
    adapter is what keeps the per-tool files thin — the model carries the
    validation (types, Field descriptions, cross-field rules), so the function
    body starts from a valid object instead of re-checking its inputs.

    LangChain binds keyword arguments straight onto the coroutine, so a
    single-model parameter cannot be filled by the framework: the wrapper
    accepts ``**fields``, validates them into the model, and delegates.

    The model is read through ``get_type_hints`` rather than raw
    ``__annotations__``: every tool module carries ``from __future__ import
    annotations``, so the raw annotation for the single parameter is the STRING
    'SomeArgs' and would never resolve to the class.
    """
    module = inspect.getmodule(handler)
    try:
        hints = get_type_hints(handler, globalns=vars(module) if module else None)
    except NameError as exc:
        raise TypeError(
            f"{handler.__name__}: its 'args' annotation could not be resolved "
            f"({exc}). Name the args model imported in the same module."
        ) from exc

    schema = hints.get("args")
    if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
        raise TypeError(
            f"{handler.__name__}: an args model annotated as 'args' is required"
        )

    async def invoke(**fields: Any) -> dict:
        # Validation must land HERE, not inside the handler: the model emits bad
        # input far more often than the handler fails, and a raised
        # ValidationError would kill the turn instead of letting the model read
        # what was wrong and correct itself.
        try:
            parsed = schema.model_validate(fields)
        except ValidationError as exc:
            return error(
                _validation_message(exc),
                invalid_arguments=_validation_details(exc),
            )
        return await handler(parsed)

    invoke.__name__ = handler.__name__
    invoke.__qualname__ = handler.__qualname__
    invoke.__doc__ = handler.__doc__
    invoke.__annotations__ = {"return": dict, "args": schema}
    return tool(invoke, args_schema=schema)
