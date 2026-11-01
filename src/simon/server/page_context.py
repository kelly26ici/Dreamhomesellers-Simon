"""Typed page context and its prompt digest.

This lives outside ``main.py`` because both transports consume it: the legacy
``POST /api/chat`` route and the Agent Streaming Protocol's ``run.start``
metadata. ``main.py`` imports the protocol router, so the protocol cannot
import these names from ``main`` without a cycle. One module, one definition,
so the streamed and non-streamed paths cannot drift apart.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SearchFilters(BaseModel):
    """Mirror of the frontend SearchFilters contract (contracts.md §10)."""

    model_config = {"extra": "ignore"}

    listing_kind: str | None = None
    property_type: str | None = None
    min_price: str | None = None
    max_price: str | None = None
    price_currency: str | None = None
    min_bedrooms: int | None = None
    min_bathrooms: int | None = None
    city: str | None = None
    area: str | None = None
    amenities: list[str] | None = None
    sort: str | None = None
    cursor: str | None = None


class ApproximateLocation(BaseModel):
    model_config = {"extra": "ignore"}

    latitude: float
    longitude: float


class PageContext(BaseModel):
    """Typed transient intent (untrusted). Consumed as context only — the
    agent's tools revalidate every fact against Postgres/Qdrant. Never
    persisted as authority."""

    model_config = {"extra": "ignore"}

    contract_version: str | None = None
    sequence: int | None = None
    route: str | None = None
    view_mode: Literal[
        "search", "property_detail", "compare", "curated", "saved",
        "workspace", "other",
    ] | None = None
    active_listing_id: str | None = None
    active_revision_id: str | None = None
    result_position: int | None = None
    result_count: int | None = None
    visible_listing_ids: list[str] = Field(default_factory=list, max_length=64)
    search_filters: SearchFilters | None = None
    compare_selection: list[str] = Field(default_factory=list, max_length=4)
    approximate_location: ApproximateLocation | None = None


def context_digest(ctx: PageContext | None) -> str:
    """Render page context into a compact preamble so the agent knows where
    the visitor is and what 'this' refers to (simon-boundary.md: context
    declares what the visitor means; the service retrieves what is true)."""
    if ctx is None:
        return ""

    bits: list[str] = []
    if ctx.view_mode == "property_detail" and ctx.active_listing_id:
        bits.append(
            f"The visitor is viewing the property detail page for listing "
            f"{ctx.active_listing_id}; 'this' most likely refers to it."
        )
    elif ctx.view_mode == "search" and ctx.visible_listing_ids:
        bits.append(
            f"The visitor is on the search grid ({len(ctx.visible_listing_ids)} "
            f"listings visible): {', '.join(ctx.visible_listing_ids[:8])}."
        )
    elif ctx.view_mode == "compare" and ctx.compare_selection:
        bits.append(
            f"The visitor is comparing listings: "
            f"{', '.join(ctx.compare_selection)}."
        )
    elif ctx.view_mode == "curated":
        bits.append("The visitor is browsing the curated 3D gallery collection.")
    elif ctx.route:
        bits.append(f"The visitor is on {ctx.route}.")

    f = ctx.search_filters
    if f is not None:
        constraints = ", ".join(
            f"{k}={v}"
            for k, v in {
                "city": f.city,
                "area": f.area,
                "min_bedrooms": f.min_bedrooms,
                "max_price": f.max_price,
                "property_type": f.property_type,
            }.items()
            if v is not None
        )
        if constraints:
            bits.append(f"Active search filters: {constraints}.")

    return ("\n\n[Visitor context] " + " ".join(bits)) if bits else ""


def page_context_from_metadata(metadata: dict | None) -> PageContext | None:
    """Read the ``pageContext`` key the Next relay puts in run.start metadata.

    Context is untrusted input: a malformed payload degrades to "no context"
    rather than failing the turn. The agent's tools still revalidate every fact.
    """
    if not isinstance(metadata, dict):
        return None
    raw = metadata.get("pageContext")
    if not isinstance(raw, dict):
        return None
    try:
        return PageContext(**raw)
    except Exception:  # noqa: BLE001 - context is optional, never fatal
        return None
