"""Fuzzy, natural-language property search — currently unavailable (D5).

Owner decision D5: this tool FAILS CLOSED with a structured error until a
published-only Qdrant projection exists. It stays registered so the concierge
can explain its own limits (and so the tool count and the agent's capability
surface are honest), but its body deliberately performs NO embedding, NO vector
query, and NO listing read.

Why the old path cannot simply be re-pointed:

* the vector collection is keyed by legacy property uuids, which do not resolve
  to ``listings.listing_id`` — so a re-fetch would return nothing, always;
* more seriously, that projection has no published-only view, so a hit could
  surface a listing that has since been withdrawn, closed, or moderated out of
  the buyer-visible surface. Returning empty is a bug; returning stale is a
  correctness and a trust failure.

The fix is a rebuildable projection keyed on listing_id that indexes only
published, buyer-visible revisions. Until that exists, the honest answer is to
say so and offer the structured filters, which is exactly what this returns.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from simon.meaning import UNAVAILABLE_MESSAGE
from simon.tools import _base as base


class SearchByMeaningArgs(BaseModel):
    """What the buyer described, in their own words.

    Accepted so the concierge can acknowledge the request shape — it is used
    only to pick which structured filters to offer as the alternative, never to
    search.
    """

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        description=(
            "The buyer's own words, e.g. 'sunny family home near good "
            "schools'."
        ),
    )
    limit: int = Field(
        default=4,
        ge=1,
        le=base.SEMANTIC_CAP,
        description="Maximum results wanted; the default cap is 4.",
    )


@base.as_tool
async def search_by_meaning(args: SearchByMeaningArgs) -> dict:
    """Find listings by what the buyer described rather than by filters.

    Currently unavailable: the vector index holds superseded ids and has no
    published-only view, so searching it could surface withdrawn properties.
    No embedding or vector query is performed. Use ``search_properties`` with
    concrete filters (price, bedrooms, city or area, property type, amenities)
    instead, and say so plainly to the buyer.
    """
    return base.error(
        UNAVAILABLE_MESSAGE,
        semantic_search="unavailable",
        reason="no published-only projection keyed on listing_id",
        alternative="search_properties",
        requested=args.query.strip(),
    )
