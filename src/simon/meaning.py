"""Semantic search over the LUXONN listing inventory.

Owner decision D5: this path is INERT. ``search_by_meaning`` must fail closed
with a structured error until a published-only vector projection exists, and
this module is the enforcement point — it never reaches Qdrant, never embeds
anything, and never reads a listing.

Two independent defects make the old re-fetch unsafe, and both are structural
rather than incidental:

1. The Qdrant collection is keyed by LEGACY property uuids. Those ids do not
   resolve to ``listings.listing_id``, so a re-fetch after the marketplace
   repoint returns nothing on every query.
2. More seriously, the projection has no published-only view. The re-fetch
   carried no lifecycle filter, so a hit could surface a listing that has since
   been withdrawn, closed, or moderated out of the buyer-visible surface.

The fix is a projection keyed on ``listing_id`` that only ever indexes
published, buyer-visible revisions — a rebuildable artifact, not a source of
authority. Until that exists the honest answer to "find me a sunny family home"
is to say so and offer the structured filters instead of silently returning
nothing (or something stale).
"""

from __future__ import annotations


class SemanticSearchUnavailable(RuntimeError):
    """The buyer asked for semantic search and there is no safe way to serve it.

    Carries a user-facing reason so the tool layer can return a structured
    error the model can act on, instead of raising and killing the turn.
    """

    def __init__(self, reason: str | None = None) -> None:
        super().__init__(
            reason
            or (
                "Semantic search is unavailable: the vector projection indexes "
                "legacy ids and has no published-only view."
            )
        )
        self.reason = self.args[0]


# The one sentence the concierge should be able to say about its own limits.
UNAVAILABLE_MESSAGE = (
    "Semantic search isn't available yet on this inventory: the vector index "
    "holds superseded ids and isn't restricted to published listings, so "
    "searching it could return withdrawn properties. I can still find "
    "listings with the filters (price, bedrooms, city/area, property type, "
    "amenities) — tell me what you're looking for."
)


async def nearest(
    text: str, *, limit: int = 6, store=None
) -> list[tuple[object, float]]:
    """Refuse, always.

    Deliberately not a lazy stub that reaches the network on the first call:
    D5 requires the failure be closed at the door, so this raises before any
    embedding, any Qdrant query, and any listing read. ``text`` and ``limit``
    are accepted so the signature stays call-compatible for the transition.
    """
    raise SemanticSearchUnavailable()


def nearest_sync_info() -> dict[str, str]:
    """Diagnostics for /api/health: what the semantic path is missing."""
    return {
        "semantic_search": "fail_closed",
        "reason": "no published-only projection keyed on listing_id",
    }
