import httpx
from qdrant_client import AsyncQdrantClient

from simon.catalog import fetch_cards
from simon.pool import get_vector_client
from simon.records import PropertyCard
from simon.settings import get_pool_settings

EMBED_URL = (
    "https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/"
    "@cf/qwen/qwen3-embedding-0.6b"
)


async def embed_query(text: str) -> list[float]:
    mine = get_pool_settings()
    url = EMBED_URL.format(account=mine.cloudflare_account_id)
    async with httpx.AsyncClient(timeout=30) as http:
        response = await http.post(
            url,
            headers={
                "Authorization": (
                    f"Bearer {mine.cloudflare_api_token.get_secret_value()}"
                ),
                "Content-Type": "application/json",
            },
            json={"text": [text]},
        )
        response.raise_for_status()
        return response.json()["result"]["data"][0]


async def nearest(
    text: str, *, limit: int = 6, store: AsyncQdrantClient | None = None
) -> list[tuple[PropertyCard, float]]:
    vector = await embed_query(text)
    client = store or get_vector_client()
    hits = await client.query_points(
        collection_name="properties",
        query=vector,
        limit=limit,
        with_payload=False,
    )
    ids = [h.id for h in hits.points]
    cards = await fetch_cards([str(i) for i in ids])
    by_id = {card.id: card for card in cards}
    matches = []
    for hit in hits.points:
        card = by_id.get(str(hit.id))
        if card:
            matches.append((card, hit.score))
    return matches
