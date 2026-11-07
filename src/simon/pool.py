from functools import lru_cache

from qdrant_client import AsyncQdrantClient
from supabase import AsyncClient, acreate_client

from simon.settings import get_settings


async def open_pool() -> AsyncClient:
    mine = get_settings()
    return await acreate_client(
        mine.supabase_url, mine.supabase_key.get_secret_value()
    )


@lru_cache(maxsize=1)
def _pool_holder() -> dict:
    return {"client": None}


async def get_pool() -> AsyncClient:
    holder = _pool_holder()
    if holder["client"] is None:
        holder["client"] = await open_pool()
    return holder["client"]


@lru_cache(maxsize=1)
def get_vector_client() -> AsyncQdrantClient:
    mine = get_settings()
    return AsyncQdrantClient(
        url=mine.qdrant_url, api_key=mine.qdrant_api_key.get_secret_value()
    )
