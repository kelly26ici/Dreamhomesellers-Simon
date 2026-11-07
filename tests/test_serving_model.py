"""ServingModel must record on the path the agent actually uses: async.

The first version passed four tests and shipped empty, because those tests
drove the sync wrapper while Simon streams through astream_events, which is
async. These tests drive awrap_model_call, the path production takes.
"""
from __future__ import annotations

import os
import warnings

warnings.filterwarnings("ignore")
for _n in ("NVIDIA_API_KEY", "POOLSIDE_API_KEY", "GROQ_API_KEY"):
    os.environ.setdefault(_n, "test-key-unused")

import pytest
from langchain.agents.middleware.types import ModelRequest

from simon.agent.middleware import build_middleware
from simon.agent.middleware import serving_model as sm


class _Model:
    model = "nvidia/nemotron-3.5-lightning-30b-a3b"


def _middleware():
    stack = build_middleware()
    return next(
        m for m in stack if type(m).__name__ == "serving_model_middleware"
    )


@pytest.mark.asyncio
async def test_records_on_the_async_path() -> None:
    """The regression test. Sync-only wrappers pass the old suite and fail here."""
    mw = _middleware()
    request = ModelRequest(
        model=_Model(), messages=[], tools=[], state={}, runtime=None
    )

    async def handler(_request):
        return {"result": "ok"}

    sm.reset_serving_model()
    await mw.awrap_model_call(request, handler)

    assert sm.serving_model_id() == "nvidia/nemotron-3.5-lightning-30b-a3b"


@pytest.mark.asyncio
async def test_async_hook_exists_at_all() -> None:
    """Guard: a sync-only decorator has no async implementation at all."""
    mw = _middleware()
    request = ModelRequest(
        model=_Model(), messages=[], tools=[], state={}, runtime=None
    )

    async def handler(_request):
        return {"result": "ok"}

    sm.reset_serving_model()
    try:
        await mw.awrap_model_call(request, handler)
    except NotImplementedError as exc:  # pragma: no cover - the guard itself
        pytest.fail(f"async path missing: {exc}")


def test_model_id_falls_back_across_provider_shapes() -> None:
    """nvidia exposes .model; openai-compatible clients expose .model_name."""
    assert sm.model_id(_Model()) == "nvidia/nemotron-3.5-lightning-30b-a3b"

    class _OnlyModelName:
        model_name = "laguna-s-2.1"

    class _Neither:
        pass

    assert sm.model_id(_OnlyModelName()) == "laguna-s-2.1"
    assert sm.model_id(_Neither()) == "_Neither"


def test_health_exposes_serving_model() -> None:
    """The route must report it; a field nothing reads is decoration."""
    from pathlib import Path

    main = Path(
        r"C:\Users\rexk6\Coding\learning\Dreamhomesellers-Simon"
        r"\src\simon\server\main.py"
    ).read_text(encoding="utf-8")
    assert '"serving_model": serving_model_id()' in main
