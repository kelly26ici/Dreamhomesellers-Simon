"""ServingModel must record, and /api/health must show it.

Exists because it shipped empty. The middleware read `.model` off the
response, ModelResponse has no such field, so it silently recorded nothing --
and because no test exercised the hook, 46 green tests said nothing about it.
"""
from __future__ import annotations

from simon.agent.middleware import serving_model as sm


class _Model:
    model = "nvidia/nemotron-3.5-lightning-30b-a3b"


class _Request:
    model = _Model()


def test_records_the_model_that_served() -> None:
    """Drives the middleware the way the agent does, not the raw function.

    `@wrap_model_call` returns an AgentMiddleware object, so calling the
    undecorated function would test a thing that never runs in production.
    The middleware exposes `awrap_model_call` / `wrap_model_call` on itself.
    """
    sm.reset_serving_model()

    def handler(_request):
        return {"result": "ok"}

    # The object create_agent receives; drive it exactly as the framework does.
    sm.serving_model_middleware.wrap_model_call(_Request(), handler)

    assert sm.serving_model_id() == "nvidia/nemotron-3.5-lightning-30b-a3b", (
        "serving_model was not recorded: the hook must read ModelRequest.model, "
        "because ModelResponse has no .model field"
    )


def test_model_id_falls_back_across_provider_shapes() -> None:
    """nvidia exposes .model; openai-compatible clients expose .model_name."""
    assert sm.model_id(_Model()) == "nvidia/nemotron-3.5-lightning-30b-a3b"

    class _OnlyModelName:
        model_name = "laguna-s-2.1"

    class _Neither:
        pass

    assert sm.model_id(_OnlyModelName()) == "laguna-s-2.1"
    assert sm.model_id(_Neither()) == "_Neither", (
        "an unidentifiable model must still produce a value, never None or ''"
    )


def test_reset_clears_it() -> None:
    sm.record_serving_model("something")
    sm.reset_serving_model()
    assert sm.serving_model_id() is None


def test_health_exposes_serving_model() -> None:
    """The route must actually report it; a field nothing reads is decoration."""
    import re
    from pathlib import Path

    main = Path(
        r"C:\Users\rexk6\Coding\learning\Dreamhomesellers-Simon"
        r"\src\simon\server\main.py"
    ).read_text(encoding="utf-8")
    assert re.search(r'"serving_model":\s*serving_model_id\(\)', main), (
        "/api/health does not report serving_model_id()"
    )
