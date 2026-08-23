"""
Tests for per-model routing health indicators (⚠️ in the models table):
router records last error per (backend, model), clears on success,
broadcasts model.error / model.ok over the realtime hub, and /api/models
exposes the registry.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.client import client
from app.providers.antigravity import AntigravityAdapter
from app.providers.gemini_api import GeminiApiAdapter
from app.providers.gemini_web import GeminiWebAdapter
from app.providers.router import MultiBackendRouter
from main import app


@pytest.fixture(autouse=True)
def isolated_model_errors():
    """Snapshot & restore the global router's error registry around each test."""
    orig = dict(client._model_errors)
    yield
    client._model_errors = orig


@pytest.fixture(autouse=True)
def stub_lifespan_network(monkeypatch):
    """Keep lifespan startup offline while using TestClient."""
    monkeypatch.setattr(
        client,
        "fetch_available_models",
        AsyncMock(return_value={"models": {}}),
    )
    monkeypatch.setattr(client, "load_code_assist", AsyncMock())


def _make_router(fail_first_backend: bool):
    """
    Build a router whose 'gemini_api' backend either fails instantly on
    generate_content (fail_first_backend=True) or always succeeds.
    Antigravity always succeeds as the fallback.
    """

    def ok_stream_factory(*args, **kwargs):
        async def gen():
            yield {
                "candidates": [],
                "usageMetadata": {"totalTokenCount": 7},
            }

        return gen()

    def _wire_common(mock: MagicMock, enabled: bool) -> None:
        mock.enabled = enabled
        mock.is_configured = MagicMock(return_value=True)
        mock.is_available = MagicMock(return_value=True)

    api = MagicMock(spec=GeminiApiAdapter)
    api.name = "gemini_api"
    _wire_common(api, True)
    # Make the router's supports_model() probe treat this backend as capable
    api._cached_models = {"models": {"gemini-3.7-flash-medium": {}}}
    if fail_first_backend:
        # Fail exactly once, then behave normally (so tests can assert that a
        # subsequent success clears the recorded error).
        api.generate_content = AsyncMock(
            side_effect=[
                ValueError("Gemini API Error (503): UNAVAILABLE"),
                {"usageMetadata": {"totalTokenCount": 5}},
            ]
        )

        stream_state = {"failed": False}

        def flaky_stream_factory(*args, **kwargs):
            if not stream_state["failed"]:
                stream_state["failed"] = True
                raise ValueError("Gemini API Error (503): UNAVAILABLE")
            return ok_stream_factory(*args, **kwargs)

        api.stream_generate_content = flaky_stream_factory
        api.embed_contents = AsyncMock(side_effect=ValueError("embed boom"))
    else:
        api.generate_content = AsyncMock(
            return_value={"usageMetadata": {"totalTokenCount": 5}}
        )
        api.stream_generate_content = ok_stream_factory
        api.embed_contents = AsyncMock(return_value={})

    agy = MagicMock(spec=AntigravityAdapter)
    agy.name = "antigravity"
    _wire_common(agy, True)
    agy.generate_content = AsyncMock(
        return_value={"usageMetadata": {"totalTokenCount": 9}}
    )
    agy.stream_generate_content = ok_stream_factory
    agy.embed_contents = AsyncMock(return_value={})

    web = MagicMock(spec=GeminiWebAdapter)
    web.name = "gemini_web"
    _wire_common(web, False)

    return MultiBackendRouter(antigravity=agy, gemini_api=api, gemini_web=web)


@pytest.mark.asyncio
async def test_generate_failure_records_error_and_success_clears_it():
    router = _make_router(fail_first_backend=True)

    # gemini_api fails -> fallback succeeds -> request OK but error recorded
    res = await router.generate_content(
        model="gemini-3.7-flash-medium",
        contents=[{"parts": [{"text": "hi"}]}],
    )
    assert res["usageMetadata"]["totalTokenCount"] == 9

    errors = router.get_model_errors()
    assert "gemini_api" in errors
    entry = errors["gemini_api"]["gemini-3.7-flash-medium"]
    assert "503" in entry["error"]
    assert entry["op"] == "generate_content"
    assert entry["at"]
    assert "antigravity" not in errors

    # Next attempt through gemini_api succeeds -> indicator must clear
    await router.generate_content(
        model="gemini-3.7-flash-medium",
        contents=[{"parts": [{"text": "hi"}]}],
    )
    assert router.get_model_errors() == {}


@pytest.mark.asyncio
async def test_stream_connect_failure_records_error():
    router = _make_router(fail_first_backend=True)

    chunks = []
    async for chunk in router.stream_generate_content(
        model="gemini-3.7-flash-medium",
        contents=[{"parts": [{"text": "hi"}]}],
    ):
        chunks.append(chunk)
    assert chunks, "fallback should still serve the stream"

    entry = router.get_model_errors()["gemini_api"]["gemini-3.7-flash-medium"]
    assert "503" in entry["error"]
    assert entry["op"] == "stream_generate_content"


@pytest.mark.asyncio
async def test_embed_failure_records_error():
    router = _make_router(fail_first_backend=True)
    await router.embed_contents(model="text-embedding-004", texts=["hello world"])
    entry = router.get_model_errors()["gemini_api"]["text-embedding-004"]
    assert "embed boom" in entry["error"]


def test_record_and_get_model_errors_normalizes_keys():
    router = _make_router(fail_first_backend=False)
    router.record_model_error("gemini_api", "Models/Gemini-2.5-Pro ", "boom")
    stored = router.get_model_errors()
    assert "gemini-2.5-pro" in stored["gemini_api"]

    router.clear_model_error("gemini_api", "models/gemini-2.5-pro")
    assert router.get_model_errors() == {}

    # Clearing an unknown entry must be a harmless no-op
    router.clear_model_error("does_not_exist", "nope")
    assert router.get_model_errors() == {}


@pytest.mark.asyncio
async def test_model_error_and_ok_events_pushed_over_websocket():
    history_snapshot = dict(client._model_errors)
    try:
        with TestClient(app) as tc, tc.websocket_connect("/ws") as ws:
            assert ws.receive_json()["type"] == "hello"

            client.record_model_error(
                "gemini_api",
                "gemini-3.7-flash-medium",
                "Gemini API Error (503): high demand",
                op="stream_generate_content",
            )
            msg = ws.receive_json()
            while msg["type"] not in ("model.error",):
                msg = ws.receive_json()
            assert msg["type"] == "model.error"
            assert msg["payload"]["backend"] == "gemini_api"
            assert msg["payload"]["model"] == "gemini-3.7-flash-medium"
            assert "high demand" in msg["payload"]["error"]

            client.clear_model_error("gemini_api", "gemini-3.7-flash-medium")
            ok_msg = ws.receive_json()
            while ok_msg["type"] not in ("model.ok",):
                ok_msg = ws.receive_json()
            assert ok_msg["type"] == "model.ok"
            assert ok_msg["payload"]["model"] == "gemini-3.7-flash-medium"
    finally:
        client._model_errors = history_snapshot


@pytest.mark.asyncio
async def test_api_models_exposes_model_errors_registry():
    with (
        patch.object(
            client.antigravity,
            "fetch_available_models",
            new_callable=AsyncMock,
            return_value={"models": {}},
        ),
        patch.object(
            client.gemini_api,
            "fetch_available_models",
            new_callable=AsyncMock,
            return_value={"models": {}},
        ),
        patch.object(
            client.gemini_web,
            "fetch_available_models",
            new_callable=AsyncMock,
            return_value={"models": {}},
        ),
        TestClient(app) as tc,
    ):
        resp = tc.get("/api/models")
        assert resp.status_code == 200
        data = resp.json()

    assert isinstance(data.get("model_errors"), dict)
