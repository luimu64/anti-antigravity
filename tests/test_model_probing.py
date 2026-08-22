import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.client import client
from app.config import DEPRECATED_MODELS
from app.keys import api_key_manager
from app.providers.gemini_api import GeminiApiAdapter
from app.providers.router import MultiBackendRouter
from main import app, lifespan


@pytest.mark.asyncio
async def test_gemini_api_adapter_dynamic_probing_and_ttl():
    """Verify GeminiApiAdapter probes upstream /models, parses capabilities, and honors TTL cache & force_refresh."""
    adapter = GeminiApiAdapter(api_key="AIzaSyTestApiKey123", model_cache_ttl=300.0)

    mock_models_response = {
        "models": [
            {
                "name": "models/gemini-2.0-flash",
                "displayName": "Gemini 2.0 Flash",
                "supportedGenerationMethods": ["generateContent"],
                "inputTokenLimit": 1048576,
            },
            {
                "name": "models/gemini-2.0-flash-thinking-exp-01-21",
                "displayName": "Gemini 2.0 Flash Thinking",
                "supportedGenerationMethods": ["generateContent"],
                "inputTokenLimit": 1048576,
            },
            {
                "name": "models/text-embedding-004",
                "displayName": "Text Embedding 004",
                "supportedGenerationMethods": ["embedContent"],
                "inputTokenLimit": 2048,
            },
            {
                "name": "models/gemini-2.5-flash",  # Deprecated model in upstream
                "displayName": "Gemini 2.5 Flash",
                "supportedGenerationMethods": ["generateContent"],
                "inputTokenLimit": 1048576,
            },
            {
                "name": "models/unsupported-method-model",
                "displayName": "Unsupported Method Model",
                "supportedGenerationMethods": ["countTokens"],
                "inputTokenLimit": 1000,
            },
        ]
    }

    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_models_response

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.is_closed = False
    mock_client.get = AsyncMock(return_value=mock_resp)

    adapter._http_client = mock_client
    adapter.get_http_client = MagicMock(return_value=mock_client)

    # 1. Initial probe calls upstream
    res1 = await adapter.fetch_available_models()
    assert mock_client.get.call_count == 1
    models = res1["models"]
    assert "gemini-2.0-flash" in models
    assert "gemini-2.0-flash-thinking-exp-01-21" in models
    assert "text-embedding-004" in models
    assert models["text-embedding-004"]["isEmbedding"] is True
    assert "thinking" in models["gemini-2.0-flash-thinking-exp-01-21"]["capabilities"]
    # Deprecated model must be filtered out
    assert "gemini-2.5-flash" not in models

    # 2. Subsequent call within TTL serves from cache (no new HTTP request)
    res2 = await adapter.fetch_available_models()
    assert mock_client.get.call_count == 1
    assert res2 == res1

    # 3. force_refresh=True triggers upstream call
    res3 = await adapter.fetch_available_models(force_refresh=True)
    assert mock_client.get.call_count == 2
    assert res3 == res1

    # 4. TTL expiration triggers upstream call
    adapter._models_fetched_at = time.time() - 301.0
    await adapter.fetch_available_models()
    assert mock_client.get.call_count == 3


@pytest.mark.asyncio
async def test_gemini_api_adapter_upstream_error_fallback():
    """Verify upstream probing errors (429, 500, network error) fall back gracefully."""
    adapter = GeminiApiAdapter(api_key="AIzaSyTestKey")

    mock_resp_429 = MagicMock(spec=httpx.Response)
    mock_resp_429.status_code = 429
    mock_resp_429.headers = {"Retry-After": "30"}
    mock_resp_429.text = "Rate limited"

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.is_closed = False
    mock_client.get = AsyncMock(return_value=mock_resp_429)

    adapter._http_client = mock_client
    adapter.get_http_client = MagicMock(return_value=mock_client)

    res = await adapter.fetch_available_models(force_refresh=True)
    assert "models" in res
    assert "gemini-2.0-flash" in res["models"]
    assert adapter.get_cooldown_remaining() > 0


@pytest.mark.asyncio
async def test_startup_lifespan_probing():
    """Verify application lifespan triggers backend model probing on startup."""
    with patch.object(
        client, "fetch_available_models", new_callable=AsyncMock
    ) as mock_probe:
        async with lifespan(app):
            pass
        mock_probe.assert_called_once_with(force_refresh=True)


def test_router_supports_model_rejection_of_deprecated():
    """Verify router supports_model rejects deprecated models and accepts active probed models."""
    api_adapter = GeminiApiAdapter(api_key="AIzaSyKey")
    api_adapter._cached_models = {
        "models": {
            "gemini-2.0-flash": {
                "displayName": "Gemini 2.0 Flash",
                "isEmbedding": False,
            },
            "gemini-1.5-pro": {"displayName": "Gemini 1.5 Pro", "isEmbedding": False},
        }
    }

    router = MultiBackendRouter(gemini_api=api_adapter)

    # Active verified models
    assert router.supports_model(api_adapter, "gemini-2.0-flash") is True
    assert router.supports_model(api_adapter, "gemini-1.5-pro") is True
    assert (
        router.supports_model(api_adapter, "gpt-4o") is True
    )  # normalizes to gemini-2.0-flash

    # Deprecated models
    for dep in DEPRECATED_MODELS:
        assert router.supports_model(api_adapter, dep) is False


@pytest.mark.asyncio
async def test_openai_endpoints_reject_deprecated_models():
    """Verify all OpenAI completion & embedding endpoints return standard 404 for deprecated models."""
    transport = httpx.ASGITransport(app=app)
    key = api_key_manager.get_first_active_key() or "test-key"

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        # 1. /v1/chat/completions non-streaming
        resp = await ac.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": "gemini-2.5-flash",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 404
        data = resp.json()
        assert data["error"]["code"] == "model_not_found"
        assert data["error"]["type"] == "invalid_request_error"
        assert "gemini-2.5-flash" in data["error"]["message"]

        # 2. /v1/chat/completions streaming
        resp_stream = await ac.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": "gemini-2.5-flash",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        assert resp_stream.status_code == 404
        data_stream = resp_stream.json()
        assert data_stream["error"]["code"] == "model_not_found"

        # 3. /v1/completions
        resp_comp = await ac.post(
            "/v1/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": "gemini-2.5-pro", "prompt": "hi"},
        )
        assert resp_comp.status_code == 404
        data_comp = resp_comp.json()
        assert data_comp["error"]["code"] == "model_not_found"

        # 4. /v1/embeddings
        resp_emb = await ac.post(
            "/v1/embeddings",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": "gemini-2.5-flash", "input": "test"},
        )
        assert resp_emb.status_code == 404
        data_emb = resp_emb.json()
        assert data_emb["error"]["code"] == "model_not_found"

        # 5. /v1/models/{model_id}
        resp_get = await ac.get(
            "/v1/models/gemini-2.5-flash",
            headers={"Authorization": f"Bearer {key}"},
        )
        assert resp_get.status_code == 404
        data_get = resp_get.json()
        assert data_get["error"]["code"] == "model_not_found"


@pytest.mark.asyncio
async def test_models_catalog_excludes_deprecated_models():
    """Verify /v1/models catalog listing does not expose deprecated models."""
    transport = httpx.ASGITransport(app=app)
    key = api_key_manager.get_first_active_key() or "test-key"

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/v1/models", headers={"Authorization": f"Bearer {key}"})
        assert resp.status_code == 200
        data = resp.json()
        model_ids = {m["id"] for m in data["data"]}

        for dep in DEPRECATED_MODELS:
            assert dep not in model_ids
