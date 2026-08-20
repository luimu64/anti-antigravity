import os
import json
import time
import pytest
import httpx
from unittest.mock import AsyncMock, patch, MagicMock

from app.providers.base import BaseAdapter, RateLimitError
from app.providers.antigravity import AntigravityAdapter
from app.providers.gemini_api import GeminiApiAdapter
from app.providers.gemini_web import GeminiWebAdapter
from app.providers.router import MultiBackendRouter
from app.client import client
from main import app

@pytest.fixture
def mock_credentials_file(tmp_path, monkeypatch):
    cred_file = tmp_path / "credentials.json"
    init_data = {
        "access_token": "ya29.test_oauth_access_token",
        "refresh_token": "1//test_oauth_refresh_token",
        "token_expiry": time.time() + 3600,
        "user_email": "tester@example.com",
        "project_id": "test-project-123",
        "tier_name": "Antigravity",
        "routing_strategy": "free_first"
    }
    with open(cred_file, "w") as f:
        json.dump(init_data, f)
    
    monkeypatch.setattr("app.providers.router.CREDENTIALS_FILE", cred_file)
    monkeypatch.setattr("app.auth.CREDENTIALS_FILE", cred_file)
    return cred_file

def test_adapter_base_and_cooldown():
    """Verify BaseAdapter cooldown mechanics and default disabled state."""
    class DummyAdapter(BaseAdapter):
        name = "dummy"
        def is_configured(self) -> bool:
            return True
        async def generate_content(self, *args, **kwargs):
            return {}
        async def stream_generate_content(self, *args, **kwargs):
            yield {}
        async def fetch_available_models(self, force_refresh=False):
            return {"models": {}}

    adapter = DummyAdapter()
    # Backends must be disabled by default
    assert adapter.enabled is False
    assert adapter.is_available() is False

    # Enable adapter
    adapter.enabled = True
    assert adapter.is_available() is True
    assert adapter.get_cooldown_remaining() == 0.0

    # Put into cooldown for 10 seconds
    adapter.set_cooldown(10.0)
    assert adapter.is_available() is False
    assert adapter.get_cooldown_remaining() > 0.0

    # Clear cooldown
    adapter.clear_cooldown()
    assert adapter.is_available() is True
    assert adapter.get_cooldown_remaining() == 0.0

def test_adapter_configuration_checks():
    """Verify configuration status and disabled-by-default state across adapters."""
    agy = AntigravityAdapter()
    g_api = GeminiApiAdapter(api_key="")
    g_web = GeminiWebAdapter(psid="", psidts="")

    assert agy.enabled is False
    assert g_api.enabled is False
    assert g_web.enabled is False

    assert g_api.is_configured() is False
    g_api.api_key = "AIzaSyTestKey123"
    assert g_api.is_configured() is True

    assert g_web.is_configured() is False
    g_web.psid = "test_psid_cookie"
    assert g_web.is_configured() is True

@pytest.mark.asyncio
async def test_free_first_routing_and_429_fallback():
    """Verify free_first routing strategy: Gemini Web -> Gemini API -> Antigravity on 429."""
    mock_agy = MagicMock(spec=AntigravityAdapter)
    mock_agy.name = "antigravity"
    mock_agy.enabled = True
    mock_agy.is_configured.return_value = True
    mock_agy.is_available.return_value = True
    mock_agy.cooldown_until = 0.0
    mock_agy.get_cooldown_remaining.return_value = 0.0
    mock_agy.generate_content = AsyncMock(return_value={"text": "Antigravity response", "candidates": [{"content": {"parts": [{"text": "Antigravity response"}]}}]})

    mock_api = MagicMock(spec=GeminiApiAdapter)
    mock_api.name = "gemini_api"
    mock_api.enabled = True
    mock_api.is_configured.return_value = True
    mock_api.is_available.return_value = True
    mock_api.cooldown_until = 0.0
    mock_api.get_cooldown_remaining.return_value = 0.0
    mock_api.generate_content = AsyncMock(side_effect=RateLimitError("API quota exhausted (429)"))

    mock_web = MagicMock(spec=GeminiWebAdapter)
    mock_web.name = "gemini_web"
    mock_web.enabled = True
    mock_web.is_configured.return_value = True
    mock_web.is_available.return_value = True
    mock_web.cooldown_until = 0.0
    mock_web.get_cooldown_remaining.return_value = 0.0
    mock_web.generate_content = AsyncMock(side_effect=RateLimitError("Web rate limited (429)"))

    router = MultiBackendRouter(
        antigravity=mock_agy,
        gemini_api=mock_api,
        gemini_web=mock_web,
        routing_strategy="free_first"
    )

    # Execute generate_content
    result = await router.generate_content(
        model="gemini-2.5-flash",
        contents=[{"role": "user", "parts": [{"text": "hello"}]}]
    )

    # Verify Gemini Web was tried first, then Gemini API, then fell back to Antigravity
    mock_web.generate_content.assert_called_once()
    mock_api.generate_content.assert_called_once()
    mock_agy.generate_content.assert_called_once()

    assert "Antigravity response" in result["candidates"][0]["content"]["parts"][0]["text"]
    assert mock_web.set_cooldown.called
    assert mock_api.set_cooldown.called

@pytest.mark.asyncio
async def test_round_robin_routing():
    """Verify round_robin distributes requests evenly across available backends."""
    mock_agy = MagicMock(spec=AntigravityAdapter)
    mock_agy.name = "antigravity"
    mock_agy.enabled = True
    mock_agy.is_configured.return_value = True
    mock_agy.is_available.return_value = True
    mock_agy.cooldown_until = 0.0
    mock_agy.get_cooldown_remaining.return_value = 0.0
    mock_agy.generate_content = AsyncMock(return_value={"backend": "antigravity"})

    mock_api = MagicMock(spec=GeminiApiAdapter)
    mock_api.name = "gemini_api"
    mock_api.enabled = True
    mock_api.is_configured.return_value = True
    mock_api.is_available.return_value = True
    mock_api.cooldown_until = 0.0
    mock_api.get_cooldown_remaining.return_value = 0.0
    mock_api.generate_content = AsyncMock(return_value={"backend": "gemini_api"})

    mock_web = MagicMock(spec=GeminiWebAdapter)
    mock_web.name = "gemini_web"
    mock_web.enabled = True
    mock_web.is_configured.return_value = True
    mock_web.is_available.return_value = True
    mock_web.cooldown_until = 0.0
    mock_web.get_cooldown_remaining.return_value = 0.0
    mock_web.generate_content = AsyncMock(return_value={"backend": "gemini_web"})

    router = MultiBackendRouter(
        antigravity=mock_agy,
        gemini_api=mock_api,
        gemini_web=mock_web,
        routing_strategy="round_robin"
    )

    calls = []
    for _ in range(6):
        res = await router.generate_content(
            model="gemini-2.5-flash",
            contents=[{"role": "user", "parts": [{"text": "hello"}]}]
        )
        calls.append(res["backend"])

    # Expect each of the 3 adapters to be called twice in round-robin sequence
    assert calls.count("antigravity") == 2
    assert calls.count("gemini_api") == 2
    assert calls.count("gemini_web") == 2

@pytest.mark.asyncio
async def test_streaming_fallback_on_429():
    """Verify stream_generate_content fallback when initial provider rate-limits."""
    async def failing_stream(*args, **kwargs):
        raise RateLimitError("Rate limited (429)", retry_after=30.0)
        yield {}

    async def success_stream(*args, **kwargs):
        yield {"candidates": [{"content": {"parts": [{"text": "streamed success"}]}}]}

    mock_api = MagicMock(spec=GeminiApiAdapter)
    mock_api.name = "gemini_api"
    mock_api.enabled = True
    mock_api.is_configured.return_value = True
    mock_api.is_available.return_value = True
    mock_api.cooldown_until = 0.0
    mock_api.get_cooldown_remaining.return_value = 0.0
    mock_api.stream_generate_content = failing_stream

    mock_agy = MagicMock(spec=AntigravityAdapter)
    mock_agy.name = "antigravity"
    mock_agy.enabled = True
    mock_agy.is_configured.return_value = True
    mock_agy.is_available.return_value = True
    mock_agy.cooldown_until = 0.0
    mock_agy.get_cooldown_remaining.return_value = 0.0
    mock_agy.stream_generate_content = success_stream

    mock_web = MagicMock(spec=GeminiWebAdapter)
    mock_web.name = "gemini_web"
    mock_web.enabled = False

    router = MultiBackendRouter(
        antigravity=mock_agy,
        gemini_api=mock_api,
        gemini_web=mock_web,
        routing_strategy="free_first"
    )

    chunks = []
    async for chunk in router.stream_generate_content(
        model="gemini-2.5-flash",
        contents=[{"role": "user", "parts": [{"text": "hi"}]}]
    ):
        chunks.append(chunk)

    assert len(chunks) == 1
    assert "streamed success" in chunks[0]["candidates"][0]["content"]["parts"][0]["text"]
    assert mock_api.set_cooldown.called

@pytest.mark.asyncio
async def test_models_aggregation_and_ranking():
    """Verify model aggregation ranks models supported across more enabled providers first."""
    mock_agy = MagicMock(spec=AntigravityAdapter)
    mock_agy.name = "antigravity"
    mock_agy.enabled = True
    mock_agy.fetch_available_models = AsyncMock(return_value={
        "models": {
            "gemini-2.5-flash": {"displayName": "Gemini 2.5 Flash"},
            "gemini-3.7-flash-high": {"displayName": "Gemini 3.7 Flash High"}
        }
    })

    mock_api = MagicMock(spec=GeminiApiAdapter)
    mock_api.name = "gemini_api"
    mock_api.enabled = True
    mock_api.fetch_available_models = AsyncMock(return_value={
        "models": {
            "gemini-2.5-flash": {"displayName": "Gemini 2.5 Flash"},
            "gemini-2.5-pro": {"displayName": "Gemini 2.5 Pro"},
            "gemini-2.0-flash": {"displayName": "Gemini 2.0 Flash"}
        }
    })

    mock_web = MagicMock(spec=GeminiWebAdapter)
    mock_web.name = "gemini_web"
    mock_web.enabled = True
    mock_web.fetch_available_models = AsyncMock(return_value={
        "models": {
            "gemini-2.5-flash": {"displayName": "Gemini 2.5 Flash"},
            "gemini-2.5-pro": {"displayName": "Gemini 2.5 Pro"},
        }
    })

    router = MultiBackendRouter(
        antigravity=mock_agy,
        gemini_api=mock_api,
        gemini_web=mock_web
    )

    models_data = await router.fetch_available_models()
    models_dict = models_data["models"]
    model_keys = list(models_dict.keys())

    # gemini-2.5-flash is present in all 3 providers -> MUST be ranked first
    assert model_keys[0] == "gemini-2.5-flash"
    assert models_dict["gemini-2.5-flash"]["provider_count"] == 3
    assert set(models_dict["gemini-2.5-flash"]["providers"]) == {"antigravity", "gemini_api", "gemini_web"}

    # gemini-2.5-pro is present in 2 providers -> ranked next
    assert models_dict["gemini-2.5-pro"]["provider_count"] == 2

    # gemini-3.7-flash-high is present in 1 provider
    assert models_dict["gemini-3.7-flash-high"]["provider_count"] == 1

@pytest.mark.asyncio
async def test_models_redundancy_and_newness_sorting():
    """Verify secondary sorting by model version / newness when redundancy count is tied."""
    mock_agy = MagicMock(spec=AntigravityAdapter)
    mock_agy.name = "antigravity"
    mock_agy.enabled = True
    mock_agy.fetch_available_models = AsyncMock(return_value={
        "models": {
            "gemini-1.5-flash": {"displayName": "Gemini 1.5 Flash"},
            "gemini-2.0-flash": {"displayName": "Gemini 2.0 Flash"},
            "gemini-2.5-flash": {"displayName": "Gemini 2.5 Flash"},
            "gemini-3.7-flash": {"displayName": "Gemini 3.7 Flash"},
        }
    })

    mock_api = MagicMock(spec=GeminiApiAdapter)
    mock_api.name = "gemini_api"
    mock_api.enabled = False

    mock_web = MagicMock(spec=GeminiWebAdapter)
    mock_web.name = "gemini_web"
    mock_web.enabled = False

    router = MultiBackendRouter(
        antigravity=mock_agy,
        gemini_api=mock_api,
        gemini_web=mock_web
    )

    models_data = await router.fetch_available_models()
    models_dict = models_data["models"]
    model_keys = list(models_dict.keys())

    # All models have provider_count = 1, so newness (semantic version) must sort 3.7 > 2.5 > 2.0 > 1.5
    assert model_keys == ["gemini-3.7-flash", "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]

@pytest.mark.asyncio
async def test_models_disabled_backend_exclusion():
    """Verify disabled backends do not contribute to aggregated model listings."""
    mock_agy = MagicMock(spec=AntigravityAdapter)
    mock_agy.name = "antigravity"
    mock_agy.enabled = True
    mock_agy.fetch_available_models = AsyncMock(return_value={
        "models": {
            "gemini-2.5-flash": {"displayName": "Gemini 2.5 Flash"},
        }
    })

    mock_api = MagicMock(spec=GeminiApiAdapter)
    mock_api.name = "gemini_api"
    mock_api.enabled = False  # Disabled
    mock_api.fetch_available_models = AsyncMock(return_value={
        "models": {
            "gemini-2.5-flash": {"displayName": "Gemini 2.5 Flash"},
            "gemini-api-exclusive": {"displayName": "API Only Model"},
        }
    })

    mock_web = MagicMock(spec=GeminiWebAdapter)
    mock_web.name = "gemini_web"
    mock_web.enabled = False

    router = MultiBackendRouter(
        antigravity=mock_agy,
        gemini_api=mock_api,
        gemini_web=mock_web
    )

    models_data = await router.fetch_available_models()
    models_dict = models_data["models"]

    # gemini-api-exclusive must NOT be present since gemini_api is disabled
    assert "gemini-api-exclusive" not in models_dict
    assert "gemini-2.5-flash" in models_dict
    assert models_dict["gemini-2.5-flash"]["providers"] == ["antigravity"]

@pytest.mark.asyncio
async def test_dashboard_backend_api_and_persistence(mock_credentials_file):
    """Verify dashboard API routes for configuring multi-backends and strategy persistence."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        # 1. GET /api/backends
        res = await ac.get("/api/backends")
        assert res.status_code == 200
        data = res.json()
        assert "routing_strategy" in data
        assert "backends" in data
        assert "antigravity" in data["backends"]
        assert "gemini_api" in data["backends"]
        assert "gemini_web" in data["backends"]

        # 2. POST /api/backends (update Gemini API key and Web cookies)
        update_payload = {
            "gemini_api_key": "AIzaSyTestKeyDashboard123",
            "gemini_api_enabled": True,
            "gemini_web_psid": "psid_cookie_val_12345",
            "gemini_web_psidts": "psidts_cookie_val_67890",
            "gemini_web_enabled": True,
            "routing_strategy": "round_robin"
        }
        res_update = await ac.post("/api/backends", json=update_payload)
        assert res_update.status_code == 200
        assert res_update.json()["status"] == "updated"

        # Verify in-memory client updated
        assert client.routing_strategy == "round_robin"
        assert client.gemini_api.api_key == "AIzaSyTestKeyDashboard123"
        assert client.gemini_web.psid == "psid_cookie_val_12345"

        # Verify disk persistence in credentials.json
        with open(mock_credentials_file, "r") as f:
            persisted = json.load(f)
        assert persisted["gemini_api_key"] == "AIzaSyTestKeyDashboard123"
        assert persisted["gemini_web_psid"] == "psid_cookie_val_12345"
        assert persisted["routing_strategy"] == "round_robin"
        # OAuth credentials must still exist
        assert persisted["access_token"] == "ya29.test_oauth_access_token"
        assert persisted["user_email"] == "tester@example.com"

        # 3. POST /api/backends/strategy
        res_strat = await ac.post("/api/backends/strategy", json={"strategy": "free_first"})
        assert res_strat.status_code == 200
        assert client.routing_strategy == "free_first"

        # 4. POST /api/backends/{id}/toggle
        res_toggle = await ac.post("/api/backends/gemini_web/toggle", json={"enabled": False})
        assert res_toggle.status_code == 200
        assert client.gemini_web.enabled is False

        # 5. POST /api/backends/cooldown/clear
        client.gemini_api.set_cooldown(60.0)
        assert client.gemini_api.get_cooldown_remaining() > 0
        res_clear = await ac.post("/api/backends/cooldown/clear", json={})
        assert res_clear.status_code == 200
        assert client.gemini_api.get_cooldown_remaining() == 0.0

        # 6. GET /api/models returns aggregated models list with capabilities metadata
        res_models = await ac.get("/api/models")
        assert res_models.status_code == 200
        models_payload = res_models.json()
        assert models_payload["status"] == "ok"
        assert "models" in models_payload
        assert isinstance(models_payload["models"], list)
        if len(models_payload["models"]) > 0:
            first_model = models_payload["models"][0]
            assert "id" in first_model
            assert "displayName" in first_model
            assert "maxTokens" in first_model
            assert "capabilities" in first_model
            assert "supportsThinking" in first_model
            assert "supportsTools" in first_model
            assert "supportsVision" in first_model
            assert "providers" in first_model

        # 7. GET /health includes backends and routing strategy
        res_health = await ac.get("/health")
        assert res_health.status_code == 200
        health_data = res_health.json()
        assert "routing_strategy" in health_data
        assert "backends" in health_data

@pytest.mark.asyncio
async def test_gemini_api_adapter_streaming_and_generation():
    """Verify GeminiApiAdapter processes stream chunks and formats generation response."""
    adapter = GeminiApiAdapter(api_key="AIzaSyDummyKey")
    
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200

    async def mock_aiter_lines():
        lines = [
            'data: {"candidates": [{"content": {"parts": [{"text": "Hello from "}]}}]}',
            "",
            'data: {"candidates": [{"content": {"parts": [{"text": "Gemini API!"}]}}]}',
            ""
        ]
        for line in lines:
            yield line

    mock_resp.aiter_lines = mock_aiter_lines
    mock_resp.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": "Hello from Gemini API!"}]}}]
    }

    class MockStreamContext:
        async def __aenter__(self):
            return mock_resp
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    mock_client = MagicMock(spec=httpx.AsyncClient)
    mock_client.is_closed = False
    mock_client.send = AsyncMock(return_value=mock_resp)
    mock_client.post = AsyncMock(return_value=mock_resp)
    mock_client.stream = MagicMock(return_value=MockStreamContext())
    adapter._http_client = mock_client
    adapter.get_http_client = MagicMock(return_value=mock_client)

    # Test non-streaming generate_content
    result = await adapter.generate_content(
        model="gemini-2.5-flash",
        contents=[{"role": "user", "parts": [{"text": "Hi"}]}]
    )
    assert "candidates" in result
    assert result["text"] == "Hello from Gemini API!"

    # Test streaming
    chunks = []
    async for chunk in adapter.stream_generate_content(
        model="gemini-2.5-flash",
        contents=[{"role": "user", "parts": [{"text": "Hi"}]}]
    ):
        chunks.append(chunk)

    assert len(chunks) == 2
    assert chunks[0]["candidates"][0]["content"]["parts"][0]["text"] == "Hello from "
    assert chunks[1]["candidates"][0]["content"]["parts"][0]["text"] == "Gemini API!"

@pytest.mark.asyncio
async def test_gemini_web_adapter_prompt_and_generation():
    """Verify GeminiWebAdapter initialization, headers, SAPISIDHASH, and thinking extraction."""
    adapter = GeminiWebAdapter(psid="test_psid_val", psidts="test_psidts_val", sapisid="test_sapisid_val")
    assert adapter.is_configured() is True
    
    headers = adapter._get_headers()
    assert "__Secure-1PSID=test_psid_val" in headers["Cookie"]
    assert "__Secure-1PSIDTS=test_psidts_val" in headers["Cookie"]
    assert "SAPISID=test_sapisid_val" in headers["Cookie"]
    assert "Authorization" in headers
    assert headers["Authorization"].startswith("SAPISIDHASH ")

    # Verify model resolution
    hex_id, model_num, thinking = adapter._resolve_model_metadata("gemini-3.7-flash")
    assert model_num == 5
    assert thinking == 2

    # Mock response with reasoning candidate [37][0][0] and response text [1][0]
    raw_rpc_text = (
        ')]}\'\n'
        '150\n'
        '[["wrb.fr",null,"[[null,null,null,null,[[null,[\\"Hello from Gemini Web!\\"],null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,[[[\\"Thinking step 1\\"]]]]]]]"]]\n'
    )
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.text = raw_rpc_text

    mock_client = AsyncMock()
    mock_client.is_closed = False
    mock_client.post = AsyncMock(return_value=mock_resp)
    mock_client.get = AsyncMock(return_value=MagicMock(status_code=200, text='{"SNlM0e":"mock_snlm0e_token"}'))
    adapter._http_client = mock_client
    adapter.get_http_client = MagicMock(return_value=mock_client)

    result = await adapter.generate_content(
        model="gemini-3.7-flash",
        contents=[{"role": "user", "parts": [{"text": "Hello"}]}]
    )

    assert result["text"] == "Hello from Gemini Web!"
    assert result["thoughts"] == "Thinking step 1"
    assert len(result["candidates"]) == 1
    assert result["candidates"][0]["thoughts"] == "Thinking step 1"
