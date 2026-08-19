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
    """Verify BaseAdapter cooldown mechanics."""
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
    """Verify configuration status across adapters."""
    agy = AntigravityAdapter()
    g_api = GeminiApiAdapter(api_key="")
    g_web = GeminiWebAdapter(psid="", psidts="")

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

    # Verify result came from Antigravity
    assert result["text"] == "Antigravity response"

    # Verify cooldown was called on failing adapters
    mock_web.set_cooldown.assert_called_once()
    mock_api.set_cooldown.assert_called_once()

@pytest.mark.asyncio
async def test_round_robin_routing():
    """Verify round_robin distributes across available adapters."""
    mock_agy = MagicMock(spec=AntigravityAdapter)
    mock_agy.name = "antigravity"
    mock_agy.enabled = True
    mock_agy.is_configured.return_value = True
    mock_agy.is_available.return_value = True
    mock_agy.cooldown_until = 0.0
    mock_agy.get_cooldown_remaining.return_value = 0.0
    mock_agy.generate_content = AsyncMock(return_value={"text": "Antigravity response"})

    mock_api = MagicMock(spec=GeminiApiAdapter)
    mock_api.name = "gemini_api"
    mock_api.enabled = True
    mock_api.is_configured.return_value = True
    mock_api.is_available.return_value = True
    mock_api.cooldown_until = 0.0
    mock_api.get_cooldown_remaining.return_value = 0.0
    mock_api.generate_content = AsyncMock(return_value={"text": "Gemini API response"})

    mock_web = MagicMock(spec=GeminiWebAdapter)
    mock_web.name = "gemini_web"
    mock_web.enabled = True
    mock_web.is_configured.return_value = True
    mock_web.is_available.return_value = True
    mock_web.cooldown_until = 0.0
    mock_web.get_cooldown_remaining.return_value = 0.0
    mock_web.generate_content = AsyncMock(return_value={"text": "Gemini Web response"})

    router = MultiBackendRouter(
        antigravity=mock_agy,
        gemini_api=mock_api,
        gemini_web=mock_web,
        routing_strategy="round_robin"
    )

    r1 = await router.generate_content(model="gemini-2.5-flash", contents=[])
    r2 = await router.generate_content(model="gemini-2.5-flash", contents=[])
    r3 = await router.generate_content(model="gemini-2.5-flash", contents=[])

    # All three adapters should have been called in round-robin order
    called_count = (
        mock_agy.generate_content.call_count +
        mock_api.generate_content.call_count +
        mock_web.generate_content.call_count
    )
    assert called_count == 3
    assert mock_agy.generate_content.call_count == 1
    assert mock_api.generate_content.call_count == 1
    assert mock_web.generate_content.call_count == 1

@pytest.mark.asyncio
async def test_streaming_fallback_on_429():
    """Verify streaming generation automatically falls back when initial connect encounters 429."""
    async def failing_web_stream(*args, **kwargs):
        raise RateLimitError("Web Stream 429 Rate Limit")
        yield {}

    async def successful_api_stream(*args, **kwargs):
        yield {"candidates": [{"content": {"parts": [{"text": "Hello from API streaming"}]}}]}
        yield {"candidates": [{"content": {"parts": [{"text": " done"}]}}]}

    mock_web = MagicMock(spec=GeminiWebAdapter)
    mock_web.name = "gemini_web"
    mock_web.enabled = True
    mock_web.is_configured.return_value = True
    mock_web.is_available.return_value = True
    mock_web.cooldown_until = 0.0
    mock_web.get_cooldown_remaining.return_value = 0.0
    mock_web.stream_generate_content = failing_web_stream

    mock_api = MagicMock(spec=GeminiApiAdapter)
    mock_api.name = "gemini_api"
    mock_api.enabled = True
    mock_api.is_configured.return_value = True
    mock_api.is_available.return_value = True
    mock_api.cooldown_until = 0.0
    mock_api.get_cooldown_remaining.return_value = 0.0
    mock_api.stream_generate_content = successful_api_stream

    mock_agy = MagicMock(spec=AntigravityAdapter)
    mock_agy.name = "antigravity"
    mock_agy.enabled = True
    mock_agy.is_configured.return_value = True
    mock_agy.is_available.return_value = True
    mock_agy.cooldown_until = 0.0
    mock_agy.get_cooldown_remaining.return_value = 0.0

    router = MultiBackendRouter(
        antigravity=mock_agy,
        gemini_api=mock_api,
        gemini_web=mock_web,
        routing_strategy="free_first"
    )

    chunks = []
    async for chunk in router.stream_generate_content(model="gemini-2.5-flash", contents=[]):
        chunks.append(chunk)

    assert len(chunks) == 2
    mock_web.set_cooldown.assert_called_once()
    assert chunks[0]["candidates"][0]["content"]["parts"][0]["text"] == "Hello from API streaming"

@pytest.mark.asyncio
async def test_models_aggregation_and_ranking():
    """Verify /v1/models sorts models supported across highest number of providers at the top."""
    mock_agy = MagicMock(spec=AntigravityAdapter)
    mock_agy.name = "antigravity"
    mock_agy.enabled = True
    mock_agy.fetch_available_models = AsyncMock(return_value={
        "models": {
            "gemini-2.5-flash": {"displayName": "Gemini 2.5 Flash"},
            "gemini-3.7-flash-high": {"displayName": "Gemini 3.7 Flash High"},
            "claude-sonnet-4-6": {"displayName": "Claude Sonnet 3.7"}
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

        # 6. GET /health includes backends and routing strategy
        res_health = await ac.get("/health")
        assert res_health.status_code == 200
        health_data = res_health.json()
        assert "routing_strategy" in health_data
        assert "backends" in health_data

@pytest.mark.asyncio
async def test_gemini_api_adapter_streaming_and_generation():
    """Verify GeminiApiAdapter processes stream chunks and formats generation response."""
    adapter = GeminiApiAdapter(api_key="AIzaSyDummyKey")
    
    # Mock httpx response with SSE lines
    sse_content = (
        b"data: {\"candidates\": [{\"content\": {\"parts\": [{\"text\": \"Hello from \"}]}, \"index\": 0}]}\n\n"
        b"data: {\"candidates\": [{\"content\": {\"parts\": [{\"text\": \"Gemini API!\"}]}, \"index\": 0, \"finishReason\": \"STOP\"}], \"usageMetadata\": {\"promptTokenCount\": 2, \"candidatesTokenCount\": 4}}\n\n"
    )

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.aiter_lines = MagicMock(return_value=_async_lines([
        'data: {"candidates": [{"content": {"parts": [{"text": "Hello from "}]}, "index": 0}]}',
        'data: {"candidates": [{"content": {"parts": [{"text": "Gemini API!"}]}, "index": 0, "finishReason": "STOP"}], "usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 4}}'
    ]))

    class MockStreamContext:
        async def __aenter__(self):
            return mock_resp
        async def __aexit__(self, *args):
            pass

    mock_http = MagicMock()
    mock_http.is_closed = False
    mock_http.stream = MagicMock(return_value=MockStreamContext())
    adapter._http_client = mock_http

    result = await adapter.generate_content(
        model="gemini-2.5-flash",
        contents=[{"role": "user", "parts": [{"text": "hi"}]}]
    )

    assert result["text"] == "Hello from Gemini API!"
    assert result["finishReason"] == "STOP"
    assert result["usageMetadata"]["promptTokenCount"] == 2
    assert result["usageMetadata"]["candidatesTokenCount"] == 4

@pytest.mark.asyncio
async def test_gemini_web_adapter_prompt_and_generation():
    """Verify GeminiWebAdapter extracts prompt and structures responses."""
    adapter = GeminiWebAdapter(psid="dummy_psid", psidts="dummy_psidts")
    
    # Test prompt flattening
    contents = [
        {"role": "user", "parts": [{"text": "What is Python?"}]},
        {"role": "model", "parts": [{"text": "Python is a programming language."}]},
        {"role": "user", "parts": [{"text": "Tell me more."}]}
    ]
    system_inst = {"parts": [{"text": "Be concise."}]}
    prompt = adapter._extract_prompt_text(contents, system_inst)
    assert "System: Be concise." in prompt
    assert "User: What is Python?" in prompt
    assert "Model: Python is a programming language." in prompt
    assert "User: Tell me more." in prompt

    # Mock web response
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = ')]}\'\n[["wrb.fr",null,"[null,null,null,null,[[null,[\\"Python is an interpreted, high-level language.\\"]]]]"]]'

    mock_http = MagicMock()
    mock_http.is_closed = False
    mock_http.post = AsyncMock(return_value=mock_resp)
    mock_http.get = AsyncMock(return_value=MagicMock(status_code=200, text='window.WIZ_global_data = {"SNlM0e":"mock_snlm0e_token"};'))
    adapter._http_client = mock_http

    result = await adapter.generate_content(
        model="gemini-2.5-flash",
        contents=contents
    )

    assert "Python is an interpreted" in result["text"]
    assert result["finishReason"] == "STOP"

async def _async_lines(lines):
    for line in lines:
        yield line
