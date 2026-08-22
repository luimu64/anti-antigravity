import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.client import client
from app.providers.antigravity import AntigravityAdapter
from app.providers.base import BaseAdapter, RateLimitError
from app.providers.gemini_api import GeminiApiAdapter
from app.providers.gemini_web import GeminiWebAdapter
from app.providers.router import MultiBackendRouter
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
        "routing_strategy": "free_first",
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
    mock_agy.generate_content = AsyncMock(
        return_value={
            "text": "Antigravity response",
            "candidates": [{"content": {"parts": [{"text": "Antigravity response"}]}}],
        }
    )

    mock_api = MagicMock(spec=GeminiApiAdapter)
    mock_api.name = "gemini_api"
    mock_api.enabled = True
    mock_api.is_configured.return_value = True
    mock_api.is_available.return_value = True
    mock_api.cooldown_until = 0.0
    mock_api.get_cooldown_remaining.return_value = 0.0
    mock_api.generate_content = AsyncMock(
        side_effect=RateLimitError("API quota exhausted (429)")
    )

    mock_web = MagicMock(spec=GeminiWebAdapter)
    mock_web.name = "gemini_web"
    mock_web.enabled = True
    mock_web.is_configured.return_value = True
    mock_web.is_available.return_value = True
    mock_web.cooldown_until = 0.0
    mock_web.get_cooldown_remaining.return_value = 0.0
    mock_web.generate_content = AsyncMock(
        side_effect=RateLimitError("Web rate limited (429)")
    )

    router = MultiBackendRouter(
        antigravity=mock_agy,
        gemini_api=mock_api,
        gemini_web=mock_web,
        routing_strategy="free_first",
    )

    # Execute generate_content
    result = await router.generate_content(
        model="gemini-2.0-flash",
        contents=[{"role": "user", "parts": [{"text": "hello"}]}],
    )

    # Verify Gemini Web was tried first, then Gemini API, then fell back to Antigravity
    mock_web.generate_content.assert_called_once()
    mock_api.generate_content.assert_called_once()
    mock_agy.generate_content.assert_called_once()

    assert (
        "Antigravity response" in result["candidates"][0]["content"]["parts"][0]["text"]
    )
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
        routing_strategy="round_robin",
    )

    calls = []
    for _ in range(6):
        res = await router.generate_content(
            model="gemini-2.0-flash",
            contents=[{"role": "user", "parts": [{"text": "hello"}]}],
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
        routing_strategy="free_first",
    )

    chunks = []
    async for chunk in router.stream_generate_content(
        model="gemini-2.0-flash", contents=[{"role": "user", "parts": [{"text": "hi"}]}]
    ):
        chunks.append(chunk)

    assert len(chunks) == 1
    assert (
        "streamed success" in chunks[0]["candidates"][0]["content"]["parts"][0]["text"]
    )
    assert mock_api.set_cooldown.called


@pytest.mark.asyncio
async def test_models_aggregation_and_ranking():
    """Verify model aggregation ranks models supported across more enabled providers first."""
    mock_agy = MagicMock(spec=AntigravityAdapter)
    mock_agy.name = "antigravity"
    mock_agy.enabled = True
    mock_agy.fetch_available_models = AsyncMock(
        return_value={
            "models": {
                "gemini-2.5-flash": {"displayName": "Gemini 2.5 Flash"},
                "gemini-3.7-flash-high": {"displayName": "Gemini 3.7 Flash High"},
            }
        }
    )

    mock_api = MagicMock(spec=GeminiApiAdapter)
    mock_api.name = "gemini_api"
    mock_api.enabled = True
    mock_api.fetch_available_models = AsyncMock(
        return_value={
            "models": {
                "gemini-2.5-flash": {"displayName": "Gemini 2.5 Flash"},
                "gemini-2.5-pro": {"displayName": "Gemini 2.5 Pro"},
                "gemini-2.0-flash": {"displayName": "Gemini 2.0 Flash"},
            }
        }
    )

    mock_web = MagicMock(spec=GeminiWebAdapter)
    mock_web.name = "gemini_web"
    mock_web.enabled = True
    mock_web.fetch_available_models = AsyncMock(
        return_value={
            "models": {
                "gemini-2.5-flash": {"displayName": "Gemini 2.5 Flash"},
                "gemini-2.5-pro": {"displayName": "Gemini 2.5 Pro"},
            }
        }
    )

    router = MultiBackendRouter(
        antigravity=mock_agy, gemini_api=mock_api, gemini_web=mock_web
    )

    models_data = await router.fetch_available_models()
    models_dict = models_data["models"]
    model_keys = list(models_dict.keys())

    # gemini-2.5-flash is present in all 3 providers -> MUST be ranked first
    assert model_keys[0] == "gemini-2.5-flash"
    assert models_dict["gemini-2.5-flash"]["provider_count"] == 3
    assert set(models_dict["gemini-2.5-flash"]["providers"]) == {
        "antigravity",
        "gemini_api",
        "gemini_web",
    }

    # gemini-2.5-pro is present in 2 providers -> ranked next
    assert models_dict["gemini-2.5-pro"]["provider_count"] == 2

    # gemini-3.7-flash-high is consolidated to canonical gemini-3.7-flash with provider_count = 1
    assert "gemini-3.7-flash" in models_dict
    assert models_dict["gemini-3.7-flash"]["provider_count"] == 1


@pytest.mark.asyncio
async def test_models_redundancy_and_newness_sorting():
    """Verify secondary sorting by model version / newness when redundancy count is tied."""
    mock_agy = MagicMock(spec=AntigravityAdapter)
    mock_agy.name = "antigravity"
    mock_agy.enabled = True
    mock_agy.fetch_available_models = AsyncMock(
        return_value={
            "models": {
                "gemini-1.5-flash": {"displayName": "Gemini 1.5 Flash"},
                "gemini-2.0-flash": {"displayName": "Gemini 2.0 Flash"},
                "gemini-2.5-flash": {"displayName": "Gemini 2.5 Flash"},
                "gemini-3.7-flash": {"displayName": "Gemini 3.7 Flash"},
            }
        }
    )

    mock_api = MagicMock(spec=GeminiApiAdapter)
    mock_api.name = "gemini_api"
    mock_api.enabled = False

    mock_web = MagicMock(spec=GeminiWebAdapter)
    mock_web.name = "gemini_web"
    mock_web.enabled = False

    router = MultiBackendRouter(
        antigravity=mock_agy, gemini_api=mock_api, gemini_web=mock_web
    )

    models_data = await router.fetch_available_models()
    models_dict = models_data["models"]
    model_keys = list(models_dict.keys())

    # All models have provider_count = 1, so newness (semantic version) must sort 3.7 > 2.5 > 2.0 > 1.5
    assert model_keys == [
        "gemini-3.7-flash",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
    ]


@pytest.mark.asyncio
async def test_models_disabled_backend_exclusion():
    """Verify disabled backends do not contribute to aggregated model listings."""
    mock_agy = MagicMock(spec=AntigravityAdapter)
    mock_agy.name = "antigravity"
    mock_agy.enabled = True
    mock_agy.fetch_available_models = AsyncMock(
        return_value={
            "models": {
                "gemini-2.5-flash": {"displayName": "Gemini 2.5 Flash"},
            }
        }
    )

    mock_api = MagicMock(spec=GeminiApiAdapter)
    mock_api.name = "gemini_api"
    mock_api.enabled = False  # Disabled
    mock_api.fetch_available_models = AsyncMock(
        return_value={
            "models": {
                "gemini-2.5-flash": {"displayName": "Gemini 2.5 Flash"},
                "gemini-api-exclusive": {"displayName": "API Only Model"},
            }
        }
    )

    mock_web = MagicMock(spec=GeminiWebAdapter)
    mock_web.name = "gemini_web"
    mock_web.enabled = False

    router = MultiBackendRouter(
        antigravity=mock_agy, gemini_api=mock_api, gemini_web=mock_web
    )

    models_data = await router.fetch_available_models()
    models_dict = models_data["models"]

    # gemini-api-exclusive must NOT be present since gemini_api is disabled
    assert "gemini-api-exclusive" not in models_dict
    assert "gemini-2.5-flash" in models_dict
    assert models_dict["gemini-2.5-flash"]["providers"] == ["antigravity"]


@pytest.mark.asyncio
async def test_models_hidden_and_reasoning_tier_mapping():
    """Verify unusable models are marked as hidden and reasoning tiers are mapped under the hood."""
    from app.translator import OpenAITranslator

    # 1. Verify reasoning tiers map silently under the hood
    assert (
        OpenAITranslator.resolve_model("gemini-3.7-flash", reasoning_effort="low")
        == "gemini-3.7-flash-low"
    )
    assert (
        OpenAITranslator.resolve_model("gemini-3.7-flash", reasoning_effort="medium")
        == "gemini-3.7-flash-medium"
    )
    assert (
        OpenAITranslator.resolve_model("gemini-3.7-flash", reasoning_effort="high")
        == "gemini-3.7-flash-high"
    )
    assert OpenAITranslator.resolve_model("gemini-3.7-flash") == "gemini-3.7-flash-high"

    assert (
        OpenAITranslator.resolve_model("gemini-3.6-flash", reasoning_effort="low")
        == "gemini-3.6-flash-low"
    )
    assert (
        OpenAITranslator.resolve_model("gemini-3.1-pro", reasoning_effort="low")
        == "gemini-3.1-pro-low"
    )
    assert OpenAITranslator.resolve_model("gemini-3.5-flash") == "gemini-3-flash-agent"
    assert OpenAITranslator.resolve_model("claude-3.7-sonnet") == "claude-sonnet-4-6"
    assert OpenAITranslator.resolve_model("claude-3-opus") == "claude-opus-4-6-thinking"
    assert OpenAITranslator.resolve_model("gpt-oss-120b") == "gpt-oss-120b-medium"

    # 2. Verify hidden model detection
    mock_agy = MagicMock(spec=AntigravityAdapter)
    mock_agy.name = "antigravity"
    mock_agy.enabled = True
    mock_agy.fetch_available_models = AsyncMock(
        return_value={
            "models": {
                "gemini-3.7-flash-high": {"displayName": "Gemini 3.7 Flash High"},
                "tab_flash_lite_preview": {"displayName": "Tab Flash Lite Preview"},
            }
        }
    )

    mock_api = MagicMock(spec=GeminiApiAdapter)
    mock_api.name = "gemini_api"
    mock_api.enabled = False

    mock_web = MagicMock(spec=GeminiWebAdapter)
    mock_web.name = "gemini_web"
    mock_web.enabled = False

    router = MultiBackendRouter(
        antigravity=mock_agy, gemini_api=mock_api, gemini_web=mock_web
    )

    models_data = await router.fetch_available_models()
    models_dict = models_data["models"]

    assert "tab_flash_lite_preview" in models_dict
    assert models_dict["tab_flash_lite_preview"]["hidden"] is True
    assert models_dict["gemini-3.7-flash"]["hidden"] is False


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
            "routing_strategy": "round_robin",
        }
        res_update = await ac.post("/api/backends", json=update_payload)
        assert res_update.status_code == 200
        assert res_update.json()["status"] == "updated"

        # Verify in-memory client updated
        assert client.routing_strategy == "round_robin"
        assert client.gemini_api.api_key == "AIzaSyTestKeyDashboard123"
        assert client.gemini_web.psid == "psid_cookie_val_12345"

        # Verify disk persistence in credentials.json
        with open(mock_credentials_file) as f:
            persisted = json.load(f)
        assert persisted["gemini_api_key"] == "AIzaSyTestKeyDashboard123"
        assert persisted["gemini_web_psid"] == "psid_cookie_val_12345"
        assert persisted["routing_strategy"] == "round_robin"
        # OAuth credentials must still exist
        assert persisted["access_token"] == "ya29.test_oauth_access_token"
        assert persisted["user_email"] == "tester@example.com"

        # 3. POST /api/backends/strategy
        res_strat = await ac.post(
            "/api/backends/strategy", json={"strategy": "free_first"}
        )
        assert res_strat.status_code == 200
        assert client.routing_strategy == "free_first"

        # 4. POST /api/backends/{id}/toggle
        res_toggle = await ac.post(
            "/api/backends/gemini_web/toggle", json={"enabled": False}
        )
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
            "",
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
        model="gemini-2.0-flash", contents=[{"role": "user", "parts": [{"text": "Hi"}]}]
    )
    assert "candidates" in result
    assert result["text"] == "Hello from Gemini API!"

    # Test streaming
    chunks = []
    async for chunk in adapter.stream_generate_content(
        model="gemini-2.0-flash", contents=[{"role": "user", "parts": [{"text": "Hi"}]}]
    ):
        chunks.append(chunk)

    assert len(chunks) == 2
    assert chunks[0]["candidates"][0]["content"]["parts"][0]["text"] == "Hello from "
    assert chunks[1]["candidates"][0]["content"]["parts"][0]["text"] == "Gemini API!"


@pytest.mark.asyncio
async def test_gemini_web_adapter_prompt_and_generation():
    """Verify GeminiWebAdapter initialization, headers, SAPISIDHASH, and thinking extraction."""
    adapter = GeminiWebAdapter(
        psid="test_psid_val", psidts="test_psidts_val", sapisid="test_sapisid_val"
    )
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
        ")]}'\n"
        "150\n"
        '[["wrb.fr",null,"[[null,null,null,null,[[null,[\\"Hello from Gemini Web!\\"],null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,[[[\\"Thinking step 1\\"]]]]]]]"]]\n'
    )
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.text = raw_rpc_text

    mock_client = AsyncMock()
    mock_client.is_closed = False
    mock_client.post = AsyncMock(return_value=mock_resp)
    mock_client.get = AsyncMock(
        return_value=MagicMock(status_code=200, text='{"SNlM0e":"mock_snlm0e_token"}')
    )
    adapter._http_client = mock_client
    adapter.get_http_client = MagicMock(return_value=mock_client)

    result = await adapter.generate_content(
        model="gemini-3.7-flash",
        contents=[{"role": "user", "parts": [{"text": "Hello"}]}],
    )

    assert result["text"] == "Hello from Gemini Web!"
    assert result["thoughts"] == "Thinking step 1"
    assert len(result["candidates"]) == 1
    assert result["candidates"][0]["thoughts"] == "Thinking step 1"


def test_vision_model_mapping_across_all_adapters():
    """Verify GeminiApiAdapter and GeminiWebAdapter map vision and -image models to latest Flash targets."""
    api_adapter = GeminiApiAdapter(api_key="test-key")
    assert api_adapter._normalize_model_name("vision") == "gemini-2.0-flash"
    assert (
        api_adapter._normalize_model_name("gemini-3.7-flash-image")
        == "gemini-2.0-flash"
    )

    web_adapter = GeminiWebAdapter(psid="test-psid")
    hex_id_v, model_num_v, thinking_v = web_adapter._resolve_model_metadata("vision")
    assert hex_id_v == "2c8a"
    assert model_num_v == 5

    hex_id_img, model_num_img, thinking_img = web_adapter._resolve_model_metadata(
        "gemini-3.7-flash-image"
    )
    assert hex_id_img == "2c8a"
    assert model_num_img == 5


def test_in_memory_rate_tracker_sliding_window():
    """Verify in-memory proactive rate limit accounting, sliding windows, and resets."""
    from app.providers.base import InMemoryRateTracker

    tracker = InMemoryRateTracker(rpm=3, tpm=100, rpd=5)
    now = 1000.0

    # Initial state
    assert tracker.has_capacity(estimated_tokens=10, now=now) is True
    assert tracker.get_window_reset_remaining(now=now) == 0.0

    # Record 3 requests within the same minute
    tracker.record_usage(tokens=30, now=now)
    tracker.record_usage(tokens=30, now=now + 10)
    tracker.record_usage(tokens=30, now=now + 20)

    # RPM limit (3) is reached
    assert tracker.has_capacity(estimated_tokens=5, now=now + 25) is False
    reset_wait = tracker.get_window_reset_remaining(now=now + 25)
    assert 34.0 <= reset_wait <= 36.0  # 60 - (1025 - 1000) = 35s

    # Advance time past the first request (> 60s from t=1000)
    now_next = now + 65.0
    assert tracker.has_capacity(estimated_tokens=5, now=now_next) is True

    # Test TPM limit
    tracker.record_usage(
        tokens=50, now=now_next
    )  # total tokens now: 30+30+50 = 110 > 100
    assert tracker.has_capacity(estimated_tokens=10, now=now_next) is False

    # Stats snapshot
    stats = tracker.get_stats(now=now_next)
    assert stats["rpm_limit"] == 3
    assert stats["tpm_limit"] == 100
    assert stats["has_capacity"] is False

    # Reset clears all in-memory counters
    tracker.reset()
    assert tracker.has_capacity(estimated_tokens=50, now=now_next) is True
    assert tracker.get_stats(now=now_next)["rpm_used"] == 0


def test_adapter_proactive_capacity_and_cooldown_hybrid():
    """Verify BaseAdapter evaluates proactive rate limits and reactive cooldowns."""
    from app.providers.gemini_api import GeminiApiAdapter

    adapter = GeminiApiAdapter(api_key="test-key", enabled=True)
    adapter.rate_limiter.rpm = 2
    assert adapter.is_available() is True

    # Consume capacity
    adapter.record_usage(tokens=10)
    adapter.record_usage(tokens=10)
    assert adapter.is_available() is False
    assert adapter.get_cooldown_remaining() > 0.0

    # Clear cooldown resets rate limiter too
    adapter.clear_cooldown()
    assert adapter.is_available() is True


@pytest.mark.asyncio
async def test_hybrid_routing_proactive_exhaustion_fallback():
    """Verify router transparently falls back to next backend when first backend reaches proactive rate limit."""
    mock_web = MagicMock(spec=GeminiWebAdapter)
    mock_web.name = "gemini_web"
    mock_web.enabled = True
    mock_web.is_configured.return_value = True
    mock_web.cooldown_until = 0.0
    mock_web.get_cooldown_remaining.return_value = 0.0
    # Simulate web backend having capacity for only 2 requests
    web_calls = 0

    def web_is_available(*args, **kwargs):
        return web_calls < 2

    mock_web.is_available = MagicMock(side_effect=web_is_available)
    mock_web.record_usage = MagicMock()

    async def web_generate(*args, **kwargs):
        nonlocal web_calls
        web_calls += 1
        return {"backend": "gemini_web"}

    mock_web.generate_content = AsyncMock(side_effect=web_generate)

    mock_api = MagicMock(spec=GeminiApiAdapter)
    mock_api.name = "gemini_api"
    mock_api.enabled = True
    mock_api.is_configured.return_value = True
    mock_api.is_available.return_value = True
    mock_api.cooldown_until = 0.0
    mock_api.get_cooldown_remaining.return_value = 0.0
    mock_api.record_usage = MagicMock()
    mock_api.generate_content = AsyncMock(return_value={"backend": "gemini_api"})

    mock_agy = MagicMock(spec=AntigravityAdapter)
    mock_agy.name = "antigravity"
    mock_agy.enabled = False

    router = MultiBackendRouter(
        antigravity=mock_agy,
        gemini_api=mock_api,
        gemini_web=mock_web,
        routing_strategy="free_first",
    )

    # First two requests go to gemini_web
    res1 = await router.generate_content(
        model="gemini-2.0-flash", contents=[{"role": "user", "parts": [{"text": "1"}]}]
    )
    res2 = await router.generate_content(
        model="gemini-2.0-flash", contents=[{"role": "user", "parts": [{"text": "2"}]}]
    )
    assert res1["backend"] == "gemini_web"
    assert res2["backend"] == "gemini_web"

    # Third request proactively skips gemini_web (out of capacity) and routes to gemini_api!
    res3 = await router.generate_content(
        model="gemini-2.0-flash", contents=[{"role": "user", "parts": [{"text": "3"}]}]
    )
    assert res3["backend"] == "gemini_api"


@pytest.mark.asyncio
async def test_all_backends_exhausted_immediate_429():
    """Verify router immediately raises RateLimitError (429) when all capable backends are exhausted."""
    mock_web = MagicMock(spec=GeminiWebAdapter)
    mock_web.name = "gemini_web"
    mock_web.enabled = True
    mock_web.is_configured.return_value = True
    mock_web.is_available.return_value = False
    mock_web.get_cooldown_remaining.return_value = 45.0

    mock_api = MagicMock(spec=GeminiApiAdapter)
    mock_api.name = "gemini_api"
    mock_api.enabled = True
    mock_api.is_configured.return_value = True
    mock_api.is_available.return_value = False
    mock_api.get_cooldown_remaining.return_value = 30.0

    mock_agy = MagicMock(spec=AntigravityAdapter)
    mock_agy.name = "antigravity"
    mock_agy.enabled = False

    router = MultiBackendRouter(
        antigravity=mock_agy,
        gemini_api=mock_api,
        gemini_web=mock_web,
        routing_strategy="free_first",
    )

    with pytest.raises(RateLimitError) as exc_info:
        await router.generate_content(
            model="gemini-2.0-flash",
            contents=[{"role": "user", "parts": [{"text": "hi"}]}],
        )

    assert exc_info.value.status_code == 429
    assert exc_info.value.retry_after == 30.0  # min cooldown remaining


@pytest.mark.asyncio
async def test_antigravity_quota_summary_exhaustion_trigger():
    """Verify AntigravityAdapter triggers cooldown when quota bucket is exhausted."""
    adapter = AntigravityAdapter()
    adapter.min_quota_fraction = 0.05
    adapter.default_cooldown = 120.0

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "groups": [
            {
                "groupId": "antigravity_general",
                "buckets": [
                    {
                        "bucketId": "weekly",
                        "displayName": "Weekly Limit",
                        "remainingFraction": 0.02,  # < 0.05 min threshold
                        "resetTime": "2030-01-01T00:00:00Z",
                    }
                ],
            }
        ]
    }

    adapter._get_headers = AsyncMock(return_value={})
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)
    mock_client.is_closed = False
    adapter._http_client = mock_client
    adapter.get_http_client = MagicMock(return_value=mock_client)

    await adapter.retrieve_user_quota_summary()

    # Cooldown should now be active
    assert adapter.get_cooldown_remaining() > 0.0


def test_base_adapter_get_rate_limit_quotas():
    """Verify BaseAdapter.get_rate_limit_quotas computes normalized quota and rate limit items."""
    api_adapter = GeminiApiAdapter(api_key="AIzaSyTest", enabled=True)
    # Default rates: rpm=15, tpm=1000000, rpd=1500
    quotas = api_adapter.get_rate_limit_quotas()
    assert len(quotas) == 3

    rpm_q = next(q for q in quotas if "RPM" in q["display_name"])
    assert rpm_q["fraction_used"] == 0.0
    assert rpm_q["remaining_fraction"] == 1.0
    assert rpm_q["fraction_remaining"] == 1.0
    assert rpm_q["backend"] == "gemini_api"
    assert rpm_q["source"] == "gemini_api"
    assert rpm_q["model_id"] == "gemini_api"

    # Record usage
    api_adapter.record_usage(tokens=250000)
    quotas_used = api_adapter.get_rate_limit_quotas()
    rpm_used = next(q for q in quotas_used if "RPM" in q["display_name"])
    tpm_used = next(q for q in quotas_used if "TPM" in q["display_name"])

    assert rpm_used["fraction_used"] > 0.0
    assert rpm_used["remaining_fraction"] < 1.0
    assert rpm_used["reset_time_seconds"] > 0.0

    assert pytest.approx(tpm_used["fraction_used"], 0.001) == 0.25
    assert pytest.approx(tpm_used["remaining_fraction"], 0.001) == 0.75

    # Test cooldown state
    api_adapter.set_cooldown(45.0)
    quotas_cd = api_adapter.get_rate_limit_quotas()
    for q in quotas_cd:
        assert q["fraction_used"] == 1.0
        assert q["remaining_fraction"] == 0.0
        assert pytest.approx(q["reset_time_seconds"], 1.0) == 45.0

    # Gemini Web (rpm=60, tpm=500000, rpd=0)
    web_adapter = GeminiWebAdapter(psid="test-psid", enabled=True)
    web_quotas = web_adapter.get_rate_limit_quotas()
    assert len(web_quotas) == 2
    web_names = [q["display_name"] for q in web_quotas]
    assert "Requests Per Minute (RPM)" in web_names
    assert "Tokens Per Minute (TPM)" in web_names
    for q in web_quotas:
        assert q["backend"] == "gemini_web"
        assert q["source"] == "gemini_web"


@pytest.mark.asyncio
async def test_dashboard_api_quotas_multi_backend_aggregation():
    """Verify /api/quotas aggregates quotas from Antigravity and enabled multi-backends."""
    transport = httpx.ASGITransport(app=app)

    mock_agy_quotas = {
        "groups": [
            {
                "groupId": "antigravity_general",
                "buckets": [
                    {
                        "bucketId": "weekly",
                        "displayName": "Weekly Quota",
                        "remainingFraction": 0.80,
                        "resetTime": "2030-01-01T00:00:00Z",
                    }
                ],
            }
        ]
    }

    # Temporarily enable gemini_api and gemini_web on global client
    client.gemini_api.enabled = True
    client.gemini_web.enabled = True
    client.gemini_api.rate_limiter.reset()
    client.gemini_web.rate_limiter.reset()
    client.gemini_api.clear_cooldown()
    client.gemini_web.clear_cooldown()

    try:
        with patch.object(
            client,
            "retrieve_user_quota_summary",
            new_callable=AsyncMock,
            return_value=mock_agy_quotas,
        ):
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as ac:
                res = await ac.get("/api/quotas")
                assert res.status_code == 200
                data = res.json()
                assert "groups" in data
                groups = data["groups"]

                # 1 antigravity bucket + 3 gemini_api buckets + 2 gemini_web buckets = 6 items
                assert len(groups) == 6

                backends_in_groups = [g["backend"] for g in groups]
                assert "antigravity" in backends_in_groups
                assert "gemini_api" in backends_in_groups
                assert "gemini_web" in backends_in_groups

                # Check antigravity item
                agy_item = next(g for g in groups if g["backend"] == "antigravity")
                assert agy_item["display_name"] == "Weekly Quota"
                assert agy_item["remaining_fraction"] == 0.80

                # Check gemini_api item
                api_item = next(g for g in groups if g["backend"] == "gemini_api")
                assert (
                    "RPM" in api_item["display_name"]
                    or "TPM" in api_item["display_name"]
                )
                assert api_item["model_id"] == "gemini_api"

                # Check gemini_web item
                web_item = next(g for g in groups if g["backend"] == "gemini_web")
                assert (
                    "RPM" in web_item["display_name"]
                    or "TPM" in web_item["display_name"]
                )
                assert web_item["model_id"] == "gemini_web"

        # When Antigravity fails but Gemini API is enabled, still return multi-backend quotas
        with patch.object(
            client,
            "retrieve_user_quota_summary",
            new_callable=AsyncMock,
            side_effect=RuntimeError("Google OAuth expired"),
        ):
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as ac:
                res_fail = await ac.get("/api/quotas")
                assert res_fail.status_code == 200
                data_fail = res_fail.json()
                assert len(data_fail["groups"]) == 5  # 3 gemini_api + 2 gemini_web
                assert all(
                    g["backend"] in ("gemini_api", "gemini_web")
                    for g in data_fail["groups"]
                )
    finally:
        client.gemini_api.enabled = False
        client.gemini_web.enabled = False


@pytest.mark.asyncio
async def test_gemini_api_probe_plan_tiers():
    """Verify GeminiApiAdapter.probe_plan identifies Free, PAYG, Invalid, and offline fallback states."""
    adapter = GeminiApiAdapter(api_key="AIzaSyTestKey")

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.is_closed = False
    adapter._http_client = mock_client
    adapter.get_http_client = MagicMock(return_value=mock_client)

    # 1. Free Tier (HTTP 400 FAILED_PRECONDITION)
    mock_resp_free = MagicMock(spec=httpx.Response)
    mock_resp_free.status_code = 400
    mock_resp_free.text = '{"error": {"status": "FAILED_PRECONDITION", "message": "User location or billing is not supported for cached content"}}'
    mock_client.post.return_value = mock_resp_free

    res_free = await adapter.probe_plan(force_refresh=True)
    assert res_free["plan_tier"] == "Free"
    assert res_free["valid"] is True
    assert adapter.plan_tier == "Free"
    assert adapter.is_valid_key is True

    # 2. Pay-As-You-Go Tier (HTTP 400 INVALID_ARGUMENT or HTTP 200)
    mock_resp_payg = MagicMock(spec=httpx.Response)
    mock_resp_payg.status_code = 400
    mock_resp_payg.text = '{"error": {"status": "INVALID_ARGUMENT", "message": "Required field contents is missing"}}'
    mock_client.post.return_value = mock_resp_payg

    res_payg = await adapter.probe_plan(force_refresh=True)
    assert res_payg["plan_tier"] == "Pay-As-You-Go"
    assert res_payg["valid"] is True
    assert adapter.plan_tier == "Pay-As-You-Go"

    mock_resp_200 = MagicMock(spec=httpx.Response)
    mock_resp_200.status_code = 200
    mock_resp_200.text = "{}"
    mock_client.post.return_value = mock_resp_200

    res_200 = await adapter.probe_plan(force_refresh=True)
    assert res_200["plan_tier"] == "Pay-As-You-Go"
    assert res_200["valid"] is True

    # 3. Invalid API Key (HTTP 400 / 403 API_KEY_INVALID)
    mock_resp_inv = MagicMock(spec=httpx.Response)
    mock_resp_inv.status_code = 400
    mock_resp_inv.text = '{"error": {"status": "INVALID_ARGUMENT", "message": "API key not valid. Please pass a valid API_KEY_INVALID key."}}'
    mock_client.post.return_value = mock_resp_inv

    res_inv = await adapter.probe_plan(force_refresh=True)
    assert res_inv["plan_tier"] == "Unknown"
    assert res_inv["valid"] is False
    assert adapter.is_valid_key is False

    # 4. Offline / Network error fallback
    mock_client.post.side_effect = httpx.NetworkError("DNS failure")
    res_err = await adapter.probe_plan(force_refresh=True)
    assert "plan_tier" in res_err
    assert "valid" in res_err

    # 5. Unconfigured key returns Unknown / False
    adapter.api_key = ""
    res_unconf = await adapter.probe_plan(force_refresh=True)
    assert res_unconf["plan_tier"] == "Unknown"
    assert res_unconf["valid"] is False


@pytest.mark.asyncio
async def test_gemini_web_profile_extraction():
    """Verify GeminiWebAdapter.fetch_user_profile extracts WIZ_global_data, account widget, and quota tier."""
    adapter = GeminiWebAdapter(psid="test_psid", psidts="test_psidts")

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.is_closed = False
    adapter._http_client = mock_client
    adapter.get_http_client = MagicMock(return_value=mock_client)

    # 1. Successful extraction from WIZ_global_data and retrieveUserQuota
    app_html = (
        "<html><script>"
        "window.WIZ_global_data = {"
        '"o6v9De":"webuser@example.com",'
        '"S06Grb":"10293847561029384756",'
        '"LVIXXb":"https:\\/\\/lh3.googleusercontent.com\\/a\\/mock-photo\\u003ds96-c",'
        '"SNlM0e":"mock_xsrf_token",'
        '"cfb2h":"boq_mock_build_label"'
        "};"
        "</script></html>"
    )
    mock_resp_app = MagicMock(spec=httpx.Response)
    mock_resp_app.status_code = 200
    mock_resp_app.text = app_html

    mock_resp_quota = MagicMock(spec=httpx.Response)
    mock_resp_quota.status_code = 200
    mock_resp_quota.json.return_value = {
        "tier": {"name": "Gemini Advanced", "id": "tier_advanced"}
    }

    async def mock_get(url, *args, **kwargs):
        if "retrieveUserQuota" in str(url):
            return mock_resp_quota
        return mock_resp_app

    mock_client.get = AsyncMock(side_effect=mock_get)

    profile = await adapter.fetch_user_profile(force_refresh=True)
    assert profile["user_email"] == "webuser@example.com"
    assert profile["account_id"] == "10293847561029384756"
    assert (
        profile["avatar_url"] == "https://lh3.googleusercontent.com/a/mock-photo=s96-c"
    )
    assert profile["subscription_tier"] == "Gemini Advanced"
    assert profile["valid"] is True

    # 2. Fallback to account widget RPC when WIZ_global_data lacks email/avatar
    mock_resp_app_empty = MagicMock(spec=httpx.Response)
    mock_resp_app_empty.status_code = 200
    mock_resp_app_empty.text = (
        '<html><script>window.WIZ_global_data = {"SNlM0e":"token"};</script></html>'
    )

    mock_resp_widget = MagicMock(spec=httpx.Response)
    mock_resp_widget.status_code = 200
    mock_resp_widget.text = (
        '<div data-identifier="987654321">'
        "<span>fallback_user@gmail.com</span>"
        '<img src="https://lh3.googleusercontent.com/widget_avatar.png">'
        "</div>"
    )

    async def mock_get_fallback(url, *args, **kwargs):
        url_str = str(url)
        if "widget/account" in url_str:
            return mock_resp_widget
        if "retrieveUserQuota" in url_str:
            mock_401 = MagicMock(spec=httpx.Response)
            mock_401.status_code = 401
            return mock_401
        return mock_resp_app_empty

    adapter._profile_fetched_at = 0.0
    adapter.user_email = None
    adapter.account_id = None
    adapter.avatar_url = None
    adapter.subscription_tier = None
    mock_client.get = AsyncMock(side_effect=mock_get_fallback)

    profile_fb = await adapter.fetch_user_profile(force_refresh=True)
    assert profile_fb["user_email"] == "fallback_user@gmail.com"
    assert profile_fb["account_id"] == "987654321"
    assert (
        profile_fb["avatar_url"]
        == "https://lh3.googleusercontent.com/widget_avatar.png"
    )
    assert profile_fb["subscription_tier"] == "Standard"
    assert profile_fb["valid"] is True


@pytest.mark.asyncio
async def test_backend_reset_and_isolation(mock_credentials_file):
    """Verify reset endpoint clears target backend credentials in memory and disk without affecting other backends."""
    transport = httpx.ASGITransport(app=app)

    # Populate initial config with all 3 backends
    client.antigravity.auth.user_email = "tester@example.com"
    client.gemini_api.api_key = "AIzaSyApiResetTestKey"
    client.gemini_api.enabled = True
    client.gemini_api.plan_tier = "Pay-As-You-Go"
    client.gemini_api.is_valid_key = True

    client.gemini_web.psid = "web_psid_reset_test"
    client.gemini_web.psidts = "web_psidts_reset_test"
    client.gemini_web.sapisid = "web_sapisid_reset_test"
    client.gemini_web.enabled = True
    client.gemini_web.user_email = "web_reset@gmail.com"
    client.gemini_web.account_id = "1122334455"
    client.gemini_web.subscription_tier = "Gemini Advanced"

    client.save_config()

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        # 1. Reset gemini_api
        res_reset_api = await ac.post("/api/backends/gemini_api/reset")
        assert res_reset_api.status_code == 200
        data_api = res_reset_api.json()
        assert data_api["status"] == "reset"
        assert data_api["backend"] == "gemini_api"

        # Verify gemini_api memory state wiped
        assert client.gemini_api.api_key == ""
        assert client.gemini_api.enabled is False
        assert client.gemini_api.plan_tier == "Unknown"
        assert client.gemini_api.is_valid_key is None

        # Verify other backends untouched
        assert client.gemini_web.psid == "web_psid_reset_test"
        assert client.gemini_web.enabled is True
        assert client.gemini_web.user_email == "web_reset@gmail.com"
        assert client.antigravity.auth.user_email == "tester@example.com"

        # Verify credentials.json persistence
        with open(mock_credentials_file) as f:
            persisted1 = json.load(f)
        assert "gemini_api_key" not in persisted1
        assert persisted1.get("gemini_web_psid") == "web_psid_reset_test"
        assert persisted1.get("access_token") == "ya29.test_oauth_access_token"

        # 2. Reset gemini_web via /api/providers/gemini_web/reset alias
        res_reset_web = await ac.post("/api/providers/gemini_web/reset")
        assert res_reset_web.status_code == 200
        data_web = res_reset_web.json()
        assert data_web["status"] == "reset"
        assert data_web["backend"] == "gemini_web"

        # Verify gemini_web memory state wiped
        assert client.gemini_web.psid == ""
        assert client.gemini_web.enabled is False
        assert client.gemini_web.user_email is None
        assert client.gemini_web.account_id is None
        assert client.gemini_web.avatar_url is None

        # Verify Antigravity OAuth tokens still preserved
        assert client.antigravity.auth.user_email == "tester@example.com"
        with open(mock_credentials_file) as f:
            persisted2 = json.load(f)
        assert "gemini_web_psid" not in persisted2
        assert persisted2.get("access_token") == "ya29.test_oauth_access_token"
        assert persisted2.get("user_email") == "tester@example.com"

        # 3. Invalid backend reset returns 404
        res_404 = await ac.post("/api/backends/nonexistent_backend/reset")
        assert res_404.status_code == 404


@pytest.mark.asyncio
async def test_backend_status_rich_metadata_endpoint():
    """Verify /api/backends and /api/providers return rich metadata for all backends."""
    transport = httpx.ASGITransport(app=app)

    # Set mock rich metadata
    client.gemini_api.api_key = "AIzaSySecretApiKey123"
    client.gemini_api.plan_tier = "Pay-As-You-Go"
    client.gemini_api.is_valid_key = True

    client.gemini_web.psid = "psid_secret_cookie_456"
    client.gemini_web.user_email = "rich_user@gmail.com"
    client.gemini_web.account_id = "1092837465"
    client.gemini_web.avatar_url = "https://lh3.googleusercontent.com/avatar.jpg"
    client.gemini_web.subscription_tier = "Gemini Advanced"
    client.gemini_web.is_valid_session = True

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        for endpoint in ("/api/backends", "/api/providers"):
            res = await ac.get(endpoint)
            assert res.status_code == 200
            data = res.json()
            assert "backends" in data
            backends = data["backends"]

            # gemini_api rich metadata
            api_info = backends["gemini_api"]
            assert api_info["plan_tier"] == "Pay-As-You-Go"
            assert api_info["valid"] is True
            assert api_info["validity_status"] == "Valid"
            assert api_info["has_api_key"] is True
            assert "AIzaSy" in api_info["masked_key"]
            assert "123" in api_info["masked_key"]

            # gemini_web rich metadata
            web_info = backends["gemini_web"]
            assert web_info["user_email"] == "rich_user@gmail.com"
            assert web_info["account_id"] == "1092837465"
            assert (
                web_info["avatar_url"] == "https://lh3.googleusercontent.com/avatar.jpg"
            )
            assert web_info["subscription_tier"] == "Gemini Advanced"
            assert web_info["valid"] is True
            assert web_info["has_psid"] is True
            assert "psid_s" in web_info["masked_psid"]
