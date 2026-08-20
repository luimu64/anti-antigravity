import pytest
import httpx
from unittest.mock import AsyncMock, patch
from main import app
from app.keys import api_key_manager
from app.client import client


@pytest.mark.asyncio
async def test_openai_error_structure_400():
    transport = httpx.ASGITransport(app=app)
    key = api_key_manager.get_first_active_key() or "test-key"
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        # Invalid body: missing required model and messages
        resp = await ac.post(
            "/v1/chat/completions", headers={"Authorization": f"Bearer {key}"}, json={}
        )
        assert resp.status_code == 400
        data = resp.json()
        assert "error" in data
        err = data["error"]
        assert "message" in err
        assert err["type"] == "invalid_request_error"
        assert err["code"] == "invalid_request_error"


@pytest.mark.asyncio
async def test_openai_error_structure_401_missing_and_invalid():
    transport = httpx.ASGITransport(app=app)
    api_key_manager.enforce_keys = True
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        # Missing auth header
        resp_missing = await ac.get("/v1/models")
        assert resp_missing.status_code == 401
        data_missing = resp_missing.json()
        assert "error" in data_missing
        assert data_missing["error"]["code"] == "missing_api_key"
        assert data_missing["error"]["type"] == "invalid_request_error"

        # Invalid auth header
        resp_invalid = await ac.get(
            "/v1/models", headers={"Authorization": "Bearer invalid-token-12345"}
        )
        assert resp_invalid.status_code == 401
        data_invalid = resp_invalid.json()
        assert "error" in data_invalid
        assert data_invalid["error"]["code"] == "invalid_api_key"
        assert data_invalid["error"]["type"] == "invalid_request_error"


@pytest.mark.asyncio
async def test_openai_error_structure_404():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/v1/unknown_endpoint")
        assert resp.status_code == 404
        data = resp.json()
        assert "error" in data
        assert data["error"]["code"] == "404"
        assert data["error"]["type"] == "invalid_request_error"


@pytest.mark.asyncio
async def test_openai_error_structure_500():
    transport = httpx.ASGITransport(app=app)
    key = api_key_manager.get_first_active_key() or "test-key"
    with patch.object(
        client,
        "generate_content",
        new_callable=AsyncMock,
        side_effect=Exception("Fatal internal error"),
    ):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "test"}],
                },
            )
            assert resp.status_code == 500
            data = resp.json()
            assert "error" in data
            assert data["error"]["code"] == "500"
            assert data["error"]["type"] == "api_error"


@pytest.mark.asyncio
async def test_all_v1_endpoints_key_enforcement():
    """
    Verify all /v1/* endpoints enforce keys when enabled and allow access when disabled.
    """
    transport = httpx.ASGITransport(app=app)
    valid_key = api_key_manager.get_first_active_key() or "test-key"

    mock_models_resp = {"models": {"gemini-3.7-flash-high": {}}}
    mock_chat_resp = {"responseId": "r1", "text": "hello", "finishReason": "STOP"}
    mock_embed_resp = {"embeddings": [{"values": [0.1, 0.2]}]}

    endpoints = [
        ("GET", "/v1/models", None),
        ("GET", "/v1/models/gpt-4o", None),
        (
            "POST",
            "/v1/chat/completions",
            {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
        ),
        ("POST", "/v1/completions", {"model": "gemini-3.7-flash-high", "prompt": "hi"}),
        ("POST", "/v1/embeddings", {"model": "text-embedding-004", "input": "hi"}),
    ]

    with (
        patch.object(
            client,
            "fetch_available_models",
            new_callable=AsyncMock,
            return_value=mock_models_resp,
        ),
        patch.object(
            client,
            "generate_content",
            new_callable=AsyncMock,
            return_value=mock_chat_resp,
        ),
        patch.object(
            client,
            "embed_contents",
            new_callable=AsyncMock,
            return_value=mock_embed_resp,
        ),
    ):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            # 1. Enforcement ENABLED: missing or bad keys must get 401
            api_key_manager.enforce_keys = True
            for method, path, body in endpoints:
                # No header
                r_none = await ac.request(method, path, json=body)
                assert r_none.status_code == 401, (
                    f"{path} should return 401 without auth"
                )
                assert r_none.json()["error"]["code"] == "missing_api_key"

                # Bad header
                r_bad = await ac.request(
                    method,
                    path,
                    headers={"Authorization": "Bearer sk-invalid"},
                    json=body,
                )
                assert r_bad.status_code == 401, (
                    f"{path} should return 401 with invalid auth"
                )
                assert r_bad.json()["error"]["code"] == "invalid_api_key"

                # Valid header
                r_ok = await ac.request(
                    method,
                    path,
                    headers={"Authorization": f"Bearer {valid_key}"},
                    json=body,
                )
                assert r_ok.status_code == 200, (
                    f"{path} should return 200 with valid key"
                )

            # 2. Enforcement DISABLED: all endpoints should pass without header
            api_key_manager.enforce_keys = False
            for method, path, body in endpoints:
                r_open = await ac.request(method, path, json=body)
                assert r_open.status_code == 200, (
                    f"{path} should return 200 when enforcement is disabled"
                )
