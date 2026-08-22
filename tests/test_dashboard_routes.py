import os
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from app.client import client
from app.keys import api_key_manager
from main import app


@pytest.mark.asyncio
async def test_dashboard_page():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/")
        assert resp.status_code == 200
        assert "Google Gate" in resp.text

        # Test with HOST environment variable set
        with patch.dict(os.environ, {"HOST": "192.168.1.100"}):
            resp_env = await ac.get("/")
            assert resp_env.status_code == 200
            assert "192.168.1.100" in resp_env.text


@pytest.mark.asyncio
async def test_dashboard_health():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["service"] == "google-gate"
        assert data["version"] == "1.0.0"
        assert "authenticated" in data
        assert "project_id" in data
        assert "api_key_enforcement" in data


@pytest.mark.asyncio
async def test_api_keys_management():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        # 1. List keys
        resp_list = await ac.get("/api/keys")
        assert resp_list.status_code == 200
        list_data = resp_list.json()
        assert "enforce_keys" in list_data
        assert isinstance(list_data["keys"], list)

        # 2. Create key
        resp_create = await ac.post("/api/keys", json={"name": "Dashboard Test Key"})
        assert resp_create.status_code == 200
        create_data = resp_create.json()
        assert create_data["status"] == "created"
        created_key = create_data["key"]
        assert created_key["name"] == "Dashboard Test Key"
        assert created_key["key"].startswith("sk-gate-")
        key_id = created_key["id"]

        # 3. Revoke existing key
        resp_revoke = await ac.delete(f"/api/keys/{key_id}")
        assert resp_revoke.status_code == 200
        assert resp_revoke.json()["status"] == "revoked"

        # 4. Revoke non-existent key -> 404
        resp_revoke_404 = await ac.delete("/api/keys/non_existent_key_id")
        assert resp_revoke_404.status_code == 404
        assert resp_revoke_404.json()["error"] == "Key not found"


@pytest.mark.asyncio
async def test_api_keys_enforcement_toggle():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        # Toggle enforcement off
        resp_off = await ac.post("/api/keys/enforcement", json={"enforce": False})
        assert resp_off.status_code == 200
        assert resp_off.json()["enforce_keys"] is False
        assert api_key_manager.enforce_keys is False

        # Toggle enforcement on
        resp_on = await ac.post("/api/keys/enforcement", json={"enforce": True})
        assert resp_on.status_code == 200
        assert resp_on.json()["enforce_keys"] is True
        assert api_key_manager.enforce_keys is True


@pytest.mark.asyncio
async def test_quotas_endpoint():
    transport = httpx.ASGITransport(app=app)
    mock_upstream_quotas = {
        "groups": [
            {
                "groupId": "antigravity_general",
                "buckets": [
                    {
                        "bucketId": "weekly",
                        "displayName": "Weekly Limit",
                        "remainingFraction": 0.85,
                        "resetTime": "2030-01-01T00:00:00Z",
                    },
                    {
                        "bucketId": "5hr",
                        "displayName": "5-Hour Burst Limit",
                        "remainingFraction": 0.98,
                        "resetTime": "2030-01-01T05:00:00Z",
                    },
                ],
            }
        ]
    }
    with patch.object(
        client,
        "retrieve_user_quota_summary",
        new_callable=AsyncMock,
        return_value=mock_upstream_quotas,
    ):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/api/quotas")
            assert resp.status_code == 200
            data = resp.json()
            assert "groups" in data
            groups = data["groups"]
            assert len(groups) == 2

            # Bucket 1
            assert groups[0]["display_name"] == "Weekly Limit"
            assert pytest.approx(groups[0]["fraction_used"], 0.001) == 0.15
            assert pytest.approx(groups[0]["remaining_fraction"], 0.001) == 0.85
            assert groups[0]["reset_time_seconds"] > 0
            assert groups[0]["model_id"] == "antigravity_general"

            # Bucket 2
            assert groups[1]["display_name"] == "5-Hour Burst Limit"
            assert pytest.approx(groups[1]["fraction_used"], 0.001) == 0.02
            assert pytest.approx(groups[1]["remaining_fraction"], 0.001) == 0.98
            assert groups[1]["reset_time_seconds"] > 0
            assert groups[1]["model_id"] == "antigravity_general"

    # Quota endpoint failure -> 500 JSON
    with patch.object(
        client,
        "retrieve_user_quota_summary",
        new_callable=AsyncMock,
        side_effect=RuntimeError("Quota service down"),
    ):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp_fail = await ac.get("/api/quotas")
            assert resp_fail.status_code == 500
            data = resp_fail.json()
            assert "error" in data
            assert data["groups"] == []


@pytest.mark.asyncio
async def test_dashboard_models_transformed():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/api/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "gemini_models" in data
        assert "the_rest" in data
        assert "models" in data
        assert isinstance(data["gemini_models"], list)
        assert isinstance(data["the_rest"], list)
        for item in data["models"]:
            assert "base_model_name" in item
            assert "selectable_model_ids" in item
            assert "available_sources" in item


@pytest.mark.asyncio
async def test_dashboard_renders_query_component():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/")
        assert resp.status_code == 200
        text = resp.text.lower()
        assert "query-card" in resp.text
        assert "query playground" in text or "query" in text
        assert "/v1/chat/completions" in resp.text
        assert "query-prompt-input" in resp.text
        assert "query-model-select" in resp.text
        assert "query-submit-btn" in resp.text
        assert "query-response-output" in resp.text

