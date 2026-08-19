import base64
import struct
import pytest
import httpx
from unittest.mock import AsyncMock, patch
from main import app
from app.client import client

@pytest.mark.asyncio
async def test_health_endpoint():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "agy-to-api"

@pytest.mark.asyncio
async def test_auth_status_endpoint():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/auth/status")
        assert response.status_code == 200
        data = response.json()
        assert "authenticated" in data
        assert "project_id" in data

from app.keys import api_key_manager

@pytest.mark.asyncio
async def test_models_endpoint():
    transport = httpx.ASGITransport(app=app)
    key = api_key_manager.get_first_active_key() or "test-key"
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/v1/models", headers={"Authorization": f"Bearer {key}"})
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) > 0
        model_ids = [m["id"] for m in data["data"]]
        assert "gemini-3.7-flash-high" in model_ids or "gpt-4o" in model_ids

@pytest.mark.asyncio
async def test_retrieve_model_endpoint():
    transport = httpx.ASGITransport(app=app)
    key = api_key_manager.get_first_active_key() or "test-key"
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/v1/models/gpt-4o", headers={"Authorization": f"Bearer {key}"})
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "gpt-4o"
        assert data["root"] == "gemini-3.7-flash-high"

@pytest.mark.asyncio
async def test_dashboard_endpoint():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/")
        assert response.status_code == 200
        assert "Antigravity API" in response.text

@pytest.mark.asyncio
async def test_embeddings_endpoint_float():
    transport = httpx.ASGITransport(app=app)
    key = api_key_manager.get_first_active_key() or "test-key"
    mock_resp = {
        "embeddings": [
            {"values": [0.1, 0.2, 0.3, 0.4]},
            {"values": [0.5, 0.6, 0.7, 0.8]}
        ],
        "usageMetadata": {
            "promptTokenCount": 16
        }
    }
    with patch.object(client, "embed_contents", new_callable=AsyncMock, return_value=mock_resp):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.post(
                "/v1/embeddings",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": "text-embedding-3-small",
                    "input": ["hello world", "second phrase"],
                    "encoding_format": "float"
                }
            )
            assert response.status_code == 200
            data = response.json()
            assert data["object"] == "list"
            assert len(data["data"]) == 2
            assert data["data"][0]["index"] == 0
            assert data["data"][0]["embedding"] == [0.1, 0.2, 0.3, 0.4]
            assert data["data"][1]["index"] == 1
            assert data["data"][1]["embedding"] == [0.5, 0.6, 0.7, 0.8]
            assert data["usage"]["prompt_tokens"] == 16
            assert data["usage"]["total_tokens"] == 16

@pytest.mark.asyncio
async def test_embeddings_endpoint_base64_and_dimensions():
    transport = httpx.ASGITransport(app=app)
    key = api_key_manager.get_first_active_key() or "test-key"
    mock_resp = {
        "embeddings": [
            {"values": [0.125, 0.25, 0.5, 1.0, 2.0]}
        ]
    }
    with patch.object(client, "embed_contents", new_callable=AsyncMock, return_value=mock_resp):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.post(
                "/v1/embeddings",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": "text-embedding-004",
                    "input": "test base64 embedding",
                    "encoding_format": "base64",
                    "dimensions": 3
                }
            )
            assert response.status_code == 200
            data = response.json()
            assert len(data["data"]) == 1
            b64_str = data["data"][0]["embedding"]
            assert isinstance(b64_str, str)
            raw_bytes = base64.b64decode(b64_str)
            unpacked = struct.unpack("<3f", raw_bytes)
            assert pytest.approx(unpacked[0], 0.001) == 0.125
            assert pytest.approx(unpacked[1], 0.001) == 0.25
            assert pytest.approx(unpacked[2], 0.001) == 0.5

@pytest.mark.asyncio
async def test_embeddings_endpoint_invalid_inputs():
    transport = httpx.ASGITransport(app=app)
    key = api_key_manager.get_first_active_key() or "test-key"
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        # Invalid dimensions <= 0
        resp = await ac.post(
            "/v1/embeddings",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": "text-embedding-004", "input": "test", "dimensions": 0}
        )
        assert resp.status_code == 400

        # Invalid encoding_format
        resp2 = await ac.post(
            "/v1/embeddings",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": "text-embedding-004", "input": "test", "encoding_format": "xml"}
        )
        assert resp2.status_code == 400

