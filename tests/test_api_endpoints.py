import base64
import struct
import pytest
import httpx
from unittest.mock import AsyncMock, patch
from main import app
from app.client import client
from app.keys import api_key_manager

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

@pytest.mark.asyncio
async def test_legacy_completions_endpoint_non_streaming():
    transport = httpx.ASGITransport(app=app)
    key = api_key_manager.get_first_active_key() or "test-key"
    mock_resp = {
        "responseId": "cmpl-test-abc",
        "text": "This is a completed text response.",
        "finishReason": "STOP",
        "usageMetadata": {
            "promptTokenCount": 5,
            "candidatesTokenCount": 7,
            "totalTokenCount": 12
        }
    }
    with patch.object(client, "generate_content", new_callable=AsyncMock, return_value=mock_resp):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/v1/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": "gpt-3.5-turbo-instruct",
                    "prompt": "Say this is a test",
                    "max_tokens": 10
                }
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["object"] == "text_completion"
            assert data["id"] == "cmpl-test-abc"
            assert len(data["choices"]) == 1
            assert data["choices"][0]["text"] == "This is a completed text response."
            assert data["choices"][0]["logprobs"] is None
            assert data["choices"][0]["finish_reason"] == "stop"
            assert data["usage"]["total_tokens"] == 12

@pytest.mark.asyncio
async def test_legacy_completions_endpoint_streaming():
    transport = httpx.ASGITransport(app=app)
    key = api_key_manager.get_first_active_key() or "test-key"

    async def mock_stream(*args, **kwargs):
        yield {
            "response": {
                "candidates": [{"content": {"role": "model", "parts": [{"text": "Streamed text"}]}}],
                "responseId": "cmpl-stream-001"
            }
        }
        yield {
            "response": {
                "candidates": [{"content": {"role": "model", "parts": []}, "finishReason": "STOP"}],
                "responseId": "cmpl-stream-001"
            }
        }

    with patch.object(client, "stream_generate_content", side_effect=mock_stream):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/v1/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": "gemini-3.7-flash-high",
                    "prompt": "Stream this test",
                    "stream": True
                }
            )
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers.get("content-type", "")
            lines = [line.strip() for line in resp.text.split("\n") if line.startswith("data: ")]
            assert len(lines) >= 2
            assert lines[-1] == "data: [DONE]"

@pytest.mark.asyncio
async def test_standardized_error_envelope():
    transport = httpx.ASGITransport(app=app)
    key = api_key_manager.get_first_active_key() or "test-key"
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        # 1. 404 Not Found
        resp_404 = await ac.get("/nonexistent/endpoint")
        assert resp_404.status_code == 404
        data_404 = resp_404.json()
        assert "error" in data_404
        assert "message" in data_404["error"]
        assert "type" in data_404["error"]

        # 2. 400 Bad Request / Validation
        resp_400 = await ac.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": "gpt-4o"} # Missing messages
        )
        assert resp_400.status_code == 400
        data_400 = resp_400.json()
        assert "error" in data_400
        assert data_400["error"]["type"] == "invalid_request_error"
        assert "messages" in data_400["error"]["message"]


