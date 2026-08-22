import base64
import json
import struct
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from app.client import client
from app.keys import api_key_manager
from main import app


@pytest.mark.asyncio
async def test_health_endpoint():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "google-gate"
        assert data["version"] == "1.0.0"
        assert "authenticated" in data
        assert "project_id" in data
        assert "api_key_enforcement" in data


@pytest.mark.asyncio
async def test_auth_status_endpoint():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/auth/status")
        assert response.status_code == 200
        data = response.json()
        assert "authenticated" in data
        assert "project_id" in data
        assert "tier_name" in data
        assert "has_refresh_token" in data


@pytest.mark.asyncio
async def test_models_endpoint():
    transport = httpx.ASGITransport(app=app)
    key = api_key_manager.get_first_active_key() or "test-key"
    mock_models = {
        "models": {
            "gemini-3.7-flash-high": {
                "displayName": "Gemini 3.7 Flash High",
                "maxTokens": 1048576,
                "supportsThinking": True,
            }
        }
    }
    with patch.object(
        client,
        "fetch_available_models",
        new_callable=AsyncMock,
        return_value=mock_models,
    ):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            # Test /v1/models
            response = await ac.get(
                "/v1/models", headers={"Authorization": f"Bearer {key}"}
            )
            assert response.status_code == 200
            data = response.json()
            assert data["object"] == "list"
            assert len(data["data"]) > 0
            model_ids = [m["id"] for m in data["data"]]
            assert "gemini-3.7-flash-high" in model_ids
            assert "gpt-4o" in model_ids
            assert "vision" in model_ids
            vision_entry = next(m for m in data["data"] if m["id"] == "vision")
            assert vision_entry["supports_vision"] is True

            # Test /models (unprefixed alias)
            response_unpref = await ac.get(
                "/models", headers={"Authorization": f"Bearer {key}"}
            )
            assert response_unpref.status_code == 200
            assert response_unpref.json()["object"] == "list"


@pytest.mark.asyncio
async def test_models_endpoint_backend_failure_fallback():
    transport = httpx.ASGITransport(app=app)
    key = api_key_manager.get_first_active_key() or "test-key"
    with patch.object(
        client,
        "fetch_available_models",
        new_callable=AsyncMock,
        side_effect=RuntimeError("Backend down"),
    ):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.get(
                "/v1/models", headers={"Authorization": f"Bearer {key}"}
            )
            assert response.status_code == 200
            data = response.json()
            assert data["object"] == "list"
            model_ids = [m["id"] for m in data["data"]]
            assert "gpt-4o" in model_ids


@pytest.mark.asyncio
async def test_retrieve_model_endpoint():
    transport = httpx.ASGITransport(app=app)
    key = api_key_manager.get_first_active_key() or "test-key"
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        # Test /v1/models/{model_id}
        response = await ac.get(
            "/v1/models/gpt-4o", headers={"Authorization": f"Bearer {key}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "gpt-4o"
        assert data["object"] == "model"
        assert data["owned_by"] == "google"
        assert data["root"] == "gemini-3.7-flash-high"

        # Test unprefixed /models/{model_id}
        response_unpref = await ac.get(
            "/models/claude-3-7-sonnet", headers={"Authorization": f"Bearer {key}"}
        )
        assert response_unpref.status_code == 200
        assert response_unpref.json()["id"] == "claude-3-7-sonnet"
        assert response_unpref.json()["root"] == "claude-sonnet-4-6"


@pytest.mark.asyncio
async def test_chat_completions_non_streaming():
    transport = httpx.ASGITransport(app=app)
    key = api_key_manager.get_first_active_key() or "test-key"
    mock_resp = {
        "responseId": "chatcmpl-test-001",
        "text": "Hello! How can I assist you today?",
        "thoughts": "User greeted, answering politely.",
        "finishReason": "STOP",
        "usageMetadata": {
            "promptTokenCount": 12,
            "candidatesTokenCount": 8,
            "totalTokenCount": 20,
        },
    }
    with patch.object(
        client, "generate_content", new_callable=AsyncMock, return_value=mock_resp
    ):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            # /v1/chat/completions
            response = await ac.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": "gpt-4o",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Hi"},
                    ],
                    "temperature": 0.7,
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert data["id"] == "chatcmpl-test-001"
            assert data["object"] == "chat.completion"
            assert data["model"] == "gpt-4o"
            assert len(data["choices"]) == 1
            assert data["choices"][0]["index"] == 0
            assert data["choices"][0]["message"]["role"] == "assistant"
            assert (
                data["choices"][0]["message"]["content"]
                == "Hello! How can I assist you today?"
            )
            assert (
                data["choices"][0]["message"]["reasoning_content"]
                == "User greeted, answering politely."
            )
            assert data["choices"][0]["finish_reason"] == "stop"
            assert data["usage"]["prompt_tokens"] == 12
            assert data["usage"]["completion_tokens"] == 8
            assert data["usage"]["total_tokens"] == 20

            # /chat/completions (unprefixed)
            resp_unpref = await ac.post(
                "/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": "gemini-3.7-flash-high",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )
            assert resp_unpref.status_code == 200
            assert resp_unpref.json()["object"] == "chat.completion"


@pytest.mark.asyncio
async def test_chat_completions_tool_calling():
    transport = httpx.ASGITransport(app=app)
    key = api_key_manager.get_first_active_key() or "test-key"
    mock_resp = {
        "responseId": "chatcmpl-tools-001",
        "text": "",
        "thoughts": "Calling weather function.",
        "toolCalls": [
            {"name": "get_weather", "args": {"location": "San Francisco, CA"}}
        ],
        "finishReason": "STOP",
        "usageMetadata": {
            "promptTokenCount": 25,
            "candidatesTokenCount": 15,
            "totalTokenCount": 40,
        },
    }
    with patch.object(
        client, "generate_content", new_callable=AsyncMock, return_value=mock_resp
    ):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": "gpt-4o",
                    "messages": [
                        {"role": "user", "content": "What's the weather in SF?"}
                    ],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "description": "Get current weather",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"location": {"type": "string"}},
                                    "required": ["location"],
                                },
                            },
                        }
                    ],
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert data["choices"][0]["finish_reason"] == "tool_calls"
            tool_calls = data["choices"][0]["message"]["tool_calls"]
            assert len(tool_calls) == 1
            assert tool_calls[0]["function"]["name"] == "get_weather"
            args = json.loads(tool_calls[0]["function"]["arguments"])
            assert args["location"] == "San Francisco, CA"


@pytest.mark.asyncio
async def test_chat_completions_streaming():
    transport = httpx.ASGITransport(app=app)
    key = api_key_manager.get_first_active_key() or "test-key"

    async def mock_stream(*args, **kwargs):
        yield {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": "Deep thought", "thought": True}],
                    }
                }
            ],
            "responseId": "chatcmpl-stream-001",
        }
        yield {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": "Final answer", "thought": False}],
                    }
                }
            ],
            "responseId": "chatcmpl-stream-001",
        }
        yield {
            "candidates": [
                {"content": {"role": "model", "parts": []}, "finishReason": "STOP"}
            ],
            "responseId": "chatcmpl-stream-001",
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 6,
                "totalTokenCount": 16,
            },
        }

    with patch.object(client, "stream_generate_content", side_effect=mock_stream):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "Tell me something"}],
                    "stream": True,
                    "stream_options": {"include_usage": True},
                },
            )
            assert response.status_code == 200
            assert "text/event-stream" in response.headers.get("content-type", "")
            lines = [
                line.strip()
                for line in response.text.split("\n")
                if line.startswith("data: ")
            ]
            assert len(lines) >= 3
            assert lines[-1] == "data: [DONE]"

            chunks = [json.loads(line[6:]) for line in lines if line != "data: [DONE]"]
            # Check reasoning chunk
            reasoning_chunks = [
                c
                for c in chunks
                if c.get("choices")
                and c["choices"][0].get("delta", {}).get("reasoning_content")
            ]
            assert len(reasoning_chunks) >= 1
            assert (
                reasoning_chunks[0]["choices"][0]["delta"]["reasoning_content"]
                == "Deep thought"
            )

            # Check content chunk
            content_chunks = [
                c
                for c in chunks
                if c.get("choices") and c["choices"][0].get("delta", {}).get("content")
            ]
            assert len(content_chunks) >= 1
            assert content_chunks[0]["choices"][0]["delta"]["content"] == "Final answer"

            # Check usage chunk at the end
            usage_chunks = [c for c in chunks if "usage" in c and c["choices"] == []]
            assert len(usage_chunks) == 1
            assert usage_chunks[0]["usage"]["total_tokens"] == 16


@pytest.mark.asyncio
async def test_chat_completions_errors():
    transport = httpx.ASGITransport(app=app)
    key = api_key_manager.get_first_active_key() or "test-key"
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        # 1. Validation error / missing messages -> 400
        resp_bad = await ac.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": "gpt-4o"},
        )
        assert resp_bad.status_code == 400
        err_json = resp_bad.json()
        assert "error" in err_json
        assert err_json["error"]["type"] == "invalid_request_error"

    # 2. Upstream 500 failure during non-streaming
    with patch.object(
        client,
        "generate_content",
        new_callable=AsyncMock,
        side_effect=RuntimeError("Google backend crash"),
    ):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp_500 = await ac.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            assert resp_500.status_code == 500
            err_json = resp_500.json()
            assert "error" in err_json
            assert "Generation failed" in err_json["error"]["message"]

    # 3. Upstream 500 failure during streaming
    with patch.object(
        client, "stream_generate_content", side_effect=RuntimeError("Stream failure")
    ):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp_stream_err = await ac.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )
            assert resp_stream_err.status_code == 500
            err_json = resp_stream_err.json()
            assert "error" in err_json
            assert "Generation failed" in err_json["error"]["message"]


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
            "totalTokenCount": 12,
        },
    }
    with patch.object(
        client, "generate_content", new_callable=AsyncMock, return_value=mock_resp
    ):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            # String prompt
            resp = await ac.post(
                "/v1/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": "gpt-3.5-turbo-instruct",
                    "prompt": "Say this is a test",
                    "max_tokens": 10,
                },
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

            # List prompt + unprefixed /completions
            resp_list = await ac.post(
                "/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": "gpt-3.5-turbo-instruct",
                    "prompt": ["Line 1", "Line 2"],
                    "max_tokens": 10,
                },
            )
            assert resp_list.status_code == 200
            assert resp_list.json()["object"] == "text_completion"


@pytest.mark.asyncio
async def test_legacy_completions_endpoint_streaming():
    transport = httpx.ASGITransport(app=app)
    key = api_key_manager.get_first_active_key() or "test-key"

    async def mock_stream(*args, **kwargs):
        yield {
            "response": {
                "candidates": [
                    {"content": {"role": "model", "parts": [{"text": "Streamed text"}]}}
                ],
                "responseId": "cmpl-stream-001",
            }
        }
        yield {
            "response": {
                "candidates": [
                    {"content": {"role": "model", "parts": []}, "finishReason": "STOP"}
                ],
                "responseId": "cmpl-stream-001",
                "usageMetadata": {
                    "promptTokenCount": 5,
                    "candidatesTokenCount": 3,
                    "totalTokenCount": 8,
                },
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
                    "stream": True,
                    "stream_options": {"include_usage": True},
                },
            )
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers.get("content-type", "")
            lines = [
                line.strip()
                for line in resp.text.split("\n")
                if line.startswith("data: ")
            ]
            assert len(lines) >= 2
            assert lines[-1] == "data: [DONE]"


@pytest.mark.asyncio
async def test_legacy_completions_errors():
    transport = httpx.ASGITransport(app=app)
    key = api_key_manager.get_first_active_key() or "test-key"
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        # Invalid non-JSON payload
        resp_invalid_json = await ac.post(
            "/v1/completions",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            content=b"invalid json",
        )
        assert resp_invalid_json.status_code == 400

    # Upstream 500 failure
    with patch.object(
        client,
        "generate_content",
        new_callable=AsyncMock,
        side_effect=RuntimeError("Legacy upstream crash"),
    ):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp_500 = await ac.post(
                "/v1/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={"prompt": "hello"},
            )
            assert resp_500.status_code == 500
            assert "Generation failed" in resp_500.json()["error"]["message"]


@pytest.mark.asyncio
async def test_embeddings_endpoint_float():
    transport = httpx.ASGITransport(app=app)
    key = api_key_manager.get_first_active_key() or "test-key"
    mock_resp = {
        "embeddings": [
            {"values": [0.1, 0.2, 0.3, 0.4]},
            {"values": [0.5, 0.6, 0.7, 0.8]},
        ],
        "usageMetadata": {"promptTokenCount": 16},
    }
    with patch.object(
        client, "embed_contents", new_callable=AsyncMock, return_value=mock_resp
    ):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            # /v1/embeddings
            response = await ac.post(
                "/v1/embeddings",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": "text-embedding-3-small",
                    "input": ["hello world", "second phrase"],
                    "encoding_format": "float",
                },
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

            # /embeddings (unprefixed)
            resp_unpref = await ac.post(
                "/embeddings",
                headers={"Authorization": f"Bearer {key}"},
                json={"model": "text-embedding-3-small", "input": "single phrase"},
            )
            assert resp_unpref.status_code == 200
            assert resp_unpref.json()["object"] == "list"


@pytest.mark.asyncio
async def test_embeddings_endpoint_token_inputs():
    transport = httpx.ASGITransport(app=app)
    key = api_key_manager.get_first_active_key() or "test-key"
    mock_resp = {"embeddings": [{"values": [0.1, 0.2, 0.3]}]}
    with patch.object(
        client, "embed_contents", new_callable=AsyncMock, return_value=mock_resp
    ):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            # Integer token list [101, 2054, 102]
            resp_tokens = await ac.post(
                "/v1/embeddings",
                headers={"Authorization": f"Bearer {key}"},
                json={"model": "text-embedding-004", "input": [101, 2054, 102]},
            )
            assert resp_tokens.status_code == 200
            assert len(resp_tokens.json()["data"]) == 1

            # Nested token list [[101, 2054], [102, 2055]]
            resp_nested = await ac.post(
                "/v1/embeddings",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": "text-embedding-004",
                    "input": [[101, 2054], [102, 2055]],
                },
            )
            assert resp_nested.status_code == 200


@pytest.mark.asyncio
async def test_embeddings_endpoint_base64_and_dimensions():
    transport = httpx.ASGITransport(app=app)
    key = api_key_manager.get_first_active_key() or "test-key"
    mock_resp = {"embeddings": [{"values": [0.125, 0.25, 0.5, 1.0, 2.0]}]}
    with patch.object(
        client, "embed_contents", new_callable=AsyncMock, return_value=mock_resp
    ):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.post(
                "/v1/embeddings",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": "text-embedding-004",
                    "input": "test base64 embedding",
                    "encoding_format": "base64",
                    "dimensions": 3,
                },
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
async def test_embeddings_endpoint_invalid_inputs_and_errors():
    transport = httpx.ASGITransport(app=app)
    key = api_key_manager.get_first_active_key() or "test-key"
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        # Empty input list -> 400
        resp_empty = await ac.post(
            "/v1/embeddings",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": "text-embedding-004", "input": []},
        )
        assert resp_empty.status_code == 400

        # Invalid dimensions <= 0 -> 400
        resp_dim = await ac.post(
            "/v1/embeddings",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": "text-embedding-004", "input": "test", "dimensions": 0},
        )
        assert resp_dim.status_code == 400

        # Invalid encoding_format -> 400
        resp_enc = await ac.post(
            "/v1/embeddings",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": "text-embedding-004",
                "input": "test",
                "encoding_format": "xml",
            },
        )
        assert resp_enc.status_code == 400

    # Upstream 500 error
    with patch.object(
        client,
        "embed_contents",
        new_callable=AsyncMock,
        side_effect=RuntimeError("Embedding upstream fail"),
    ):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp_500 = await ac.post(
                "/v1/embeddings",
                headers={"Authorization": f"Bearer {key}"},
                json={"model": "text-embedding-004", "input": "test"},
            )
            assert resp_500.status_code == 500
            assert "Embedding failed" in resp_500.json()["error"]["message"]


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
        assert data_404["error"]["type"] == "invalid_request_error"
        assert data_404["error"]["code"] == "404"

        # 2. 400 Bad Request / Validation (missing required fields)
        resp_400 = await ac.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": "gpt-4o"},  # Missing messages
        )
        assert resp_400.status_code == 400
        data_400 = resp_400.json()
        assert "error" in data_400
        assert data_400["error"]["type"] == "invalid_request_error"
        assert "messages" in data_400["error"]["message"]
