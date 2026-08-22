import threading
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from app.client import client
from app.history import QueryHistoryManager, history_manager
from app.keys import api_key_manager
from main import app


def test_query_history_manager_basic_crud():
    mgr = QueryHistoryManager(max_records=5)
    assert len(mgr) == 0
    assert mgr.list_history() == []

    # Record 1
    e1 = mgr.record(
        model="gpt-4o",
        resolved_model="gemini-2.5-flash",
        backend="antigravity",
        duration_ms=150.2,
        status="success",
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        request_id="req_1",
    )
    assert len(mgr) == 1
    assert e1["id"] == "req_1"
    assert e1["status"] == "success"
    assert e1["model"] == "gpt-4o"
    assert e1["resolved_model"] == "gemini-2.5-flash"
    assert e1["backend"] == "antigravity"
    assert e1["duration_ms"] == 150.2
    assert e1["prompt_tokens"] == 10
    assert e1["completion_tokens"] == 20
    assert e1["total_tokens"] == 30
    assert e1["error_message"] is None

    # Record error
    e2 = mgr.record(
        model="claude-3-7-sonnet",
        resolved_model="claude-sonnet-4-6",
        backend="gemini_api",
        duration_ms=55.0,
        status="error",
        error_message="Connection timeout",
        request_id="req_2",
    )
    assert len(mgr) == 2
    assert e2["status"] == "error"
    assert e2["error_message"] == "Connection timeout"
    assert e2["prompt_tokens"] == 0

    # Newest is first (reverse chronological)
    history = mgr.list_history()
    assert len(history) == 2
    assert history[0]["id"] == "req_2"
    assert history[1]["id"] == "req_1"

    # Test rolling buffer capacity limit (max 5)
    for i in range(3, 10):
        mgr.record(
            model="gpt-4o",
            resolved_model="gemini-2.5-flash",
            backend="antigravity",
            duration_ms=10.0,
            request_id=f"req_{i}",
        )

    assert len(mgr) == 5
    ids = [entry["id"] for entry in mgr.list_history()]
    assert ids == ["req_9", "req_8", "req_7", "req_6", "req_5"]

    # Clear
    mgr.clear()
    assert len(mgr) == 0
    assert mgr.list_history() == []


def test_query_history_manager_thread_safety():
    mgr = QueryHistoryManager(max_records=50)

    def worker(worker_id: int):
        for i in range(20):
            mgr.record(
                model=f"model_{worker_id}",
                resolved_model="resolved",
                backend="test_backend",
                duration_ms=float(i),
                request_id=f"req_{worker_id}_{i}",
            )

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Total buffer should cap at max 50
    assert len(mgr) == 50
    history = mgr.list_history()
    assert len(history) == 50


@pytest.mark.asyncio
async def test_dashboard_history_endpoints():
    history_manager.clear()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        # 1. Initially empty
        resp = await ac.get("/api/history")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["history"] == []
        assert data["total"] == 0

        # 2. Add an entry directly
        history_manager.record(
            model="gpt-4o",
            resolved_model="gemini-2.5-flash",
            backend="antigravity",
            duration_ms=120.5,
            status="success",
            prompt_tokens=15,
            completion_tokens=25,
            total_tokens=40,
        )

        resp2 = await ac.get("/api/history")
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert data2["total"] == 1
        assert len(data2["history"]) == 1
        assert data2["history"][0]["model"] == "gpt-4o"
        assert data2["history"][0]["total_tokens"] == 40

        # 3. Delete history
        resp_del = await ac.delete("/api/history")
        assert resp_del.status_code == 200
        assert resp_del.json()["status"] == "cleared"

        # 4. Verify cleared
        resp3 = await ac.get("/api/history")
        assert resp3.status_code == 200
        assert resp3.json()["total"] == 0
        assert resp3.json()["history"] == []


@pytest.mark.asyncio
async def test_chat_completions_records_history_non_streaming():
    history_manager.clear()
    transport = httpx.ASGITransport(app=app)
    key = api_key_manager.get_first_active_key() or "test-key"

    mock_resp = {
        "responseId": "chatcmpl-hist-001",
        "text": "History test response",
        "finishReason": "STOP",
        "usageMetadata": {
            "promptTokenCount": 14,
            "candidatesTokenCount": 6,
            "totalTokenCount": 20,
        },
    }

    with patch.object(
        client, "generate_content", new_callable=AsyncMock, return_value=mock_resp
    ):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )
            assert resp.status_code == 200

    history = history_manager.list_history()
    assert len(history) == 1
    entry = history[0]
    assert entry["status"] == "success"
    assert entry["model"] == "gpt-4o"
    assert entry["resolved_model"] == "gemini-3.7-flash-high"
    assert entry["prompt_tokens"] == 14
    assert entry["completion_tokens"] == 6
    assert entry["total_tokens"] == 20
    assert entry["duration_ms"] >= 0


@pytest.mark.asyncio
async def test_chat_completions_records_history_streaming():
    history_manager.clear()
    transport = httpx.ASGITransport(app=app)
    key = api_key_manager.get_first_active_key() or "test-key"

    async def mock_stream(*args, **kwargs):
        yield {
            "response": {
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": "Streamed text response"}],
                            "role": "model",
                        }
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 22,
                    "candidatesTokenCount": 11,
                    "totalTokenCount": 33,
                },
            }
        }

    with patch.object(client, "stream_generate_content", side_effect=mock_stream):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": "gemini-3.7-flash-high",
                    "messages": [{"role": "user", "content": "Stream query"}],
                    "stream": True,
                },
            )
            assert resp.status_code == 200
            # Read full body stream
            content = resp.text
            assert "Streamed text response" in content

    history = history_manager.list_history()
    assert len(history) == 1
    entry = history[0]
    assert entry["status"] == "success"
    assert entry["model"] == "gemini-3.7-flash-high"
    assert entry["prompt_tokens"] == 22
    assert entry["completion_tokens"] == 11
    assert entry["total_tokens"] == 33


@pytest.mark.asyncio
async def test_legacy_completions_records_history():
    history_manager.clear()
    transport = httpx.ASGITransport(app=app)
    key = api_key_manager.get_first_active_key() or "test-key"

    mock_resp = {
        "responseId": "cmpl-hist-001",
        "text": "Text completion answer",
        "finishReason": "STOP",
        "usageMetadata": {
            "promptTokenCount": 8,
            "candidatesTokenCount": 5,
            "totalTokenCount": 13,
        },
    }

    with patch.object(
        client, "generate_content", new_callable=AsyncMock, return_value=mock_resp
    ):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/v1/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": "gpt-4o",
                    "prompt": "Say hello",
                },
            )
            assert resp.status_code == 200

    history = history_manager.list_history()
    assert len(history) == 1
    entry = history[0]
    assert entry["status"] == "success"
    assert entry["model"] == "gpt-4o"
    assert entry["prompt_tokens"] == 8
    assert entry["completion_tokens"] == 5
    assert entry["total_tokens"] == 13


@pytest.mark.asyncio
async def test_error_records_in_history():
    history_manager.clear()
    transport = httpx.ASGITransport(app=app)
    key = api_key_manager.get_first_active_key() or "test-key"

    # Upstream exception
    with patch.object(
        client,
        "generate_content",
        new_callable=AsyncMock,
        side_effect=RuntimeError("Google backend error"),
    ):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )
            assert resp.status_code == 500

    history = history_manager.list_history()
    assert len(history) == 1
    entry = history[0]
    assert entry["status"] == "error"
    assert entry["model"] == "gpt-4o"
    assert entry["total_tokens"] == 0
    assert "Google backend error" in entry["error_message"]


@pytest.mark.asyncio
async def test_dashboard_renders_history_component():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/")
        assert resp.status_code == 200
        assert "Query History" in resp.text
        assert "history-card" in resp.text
        assert "loadHistory()" in resp.text
