from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

import app.realtime as realtime_module
from app.client import client
from app.history import QueryHistoryManager, history_manager
from app.realtime import RealtimeHub, hub
from app.routes.dashboard import (
    _realtime_quota_collector,
    collect_quota_groups,
)
from main import app


@pytest.fixture(autouse=True)
def isolated_hub_state():
    """
    Snapshot & restore realtime hub state around each test.
    """
    orig_collector = hub._quota_collector
    orig_loop = hub._loop
    orig_dirty = hub._quota_dirty
    yield
    hub._quota_collector = orig_collector
    hub._loop = orig_loop
    hub._quota_dirty = orig_dirty
    history_manager.clear()


@pytest.fixture(autouse=True)
def stub_lifespan_network(monkeypatch):
    """
    Keep lifespan startup offline while using TestClient.
    """
    monkeypatch.setattr(
        client,
        "fetch_available_models",
        AsyncMock(return_value={"models": {}}),
    )
    monkeypatch.setattr(client, "load_code_assist", AsyncMock())


@pytest.fixture
def stub_quota_collector():
    """
    Replace the upstream quota collector with a controllable stub.
    """
    collector = AsyncMock(return_value=[])
    hub.set_quota_collector(collector)
    return collector


@pytest.fixture
def fast_debounce(monkeypatch):
    monkeypatch.setattr(realtime_module, "QUOTA_REFRESH_DEBOUNCE_S", 0.05)


def test_websocket_hello_and_ping_pong():
    with TestClient(app) as tc, tc.websocket_connect("/ws") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "hello"
        assert hello["payload"]["clients"] >= 1

        ws.send_text("ping")
        pong = ws.receive_json()
        assert pong["type"] == "pong"


def test_client_count_tracked_on_connect_and_disconnect():
    assert hub.client_count == 0
    with TestClient(app) as tc:
        with tc.websocket_connect("/ws"):
            assert hub.client_count == 1
        assert hub.client_count == 0


def test_history_new_event_pushed_over_websocket(stub_quota_collector):
    history_manager.clear()
    with TestClient(app) as tc, tc.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "hello"

        entry = history_manager.record(
            model="gpt-4o",
            resolved_model="gemini-3.7-flash-high",
            backend="antigravity",
            duration_ms=42.0,
            status="success",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            request_id="req_ws_live_1",
        )
        assert entry["id"] == "req_ws_live_1"

        msg = ws.receive_json()
        assert msg["type"] == "history.new"
        assert msg["payload"]["id"] == "req_ws_live_1"
        assert msg["payload"]["model"] == "gpt-4o"
        assert msg["payload"]["total_tokens"] == 30


def test_history_clear_event_pushed_over_websocket(stub_quota_collector):
    with TestClient(app) as tc, tc.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "hello"

        history_manager.clear()
        msg = ws.receive_json()
        assert msg["type"] == "history.clear"


def test_quota_update_pushed_after_activity(fast_debounce, stub_quota_collector):
    stub_quota_collector.return_value = [
        {
            "display_name": "Gemini Flash",
            "backend": "antigravity",
            "remaining_fraction": 0.75,
        }
    ]
    with TestClient(app) as tc, tc.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "hello"

        # Recording a query marks quotas dirty -> debounced push follows
        history_manager.record(
            model="gpt-4o",
            resolved_model="gemini-3.7-flash-high",
            backend="antigravity",
            duration_ms=10.0,
            request_id="req_ws_quota_trigger",
        )

        first = ws.receive_json()
        assert first["type"] == "history.new"

        second = ws.receive_json()
        assert second["type"] == "quotas.update"
        groups = second["payload"]["groups"]
        assert len(groups) == 1
        assert groups[0]["display_name"] == "Gemini Flash"
        stub_quota_collector.assert_awaited()


def test_manual_publish_reaches_clients(stub_quota_collector):
    with (
        TestClient(app) as tc,
        tc.websocket_connect("/ws") as ws_a,
        tc.websocket_connect("/ws") as ws_b,
    ):
        assert ws_a.receive_json()["type"] == "hello"
        assert ws_b.receive_json()["type"] == "hello"

        hub.publish("backends.updated", {"foo": "bar"})

        for ws in (ws_a, ws_b):
            msg = ws.receive_json()
            assert msg["type"] == "backends.updated"
            assert msg["payload"] == {"foo": "bar"}


def test_realtime_quota_collector_registered_by_dashboard():
    assert hub._quota_collector is _realtime_quota_collector


@pytest.mark.asyncio
async def test_dashboard_quota_collector_returns_groups():
    with patch.object(
        client,
        "retrieve_user_quota_summary",
        new_callable=AsyncMock,
        return_value={"buckets": []},
    ):
        groups, err = await collect_quota_groups()
    assert isinstance(groups, list)
    assert err is None


def test_publish_without_event_loop_is_silent_noop():
    fresh_hub = RealtimeHub()
    # No loop attached and none running - must not raise
    fresh_hub.publish("history.new", {"id": "x"})
    fresh_hub.request_quota_refresh()
    fresh_hub.disconnect(None)


def test_record_without_loop_does_not_break_history_buffering():
    mgr = QueryHistoryManager(max_records=3)
    entry = mgr.record(
        model="m",
        resolved_model="r",
        backend="b",
        duration_ms=1.0,
        request_id="req_offline",
    )
    assert len(mgr) == 1
    assert entry["id"] == "req_offline"
