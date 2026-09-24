"""Live route tests: WebSocket handshake, key enforcement, session wiring."""

import asyncio

import pytest
from fastapi.testclient import TestClient

import app.routes.live as live_module
from app.keys import api_key_manager
from app.live.transport import LiveTransport, TransportEvent
from main import app


class StubTransport(LiveTransport):
    name = "stub"

    def __init__(self):
        self.opened = 0
        self.closed = 0
        self.requests = []

    def is_available(self) -> bool:
        return True

    async def open(self, setup) -> None:
        self.opened += 1

    async def abort(self) -> None:
        pass

    async def close(self) -> None:
        self.closed += 1

    async def run_turn(self, request):
        self.requests.append(request)
        yield TransportEvent.text_event("spoken reply")
        yield TransportEvent.done(
            {"promptTokenCount": 1, "responseTokenCount": 2, "totalTokenCount": 3}
        )


@pytest.fixture(autouse=True)
def offline_lifespan(monkeypatch):
    """Keep TestClient lifespan startup offline (no upstream probing)."""
    from unittest.mock import AsyncMock

    from app.client import client

    monkeypatch.setattr(
        client, "fetch_available_models", AsyncMock(return_value={"models": {}})
    )
    monkeypatch.setattr(client, "load_code_assist", AsyncMock())


@pytest.fixture
def stub_transport(monkeypatch):
    transport = StubTransport()
    monkeypatch.setattr(live_module, "get_transport", lambda: transport)
    return transport


def test_live_session_exchanges_frames(stub_transport):
    api_key_manager.enforce_keys = False
    with TestClient(app) as client, client.websocket_connect("/v1/live") as ws:
        ws.send_json({"setup": {"model": "gemini-3.7-flash"}})
        assert ws.receive_json() == {"setupComplete": {}}

        ws.send_json(
            {
                "clientContent": {
                    "turns": [{"role": "user", "parts": [{"text": "hi"}]}],
                    "turnComplete": True,
                }
            }
        )
        frames = [ws.receive_json() for _ in range(3)]

    kinds = [f.get("serverContent", {}) for f in frames]
    assert {"outputTranscription": {"text": "spoken reply"}} in kinds
    assert {"turnComplete": True} in kinds
    assert stub_transport.opened == 1
    assert stub_transport.requests[0].prompt == "hi"


def test_live_requires_key_when_enforcement_is_on(stub_transport):
    api_key_manager.enforce_keys = True
    with (
        TestClient(app) as client,
        client.websocket_connect("/v1/live?key=sk-gate-nope") as ws,
    ):
        frame = ws.receive_json()
    assert frame["error"]["code"] == "unauthorized"
    assert "incorrect API key" in frame["error"]["message"]


def test_invalid_json_frame_gets_an_error_not_a_drop(stub_transport):
    api_key_manager.enforce_keys = False
    with TestClient(app) as client, client.websocket_connect("/v1/live") as ws:
        ws.send_text("{not json")
        frame = ws.receive_json()
    assert frame["error"]["code"] == "invalid_argument"


def test_live_status_reports_capabilities(stub_transport):
    api_key_manager.enforce_keys = False
    with TestClient(app) as client:
        response = client.get("/v1/live/status")
    payload = response.json()
    assert response.status_code == 200
    assert payload["schema"] == "bidiGenerateContent"
    assert payload["capabilities"]["voice_in"] is True
    assert payload["capabilities"]["tools"] is False


def test_transport_status_is_serialisable(stub_transport):
    # The status payload is served straight to the dashboard/health consumers,
    # so it must be JSON-clean even with a browser-backed lane behind it.
    import json

    payload = live_module.get_transport().status()
    json.dumps(payload)


@pytest.mark.asyncio
async def test_browser_lane_reports_unavailable_without_playwright(monkeypatch):
    """Without the playwright dependency the lane must refuse, not crash."""
    from app.live.gemini_web_transport import GeminiWebLiveTransport

    transport = GeminiWebLiveTransport()
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "playwright":
            raise ImportError("no playwright")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert transport.is_available() is False

    sent = []

    async def emit(frame):
        sent.append(frame)

    from app.live.session import LiveSession

    session = LiveSession(transport, emit)
    await session.handle({"setup": {"model": "gemini-3.7-flash"}})
    assert sent[-1]["error"]["code"] == "unavailable"
    # No transport work is attempted on an unavailable lane.
    assert session.setup is None
    await asyncio.sleep(0)
