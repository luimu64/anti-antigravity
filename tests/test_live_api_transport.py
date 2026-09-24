"""Tests for the native Live API lane (``app/live/gemini_live_api.py``).

The socket is faked at ``websockets.connect``: these tests pin the frame
contract (setup translation, verbatim pass-through, server-owned interruption)
and the auth handling, not Google's behaviour.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from app.live import gemini_live_api as api
from app.live.protocol import parse_setup
from app.live.session import LiveSession
from app.live.transport import (
    StreamingLiveTransport,
    TransportEvent,
)


class SocketClosed(Exception):
    """Stand-in for ``websockets.exceptions.ConnectionClosed``."""


class FakeWS:
    def __init__(self, incoming: list | None = None) -> None:
        self.sent: list[str] = []
        self.incoming = list(incoming or [])
        self.closed = False

    async def send(self, data) -> None:
        self.sent.append(data)

    async def recv(self):
        if not self.incoming:
            raise SocketClosed("received 1008 (policy violation) connection closed")
        item = self.incoming.pop(0)
        if isinstance(item, (bytes, bytearray)):
            return bytes(item)
        return item if isinstance(item, str) else json.dumps(item)

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_connect(monkeypatch):
    state: dict = {"connections": [], "urls": [], "headers": [], "incoming": []}

    async def connect(url, additional_headers=None, **kwargs):
        state["urls"].append(url)
        state["headers"].append(additional_headers)
        if state.get("fail_attempts", 0) > 0:
            state["fail_attempts"] -= 1
            raise SocketClosed(
                "received 1008 (policy violation) Expected OAuth 2 access token, "
                "login cookie or other valid authentication credential"
            )
        ws = FakeWS(incoming=state.get("incoming"))
        state["connections"].append(ws)
        return ws

    monkeypatch.setattr(api.websockets, "connect", connect)
    return state


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
def test_map_model_only_accepts_live_capable_ids():
    assert api.map_model("") == api.DEFAULT_LIVE_MODEL
    assert api.map_model("gemini-3.7-flash") == api.DEFAULT_LIVE_MODEL
    assert api.map_model("models/gemini-3.7-flash") == api.DEFAULT_LIVE_MODEL
    assert (
        api.map_model("gemini-2.5-flash-native-audio-preview")
        == "models/gemini-2.5-flash-native-audio-preview"
    )
    assert api.map_model("gemini-3.7-flash-live") == "models/gemini-3.7-flash-live"


def test_caller_message_names_the_actual_failure():
    assert "API key not valid" in api.caller_message(ValueError("API key not valid. x"))
    assert "OAuth" in api.caller_message(
        ValueError("Expected OAuth 2 access token, login cookie")
    )
    assert "ephemeral" in api.caller_message(
        ValueError("1007 Missing or malformed auth token; access_token query param")
    )
    single = api.caller_message(ValueError("boom\nsecond line"))
    assert "\n" not in single and "boom" in single


def test_is_available_needs_a_credential():
    assert api.GeminiLiveApiTransport().is_available() is False

    async def provider() -> str:
        return "tok"

    assert api.GeminiLiveApiTransport(token_provider=provider).is_available() is True
    assert api.GeminiLiveApiTransport(api_key="k").is_available() is True


# ---------------------------------------------------------------------------
# Setup / auth
# ---------------------------------------------------------------------------
async def test_open_uses_api_key_query_param(fake_connect):
    transport = api.GeminiLiveApiTransport(api_key="secret-key")
    fake_connect["incoming"] = [{"setupComplete": {}}]
    await transport.open(parse_setup({"model": "gemini-3.7-flash"}))
    assert fake_connect["urls"][0].endswith("?key=secret-key")
    assert fake_connect["headers"][0] == {}
    await transport.close()


async def test_open_uses_bearer_token_for_account_auth(fake_connect):
    async def provider() -> str:
        return "account-access-token"

    transport = api.GeminiLiveApiTransport(token_provider=provider)
    fake_connect["incoming"] = [{"setupComplete": {}}]
    await transport.open(parse_setup({"model": "gemini-3.7-flash"}))
    assert fake_connect["headers"][0] == {
        "Authorization": "Bearer account-access-token"
    }
    assert "key=" not in fake_connect["urls"][0]
    await transport.close()


async def test_auth_failure_refreshes_once_then_retries(fake_connect):
    refreshes = []

    async def provider() -> str:
        return "stale-token"

    async def refresher():
        refreshes.append(1)
        return "fresh-token"

    transport = api.GeminiLiveApiTransport(
        token_provider=provider, token_refresher=refresher
    )
    fake_connect["fail_attempts"] = 1
    fake_connect["incoming"] = [{"setupComplete": {}}]
    await transport.open(parse_setup({"model": "gemini-3.7-flash"}))
    assert len(refreshes) == 1
    assert len(fake_connect["urls"]) == 2
    # The retry must carry the refreshed token, not the stale one (a credential
    # store can keep handing back the old token until it expires).
    assert fake_connect["headers"][1] == {"Authorization": "Bearer fresh-token"}
    await transport.close()


async def test_auth_failure_without_refresher_is_an_actionable_error(fake_connect):
    async def provider() -> str:
        return "stale-token"

    transport = api.GeminiLiveApiTransport(token_provider=provider)
    fake_connect["fail_attempts"] = 1
    with pytest.raises(api.LiveApiError) as excinfo:
        await transport.open(parse_setup({"model": "gemini-3.7-flash"}))
    assert "OAuth" in str(excinfo.value)


async def test_upstream_non_setup_frame_raises(fake_connect):
    fake_connect["incoming"] = [{"error": {"code": 400, "message": "bad model"}}]
    transport = api.GeminiLiveApiTransport(api_key="k")
    with pytest.raises(api.LiveApiError) as excinfo:
        await transport.open(parse_setup({"model": "gemini-3.7-flash"}))
    assert "bad model" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Setup translation + frames
# ---------------------------------------------------------------------------
async def test_upstream_setup_is_live_shaped(fake_connect):
    fake_connect["incoming"] = [{"setupComplete": {}}]
    transport = api.GeminiLiveApiTransport(api_key="k")
    raw_setup = {
        "model": "gemini-3.7-flash",
        "systemInstruction": {"parts": [{"text": "be brief"}]},
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Kore"}}
            },
        },
    }
    setup = parse_setup(raw_setup)
    raw_setup["tools"] = [{"functionDeclarations": [{"name": "get_time"}]}]
    await transport.open(setup, raw_setup)

    sent = json.loads(fake_connect["connections"][0].sent[0])
    upstream = sent["setup"]
    assert upstream["model"] == api.DEFAULT_LIVE_MODEL
    assert upstream["generationConfig"]["responseModalities"] == ["AUDIO"]
    assert (
        upstream["generationConfig"]["speechConfig"]["voiceConfig"][
            "prebuiltVoiceConfig"
        ]["voiceName"]
        == "Kore"
    )
    assert upstream["systemInstruction"] == {"parts": [{"text": "be brief"}]}
    assert upstream["tools"] == [{"functionDeclarations": [{"name": "get_time"}]}]
    await transport.close()


async def test_media_frames_pass_through_verbatim(fake_connect):
    fake_connect["incoming"] = [{"setupComplete": {}}]
    transport = api.GeminiLiveApiTransport(api_key="k")
    await transport.open(parse_setup({"model": "gemini-3.7-flash"}))
    frame = {"realtimeInput": {"mediaChunks": [{"mimeType": "audio/pcm;rate=16000"}]}}
    await transport.send_frame(frame)
    assert json.loads(fake_connect["connections"][0].sent[1]) == frame
    await transport.close()


async def test_events_decode_frames_and_report_socket_end(fake_connect):
    fake_connect["incoming"] = [
        {"setupComplete": {}},
        {"serverContent": {"interrupted": True}},
        b'{"x": 1}',
    ]
    transport = api.GeminiLiveApiTransport(api_key="k")
    await transport.open(parse_setup({"model": "gemini-3.7-flash"}))
    kinds = []
    async for event in transport.events():
        kinds.append((event.kind, event.raw))
    assert kinds[0] == ("frame", {"serverContent": {"interrupted": True}})
    assert kinds[1] == ("frame", {"x": 1})
    assert kinds[2][0] == "error"
    await transport.close()


async def test_send_frame_without_socket_is_an_error():
    transport = api.GeminiLiveApiTransport(api_key="k")
    with pytest.raises(api.LiveApiError):
        await transport.send_frame({"realtimeInput": {}})


# ---------------------------------------------------------------------------
# Streaming session pass-through
# ---------------------------------------------------------------------------
class FakeStreamingTransport(StreamingLiveTransport):
    name = "fake_stream"

    def __init__(self, upstream: list[dict]) -> None:
        self.upstream = upstream
        self.sent: list[dict] = []
        self.setups: list[tuple] = []
        self.closed = 0

    def is_available(self) -> bool:
        return True

    async def open(self, setup, raw_setup=None) -> None:
        self.setups.append((setup, raw_setup))

    async def send_frame(self, frame) -> None:
        self.sent.append(frame)

    async def events(self):
        for frame in self.upstream:
            yield TransportEvent.frame_event(frame)
        await asyncio.sleep(30)

    async def abort(self) -> None:
        return None

    async def close(self) -> None:
        self.closed += 1


async def _drain(times: int = 6) -> None:
    for _ in range(times):
        await asyncio.sleep(0)


async def test_streaming_session_pipes_frames_both_ways():
    upstream = [
        {"serverContent": {"modelTurn": {"parts": [{"inlineData": {"data": "AAA"}}]}}},
        {"serverContent": {"outputTranscription": {"text": "hi there"}}},
        {"serverContent": {"interrupted": True}},
        {"serverContent": {"turnComplete": True}},
        {"usageMetadata": {"promptTokenCount": 7, "responseTokenCount": 9}},
    ]
    transport = FakeStreamingTransport(upstream)
    emitted: list[dict] = []

    async def emit(frame: dict) -> None:
        emitted.append(frame)

    session = LiveSession(transport, emit)
    await session.handle({"setup": {"model": "gemini-3.7-flash"}})
    assert emitted[0] == {"setupComplete": {}}

    # Client audio goes upstream unmodified — no local turn assembly.
    audio = {"realtimeInput": {"mediaChunks": [{"mimeType": "audio/pcm;rate=16000"}]}}
    await session.handle(audio)
    assert transport.sent == [audio]
    assert transport.setups[0][1] == {"model": "gemini-3.7-flash"}

    # Tool responses are supported on this lane.
    tool = {"toolResponse": {"functionResponses": [{"id": "1"}]}}
    await session.handle(tool)
    assert transport.sent[-1] == tool
    assert not any(f.get("error") for f in emitted)

    await _drain(10)
    assert {"serverContent": {"interrupted": True}} in emitted
    assert {
        "usageMetadata": {"promptTokenCount": 7, "responseTokenCount": 9}
    } in emitted
    assert session.stats["interruptions"] == 1
    assert session.stats["turns"] == 1
    assert session.stats["streaming"] is True
    assert session.stats["prompt_tokens"] == 7
    assert session.stats["response_tokens"] == 9
    assert session.transcript == "hi there"

    await session.close()
    assert transport.closed == 1
    assert session.stats["generating"] is False


async def test_streaming_session_oversized_chunk_is_rejected():
    transport = FakeStreamingTransport([])
    emitted: list[dict] = []

    async def emit(frame: dict) -> None:
        emitted.append(frame)

    session = LiveSession(transport, emit)
    await session.handle({"setup": {"model": "gemini-3.7-flash"}})
    import base64

    from app.live import protocol as p

    huge = base64.b64encode(b"\x00" * (p.MAX_MEDIA_CHUNK_BYTES + 4096)).decode()
    await session.handle(
        {"realtimeInput": {"mediaChunks": [{"mimeType": "audio/pcm", "data": huge}]}}
    )
    assert transport.sent == []
    assert emitted[-1]["error"]["code"] == "invalid_argument"
    await session.close()


# ---------------------------------------------------------------------------
# Lane selection
# ---------------------------------------------------------------------------
def test_build_transport_prefers_the_live_lane(monkeypatch):
    from app.routes import live as live_route

    monkeypatch.delenv("LIVE_LANE", raising=False)
    monkeypatch.setenv("GEMINI_LIVE_API_KEY", "k")
    transport = live_route.build_transport()
    assert transport.name == "gemini_live_api"
    assert transport.is_available() is True


def test_build_transport_falls_back_to_the_cookie_lane(monkeypatch):
    from app.routes import live as live_route

    monkeypatch.delenv("GEMINI_LIVE_API_KEY", raising=False)
    monkeypatch.setattr(live_route, "_account_auth", lambda: (None, None))
    monkeypatch.delenv("LIVE_LANE", raising=False)
    assert live_route.build_transport().name == "gemini_web"
    monkeypatch.setenv("LIVE_LANE", "web")
    assert live_route.build_transport().name == "gemini_web"


def test_build_transport_uses_account_oauth_when_signed_in(monkeypatch):
    from app.routes import live as live_route

    async def provider() -> str:
        return "tok"

    async def refresher():
        return "tok"

    monkeypatch.delenv("GEMINI_LIVE_API_KEY", raising=False)
    monkeypatch.delenv("LIVE_LANE", raising=False)
    monkeypatch.setattr(live_route, "_account_auth", lambda: (provider, refresher))
    transport = live_route.build_transport()
    assert transport.name == "gemini_live_api"
    assert transport.status()["auth"] == "oauth_access_token"


def test_empty_env_does_not_wipe_defaults(monkeypatch):
    """compose passes `VAR=${VAR:-}`, which sets an empty string, not "unset"."""
    monkeypatch.setenv("GEMINI_LIVE_MODEL", "")
    monkeypatch.setenv("GEMINI_LIVE_WS_URL", "")
    monkeypatch.setenv("GEMINI_LIVE_SETUP_TIMEOUT_S", "")
    assert api._env("GEMINI_LIVE_MODEL", "fallback") == "fallback"
    assert api.DEFAULT_LIVE_MODEL.startswith("models/gemini-")
    assert api.LIVE_WS_URL.startswith("wss://")
    assert api.SETUP_TIMEOUT_S > 0
