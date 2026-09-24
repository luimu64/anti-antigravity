"""Live session state-machine tests (fake transport, no browser, no network).

Covers: setup handshake, voice-in turn assembly (explicit activity markers and
silence flush), voice-out frames, barge-in semantics (interrupted frame, no
turnComplete for the cut turn, new turn proceeds), unsupported tool calling and
usage accounting.
"""

import asyncio
import base64

import pytest

from app.live.session import LiveSession
from app.live.transport import LiveTransport, TransportEvent


class FakeTransport(LiveTransport):
    name = "fake"

    def __init__(self, *, audio_out: bytes = b"", text_out: str = "ok", fail: str = ""):
        self.opened = 0
        self.aborted = 0
        self.closed = 0
        self.requests = []
        self.audio_out = audio_out
        self.text_out = text_out
        self.fail = fail
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    def is_available(self) -> bool:
        return True

    async def open(self, setup) -> None:
        self.opened += 1

    async def abort(self) -> None:
        self.aborted += 1

    async def close(self) -> None:
        self.closed += 1

    async def run_turn(self, request):
        self.requests.append(request)
        self.started.set()
        if self.fail:
            yield TransportEvent.error(self.fail)
            yield TransportEvent.done()
            return
        # Wait to be released so barge-in can be exercised deterministically.
        await self.release.wait()
        if self.audio_out:
            yield TransportEvent.audio_event(self.audio_out, "audio/mpeg")
        if self.text_out:
            yield TransportEvent.text_event(self.text_out)
        yield TransportEvent.done(
            {"promptTokenCount": 2, "responseTokenCount": 3, "totalTokenCount": 5}
        )


@pytest.fixture
def frames():
    sent: list[dict] = []

    async def emit(frame: dict) -> None:
        sent.append(frame)

    return sent, emit


def _pcm(n: int) -> str:
    return base64.b64encode(bytes(n)).decode()


async def _setup(session: LiveSession) -> None:
    await session.handle(
        {
            "setup": {
                "model": "gemini-3.7-flash",
                "systemInstruction": {"parts": [{"text": "Be brief."}]},
            }
        }
    )


@pytest.mark.asyncio
async def test_setup_handshake(frames):
    sent, emit = frames
    transport = FakeTransport()
    session = LiveSession(transport, emit)
    await _setup(session)

    assert sent == [{"setupComplete": {}}]
    assert transport.opened == 1
    assert session.setup.model == "gemini-3.7-flash"


@pytest.mark.asyncio
async def test_frames_before_setup_are_rejected(frames):
    sent, emit = frames
    session = LiveSession(FakeTransport(), emit)
    await session.handle({"realtimeInput": {"mediaChunks": []}})
    assert sent[0]["error"]["code"] == "failed_precondition"


@pytest.mark.asyncio
async def test_voice_in_turn_assembles_audio_and_streams_audio_out(frames):
    sent, emit = frames
    transport = FakeTransport(audio_out=b"ID3" + bytes(10), text_out="hello there")
    session = LiveSession(transport, emit, silence_flush_s=0)
    await _setup(session)

    await session.handle({"realtimeInput": {"activityStart": {}}})
    await session.handle(
        {
            "realtimeInput": {
                "mediaChunks": [{"mimeType": "audio/pcm;rate=16000", "data": _pcm(320)}]
            }
        }
    )
    await session.handle({"realtimeInput": {"activityEnd": {}}})
    transport.release.set()
    await asyncio.wait_for(session._turn_task, timeout=2)

    request = transport.requests[0]
    assert len(request.audio) == 320
    assert request.audio_rate == 16000
    assert request.first_turn is True
    assert request.system_instruction == "Be brief."

    kinds = [f.get("serverContent", {}) for f in sent]
    assert {
        "modelTurn": {
            "role": "model",
            "parts": [
                {
                    "inlineData": {
                        "mimeType": "audio/mpeg",
                        "data": base64.b64encode(b"ID3" + bytes(10)).decode(),
                    }
                }
            ],
        }
    } in kinds
    assert {"outputTranscription": {"text": "hello there"}} in kinds
    assert {"turnComplete": True} in kinds
    assert session.stats["prompt_tokens"] == 2
    assert session.stats["response_tokens"] == 3


@pytest.mark.asyncio
async def test_silence_flush_closes_a_turn_without_activity_end(frames):
    sent, emit = frames
    transport = FakeTransport(text_out="flushed")
    session = LiveSession(transport, emit, silence_flush_s=0.05)
    await _setup(session)

    await session.handle(
        {
            "realtimeInput": {
                "mediaChunks": [{"mimeType": "audio/pcm", "data": _pcm(64)}]
            }
        }
    )
    transport.release.set()
    for _ in range(50):
        await asyncio.sleep(0.02)
        if transport.requests:
            break
    assert transport.requests, "silence flush never released the turn"
    await asyncio.wait_for(session._turn_task, timeout=2)
    assert any("turnComplete" in (f.get("serverContent") or {}) for f in sent)


@pytest.mark.asyncio
async def test_barge_in_interrupts_without_closing_cut_turn(frames):
    sent, emit = frames
    transport = FakeTransport(text_out="first")
    session = LiveSession(transport, emit, silence_flush_s=0)
    await _setup(session)

    await session.handle({"realtimeInput": {"activityStart": {}}})
    await session.handle(
        {
            "realtimeInput": {
                "mediaChunks": [{"mimeType": "audio/pcm", "data": _pcm(64)}]
            }
        }
    )
    await session.handle({"realtimeInput": {"activityEnd": {}}})
    await asyncio.wait_for(transport.started.wait(), timeout=2)

    # User starts talking over the model.
    await session.handle({"realtimeInput": {"activityStart": {}}})
    assert sent[-1] == {"serverContent": {"interrupted": True}}
    assert transport.aborted == 1
    assert session.stats["interruptions"] == 1
    # The cut turn must not have produced a turnComplete frame.
    assert not any(f.get("serverContent", {}).get("turnComplete") for f in sent)

    # The replacement turn runs normally.
    transport.release.set()
    await session.handle(
        {
            "realtimeInput": {
                "mediaChunks": [{"mimeType": "audio/pcm", "data": _pcm(32)}]
            }
        }
    )
    await session.handle({"realtimeInput": {"activityEnd": {}}})
    await asyncio.wait_for(session._turn_task, timeout=2)
    assert len(transport.requests) == 2
    assert any(f.get("serverContent", {}).get("turnComplete") for f in sent)


@pytest.mark.asyncio
async def test_client_content_text_turn(frames):
    sent, emit = frames
    transport = FakeTransport(text_out="answer")
    session = LiveSession(transport, emit, silence_flush_s=0)
    await _setup(session)

    transport.release.set()
    await session.handle(
        {
            "clientContent": {
                "turns": [{"role": "user", "parts": [{"text": "what time is it"}]}],
                "turnComplete": True,
            }
        }
    )
    await asyncio.wait_for(session._turn_task, timeout=2)
    assert transport.requests[0].prompt == "what time is it"
    assert transport.requests[0].audio == b""


@pytest.mark.asyncio
async def test_tool_response_is_reported_unsupported(frames):
    sent, emit = frames
    session = LiveSession(FakeTransport(), emit)
    await _setup(session)
    await session.handle({"toolResponse": {"functionResponses": []}})
    assert sent[-1]["error"]["code"] == "unimplemented"


@pytest.mark.asyncio
async def test_oversized_media_chunk_is_rejected(frames):
    sent, emit = frames
    session = LiveSession(FakeTransport(), emit)
    await _setup(session)
    await session.handle(
        {
            "realtimeInput": {
                "mediaChunks": [
                    {
                        "mimeType": "audio/pcm",
                        "data": base64.b64encode(bytes((1 << 20) + 2)).decode(),
                    }
                ]
            }
        }
    )
    assert sent[-1]["error"]["code"] == "invalid_argument"


@pytest.mark.asyncio
async def test_transport_error_reaches_the_client(frames):
    sent, emit = frames
    transport = FakeTransport(fail="voice-out unavailable: no TTS audio captured")
    session = LiveSession(transport, emit, silence_flush_s=0)
    await _setup(session)
    await session.handle(
        {
            "realtimeInput": {
                "mediaChunks": [{"mimeType": "audio/pcm", "data": _pcm(64)}]
            }
        }
    )
    transport.release.set()
    await session.handle({"realtimeInput": {"activityEnd": {}}})
    await asyncio.wait_for(session._turn_task, timeout=2)
    errors = [f["error"]["message"] for f in sent if "error" in f]
    assert errors and errors[-1].startswith("voice-out unavailable")
    assert session.stats["last_error"].startswith("voice-out unavailable")


@pytest.mark.asyncio
async def test_close_cancels_turn_and_closes_transport(frames):
    sent, emit = frames
    transport = FakeTransport()
    session = LiveSession(transport, emit, silence_flush_s=0)
    await _setup(session)
    await session.handle({"realtimeInput": {"activityStart": {}}})
    await session.handle(
        {
            "realtimeInput": {
                "mediaChunks": [{"mimeType": "audio/pcm", "data": _pcm(64)}]
            }
        }
    )
    await session.handle({"realtimeInput": {"activityEnd": {}}})
    await asyncio.wait_for(transport.started.wait(), timeout=2)
    await session.close()
    assert transport.closed == 1
    assert session.stats["generating"] is False
