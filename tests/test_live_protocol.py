"""Live protocol tests: setup parsing, audio helpers, RPC audio extraction."""

import base64
import json
import struct

import pytest

from app.live import protocol as p
from app.live.gemini_web_transport import (
    chunk_audio,
    clean_reply_text,
    estimate_tokens,
    extract_dom_state,
    resolve_audio_out,
    turn_prompt,
)
from app.live.transport import TurnRequest


def _rpc_body(audio: bytes, marker: str = "JLpPJe") -> str:
    """Build a batchexecute response envelope carrying base64 audio."""
    inner = json.dumps([None, None, None, base64.b64encode(audio).decode()])
    envelope = json.dumps(
        [["wrb.fr", marker, inner, None, None, None, None, "generic"]]
    )
    return ")]}'\n\n" + envelope + "\n1234\n"


MP3 = b"ID3\x04\x00\x00\x00\x00\x00\x00" + bytes(4000)


# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------
def test_parse_setup_full():
    setup = p.parse_setup(
        {
            "model": "models/gemini-3.7-flash",
            "systemInstruction": {
                "parts": [{"text": "Be terse."}, {"text": "Finnish."}]
            },
            "generationConfig": {
                "responseModalities": ["audio", "text"],
                "temperature": 0.4,
                "maxOutputTokens": 512,
                "speechConfig": {
                    "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Kore"}}
                },
            },
            "tools": [{"functionDeclarations": [{"name": "noop"}]}],
        }
    )
    assert setup.model == "gemini-3.7-flash"
    assert setup.system_instruction == "Be terse.\nFinnish."
    assert setup.response_modalities == ["AUDIO", "TEXT"]
    assert setup.voice == "Kore"
    assert setup.temperature == 0.4
    assert setup.max_output_tokens == 512
    assert setup.tools_requested is True


def test_parse_setup_minimal_and_errors():
    setup = p.parse_setup({"model": "gemini-live"})
    assert setup.response_modalities == ["AUDIO"]
    assert setup.input_audio_rate == p.DEFAULT_INPUT_RATE

    with pytest.raises(p.LiveMessageError):
        p.parse_setup({"model": "   "})
    with pytest.raises(p.LiveMessageError):
        p.parse_setup("nope")


def test_parse_rate_guards():
    assert p.parse_rate("audio/pcm;rate=16000", 24000) == 16000
    assert p.parse_rate("audio/pcm", 24000) == 24000
    assert p.parse_rate("audio/pcm;rate=3", 24000) == 24000  # absurd -> default
    assert p.parse_rate(None, 16000) == 16000


# ---------------------------------------------------------------------------
# audio helpers
# ---------------------------------------------------------------------------
def test_pcm16_to_wav_header():
    pcm = struct.pack("<4h", 1, -2, 3, -4)
    wav = p.pcm16_to_wav(pcm, rate=16000)
    assert wav[:4] == b"RIFF"
    assert wav[8:12] == b"WAVE"
    assert struct.unpack("<I", wav[4:8])[0] == 36 + len(pcm)
    assert wav[-len(pcm) :] == pcm
    assert struct.unpack("<I", wav[24:28])[0] == 16000


def test_sniff_and_resolve_audio_out():
    assert p.sniff_audio_mime(MP3) == "audio/mpeg"
    assert p.sniff_audio_mime(b"RIFF\x00\x00\x00\x00WAVE") == "audio/wav"
    assert p.sniff_audio_mime(bytes(64)) == "audio/pcm;rate=24000"
    assert p.sniff_audio_mime(MP3) == "audio/mpeg"

    audio, mime = resolve_audio_out(b"", 24000)
    assert (audio, mime) == (b"", "")

    raw = bytes(128)
    audio, mime = resolve_audio_out(raw, 16000)
    assert audio == raw and mime == "audio/pcm;rate=16000"


def test_decode_base64_tolerates_garbage():
    assert p.decode_base64(None) == b""
    assert p.decode_base64("not base64 !!") == b""
    assert p.decode_base64(base64.b64encode(b"abc").decode()) == b"abc"


def test_extract_audio_from_rpc_body_finds_nested_audio():
    assert p.extract_audio_from_rpc_body(_rpc_body(MP3)) == MP3


def test_extract_audio_from_rpc_body_ignores_text_payloads():
    body = ")]}'\n\n" + json.dumps(
        [["wrb.fr", "RxAFq", json.dumps([None, ["hello there", 0]])]]
    )
    assert p.extract_audio_from_rpc_body(body) == b""


def test_extract_audio_from_rpc_body_picks_largest():
    small = b"ID3\x04" + bytes(300)
    large = b"ID3\x04" + bytes(9000)
    body = _rpc_body(small) + _rpc_body(large)
    assert p.extract_audio_from_rpc_body(body) == large


def test_chunk_audio():
    assert chunk_audio(b"12345", 2) == [b"12", b"34", b"5"]
    assert chunk_audio(b"", 2) == []
    assert chunk_audio(b"abc", 0) == [b"abc"]


# ---------------------------------------------------------------------------
# frames
# ---------------------------------------------------------------------------
def test_frame_builders():
    assert p.setup_complete() == {"setupComplete": {}}
    assert p.turn_complete() == {"serverContent": {"turnComplete": True}}
    assert p.interrupted() == {"serverContent": {"interrupted": True}}

    frame = p.server_audio(b"\x01\x02", "audio/pcm;rate=24000")
    part = frame["serverContent"]["modelTurn"]["parts"][0]
    assert part["inlineData"]["mimeType"] == "audio/pcm;rate=24000"
    assert base64.b64decode(part["inlineData"]["data"]) == b"\x01\x02"

    assert p.server_text("hi")["serverContent"]["modelTurn"]["parts"][0]["text"] == "hi"
    assert p.output_transcription("hi") == {
        "serverContent": {"outputTranscription": {"text": "hi"}}
    }
    usage = p.usage(3, 4, 7)["usageMetadata"]
    assert usage["promptTokenCount"] == 3 and usage["totalTokenCount"] == 7
    assert p.error("nope")["error"]["message"] == "nope"


# ---------------------------------------------------------------------------
# transport pure helpers
# ---------------------------------------------------------------------------
def test_clean_reply_text_strips_chrome():
    raw = "Gemini said\nHere is the answer.\nthumb_up\nthumb_down\nListen\nCopy"
    assert clean_reply_text(raw) == "Here is the answer."
    assert clean_reply_text("") == ""


def test_extract_dom_state_normalises_shapes():
    assert extract_dom_state(None) == {}
    state = extract_dom_state(
        json.dumps(
            {
                "reply": "Gemini said\nHello.\nCopy",
                "generating": False,
                "hasListen": True,
                "turns": "3",
                "inputReady": True,
            }
        )
    )
    assert state["reply"] == "Hello."
    assert state["turns"] == 3
    assert state["hasListen"] is True
    assert state["hasStop"] is False


def test_turn_prompt_prefixes_persona_once():
    first = turn_prompt(
        TurnRequest(prompt="hello", system_instruction="Be terse.", first_turn=True)
    )
    assert first.startswith("[Instructions]\nBe terse.\n[End instructions]")
    assert first.endswith("hello")

    later = turn_prompt(TurnRequest(prompt="again", system_instruction="Be terse."))
    assert later == "again"

    persona_only = turn_prompt(
        TurnRequest(prompt="", system_instruction="Be terse.", first_turn=True)
    )
    assert persona_only.startswith("[Instructions]")


def test_estimate_tokens():
    assert estimate_tokens("") == 0
    assert estimate_tokens("   ") == 0
    assert estimate_tokens("one two three") == 3


def test_launch_error_is_a_single_caller_safe_line():
    from app.live.gemini_web_transport import _launch_error

    raw = (
        "BrowserType.launch_persistent_context: Executable doesn't exist at "
        "/opt/ms-playwright/chromium\n"
        "╔═══════════════════════════════════════╗\n"
        "║ Please run the following command       ║\n"
        "║     playwright install                 ║\n"
        "╚═══════════════════════════════════════╝"
    )
    message = _launch_error(RuntimeError(raw))
    assert "\n" not in message
    assert message.startswith("browser launch failed:")
    assert "Executable doesn't exist" in message
    assert "chromium" in message
    assert "install a browser" in message

    plain = _launch_error(RuntimeError("some other problem"))
    assert plain == "browser launch failed: some other problem"

    long = _launch_error(RuntimeError("x" * 500))
    assert len(long) < 300
