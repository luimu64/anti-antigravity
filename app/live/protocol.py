"""Gemini Live API (BidiGenerateContent) wire schema + audio helpers.

Only the subset a voice-in / voice-out session needs is modelled:

client -> gateway
    ``{"setup": {"model": ..., "generationConfig": {...}, "systemInstruction": {...}}}``
    ``{"realtimeInput": {"mediaChunks": [{"mimeType": "audio/pcm;rate=16000", "data": "<b64>"}]}}``
    ``{"realtimeInput": {"activityStart": {}}}`` / ``{"activityEnd": {}}``
    ``{"clientContent": {"turns": [{"role": "user", "parts": [{"text": ...}]}], "turnComplete": true}}``
    ``{"toolResponse": {...}}``

gateway -> client
    ``{"setupComplete": {}}``
    ``{"serverContent": {"modelTurn": {"role": "model", "parts": [{"inlineData": {"mimeType", "data"}}]}}}``
    ``{"serverContent": {"outputTranscription": {"text": ...}}}``
    ``{"serverContent": {"turnComplete": true}}`` / ``{"serverContent": {"interrupted": true}}``
    ``{"usageMetadata": {...}}``

Frames are plain dicts; :func:`dumps` is the only serializer so tests and the
route agree on separators.
"""

from __future__ import annotations

import base64
import binascii
import json
import re
import struct
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Audio formats
# ---------------------------------------------------------------------------
PCM_MIME_TEMPLATE = "audio/pcm;rate={rate}"
DEFAULT_INPUT_RATE = 16000
DEFAULT_OUTPUT_RATE = 24000
# A single mediaChunk larger than this is refused: the client is expected to
# stream ~20-100 ms frames, and an unbounded append is a memory leak.
MAX_MEDIA_CHUNK_BYTES = 1 << 20
# Ceiling for one utterance buffer (PCM16 @16 kHz ≈ 32 kB/s → ~2 minutes).
MAX_TURN_AUDIO_BYTES = 4 << 20

_MP3_MAGIC = (b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2", b"\xff\xfa")
_AUDIO_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"RIFF", "audio/wav"),
    (b"OggS", "audio/ogg"),
    (b"fLaC", "audio/flac"),
    (b"ID3", "audio/mpeg"),
    (b"\xff\xfb", "audio/mpeg"),
    (b"\xff\xf3", "audio/mpeg"),
    (b"\xff\xf2", "audio/mpeg"),
    (b"\xff\xfa", "audio/mpeg"),
    (b"ftyp", "audio/mp4"),
)

MIME_RATE_RE = re.compile(r"rate=(\d+)")


class LiveMessageError(ValueError):
    """A client frame violated the Live schema."""


# ---------------------------------------------------------------------------
# Setup parsing
# ---------------------------------------------------------------------------
@dataclass
class LiveSetup:
    """Normalised ``setup`` frame contents."""

    model: str
    system_instruction: str = ""
    voice: str = ""
    response_modalities: list[str] = field(default_factory=lambda: ["AUDIO"])
    input_audio_rate: int = DEFAULT_INPUT_RATE
    output_audio_rate: int = DEFAULT_OUTPUT_RATE
    temperature: float | None = None
    max_output_tokens: int | None = None
    language: str = ""
    barge_in: bool = True
    tools_requested: bool = False


def _parts_text(parts: Any) -> str:
    if not isinstance(parts, list):
        return ""
    out = []
    for part in parts:
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            out.append(part["text"])
    return "\n".join(out).strip()


# Public alias: the session state machine reads client turns through this.
parts_text = _parts_text


def parse_rate(mime: str | None, default: int) -> int:
    """Extract the ``rate=`` parameter from a Live audio mime type."""
    if not mime:
        return default
    match = MIME_RATE_RE.search(str(mime))
    if not match:
        return default
    try:
        rate = int(match.group(1))
    except ValueError:
        return default
    # Guard against nonsense values that would make WAV headers lie.
    return rate if 4000 <= rate <= 192000 else default


def parse_setup(frame: dict[str, Any]) -> LiveSetup:
    """Validate and normalise a ``setup`` frame.

    The Live API requires ``setup.model``; everything else is optional. Unknown
    keys are ignored rather than rejected — clients add fields over time and a
    hard failure there would be a false negative.
    """
    if not isinstance(frame, dict):
        raise LiveMessageError("setup must be an object")
    model = frame.get("model") or frame.get("model_id") or ""
    if not isinstance(model, str) or not model.strip():
        raise LiveMessageError("setup.model is required")

    setup = LiveSetup(model=model.strip().removeprefix("models/"))

    instructions = frame.get("systemInstruction") or frame.get("system_instruction")
    if isinstance(instructions, dict):
        setup.system_instruction = (
            _parts_text(instructions.get("parts"))
            or str(instructions.get("text") or "").strip()
        )
    elif isinstance(instructions, str):
        setup.system_instruction = instructions.strip()

    generation = frame.get("generationConfig") or frame.get("generation_config") or {}
    if isinstance(generation, dict):
        modalities = generation.get("responseModalities") or generation.get(
            "response_modalities"
        )
        if isinstance(modalities, list) and modalities:
            setup.response_modalities = [str(m).upper() for m in modalities]
        for key, attr in (
            ("temperature", "temperature"),
            ("maxOutputTokens", "max_output_tokens"),
            ("max_output_tokens", "max_output_tokens"),
        ):
            value = generation.get(key)
            if isinstance(value, int | float) and attr == "max_output_tokens":
                setattr(setup, attr, int(value))
            elif isinstance(value, int | float):
                setattr(setup, attr, float(value))
        speech = generation.get("speechConfig") or {}
        if isinstance(speech, dict):
            voice_config = speech.get("voiceConfig") or {}
            prebuilt = (
                voice_config.get("prebuiltVoiceConfig")
                if isinstance(voice_config, dict)
                else {}
            )
            if isinstance(prebuilt, dict) and prebuilt.get("voiceName"):
                setup.voice = str(prebuilt["voiceName"])
        media = generation.get("mediaResolution")
        if isinstance(media, str):
            setup.language = ""  # reserved; no wire meaning on this lane

    language = frame.get("language") or frame.get("languageCode")
    if isinstance(language, str):
        setup.language = language.strip()

    if isinstance(frame.get("tools"), list) and frame["tools"]:
        setup.tools_requested = True

    return setup


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------
def decode_base64(data: Any) -> bytes:
    """Decode a Live ``inlineData.data`` payload, tolerating URL-safe input."""
    if not isinstance(data, str) or not data:
        return b""
    try:
        return base64.b64decode(data, validate=False)
    except (binascii.Error, ValueError):
        return b""


def sniff_audio_mime(data: bytes) -> str:
    """Return the container mime for raw audio bytes, else raw PCM16."""
    head = data[:12]
    for magic, mime in _AUDIO_MAGIC:
        if head.startswith(magic):
            return mime
    if head[4:8] == b"ftyp":
        return "audio/mp4"
    return PCM_MIME_TEMPLATE.format(rate=DEFAULT_OUTPUT_RATE)


def pcm16_to_wav(
    pcm: bytes, rate: int = DEFAULT_INPUT_RATE, channels: int = 1
) -> bytes:
    """Wrap raw little-endian PCM16 in a RIFF/WAVE header."""
    byte_rate = rate * channels * 2
    header = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVE"
    header += b"fmt " + struct.pack("<IHHIIHH", 16, 1, channels, rate, byte_rate, 2, 16)
    header += b"data" + struct.pack("<I", len(pcm))
    return header + pcm


def looks_like_audio(data: bytes) -> bool:
    """Cheap container test used when fishing audio out of RPC payloads."""
    if len(data) < 16:
        return False
    head = data[:4]
    if any(head.startswith(magic[: len(head)]) for magic in _MP3_MAGIC):
        return True
    return any(data.startswith(magic) for magic, _ in _AUDIO_MAGIC) or (
        data[4:8] == b"ftyp"
    )


def iter_base64_strings(node: Any, min_length: int = 64):
    """Walk a decoded JSON tree yielding base64-looking strings."""
    if isinstance(node, str):
        if len(node) >= min_length and re.fullmatch(r"[A-Za-z0-9+/=_-]+", node):
            yield node
    elif isinstance(node, list):
        for item in node:
            yield from iter_base64_strings(item, min_length)
    elif isinstance(node, dict):
        for item in node.values():
            yield from iter_base64_strings(item, min_length)


def extract_audio_from_rpc_body(body: str, min_length: int = 256) -> bytes:
    """Pull the largest audio blob out of a batchexecute/JSPB response body.

    Web RPC responses are chunked JSON envelopes prefixed with ``)]}'``; the
    audio arrives base64-encoded somewhere in the tree. The layout is not
    documented upstream, so this walks every plausible position instead of
    hard-coding an index — it must survive a payload reshuffle.
    """
    best = b""
    for line in (body or "").split("\n"):
        line = line.strip()
        if not line or line.startswith(")]}'") or line.isdigit():
            continue
        try:
            parsed = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        candidates: list[Any] = [parsed]
        # Unwrap the standard wrb.fr envelope: [["wrb.fr", null, "<json>"]].
        if isinstance(parsed, list):
            for item in parsed:
                if (
                    isinstance(item, list)
                    and len(item) > 2
                    and isinstance(item[2], str)
                ):
                    try:
                        candidates.append(json.loads(item[2]))
                    except (json.JSONDecodeError, ValueError):
                        continue
        for candidate in candidates:
            for blob in iter_base64_strings(candidate, min_length):
                for decoder in (
                    lambda s: base64.b64decode(s, validate=False),
                    lambda s: base64.b64decode(s + "=" * (-len(s) % 4), validate=False),
                ):
                    try:
                        raw = decoder(blob)
                    except (binascii.Error, ValueError):
                        continue
                    if looks_like_audio(raw) and len(raw) > len(best):
                        best = raw
    return best


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


# ---------------------------------------------------------------------------
# Frame builders
# ---------------------------------------------------------------------------
def dumps(frame: dict[str, Any]) -> str:
    return json.dumps(frame, separators=(",", ":"))


def setup_complete() -> dict[str, Any]:
    return {"setupComplete": {}}


def server_audio(data: bytes, mime: str) -> dict[str, Any]:
    return {
        "serverContent": {
            "modelTurn": {
                "role": "model",
                "parts": [{"inlineData": {"mimeType": mime, "data": b64(data)}}],
            }
        }
    }


def server_text(text: str) -> dict[str, Any]:
    """Model text as a ``modelTurn`` part (transcript of what is spoken)."""
    return {
        "serverContent": {"modelTurn": {"role": "model", "parts": [{"text": text}]}}
    }


def output_transcription(text: str) -> dict[str, Any]:
    return {"serverContent": {"outputTranscription": {"text": text}}}


def input_transcription(text: str) -> dict[str, Any]:
    return {"serverContent": {"inputTranscription": {"text": text}}}


def turn_complete() -> dict[str, Any]:
    return {"serverContent": {"turnComplete": True}}


def interrupted() -> dict[str, Any]:
    return {"serverContent": {"interrupted": True}}


def usage(
    prompt_tokens: int, response_tokens: int, total_tokens: int
) -> dict[str, Any]:
    return {
        "usageMetadata": {
            "promptTokenCount": prompt_tokens,
            "responseTokenCount": response_tokens,
            "totalTokenCount": total_tokens,
        }
    }


def error(message: str, code: str = "invalid_argument") -> dict[str, Any]:
    return {"error": {"code": code, "message": message}}
