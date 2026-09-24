"""Transport-agnostic Live session state machine.

Owns the parts of a live voice session that do not depend on the upstream lane:
turn assembly (``activityStart`` / ``activityEnd`` / bare ``mediaChunks`` silence
flush), barge-in, transcript bookkeeping and usage accounting. The transport only
has to answer "here is what the user said (audio/text) — here is what the model
produced (audio/text)".

Concurrency model: one turn runs at a time. Speech detected while a turn is in
flight is a *barge-in*: the in-flight turn is cancelled, the client is told
(``serverContent.interrupted``), upstream generation is aborted, and the new
utterance becomes the next turn.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from collections.abc import Awaitable, Callable
from typing import Any

from app.live import protocol as p
from app.live.transport import LiveTransport, TurnRequest

logger = logging.getLogger("google_gate.live.session")

# Clients that never send activityEnd (older Live clients stream bare
# mediaChunks) still need turn boundaries: flush after this much quiet.
SILENCE_FLUSH_S = float(os.getenv("LIVE_SILENCE_FLUSH_S", "1.5"))
# Reject a media chunk that is obviously not a 20-100 ms frame.
MAX_MEDIA_CHUNK_BYTES = p.MAX_MEDIA_CHUNK_BYTES
MAX_TURN_AUDIO_BYTES = p.MAX_TURN_AUDIO_BYTES

EmitFn = Callable[[dict[str, Any]], Awaitable[None]]


class LiveSessionError(RuntimeError):
    """A session-level failure (setup violated, lane unavailable, ...)."""


class LiveSession:
    """One WebSocket client, one conversation, one upstream lane."""

    def __init__(
        self,
        transport: LiveTransport,
        emit: EmitFn,
        silence_flush_s: float = SILENCE_FLUSH_S,
    ) -> None:
        self.transport = transport
        self._emit = emit
        self.silence_flush_s = silence_flush_s
        self.setup: p.LiveSetup | None = None
        self._audio = bytearray()
        self._activity_open = False
        self._turn_task: asyncio.Task | None = None
        self._flush_timer: asyncio.Task | None = None
        self._closed = False
        self._turns = 0
        self._interruptions = 0
        self._prompt_tokens = 0
        self._response_tokens = 0
        self._last_error = ""
        self._last_reply = ""
        self._transcript: list[str] = []
        self._suppress_events = False

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def stats(self) -> dict[str, Any]:
        return {
            "turns": self._turns,
            "interruptions": self._interruptions,
            "prompt_tokens": self._prompt_tokens,
            "response_tokens": self._response_tokens,
            "buffered_audio_bytes": len(self._audio),
            "generating": self._turn_task is not None and not self._turn_task.done(),
            "last_error": self._last_error,
            "transport": self.transport.name,
        }

    @property
    def transcript(self) -> str:
        return "\n".join(self._transcript[-20:])

    # ------------------------------------------------------------------
    # Client frames
    # ------------------------------------------------------------------
    async def handle(self, frame: Any) -> None:
        """Dispatch one decoded client frame."""
        if not isinstance(frame, dict):
            await self._send(p.error("frame must be a JSON object", "invalid_argument"))
            return

        if "setup" in frame:
            await self._handle_setup(frame["setup"])
            return

        if self.setup is None:
            await self._send(
                p.error("setup frame required first", "failed_precondition")
            )
            return

        if "realtimeInput" in frame or "realtime_input" in frame:
            await self._handle_realtime(
                frame.get("realtimeInput") or frame.get("realtime_input")
            )
            return
        if "clientContent" in frame or "client_content" in frame:
            await self._handle_client_content(
                frame.get("clientContent") or frame.get("client_content")
            )
            return
        if "toolResponse" in frame or "tool_response" in frame:
            await self._send(
                p.error(
                    "tool calling is not available on this lane: the Gemini web "
                    "backend exposes no function-calling wire",
                    "unimplemented",
                )
            )
            return
        await self._send(p.error("unrecognised frame", "invalid_argument"))

    async def _handle_setup(self, raw: Any) -> None:
        if self.setup is not None:
            await self._send(p.error("session already set up", "failed_precondition"))
            return
        try:
            setup = p.parse_setup(raw)
        except p.LiveMessageError as exc:
            await self._send(p.error(str(exc), "invalid_argument"))
            return

        if not self.transport.is_available():
            await self._send(
                p.error(
                    f"transport '{self.transport.name}' is not configured "
                    "(no browser profile / credentials)",
                    "unavailable",
                )
            )
            return

        try:
            await self.transport.open(setup)
        except Exception as exc:
            self._last_error = str(exc)
            logger.warning(f"[live] transport open failed: {exc}")
            await self._send(p.error(f"transport unavailable: {exc}", "unavailable"))
            return

        self.setup = setup
        if setup.tools_requested:
            logger.info(
                "[live] tools requested but unsupported on the %s lane; "
                "continuing without function calling",
                self.transport.name,
            )
        await self._send(p.setup_complete())

    # -- realtime input -------------------------------------------------
    async def _handle_realtime(self, block: Any) -> None:
        if not isinstance(block, dict):
            await self._send(p.error("realtimeInput must be an object"))
            return

        if (
            block.get("activityStart") is not None
            or block.get("activity_start") is not None
        ):
            await self._begin_activity()
        if (
            block.get("activityEnd") is not None
            or block.get("activity_end") is not None
        ):
            await self._end_activity()
        chunks = (
            block.get("mediaChunks")
            or block.get("media_chunks")
            or block.get("audio")
            or []
        )
        for chunk in chunks if isinstance(chunks, list) else []:
            await self._append_media(chunk)

    async def _append_media(self, chunk: Any) -> None:
        if not isinstance(chunk, dict):
            return
        data = p.decode_base64(chunk.get("data"))
        if not data:
            return
        if len(data) > MAX_MEDIA_CHUNK_BYTES:
            await self._send(
                p.error(
                    f"media chunk too large ({len(data)} bytes > {MAX_MEDIA_CHUNK_BYTES}); "
                    "stream ~20-100 ms frames",
                    "invalid_argument",
                )
            )
            return
        # A client that streams audio without activity markers is still
        # speaking: arm the silence flush so the turn eventually closes.
        if not self._activity_open and not self._audio:
            await self._cancel_turn_for_barge_in(reason="speech detected")
        self._audio.extend(data)
        if len(self._audio) > MAX_TURN_AUDIO_BYTES:
            logger.info("[live] utterance exceeded buffer cap; flushing turn")
            await self._flush_turn()
            return
        self._arm_flush_timer()

    async def _begin_activity(self) -> None:
        if self._activity_open:
            return
        self._activity_open = True
        self._cancel_flush_timer()
        if self._audio:
            # Re-arm mid-utterance: keep the buffer, just resume.
            return
        await self._cancel_turn_for_barge_in(reason="user started speaking")

    async def _end_activity(self) -> None:
        self._activity_open = False
        self._cancel_flush_timer()
        await self._flush_turn()

    async def _handle_client_content(self, block: Any) -> None:
        if not isinstance(block, dict):
            await self._send(p.error("clientContent must be an object"))
            return
        turns = block.get("turns") or []
        text_parts: list[str] = []
        if isinstance(turns, list):
            for turn in turns:
                if isinstance(turn, dict):
                    text_parts.append(p._parts_text(turn.get("parts")))
        text = "\n".join(part for part in text_parts if part).strip()
        if not text and not self._audio:
            await self._send(
                p.error("clientContent carried no text", "invalid_argument")
            )
            return
        await self._cancel_turn_for_barge_in(reason="new client content")
        await self._start_turn(prompt=text)

    # ------------------------------------------------------------------
    # Turn lifecycle
    # ------------------------------------------------------------------
    def _arm_flush_timer(self) -> None:
        if self.silence_flush_s <= 0:
            return
        self._cancel_flush_timer()
        self._flush_timer = asyncio.create_task(self._silence_flush())

    def _cancel_flush_timer(self) -> None:
        if self._flush_timer and not self._flush_timer.done():
            self._flush_timer.cancel()
        self._flush_timer = None

    async def _silence_flush(self) -> None:
        try:
            await asyncio.sleep(self.silence_flush_s)
            if self._activity_open:
                return  # explicit activityEnd owns the boundary
            await self._flush_turn()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(f"[live] silence flush failed: {exc}")

    async def _flush_turn(self) -> None:
        audio = bytes(self._audio)
        self._audio.clear()
        if not audio:
            return
        await self._start_turn(audio=audio)

    async def _cancel_turn_for_barge_in(self, reason: str) -> None:
        """Stop an in-flight turn because the user spoke (or sent content).

        Ordering matters: the client must see ``interrupted``, and must NOT see
        a ``turnComplete``/usage pair for a turn that was cut off — so events
        from the cancelled turn are suppressed ahead of the cancel.
        """
        if self._turn_task is None or self._turn_task.done():
            self._turn_task = None
            return
        self._interruptions += 1
        logger.info(f"[live] barge-in: {reason}")
        self._suppress_events = True
        with contextlib.suppress(Exception):
            await self._send(p.interrupted())
        self._turn_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await self._turn_task
        self._turn_task = None
        self._suppress_events = False
        with contextlib.suppress(Exception):
            await self.transport.abort()

    async def _start_turn(self, prompt: str = "", audio: bytes = b"") -> None:
        if self.setup is None:
            return
        request = TurnRequest(
            prompt=prompt,
            audio=audio,
            audio_rate=self.setup.input_audio_rate,
            system_instruction=self.setup.system_instruction,
            voice=self.setup.voice,
            language=self.setup.language,
            first_turn=self._turns == 0,
        )
        self._turn_task = asyncio.create_task(self._run_turn(request))

    async def _run_turn(self, request: TurnRequest) -> None:
        """Consume the transport's events and translate them to Live frames."""
        try:
            async for event in self.transport.run_turn(request):
                if self._closed:
                    return
                if event.kind == "text":
                    if event.text:
                        self._last_reply = event.text
                        self._transcript.append(event.text)
                        await self._send(p.output_transcription(event.text))
                        await self._send(p.server_text(event.text))
                elif event.kind == "audio":
                    if event.audio:
                        await self._send(p.server_audio(event.audio, event.mime))
                elif event.kind == "error":
                    self._last_error = event.message
                    logger.warning(f"[live] transport error: {event.message}")
                    await self._send(p.error(event.message, "unavailable"))
                elif event.kind == "done":
                    usage = event.usage or {}
                    self._prompt_tokens += int(usage.get("promptTokenCount") or 0)
                    self._response_tokens += int(usage.get("responseTokenCount") or 0)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._last_error = str(exc)
            logger.warning(f"[live] turn failed: {exc}")
            await self._send(p.error(f"turn failed: {exc}", "internal"))
        finally:
            self._turns += 1
            if not self._closed and not self._suppress_events:
                with contextlib.suppress(Exception):
                    await self._send(p.turn_complete())
                with contextlib.suppress(Exception):
                    await self._send(
                        p.usage(
                            self._prompt_tokens,
                            self._response_tokens,
                            self._prompt_tokens + self._response_tokens,
                        )
                    )

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------
    async def close(self) -> None:
        self._closed = True
        self._cancel_flush_timer()
        if self._turn_task and not self._turn_task.done():
            self._turn_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._turn_task
        self._turn_task = None
        with contextlib.suppress(Exception):
            await self.transport.close()

    async def _send(self, frame: dict[str, Any]) -> None:
        if self._closed:
            return
        await self._emit(frame)
