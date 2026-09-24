"""Transport contract for live voice sessions.

A transport owns one upstream lane and implements a *turn*: give it the audio
the user just spoke (and/or text) and it yields what the model said back, as
text and/or audio. :class:`app.live.session.LiveSession` owns everything above
that line — the Live API schema, turn assembly and barge-in — so a second lane
(AI Studio web, an API-key Live socket, ...) only has to implement this
interface.
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from app.live.protocol import LiveSetup


@dataclass
class TurnRequest:
    """One user turn: what the user said, as audio and/or text."""

    prompt: str = ""
    audio: bytes = b""
    audio_rate: int = 16000
    system_instruction: str = ""
    voice: str = ""
    language: str = ""
    first_turn: bool = False


class TransportEvent:
    """A typed event yielded by :meth:`LiveTransport.run_turn`."""

    __slots__ = ("kind", "text", "audio", "mime", "usage", "message")

    def __init__(
        self,
        kind: str,
        text: str = "",
        audio: bytes = b"",
        mime: str = "",
        usage: dict[str, Any] | None = None,
        message: str = "",
    ) -> None:
        self.kind = kind
        self.text = text
        self.audio = audio
        self.mime = mime
        self.usage = usage or {}
        self.message = message

    @classmethod
    def text_event(cls, text: str) -> TransportEvent:
        return cls("text", text=text)

    @classmethod
    def audio_event(cls, audio: bytes, mime: str) -> TransportEvent:
        return cls("audio", audio=audio, mime=mime)

    @classmethod
    def done(cls, usage: dict[str, Any] | None = None) -> TransportEvent:
        return cls("done", usage=usage)

    @classmethod
    def error(cls, message: str) -> TransportEvent:
        return cls("error", message=message)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        detail = (
            self.text or self.message or (f"{len(self.audio)}B" if self.audio else "")
        )
        return f"<TransportEvent {self.kind} {detail!r}>"


class LiveTransport(abc.ABC):
    """One upstream lane, one serialised session at a time."""

    name: str = "transport"

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Whether this lane can serve a session right now (credentials present)."""

    @abc.abstractmethod
    async def open(self, setup: LiveSetup) -> None:
        """Bring the lane up for a session (browser/page, socket, ...)."""

    @abc.abstractmethod
    def run_turn(self, request: TurnRequest) -> AsyncIterator[TransportEvent]:
        """Yield what the model produced for one user turn.

        Implementations are async generators; the session consumes them and
        cancels the iterator on barge-in.
        """

    @abc.abstractmethod
    async def abort(self) -> None:
        """Stop the in-flight model turn as fast as the lane allows."""

    @abc.abstractmethod
    async def close(self) -> None:
        """Release session-scoped resources (the lane may stay warm)."""

    def status(self) -> dict[str, Any]:  # pragma: no cover - overridden
        return {"transport": self.name, "available": self.is_available()}
