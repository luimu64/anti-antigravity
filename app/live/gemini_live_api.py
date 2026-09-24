"""Gemini Live API transport: true full-duplex audio over ``BidiGenerateContent``.

The cookie lane (:mod:`app.live.gemini_web_transport`) can only *fake* duplex out
of the Gemini web UI, because gemini.google.com exposes no bidi audio RPC (see
``INTERNAL_API.md`` §7c). The mobile app has no such limitation: it drives
Google's Live API directly, authenticating with the signed-in account's OAuth 2
access token rather than an API key.

Endpoint probe (2026-09-24, no credentials held locally) confirms the auth modes:

* ``v1beta.BidiGenerateContent``, no creds → 1008 *"Method doesn't allow
  unregistered callers ... Please use API Key or other form of API credential"*
* ``...BidiGenerateContent?key=<bogus>`` → 1007 *"API key not valid"*
* ``...BidiGenerateContent``, ``Bearer <bogus>`` → 1008 *"Expected OAuth 2 access
  token, login cookie or other valid authentication credential"* — the phone's
  auth mode
* ``...BidiGenerateContentConstrained`` → 1007 *"Missing or malformed auth token
  in request. Obtain one from CreateAuthToken and pass it in an `access_token`
  query parameter"* — the ephemeral-token variant

Because the server owns VAD and turn boundaries here, nothing in this module
buffers audio: chunks go upstream as they arrive and server events are handed
straight to the client. Interruption is signalled by the server
(``serverContent.interrupted``), not fabricated locally.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any
from urllib.parse import urlencode

import websockets

from app.live.protocol import LiveSetup
from app.live.transport import StreamingLiveTransport, TransportEvent

logger = logging.getLogger("google_gate.live.gemini_live_api")


def _env(name: str, default: str) -> str:
    """Read an env var, treating empty as unset.

    docker-compose passes ``VAR=${VAR:-}``, which *sets* the variable to an empty
    string — so ``os.getenv(name, default)`` would return "" and silently wipe
    the default.
    """
    return (os.getenv(name) or "").strip() or default


LIVE_WS_URL = _env(
    "GEMINI_LIVE_WS_URL",
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent",
)
# Gateway model aliases (LIVE_DEFAULT_MODEL="gemini-3.7-flash") are chat models;
# the Live API only serves native-audio / live variants. Map, don't guess.
DEFAULT_LIVE_MODEL = _env(
    "GEMINI_LIVE_MODEL", "models/gemini-2.5-flash-native-audio-preview-09-2025"
)
SETUP_TIMEOUT_S = float(_env("GEMINI_LIVE_SETUP_TIMEOUT_S", "30"))


class LiveApiError(RuntimeError):
    """Upstream refused or dropped the live socket; message is caller-safe."""


def map_model(model: str) -> str:
    """Map a gateway model alias onto a Live-capable upstream model id."""
    name = (model or "").strip()
    if not name:
        return DEFAULT_LIVE_MODEL
    bare = name.split("/", 1)[-1]
    if "native-audio" in bare or bare.endswith("-live") or "-live-" in bare:
        return f"models/{bare}"
    return DEFAULT_LIVE_MODEL


def _first_line(text: str, limit: int = 200) -> str:
    line = (text or "").strip().splitlines()[0] if text.strip() else ""
    return line[:limit]


def caller_message(exc: BaseException) -> str:
    """Turn a transport exception into a one-liner a WS client can act on."""
    text = str(exc)
    code = getattr(exc, "rcvd", None) or getattr(exc, "code", None)
    lowered = text.lower()
    if "api key not valid" in lowered:
        return "upstream: API key not valid (check GEMINI_LIVE_API_KEY)"
    if "unregistered callers" in lowered or "invalid authentication" in lowered:
        return (
            "upstream: account credentials rejected; the Live API needs an OAuth "
            "access token (or API key) for this project"
        )
    if "1007" in text and "access_token" in lowered:
        return "upstream: ephemeral token required (CreateAuthToken access_token)"
    detail = _first_line(text) or exc.__class__.__name__
    return f"upstream refused the live socket: {detail}" + (
        f" (close code {code})" if code else ""
    )


def _is_auth_failure(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "1008",
            "invalid authentication",
            "expected oauth 2 access token",
            "unregistered callers",
            "401",
            "unauthenticated",
        )
    )


class GeminiLiveApiTransport(StreamingLiveTransport):
    """One Live API socket, speaking ``BidiGenerateContent`` natively.

    Auth is whichever credential the caller can supply:

    * ``api_key`` → ``?key=`` query parameter
    * ``token_provider`` → ``Authorization: Bearer`` (an async callable returning
      a *valid* access token; the gateway's account token works here)
    * ``token_refresher`` → called once if the first attempt is an auth failure
    """

    name = "gemini_live_api"
    streaming = True

    def __init__(
        self,
        api_key: str = "",
        token_provider: Callable[[], Awaitable[str]] | None = None,
        token_refresher: Callable[[], Awaitable[Any]] | None = None,
        url: str = LIVE_WS_URL,
        model: str = "",
        setup_timeout_s: float = SETUP_TIMEOUT_S,
    ) -> None:
        self.api_key = (api_key or "").strip()
        self.token_provider = token_provider
        self.token_refresher = token_refresher
        self.url = url
        self.model = model
        self.setup_timeout_s = setup_timeout_s
        self._ws: Any = None
        self._token = ""
        #: Set when a refresh produced a token; retries use it directly, because
        #: the credential store may still hand back the stale one.
        self._forced_token = ""
        self._closed = False

    # ------------------------------------------------------------------
    # Capability
    # ------------------------------------------------------------------
    def is_available(self) -> bool:
        return bool(self.api_key or self.token_provider)

    def status(self) -> dict[str, Any]:
        return {
            "transport": self.name,
            "available": self.is_available(),
            "url": self.url,
            "auth": self._auth_mode(),
            "model": self.model or DEFAULT_LIVE_MODEL,
            "streaming": True,
        }

    def _auth_mode(self) -> str:
        if self.api_key:
            return "api_key"
        if self.token_provider:
            return "oauth_access_token"
        return "none"

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------
    async def open(
        self, setup: LiveSetup, raw_setup: dict[str, Any] | None = None
    ) -> None:
        self._closed = False
        attempt = 0
        while True:
            try:
                await self._connect()
                await self.send_frame(self.upstream_setup(setup, raw_setup))
                first = await asyncio.wait_for(self._recv_frame(), self.setup_timeout_s)
                if not isinstance(first, dict) or "setupComplete" not in first:
                    await self._teardown()
                    raise LiveApiError(
                        caller_message(ValueError(f"unexpected frame {first!r}"))
                    )
                return
            except LiveApiError:
                raise
            except Exception as exc:
                await self._teardown()
                if attempt == 0 and self._auth_refreshable(exc):
                    attempt = 1
                    self._forced_token = await self._refresh_token()
                    continue
                raise LiveApiError(caller_message(exc)) from exc

    async def close(self) -> None:
        self._closed = True
        await self._teardown()

    async def abort(self) -> None:
        """No-op: the Live API owns interruption.

        The server emits ``serverContent.interrupted`` when it detects that the
        user started talking over the model; the client's job is to stop playing
        its local buffer when it sees that frame (which the session forwards).
        """

    # ------------------------------------------------------------------
    # Frames
    # ------------------------------------------------------------------
    async def send_frame(self, frame: dict[str, Any]) -> None:
        if self._ws is None:
            raise LiveApiError("live socket is not open")
        try:
            await self._ws.send(json.dumps(frame))
        except Exception as exc:  # pragma: no cover - socket-level failure
            raise LiveApiError(caller_message(exc)) from exc

    async def events(self) -> AsyncIterator[TransportEvent]:
        """Yield upstream server frames, verbatim, until the socket closes."""
        while not self._closed and self._ws is not None:
            try:
                frame = await self._recv_frame()
            except Exception as exc:
                if not self._closed:
                    logger.warning(f"[GeminiLiveApi] socket ended: {exc}")
                    yield TransportEvent.error(caller_message(exc))
                return
            if frame is None:
                return
            if isinstance(frame, dict):
                yield TransportEvent.frame_event(frame)
            else:
                yield TransportEvent.error("upstream sent a non-object frame")

    # ------------------------------------------------------------------
    # Setup translation
    # ------------------------------------------------------------------
    def upstream_setup(
        self, setup: LiveSetup, raw_setup: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Translate the edge setup frame into the upstream Live setup frame."""
        config: dict[str, Any] = {"responseModalities": ["AUDIO"]}
        if setup.voice:
            config["speechConfig"] = {
                "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": setup.voice}}
            }
        if setup.language:
            config["speechConfig"] = config.get("speechConfig", {})
            config["speechConfig"]["languageCode"] = setup.language
        upstream: dict[str, Any] = {
            "model": self.model or map_model(setup.model),
            "generationConfig": config,
            "inputAudioTranscription": {},
            "outputAudioTranscription": {},
        }
        if setup.system_instruction:
            upstream["systemInstruction"] = {
                "parts": [{"text": setup.system_instruction}]
            }
        # Tools pass through untouched: unlike the web lane, this one supports
        # function calling natively.
        tools = (raw_setup or {}).get("tools")
        if tools:
            upstream["tools"] = tools
        return {"setup": upstream}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _url(self) -> str:
        if not self.api_key:
            return self.url
        sep = "&" if "?" in self.url else "?"
        return f"{self.url}{sep}{urlencode({'key': self.api_key})}"

    async def _connect(self) -> None:
        headers: dict[str, str] = {}
        if not self.api_key:
            if self.token_provider is None:
                raise LiveApiError(
                    "no live credentials: set GEMINI_LIVE_API_KEY or sign in an "
                    "account (OAuth access token) for this project"
                )
            self._token = self._forced_token or await self.token_provider()
            if not self._token:
                raise LiveApiError("account token unavailable; re-auth the project")
            headers["Authorization"] = f"Bearer {self._token}"
        kwargs: dict[str, Any] = {
            "max_size": None,
            "open_timeout": self.setup_timeout_s,
            "ping_interval": 20,
            "ping_timeout": 20,
        }
        try:
            self._ws = await websockets.connect(
                self._url(), additional_headers=headers, **kwargs
            )
        except TypeError:  # websockets < 14
            self._ws = await websockets.connect(
                self._url(), extra_headers=headers, **kwargs
            )

    async def _recv_frame(self) -> Any:
        if self._ws is None:
            return None
        raw = await self._ws.recv()
        if isinstance(raw, (bytes, bytearray)):
            raw = bytes(raw).decode("utf-8", "replace")
        if not isinstance(raw, str):
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"raw": raw}

    async def _teardown(self) -> None:
        ws, self._ws = self._ws, None
        if ws is None:
            return
        with contextlib.suppress(Exception):
            await ws.close()

    def _auth_refreshable(self, exc: BaseException) -> bool:
        return bool(self.token_refresher) and not self.api_key and _is_auth_failure(exc)

    async def _refresh_token(self) -> str:
        """Refresh the account token, returning it (or "" if it failed)."""
        if self.token_refresher is None:
            return ""
        try:
            token = await self.token_refresher()
        except Exception as exc:
            logger.warning(f"[GeminiLiveApi] token refresh failed: {exc}")
            return ""
        return token if isinstance(token, str) and token.strip() else ""
