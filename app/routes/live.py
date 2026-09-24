"""Live voice session endpoint (``/v1/live``).

Speaks Google's Live API (``BidiGenerateContent``) over a WebSocket so a
Live-capable client can point at the gateway instead of Google:

    ws://<gateway>/v1/live?key=<gateway key>

Auth is the same bridge-key registry as ``/v1/*``: the key may arrive as a
``Authorization: Bearer`` header, an ``x-goog-api-key`` header, or a ``key``/
``api_key`` query parameter (browser-side Live clients cannot set headers, and
the vendor SDK's websocket path puts credentials in a header or query).

The upstream lane is pluggable (see :mod:`app.live.transport`); the cookie lane
(``gemini_web``) drives the Gemini web app in a persistent signed-in browser.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from app.history import history_manager
from app.keys import api_key_manager
from app.live.gemini_web_transport import GeminiWebLiveTransport
from app.live.session import LiveSession
from app.telemetry import log_event

logger = logging.getLogger("google_gate.live")
telemetry_logger = logging.getLogger("google_gate.requests")

router = APIRouter(tags=["Live"])

# One transport instance per process: the lane owns a single browser tab, and
# turn execution is serialised inside the transport.
_transport = GeminiWebLiveTransport()

# Close codes: 4401 = unauthorized, 4408 = setup timeout, 4400 = bad request.
CLOSE_UNAUTHORIZED = 4401
CLOSE_SETUP_TIMEOUT = 4408


def get_transport() -> GeminiWebLiveTransport:
    """Return the process-wide live transport (overridable in tests)."""
    return _transport


def _extract_key(
    websocket: WebSocket, query_key: str | None, header_key: str | None
) -> str:
    query = (
        query_key
        or websocket.query_params.get("key")
        or websocket.query_params.get("api_key")
    )
    if query:
        return str(query)
    if header_key:
        return str(header_key)
    authorization = websocket.headers.get("authorization") or ""
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() == "bearer" and token:
        return token
    return ""


async def _authorize(websocket: WebSocket, key: str) -> tuple[bool, str]:
    if not api_key_manager.enforce_keys:
        return True, ""
    if not key:
        return False, (
            "missing API key: pass it as the 'key' query parameter, or in an "
            "Authorization: Bearer / x-goog-api-key header"
        )
    if not api_key_manager.validate_key(key):
        return False, f"incorrect API key: {key[:8]}***"
    return True, ""


@router.websocket("/v1/live")
async def live_session(
    websocket: WebSocket,
    key: str | None = Query(default=None, alias="key"),
    api_key: str | None = Query(default=None, alias="api_key"),
) -> None:
    """Bidirectional voice session: audio in, audio out, barge-in."""
    requested = websocket.headers.get("sec-websocket-protocol") or ""
    subprotocols = [p.strip() for p in requested.split(",") if p.strip()]
    await websocket.accept(subprotocol=subprotocols[0] if subprotocols else None)

    authorized, reason = await _authorize(
        websocket,
        _extract_key(
            websocket, key or api_key, websocket.headers.get("x-goog-api-key")
        ),
    )
    if not authorized:
        await websocket.send_json(
            {"error": {"code": "unauthorized", "message": reason}}
        )
        # Give the frame a chance to flush before the close frame races it.
        await asyncio.sleep(0.05)
        await websocket.close(code=CLOSE_UNAUTHORIZED, reason=reason[:120])
        return

    transport = get_transport()

    async def emit(frame: dict) -> None:
        await websocket.send_json(frame)

    session = LiveSession(transport, emit)
    started = time.monotonic()
    client = websocket.client.host if websocket.client else "unknown"
    log_event(
        logger,
        logging.INFO,
        "live.session.open",
        "Live voice session opened",
        client_host=client,
        transport=transport.name,
    )

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                frame = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                await websocket.send_json(
                    {
                        "error": {
                            "code": "invalid_argument",
                            "message": "frames must be JSON",
                        }
                    }
                )
                continue
            await session.handle(frame)
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # pragma: no cover - transport-level failure
        logger.warning(f"[live] session error: {exc}")
    finally:
        stats = session.stats
        await session.close()
        duration_ms = (time.monotonic() - started) * 1000
        status = "error" if stats["turns"] == 0 and stats["last_error"] else "success"
        history_manager.record(
            model=(session.setup.model if session.setup else "gemini-live"),
            resolved_model=(session.setup.model if session.setup else "gemini-live"),
            backend=transport.name,
            duration_ms=duration_ms,
            status=status,
            prompt_tokens=stats["prompt_tokens"],
            completion_tokens=stats["response_tokens"],
            total_tokens=stats["prompt_tokens"] + stats["response_tokens"],
            error_message=stats["last_error"] or None,
        )
        log_event(
            logger,
            logging.INFO,
            "live.session.close" if status == "success" else "live.session.error",
            "Live voice session closed",
            client_host=client,
            transport=transport.name,
            duration_ms=round(duration_ms, 2),
            turns=stats["turns"],
            interruptions=stats["interruptions"],
            prompt_tokens=stats["prompt_tokens"],
            response_tokens=stats["response_tokens"],
            error=stats["last_error"] or None,
        )


@router.get("/v1/live/status")
async def live_status() -> dict:
    """Report the live lane's readiness (profile present, last error, turns)."""
    transport = get_transport()
    status = transport.status()
    status["endpoint"] = "/v1/live"
    status["schema"] = "bidiGenerateContent"
    status["audio"] = {
        "input": "audio/pcm;rate=16000",
        "output": "audio/pcm;rate=24000",
    }
    status["capabilities"] = {
        "voice_in": True,
        "voice_out": True,
        "barge_in": True,
        "tools": False,
    }
    return status
