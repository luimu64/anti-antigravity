"""
Structured telemetry & logging for the Google Gate routing system.

Every log line carries enough context to answer, for a real deployment:
  * which request was affected (request_id correlation across modules),
  * which backends were considered and WHY one was picked or skipped
    (enabled / configured / capable / cooldown / rate-capacity states),
  * what happened on every attempt (latency, tokens, error class),
  * how the request ended (success / fallback chain / exhausted / error).

Configuration (env vars):
  LOG_LEVEL   - DEBUG | INFO | WARNING | ERROR   (default INFO)
  LOG_FORMAT  - json | text                      (default text)
  LOG_FILE    - optional path; also writes JSON lines there with rotation.
"""

import contextvars
import json
import logging
import logging.handlers
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_REQUEST_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "google_gate_request_id", default=None
)

_MAX_FIELD_LEN = 512


def new_request_id() -> str:
    """Generate a fresh correlation id for an incoming request."""
    return f"req_{os.urandom(6).hex()}"


def bind_request_id(request_id: str | None) -> None:
    """Bind the correlation id to the current execution context."""
    _REQUEST_ID.set(request_id)


def get_request_id() -> str | None:
    return _REQUEST_ID.get()


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, str) and len(value) > _MAX_FIELD_LEN:
        return value[:_MAX_FIELD_LEN] + "...(truncated)"
    if value is not None and not isinstance(value, (int, bool, str, float)):
        return str(value)
    return value


class RequestIdFilter(logging.Filter):
    """Attach the current correlation id to every record passing through."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = _REQUEST_ID.get()
        return True


class JsonFormatter(logging.Formatter):
    """Single-line JSON output, ideal for ingestion by Loki/ELK/Datadog etc."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        rid = getattr(record, "request_id", None)
        if rid:
            payload["request_id"] = rid
        event = getattr(record, "event", None)
        if event:
            payload["event"] = event
        fields = getattr(record, "fields", None)
        if fields:
            payload.update(_sanitize(fields))
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class TextFormatter(logging.Formatter):
    """Human-friendly console format with event + structured field appendix.

    The asctime prefix is dropped when the output stream is not a TTY
    (Docker, journald, process managers), because those collectors already
    prepend their own timestamp to every captured line - keeping ours
    would log every date twice.
    """

    def __init__(self, show_time: bool | None = None):
        if show_time is None:
            show_time = sys.stderr.isatty()
        fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        datefmt: str | None = "%Y-%m-%d %H:%M:%S"
        if not show_time:
            fmt = "[%(levelname)s] %(name)s: %(message)s"
            datefmt = None
        super().__init__(fmt, datefmt=datefmt)

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = []
        rid = getattr(record, "request_id", None)
        if rid:
            extras.append(f"request_id={rid}")
        event = getattr(record, "event", None)
        if event:
            extras.append(f"event={event}")
        fields = getattr(record, "fields", None)
        if fields:
            try:
                extras.append(json.dumps(_sanitize(fields), default=str))
            except Exception:
                extras.append(str(fields))
        if record.exc_info and not record.getMessage():
            base += "\n" + self.formatException(record.exc_info)
        return base + (" | " + " ".join(extras) if extras else "")


def log_event(
    logger: logging.Logger,
    level: int,
    event: str,
    msg: str = "",
    **fields: Any,
) -> None:
    """Emit a structured routing-telemetry event."""
    if not logger.isEnabledFor(level):
        return
    logger.log(
        level,
        msg or event,
        extra={"event": event, "fields": fields},
    )


def setup_logging() -> None:
    """
    Configure the 'google_gate' logging namespace.

    Idempotent: safe to call multiple times (module reloads in tests).
    All google_gate records flow through our handlers only (propagate=False)
    so JSON lines are never duplicated by root handlers.
    """
    root = logging.getLogger("google_gate")
    if getattr(root, "_telemetry_configured", False):
        return

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    fmt = os.getenv("LOG_FORMAT", "text").lower()
    log_file = os.getenv("LOG_FILE", "")

    root.setLevel(level)
    root.propagate = False

    req_filter = RequestIdFilter()
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.addFilter(req_filter)
    console_handler.setFormatter(JsonFormatter() if fmt == "json" else TextFormatter())
    root.addHandler(console_handler)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            path, maxBytes=50 * 1024 * 1024, backupCount=5
        )
        file_handler.setLevel(level)
        file_handler.addFilter(req_filter)
        file_handler.setFormatter(JsonFormatter())
        root.addHandler(file_handler)

    root._telemetry_configured = True  # type: ignore[attr-defined]
