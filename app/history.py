import threading
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Any


class QueryHistoryManager:
    """In-memory rolling buffer tracking up to 50 recent API queries."""

    def __init__(self, max_records: int = 50):
        self._max_records = max_records
        self._buffer: deque[dict[str, Any]] = deque(maxlen=max_records)
        self._lock = threading.Lock()

    def record(
        self,
        model: str,
        resolved_model: str,
        backend: str,
        duration_ms: float,
        status: str = "success",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        error_message: str | None = None,
        request_id: str | None = None,
        timestamp: str | None = None,
    ) -> dict[str, Any]:
        """Record a query entry into the rolling history buffer."""
        entry = {
            "id": request_id or f"req_{uuid.uuid4().hex[:12]}",
            "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
            "status": status,
            "model": model,
            "resolved_model": resolved_model,
            "backend": backend,
            "duration_ms": round(duration_ms, 2),
            "prompt_tokens": prompt_tokens or 0,
            "completion_tokens": completion_tokens or 0,
            "total_tokens": total_tokens or 0,
            "error_message": error_message,
        }
        with self._lock:
            self._buffer.appendleft(entry)
        return entry

    # Aliases
    add = record

    def list_history(self) -> list[dict[str, Any]]:
        """Return all query records in reverse chronological order."""
        with self._lock:
            return list(self._buffer)

    # Aliases
    get_history = list_history

    def clear(self) -> None:
        """Clear all stored query records."""
        with self._lock:
            self._buffer.clear()

    # Aliases
    clear_history = clear

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)


history_manager = QueryHistoryManager()
