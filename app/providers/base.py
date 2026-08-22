import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import AsyncGenerator
from typing import Any

logger = logging.getLogger("google_gate.providers.base")


class RateLimitError(Exception):
    """Raised when an upstream provider returns 429 Too Many Requests or quota exceeded."""

    def __init__(
        self,
        message: str,
        status_code: int = 429,
        retry_after: float = 60.0,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


class ModelNotFoundError(Exception):
    """Raised when a requested model is not found, deprecated, or not supported by any configured backend."""

    def __init__(
        self,
        message: str,
        model: str = "",
        status_code: int = 404,
    ):
        super().__init__(message)
        self.model = model
        self.status_code = status_code


class InMemoryRateTracker:
    """
    In-memory proactive sliding-window rate limit tracker per running process.
    Tracks requests per minute (RPM), tokens per minute (TPM), and requests per day (RPD).
    """

    def __init__(self, rpm: int = 0, tpm: int = 0, rpd: int = 0):
        self.rpm = rpm
        self.tpm = tpm
        self.rpd = rpd
        self._minute_requests: deque[float] = deque()
        self._minute_tokens: deque[tuple[float, int]] = deque()
        self._day_requests: deque[float] = deque()

    def _prune(self, now: float | None = None) -> None:
        current = now if now is not None else time.time()
        minute_cutoff = current - 60.0
        day_cutoff = current - 86400.0

        while self._minute_requests and self._minute_requests[0] <= minute_cutoff:
            self._minute_requests.popleft()

        while self._minute_tokens and self._minute_tokens[0][0] <= minute_cutoff:
            self._minute_tokens.popleft()

        while self._day_requests and self._day_requests[0] <= day_cutoff:
            self._day_requests.popleft()

    def has_capacity(self, estimated_tokens: int = 0, now: float | None = None) -> bool:
        """Check if provider has capacity within configured proactive rate limit windows."""
        current = now if now is not None else time.time()
        self._prune(current)

        if self.rpm > 0 and len(self._minute_requests) >= self.rpm:
            return False

        if self.tpm > 0:
            current_tokens = sum(t[1] for t in self._minute_tokens)
            if current_tokens + max(0, estimated_tokens) > self.tpm:
                return False

        return not (self.rpd > 0 and len(self._day_requests) >= self.rpd)

    def get_window_reset_remaining(self, now: float | None = None) -> float:
        """Return seconds until capacity frees up in the current sliding window."""
        current = now if now is not None else time.time()
        self._prune(current)

        waits = []
        if (
            self.rpm > 0
            and len(self._minute_requests) >= self.rpm
            and self._minute_requests
        ):
            waits.append(max(0.1, 60.0 - (current - self._minute_requests[0])))

        if self.tpm > 0 and self._minute_tokens:
            current_tokens = sum(t[1] for t in self._minute_tokens)
            if current_tokens >= self.tpm:
                waits.append(max(0.1, 60.0 - (current - self._minute_tokens[0][0])))

        if self.rpd > 0 and len(self._day_requests) >= self.rpd and self._day_requests:
            waits.append(max(0.1, 86400.0 - (current - self._day_requests[0])))

        return max(waits) if waits else 0.0

    def record_usage(self, tokens: int = 0, now: float | None = None) -> None:
        """Record a completed or in-flight request and token consumption."""
        current = now if now is not None else time.time()
        self._prune(current)
        self._minute_requests.append(current)
        if tokens > 0 or self.tpm > 0:
            self._minute_tokens.append((current, max(1, tokens)))
        self._day_requests.append(current)

    def reset(self) -> None:
        """Reset all in-memory counters."""
        self._minute_requests.clear()
        self._minute_tokens.clear()
        self._day_requests.clear()

    def get_stats(self, now: float | None = None) -> dict[str, Any]:
        """Return snapshot of current rate limit tracking state."""
        current = now if now is not None else time.time()
        self._prune(current)
        return {
            "rpm_limit": self.rpm,
            "rpm_used": len(self._minute_requests),
            "tpm_limit": self.tpm,
            "tpm_used": sum(t[1] for t in self._minute_tokens),
            "rpd_limit": self.rpd,
            "rpd_used": len(self._day_requests),
            "has_capacity": self.has_capacity(now=current),
            "reset_remaining": round(self.get_window_reset_remaining(now=current), 1),
        }


class BaseAdapter(ABC):
    """Abstract base class for all backend adapters."""

    name: str = "base"
    default_cooldown: float = 60.0
    min_quota_fraction: float = 0.0

    def __init__(
        self,
        enabled: bool = False,
        rpm: int = 0,
        tpm: int = 0,
        rpd: int = 0,
        default_cooldown: float = 60.0,
        min_quota_fraction: float = 0.0,
        model_cache_ttl: float = 300.0,
    ):
        self.enabled: bool = enabled
        self.cooldown_until: float = 0.0
        self.default_cooldown = default_cooldown
        self.min_quota_fraction = min_quota_fraction
        self.model_cache_ttl = model_cache_ttl
        self.rate_limiter = InMemoryRateTracker(rpm=rpm, tpm=tpm, rpd=rpd)
        self._cached_models: dict[str, Any] | None = None
        self._models_fetched_at: float = 0.0

    @abstractmethod
    def is_configured(self) -> bool:
        """Return True if required credentials or configuration are present."""

    def is_available(self, estimated_tokens: int = 0) -> bool:
        """Return True if backend is enabled, configured, not in cooldown, and has proactive capacity."""
        if not (self.enabled and self.is_configured()):
            return False
        if time.time() < self.cooldown_until:
            return False
        return not (
            hasattr(self, "rate_limiter")
            and not self.rate_limiter.has_capacity(estimated_tokens)
        )

    def set_cooldown(self, seconds: float | None = None) -> None:
        """Mark backend as cooled down due to rate limits or errors."""
        duration = seconds if seconds is not None else self.default_cooldown
        self.cooldown_until = time.time() + duration
        logger.warning(
            f"Backend '{self.name}' entered cooldown for {duration:.1f}s until {self.cooldown_until:.1f}"
        )

    def clear_cooldown(self) -> None:
        """Clear cooldown status and reset in-memory proactive rate tracker."""
        self.cooldown_until = 0.0
        if hasattr(self, "rate_limiter"):
            self.rate_limiter.reset()

    def get_cooldown_remaining(self) -> float:
        """Return remaining cooldown in seconds, or window reset if rate-limited, else 0."""
        cooldown_wait = max(0.0, self.cooldown_until - time.time())
        if cooldown_wait > 0:
            return cooldown_wait
        if hasattr(self, "rate_limiter") and not self.rate_limiter.has_capacity():
            return self.rate_limiter.get_window_reset_remaining()
        return 0.0

    def get_rate_limit_quotas(self) -> list[dict[str, Any]]:
        """Return normalized rate limit and cooldown quota items for this adapter."""
        if not hasattr(self, "rate_limiter"):
            return []
        items: list[dict[str, Any]] = []
        now = time.time()
        cooldown = self.get_cooldown_remaining()
        rl = self.rate_limiter
        rl._prune(now)

        if rl.rpm > 0:
            used = len(rl._minute_requests)
            frac_used = (
                1.0 if cooldown > 0 else min(1.0, max(0.0, round(used / rl.rpm, 4)))
            )
            rem_frac = max(0.0, min(1.0, round(1.0 - frac_used, 4)))
            reset_secs = (
                round(cooldown, 1)
                if cooldown > 0
                else (
                    max(0.0, round(60.0 - (now - rl._minute_requests[0]), 1))
                    if rl._minute_requests
                    else 0.0
                )
            )
            items.append(
                {
                    "display_name": "Requests Per Minute (RPM)",
                    "fraction_used": frac_used,
                    "remaining_fraction": rem_frac,
                    "fraction_remaining": rem_frac,
                    "reset_time_seconds": reset_secs,
                    "model_id": self.name,
                    "backend": self.name,
                    "source": self.name,
                }
            )

        if rl.tpm > 0:
            used = sum(t[1] for t in rl._minute_tokens)
            frac_used = (
                1.0 if cooldown > 0 else min(1.0, max(0.0, round(used / rl.tpm, 4)))
            )
            rem_frac = max(0.0, min(1.0, round(1.0 - frac_used, 4)))
            reset_secs = (
                round(cooldown, 1)
                if cooldown > 0
                else (
                    max(0.0, round(60.0 - (now - rl._minute_tokens[0][0]), 1))
                    if rl._minute_tokens
                    else 0.0
                )
            )
            items.append(
                {
                    "display_name": "Tokens Per Minute (TPM)",
                    "fraction_used": frac_used,
                    "remaining_fraction": rem_frac,
                    "fraction_remaining": rem_frac,
                    "reset_time_seconds": reset_secs,
                    "model_id": self.name,
                    "backend": self.name,
                    "source": self.name,
                }
            )

        if rl.rpd > 0:
            used = len(rl._day_requests)
            frac_used = (
                1.0 if cooldown > 0 else min(1.0, max(0.0, round(used / rl.rpd, 4)))
            )
            rem_frac = max(0.0, min(1.0, round(1.0 - frac_used, 4)))
            reset_secs = (
                round(cooldown, 1)
                if cooldown > 0
                else (
                    max(0.0, round(86400.0 - (now - rl._day_requests[0]), 1))
                    if rl._day_requests
                    else 0.0
                )
            )
            items.append(
                {
                    "display_name": "Requests Per Day (RPD)",
                    "fraction_used": frac_used,
                    "remaining_fraction": rem_frac,
                    "fraction_remaining": rem_frac,
                    "reset_time_seconds": reset_secs,
                    "model_id": self.name,
                    "backend": self.name,
                    "source": self.name,
                }
            )

        if not items and cooldown > 0:
            items.append(
                {
                    "display_name": "Cooldown",
                    "fraction_used": 1.0,
                    "remaining_fraction": 0.0,
                    "fraction_remaining": 0.0,
                    "reset_time_seconds": round(cooldown, 1),
                    "model_id": self.name,
                    "backend": self.name,
                    "source": self.name,
                }
            )

        return items

    def record_usage(self, tokens: int = 0) -> None:
        """Record proactive usage."""
        if hasattr(self, "rate_limiter"):
            self.rate_limiter.record_usage(tokens)

    @abstractmethod
    async def generate_content(
        self,
        model: str,
        contents: list[dict[str, Any]],
        system_instruction: dict[str, Any] | None = None,
        generation_config: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Generate complete content response non-streaming."""

    @abstractmethod
    async def stream_generate_content(
        self,
        model: str,
        contents: list[dict[str, Any]],
        system_instruction: dict[str, Any] | None = None,
        generation_config: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream generated content chunks."""

    @abstractmethod
    async def fetch_available_models(
        self, force_refresh: bool = False
    ) -> dict[str, Any]:
        """Fetch models supported by this adapter."""

    async def embed_contents(
        self, model: str, texts: list[str], dimensions: int | None = None
    ) -> dict[str, Any]:
        """Generate text embeddings (optional support)."""
        raise NotImplementedError(f"Backend '{self.name}' does not support embeddings.")
