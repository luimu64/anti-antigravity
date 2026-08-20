import time
import logging
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Dict, Any, List, Optional

logger = logging.getLogger("agy_to_api.providers.base")

class RateLimitError(Exception):
    """Raised when an upstream provider returns 429 Too Many Requests or quota exceeded."""
    def __init__(self, message: str, status_code: int = 429, retry_after: float = 60.0):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after

class BaseAdapter(ABC):
    """Abstract base class for all backend adapters."""
    name: str = "base"
    default_cooldown: float = 60.0

    def __init__(self, enabled: bool = False):
        self.enabled: bool = enabled
        self.cooldown_until: float = 0.0

    @abstractmethod
    def is_configured(self) -> bool:
        """Return True if required credentials or configuration are present."""
        pass

    def is_available(self) -> bool:
        """Return True if backend is enabled, configured, and not currently in cooldown."""
        return self.enabled and self.is_configured() and (time.time() >= self.cooldown_until)

    def set_cooldown(self, seconds: Optional[float] = None) -> None:
        """Mark backend as cooled down due to rate limits or errors."""
        duration = seconds if seconds is not None else self.default_cooldown
        self.cooldown_until = time.time() + duration
        logger.warning(f"Backend '{self.name}' entered cooldown for {duration}s until {self.cooldown_until:.1f}")

    def clear_cooldown(self) -> None:
        """Clear cooldown status immediately."""
        self.cooldown_until = 0.0

    def get_cooldown_remaining(self) -> float:
        """Return remaining cooldown in seconds, or 0 if not cooling down."""
        return max(0.0, self.cooldown_until - time.time())

    @abstractmethod
    async def generate_content(
        self,
        model: str,
        contents: List[Dict[str, Any]],
        system_instruction: Optional[Dict[str, Any]] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Generate complete content response non-streaming."""
        pass

    @abstractmethod
    async def stream_generate_content(
        self,
        model: str,
        contents: List[Dict[str, Any]],
        system_instruction: Optional[Dict[str, Any]] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream generated content chunks."""
        pass

    @abstractmethod
    async def fetch_available_models(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Fetch models supported by this adapter."""
        pass

    async def embed_contents(
        self,
        model: str,
        texts: List[str],
        dimensions: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate text embeddings (optional support)."""
        raise NotImplementedError(f"Backend '{self.name}' does not support embeddings.")
