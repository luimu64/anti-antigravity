import logging
from app.providers.base import BaseAdapter, RateLimitError
from app.providers.antigravity import AntigravityAdapter
from app.providers.gemini_api import GeminiApiAdapter
from app.providers.gemini_web import GeminiWebAdapter
from app.providers.router import MultiBackendRouter, router_client

logger = logging.getLogger("agy_to_api.client")

# Backwards compatibility alias
AntigravityClient = AntigravityAdapter

# Global router client singleton
client = router_client
