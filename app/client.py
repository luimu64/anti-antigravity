import logging

from app.providers.antigravity import AntigravityAdapter
from app.providers.router import router_client

logger = logging.getLogger("agy_to_api.client")

# Backwards compatibility alias
AntigravityClient = AntigravityAdapter

# Global router client singleton
client = router_client
