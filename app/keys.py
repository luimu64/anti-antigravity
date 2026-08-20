import json
import logging
import os
import secrets
import time
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from app.config import DATA_DIR

logger = logging.getLogger("agy_to_api.keys")
API_KEYS_FILE = Path(os.getenv("API_KEYS_FILE", str(DATA_DIR / "api_keys.json")))


class APIKeyItem(BaseModel):
    id: str
    name: str
    key: str  # hashed or raw for storage
    key_preview: str
    created_at: int
    last_used_at: int | None = None
    is_active: bool = True


class APIKeyManager:
    def __init__(self, storage_file: Path = API_KEYS_FILE):
        self.storage_file = storage_file
        self.keys: dict[str, dict[str, Any]] = {}
        self.enforce_keys: bool = os.getenv("ENFORCE_API_KEY", "true").lower() in (
            "true",
            "1",
            "yes",
        )
        self.load_keys()

    def load_keys(self):
        """Load API keys from disk."""
        if self.storage_file.exists():
            try:
                with open(self.storage_file, "r") as f:
                    data = json.load(f)
                self.keys = data.get("keys", {})
                self.enforce_keys = data.get("enforce_keys", self.enforce_keys)
                logger.info(
                    f"Loaded {len(self.keys)} API keys from {self.storage_file}"
                )
            except Exception as e:
                logger.error(f"Failed to load API keys: {e}")
                self.keys = {}
        else:
            # Generate a default initial key if none exists
            initial_key = os.getenv("API_KEY") or f"sk-agy-{secrets.token_hex(16)}"
            key_id = f"key_{uuid.uuid4().hex[:8]}"
            self.keys[key_id] = {
                "id": key_id,
                "name": "Default Key",
                "key": initial_key,
                "key_preview": initial_key[:10] + "..." + initial_key[-4:],
                "created_at": int(time.time()),
                "last_used_at": None,
                "is_active": True,
            }
            self.save_keys()
            logger.info(f"Initialized default API key: {initial_key[:10]}...")

    def save_keys(self):
        """Save API keys to disk."""
        try:
            os.makedirs(self.storage_file.parent, exist_ok=True)
            with open(self.storage_file, "w") as f:
                json.dump(
                    {"keys": self.keys, "enforce_keys": self.enforce_keys}, f, indent=2
                )
        except Exception as e:
            logger.error(f"Failed to save API keys to {self.storage_file}: {e}")

    def create_key(self, name: str = "New Key") -> dict[str, Any]:
        """Create a new API key for client access to the bridge."""
        raw_key = f"sk-agy-{secrets.token_hex(20)}"
        key_id = f"key_{uuid.uuid4().hex[:8]}"
        created_at = int(time.time())

        item = {
            "id": key_id,
            "name": name or "Unnamed Key",
            "key": raw_key,
            "key_preview": raw_key[:10] + "..." + raw_key[-4:],
            "created_at": created_at,
            "last_used_at": None,
            "is_active": True,
        }
        self.keys[key_id] = item
        self.save_keys()
        return item

    def revoke_key(self, key_id: str) -> bool:
        """Revoke / delete an API key."""
        if key_id in self.keys:
            del self.keys[key_id]
            self.save_keys()
            return True
        return False

    def list_keys(self) -> list[dict[str, Any]]:
        """List all keys (with masked values)."""
        result = []
        for k in self.keys.values():
            result.append(
                {
                    "id": k["id"],
                    "name": k["name"],
                    "key_preview": k["key_preview"],
                    "created_at": k["created_at"],
                    "last_used_at": k.get("last_used_at"),
                    "is_active": k.get("is_active", True),
                }
            )
        return sorted(result, key=lambda x: x["created_at"], reverse=True)

    def validate_key(self, token: str) -> bool:
        """
        Validate incoming Bearer token against stored keys or env var.
        Returns True if valid, False otherwise.
        """
        if not self.enforce_keys:
            return True

        if not token:
            return False

        clean_token = token.strip()

        # Check against env var API_KEY if set
        env_key = os.getenv("API_KEY")
        if env_key and clean_token == env_key:
            return True

        # Check against active registered keys
        for k in self.keys.values():
            if k.get("is_active", True) and k.get("key") == clean_token:
                # Update last used timestamp
                k["last_used_at"] = int(time.time())
                self.save_keys()
                return True

        return False

    def get_first_active_key(self) -> str | None:
        """Get the first active raw key (for UI testing)."""
        for k in self.keys.values():
            if k.get("is_active", True):
                return k.get("key")
        return os.getenv("API_KEY")


# Global singleton
api_key_manager = APIKeyManager()
