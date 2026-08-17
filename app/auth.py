import os
import json
import time
import base64
import hashlib
import secrets
import logging
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timezone
import httpx

from app.config import (
    DEFAULT_CLIENT_ID,
    DEFAULT_CLIENT_SECRET,
    OAUTH_AUTH_URL,
    OAUTH_TOKEN_URL,
    OAUTH_SCOPES,
    CREDENTIALS_FILE,
    REDIRECT_URI,
    PROJECT_ID_OVERRIDE
)

logger = logging.getLogger("agy_to_api.auth")

class OAuthManager:
    def __init__(
        self,
        client_id: str = DEFAULT_CLIENT_ID,
        client_secret: str = DEFAULT_CLIENT_SECRET,
        credentials_file: str = str(CREDENTIALS_FILE)
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.credentials_file = credentials_file
        
        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.token_expiry: float = 0
        self.id_token: Optional[str] = None
        self.user_email: Optional[str] = None
        self.project_id: Optional[str] = PROJECT_ID_OVERRIDE or None
        self.tier_name: Optional[str] = None
        
        # In-memory PKCE state mapping: state -> code_verifier
        self._pkce_verifier_cache: Dict[str, str] = {}
        
        # Try loading credentials automatically
        self.load_credentials()

    def generate_pkce(self) -> Tuple[str, str]:
        """Generate PKCE code_verifier and code_challenge."""
        verifier = secrets.token_urlsafe(64)
        digest = hashlib.sha256(verifier.encode("utf-8")).digest()
        challenge = base64.urlsafe_b64encode(digest).decode("utf-8").rstrip("=")
        return verifier, challenge

    def get_authorization_url(self, redirect_uri: str = REDIRECT_URI, state: Optional[str] = None) -> Tuple[str, str, str]:
        """
        Generate the Google OAuth2 authorization URL.
        Returns: (auth_url, state, code_verifier)
        """
        if not state:
            state = secrets.token_urlsafe(16)
        
        verifier, challenge = self.generate_pkce()
        self._pkce_verifier_cache[state] = verifier

        params = {
            "client_id": self.client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": " ".join(OAUTH_SCOPES),
            "access_type": "offline",
            "prompt": "consent",
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256"
        }
        
        req = httpx.Request("GET", OAUTH_AUTH_URL, params=params)
        auth_url = str(req.url)
        return auth_url, state, verifier

    async def exchange_code(
        self,
        code: str,
        redirect_uri: str = REDIRECT_URI,
        state: Optional[str] = None,
        code_verifier: Optional[str] = None
    ) -> Dict[str, Any]:
        """Exchange authorization code for access and refresh tokens."""
        verifier = code_verifier
        if not verifier and state and state in self._pkce_verifier_cache:
            verifier = self._pkce_verifier_cache.pop(state)

        data = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": redirect_uri
        }
        if verifier:
            data["code_verifier"] = verifier

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(OAUTH_TOKEN_URL, data=data)
            if resp.status_code != 200:
                logger.error(f"Failed to exchange authorization code: {resp.status_code} {resp.text}")
                raise ValueError(f"Failed to exchange authorization code: {resp.text}")
            
            token_data = resp.json()

        self._apply_token_response(token_data)
        self.save_credentials()
        return token_data

    async def refresh_access_token(self) -> str:
        """Refresh the access token using the current refresh token."""
        if not self.refresh_token:
            raise ValueError("No refresh token available. Please log in first.")

        data = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "refresh_token": self.refresh_token,
            "grant_type": "refresh_token"
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(OAUTH_TOKEN_URL, data=data)
            if resp.status_code != 200:
                logger.error(f"Failed to refresh access token: {resp.status_code} {resp.text}")
                raise ValueError(f"Failed to refresh access token: {resp.text}")
            
            token_data = resp.json()

        self._apply_token_response(token_data)
        self.save_credentials()
        return self.access_token

    async def get_valid_access_token(self) -> str:
        """Get a valid access token, refreshing it automatically if expired or expiring within 2 minutes."""
        now = time.time()
        # If token is missing or expiring in less than 120 seconds
        if not self.access_token or (self.token_expiry and (self.token_expiry - now < 120)):
            if self.refresh_token:
                logger.info("Access token expired or near expiration, refreshing...")
                return await self.refresh_access_token()
            elif self.access_token:
                # If we only have access token without refresh token, return it
                return self.access_token
            else:
                raise ValueError("Not authenticated. Please log in with Google first.")
        
        return self.access_token

    def _apply_token_response(self, token_data: Dict[str, Any]):
        """Parse and store token response data."""
        self.access_token = token_data.get("access_token")
        if "refresh_token" in token_data and token_data["refresh_token"]:
            self.refresh_token = token_data["refresh_token"]
        
        expires_in = token_data.get("expires_in", 3600)
        self.token_expiry = time.time() + float(expires_in)
        
        if "id_token" in token_data:
            self.id_token = token_data["id_token"]
            self._extract_email_from_id_token(self.id_token)

    def _extract_email_from_id_token(self, id_token: str):
        """Extract email claim from JWT ID token without full verification."""
        try:
            parts = id_token.split(".")
            if len(parts) >= 2:
                padded = parts[1] + "=" * ((4 - len(parts[1]) % 4) % 4)
                payload = json.loads(base64.urlsafe_b64decode(padded.encode("utf-8")))
                if "email" in payload:
                    self.user_email = payload["email"]
        except Exception as e:
            logger.debug(f"Could not extract email from id_token: {e}")

    def save_credentials(self):
        """Save current credentials and metadata to disk."""
        data = {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "token_expiry": self.token_expiry,
            "user_email": self.user_email,
            "project_id": self.project_id,
            "tier_name": self.tier_name,
            "id_token": self.id_token,
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        try:
            os.makedirs(os.path.dirname(self.credentials_file), exist_ok=True)
            with open(self.credentials_file, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved credentials to {self.credentials_file}")
        except Exception as e:
            logger.error(f"Failed to save credentials to {self.credentials_file}: {e}")

    def load_credentials(self) -> bool:
        """
        Load credentials from:
        1. Environment variables
        2. Local credentials file
        3. Linux Secret Service (DBus) keyring if available
        """
        # 1. Environment variables
        env_refresh = os.getenv("REFRESH_TOKEN")
        env_access = os.getenv("ACCESS_TOKEN") or os.getenv("GOOGLE_OAUTH_TOKEN")
        if env_refresh or env_access:
            self.refresh_token = env_refresh or self.refresh_token
            self.access_token = env_access or self.access_token
            if env_access and not self.token_expiry:
                self.token_expiry = time.time() + 3600
            logger.info("Loaded credentials from environment variables.")
            return True

        # 2. Local credentials file
        if os.path.exists(self.credentials_file):
            try:
                with open(self.credentials_file, "r") as f:
                    data = json.load(f)
                self.access_token = data.get("access_token")
                self.refresh_token = data.get("refresh_token")
                self.token_expiry = data.get("token_expiry", 0)
                self.user_email = data.get("user_email")
                self.project_id = PROJECT_ID_OVERRIDE or data.get("project_id")
                self.tier_name = data.get("tier_name")
                self.id_token = data.get("id_token")
                logger.info(f"Loaded credentials from {self.credentials_file}")
                return True
            except Exception as e:
                logger.warning(f"Failed to read {self.credentials_file}: {e}")

        # 3. Linux Secret Service (DBus) auto-discovery
        if self._load_from_secret_service():
            self.save_credentials()
            return True

        return False

    def _load_from_secret_service(self) -> bool:
        """Attempt to read antigravity token from system SecretService via secret-tool CLI or dbus."""
        # Try secret-tool CLI first
        try:
            import subprocess
            res = subprocess.run(
                ["secret-tool", "lookup", "service", "gemini", "username", "antigravity"],
                capture_output=True,
                text=True,
                timeout=3.0
            )
            if res.returncode == 0 and res.stdout.strip():
                parsed = json.loads(res.stdout.strip())
                token_obj = parsed.get("token", {})
                if isinstance(token_obj, dict):
                    self.access_token = token_obj.get("access_token")
                    self.refresh_token = token_obj.get("refresh_token")
                    expiry_str = token_obj.get("expiry")
                    if expiry_str:
                        try:
                            # Parse full ISO with timezone or clean timestamp
                            dt = datetime.fromisoformat(expiry_str)
                            self.token_expiry = dt.timestamp()
                        except Exception:
                            self.token_expiry = time.time() + 3600
                    logger.info("Successfully loaded antigravity credentials via secret-tool!")
                    return True
        except Exception as e:
            logger.debug(f"secret-tool discovery skipped: {e}")

        # Try dbus next
        try:
            import dbus
            bus = dbus.SessionBus()
            service = bus.get_object("org.freedesktop.secrets", "/org/freedesktop/secrets")
            iface = dbus.Interface(service, "org.freedesktop.Secret.Service")
            session_path = iface.OpenSession("plain", "")[1]

            search_res = iface.SearchItems({"service": "gemini", "username": "antigravity"})
            items = search_res[0] + search_res[1]
            if not items:
                return False

            item = bus.get_object("org.freedesktop.secrets", items[0])
            item_iface = dbus.Interface(item, "org.freedesktop.Secret.Item")
            secret = item_iface.GetSecret(session_path)
            secret_bytes = bytes(secret[2])
            parsed = json.loads(secret_bytes.decode("utf-8"))
            
            token_obj = parsed.get("token", {})
            if isinstance(token_obj, dict):
                self.access_token = token_obj.get("access_token")
                self.refresh_token = token_obj.get("refresh_token")
                
                expiry_str = token_obj.get("expiry")
                if expiry_str:
                    try:
                        clean_exp = expiry_str.split("+")[0].split("Z")[0]
                        dt = datetime.fromisoformat(clean_exp)
                        self.token_expiry = dt.replace(tzinfo=timezone.utc).timestamp()
                    except Exception:
                        self.token_expiry = time.time() + 3600
                
                logger.info("Successfully discovered and imported antigravity token from System Keyring DBus!")
                return True
        except Exception as e:
            logger.debug(f"System keyring discovery skipped or unavailable: {e}")
        return False

    def set_tokens(self, access_token: Optional[str], refresh_token: Optional[str], project_id: Optional[str] = None):
        """Manually configure tokens."""
        if access_token:
            self.access_token = access_token
            self.token_expiry = time.time() + 3600
        if refresh_token:
            self.refresh_token = refresh_token
        if project_id:
            self.project_id = project_id
        self.save_credentials()

    def logout(self):
        """Clear all stored tokens."""
        self.access_token = None
        self.refresh_token = None
        self.token_expiry = 0
        self.id_token = None
        self.user_email = None
        self.project_id = PROJECT_ID_OVERRIDE or None
        self.tier_name = None
        if os.path.exists(self.credentials_file):
            try:
                os.remove(self.credentials_file)
            except Exception as e:
                logger.warning(f"Could not delete {self.credentials_file}: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Return current authentication status."""
        is_authenticated = bool(self.access_token or self.refresh_token)
        now = time.time()
        is_expired = self.token_expiry > 0 and (self.token_expiry < now)
        expires_in_seconds = max(0, int(self.token_expiry - now)) if self.token_expiry else None

        return {
            "authenticated": is_authenticated,
            "user_email": self.user_email,
            "project_id": self.project_id,
            "tier_name": self.tier_name,
            "has_refresh_token": bool(self.refresh_token),
            "token_expired": is_expired,
            "expires_in_seconds": expires_in_seconds,
            "credentials_file": self.credentials_file
        }

# Global singleton
auth_manager = OAuthManager()
