import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Default base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
CREDENTIALS_FILE = Path(
    os.getenv("CREDENTIALS_FILE", str(DATA_DIR / "credentials.json"))
)

# Google OAuth Constants (extracted from agy binary)
DEFAULT_CLIENT_ID = os.getenv(
    "GOOGLE_CLIENT_ID",
    "1071006060591-tmhssin2h21lcre235vtolojh4g403ep.apps.googleusercontent.com",
)
DEFAULT_CLIENT_SECRET = os.getenv(
    "GOOGLE_CLIENT_SECRET", "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"
)

# Alternative fallback Client ID / Secret if needed
FALLBACK_CLIENT_ID = (
    "884354919052-36trc1jjb3tguiac32ov6cod268c5blh.apps.googleusercontent.com"
)
FALLBACK_CLIENT_SECRET = "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"

OAUTH_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
OAUTH_TOKEN_URL = "https://oauth2.googleapis.com/token"
OAUTH_SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/cclog",
    "https://www.googleapis.com/auth/experimentsandconfigs",
    "openid",
]

# Cloud Code / Antigravity internal backend
CLOUD_CODE_BASE_URL = os.getenv(
    "CLOUD_CODE_URL", "https://daily-cloudcode-pa.googleapis.com"
)
USER_AGENT = "antigravity/cli/1.1.13 (aidev_client; os_type=linux; arch=amd64; cl=964361259; auth_method=consumer)"

# Server Configuration
SERVER_HOST = os.getenv("HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("PORT", "8000"))
API_KEY = os.getenv(
    "API_KEY", ""
)  # If set, incoming OpenAI requests require Authorization: Bearer <API_KEY>
PROJECT_ID_OVERRIDE = os.getenv("GOOGLE_PROJECT_ID", "")
REDIRECT_URI = os.getenv(
    "REDIRECT_URI", f"http://localhost:{SERVER_PORT}/auth/callback"
)

# Model aliases mapping standard OpenAI / Anthropic names to Antigravity internal models
MODEL_ALIASES = {
    # Gemini 3.7
    "gemini-3.7-flash": "gemini-3.7-flash-high",
    "gemini-3.7-flash-high": "gemini-3.7-flash-high",
    "gemini-3.7-flash-medium": "gemini-3.7-flash-medium",
    "gemini-3.7-flash-low": "gemini-3.7-flash-low",
    # Gemini 3.6
    "gemini-3.6-flash": "gemini-3.6-flash-high",
    "gemini-3.6-flash-high": "gemini-3.6-flash-high",
    "gemini-3.6-flash-medium": "gemini-3.6-flash-medium",
    "gemini-3.6-flash-low": "gemini-3.6-flash-low",
    # Gemini 3.5 & Pro
    "gemini-3.5-flash": "gemini-3-flash-agent",
    "gemini-3.5-flash-high": "gemini-3-flash-agent",
    "gemini-3.1-pro": "gemini-3.1-pro-high",
    "gemini-3.1-pro-high": "gemini-3.1-pro-high",
    "gemini-3.1-pro-low": "gemini-3.1-pro-low",
    "gemini-2.5-pro": "gemini-2.5-pro",
    "gemini-2.5-flash": "gemini-2.5-flash",
    # Claude models
    "claude-sonnet-4-6": "claude-sonnet-4-6",
    "claude-3-7-sonnet": "claude-sonnet-4-6",
    "claude-3-5-sonnet": "claude-sonnet-4-6",
    "claude-sonnet-3.7": "claude-sonnet-4-6",
    "claude-opus-4-6-thinking": "claude-opus-4-6-thinking",
    "claude-3-opus": "claude-opus-4-6-thinking",
    "claude-opus-4.6": "claude-opus-4-6-thinking",
    # OpenAI aliases
    "gpt-4o": "gemini-3.7-flash-high",
    "gpt-4o-mini": "gemini-3.6-flash-high",
    "gpt-4-turbo": "gemini-3.1-pro-high",
    "o1": "claude-opus-4-6-thinking",
    "o3-mini": "gemini-3.7-flash-high",
    "gpt-oss-120b": "gpt-oss-120b-medium",
    "gpt-oss-120b-medium": "gpt-oss-120b-medium",
    # Embedding aliases
    "text-embedding-3-small": "text-embedding-004",
    "text-embedding-3-large": "text-embedding-004",
    "text-embedding-ada-002": "text-embedding-004",
    "text-embedding-004": "text-embedding-004",
}

# Lookup table mapping base canonical models to Antigravity internal reasoning tiers
ANTIGRAVITY_TIER_MAP = {
    "gemini-3.7-flash": {
        "low": "gemini-3.7-flash-low",
        "medium": "gemini-3.7-flash-medium",
        "high": "gemini-3.7-flash-high",
        "default": "gemini-3.7-flash-high",
    },
    "gemini-3.6-flash": {
        "low": "gemini-3.6-flash-low",
        "medium": "gemini-3.6-flash-medium",
        "high": "gemini-3.6-flash-high",
        "default": "gemini-3.6-flash-high",
    },
    "gemini-3.1-pro": {
        "low": "gemini-3.1-pro-low",
        "medium": "gemini-3.1-pro-high",
        "high": "gemini-3.1-pro-high",
        "default": "gemini-3.1-pro-high",
    },
    "gemini-3.5-flash": {"default": "gemini-3-flash-agent"},
    "gpt-oss-120b": {"default": "gpt-oss-120b-medium"},
    "claude-3.7-sonnet": {"default": "claude-sonnet-4-6"},
    "claude-3.5-sonnet": {"default": "claude-sonnet-4-6"},
    "claude-3-opus": {"default": "claude-opus-4-6-thinking"},
}

# Canonical model consolidation mapping (Antigravity quirk -> Public clean name)
CANONICAL_MODEL_MAP = {
    "gemini-3.7-flash-high": "gemini-3.7-flash",
    "gemini-3.7-flash-medium": "gemini-3.7-flash",
    "gemini-3.7-flash-low": "gemini-3.7-flash",
    "gemini-3.6-flash-high": "gemini-3.6-flash",
    "gemini-3.6-flash-medium": "gemini-3.6-flash",
    "gemini-3.6-flash-low": "gemini-3.6-flash",
    "gemini-3.1-pro-high": "gemini-3.1-pro",
    "gemini-3.1-pro-low": "gemini-3.1-pro",
    "gemini-3-flash-agent": "gemini-3.5-flash",
    "gpt-oss-120b-medium": "gpt-oss-120b",
    "claude-sonnet-4-6": "claude-3.7-sonnet",
    "claude-opus-4-6-thinking": "claude-3-opus",
}

# Internal or unusable models hidden by default in the catalog
HIDDEN_MODELS = {
    "tab_flash_lite_preview",
    "tab_flash_preview",
    "tab_pro_preview",
    "tab_flash_lite",
    "gemini-3.6-flash-low",
    "gemini-3.6-flash-medium",
    "gemini-3.7-flash-low",
    "gemini-3.7-flash-medium",
    "gemini-3.1-pro-low",
}
