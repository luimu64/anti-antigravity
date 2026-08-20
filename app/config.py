import os
from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# Default base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
CREDENTIALS_FILE = Path(os.getenv("CREDENTIALS_FILE", str(DATA_DIR / "credentials.json")))

# Google OAuth Constants (extracted from agy binary)
DEFAULT_CLIENT_ID = os.getenv(
    "GOOGLE_CLIENT_ID",
    "1071006060591-tmhssin2h21lcre235vtolojh4g403ep.apps.googleusercontent.com"
)
DEFAULT_CLIENT_SECRET = os.getenv(
    "GOOGLE_CLIENT_SECRET",
    "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"
)

# Alternative fallback Client ID / Secret if needed
FALLBACK_CLIENT_ID = "884354919052-36trc1jjb3tguiac32ov6cod268c5blh.apps.googleusercontent.com"
FALLBACK_CLIENT_SECRET = "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"

OAUTH_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
OAUTH_TOKEN_URL = "https://oauth2.googleapis.com/token"
OAUTH_SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/cclog",
    "https://www.googleapis.com/auth/experimentsandconfigs",
    "openid"
]

# Cloud Code / Antigravity internal backend
CLOUD_CODE_BASE_URL = os.getenv("CLOUD_CODE_URL", "https://daily-cloudcode-pa.googleapis.com")
USER_AGENT = "antigravity/cli/1.1.13 (aidev_client; os_type=linux; arch=amd64; cl=964361259; auth_method=consumer)"

# Server Configuration
SERVER_HOST = os.getenv("HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("PORT", "8000"))
API_KEY = os.getenv("API_KEY", "")  # If set, incoming OpenAI requests require Authorization: Bearer <API_KEY>
PROJECT_ID_OVERRIDE = os.getenv("GOOGLE_PROJECT_ID", "")
REDIRECT_URI = os.getenv("REDIRECT_URI", f"http://localhost:{SERVER_PORT}/auth/callback")

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
    "text-embedding-004": "text-embedding-004"
}

# Predefined specifications for models: contextWindow (total tokens), maxOutputTokens (generation limit), supportsThinking
MODEL_SPECS = {
    # Gemini 3.7 (1M context window, 64k max output)
    "gemini-3.7-flash-high": {
        "displayName": "Gemini 3.7 Flash (High)",
        "contextWindow": 1048576,
        "maxOutputTokens": 65536,
        "supportsThinking": True,
    },
    "gemini-3.7-flash-medium": {
        "displayName": "Gemini 3.7 Flash (Medium)",
        "contextWindow": 1048576,
        "maxOutputTokens": 65536,
        "supportsThinking": True,
    },
    "gemini-3.7-flash-low": {
        "displayName": "Gemini 3.7 Flash (Low)",
        "contextWindow": 1048576,
        "maxOutputTokens": 65536,
        "supportsThinking": True,
    },
    "gemini-3.7-flash": {
        "displayName": "Gemini 3.7 Flash",
        "contextWindow": 1048576,
        "maxOutputTokens": 65536,
        "supportsThinking": True,
    },

    # Gemini 3.6 (1M context window, 64k max output)
    "gemini-3.6-flash-high": {
        "displayName": "Gemini 3.6 Flash (High)",
        "contextWindow": 1048576,
        "maxOutputTokens": 65536,
        "supportsThinking": True,
    },
    "gemini-3.6-flash-medium": {
        "displayName": "Gemini 3.6 Flash (Medium)",
        "contextWindow": 1048576,
        "maxOutputTokens": 65536,
        "supportsThinking": True,
    },
    "gemini-3.6-flash-low": {
        "displayName": "Gemini 3.6 Flash (Low)",
        "contextWindow": 1048576,
        "maxOutputTokens": 65536,
        "supportsThinking": True,
    },
    "gemini-3.6-flash": {
        "displayName": "Gemini 3.6 Flash",
        "contextWindow": 1048576,
        "maxOutputTokens": 65536,
        "supportsThinking": True,
    },

    # Gemini 3.5 & Agent
    "gemini-3-flash-agent": {
        "displayName": "Gemini 3 Flash Agent",
        "contextWindow": 1048576,
        "maxOutputTokens": 65536,
        "supportsThinking": True,
    },
    "gemini-3.5-flash": {
        "displayName": "Gemini 3.5 Flash",
        "contextWindow": 1048576,
        "maxOutputTokens": 65536,
        "supportsThinking": True,
    },
    "gemini-3.5-flash-high": {
        "displayName": "Gemini 3.5 Flash (High)",
        "contextWindow": 1048576,
        "maxOutputTokens": 65536,
        "supportsThinking": True,
    },

    # Gemini 3.1 Pro & 2.5
    "gemini-3.1-pro-high": {
        "displayName": "Gemini 3.1 Pro (High)",
        "contextWindow": 1048576,
        "maxOutputTokens": 65536,
        "supportsThinking": True,
    },
    "gemini-3.1-pro-low": {
        "displayName": "Gemini 3.1 Pro (Low)",
        "contextWindow": 1048576,
        "maxOutputTokens": 65536,
        "supportsThinking": True,
    },
    "gemini-3.1-pro": {
        "displayName": "Gemini 3.1 Pro",
        "contextWindow": 1048576,
        "maxOutputTokens": 65536,
        "supportsThinking": True,
    },
    "gemini-2.5-pro": {
        "displayName": "Gemini 2.5 Pro",
        "contextWindow": 2097152,
        "maxOutputTokens": 65536,
        "supportsThinking": True,
    },
    "gemini-2.5-flash": {
        "displayName": "Gemini 2.5 Flash",
        "contextWindow": 1048576,
        "maxOutputTokens": 65536,
        "supportsThinking": True,
    },
    "gemini-2.0-flash": {
        "displayName": "Gemini 2.0 Flash",
        "contextWindow": 1048576,
        "maxOutputTokens": 65536,
        "supportsThinking": False,
    },
    "gemini-1.5-pro": {
        "displayName": "Gemini 1.5 Pro",
        "contextWindow": 2097152,
        "maxOutputTokens": 65536,
        "supportsThinking": False,
    },
    "gemini-1.5-flash": {
        "displayName": "Gemini 1.5 Flash",
        "contextWindow": 1048576,
        "maxOutputTokens": 65536,
        "supportsThinking": False,
    },

    # Claude models (250k / 200k context window, 64k max output)
    "claude-sonnet-4-6": {
        "displayName": "Claude Sonnet 4.6",
        "contextWindow": 250000,
        "maxOutputTokens": 64000,
        "supportsThinking": True,
    },
    "claude-opus-4-6-thinking": {
        "displayName": "Claude Opus 4.6 (Thinking)",
        "contextWindow": 250000,
        "maxOutputTokens": 64000,
        "supportsThinking": True,
    },
    "claude-3-7-sonnet": {
        "displayName": "Claude 3.7 Sonnet",
        "contextWindow": 250000,
        "maxOutputTokens": 64000,
        "supportsThinking": True,
    },
    "claude-3-5-sonnet": {
        "displayName": "Claude 3.5 Sonnet",
        "contextWindow": 200000,
        "maxOutputTokens": 64000,
        "supportsThinking": True,
    },
    "claude-sonnet-3.7": {
        "displayName": "Claude Sonnet 3.7",
        "contextWindow": 250000,
        "maxOutputTokens": 64000,
        "supportsThinking": True,
    },
    "claude-3-opus": {
        "displayName": "Claude 3 Opus",
        "contextWindow": 200000,
        "maxOutputTokens": 64000,
        "supportsThinking": True,
    },
    "claude-opus-4.6": {
        "displayName": "Claude Opus 4.6",
        "contextWindow": 250000,
        "maxOutputTokens": 64000,
        "supportsThinking": True,
    },

    # GPT-OSS (128k/131k context window, 32k max output)
    "gpt-oss-120b-medium": {
        "displayName": "GPT-OSS 120B",
        "contextWindow": 131072,
        "maxOutputTokens": 32768,
        "supportsThinking": False,
    },
    "gpt-oss-120b": {
        "displayName": "GPT-OSS 120B",
        "contextWindow": 131072,
        "maxOutputTokens": 32768,
        "supportsThinking": False,
    },

    # OpenAI aliases
    "gpt-4o": {
        "displayName": "GPT-4o (Gemini 3.7 Flash)",
        "contextWindow": 1048576,
        "maxOutputTokens": 65536,
        "supportsThinking": True,
    },
    "gpt-4o-mini": {
        "displayName": "GPT-4o mini (Gemini 3.6 Flash)",
        "contextWindow": 1048576,
        "maxOutputTokens": 65536,
        "supportsThinking": True,
    },
    "gpt-4-turbo": {
        "displayName": "GPT-4 Turbo (Gemini 3.1 Pro)",
        "contextWindow": 1048576,
        "maxOutputTokens": 65536,
        "supportsThinking": True,
    },
    "o1": {
        "displayName": "o1 (Claude Opus 4.6 Thinking)",
        "contextWindow": 250000,
        "maxOutputTokens": 64000,
        "supportsThinking": True,
    },
    "o3-mini": {
        "displayName": "o3-mini (Gemini 3.7 Flash)",
        "contextWindow": 1048576,
        "maxOutputTokens": 65536,
        "supportsThinking": True,
    },

    # Embedding models
    "text-embedding-004": {
        "displayName": "Text Embedding 004",
        "contextWindow": 2048,
        "maxOutputTokens": 2048,
        "supportsThinking": False,
    },
    "text-embedding-3-small": {
        "displayName": "Text Embedding 3 Small",
        "contextWindow": 8192,
        "maxOutputTokens": 8192,
        "supportsThinking": False,
    },
    "text-embedding-3-large": {
        "displayName": "Text Embedding 3 Large",
        "contextWindow": 8192,
        "maxOutputTokens": 8192,
        "supportsThinking": False,
    },
    "text-embedding-ada-002": {
        "displayName": "Text Embedding Ada 002",
        "contextWindow": 8192,
        "maxOutputTokens": 8192,
        "supportsThinking": False,
    },
}

def resolve_model_alias(model_id: str) -> str:
    """Map user-requested model to internal model name."""
    clean = model_id.lower().strip()
    if clean in MODEL_ALIASES:
        return MODEL_ALIASES[clean]
    for alias, internal in MODEL_ALIASES.items():
        if clean.startswith(alias):
            return internal
    return model_id

def get_model_metadata(
    model_id: str,
    raw_info: Optional[dict] = None,
    created_time: int = 1700000000
) -> dict:
    """
    Build standard OpenAI model dictionary with accurate context window and token limits.
    Broadcasts context_window / context_length (total context tokens) and
    max_tokens / max_output_tokens (max completion output tokens).
    """
    raw_info = raw_info or {}
    clean_id = model_id.lower().strip()
    root_id = resolve_model_alias(clean_id)

    # 1. Spec lookup
    spec = MODEL_SPECS.get(clean_id) or MODEL_SPECS.get(root_id) or {}

    # 2. Context Window (total input + output capacity)
    context_window = (
        raw_info.get("context_window")
        or raw_info.get("contextWindow")
        or raw_info.get("context_length")
        or raw_info.get("contextLength")
    )
    if not context_window:
        raw_max_tokens = raw_info.get("maxTokens")
        raw_max_output = raw_info.get("maxOutputTokens")
        if raw_max_tokens is not None and (raw_max_tokens > 65536 or raw_max_output is not None):
            context_window = raw_max_tokens
        else:
            context_window = spec.get("contextWindow") or spec.get("context_window") or 1048576

    # 3. Max Output Tokens (max generation limit)
    max_output_tokens = (
        raw_info.get("max_output_tokens")
        or raw_info.get("maxOutputTokens")
    )
    if not max_output_tokens:
        raw_max_tokens = raw_info.get("maxTokens")
        if raw_max_tokens is not None and raw_max_tokens <= 65536 and not raw_info.get("maxOutputTokens"):
            max_output_tokens = raw_max_tokens
        else:
            max_output_tokens = spec.get("maxOutputTokens") or spec.get("max_output_tokens")
            if not max_output_tokens:
                if "claude" in root_id:
                    max_output_tokens = 64000
                elif "gpt-oss" in root_id:
                    max_output_tokens = 32768
                elif "embedding" in root_id:
                    max_output_tokens = 2048
                else:
                    max_output_tokens = 65536

    # 4. Display Name & Name
    display_name = (
        raw_info.get("displayName")
        or raw_info.get("display_name")
        or raw_info.get("name")
        or spec.get("displayName")
        or spec.get("display_name")
        or model_id
    )

    # 5. Supports Thinking
    if "supportsThinking" in raw_info:
        supports_thinking = bool(raw_info["supportsThinking"])
    elif "supports_thinking" in raw_info:
        supports_thinking = bool(raw_info["supports_thinking"])
    else:
        supports_thinking = spec.get(
            "supportsThinking",
            spec.get(
                "supports_thinking",
                bool("3.7" in root_id or "thinking" in root_id or "claude" in root_id or "o1" in clean_id or "o3" in clean_id)
            )
        )

    owned_by = "google"
    if clean_id in MODEL_ALIASES and clean_id != root_id:
        owned_by = "google-antigravity"

    return {
        "id": model_id,
        "object": "model",
        "created": created_time,
        "owned_by": owned_by,
        "permission": [],
        "root": root_id,
        "parent": None,
        "name": display_name,
        "display_name": display_name,
        "context_window": int(context_window),
        "context_length": int(context_window),
        "max_tokens": int(max_output_tokens),
        "max_output_tokens": int(max_output_tokens),
        "supports_thinking": supports_thinking,
    }
