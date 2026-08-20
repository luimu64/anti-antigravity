import os
from pathlib import Path
from typing import Optional, Dict, Any
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

def translate_model_to_openai(
    model_id: str,
    info: Optional[Dict[str, Any]] = None,
    created_time: int = 1700000000
) -> Dict[str, Any]:
    """
    Dynamically translate internal backend model information into standard OpenAI Model schema.
    Extracts context window (maxTokens / context_window) and output tokens limit (maxOutputTokens)
    directly from upstream response without hardcoded model tables.
    """
    info = info or {}
    clean_id = model_id.strip()
    if clean_id.startswith("models/"):
        clean_id = clean_id[7:]

    display_name = (
        info.get("displayName")
        or info.get("display_name")
        or info.get("name")
        or clean_id
    )

    # In Google Cloud Code / Antigravity backend:
    # - maxTokens represents the full context window capacity
    # - maxOutputTokens represents the maximum generation output tokens limit
    context_window = (
        info.get("context_window")
        or info.get("contextWindow")
        or info.get("context_length")
        or info.get("contextLength")
        or info.get("maxTokens")
    )
    max_output_tokens = (
        info.get("max_output_tokens")
        or info.get("maxOutputTokens")
    )

    context_val = int(context_window) if context_window is not None else None
    max_output_val = int(max_output_tokens) if max_output_tokens is not None else context_val

    model_obj = {
        "id": clean_id,
        "object": "model",
        "created": created_time,
        "owned_by": "google",
        "permission": [],
        "root": clean_id,
        "parent": None,
        "name": display_name,
        "display_name": display_name,
    }

    if context_val is not None:
        model_obj["context_window"] = context_val
        model_obj["context_length"] = context_val
    if max_output_val is not None:
        model_obj["max_tokens"] = max_output_val
        model_obj["max_output_tokens"] = max_output_val
    if "supportsThinking" in info:
        model_obj["supports_thinking"] = bool(info["supportsThinking"])
    elif "supports_thinking" in info:
        model_obj["supports_thinking"] = bool(info["supports_thinking"])

    return model_obj
