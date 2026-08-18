import os
import logging
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from jinja2 import Environment, FileSystemLoader

from app.config import SERVER_PORT, BASE_DIR
from app.auth import auth_manager
from app.keys import api_key_manager
from app.client import client
from pydantic import BaseModel

logger = logging.getLogger("agy_to_api.dashboard")
router = APIRouter(tags=["Dashboard"])

templates_dir = Path(__file__).resolve().parent.parent / "templates"
jinja_env = Environment(loader=FileSystemLoader(str(templates_dir)))

class CreateKeyRequest(BaseModel):
    name: Optional[str] = "New Key"

class ToggleEnforcementRequest(BaseModel):
    enforce: bool

@router.get("/", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    """
    Render main interactive web dashboard.
    """
    status = auth_manager.get_status()
    template = jinja_env.get_template("dashboard.html")

    # Determine base URL dynamically using HOST env or request host header
    env_host = os.getenv("HOST")
    if env_host and env_host != "0.0.0.0":
        base_url = f"http://{env_host}:{SERVER_PORT}/v1"
    else:
        req_host = request.headers.get("host")
        if req_host:
            base_url = f"{request.url.scheme}://{req_host}/v1"
        else:
            base_url = f"http://localhost:{SERVER_PORT}/v1"

    html_content = template.render(
        auth_status=status,
        port=SERVER_PORT,
        base_url=base_url,
        enforce_keys=api_key_manager.enforce_keys
    )
    return HTMLResponse(content=html_content)

@router.get("/api/keys")
async def get_api_keys():
    """
    List all API keys.
    """
    return {
        "enforce_keys": api_key_manager.enforce_keys,
        "keys": api_key_manager.list_keys()
    }

@router.post("/api/keys")
async def create_api_key(payload: CreateKeyRequest):
    """
    Generate a new API key.
    """
    new_key = api_key_manager.create_key(name=payload.name or "New Key")
    return {
        "status": "created",
        "key": new_key
    }

@router.delete("/api/keys/{key_id}")
async def revoke_api_key(key_id: str):
    """
    Revoke / delete an API key.
    """
    success = api_key_manager.revoke_key(key_id)
    if not success:
        return JSONResponse(status_code=404, content={"error": "Key not found"})
    return {"status": "revoked", "key_id": key_id}

@router.post("/api/keys/enforcement")
async def toggle_key_enforcement(payload: ToggleEnforcementRequest):
    """
    Enable or disable API key enforcement.
    """
    api_key_manager.enforce_keys = payload.enforce
    api_key_manager.save_keys()
    return {
        "status": "updated",
        "enforce_keys": api_key_manager.enforce_keys
    }

@router.get("/api/quotas")
async def get_quotas():
    """
    Retrieve user quota usage.
    """
    try:
        data = await client.retrieve_user_quota_summary()
        return JSONResponse(content=data)
    except Exception as e:
        logger.warning(f"Failed to fetch quotas: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "groups": []}
        )

@router.get("/health")
async def health_check():
    """
    Service health check endpoint.
    """
    status = auth_manager.get_status()
    return {
        "status": "healthy",
        "service": "agy-to-api",
        "authenticated": status["authenticated"],
        "project_id": status["project_id"],
        "api_key_enforcement": api_key_manager.enforce_keys
    }
