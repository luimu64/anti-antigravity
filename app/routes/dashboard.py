import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel

from app.config import SERVER_PORT, BASE_DIR
from app.auth import auth_manager
from app.keys import api_key_manager
from app.client import client

logger = logging.getLogger("agy_to_api.dashboard")
router = APIRouter(tags=["Dashboard"])

templates_dir = Path(__file__).resolve().parent.parent / "templates"
jinja_env = Environment(loader=FileSystemLoader(str(templates_dir)))

class CreateKeyRequest(BaseModel):
    name: Optional[str] = "New Key"

class ToggleEnforcementRequest(BaseModel):
    enforce: bool

class UpdateBackendsRequest(BaseModel):
    routing_strategy: Optional[str] = None
    gemini_api_key: Optional[str] = None
    gemini_api_enabled: Optional[bool] = None
    gemini_web_psid: Optional[str] = None
    gemini_web_psidts: Optional[str] = None
    gemini_web_enabled: Optional[bool] = None
    antigravity_enabled: Optional[bool] = None

class ToggleBackendRequest(BaseModel):
    enabled: bool

class SetStrategyRequest(BaseModel):
    strategy: str

class ClearCooldownRequest(BaseModel):
    backend: Optional[str] = None

@router.get("/", response_class=HTMLResponse)
@router.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    """
    Render main interactive web dashboard.
    """
    auth_status = auth_manager.get_status()
    backend_status = client.get_status() if hasattr(client, "get_status") else {}
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
        auth_status=auth_status,
        backend_status=backend_status,
        port=SERVER_PORT,
        base_url=base_url,
        enforce_keys=api_key_manager.enforce_keys
    )
    return HTMLResponse(content=html_content)

@router.get("/api/backends")
@router.get("/api/providers")
async def get_backends():
    """
    Get current status and configuration of all backend adapters.
    """
    if hasattr(client, "get_status"):
        return client.get_status()
    return {"routing_strategy": "free_first", "backends": {}}

@router.post("/api/backends")
@router.post("/api/providers")
async def update_backends(payload: UpdateBackendsRequest):
    """
    Update backend configurations and routing strategy.
    """
    if hasattr(client, "update_config"):
        updates = payload.model_dump(exclude_unset=True, exclude_none=True)
        updated_status = client.update_config(updates)
        return {
            "status": "updated",
            "config": updated_status
        }
    return {"status": "error", "message": "Router not available"}

@router.post("/api/backends/{backend_id}/toggle")
async def toggle_backend(backend_id: str, payload: ToggleBackendRequest):
    """
    Toggle a specific backend adapter on or off.
    """
    if hasattr(client, "get_adapter"):
        adapter = client.get_adapter(backend_id)
        if not adapter:
            raise HTTPException(status_code=404, detail=f"Backend '{backend_id}' not found")
        
        adapter.enabled = payload.enabled
        key_name = f"{backend_id}_enabled"
        if hasattr(client, "save_config"):
            client.save_config()
        return {
            "status": "updated",
            "backend": backend_id,
            "enabled": adapter.enabled,
            "config": client.get_status()
        }
    raise HTTPException(status_code=500, detail="Router not available")

@router.post("/api/backends/strategy")
async def set_routing_strategy(payload: SetStrategyRequest):
    """
    Set active routing strategy: 'free_first' or 'round_robin'.
    """
    strategy = payload.strategy.strip().lower()
    if strategy not in ("free_first", "round_robin"):
        raise HTTPException(status_code=400, detail="Strategy must be 'free_first' or 'round_robin'")
    
    if hasattr(client, "update_config"):
        res = client.update_config({"routing_strategy": strategy})
        return {"status": "updated", "routing_strategy": strategy, "config": res}
    
    raise HTTPException(status_code=500, detail="Router not available")

@router.post("/api/backends/cooldown/clear")
async def clear_cooldowns(payload: Optional[ClearCooldownRequest] = None):
    """
    Clear rate limit cooldown for a specific backend or all backends.
    """
    backend = payload.backend if payload else None
    if hasattr(client, "clear_all_cooldowns"):
        client.clear_all_cooldowns(backend)
        return {"status": "cleared", "backend": backend or "all", "config": client.get_status()}
    return {"status": "error"}

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
    backend_status = client.get_status() if hasattr(client, "get_status") else {}
    return {
        "status": "healthy",
        "service": "agy-to-api",
        "authenticated": status["authenticated"],
        "project_id": status["project_id"],
        "api_key_enforcement": api_key_manager.enforce_keys,
        "routing_strategy": backend_status.get("routing_strategy", "free_first"),
        "backends": backend_status.get("backends", {})
    }
