import logging
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel

from app.auth import auth_manager
from app.client import client
from app.config import SERVER_PORT
from app.history import history_manager
from app.keys import api_key_manager
from app.transformer import transform_model_catalog, transform_quota_summary

logger = logging.getLogger("google_gate.dashboard")
router = APIRouter(tags=["Dashboard"])

templates_dir = Path(__file__).resolve().parent.parent / "templates"
jinja_env = Environment(loader=FileSystemLoader(str(templates_dir)))


class CreateKeyRequest(BaseModel):
    name: str | None = "New Key"


class ToggleEnforcementRequest(BaseModel):
    enforce: bool


class UpdateBackendsRequest(BaseModel):
    routing_strategy: str | None = None
    gemini_api_key: str | None = None
    gemini_api_enabled: bool | None = None
    gemini_web_psid: str | None = None
    gemini_web_psidts: str | None = None
    gemini_web_sapisid: str | None = None
    gemini_web_enabled: bool | None = None
    antigravity_enabled: bool | None = None


class ToggleBackendRequest(BaseModel):
    enabled: bool


class SetStrategyRequest(BaseModel):
    strategy: str


class ClearCooldownRequest(BaseModel):
    backend: str | None = None


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
        enforce_keys=api_key_manager.enforce_keys,
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
        return {"status": "updated", "config": updated_status}
    return {"status": "error", "message": "Router not available"}


@router.post("/api/backends/{backend_id}/toggle")
async def toggle_backend(backend_id: str, payload: ToggleBackendRequest):
    """
    Toggle a specific backend adapter on or off.
    """
    if hasattr(client, "get_adapter"):
        adapter = client.get_adapter(backend_id)
        if not adapter:
            raise HTTPException(
                status_code=404, detail=f"Backend '{backend_id}' not found"
            )

        adapter.enabled = payload.enabled
        if hasattr(client, "save_config"):
            client.save_config()
        return {
            "status": "updated",
            "backend": backend_id,
            "enabled": adapter.enabled,
            "config": client.get_status(),
        }
    raise HTTPException(status_code=500, detail="Router not available")


@router.post("/api/backends/{backend_id}/reset")
async def reset_backend_credentials(backend_id: str):
    """
    Wipe stored credentials for a backend and disable it.
    """
    if not hasattr(client, "reset_credentials"):
        raise HTTPException(status_code=500, detail="Router not available")
    try:
        status = client.reset_credentials(backend_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return {"status": "reset", "backend": backend_id, "config": status}


@router.post("/api/backends/strategy")
async def set_routing_strategy(payload: SetStrategyRequest):
    """
    Set active routing strategy: 'free_first' or 'round_robin'.
    """
    strategy = payload.strategy.strip().lower()
    if strategy not in ("free_first", "round_robin"):
        raise HTTPException(
            status_code=400, detail="Strategy must be 'free_first' or 'round_robin'"
        )

    if hasattr(client, "update_config"):
        res = client.update_config({"routing_strategy": strategy})
        return {"status": "updated", "routing_strategy": strategy, "config": res}

    raise HTTPException(status_code=500, detail="Router not available")


@router.post("/api/backends/cooldown/clear")
async def clear_cooldowns(payload: ClearCooldownRequest | None = None):
    """
    Clear rate limit cooldown for a specific backend or all backends.
    """
    backend = payload.backend if payload else None
    if hasattr(client, "clear_all_cooldowns"):
        client.clear_all_cooldowns(backend)
        return {
            "status": "cleared",
            "backend": backend or "all",
            "config": client.get_status(),
        }
    return {"status": "error"}


@router.get("/api/keys")
async def get_api_keys():
    """
    List all API keys.
    """
    return {
        "enforce_keys": api_key_manager.enforce_keys,
        "keys": api_key_manager.list_keys(),
    }


@router.post("/api/keys")
async def create_api_key(payload: CreateKeyRequest):
    """
    Generate a new API key.
    """
    new_key = api_key_manager.create_key(name=payload.name or "New Key")
    return {"status": "created", "key": new_key}


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
    return {"status": "updated", "enforce_keys": api_key_manager.enforce_keys}


@router.get("/api/history")
async def get_query_history():
    """
    Retrieve recent query history (up to 50 entries) in reverse chronological order.
    """
    history = history_manager.list_history()
    return {
        "status": "ok",
        "history": history,
        "total": len(history),
    }


@router.delete("/api/history")
async def clear_query_history():
    """
    Clear all in-memory query history records.
    """
    history_manager.clear()
    return {
        "status": "cleared",
        "message": "Query history cleared successfully",
    }


@router.get("/api/quotas")
async def get_quotas():
    """
    Retrieve user quota usage and rate limits across configured backends.
    """
    groups: list[dict[str, Any]] = []
    antigravity_err = None

    # 1. Antigravity Quota
    try:
        data = await client.retrieve_user_quota_summary()
        normalized = transform_quota_summary(data, backend="antigravity")
        groups.extend(normalized.get("groups", []))
    except Exception as e:
        logger.warning(f"Failed to fetch Antigravity quotas: {e}")
        antigravity_err = e

    # 2. Multi-backend rate limit quotas (gemini_api, gemini_web, etc.)
    if hasattr(client, "adapters"):
        for name, adapter in client.adapters.items():
            if name == "antigravity":
                continue
            if not adapter.enabled:
                continue
            try:
                if hasattr(adapter, "get_rate_limit_quotas"):
                    groups.extend(adapter.get_rate_limit_quotas())
            except Exception as e:
                logger.warning(f"Failed to get rate limit quotas for '{name}': {e}")

    if not groups and antigravity_err:
        return JSONResponse(
            status_code=500, content={"error": str(antigravity_err), "groups": []}
        )

    return JSONResponse(content={"groups": groups})


@router.get("/api/models")
async def get_dashboard_models():
    """
    Get simplified and normalized list of models across all active providers.
    """
    try:
        raw_records = []
        if hasattr(client, "adapters"):
            for name, adapter in client.adapters.items():
                if not adapter.enabled:
                    continue
                try:
                    res = await adapter.fetch_available_models()
                    for m_id, m_info in res.get("models", {}).items():
                        raw_id = m_id.replace("models/", "")
                        is_embedding = "embedding" in raw_id.lower()
                        supports_thinking = bool(
                            m_info.get("supportsThinking", False)
                            or "thinking" in raw_id.lower()
                            or "3.7" in raw_id
                        )
                        supports_tools = not is_embedding and any(
                            k in raw_id.lower() for k in ("gemini", "claude", "gpt")
                        )
                        supports_vision = not is_embedding and any(
                            k in raw_id.lower() for k in ("gemini", "claude", "gpt-4")
                        )
                        capabilities = []
                        if supports_thinking:
                            capabilities.append("thinking")
                        if supports_tools:
                            capabilities.append("tools")
                        if supports_vision:
                            capabilities.append("vision")
                        if is_embedding:
                            capabilities.append("embeddings")

                        raw_records.append(
                            {
                                "model_id": raw_id,
                                "raw_name": m_info.get("displayName", raw_id),
                                "context_window": m_info.get("maxTokens", 1048576),
                                "capabilities": capabilities,
                                "source_antigravity": (name == "antigravity"),
                                "source_gemini_api": (name == "gemini_api"),
                                "source_gemini_web": (name == "gemini_web"),
                            }
                        )
                except Exception as e:
                    logger.warning(f"Error fetching models from '{name}': {e}")
        elif hasattr(client, "fetch_available_models"):
            raw = await client.fetch_available_models()
            for m_id, m_info in raw.get("models", {}).items():
                raw_id = m_id.replace("models/", "")
                provs = m_info.get("providers", ["antigravity"])
                raw_records.append(
                    {
                        "model_id": raw_id,
                        "raw_name": m_info.get("displayName", raw_id),
                        "context_window": m_info.get("maxTokens", 1048576),
                        "capabilities": m_info.get("capabilities", []),
                        "source_antigravity": "antigravity" in provs,
                        "source_gemini_api": "gemini_api" in provs,
                        "source_gemini_web": "gemini_web" in provs,
                    }
                )

        transformed = transform_model_catalog(raw_records)
        gemini_models = transformed["gemini_models"]
        the_rest = transformed["the_rest"]
        all_models = gemini_models + the_rest

        return {
            "status": "ok",
            "gemini_models": gemini_models,
            "non_google_models": the_rest,
            "the_rest": the_rest,
            "models": all_models,
            "total": len(all_models),
        }
    except Exception as e:
        logger.warning(f"Failed to fetch models for dashboard: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error": str(e), "models": [], "total": 0},
        )


@router.get("/health")
async def health_check():
    """
    Service health check endpoint.
    """
    status = auth_manager.get_status()
    backend_status = client.get_status() if hasattr(client, "get_status") else {}
    return {
        "status": "ok",
        "service": "google-gate",
        "version": "1.0.0",
        "authenticated": status["authenticated"],
        "project_id": status["project_id"],
        "api_key_enforcement": api_key_manager.enforce_keys,
        "routing_strategy": backend_status.get("routing_strategy", "free_first"),
        "backends": backend_status.get("backends", {}),
    }
