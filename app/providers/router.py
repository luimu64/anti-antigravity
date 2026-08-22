import json
import logging
import os
import re
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from app.auth import OAuthManager, auth_manager
from app.config import (
    CANONICAL_MODEL_MAP,
    CLOUD_CODE_BASE_URL,
    CREDENTIALS_FILE,
    HIDDEN_MODELS,
)
from app.providers.antigravity import AntigravityAdapter
from app.providers.base import BaseAdapter, RateLimitError
from app.providers.gemini_api import GeminiApiAdapter
from app.providers.gemini_web import GeminiWebAdapter

logger = logging.getLogger("agy_to_api.providers.router")


def _extract_model_version(model_id: str) -> tuple[int, ...]:
    """
    Extract major/minor version numbers from a model ID string for semantic sorting.
    Examples:
      'gemini-3.7-flash' -> (3, 7, 0, 0)
      'gemini-3.5-pro' -> (3, 5, 0, 0)
      'gemini-2.5-flash-thinking' -> (2, 5, 0, 0)
      'claude-sonnet-4-6' -> (4, 6, 0, 0)
      'gpt-oss-120b-medium' -> (120, 0, 0, 0)
    """
    # 1. Look for patterns like '3.7', '2.5', '1.5'
    match_dot = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", model_id)
    if match_dot:
        major = int(match_dot.group(1))
        minor = int(match_dot.group(2))
        patch = int(match_dot.group(3)) if match_dot.group(3) else 0
        return (major, minor, patch, 0)

    # 2. Look for dash version patterns like '4-6', '3-5' (Claude/GPT style)
    match_dash = re.search(
        r"(?:sonnet|opus|haiku|gpt)[^\d]*(\d+)-(\d+)", model_id, re.IGNORECASE
    )
    if match_dash:
        return (int(match_dash.group(1)), int(match_dash.group(2)), 0, 0)

    # 3. Look for number prefixes or standalone numbers like '120b', '3'
    match_num = re.search(r"(\d+)", model_id)
    if match_num:
        try:
            return (int(match_num.group(1)), 0, 0, 0)
        except Exception:
            pass

    return (0, 0, 0, 0)


class MultiBackendRouter(BaseAdapter):
    name = "router"

    def __init__(
        self,
        antigravity: AntigravityAdapter | None = None,
        gemini_api: GeminiApiAdapter | None = None,
        gemini_web: GeminiWebAdapter | None = None,
        routing_strategy: str = "free_first",
    ):
        super().__init__(enabled=True)
        custom_instances = bool(antigravity or gemini_api or gemini_web)
        self.antigravity = antigravity or AntigravityAdapter()
        self.gemini_api = gemini_api or GeminiApiAdapter()
        self.gemini_web = gemini_web or GeminiWebAdapter()

        self.adapters: dict[str, BaseAdapter] = {
            "antigravity": self.antigravity,
            "gemini_api": self.gemini_api,
            "gemini_web": self.gemini_web,
        }

        self.routing_strategy = routing_strategy
        self._rr_counter = 0

        # Backwards-compatibility attributes for direct AntigravityClient callers
        self.auth: OAuthManager = getattr(self.antigravity, "auth", auth_manager)
        self.base_url: str = CLOUD_CODE_BASE_URL

        # Load persisted credentials & configuration if using default instances
        if not custom_instances:
            self.load_config()

    def is_configured(self) -> bool:
        return any(a.is_configured() for a in self.adapters.values() if a.enabled)

    def load_config(self) -> None:
        """Load backend configs and routing strategy from environment and credentials file."""
        # 1. Environment variables
        env_strategy = os.getenv("ROUTING_STRATEGY")
        if env_strategy in ("free_first", "round_robin"):
            self.routing_strategy = env_strategy

        env_api_key = os.getenv("GEMINI_API_KEY")
        if env_api_key:
            self.gemini_api.api_key = env_api_key

        if os.getenv("ANTIGRAVITY_ENABLED"):
            self.antigravity.enabled = os.getenv("ANTIGRAVITY_ENABLED", "").lower() in (
                "true",
                "1",
            )
        if os.getenv("GEMINI_API_ENABLED"):
            self.gemini_api.enabled = os.getenv("GEMINI_API_ENABLED", "").lower() in (
                "true",
                "1",
            )
        if os.getenv("GEMINI_WEB_ENABLED"):
            self.gemini_web.enabled = os.getenv("GEMINI_WEB_ENABLED", "").lower() in (
                "true",
                "1",
            )

        env_psid = (
            os.getenv("GEMINI_WEB_PSID")
            or os.getenv("SECURE_1PSID")
            or os.environ.get("__Secure-1PSID")
        )
        if env_psid:
            self.gemini_web.psid = env_psid

        env_psidts = (
            os.getenv("GEMINI_WEB_PSIDTS")
            or os.getenv("SECURE_1PSIDTS")
            or os.environ.get("__Secure-1PSIDTS")
        )
        if env_psidts:
            self.gemini_web.psidts = env_psidts

        env_sapisid = (
            os.getenv("GEMINI_WEB_SAPISID")
            or os.getenv("SAPISID")
            or os.getenv("SECURE_3PSID")
            or os.environ.get("__Secure-3PSID")
        )
        if env_sapisid:
            self.gemini_web.sapisid = env_sapisid

        # 2. Local credentials file
        if os.path.exists(CREDENTIALS_FILE):
            try:
                with open(CREDENTIALS_FILE) as f:
                    data = json.load(f)

                if "routing_strategy" in data and data["routing_strategy"] in (
                    "free_first",
                    "round_robin",
                ):
                    self.routing_strategy = data["routing_strategy"]

                if data.get("gemini_api_key"):
                    self.gemini_api.api_key = data["gemini_api_key"]
                if "gemini_api_enabled" in data:
                    self.gemini_api.enabled = bool(data["gemini_api_enabled"])

                if data.get("gemini_web_psid"):
                    self.gemini_web.psid = data["gemini_web_psid"]
                if data.get("gemini_web_psidts"):
                    self.gemini_web.psidts = data["gemini_web_psidts"]
                if data.get("gemini_web_sapisid"):
                    self.gemini_web.sapisid = data["gemini_web_sapisid"]
                if "gemini_web_enabled" in data:
                    self.gemini_web.enabled = bool(data["gemini_web_enabled"])

                if "antigravity_enabled" in data:
                    self.antigravity.enabled = bool(data["antigravity_enabled"])

            except Exception as e:
                logger.warning(
                    f"Failed to load backend config from {CREDENTIALS_FILE}: {e}"
                )

    def save_config(self) -> None:
        """Persist multi-backend credentials and settings to credentials file."""
        existing_data = {}
        if os.path.exists(CREDENTIALS_FILE):
            try:
                with open(CREDENTIALS_FILE) as f:
                    existing_data = json.load(f)
            except Exception:
                existing_data = {}

        existing_data.update(
            {
                "routing_strategy": self.routing_strategy,
                "gemini_api_key": self.gemini_api.api_key,
                "gemini_api_enabled": self.gemini_api.enabled,
                "gemini_web_psid": self.gemini_web.psid,
                "gemini_web_psidts": self.gemini_web.psidts,
                "gemini_web_sapisid": self.gemini_web.sapisid,
                "gemini_web_enabled": self.gemini_web.enabled,
                "antigravity_enabled": self.antigravity.enabled,
            }
        )

        try:
            os.makedirs(os.path.dirname(CREDENTIALS_FILE), exist_ok=True)
            with open(CREDENTIALS_FILE, "w") as f:
                json.dump(existing_data, f, indent=2)
            logger.info(f"Saved multi-backend configuration to {CREDENTIALS_FILE}")
        except Exception as e:
            logger.error(f"Failed to save backend config to {CREDENTIALS_FILE}: {e}")

    def update_config(self, updates: dict[str, Any]) -> dict[str, Any]:
        """Apply updates to backends/strategy and persist."""
        if "routing_strategy" in updates and updates["routing_strategy"] in (
            "free_first",
            "round_robin",
        ):
            self.routing_strategy = updates["routing_strategy"]

        if "gemini_api_key" in updates and updates["gemini_api_key"] is not None:
            self.gemini_api.api_key = str(updates["gemini_api_key"]).strip()
        if (
            "gemini_api_enabled" in updates
            and updates["gemini_api_enabled"] is not None
        ):
            self.gemini_api.enabled = bool(updates["gemini_api_enabled"])

        if "gemini_web_psid" in updates and updates["gemini_web_psid"] is not None:
            self.gemini_web.psid = str(updates["gemini_web_psid"]).strip()
        if "gemini_web_psidts" in updates and updates["gemini_web_psidts"] is not None:
            self.gemini_web.psidts = str(updates["gemini_web_psidts"]).strip()
        if (
            "gemini_web_sapisid" in updates
            and updates["gemini_web_sapisid"] is not None
        ):
            self.gemini_web.sapisid = str(updates["gemini_web_sapisid"]).strip()
        if (
            "gemini_web_enabled" in updates
            and updates["gemini_web_enabled"] is not None
        ):
            self.gemini_web.enabled = bool(updates["gemini_web_enabled"])

        if (
            "antigravity_enabled" in updates
            and updates["antigravity_enabled"] is not None
        ):
            self.antigravity.enabled = bool(updates["antigravity_enabled"])

        self.save_config()
        return self.get_status()

    def get_status(self) -> dict[str, Any]:
        """Get status of all backends and routing configuration."""

        def mask_secret(s: str | None) -> str:
            if not s:
                return ""
            if len(s) > 10:
                return f"{s[:6]}...{s[-4:]}"
            return "***"

        return {
            "routing_strategy": self.routing_strategy,
            "backends": {
                "antigravity": {
                    "id": "antigravity",
                    "name": "Antigravity (OAuth)",
                    "enabled": self.antigravity.enabled,
                    "configured": self.antigravity.is_configured(),
                    "available": self.antigravity.is_available(),
                    "cooldown_remaining": round(
                        self.antigravity.get_cooldown_remaining(), 1
                    ),
                    "authenticated": bool(
                        self.antigravity.auth.access_token
                        or self.antigravity.auth.refresh_token
                    ),
                    "user_email": self.antigravity.auth.user_email,
                    "tier_name": self.antigravity.auth.tier_name,
                    "project_id": self.antigravity.auth.project_id,
                },
                "gemini_api": {
                    "id": "gemini_api",
                    "name": "Google Gemini AI Studio API",
                    "enabled": self.gemini_api.enabled,
                    "configured": self.gemini_api.is_configured(),
                    "available": self.gemini_api.is_available(),
                    "cooldown_remaining": round(
                        self.gemini_api.get_cooldown_remaining(), 1
                    ),
                    "has_api_key": bool(self.gemini_api.api_key),
                    "masked_key": mask_secret(self.gemini_api.api_key),
                },
                "gemini_web": {
                    "id": "gemini_web",
                    "name": "Google Gemini Web UI (Cookies)",
                    "enabled": self.gemini_web.enabled,
                    "configured": self.gemini_web.is_configured(),
                    "available": self.gemini_web.is_available(),
                    "cooldown_remaining": round(
                        self.gemini_web.get_cooldown_remaining(), 1
                    ),
                    "has_psid": bool(self.gemini_web.psid),
                    "has_psidts": bool(self.gemini_web.psidts),
                    "masked_psid": mask_secret(self.gemini_web.psid),
                },
            },
        }

    def clear_all_cooldowns(self, backend: str | None = None) -> None:
        """Clear cooldown status for specific backend or all backends."""
        if backend and backend in self.adapters:
            self.adapters[backend].clear_cooldown()
        else:
            for a in self.adapters.values():
                a.clear_cooldown()

    def get_adapter(self, name: str) -> BaseAdapter | None:
        return self.adapters.get(name)

    def get_available_adapters(self) -> list[BaseAdapter]:
        available = [a for a in self.adapters.values() if a.is_available()]
        if not available:
            # If all are in cooldown but configured & enabled, fallback to configured
            available = [
                a for a in self.adapters.values() if a.enabled and a.is_configured()
            ]
        return available

    def get_ordered_adapters(self) -> list[BaseAdapter]:
        """Order adapters according to the selected routing strategy."""
        if self.routing_strategy == "round_robin":
            available = self.get_available_adapters()
            if not available:
                return [self.antigravity]
            idx = self._rr_counter % len(available)
            self._rr_counter += 1
            return available[idx:] + available[:idx]

        # Default "free_first": Gemini Web -> Gemini AI Studio API -> Antigravity
        priority_order = [self.gemini_web, self.gemini_api, self.antigravity]
        available = [a for a in priority_order if a.is_available()]
        if not available:
            # Fallback to any enabled/configured
            available = [a for a in priority_order if a.enabled and a.is_configured()]
        return available or [self.antigravity]

    def _is_rate_limit_exception(self, e: Exception) -> bool:
        """Check if exception represents a 429 / rate limit / quota exceeded."""
        if isinstance(e, RateLimitError):
            return True
        if isinstance(e, httpx.HTTPStatusError) and e.response.status_code == 429:
            return True
        msg = str(e).lower()
        return (
            "429" in msg
            or "resource_exhausted" in msg
            or "quota" in msg
            or "too many requests" in msg
        )

    async def generate_content(
        self,
        model: str,
        contents: list[dict[str, Any]],
        system_instruction: dict[str, Any] | None = None,
        generation_config: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Execute non-streaming content generation with automatic 429 fallback."""
        candidates = self.get_ordered_adapters()
        last_exception = None

        for adapter in candidates:
            try:
                logger.info(
                    f"Routing generate_content (strategy={self.routing_strategy}) to '{adapter.name}'"
                )
                return await adapter.generate_content(
                    model=model,
                    contents=contents,
                    system_instruction=system_instruction,
                    generation_config=generation_config,
                    tools=tools,
                )
            except Exception as e:
                if self._is_rate_limit_exception(e):
                    retry_after = getattr(e, "retry_after", 60.0) or 60.0
                    logger.warning(
                        f"Backend '{adapter.name}' hit rate limit: {e}. Falling back to next backend..."
                    )
                    adapter.set_cooldown(retry_after)
                else:
                    logger.warning(
                        f"Backend '{adapter.name}' failed with error: {e}. Attempting fallback..."
                    )
                last_exception = e
                continue

        if last_exception:
            raise last_exception
        raise ValueError("No backends available to fulfill generate_content request.")

    async def stream_generate_content(
        self,
        model: str,
        contents: list[dict[str, Any]],
        system_instruction: dict[str, Any] | None = None,
        generation_config: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream generated content with automatic fallback if initial connect fails with 429."""
        candidates = self.get_ordered_adapters()
        last_exception = None

        for adapter in candidates:
            success = False
            try:
                logger.info(
                    f"Routing stream_generate_content (strategy={self.routing_strategy}) to '{adapter.name}'"
                )
                stream_gen = adapter.stream_generate_content(
                    model=model,
                    contents=contents,
                    system_instruction=system_instruction,
                    generation_config=generation_config,
                    tools=tools,
                )

                # Peek first item to ensure connection and catch immediate rate limit
                async for chunk in stream_gen:
                    yield chunk
                    success = True
                    break

                if success:
                    async for chunk in stream_gen:
                        yield chunk
                    return

            except Exception as e:
                is_rate_limit = self._is_rate_limit_exception(e)
                if is_rate_limit:
                    retry_after = getattr(e, "retry_after", 60.0) or 60.0
                    logger.warning(
                        f"Backend '{adapter.name}' streaming hit rate limit: {e}. Falling back..."
                    )
                    adapter.set_cooldown(retry_after)

                if not success:
                    if not is_rate_limit:
                        logger.warning(
                            f"Backend '{adapter.name}' stream failed before data: {e}. Falling back..."
                        )
                    last_exception = e
                    continue
                else:
                    logger.error(
                        f"Backend '{adapter.name}' stream broke mid-generation: {e}"
                    )
                    raise

        if last_exception:
            raise last_exception
        raise ValueError(
            "No backends available to fulfill stream_generate_content request."
        )

    async def embed_contents(
        self, model: str, texts: list[str], dimensions: int | None = None
    ) -> dict[str, Any]:
        """Route embedding requests to available backend supporting embeddings."""
        candidates = [
            a for a in self.get_ordered_adapters() if hasattr(a, "embed_contents")
        ]
        last_exception = None

        for adapter in candidates:
            try:
                return await adapter.embed_contents(
                    model=model, texts=texts, dimensions=dimensions
                )
            except Exception as e:
                if self._is_rate_limit_exception(e):
                    retry_after = getattr(e, "retry_after", 60.0) or 60.0
                    logger.warning(
                        f"Backend '{adapter.name}' embeddings hit rate limit: {e}. Falling back..."
                    )
                    adapter.set_cooldown(retry_after)
                else:
                    logger.warning(
                        f"Backend '{adapter.name}' embeddings failed: {e}. Falling back..."
                    )
                last_exception = e
                continue

        if last_exception:
            raise last_exception
        raise ValueError("No backends available for embeddings.")

    async def fetch_available_models(
        self, force_refresh: bool = False
    ) -> dict[str, Any]:
        """
        Aggregate models across all active providers, count provider support,
        and sort models supported across the highest number of active providers first.
        """
        aggregated_models: dict[str, dict[str, Any]] = {}
        provider_counts: dict[str, int] = {}

        # Fetch models from all enabled adapters
        for name, adapter in self.adapters.items():
            if not adapter.enabled:
                continue
            try:
                res = await adapter.fetch_available_models(force_refresh=force_refresh)
                models_dict = res.get("models", {})
                for m_id, m_info in models_dict.items():
                    raw_id = m_id.replace("models/", "")

                    # Consolidate Antigravity reasoning tiers into clean base models
                    canon_id = raw_id
                    if adapter.name == "antigravity" and raw_id in CANONICAL_MODEL_MAP:
                        canon_id = CANONICAL_MODEL_MAP[raw_id]

                    is_hidden = bool(
                        raw_id in HIDDEN_MODELS
                        or canon_id in HIDDEN_MODELS
                        or raw_id.startswith("tab_")
                        or canon_id.startswith("tab_")
                    )

                    is_embedding = "embedding" in canon_id.lower()
                    supports_thinking = bool(
                        m_info.get("supportsThinking", False)
                        or "thinking" in canon_id.lower()
                        or "3.7" in canon_id
                    )
                    supports_tools = not is_embedding and any(
                        k in canon_id.lower() for k in ("gemini", "claude", "gpt")
                    )
                    supports_vision = not is_embedding and any(
                        k in canon_id.lower() for k in ("gemini", "claude", "gpt-4")
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

                    # Friendly display name
                    disp_name = m_info.get("displayName", canon_id)
                    if canon_id == "gemini-3.7-flash":
                        disp_name = "Gemini 3.7 Flash"
                    elif canon_id == "gemini-3.6-flash":
                        disp_name = "Gemini 3.6 Flash"
                    elif canon_id == "gemini-3.5-flash":
                        disp_name = "Gemini 3.5 Flash"
                    elif canon_id == "gemini-3.1-pro":
                        disp_name = "Gemini 3.1 Pro"
                    elif canon_id == "gpt-oss-120b":
                        disp_name = "GPT-OSS 120B"
                    elif canon_id == "claude-3.7-sonnet":
                        disp_name = "Claude 3.7 Sonnet"
                    elif canon_id == "claude-3-opus":
                        disp_name = "Claude 3 Opus"

                    if canon_id not in aggregated_models:
                        aggregated_models[canon_id] = {
                            "displayName": disp_name,
                            "maxTokens": m_info.get("maxTokens", 1048576),
                            "supportsThinking": supports_thinking,
                            "supportsTools": supports_tools,
                            "supportsVision": supports_vision,
                            "isEmbedding": is_embedding,
                            "capabilities": capabilities,
                            "hidden": is_hidden,
                            "providers": [],
                        }

                    if not is_hidden:
                        aggregated_models[canon_id]["hidden"] = False

                    if name not in aggregated_models[canon_id]["providers"]:
                        aggregated_models[canon_id]["providers"].append(name)
                        provider_counts[canon_id] = provider_counts.get(canon_id, 0) + 1
            except Exception as e:
                logger.warning(f"Error fetching models from '{name}': {e}")

        # If empty (e.g. offline), ensure fallback
        if not aggregated_models:
            res = await self.antigravity.fetch_available_models(
                force_refresh=force_refresh
            )
            for m_id, m_info in res.get("models", {}).items():
                raw_id = m_id.replace("models/", "")
                canon_id = CANONICAL_MODEL_MAP.get(raw_id, raw_id)
                is_hidden = bool(
                    raw_id in HIDDEN_MODELS
                    or canon_id in HIDDEN_MODELS
                    or raw_id.startswith("tab_")
                    or canon_id.startswith("tab_")
                )
                is_embedding = "embedding" in canon_id.lower()
                supports_thinking = bool(
                    m_info.get("supportsThinking", False)
                    or "thinking" in canon_id.lower()
                    or "3.7" in canon_id
                )
                supports_tools = not is_embedding and any(
                    k in canon_id.lower() for k in ("gemini", "claude", "gpt")
                )
                supports_vision = not is_embedding and any(
                    k in canon_id.lower() for k in ("gemini", "claude", "gpt-4")
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

                disp_name = m_info.get("displayName", canon_id)
                if canon_id == "gemini-3.7-flash":
                    disp_name = "Gemini 3.7 Flash"
                elif canon_id == "gemini-3.6-flash":
                    disp_name = "Gemini 3.6 Flash"
                elif canon_id == "gemini-3.5-flash":
                    disp_name = "Gemini 3.5 Flash"
                elif canon_id == "gemini-3.1-pro":
                    disp_name = "Gemini 3.1 Pro"
                elif canon_id == "gpt-oss-120b":
                    disp_name = "GPT-OSS 120B"

                aggregated_models[canon_id] = {
                    "displayName": disp_name,
                    "maxTokens": m_info.get("maxTokens", 1048576),
                    "supportsThinking": supports_thinking,
                    "supportsTools": supports_tools,
                    "supportsVision": supports_vision,
                    "isEmbedding": is_embedding,
                    "capabilities": capabilities,
                    "hidden": is_hidden,
                    "providers": ["antigravity"],
                }
                provider_counts[canon_id] = 1

        # Sort models descending by provider support count (redundancy), then by version (newness), then by model ID
        sorted_model_keys = sorted(
            aggregated_models.keys(),
            key=lambda m: (
                -provider_counts.get(m, 0),
                tuple(-x for x in _extract_model_version(m)),
                m,
            ),
        )

        sorted_models = {
            m: {
                **aggregated_models[m],
                "provider_count": provider_counts.get(m, 0),
                "providers": aggregated_models[m].get("providers", []),
            }
            for m in sorted_model_keys
        }

        return {"models": sorted_models}

    # Delegation methods for Antigravity-specific calls
    async def load_code_assist(self) -> dict[str, Any]:
        return await self.antigravity.load_code_assist()

    async def retrieve_user_quota_summary(self) -> dict[str, Any]:
        return await self.antigravity.retrieve_user_quota_summary()

    async def get_project_id(self) -> str:
        return await self.antigravity.get_project_id()


# Global router client singleton
router_client = MultiBackendRouter()
