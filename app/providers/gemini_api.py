import json
import logging
import os
import time
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from app.config import DEPRECATED_MODELS, MODEL_CACHE_TTL, PROVIDER_RATE_LIMITS
from app.providers.base import BaseAdapter, RateLimitError

logger = logging.getLogger("google_gate.providers.gemini_api")

# Fallback models for Google AI Studio API when unconfigured or offline
FALLBACK_MODELS = {
    "gemini-2.0-flash": {
        "displayName": "Gemini 2.0 Flash",
        "maxTokens": 1048576,
        "supportsThinking": False,
        "capabilities": ["tools", "vision"],
    },
    "gemini-2.0-flash-lite": {
        "displayName": "Gemini 2.0 Flash Lite",
        "maxTokens": 1048576,
        "supportsThinking": False,
        "capabilities": ["tools", "vision"],
    },
    "gemini-2.0-pro-exp-02-05": {
        "displayName": "Gemini 2.0 Pro Experimental",
        "maxTokens": 2097152,
        "supportsThinking": True,
        "capabilities": ["thinking", "tools", "vision"],
    },
    "gemini-2.0-flash-thinking-exp-01-21": {
        "displayName": "Gemini 2.0 Flash Thinking",
        "maxTokens": 1048576,
        "supportsThinking": True,
        "capabilities": ["thinking", "tools", "vision"],
    },
    "gemini-1.5-pro": {
        "displayName": "Gemini 1.5 Pro",
        "maxTokens": 2097152,
        "supportsThinking": False,
        "capabilities": ["tools", "vision"],
    },
    "gemini-1.5-flash": {
        "displayName": "Gemini 1.5 Flash",
        "maxTokens": 1048576,
        "supportsThinking": False,
        "capabilities": ["tools", "vision"],
    },
    "gemini-1.5-flash-8b": {
        "displayName": "Gemini 1.5 Flash 8B",
        "maxTokens": 1048576,
        "supportsThinking": False,
        "capabilities": ["tools", "vision"],
    },
    "gemini-1.0-pro": {
        "displayName": "Gemini 1.0 Pro",
        "maxTokens": 32768,
        "supportsThinking": False,
        "capabilities": ["tools"],
    },
    "text-embedding-004": {
        "displayName": "Text Embedding 004",
        "maxTokens": 2048,
        "supportsThinking": False,
        "capabilities": ["embeddings"],
    },
}


def _extract_retry_after(resp: httpx.Response, default: float = 60.0) -> float:
    header = resp.headers.get("retry-after") or resp.headers.get("Retry-After")
    if header:
        try:
            return max(1.0, float(header.strip()))
        except ValueError:
            pass
    return default


class GeminiApiAdapter(BaseAdapter):
    name = "gemini_api"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        enabled: bool = False,
        model_cache_ttl: float = MODEL_CACHE_TTL,
    ):
        limits = PROVIDER_RATE_LIMITS.get("gemini_api", {})
        super().__init__(
            enabled=enabled,
            rpm=limits.get("rpm", 15),
            tpm=limits.get("tpm", 1000000),
            rpd=limits.get("rpd", 1500),
            default_cooldown=limits.get("default_cooldown", 60.0),
            min_quota_fraction=limits.get("min_quota_fraction", 0.0),
            model_cache_ttl=model_cache_ttl,
        )
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.plan_tier: str = "Unknown"
        self.is_valid_key: bool | None = None
        self._plan_probed_at: float = 0.0
        self._http_client: httpx.AsyncClient | None = None
        self._cached_models: dict[str, Any] | None = None
        self._models_fetched_at: float = 0.0

    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_key.strip())

    def reset_credentials(self) -> None:
        """Clear API key, cached models, and plan metadata."""
        self.api_key = ""
        self.enabled = False
        self.plan_tier = "Unknown"
        self.is_valid_key = None
        self._cached_models = None
        self._models_fetched_at = 0.0
        self._plan_probed_at = 0.0
        self.clear_cooldown()
        if hasattr(self, "rate_limiter"):
            self.rate_limiter.reset()

    async def probe_plan(self, force_refresh: bool = False) -> dict[str, Any]:
        """
        Probe Google AI Studio API key plan tier:
        POST https://generativelanguage.googleapis.com/v1beta/cachedContents?key=<API_KEY>
          - HTTP 400 FAILED_PRECONDITION -> Free
          - HTTP 400 INVALID_ARGUMENT or HTTP 200 -> Pay-As-You-Go
          - HTTP 400 API_KEY_INVALID -> Invalid
          - Other / network error -> Fallback gracefully
        """
        if not self.is_configured():
            self.plan_tier = "Unknown"
            self.is_valid_key = False
            return {"plan_tier": self.plan_tier, "valid": False}

        now = time.time()
        if (
            not force_refresh
            and self._plan_probed_at > 0
            and (now - self._plan_probed_at < self.model_cache_ttl)
        ):
            return {
                "plan_tier": self.plan_tier,
                "valid": bool(
                    self.is_valid_key if self.is_valid_key is not None else True
                ),
            }

        url = f"{self.base_url}/cachedContents?key={self.api_key}"
        http = self.get_http_client()
        try:
            resp = await http.post(url, json={}, headers=self._get_headers())
            text = resp.text

            if resp.status_code == 200:
                self.plan_tier = "Pay-As-You-Go"
                self.is_valid_key = True
            elif resp.status_code == 400:
                if "API_KEY_INVALID" in text or "API key not valid" in text:
                    self.plan_tier = "Unknown"
                    self.is_valid_key = False
                elif "FAILED_PRECONDITION" in text:
                    self.plan_tier = "Free"
                    self.is_valid_key = True
                elif "INVALID_ARGUMENT" in text:
                    self.plan_tier = "Pay-As-You-Go"
                    self.is_valid_key = True
                else:
                    self.plan_tier = "Unknown"
                    self.is_valid_key = True
            elif resp.status_code in (401, 403):
                self.plan_tier = "Unknown"
                self.is_valid_key = False
            else:
                self.plan_tier = "Unknown"
                self.is_valid_key = True
        except Exception as e:
            logger.warning(f"Error probing Gemini API plan tier: {e}")
            if self.is_valid_key is None:
                self.is_valid_key = True

        self._plan_probed_at = now
        return {
            "plan_tier": self.plan_tier,
            "valid": bool(self.is_valid_key if self.is_valid_key is not None else True),
        }

    def get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(300.0, connect=30.0)
            )
        return self._http_client

    def _normalize_model_name(self, model: str) -> str:
        """Map internal/alias names to standard Gemini API model names."""
        clean = model.replace("models/", "")
        if self._cached_models and clean in self._cached_models.get("models", {}):
            return clean

        mapping = {
            "gemini-3.7-flash-high": "gemini-2.0-flash",
            "gemini-3.7-flash-medium": "gemini-2.0-flash",
            "gemini-3.7-flash-low": "gemini-2.0-flash",
            "gemini-3.7-flash-image": "gemini-2.0-flash",
            "vision": "gemini-2.0-flash",
            "gemini-3.6-flash-high": "gemini-2.0-flash",
            "gemini-3.6-flash-medium": "gemini-2.0-flash",
            "gemini-3.1-pro-high": "gemini-1.5-pro",
            "gemini-3-flash-agent": "gemini-2.0-flash",
            "gpt-4o": "gemini-2.0-flash",
            "gpt-4o-mini": "gemini-2.0-flash",
            "claude-sonnet-4-6": "gemini-1.5-pro",
            "claude-opus-4-6-thinking": "gemini-1.5-pro",
            "gpt-oss-120b-medium": "gemini-2.0-flash",
        }
        return mapping.get(clean, clean)

    def _get_headers(self) -> dict[str, str]:
        return {
            "x-goog-api-key": self.api_key or "",
            "Content-Type": "application/json",
            "User-Agent": "google-gate/1.0.0 (GeminiApiAdapter)",
        }

    async def fetch_available_models(
        self, force_refresh: bool = False
    ) -> dict[str, Any]:
        """Fetch available models from Google AI Studio with TTL caching."""
        now = time.time()
        if (
            self._cached_models
            and not force_refresh
            and (now - self._models_fetched_at < self.model_cache_ttl)
        ):
            return self._cached_models

        if not self.is_configured():
            self._cached_models = {"models": FALLBACK_MODELS}
            self._models_fetched_at = now
            return self._cached_models

        # Trigger plan probe if not yet probed
        if self._plan_probed_at == 0 or force_refresh:
            try:
                await self.probe_plan(force_refresh=force_refresh)
            except Exception as e:
                logger.debug(f"Plan probe in fetch_available_models skipped: {e}")

        http = self.get_http_client()
        models_dict = {}
        page_token = None
        page_count = 0

        try:
            while True:
                params = {"pageSize": 100}
                if page_token:
                    params["pageToken"] = page_token

                url = f"{self.base_url}/models"
                resp = await http.get(url, params=params, headers=self._get_headers())
                if resp.status_code == 429:
                    retry_after = _extract_retry_after(resp, self.default_cooldown)
                    self.set_cooldown(retry_after)
                    if self._cached_models:
                        return self._cached_models
                    return {"models": models_dict or FALLBACK_MODELS}

                if resp.status_code != 200:
                    logger.warning(
                        f"Gemini API /models returned status {resp.status_code}: {resp.text[:200]}"
                    )
                    break

                data = resp.json()
                for m in data.get("models", []):
                    m_name = m.get("name", "").replace("models/", "")
                    if not m_name or m_name in DEPRECATED_MODELS:
                        continue

                    methods = m.get("supportedGenerationMethods", [])
                    is_embedding = (
                        "embedContent" in methods or "embedding" in m_name.lower()
                    )
                    supports_thinking = bool(
                        "thinking" in m_name.lower()
                        or "2.0-flash-thinking" in m_name.lower()
                        or "3.7" in m_name
                        or "3.6" in m_name
                        or "3.1" in m_name
                    )
                    supports_tools = not is_embedding and (
                        "generateContent" in methods or "gemini" in m_name.lower()
                    )
                    supports_vision = not is_embedding and any(
                        k in m_name.lower()
                        for k in (
                            "flash",
                            "pro",
                            "2.0",
                            "3.0",
                            "3.1",
                            "3.5",
                            "3.6",
                            "3.7",
                            "1.5",
                        )
                    )

                    caps = []
                    if supports_thinking:
                        caps.append("thinking")
                    if supports_tools:
                        caps.append("tools")
                    if supports_vision:
                        caps.append("vision")
                    if is_embedding:
                        caps.append("embeddings")

                    # InputTokenLimit is the context window size in Google AI Studio
                    ctx_window = m.get("inputTokenLimit") or (
                        2097152 if "pro" in m_name.lower() else 1048576
                    )

                    models_dict[m_name] = {
                        "displayName": m.get("displayName", m_name),
                        "maxTokens": ctx_window,
                        "supportsThinking": supports_thinking,
                        "supportsTools": supports_tools,
                        "supportsVision": supports_vision,
                        "isEmbedding": is_embedding,
                        "capabilities": caps,
                    }

                page_token = data.get("nextPageToken")
                page_count += 1
                if not page_token or page_count >= 10:
                    break

            if models_dict:
                self._cached_models = {"models": models_dict}
                self._models_fetched_at = now
                return self._cached_models
            elif self._cached_models:
                return self._cached_models
        except Exception as e:
            logger.warning(f"Failed to fetch models from Gemini API: {e}")

        if self._cached_models:
            return self._cached_models

        self._cached_models = {"models": FALLBACK_MODELS}
        self._models_fetched_at = now
        return self._cached_models

    async def stream_generate_content(
        self,
        model: str,
        contents: list[dict[str, Any]],
        system_instruction: dict[str, Any] | None = None,
        generation_config: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream generated content from Google AI Studio."""
        if not self.is_configured():
            raise ValueError("Gemini API key is not configured.")

        normalized_model = self._normalize_model_name(model)
        payload: dict[str, Any] = {"contents": contents}
        if system_instruction:
            payload["systemInstruction"] = system_instruction
        if generation_config:
            payload["generationConfig"] = generation_config
        if tools:
            payload["tools"] = tools

        url = f"{self.base_url}/models/{normalized_model}:streamGenerateContent?alt=sse"
        http = self.get_http_client()

        logger.info(
            f"[GeminiApi] Sending streamGenerateContent for model={normalized_model}"
        )
        async with http.stream(
            "POST", url, json=payload, headers=self._get_headers()
        ) as resp:
            if resp.status_code == 429:
                retry_after = _extract_retry_after(resp, self.default_cooldown)
                self.set_cooldown(retry_after)
                err_text = (await resp.aread()).decode("utf-8", errors="replace")
                raise RateLimitError(
                    f"Gemini API rate limited (429): {err_text}",
                    status_code=429,
                    retry_after=retry_after,
                )

            if resp.status_code != 200:
                err_text = (await resp.aread()).decode("utf-8", errors="replace")
                logger.error(f"Gemini API error ({resp.status_code}): {err_text}")
                raise ValueError(f"Gemini API Error ({resp.status_code}): {err_text}")

            async for line in resp.aiter_lines():
                line = line.strip()
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str and data_str != "[DONE]":
                        try:
                            parsed = json.loads(data_str)
                            yield parsed
                        except json.JSONDecodeError as e:
                            logger.warning(
                                f"Failed to parse Gemini API SSE JSON: {data_str} ({e})"
                            )

    async def generate_content(
        self,
        model: str,
        contents: list[dict[str, Any]],
        system_instruction: dict[str, Any] | None = None,
        generation_config: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Non-streaming generate content call. Merges SSE stream chunks."""
        candidates_map: dict[int, dict[str, Any]] = {}
        usage_metadata: dict[str, Any] = {}
        finish_reason = "STOP"
        response_id = None
        model_version = model
        last_thought_signature = None

        async for event in self.stream_generate_content(
            model=model,
            contents=contents,
            system_instruction=system_instruction,
            generation_config=generation_config,
            tools=tools,
        ):
            resp_obj = (
                event.get("response")
                if isinstance(event.get("response"), dict)
                else event
            )
            if "responseId" in resp_obj:
                response_id = resp_obj["responseId"]
            if "modelVersion" in resp_obj:
                model_version = resp_obj["modelVersion"]
            if "usageMetadata" in resp_obj:
                usage_metadata.update(resp_obj["usageMetadata"])
            elif "usageMetadata" in event:
                usage_metadata.update(event["usageMetadata"])

            candidates = resp_obj.get("candidates", [])
            for default_idx, cand in enumerate(candidates):
                idx = cand.get("index", default_idx)
                if idx not in candidates_map:
                    candidates_map[idx] = {
                        "text": [],
                        "thoughts": [],
                        "toolCalls": [],
                        "finishReason": "STOP",
                        "thoughtSignature": None,
                    }

                if cand.get("finishReason"):
                    candidates_map[idx]["finishReason"] = cand["finishReason"]
                    finish_reason = cand["finishReason"]

                parts = cand.get("content", {}).get("parts", [])
                for p in parts:
                    if p.get("thoughtSignature"):
                        last_thought_signature = p["thoughtSignature"]
                        candidates_map[idx]["thoughtSignature"] = last_thought_signature
                    if p.get("thought"):
                        candidates_map[idx]["thoughts"].append(p.get("text", ""))
                    elif p.get("text"):
                        candidates_map[idx]["text"].append(p.get("text", ""))
                    if p.get("functionCall"):
                        candidates_map[idx]["toolCalls"].append(p["functionCall"])

        formatted_candidates = []
        if candidates_map:
            for idx in sorted(candidates_map.keys()):
                c = candidates_map[idx]
                formatted_candidates.append(
                    {
                        "index": idx,
                        "text": "".join(c["text"]),
                        "thoughts": "".join(c["thoughts"]),
                        "toolCalls": c["toolCalls"],
                        "finishReason": c["finishReason"],
                        "thoughtSignature": c["thoughtSignature"],
                    }
                )
        else:
            formatted_candidates.append(
                {
                    "index": 0,
                    "text": "",
                    "thoughts": "",
                    "toolCalls": [],
                    "finishReason": finish_reason,
                    "thoughtSignature": None,
                }
            )

        primary = formatted_candidates[0]
        return {
            "responseId": response_id,
            "modelVersion": model_version,
            "candidates": formatted_candidates,
            "text": primary["text"],
            "thoughts": primary["thoughts"],
            "toolCalls": primary["toolCalls"],
            "finishReason": primary["finishReason"],
            "usageMetadata": usage_metadata,
            "thoughtSignature": last_thought_signature,
        }

    async def embed_contents(
        self, model: str, texts: list[str], dimensions: int | None = None
    ) -> dict[str, Any]:
        """Call Gemini API batchEmbedContents."""
        if not self.is_configured():
            raise ValueError("Gemini API key is not configured.")

        clean_model = model.replace("models/", "")
        requests_payload = []
        for text in texts:
            req_item: dict[str, Any] = {
                "model": f"models/{clean_model}",
                "content": {"parts": [{"text": text}]},
            }
            if dimensions is not None:
                req_item["outputDimensionality"] = dimensions
            requests_payload.append(req_item)

        payload = {"requests": requests_payload}
        url = f"{self.base_url}/models/{clean_model}:batchEmbedContents"
        http = self.get_http_client()
        resp = await http.post(url, json=payload, headers=self._get_headers())

        if resp.status_code == 429:
            retry_after = _extract_retry_after(resp, self.default_cooldown)
            self.set_cooldown(retry_after)
            raise RateLimitError(
                f"Gemini API embedding rate limited (429): {resp.text}",
                status_code=429,
                retry_after=retry_after,
            )

        if resp.status_code != 200:
            logger.error(
                f"Gemini API batchEmbedContents failed: {resp.status_code} {resp.text}"
            )
            raise ValueError(f"Gemini API Error ({resp.status_code}): {resp.text}")

        return resp.json()
