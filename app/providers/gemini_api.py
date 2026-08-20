import os
import json
import logging
from typing import AsyncGenerator, Dict, Any, List, Optional
import httpx

from app.providers.base import BaseAdapter, RateLimitError

logger = logging.getLogger("agy_to_api.providers.gemini_api")

class GeminiApiAdapter(BaseAdapter):
    name = "gemini_api"

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        enabled: bool = False
    ):
        super().__init__(enabled=enabled)
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self._http_client: Optional[httpx.AsyncClient] = None
        self._cached_models: Optional[Dict[str, Any]] = None

    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_key.strip())

    def get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=30.0))
        return self._http_client

    def _normalize_model_name(self, model: str) -> str:
        """Map internal/alias names to standard Gemini API model names."""
        clean = model.replace("models/", "")
        mapping = {
            "gemini-3.7-flash-high": "gemini-2.5-flash",
            "gemini-3.7-flash-medium": "gemini-2.5-flash",
            "gemini-3.7-flash-low": "gemini-2.5-flash",
            "gemini-3.6-flash-high": "gemini-2.5-flash",
            "gemini-3.6-flash-medium": "gemini-2.5-flash",
            "gemini-3.1-pro-high": "gemini-2.5-pro",
            "gemini-3-flash-agent": "gemini-2.5-flash",
            "gpt-4o": "gemini-2.5-flash",
            "gpt-4o-mini": "gemini-2.5-flash",
            "claude-sonnet-4-6": "gemini-2.5-pro",
            "claude-opus-4-6-thinking": "gemini-2.5-pro",
            "gpt-oss-120b-medium": "gemini-2.5-flash",
        }
        return mapping.get(clean, clean)

    def _get_headers(self) -> Dict[str, str]:
        return {
            "x-goog-api-key": self.api_key or "",
            "Content-Type": "application/json",
            "User-Agent": "agy-to-api/1.0.0 (GeminiApiAdapter)"
        }

    async def fetch_available_models(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Fetch available models from Google AI Studio."""
        if self._cached_models and not force_refresh:
            return self._cached_models

        fallback_models = {
            "gemini-2.5-flash": {"displayName": "Gemini 2.5 Flash", "maxTokens": 1048576, "supportsThinking": True},
            "gemini-2.5-pro": {"displayName": "Gemini 2.5 Pro", "maxTokens": 2097152, "supportsThinking": True},
            "gemini-2.0-flash": {"displayName": "Gemini 2.0 Flash", "maxTokens": 1048576, "supportsThinking": False},
            "gemini-1.5-pro": {"displayName": "Gemini 1.5 Pro", "maxTokens": 2097152, "supportsThinking": False},
            "gemini-1.5-flash": {"displayName": "Gemini 1.5 Flash", "maxTokens": 1048576, "supportsThinking": False},
            "text-embedding-004": {"displayName": "Text Embedding 004", "maxTokens": 2048, "supportsThinking": False},
        }

        if not self.is_configured():
            return {"models": fallback_models}

        url = f"{self.base_url}/models"
        http = self.get_http_client()
        try:
            resp = await http.get(url, headers=self._get_headers())
            if resp.status_code == 429:
                self.set_cooldown(60.0)
                return {"models": fallback_models}

            if resp.status_code == 200:
                data = resp.json()
                models_dict = {}
                for m in data.get("models", []):
                    m_name = m.get("name", "").replace("models/", "")
                    if m_name:
                        models_dict[m_name] = {
                            "displayName": m.get("displayName", m_name),
                            "maxTokens": m.get("outputTokenLimit", 1048576),
                            "supportsThinking": "2.5" in m_name or "thinking" in m_name
                        }
                self._cached_models = {"models": models_dict or fallback_models}
                return self._cached_models
        except Exception as e:
            logger.warning(f"Failed to fetch models from Gemini API: {e}")

        return {"models": fallback_models}

    async def stream_generate_content(
        self,
        model: str,
        contents: List[Dict[str, Any]],
        system_instruction: Optional[Dict[str, Any]] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream generated content from Google AI Studio."""
        if not self.is_configured():
            raise ValueError("Gemini API key is not configured.")

        normalized_model = self._normalize_model_name(model)
        payload: Dict[str, Any] = {"contents": contents}
        if system_instruction:
            payload["systemInstruction"] = system_instruction
        if generation_config:
            payload["generationConfig"] = generation_config
        if tools:
            payload["tools"] = tools

        url = f"{self.base_url}/models/{normalized_model}:streamGenerateContent?alt=sse"
        http = self.get_http_client()

        logger.info(f"[GeminiApi] Sending streamGenerateContent for model={normalized_model}")
        async with http.stream("POST", url, json=payload, headers=self._get_headers()) as resp:
            if resp.status_code == 429:
                self.set_cooldown(60.0)
                err_text = (await resp.aread()).decode("utf-8", errors="replace")
                raise RateLimitError(f"Gemini API rate limited (429): {err_text}")

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
                            logger.warning(f"Failed to parse Gemini API SSE JSON: {data_str} ({e})")

    async def generate_content(
        self,
        model: str,
        contents: List[Dict[str, Any]],
        system_instruction: Optional[Dict[str, Any]] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Non-streaming generate content call. Merges SSE stream chunks."""
        candidates_map: Dict[int, Dict[str, Any]] = {}
        usage_metadata: Dict[str, Any] = {}
        finish_reason = "STOP"
        response_id = None
        model_version = model
        last_thought_signature = None

        async for event in self.stream_generate_content(
            model=model,
            contents=contents,
            system_instruction=system_instruction,
            generation_config=generation_config,
            tools=tools
        ):
            resp_obj = event.get("response") if isinstance(event.get("response"), dict) else event
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
                        "thoughtSignature": None
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
                formatted_candidates.append({
                    "index": idx,
                    "text": "".join(c["text"]),
                    "thoughts": "".join(c["thoughts"]),
                    "toolCalls": c["toolCalls"],
                    "finishReason": c["finishReason"],
                    "thoughtSignature": c["thoughtSignature"]
                })
        else:
            formatted_candidates.append({
                "index": 0,
                "text": "",
                "thoughts": "",
                "toolCalls": [],
                "finishReason": finish_reason,
                "thoughtSignature": None
            })

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
            "thoughtSignature": last_thought_signature
        }

    async def embed_contents(
        self,
        model: str,
        texts: List[str],
        dimensions: Optional[int] = None
    ) -> Dict[str, Any]:
        """Call Gemini API batchEmbedContents."""
        if not self.is_configured():
            raise ValueError("Gemini API key is not configured.")

        clean_model = model.replace("models/", "")
        requests_payload = []
        for text in texts:
            req_item: Dict[str, Any] = {
                "model": f"models/{clean_model}",
                "content": {"parts": [{"text": text}]}
            }
            if dimensions is not None:
                req_item["outputDimensionality"] = dimensions
            requests_payload.append(req_item)

        payload = {"requests": requests_payload}
        url = f"{self.base_url}/models/{clean_model}:batchEmbedContents"
        http = self.get_http_client()
        resp = await http.post(url, json=payload, headers=self._get_headers())

        if resp.status_code == 429:
            self.set_cooldown(60.0)
            raise RateLimitError(f"Gemini API embedding rate limited (429): {resp.text}")

        if resp.status_code != 200:
            logger.error(f"Gemini API batchEmbedContents failed: {resp.status_code} {resp.text}")
            raise ValueError(f"Gemini API Error ({resp.status_code}): {resp.text}")

        return resp.json()
