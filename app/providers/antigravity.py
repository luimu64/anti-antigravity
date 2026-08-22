import datetime
import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from app.auth import OAuthManager, auth_manager
from app.config import CLOUD_CODE_BASE_URL, PROVIDER_RATE_LIMITS, USER_AGENT
from app.providers.base import BaseAdapter, RateLimitError

logger = logging.getLogger("agy_to_api.providers.antigravity")


def _extract_retry_after(resp: httpx.Response, default: float = 60.0) -> float:
    header = resp.headers.get("retry-after") or resp.headers.get("Retry-After")
    if header:
        try:
            return max(1.0, float(header.strip()))
        except ValueError:
            pass
    return default


class AntigravityAdapter(BaseAdapter):
    name = "antigravity"

    def __init__(
        self,
        base_url: str = CLOUD_CODE_BASE_URL,
        auth: OAuthManager = auth_manager,
        enabled: bool = False,
    ):
        limits = PROVIDER_RATE_LIMITS.get("antigravity", {})
        super().__init__(
            enabled=enabled,
            rpm=limits.get("rpm", 100),
            tpm=limits.get("tpm", 1000000),
            rpd=limits.get("rpd", 0),
            default_cooldown=limits.get("default_cooldown", 60.0),
            min_quota_fraction=limits.get("min_quota_fraction", 0.01),
        )
        self.base_url = base_url.rstrip("/")
        self.auth = auth
        self._cached_models: dict[str, Any] | None = None
        self._http_client: httpx.AsyncClient | None = None

    def is_configured(self) -> bool:
        return bool(self.auth.refresh_token or self.auth.access_token)

    def get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(300.0, connect=30.0)
            )
        return self._http_client

    async def _get_headers(self) -> dict[str, str]:
        token = await self.auth.get_valid_access_token()
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
            "Accept-Encoding": "gzip, deflate",
        }

    async def load_code_assist(self) -> dict[str, Any]:
        """Call /v1internal:loadCodeAssist to retrieve project ID and tier metadata."""
        headers = await self._get_headers()
        payload = {"metadata": {"ideType": "ANTIGRAVITY"}}
        url = f"{self.base_url}/v1internal:loadCodeAssist"
        http = self.get_http_client()
        resp = await http.post(url, json=payload, headers=headers)
        if resp.status_code == 401 and self.auth.refresh_token:
            logger.info(
                "Received 401 on loadCodeAssist, refreshing token and retrying..."
            )
            await self.auth.refresh_access_token()
            headers = await self._get_headers()
            resp = await http.post(url, json=payload, headers=headers)

        if resp.status_code == 429:
            retry_after = _extract_retry_after(resp, self.default_cooldown)
            self.set_cooldown(retry_after)
            raise RateLimitError(
                f"Antigravity rate limited (429): {resp.text}",
                status_code=429,
                retry_after=retry_after,
            )

        if resp.status_code != 200:
            logger.error(f"loadCodeAssist failed: {resp.status_code} {resp.text}")
            raise ValueError(f"loadCodeAssist failed: {resp.status_code} {resp.text}")

        data = resp.json()
        project_id = data.get("cloudaicompanionProject")
        if project_id and not self.auth.project_id:
            self.auth.project_id = project_id

        current_tier = data.get("currentTier", {})
        if current_tier:
            self.auth.tier_name = current_tier.get("name") or current_tier.get("id")

        user_limits = data.get("userLimits", {})
        if isinstance(user_limits, dict) and "rateLimit" in user_limits:
            try:
                discovered_rpm = int(user_limits["rateLimit"])
                if discovered_rpm > 0 and hasattr(self, "rate_limiter"):
                    self.rate_limiter.rpm = discovered_rpm
            except Exception:
                pass

        self.auth.save_credentials()
        return data

    async def get_project_id(self) -> str:
        """Get or discover active project ID."""
        if self.auth.project_id:
            return self.auth.project_id

        data = await self.load_code_assist()
        project_id = data.get("cloudaicompanionProject")
        if not project_id:
            raise ValueError("Failed to retrieve project ID from Antigravity backend.")
        return project_id

    async def fetch_available_models(
        self, force_refresh: bool = False
    ) -> dict[str, Any]:
        """Call /v1internal:fetchAvailableModels to get all supported models."""
        if self._cached_models and not force_refresh:
            return self._cached_models

        headers = await self._get_headers()
        url = f"{self.base_url}/v1internal:fetchAvailableModels"
        http = self.get_http_client()
        resp = await http.post(url, json={}, headers=headers)
        if resp.status_code == 401 and self.auth.refresh_token:
            logger.info(
                "Received 401 on fetchAvailableModels, refreshing token and retrying..."
            )
            await self.auth.refresh_access_token()
            headers = await self._get_headers()
            resp = await http.post(url, json={}, headers=headers)

        if resp.status_code == 429:
            retry_after = _extract_retry_after(resp, self.default_cooldown)
            self.set_cooldown(retry_after)
            raise RateLimitError(
                f"Antigravity rate limited (429): {resp.text}",
                status_code=429,
                retry_after=retry_after,
            )

        if resp.status_code != 200:
            logger.error(f"fetchAvailableModels failed: {resp.status_code} {resp.text}")
            raise ValueError(
                f"fetchAvailableModels failed: {resp.status_code} {resp.text}"
            )

        self._cached_models = resp.json()
        return self._cached_models

    async def retrieve_user_quota_summary(self) -> dict[str, Any]:
        """Call /v1internal:retrieveUserQuotaSummary to get quota details."""
        headers = await self._get_headers()
        url = f"{self.base_url}/v1internal:retrieveUserQuotaSummary"
        http = self.get_http_client()
        resp = await http.post(url, json={}, headers=headers)
        if resp.status_code == 401 and self.auth.refresh_token:
            logger.info(
                "Received 401 on retrieveUserQuotaSummary, refreshing token and retrying..."
            )
            await self.auth.refresh_access_token()
            headers = await self._get_headers()
            resp = await http.post(url, json={}, headers=headers)

        if resp.status_code == 429:
            retry_after = _extract_retry_after(resp, self.default_cooldown)
            self.set_cooldown(retry_after)
            raise RateLimitError(
                f"Antigravity rate limited (429): {resp.text}",
                status_code=429,
                retry_after=retry_after,
            )

        if resp.status_code != 200:
            logger.error(
                f"retrieveUserQuotaSummary failed: {resp.status_code} {resp.text}"
            )
            raise ValueError(
                f"retrieveUserQuotaSummary failed: {resp.status_code} {resp.text}"
            )

        data = resp.json()
        groups = data.get("groups", [])
        for grp in groups:
            for bucket in grp.get("buckets", []):
                rem = bucket.get("remainingFraction")
                if rem is not None and rem <= self.min_quota_fraction:
                    cooldown_secs = self.default_cooldown
                    reset_time_str = bucket.get("resetTime")
                    if reset_time_str:
                        try:
                            dt = datetime.datetime.fromisoformat(
                                reset_time_str.replace("Z", "+00:00")
                            )
                            delta = (
                                dt.timestamp()
                                - datetime.datetime.now(
                                    datetime.timezone.utc
                                ).timestamp()
                            )
                            if delta > 0:
                                cooldown_secs = min(delta, 86400.0)
                        except Exception:
                            pass
                    self.set_cooldown(cooldown_secs)
                    logger.warning(
                        f"Antigravity quota bucket '{bucket.get('displayName', bucket.get('bucketId'))}' exhausted "
                        f"({rem * 100:.1f}% remaining). Placed in cooldown for {cooldown_secs:.1f}s."
                    )
        return data

    async def stream_generate_content(
        self,
        model: str,
        contents: list[dict[str, Any]],
        system_instruction: dict[str, Any] | None = None,
        generation_config: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Call /v1internal:streamGenerateContent?alt=sse and stream events."""
        headers = await self._get_headers()
        project_id = await self.get_project_id()

        inner_request: dict[str, Any] = {"contents": contents}
        if system_instruction:
            inner_request["systemInstruction"] = system_instruction
        if generation_config:
            inner_request["generationConfig"] = generation_config
        if tools:
            inner_request["tools"] = tools

        payload = {"project": project_id, "model": model, "request": inner_request}

        logger.info(f"[Antigravity] Sending streamGenerateContent for model={model}")
        url = f"{self.base_url}/v1internal:streamGenerateContent?alt=sse"
        http = self.get_http_client()

        async with http.stream("POST", url, json=payload, headers=headers) as resp:
            if resp.status_code == 401 and self.auth.refresh_token:
                logger.info(
                    "Received 401 on streamGenerateContent, refreshing token and retrying..."
                )
                await self.auth.refresh_access_token()
                headers = await self._get_headers()
                async with http.stream(
                    "POST", url, json=payload, headers=headers
                ) as retry_resp:
                    if retry_resp.status_code == 429:
                        retry_after = _extract_retry_after(
                            retry_resp, self.default_cooldown
                        )
                        self.set_cooldown(retry_after)
                        raise RateLimitError(
                            f"Antigravity rate limited (429): {await retry_resp.aread()}",
                            status_code=429,
                            retry_after=retry_after,
                        )
                    if retry_resp.status_code != 200:
                        err_body = await retry_resp.aread()
                        err_text = err_body.decode("utf-8", errors="replace")
                        logger.error(
                            f"streamGenerateContent retry error: {retry_resp.status_code} - {err_text}"
                        )
                        raise ValueError(
                            f"Antigravity API Error ({retry_resp.status_code}): {err_text}"
                        )

                    async for line in retry_resp.aiter_lines():
                        line = line.strip()
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            if data_str and data_str != "[DONE]":
                                try:
                                    parsed = json.loads(data_str)
                                    yield parsed
                                except json.JSONDecodeError as e:
                                    logger.warning(
                                        f"Failed to parse SSE JSON: {data_str} ({e})"
                                    )
                return

            if resp.status_code == 429:
                retry_after = _extract_retry_after(resp, self.default_cooldown)
                self.set_cooldown(retry_after)
                err_body = await resp.aread()
                raise RateLimitError(
                    f"Antigravity rate limited (429): {err_body.decode('utf-8', errors='replace')}",
                    status_code=429,
                    retry_after=retry_after,
                )

            if resp.status_code != 200:
                err_body = await resp.aread()
                err_text = err_body.decode("utf-8", errors="replace")
                logger.error(
                    f"streamGenerateContent error: {resp.status_code} - {err_text}"
                )
                raise ValueError(
                    f"Antigravity API Error ({resp.status_code}): {err_text}"
                )

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
                                f"Failed to parse SSE JSON: {data_str} ({e})"
                            )

    async def generate_content(
        self,
        model: str,
        contents: list[dict[str, Any]],
        system_instruction: dict[str, Any] | None = None,
        generation_config: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Non-streaming generate content call. Merges stream chunks into complete response."""
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
        """Call /v1internal:batchEmbedContents to generate text embeddings."""
        headers = await self._get_headers()
        project_id = await self.get_project_id()

        internal_model = (
            model.replace("models/", "") if model.startswith("models/") else model
        )

        requests_payload = []
        for text in texts:
            req_item: dict[str, Any] = {"content": {"parts": [{"text": text}]}}
            if dimensions is not None:
                req_item["outputDimensionality"] = dimensions
            requests_payload.append(req_item)

        payload = {
            "project": project_id,
            "model": internal_model,
            "requests": requests_payload,
        }

        url = f"{self.base_url}/v1internal:batchEmbedContents"
        http = self.get_http_client()
        resp = await http.post(url, json=payload, headers=headers)

        if resp.status_code == 401 and self.auth.refresh_token:
            logger.info(
                "Received 401 on batchEmbedContents, refreshing token and retrying..."
            )
            await self.auth.refresh_access_token()
            headers = await self._get_headers()
            resp = await http.post(url, json=payload, headers=headers)

        if resp.status_code == 429:
            retry_after = _extract_retry_after(resp, self.default_cooldown)
            self.set_cooldown(retry_after)
            raise RateLimitError(
                f"Antigravity embedding rate limited (429): {resp.text}",
                status_code=429,
                retry_after=retry_after,
            )

        if resp.status_code != 200:
            logger.error(f"batchEmbedContents failed: {resp.status_code} {resp.text}")
            raise ValueError(f"Antigravity API Error ({resp.status_code}): {resp.text}")

        return resp.json()
