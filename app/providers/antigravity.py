import datetime
import json
import logging
import os
import time
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from app.auth import OAuthManager, auth_manager
from app.config import (
    CLOUD_CODE_BASE_URL,
    DEPRECATED_MODELS,
    MODEL_CACHE_TTL,
    PROVIDER_RATE_LIMITS,
    USER_AGENT,
)
from app.providers.base import BaseAdapter, RateLimitError

logger = logging.getLogger("google_gate.providers.antigravity")


def _extract_retry_after(resp: httpx.Response, default: float = 60.0) -> float:
    header = resp.headers.get("retry-after") or resp.headers.get("Retry-After")
    if header:
        try:
            return max(1.0, float(header.strip()))
        except ValueError:
            pass
    return default


def _extract_retry_after_header(
    resp: httpx.Response,
) -> float | None:
    """Return the upstream Retry-After as seconds, or None when absent."""
    header = resp.headers.get("retry-after") or resp.headers.get("Retry-After")
    if header:
        try:
            return max(1.0, float(header.strip()))
        except ValueError:
            pass
    return None


# Cooldown applied to a 429 that carries NO Retry-After header. Upstream
# Antigravity burst-rejections are typically transient (retried immediately
# they succeed) — empirically 12 consecutive probes at 0.4s spacing all
# passed right after a burst of upstream 429s — so a full default_cooldown
# (60s) lockout punishes the account for one transient rejection: every
# parallel request during the window fails instantly with "all backends
# exhausted". A short cooldown keeps the next attempt nearby while still
# spacing out the retry.
TRANSIENT_429_COOLDOWN_S = float(os.getenv("ANTIGRAVITY_TRANSIENT_429_COOLDOWN", "3.0"))


def _parse_duration(value: Any) -> float | None:
    """Parse '312.252235ms' / '0.057395018s' style durations to seconds."""
    try:
        s = str(value).strip()
        if s.endswith("ms"):
            return max(0.0, float(s[:-2])) / 1000.0
        if s.endswith("s"):
            return max(0.0, float(s[:-1]))
    except (ValueError, TypeError):
        return None
    return None


def _retry_delay_from_body(resp: httpx.Response) -> float | None:
    """Parse the gRPC error body's retryDelay / quotaResetDelay.

    Antigravity 429s carry no Retry-After header but DO embed the precise
    remaining reset delay in the JSON body:
      details[RetryInfo].retryDelay           e.g. "0.057395018s"
      details[ErrorInfo].metadata.quotaResetDelay   e.g. "312.252235ms"
    Returns seconds, or None when absent/unparseable.
    """
    try:
        data = resp.json().get("error", {})
    except Exception:
        return None
    details = data.get("details")
    if not isinstance(details, list):
        return None

    best: float | None = None
    for d in details:
        if not isinstance(d, dict):
            continue
        # RetryInfo: "@type": "type.googleapis.com/google.rpc.RetryInfo"
        if str(d.get("@type", "")).endswith("RetryInfo"):
            v = _parse_duration(d.get("retryDelay"))
            if v is not None:
                best = v if best is None else max(best, v)
        # ErrorInfo metadata fallback: quotaResetDelay
        elif str(d.get("@type", "")).endswith("ErrorInfo"):
            meta = d.get("metadata")
            if isinstance(meta, dict):
                v = _parse_duration(meta.get("quotaResetDelay"))
                if v is not None:
                    best = v if best is None else max(best, v)
    return best


def _retry_delay_from_body_str(text: str) -> float | None:
    """Parse retryDelay / quotaResetDelay from a raw JSON/gRPC body string."""
    import json as _json

    try:
        data = _json.loads(text)
    except Exception:
        return None
    details = (data.get("error", {}) or {}).get("details") or []
    for d in details:
        if isinstance(d, dict) and d.get("@type", "").endswith("ErrorInfo"):
            meta = d.get("metadata", {}) or {}
            for key in ("quotaResetDelay", "retryDelay"):
                v = meta.get(key)
                if not v:
                    continue
                parsed = _parse_duration(v)
                if parsed is not None:
                    return max(0.05, parsed)
    return None


def _cooldown_for_429(resp: httpx.Response, default: float) -> float:
    """Cooldown for an upstream 429, in precision order:
    1. Retry-After header (authoritative when present),
    2. retryDelay/quotaResetDelay parsed from the error body,
    3. short transient cooldown instead of the full default lockout."""
    header_secs = _extract_retry_after_header(resp)
    if header_secs is not None:
        return header_secs
    body_secs = _retry_delay_from_body(resp)
    if body_secs is not None:
        return max(0.5, body_secs)
    return TRANSIENT_429_COOLDOWN_S


# Escalating backoff for CONSECUTIVE upstream 429s. A single 429 whose body
# says "resets in 300ms" is real (bucket refresh imminent) — but a retry
# loop hitting 429s back-to-back means the account is being hammered and
# every rejected attempt still burns quota tally on Google's side. Without
# escalation, a misbehaving client (observed: Hindsight memory worker via
# Bifrost) retried at ~4 req/s and consumed an entire 5-hour quota bucket
# in minutes with requests never accepted. Cooldown doubles per consecutive
# 429 up to the default; a successful request resets the strike counter.
MAX_CONSECUTIVE_429_ESCALATION = 5  # cap: stops at default_cooldown ≈ 60s


def _escalate_consecutive_429(current_strikes: int) -> float:
    """Cooldown for the Nth consecutive 429 (0-indexed): 2^n * base, capped."""
    factor = min(current_strikes, MAX_CONSECUTIVE_429_ESCALATION)
    return min(
        TRANSIENT_429_COOLDOWN_S * (2**factor),
        PROVIDER_RATE_LIMITS["antigravity"].get("default_cooldown", 60.0),
    )


class AntigravityAdapter(BaseAdapter):
    name = "antigravity"

    def __init__(
        self,
        base_url: str = CLOUD_CODE_BASE_URL,
        auth: OAuthManager = auth_manager,
        enabled: bool = False,
        model_cache_ttl: float = MODEL_CACHE_TTL,
    ):
        limits = PROVIDER_RATE_LIMITS.get("antigravity", {})
        super().__init__(
            enabled=enabled,
            rpm=limits.get("rpm", 100),
            tpm=limits.get("tpm", 1000000),
            rpd=limits.get("rpd", 0),
            default_cooldown=limits.get("default_cooldown", 60.0),
            min_quota_fraction=limits.get("min_quota_fraction", 0.01),
            model_cache_ttl=model_cache_ttl,
        )
        self.base_url = base_url.rstrip("/")
        self.auth = auth
        self._cached_models: dict[str, Any] | None = None
        self._models_fetched_at: float = 0.0
        self._http_client: httpx.AsyncClient | None = None
        # Quota-deny map: model family -> reset timestamp (epoch seconds).
        # Set ONLY on evidence that a quota bucket is genuinely exhausted;
        # while set, every request to that family is denied outright at the
        # router level (no cooldown timer games — the deny lasts until the
        # bucket's own reset time). Never populated by transient 429s.
        self._quota_exhausted_until: dict[str, float] = {}

    def _family_of(self, model: str) -> str:
        """Bucket key for a model, e.g. 'gemini-3.8-flash-medium' ->
        'gemini-3.8-flash' (tier suffixes share one bucket)."""
        base = model
        for suffix in ("-low", "-medium", "-high", "-tiered"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        return base

    def mark_quota_exhausted(self, model: str, reset_ts: float) -> None:
        """Deny this model's family until the epoch second reset_ts."""
        fam = self._family_of(model)
        self._quota_exhausted_until[fam] = reset_ts

    def quota_family_exhausted(self, model: str) -> bool:
        now = time.time()

        def _live(fam: str) -> bool:
            exp = self._quota_exhausted_until.get(fam)
            return bool(exp and now < exp)

        # Direct family key (tier suffixes share one bucket).
        if _live(self._family_of(model)):
            return True
        # Lane keys set from quota buckets: 'gemini-5h' covers every gemini-*
        # model, 'claude-lane' covers every claude*/o* alias. A drained 5h
        # bucket must not deny 3p models, and vice versa.
        m = model.lower()
        if m.startswith("gemini") and _live("gemini-5h"):
            return True
        if m.startswith(("claude", "o1", "o3", "gpt-oss")) and _live("claude-lane"):
            return True
        # gpt-oss resolves to gpt-oss-120b-medium via MODEL_ALIASES; the
        # bucket marks the canonical 'gpt-oss-120b' family directly.
        return m.startswith("gpt-oss") and _live("gpt-oss-120b")

    def _apply_429(self, resp: httpx.Response, model: str | None) -> float:
        """Handle an upstream 429. Returns seconds until a retry makes sense.

        No cooldowns. If the body says the quota is exhausted, the model's
        family is denied until the reset time parsed from the body (falling
        back to default_cooldown). Otherwise it's a transient limit: return
        the body's own retry delay and change nothing.
        """
        reset_ts = None
        try:
            data = resp.json()
            details = (data.get("error", {}) or {}).get("details") or []
            for d in details:
                if not isinstance(d, dict):
                    continue
                meta = d.get("metadata", {}) or {}
                for key in ("quotaResetDelay", "retryDelay"):
                    delay = _parse_duration(meta.get(key))
                    if delay is not None:
                        reset_ts = time.time() + delay
        except Exception:
            pass

        if reset_ts is None:
            body_delay = _retry_delay_from_body(resp.text)
            if body_delay is not None:
                reset_ts = time.time() + body_delay

        # Determine whether this 429 is model-scoped exhaustion (quota) or
        # account-wide: bodies carrying a 'model' metadata entry are per-model.
        model_scoped = False
        try:
            data = resp.json()
            details = data.get("error", {}).get("details") or []
            for d in details:
                if isinstance(d, dict) and d.get("@type", "").endswith("ErrorInfo"):
                    meta = d.get("metadata", {}) or {}
                    if meta.get("model"):
                        model_scoped = True
        except Exception:
            pass

        if reset_ts is not None and model and model_scoped:
            self.mark_quota_exhausted(model, reset_ts)
            logger.warning(
                "Antigravity quota exhausted for '%s' family; denying until %s",
                model,
                datetime.datetime.fromtimestamp(
                    reset_ts, datetime.timezone.utc
                ).isoformat(),
            )
            # Scoped deny is registered via _quota_exhausted_until; the caller
            # must NOT also slap an adapter-wide cooldown — that would deny
            # unrelated models (e.g. 3p claude when a gemini bucket drains).
            return max(0.5, reset_ts - time.time())

        if reset_ts is None:
            reset_ts = time.time() + TRANSIENT_429_COOLDOWN_S
        return max(0.5, reset_ts - time.time())

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
            retry_after = self._apply_429(resp, None)
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
        """Call /v1internal:fetchAvailableModels to get all supported models with TTL caching."""
        now = time.time()
        if (
            self._cached_models
            and not force_refresh
            and (now - self._models_fetched_at < self.model_cache_ttl)
        ):
            return self._cached_models

        fallback_models = {
            "gemini-3.7-flash-high": {
                "displayName": "Gemini 3.7 Flash High",
                "maxTokens": 1048576,
                "supportsThinking": True,
            },
            "gemini-3.6-flash-high": {
                "displayName": "Gemini 3.6 Flash High",
                "maxTokens": 1048576,
                "supportsThinking": True,
            },
            "gemini-3.1-pro-high": {
                "displayName": "Gemini 3.1 Pro High",
                "maxTokens": 2097152,
                "supportsThinking": True,
            },
            "gemini-3-flash-agent": {
                "displayName": "Gemini 3.5 Flash",
                "maxTokens": 1048576,
                "supportsThinking": True,
            },
            "claude-sonnet-4-6": {
                "displayName": "Claude 3.7 Sonnet",
                "maxTokens": 200000,
                "supportsThinking": True,
            },
            "claude-opus-4-6-thinking": {
                "displayName": "Claude 3 Opus",
                "maxTokens": 200000,
                "supportsThinking": True,
            },
            "gpt-oss-120b-medium": {
                "displayName": "GPT-OSS 120B",
                "maxTokens": 32768,
                "supportsThinking": True,
            },
            "text-embedding-004": {
                "displayName": "Text Embedding 004",
                "maxTokens": 2048,
                "supportsThinking": False,
            },
        }

        if not self.is_configured():
            self._cached_models = {"models": fallback_models}
            self._models_fetched_at = now
            return self._cached_models

        headers = await self._get_headers()
        url = f"{self.base_url}/v1internal:fetchAvailableModels"
        http = self.get_http_client()
        try:
            resp = await http.post(url, json={}, headers=headers)
            if resp.status_code == 401 and self.auth.refresh_token:
                logger.info(
                    "Received 401 on fetchAvailableModels, refreshing token and retrying..."
                )
                await self.auth.refresh_access_token()
                headers = await self._get_headers()
                resp = await http.post(url, json={}, headers=headers)

            if resp.status_code == 429:
                self._apply_429(resp, None)
                if self._cached_models:
                    return self._cached_models
                return {"models": fallback_models}

            if resp.status_code != 200:
                logger.error(
                    f"fetchAvailableModels failed: {resp.status_code} {resp.text}"
                )
                if self._cached_models:
                    return self._cached_models
                return {"models": fallback_models}

            data = resp.json()
            if isinstance(data, dict) and "models" in data:
                filtered = {
                    k: v
                    for k, v in data["models"].items()
                    if k not in DEPRECATED_MODELS
                    and k.replace("models/", "") not in DEPRECATED_MODELS
                }
                self._cached_models = {"models": filtered}
            else:
                self._cached_models = data
            self._models_fetched_at = now
            return self._cached_models
        except Exception as e:
            logger.warning(f"Error fetching models from Antigravity: {e}")
            if self._cached_models:
                return self._cached_models
            return {"models": fallback_models}

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
            retry_after = self._apply_429(resp, None)
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
                    # Known-exhausted bucket: deny the models it serves until
                    # reset. Bucket IDs look like 'gemini-5h' / '3p-5h' etc —
                    # model-family mapping happens via models built from these.
                    reset_ts = time.time() + self.default_cooldown
                    reset_time_str = bucket.get("resetTime")
                    if reset_time_str:
                        try:
                            dt = datetime.datetime.fromisoformat(
                                reset_time_str.replace("Z", "+00:00")
                            )
                            reset_ts = dt.timestamp()
                        except Exception:
                            pass
                    bucket_id = str(
                        bucket.get("modelId")
                        or bucket.get("model_id")
                        or bucket.get("bucketId")
                        or bucket.get("displayName", "")
                    )
                    for fam in self._bucket_families(bucket):
                        self.mark_quota_exhausted(fam, reset_ts)
                    logger.warning(
                        f"Antigravity quota bucket '{bucket_id}' exhausted "
                        f"({rem * 100:.1f}% remaining). Denying its models until reset."
                    )

        # Quota summary tells us which aggregates are drained — but buckets
        # are account aggregates, not per-model. Mark every cached model
        # family: when the shared bucket is empty ALL models draw it.
        # (Per-model 429s add finer-grained denies via _apply_429.)
        # NOTE: intentionally NOT marking here — bucket->model mapping is
        # unknown (a 'gemini-5h' bucket serves everything). Aggregate
        # exhaustion without a per-model failure is informational only;
        # per-model denies from real 429s are the enforcement point.
        return data

    def _bucket_families(self, bucket: dict) -> list[str]:
        """Map a quota bucket (raw upstream dict) to the model families it
        serves. Identification priority: modelId/model_id/bucketId
        ('gemini-5h', '3p-5h', 'gemini-weekly'...) — displayName
        ('Five Hour Limit Remaining') does NOT carry the discriminator.

        Verified bucket ids from retrieveUserQuotaSummary:
        - 'gemini-weekly' / 'gemini-5h'      -> gemini-* models only
        - '3p-weekly' / '3p-5h'              -> third-party models only
          (claude-*, gpt-oss-*)
        Aggregate markers are stored with a 'marker:' prefix so they can
        never accidentally match a real model family key.
        """
        b = str(
            bucket.get("modelId")
            or bucket.get("model_id")
            or bucket.get("bucketId")
            or bucket.get("displayName")
            or ""
        ).lower()
        if "3p-" in b:
            return [
                "marker:3p-weekly" if "weekly" in b else "marker:3p-5h",
                "claude-lane",
                "gpt-oss-120b",
            ]
        if "5h" in b:
            return ["gemini-5h"]
        if "weekly" in b:
            return ["marker:gemini-weekly"]  # aggregate marker, not a family
        return []

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
                        retry_after = self._apply_429(retry_resp, model)
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
                                except json.JSONDecodeError as e:
                                    logger.warning(
                                        f"Failed to parse SSE JSON: {data_str} ({e})"
                                    )
                                    continue
                                    yield parsed
                return

            if resp.status_code == 429:
                retry_after = self._apply_429(resp, model)
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
                        except json.JSONDecodeError as e:
                            logger.warning(
                                f"Failed to parse SSE JSON: {data_str} ({e})"
                            )
                            continue
                        yield parsed

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
        # Model-unavailable guard: Antigravity sunsets return HTTP 200 with
        # only a canned notice (no usageMetadata at all). Surface that as a
        # gateway error instead of a 0-token "success" that misleads clients
        # (and made a dead model look like the caller's fault).
        if not usage_metadata:
            notice = (primary["text"] or "").strip()
            if notice:
                logger.error(
                    "Upstream returned no usageMetadata with text (model %s): %s",
                    model,
                    notice[:120],
                )
                raise RuntimeError(
                    f"Antigravity model '{model}' returned no usage metadata — "
                    f"likely decommissioned upstream. Response text: {notice[:200]}"
                )

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
            retry_after = self._apply_429(resp, None)
            raise RateLimitError(
                f"Antigravity embedding rate limited (429): {resp.text}",
                status_code=429,
                retry_after=retry_after,
            )

        if resp.status_code != 200:
            logger.error(f"batchEmbedContents failed: {resp.status_code} {resp.text}")
            raise ValueError(f"Antigravity API Error ({resp.status_code}): {resp.text}")

        return resp.json()
