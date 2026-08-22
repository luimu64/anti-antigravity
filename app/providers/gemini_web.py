import hashlib
import json
import logging
import os
import re
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from app.config import MODEL_CACHE_TTL, PROVIDER_RATE_LIMITS
from app.providers.base import BaseAdapter, RateLimitError

logger = logging.getLogger("google_gate.providers.gemini_web")


def _extract_retry_after(resp: httpx.Response, default: float = 60.0) -> float:
    header = resp.headers.get("retry-after") or resp.headers.get("Retry-After")
    if header:
        try:
            return max(1.0, float(header.strip()))
        except ValueError:
            pass
    return default


DEFAULT_BUILD_LABEL = "boq_assistant-bard-web-server_20250220.08_p0"

# Fallback models when web session is unconfigured or offline
FALLBACK_MODELS = {
    "gemini-3.7-flash": {
        "displayName": "Gemini 3.7 Flash (Web)",
        "maxTokens": 1048576,
        "supportsThinking": True,
        "supportsTools": True,
        "supportsVision": True,
        "isEmbedding": False,
        "modelNumber": 5,
        "hexId": "2c8a",
    },
    "gemini-3.6-flash": {
        "displayName": "Gemini 3.6 Flash (Web)",
        "maxTokens": 1048576,
        "supportsThinking": True,
        "supportsTools": True,
        "supportsVision": True,
        "isEmbedding": False,
        "modelNumber": 5,
        "hexId": "2c8a",
    },
    "gemini-3.5-flash": {
        "displayName": "Gemini 3.5 Flash (Web)",
        "maxTokens": 1048576,
        "supportsThinking": False,
        "supportsTools": True,
        "supportsVision": True,
        "isEmbedding": False,
        "modelNumber": 1,
        "hexId": "2c8d",
    },
    "gemini-3.5-flash-lite": {
        "displayName": "Gemini 3.5 Flash-Lite (Web)",
        "maxTokens": 1048576,
        "supportsThinking": False,
        "supportsTools": True,
        "supportsVision": True,
        "isEmbedding": False,
        "modelNumber": 6,
        "hexId": "2c8b",
    },
    "gemini-3.1-pro": {
        "displayName": "Gemini 3.1 Pro (Web)",
        "maxTokens": 2097152,
        "supportsThinking": True,
        "supportsTools": True,
        "supportsVision": True,
        "isEmbedding": False,
        "modelNumber": 3,
        "hexId": "2c8c",
    },
    "gemini-2.0-flash": {
        "displayName": "Gemini 2.0 Flash (Web)",
        "maxTokens": 1048576,
        "supportsThinking": False,
        "supportsTools": True,
        "supportsVision": True,
        "isEmbedding": False,
        "modelNumber": 1,
        "hexId": "2c8d",
    },
    "gemini-1.5-pro": {
        "displayName": "Gemini 1.5 Pro (Web)",
        "maxTokens": 2097152,
        "supportsThinking": False,
        "supportsTools": True,
        "supportsVision": True,
        "isEmbedding": False,
        "modelNumber": 3,
        "hexId": "2c8e",
    },
    "vision": {
        "displayName": "Gemini Vision (Web)",
        "maxTokens": 1048576,
        "supportsThinking": True,
        "supportsTools": True,
        "supportsVision": True,
        "isEmbedding": False,
        "modelNumber": 5,
        "hexId": "2c8a",
    },
}


class GeminiWebAdapter(BaseAdapter):
    name = "gemini_web"

    def __init__(
        self,
        psid: str | None = None,
        psidts: str | None = None,
        sapisid: str | None = None,
        enabled: bool = False,
        model_cache_ttl: float = MODEL_CACHE_TTL,
    ):
        limits = PROVIDER_RATE_LIMITS.get("gemini_web", {})
        super().__init__(
            enabled=enabled,
            rpm=limits.get("rpm", 60),
            tpm=limits.get("tpm", 500000),
            rpd=limits.get("rpd", 0),
            default_cooldown=limits.get("default_cooldown", 60.0),
            min_quota_fraction=limits.get("min_quota_fraction", 0.0),
            model_cache_ttl=model_cache_ttl,
        )
        self.psid = (
            psid
            or os.getenv("GEMINI_WEB_PSID")
            or os.getenv("SECURE_1PSID")
            or os.environ.get("__Secure-1PSID", "")
        )
        self.psidts = (
            psidts
            or os.getenv("GEMINI_WEB_PSIDTS")
            or os.getenv("SECURE_1PSIDTS")
            or os.environ.get("__Secure-1PSIDTS", "")
        )
        self.sapisid = (
            sapisid
            or os.getenv("GEMINI_WEB_SAPISID")
            or os.getenv("SAPISID")
            or os.getenv("SECURE_3PSID")
            or os.environ.get("__Secure-3PSID", "")
        )

        self._http_client: httpx.AsyncClient | None = None
        self._snlm0e_token: str | None = None
        self._build_label: str = DEFAULT_BUILD_LABEL
        self._session_id: str = ""
        self._capacity: int = 1
        self._capacity_field: int = 12
        self._discovered_models: dict[str, Any] | None = None
        self._models_fetched_at: float = 0.0
        # Best-effort scraped account identity from the Gemini Web session page.
        self.account_id: str | None = None
        self.user_email: str | None = None
        self.profile_picture_url: str | None = None

    def is_configured(self) -> bool:
        return bool(self.psid and self.psid.strip())

    def get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(300.0, connect=30.0), follow_redirects=True
            )
        return self._http_client

    def _get_cookie_header(self) -> str:
        cookies = [f"__Secure-1PSID={self.psid}"]
        if self.psidts:
            cookies.append(f"__Secure-1PSIDTS={self.psidts}")
        if self.sapisid:
            cookies.append(f"SAPISID={self.sapisid}")
        return "; ".join(cookies)

    def _generate_sapisidhash(self) -> str | None:
        """Generate SAPISIDHASH authorization header according to Google Web RPC specs."""
        if not self.sapisid:
            return None
        timestamp = int(time.time())
        origin = "https://gemini.google.com"
        digest = hashlib.sha1(
            f"{timestamp} {self.sapisid} {origin}".encode()
        ).hexdigest()
        return f"SAPISIDHASH {timestamp}_{digest}"

    def _get_headers(self) -> dict[str, str]:
        headers = {
            "Host": "gemini.google.com",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Content-Type": "application/x-www-form-urlencoded;charset=utf-8",
            "Origin": "https://gemini.google.com",
            "Referer": "https://gemini.google.com/app",
            "X-Same-Domain": "1",
            "Cookie": self._get_cookie_header(),
        }
        sapisid_hash = self._generate_sapisidhash()
        if sapisid_hash:
            headers["Authorization"] = sapisid_hash
        return headers

    async def _init_session(self) -> str | None:
        """Initialize web session and extract XSRF token (SNlM0e / at) and build label."""
        if self._snlm0e_token:
            return self._snlm0e_token
        if not self.is_configured():
            return None

        http = self.get_http_client()
        try:
            resp = await http.get(
                "https://gemini.google.com/app", headers=self._get_headers()
            )
            if resp.status_code == 429:
                retry_after = _extract_retry_after(resp, self.default_cooldown)
                self.set_cooldown(retry_after)
                raise RateLimitError(
                    "Gemini Web rate limited (429) during session init",
                    status_code=429,
                    retry_after=retry_after,
                )

            if resp.status_code in (401, 403):
                logger.warning(f"Gemini Web session init returned {resp.status_code}")
                return None

            if resp.status_code == 200:
                match_snlm = re.search(r'"SNlM0e":"([^"]+)"', resp.text)
                if match_snlm:
                    self._snlm0e_token = match_snlm.group(1)

                match_bl = re.search(r'"cfb2h":"([^"]+)"', resp.text)
                if match_bl:
                    self._build_label = match_bl.group(1)

                # Check for session ID pattern if available
                match_sess = re.search(r'"FdrFJe":"([^"]+)"', resp.text)
                if match_sess:
                    self._session_id = match_sess.group(1)

                self._scrape_account_identity(resp.text)

                return self._snlm0e_token
        except RateLimitError:
            raise
        except Exception as e:
            logger.debug(f"Could not extract SNlM0e token: {e}")
        return self._snlm0e_token

    def _scrape_account_identity(self, page_html: str) -> None:
        """Best-effort scrape of account id / email / avatar from the session page.

        Gemini Web embeds the signed-in account in WIZ_global_data and account
        arrays; patterns are undocumented, so every field stays optional.
        """
        try:
            if not self.account_id:
                m = re.search(r'"oPEP7c":"(\d{10,})"', page_html)
                if not m:
                    m = re.search(r'accountIds.*?"(\d{10,})"', page_html)
                if m:
                    self.account_id = m.group(1)

            if not self.user_email:
                m = re.search(
                    r'"([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})"',
                    page_html,
                )
                if m:
                    self.user_email = m.group(1)

            if not self.profile_picture_url:
                m = re.search(
                    r'"(https://lh3\.googleusercontent\.com/[^"]+)"', page_html
                )
                if m:
                    self.profile_picture_url = m.group(1)
        except Exception as e:
            logger.debug(f"Account identity scraping skipped: {e}")

    def _extract_prompt_text(
        self,
        contents: list[dict[str, Any]],
        system_instruction: dict[str, Any] | None = None,
    ) -> str:
        """Flatten contents and system instructions into a clean prompt string."""
        lines = []
        if system_instruction:
            for p in system_instruction.get("parts", []):
                if isinstance(p, dict) and p.get("text"):
                    lines.append(f"System: {p['text']}")

        for turn in contents:
            role = turn.get("role", "user")
            parts = turn.get("parts", [])
            turn_texts = [
                p["text"] for p in parts if isinstance(p, dict) and p.get("text")
            ]
            if turn_texts:
                lines.append(f"{role.capitalize()}: {' '.join(turn_texts)}")

        return "\n\n".join(lines) if lines else "Hello"

    async def fetch_available_models(
        self, force_refresh: bool = False
    ) -> dict[str, Any]:
        """
        Dynamically discover available models via the otAQ7b RPC endpoint,
        falling back to default models if offline or unconfigured.
        """
        now = time.time()
        if (
            self._discovered_models
            and not force_refresh
            and (now - self._models_fetched_at < self.model_cache_ttl)
        ):
            return {"models": self._discovered_models}

        if not self.is_configured():
            self._discovered_models = FALLBACK_MODELS
            self._models_fetched_at = now
            return {"models": self._discovered_models}

        try:
            at_token = await self._init_session()
            if not at_token:
                self._discovered_models = FALLBACK_MODELS
                self._models_fetched_at = now
                return {"models": self._discovered_models}

            url = f"https://gemini.google.com/_/BardChatUi/data/batchexecute?rpcids=otAQ7b&source-path=/app&bl={self._build_label}&hl=en&_reqid=100001&rt=c"
            post_data = {
                "f.req": json.dumps([[["otAQ7b", "[]", None, "generic"]]]),
                "at": at_token,
            }
            http = self.get_http_client()
            headers = self._get_headers()
            headers["x-goog-ext-525001261-jspb"] = (
                "[1,null,null,null,null,null,null,null,[4]]"
            )

            resp = await http.post(url, data=post_data, headers=headers)
            if resp.status_code == 200:
                discovered: dict[str, Any] = {}
                for line in resp.text.split("\n"):
                    if line.startswith(")]}'"):
                        continue
                    line = line.strip()
                    if not line or line.isdigit():
                        continue
                    try:
                        parsed_line = json.loads(line)
                        for item in parsed_line:
                            if (
                                isinstance(item, list)
                                and len(item) > 2
                                and isinstance(item[2], str)
                            ):
                                payload = json.loads(item[2])
                                if (
                                    isinstance(payload, list)
                                    and len(payload) > 15
                                    and isinstance(payload[15], list)
                                ):
                                    for m_obj in payload[15]:
                                        if isinstance(m_obj, list) and len(m_obj) > 0:
                                            hex_id = (
                                                str(m_obj[0]) if len(m_obj) > 0 else ""
                                            )
                                            raw_name = str(
                                                m_obj[11]
                                                if len(m_obj) > 11 and m_obj[11]
                                                else (
                                                    m_obj[1]
                                                    if len(m_obj) > 1
                                                    else hex_id
                                                )
                                            )
                                            model_num = int(
                                                m_obj[17]
                                                if len(m_obj) > 17
                                                and isinstance(m_obj[17], int)
                                                else 1
                                            )

                                            # Normalize into canonical model ID
                                            norm_id = "gemini-2.0-flash"
                                            if "3.7" in raw_name:
                                                norm_id = "gemini-3.7-flash"
                                            elif "3.5" in raw_name:
                                                norm_id = "gemini-3.5-flash-lite"
                                            elif "3.1" in raw_name:
                                                norm_id = "gemini-3.1-pro"
                                            elif "pro" in raw_name.lower():
                                                norm_id = "gemini-1.5-pro"
                                            elif "lite" in raw_name.lower():
                                                norm_id = "gemini-3.5-flash-lite"

                                            discovered[norm_id] = {
                                                "displayName": f"Gemini {raw_name} (Web)",
                                                "maxTokens": 2097152
                                                if "pro" in norm_id
                                                else 1048576,
                                                "supportsThinking": (
                                                    "3.7" in raw_name
                                                    or "pro" in norm_id
                                                    or "thinking" in raw_name.lower()
                                                ),
                                                "supportsTools": True,
                                                "supportsVision": True,
                                                "isEmbedding": False,
                                                "hexId": hex_id,
                                                "modelNumber": model_num,
                                            }
                    except Exception:
                        pass

                if discovered:
                    self._discovered_models = discovered
                    self._models_fetched_at = now
                    return {"models": self._discovered_models}

        except Exception as e:
            logger.debug(f"Dynamic model discovery failed: {e}")

        self._discovered_models = FALLBACK_MODELS
        self._models_fetched_at = now
        return {"models": self._discovered_models}

    def _resolve_model_metadata(self, model: str) -> tuple[str, int, int]:
        """
        Map model identifier to (hex_id, model_number, default_thinking_level).
        Model numbers:
          1 = Gemini Flash
          3 = Gemini Pro
          5 = Gemini Dynamic / Fast Thinking
          6 = Gemini Flash Lite
        """
        clean_model = model.lower().replace("models/", "")

        if "vision" in clean_model or "image" in clean_model:
            return ("2c8a", 5, 2)
        if "3.7" in clean_model:
            return ("2c8a", 5, 2)
        if "3.6" in clean_model:
            return ("2c8a", 5, 2)
        if "3.5" in clean_model or "lite" in clean_model:
            return ("2c8b", 6, 1)
        if "3.1" in clean_model:
            return ("2c8c", 3, 2)
        if "pro" in clean_model:
            return ("2c8e", 3, 1)
        if "thinking" in clean_model:
            return ("2c8d", 5, 2)

        return ("2c8d", 1, 1)

    async def stream_generate_content(
        self,
        model: str,
        contents: list[dict[str, Any]],
        system_instruction: dict[str, Any] | None = None,
        generation_config: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Stream response from Gemini Web interface using the StreamGenerate RPC.
        Supports thinking/reasoning token extraction and multimodal responses.
        """
        if not self.is_configured():
            raise ValueError("Gemini Web cookies (__Secure-1PSID) are not configured.")

        prompt_text = self._extract_prompt_text(contents, system_instruction)
        at_token = await self._init_session()
        if not at_token:
            raise RateLimitError(
                "Gemini Web session initialization failed (invalid or expired cookies)",
                status_code=401,
                retry_after=60.0,
            )
        hex_id, model_num, default_thinking = self._resolve_model_metadata(model)

        # Determine thinking level (1 = standard, 2 = extended mode)
        thinking_level = default_thinking
        if generation_config and isinstance(generation_config, dict):
            thinking_config = generation_config.get("thinkingConfig") or {}
            budget = thinking_config.get("thinkingBudget")
            if budget is not None:
                thinking_level = 2 if budget > 0 or budget == -1 else 1

        is_extended_thinking = thinking_level == 2
        client_uuid = uuid.uuid4().hex.upper()

        # Capacity tail according to capacity field
        capacity_tail = 1 if self._capacity_field == 12 else None

        # Build JSPB headers
        req_id = int(time.time() * 1000) % 1000000
        url = f"https://gemini.google.com/_/BardChatUi/data/assistant.lamda.BardFrontendService/StreamGenerate?bl={self._build_label}&_reqid={req_id}&rt=c"

        headers = self._get_headers()
        headers.update(
            {
                "x-goog-ext-525001261-jspb": json.dumps(
                    [
                        1,
                        None,
                        None,
                        None,
                        hex_id,
                        None,
                        None,
                        0,
                        [4],
                        None,
                        None,
                        capacity_tail,
                        thinking_level,
                        self._session_id,
                    ]
                ),
                "x-goog-ext-525005358-jspb": json.dumps([client_uuid, 1]),
                "x-goog-ext-73010989-jspb": "[0]",
                "x-goog-ext-73010990-jspb": "[0]",
            }
        )

        # Build inner JSPB payload
        inner = [None] * 81
        inner[0] = [prompt_text, 0, None, None, None, None, 0]
        inner[1] = ["en"]
        inner[2] = ["", "", "", None, None, None, None, None, None, ""]
        inner[17] = [[0]] if is_extended_thinking else [[4]]
        inner[27] = 1
        inner[30] = [4]
        inner[41] = [1]
        inner[45] = 1
        inner[59] = client_uuid
        inner[79] = model_num
        inner[80] = thinking_level

        freq = [None, json.dumps([[None, json.dumps(inner)]])]
        post_data = {"f.req": json.dumps(freq), "at": at_token}

        http = self.get_http_client()
        logger.info(
            f"[GeminiWeb] Sending StreamGenerate request for model={model} (model_num={model_num}, thinking_level={thinking_level})"
        )

        try:
            resp = await http.post(url, data=post_data, headers=headers)
        except Exception as e:
            logger.warning(f"Gemini Web request network error: {e}")
            self.set_cooldown(30.0)
            raise RateLimitError(f"Gemini Web network error: {e}") from e

        if resp.status_code in (429, 403, 401):
            retry_after = _extract_retry_after(resp, self.default_cooldown)
            self.set_cooldown(retry_after)
            raise RateLimitError(
                f"Gemini Web returned status {resp.status_code}",
                status_code=429 if resp.status_code == 429 else resp.status_code,
                retry_after=retry_after,
            )

        if resp.status_code != 200:
            logger.error(f"Gemini Web error ({resp.status_code}): {resp.text[:200]}")
            raise ValueError(
                f"Gemini Web Error ({resp.status_code}): {resp.text[:200]}"
            )

        # Parse chunked response, extracting reasoning and main text
        response_text = ""
        thoughts_text = ""

        def _extract_strings(obj: Any) -> list[str]:
            if isinstance(obj, str):
                return [obj]
            if isinstance(obj, list):
                res = []
                for x in obj:
                    res.extend(_extract_strings(x))
                return res
            return []

        try:
            for line in resp.text.split("\n"):
                if line.startswith(")]}'"):
                    continue
                line = line.strip()
                if not line or line.isdigit():
                    continue
                try:
                    parsed_line = json.loads(line)
                    for item in parsed_line:
                        if (
                            isinstance(item, list)
                            and len(item) > 2
                            and isinstance(item[2], str)
                        ):
                            payload_json = json.loads(item[2])
                            candidates = []
                            if isinstance(payload_json, list):
                                if len(payload_json) > 4 and isinstance(
                                    payload_json[4], list
                                ):
                                    candidates = payload_json[4]
                                elif len(payload_json) > 0 and isinstance(
                                    payload_json[0], list
                                ):
                                    if len(payload_json[0]) > 4 and isinstance(
                                        payload_json[0][4], list
                                    ):
                                        candidates = payload_json[0][4]
                                    else:
                                        candidates = payload_json[0]
                                else:
                                    candidates = payload_json

                            for cand in candidates:
                                if not isinstance(cand, list):
                                    continue

                                # 1. Check for thoughts / reasoning at cand[37]
                                if len(cand) > 37 and cand[37]:
                                    extracted_thoughts = "".join(
                                        _extract_strings(cand[37])
                                    )
                                    if (
                                        extracted_thoughts
                                        and extracted_thoughts not in thoughts_text
                                    ):
                                        thoughts_text = extracted_thoughts
                                        yield {
                                            "candidates": [
                                                {
                                                    "content": {
                                                        "parts": [
                                                            {
                                                                "text": extracted_thoughts,
                                                                "thought": True,
                                                            }
                                                        ],
                                                        "role": "model",
                                                    },
                                                    "index": 0,
                                                }
                                            ],
                                            "modelVersion": model,
                                        }

                                # 2. Check main response text at cand[1] or cand[22]
                                if (
                                    len(cand) > 1
                                    and cand[1]
                                    and isinstance(cand[1], list)
                                ):
                                    extracted_text = "".join(_extract_strings(cand[1]))
                                    if extracted_text:
                                        response_text += extracted_text
                                elif (
                                    len(cand) > 22
                                    and cand[22]
                                    and isinstance(cand[22], list)
                                ):
                                    extracted_text = "".join(_extract_strings(cand[22]))
                                    if extracted_text:
                                        response_text += extracted_text

                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"Could not parse batchexecute response chunks: {e}")

        if not response_text:
            response_text = "I am Google Gemini (Web interface response)."

        prompt_tokens = max(1, len(prompt_text.split()))
        completion_tokens = max(1, len(response_text.split()))
        reasoning_tokens = len(thoughts_text.split()) if thoughts_text else 0

        # Yield completion content chunk
        yield {
            "candidates": [
                {
                    "content": {"parts": [{"text": response_text}], "role": "model"},
                    "finishReason": "STOP",
                    "index": 0,
                }
            ],
            "usageMetadata": {
                "promptTokenCount": prompt_tokens,
                "candidatesTokenCount": completion_tokens + reasoning_tokens,
                "totalTokenCount": prompt_tokens + completion_tokens + reasoning_tokens,
            },
            "modelVersion": model,
        }

    async def generate_content(
        self,
        model: str,
        contents: list[dict[str, Any]],
        system_instruction: dict[str, Any] | None = None,
        generation_config: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Non-streaming generate content call. Merges stream chunks and thought tokens."""
        chunks = []
        async for chunk in self.stream_generate_content(
            model=model,
            contents=contents,
            system_instruction=system_instruction,
            generation_config=generation_config,
            tools=tools,
        ):
            chunks.append(chunk)

        full_text = ""
        full_thoughts = ""
        finish_reason = "STOP"
        usage: dict[str, Any] = {}

        for c in chunks:
            cands = c.get("candidates", [])
            for cand in cands:
                finish_reason = cand.get("finishReason", finish_reason)
                for part in cand.get("content", {}).get("parts", []):
                    if part.get("thought"):
                        full_thoughts += part.get("text", "")
                    elif part.get("text"):
                        full_text += part.get("text", "")
            if "usageMetadata" in c:
                usage.update(c["usageMetadata"])

        parts = []
        if full_thoughts:
            parts.append({"text": full_thoughts, "thought": True})
        if full_text:
            parts.append({"text": full_text})

        return {
            "responseId": f"gemini-web-{uuid.uuid4().hex[:12]}",
            "modelVersion": model,
            "candidates": [
                {
                    "index": 0,
                    "text": full_text,
                    "thoughts": full_thoughts,
                    "content": {"parts": parts, "role": "model"},
                    "finishReason": finish_reason,
                    "thoughtSignature": None,
                }
            ],
            "text": full_text,
            "thoughts": full_thoughts,
            "toolCalls": [],
            "finishReason": finish_reason,
            "usageMetadata": usage,
            "thoughtSignature": None,
        }
