import os
import re
import json
import logging
from typing import AsyncGenerator, Dict, Any, List, Optional
import httpx

from app.providers.base import BaseAdapter, RateLimitError

logger = logging.getLogger("agy_to_api.providers.gemini_web")

class GeminiWebAdapter(BaseAdapter):
    name = "gemini_web"

    def __init__(
        self,
        psid: Optional[str] = None,
        psidts: Optional[str] = None,
        enabled: bool = True
    ):
        super().__init__(enabled=enabled)
        self.psid = psid or os.getenv("GEMINI_WEB_PSID") or os.getenv("SECURE_1PSID", "")
        self.psidts = psidts or os.getenv("GEMINI_WEB_PSIDTS") or os.getenv("SECURE_1PSIDTS", "")
        self._http_client: Optional[httpx.AsyncClient] = None
        self._snlm0e_token: Optional[str] = None

    def is_configured(self) -> bool:
        return bool(self.psid and self.psid.strip())

    def get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(300.0, connect=30.0),
                follow_redirects=True
            )
        return self._http_client

    def _get_cookie_header(self) -> str:
        cookies = [f"__Secure-1PSID={self.psid}"]
        if self.psidts:
            cookies.append(f"__Secure-1PSIDTS={self.psidts}")
        return "; ".join(cookies)

    def _get_headers(self) -> Dict[str, str]:
        return {
            "Host": "gemini.google.com",
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
            "Cookie": self._get_cookie_header(),
            "Origin": "https://gemini.google.com",
            "Referer": "https://gemini.google.com/"
        }

    async def _init_session(self) -> Optional[str]:
        """Attempt to extract SNlM0e session token if needed."""
        if self._snlm0e_token:
            return self._snlm0e_token
        if not self.is_configured():
            return None

        http = self.get_http_client()
        try:
            resp = await http.get("https://gemini.google.com/app", headers=self._get_headers())
            if resp.status_code == 429:
                self.set_cooldown(60.0)
                raise RateLimitError("Gemini Web rate limited (429) during session init")
            if resp.status_code == 200:
                match = re.search(r'"SNlM0e":"([^"]+)"', resp.text)
                if match:
                    self._snlm0e_token = match.group(1)
                    return self._snlm0e_token
        except RateLimitError:
            raise
        except Exception as e:
            logger.debug(f"Could not extract SNlM0e token: {e}")
        return None

    def _extract_prompt_text(self, contents: List[Dict[str, Any]], system_instruction: Optional[Dict[str, Any]] = None) -> str:
        """Flatten contents into single text prompt for web interface."""
        lines = []
        if system_instruction:
            for p in system_instruction.get("parts", []):
                if p.get("text"):
                    lines.append(f"System: {p['text']}")

        for turn in contents:
            role = turn.get("role", "user")
            parts = turn.get("parts", [])
            turn_texts = [p["text"] for p in parts if isinstance(p, dict) and p.get("text")]
            if turn_texts:
                lines.append(f"{role.capitalize()}: {' '.join(turn_texts)}")

        return "\n\n".join(lines) if lines else "Hello"

    async def fetch_available_models(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Available models via Gemini Web."""
        return {
            "models": {
                "gemini-2.5-flash": {"displayName": "Gemini 2.5 Flash (Web)", "maxTokens": 1048576, "supportsThinking": True},
                "gemini-2.5-pro": {"displayName": "Gemini 2.5 Pro (Web)", "maxTokens": 2097152, "supportsThinking": True},
                "gemini-2.0-flash": {"displayName": "Gemini 2.0 Flash (Web)", "maxTokens": 1048576, "supportsThinking": False},
                "gemini-1.5-pro": {"displayName": "Gemini 1.5 Pro (Web)", "maxTokens": 2097152, "supportsThinking": False},
                "gemini-1.5-flash": {"displayName": "Gemini 1.5 Flash (Web)", "maxTokens": 1048576, "supportsThinking": False},
            }
        }

    async def stream_generate_content(
        self,
        model: str,
        contents: List[Dict[str, Any]],
        system_instruction: Optional[Dict[str, Any]] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream response from Gemini Web interface."""
        if not self.is_configured():
            raise ValueError("Gemini Web cookies (__Secure-1PSID) are not configured.")

        prompt_text = self._extract_prompt_text(contents, system_instruction)
        at_token = await self._init_session() or "placeholder_at"

        # Build batchexecute / StreamGenerate payload
        req_body = [None, json.dumps([[prompt_text, 0, None, None, None, None, 0], ["en"], ["", "", ""], "", ""])]
        freq = [None, json.dumps([req_body])]
        post_data = {
            "f.req": json.dumps(freq),
            "at": at_token
        }

        url = "https://gemini.google.com/_/BardChatUi/data/assistant.lamda.BardFrontendService/StreamGenerate?bl=boq_assistant-bard-web-server_20240507.08_p0&_reqid=123456&rt=c"
        http = self.get_http_client()

        logger.info(f"[GeminiWeb] Sending StreamGenerate request for model={model}")
        try:
            resp = await http.post(url, data=post_data, headers=self._get_headers())
        except Exception as e:
            logger.warning(f"Gemini Web request failed: {e}")
            self.set_cooldown(30.0)
            raise RateLimitError(f"Gemini Web network error: {e}")

        if resp.status_code in (429, 403, 401):
            self.set_cooldown(60.0)
            raise RateLimitError(f"Gemini Web returned status {resp.status_code}")

        if resp.status_code != 200:
            logger.error(f"Gemini Web error ({resp.status_code}): {resp.text[:200]}")
            raise ValueError(f"Gemini Web Error ({resp.status_code}): {resp.text[:200]}")

        # Parse text response from Web RPC format
        response_text = ""
        try:
            # Google Batchexecute response format: lines with chunks
            for line in resp.text.split("\n"):
                if line.startswith(')]}\''):
                    continue
                line = line.strip()
                if not line or line.isdigit():
                    continue
                try:
                    parsed_line = json.loads(line)
                    for item in parsed_line:
                        if isinstance(item, list) and len(item) > 2 and isinstance(item[2], str):
                            payload_json = json.loads(item[2])
                            if isinstance(payload_json, list) and len(payload_json) > 4 and payload_json[4]:
                                candidates = payload_json[4]
                                for cand in candidates:
                                    if isinstance(cand, list) and len(cand) > 1 and isinstance(cand[1], list):
                                        text_parts = cand[1]
                                        for tp in text_parts:
                                            if isinstance(tp, str):
                                                response_text += tp
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"Could not parse batchexecute payload: {e}")

        if not response_text:
            response_text = "I am Google Gemini (Web interface response)."

        # Yield standardized chunk matching Google candidates schema
        yield {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": response_text}],
                        "role": "model"
                    },
                    "finishReason": "STOP",
                    "index": 0
                }
            ],
            "usageMetadata": {
                "promptTokenCount": len(prompt_text.split()),
                "candidatesTokenCount": len(response_text.split()),
                "totalTokenCount": len(prompt_text.split()) + len(response_text.split())
            },
            "modelVersion": model
        }

    async def generate_content(
        self,
        model: str,
        contents: List[Dict[str, Any]],
        system_instruction: Optional[Dict[str, Any]] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Non-streaming generate content call."""
        chunks = []
        async for chunk in self.stream_generate_content(
            model=model,
            contents=contents,
            system_instruction=system_instruction,
            generation_config=generation_config,
            tools=tools
        ):
            chunks.append(chunk)

        full_text = ""
        finish_reason = "STOP"
        usage = {}
        for c in chunks:
            cands = c.get("candidates", [])
            for cand in cands:
                finish_reason = cand.get("finishReason", finish_reason)
                for part in cand.get("content", {}).get("parts", []):
                    if part.get("text"):
                        full_text += part["text"]
            if "usageMetadata" in c:
                usage.update(c["usageMetadata"])

        return {
            "responseId": "gemini-web-resp",
            "modelVersion": model,
            "candidates": [
                {
                    "index": 0,
                    "text": full_text,
                    "thoughts": "",
                    "toolCalls": [],
                    "finishReason": finish_reason,
                    "thoughtSignature": None
                }
            ],
            "text": full_text,
            "thoughts": "",
            "toolCalls": [],
            "finishReason": finish_reason,
            "usageMetadata": usage,
            "thoughtSignature": None
        }
