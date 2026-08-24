import base64
import contextlib
import hashlib
import json
import logging
import os
import re
import secrets
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from app.config import MODEL_CACHE_TTL, PROVIDER_RATE_LIMITS
from app.providers.base import BaseAdapter, RateLimitError

logger = logging.getLogger("google_gate.providers.aistudio_web")

RPC_BASE = (
    "https://alkalimakersuite-pa.clients6.google.com/$rpc/"
    "google.internal.alkali.applications.makersuite.v1.MakerSuiteService"
)
ORIGIN = "https://aistudio.google.com"
# Public web client key embedded in the AI Studio frontend (same for all users).
DEFAULT_WEB_API_KEY = "AIzaSyDdP816MREB3SkjZO04QXbjsigfcI0GWOs"
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/150.0.0.0 Safari/537.36"
)
# Static client-context header present in every live AI Studio capture.
X_GOOG_EXT_CLIENT_BIN = "CAASA1JGSRgBMAE4BEAAUARYAWICRklwAHgB"
MAX_OUTPUT_TOKENS_LIMIT = 65536

# Harm categories 7-10 with threshold 5 (BLOCK_NONE) exactly as sent by the
# live AI Studio web client in the tool/safety config slot.
SAFETY_CONFIG: list[list[Any]] = [
    [None, None, 7, 5],
    [None, None, 8, 5],
    [None, None, 9, 5],
    [None, None, 10, 5],
]

FINISH_REASONS = {
    "STOP",
    "MAX_TOKENS",
    "SAFETY",
    "RECITATION",
    "BLOCKLIST",
    "PROHIBITED_CONTENT",
    "SPII",
    "MALFORMED_FUNCTION_CALL",
}

# Static catalog of models exposed by the AI Studio web UI. The MakerSuite
# RPC surface has no reliable public ListModels wire format, so this catalog
# is maintained manually (mirrors the gemini_web fallback approach).
# Note: gateway-wide deprecated models (e.g. gemini-2.5-*) are intentionally
# excluded - the router rejects them before adapter capability checks.
FALLBACK_MODELS: dict[str, dict[str, Any]] = {
    "gemini-3.7-flash": {
        "displayName": "Gemini 3.7 Flash (AI Studio Web)",
        "maxTokens": 1048576,
        "supportsThinking": True,
        "supportsTools": False,
        "supportsVision": True,
        "isEmbedding": False,
    },
    "gemini-3.6-flash": {
        "displayName": "Gemini 3.6 Flash (AI Studio Web)",
        "maxTokens": 1048576,
        "supportsThinking": True,
        "supportsTools": False,
        "supportsVision": True,
        "isEmbedding": False,
    },
    "gemini-3.1-pro": {
        "displayName": "Gemini 3.1 Pro (AI Studio Web)",
        "maxTokens": 1048576,
        "supportsThinking": True,
        "supportsTools": False,
        "supportsVision": True,
        "isEmbedding": False,
    },
    "gemini-3.5-flash-lite": {
        "displayName": "Gemini 3.5 Flash-Lite (AI Studio Web)",
        "maxTokens": 1048576,
        "supportsThinking": False,
        "supportsTools": False,
        "supportsVision": True,
        "isEmbedding": False,
    },
    "gemini-2.0-flash": {
        "displayName": "Gemini 2.0 Flash (AI Studio Web)",
        "maxTokens": 1048576,
        "supportsThinking": False,
        "supportsTools": False,
        "supportsVision": True,
        "isEmbedding": False,
    },
}

# Antigravity-internal reasoning-tier suffixes stripped before lookup; the
# AI Studio web UI only knows base model names.
TIER_SUFFIXES = ("-high", "-medium", "-low")


def _iter_text_parts(node: Any):
    """Yield text values from protobuf-as-json DataItem nodes ([None, "<text>"])."""
    if isinstance(node, list):
        if len(node) >= 2 and node[0] is None and isinstance(node[1], str):
            yield node[1]
            return
        for child in node:
            yield from _iter_text_parts(child)


def _find_finish_reason(node: Any) -> str | None:
    """Locate a bare finish-reason enum string anywhere in the response tree."""
    if isinstance(node, str):
        return node if node in FINISH_REASONS else None
    if isinstance(node, list):
        for child in node:
            found = _find_finish_reason(child)
            if found:
                return found
    return None


def _extract_error_payload(body: Any) -> dict[str, Any] | None:
    """Detect gRPC-style error objects ({\"error\": {...}}) in decoded bodies."""
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            return err
    return None


def _extract_grpc_detail(text: str) -> str:
    """Pull a human-readable message out of protobuf-as-json google.rpc.Status
    bodies (e.g. [,[3,\"Invalid value at 'generation_config...'\",[...]]])."""
    try:
        parsed = json.loads(text)
    except ValueError:
        return text[:300]

    strings: list[str] = []

    def _walk(node: Any) -> None:
        if isinstance(node, str):
            strings.append(node)
        elif isinstance(node, list):
            for child in node:
                _walk(child)

    _walk(parsed)
    meaningful = [
        s
        for s in strings
        if "Invalid value" in s or "at '" in s or "failed" in s.lower()
    ]
    if meaningful:
        return max(meaningful, key=len)[:300]
    if strings:
        # Skip short enum-ish fragments; prefer the longest descriptive string.
        return max(strings, key=len)[:300]
    return text[:300]


class AIStudioWebAdapter(BaseAdapter):
    name = "aistudio_web"

    def __init__(
        self,
        cookies: str | None = None,
        api_key: str | None = None,
        session: str | None = None,
        enabled: bool = False,
        model_cache_ttl: float = MODEL_CACHE_TTL,
    ):
        limits = PROVIDER_RATE_LIMITS.get("aistudio_web", {})
        super().__init__(
            enabled=enabled,
            rpm=limits.get("rpm", 10),
            tpm=limits.get("tpm", 250000),
            rpd=limits.get("rpd", 0),
            default_cooldown=limits.get("default_cooldown", 60.0),
            min_quota_fraction=limits.get("min_quota_fraction", 0.0),
            model_cache_ttl=model_cache_ttl,
        )
        self.cookies = (
            cookies
            or os.getenv("AISTUDIO_WEB_COOKIES")
            or os.getenv("AISTUDIO_COOKIES")
            or ""
        )
        self.api_key = (
            api_key or os.getenv("AISTUDIO_WEB_API_KEY") or DEFAULT_WEB_API_KEY
        )
        # Opaque client-context blob copied from a live browser request
        # (payload slot 4). Optional; a synthetic value is used when absent.
        self.session_blob = session or os.getenv("AISTUDIO_WEB_SESSION") or ""
        self.proxy = os.getenv("AISTUDIO_WEB_PROXY") or None
        # The AI Studio web client authenticates purely via cookies; sending a
        # SAPISIDHASH that upstream cannot validate yields PERMISSION_DENIED.
        # Opt in only for debugging via AISTUDIO_WEB_SEND_AUTH=1.
        self.send_authorization = os.getenv("AISTUDIO_WEB_SEND_AUTH", "").lower() in (
            "1",
            "true",
        )

        self._http_client: httpx.AsyncClient | None = None
        self._visit_id = self._generate_visit_id()
        self.is_valid_session: bool | None = None
        # Timestamp of the last successful PSIDTS rotation (auto token refresh).
        self._last_rotation: float = 0.0
        # Optional callback (usually router.save_config) invoked after cookie
        # mutations so rotated tokens survive restarts.
        self.persist_cb = None

        # File-upload state (GetAppFolder / GenerateAccessToken / Drive).
        self._app_folder_id: str | None = None
        self._drive_token: str | None = None
        self._drive_token_fetched_at: float = 0.0
        # (sha256, mime) -> Drive file id cache for inline attachments.
        self._uploaded_files: dict[tuple[str, str], str] = {}
        self._discovered_models: dict[str, Any] | None = None
        self._models_fetched_at: float = 0.0
        # Head of the last request body (DEBUG diagnostics for upstream 400s).
        self._last_request_head: str = ""

    # ------------------------------------------------------------------
    # Credentials & headers
    # ------------------------------------------------------------------
    def is_configured(self) -> bool:
        return bool(self.cookies.strip()) and bool(self.extract_sapisid(self.cookies))

    @staticmethod
    def extract_sapisid(cookie_header: str) -> str:
        match = re.search(r"(?:^|;\s*)SAPISID=([^;]+)", cookie_header or "")
        return match.group(1).strip() if match else ""

    def reset_credentials(self) -> None:
        """Clear cookies and cached session state."""
        self.cookies = ""
        self.session_blob = ""
        self.enabled = False
        self.is_valid_session = None
        self._visit_id = self._generate_visit_id()
        self._app_folder_id = None
        self._drive_token = None
        self._drive_token_fetched_at = 0.0
        self._uploaded_files.clear()
        self.clear_cooldown()
        if hasattr(self, "rate_limiter"):
            self.rate_limiter.reset()

    def get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            # Google's $rpc bridges are gRPC-web endpoints served over HTTP/2;
            # match the browser transport when the h2 package is available.
            try:
                import h2  # noqa: F401

                http2 = True
            except ImportError:
                http2 = False
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(300.0, connect=30.0),
                proxy=self.proxy,
                http2=http2,
            )
        return self._http_client

    @staticmethod
    def _sapisidhash(sapisid: str, origin: str = ORIGIN) -> str:
        timestamp = int(time.time())
        digest = hashlib.sha1(f"{timestamp} {sapisid} {origin}".encode()).hexdigest()
        return f"{timestamp}_{digest}"

    def _authorization_header(self) -> str | None:
        sapisid = self.extract_sapisid(self.cookies)
        if not sapisid:
            return None
        # The web client sends all three hash variants; 1P/3P APISID cookies
        # mirror SAPISID for consumer accounts.
        core = self._sapisidhash(sapisid)
        return f"SAPISIDHASH {core} SAPISID1PHASH {core} SAPISID3PHASH {core}"

    def _get_headers(self) -> dict[str, str]:
        headers = {
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "Content-Type": "application/json+protobuf",
            "Origin": ORIGIN,
            "Referer": f"{ORIGIN}/",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-site",
            "User-Agent": USER_AGENT,
            # Chrome client hints matching the UA above (present in every
            # browser capture; frontends gate cookie-authed RPCs on these).
            "sec-ch-ua": '"Not;A=Brand";v="8", "Chromium";v="150", '
            '"Google Chrome";v="150"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Linux"',
            "X-Goog-Api-Key": self.api_key,
            "X-Goog-Authuser": "0",
            "X-Goog-Ext-519733851-Bin": X_GOOG_EXT_CLIENT_BIN,
            "X-User-Agent": "grpc-web-javascript/0.1",
            "X-AiStudio-G1-Tier": "TIER1",
            "X-AiStudio-Visit-Id": self._visit_id,
        }
        auth = self._authorization_header()
        if auth and self.send_authorization:
            headers["Authorization"] = auth
        cookie_header = self.cookies.strip()
        if cookie_header:
            headers["Cookie"] = cookie_header
        return headers

    REQUIRED_COOKIE_NAMES = (
        "SID",
        "HSID",
        "SSID",
        "SAPISID",
        "__Secure-1PSID",
        "__Secure-3PSID",
    )

    def missing_cookie_names(self) -> list[str]:
        """Return required Google session cookies absent from the pasted header."""
        present = self.present_cookie_names()
        return [name for name in self.REQUIRED_COOKIE_NAMES if name not in present]

    def present_cookie_names(self) -> set[str]:
        """Return the cookie names present in the pasted header."""
        return {
            pair.split("=", 1)[0].strip()
            for pair in self.cookies.split(";")
            if "=" in pair
        }

    # ------------------------------------------------------------------
    # PSIDTS auto-refresh (Google rotates these tokens aggressively)
    # ------------------------------------------------------------------
    @staticmethod
    def _replace_cookie(cookie_header: str, name: str, value: str) -> str:
        pattern = re.compile(
            rf"(?P<pre>(?:^|;\s*){re.escape(name)}=)[^;]*", re.MULTILINE
        )
        replaced, count = pattern.subn(
            lambda m: f"{m.group('pre')}{value}", cookie_header
        )
        if count:
            return replaced
        sep = "" if not cookie_header.strip() else "; "
        return f"{cookie_header}{sep}{name}={value}"

    async def rotate_psidts(self) -> bool:
        """Refresh __Secure-1PSIDTS/__Secure-3PSIDTS via RotateCookies.

        Google rotates these tokens frequently and silently degrades sessions
        carrying stale values. Returns True when fresh tokens were applied.
        """
        if not self.is_configured() or "__Secure-1PSIDTS=" not in self.cookies:
            return False

        http = self.get_http_client()
        try:
            resp = await http.post(
                "https://accounts.google.com/RotateCookies",
                content='[000,"-0000000000000000000"]',
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": USER_AGENT,
                    "Cookie": self.cookies.strip(),
                },
            )
        except Exception as e:
            logger.debug(f"[AIStudioWeb] RotateCookies request failed: {e}")
            return False

        updated = False
        for set_cookie in resp.headers.get_list("set-cookie"):
            name, _, remainder = set_cookie.partition("=")
            name = name.strip()
            if name not in ("__Secure-1PSIDTS", "__Secure-3PSIDTS"):
                continue
            fresh = remainder.split(";", 1)[0].strip()
            if fresh and fresh not in self.cookies:
                self.cookies = self._replace_cookie(self.cookies, name, fresh)
                updated = True
                logger.info(f"[AIStudioWeb] Rotated {name}")

        if updated:
            self._last_rotation = time.time()
            if self.persist_cb:
                try:
                    self.persist_cb()
                except Exception as e:
                    logger.warning(f"[AIStudioWeb] Failed to persist rotation: {e}")
        else:
            logger.debug("[AIStudioWeb] RotateCookies issued no new tokens")
        return updated

    async def ensure_fresh_session(self) -> None:
        """Lazily rotate PSIDTS cookies when nearing staleness."""
        if not (self.enabled and self.is_configured()):
            return
        now = time.time()
        if self._last_rotation and now - self._last_rotation < 480.0:
            return
        await self.rotate_psidts()

    @staticmethod
    def _generate_visit_id() -> str:
        # Header variant observed in live traffic: "v1_" + b64(uuid hex).
        return "v1_" + base64.b64encode(uuid.uuid4().hex.encode()).decode()

    @staticmethod
    def _generate_session_blob() -> str:
        # Synthetic stand-in for the opaque payload slot 4 client context.
        alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        raw = "".join(secrets.choice(alphabet) for _ in range(22))
        prefix = "LS" + "".join(secrets.choice(alphabet) for _ in range(8))
        filler = "".join(secrets.choice(alphabet) for _ in range(512))
        return f"!{prefix}ARg{filler}{raw}"

    # ------------------------------------------------------------------
    # File uploads (GetAppFolder -> GenerateAccessToken -> Drive multipart)
    # ------------------------------------------------------------------
    DRIVE_DRIVE_UPLOAD_HOST = "https://content.googleapis.com"

    async def _get_app_folder(self) -> str | None:
        """Fetch the AI Studio app Drive folder id used as upload parent."""
        if self._app_folder_id:
            return self._app_folder_id
        http = self.get_http_client()
        try:
            resp = await http.post(
                f"{RPC_BASE}/GetAppFolder", content="[]", headers=self._get_headers()
            )
            if resp.status_code == 200:
                parsed = resp.json()
                if isinstance(parsed, list) and parsed and isinstance(parsed[0], str):
                    self._app_folder_id = parsed[0]
        except Exception as e:
            logger.debug(f"[AIStudioWeb] GetAppFolder failed: {e}")
        return self._app_folder_id

    async def _get_drive_token(self, force_refresh: bool = False) -> str | None:
        """Obtain a short-lived Drive Bearer token via GenerateAccessToken."""
        now = time.time()
        if (
            not force_refresh
            and self._drive_token
            and now - self._drive_token_fetched_at < 3000.0
        ):
            return self._drive_token
        http = self.get_http_client()
        try:
            resp = await http.post(
                f"{RPC_BASE}/GenerateAccessToken",
                content='["users/me"]',
                headers=self._get_headers(),
            )
            if resp.status_code == 200:
                parsed = resp.json()
                if isinstance(parsed, list) and parsed and isinstance(parsed[0], str):
                    self._drive_token = parsed[0]
                    self._drive_token_fetched_at = now
        except Exception as e:
            logger.debug(f"[AIStudioWeb] GenerateAccessToken failed: {e}")
        return self._drive_token

    async def upload_file(
        self,
        data: bytes,
        mime_type: str,
        filename: str | None = None,
    ) -> str:
        """Upload attachment bytes to the AI Studio Drive app folder.

        Returns the Drive file id referenced by generation requests as a
        DataItem file part ([null,null,null,null,null,["<id>"]]).
        Identical payloads (hash + mime) are uploaded once per session.
        """
        digest = hashlib.sha256(data).hexdigest()
        cache_key = (digest, mime_type)
        cached = self._uploaded_files.get(cache_key)
        if cached:
            return cached

        folder_id = await self._get_app_folder()
        metadata: dict[str, Any] = {
            "name": filename or f"attachment-{digest[:12]}",
        }
        if folder_id:
            metadata["parents"] = [folder_id]

        boundary = uuid.uuid4().hex
        body = (
            f"--{boundary}\r\n"
            f"Content-Type: application/json; charset=UTF-8\r\n\r\n"
            f"{json.dumps(metadata)}\r\n"
            f"--{boundary}\r\n"
            f"Content-Type: {mime_type}\r\n"
            f"Content-Transfer-Encoding: base64\r\n\r\n"
            f"{base64.b64encode(data).decode()}\r\n"
            f"--{boundary}--\r\n"
        ).encode()

        url = (
            f"{self.DRIVE_DRIVE_UPLOAD_HOST}/upload/drive/v3/files"
            f"?uploadType=multipart&key={self.api_key}"
        )

        http = self.get_http_client()
        token = await self._get_drive_token()

        for attempt in range(2):
            if not token:
                raise ValueError(
                    "AI Studio Web could not obtain a Drive access token "
                    "(GenerateAccessToken failed); check cookies."
                )
            try:
                resp = await http.post(
                    url,
                    content=body,
                    headers={
                        "Authorization": f"Bearer {token}",
                        "X-Goog-Api-Key": self.api_key,
                        "X-JavaScript-User-Agent": "google-api-javascript-client/1.1.0",
                        "Origin": ORIGIN,
                        "Referer": f"{ORIGIN}/",
                        "User-Agent": USER_AGENT,
                        "Content-Type": f'multipart/related; boundary="{boundary}"',
                    },
                )
            except Exception as e:
                logger.warning(f"[AIStudioWeb] Drive upload network error: {e}")
                raise ValueError(f"AI Studio Web file upload failed: {e}") from e

            if resp.status_code == 401 and attempt == 0:
                # ya29 tokens expire quickly; force one refresh and retry.
                token = await self._get_drive_token(force_refresh=True)
                continue

            if resp.status_code not in (200, 201):
                raise ValueError(
                    f"AI Studio Web file upload failed "
                    f"({resp.status_code}): {resp.text[:200]}"
                )
            break
        else:
            raise ValueError("AI Studio Web file upload failed after token refresh.")

        file_id = str(resp.json().get("id") or "")
        if not file_id:
            raise ValueError("AI Studio Web file upload returned no file id.")
        self._uploaded_files[cache_key] = file_id
        logger.info(
            f"[AIStudioWeb] Uploaded attachment '{metadata['name']}' "
            f"({len(data)} bytes, {mime_type}) -> {file_id}"
        )
        return file_id

    # ------------------------------------------------------------------
    # Request building (protobuf-as-json wire schema)
    # ------------------------------------------------------------------
    async def _convert_part(self, part: dict[str, Any]) -> list[Any] | None:
        text = part.get("text")
        if isinstance(text, str) and not part.get("thought"):
            return [None, text]

        inline = part.get("inlineData") or part.get("inline_data")
        if isinstance(inline, dict):
            data_b64 = inline.get("data") or ""
            mime_type = inline.get("mimeType") or inline.get("mime_type") or ""
            if data_b64:
                try:
                    data = base64.b64decode(data_b64)
                except Exception:
                    logger.debug(
                        "[AIStudioWeb] Dropping attachment with invalid base64 data"
                    )
                    return None
                file_id = await self.upload_file(
                    data,
                    mime_type or "application/octet-stream",
                    filename=part.get("fileName") or inline.get("fileName"),
                )
                # File-reference DataItem observed in live captures:
                # field 6 holds a single-element file id list.
                return [None, None, None, None, None, [file_id]]
            return None

        # Other non-text parts (function calls etc.) have unverified wire
        # slots; skip them rather than corrupting the request.
        return None

    async def _convert_contents(
        self,
        contents: list[dict[str, Any]] | None,
        system_instruction: dict[str, Any] | None = None,
    ) -> tuple[list[Any], list[Any] | None]:
        """Convert Google-schema contents into makersuite wire contents.

        Returns (contents_wire, system_wire) where system_wire uses the same
        Content shape and is placed at payload slot 5 by the caller.
        """
        wire_contents: list[Any] = []
        for turn in contents or []:
            role = "model" if turn.get("role") in ("model", "assistant") else "user"
            parts_wire: list[Any] = []
            for part in turn.get("parts", []) or []:
                if not isinstance(part, dict):
                    continue
                converted = await self._convert_part(part)
                if converted is not None:
                    parts_wire.append(converted)
                sig = part.get("thoughtSignature")
                if sig:
                    # Thought-signature carrier part observed in live traffic:
                    # empty text with the signature blob at DataItem index 14.
                    sig_part: list[Any] = [None] * 15
                    sig_part[1] = ""
                    sig_part[14] = sig
                    parts_wire.append(sig_part)
            if parts_wire:
                wire_contents.append([parts_wire, role])

        system_wire: list[Any] | None = None
        if system_instruction:
            sys_texts = [
                p["text"]
                for p in system_instruction.get("parts", [])
                if isinstance(p, dict) and p.get("text")
            ]
            if sys_texts:
                system_wire = [[[None, "\n\n".join(sys_texts)]], "user"]

        return wire_contents, system_wire

    def _build_generation_config(
        self, generation_config: dict[str, Any] | None
    ) -> list[Any]:
        gc = generation_config or {}

        max_tokens = gc.get("maxOutputTokens") or MAX_OUTPUT_TOKENS_LIMIT
        # 17-slot layout per live capture; slot indexes map to proto fields:
        #   3=maxOutputTokens, 4=temperature, 5=topP, 6=topK,
        #   7=responseMimeType-ish (older captures), 12=speechConfig
        #   (server rejects scalars here!), 13=candidateCount, 16=thinking.
        cfg: list[Any] = [None] * 17
        cfg[3] = max(1, min(int(max_tokens), MAX_OUTPUT_TOKENS_LIMIT))
        if gc.get("temperature") is not None:
            cfg[4] = float(gc["temperature"])
        if gc.get("topP") is not None:
            cfg[5] = float(gc["topP"])
        if gc.get("topK") is not None:
            cfg[6] = int(gc["topK"])
        candidates = int(gc.get("candidateCount") or 1)
        cfg[13] = max(1, candidates)

        thinking = gc.get("thinkingConfig") or {}
        budget = thinking.get("thinkingBudget")
        if thinking.get("includeThoughts") and (budget is None or budget != 0):
            # [includeThoughts=1, ..., level] per live captures; level 2 is the
            # standard extended-thinking mode, 1 maps to low effort.
            level = 1 if budget == 1 else 2
            cfg[16] = [1, None, None, level]

        unsupported = [
            k
            for k in (
                "stopSequences",
                "responseMimeType",
                "responseSchema",
                "presencePenalty",
                "frequencyPenalty",
                "seed",
            )
            if k in gc
        ]
        if unsupported:
            logger.debug(
                f"[AIStudioWeb] Ignoring unsupported generation config keys: "
                f"{unsupported}"
            )
        return cfg

    def normalize_model(self, model: str) -> str:
        """Map gateway model ids to AI Studio web catalog names.

        Strips Antigravity reasoning-tier suffixes (gemini-3.6-flash-medium ->
        gemini-3.6-flash) and resolves to the closest known catalog entry;
        upstream rejects unknown models with an opaque HTTP 400.
        """
        clean = model.lower().replace("models/", "").strip()
        changed = True
        while changed:
            changed = False
            for suffix in TIER_SUFFIXES:
                if clean.endswith(suffix):
                    clean = clean[: -len(suffix)]
                    changed = True
                    break

        known = set(FALLBACK_MODELS)
        if isinstance(self._discovered_models, dict):
            known.update(k.lower() for k in self._discovered_models)

        # Exact match (pre-strip names count too - ListModels exposes tiers
        # under different ids than the Antigravity backend).
        for candidate in (clean, model.lower().replace("models/", "").strip()):
            if candidate in known:
                return candidate
        matches = [k for k in known if clean.startswith(k)]
        if matches:
            return max(matches, key=len)
        return "gemini-2.0-flash"

    async def build_request_payload(
        self,
        model: str,
        contents: list[dict[str, Any]],
        system_instruction: dict[str, Any] | None = None,
        generation_config: dict[str, Any] | None = None,
    ) -> list[Any]:
        contents_wire, system_wire = await self._convert_contents(
            contents, system_instruction
        )
        clean_model = self.normalize_model(model)
        wire_model = (
            clean_model
            if clean_model.startswith("models/")
            else f"models/{clean_model}"
        )
        session_blob = self.session_blob or self._generate_session_blob()
        payload: list[Any] = [
            wire_model,
            contents_wire,
            SAFETY_CONFIG,
            self._build_generation_config(generation_config),
            session_blob,
            system_wire,
            None,
            None,
            None,
            None,
            1,
            self._visit_id,
        ]
        return payload

    # ------------------------------------------------------------------
    # Response handling
    # ------------------------------------------------------------------
    async def _raise_for_response(
        self, resp: httpx.Response, decoded_error: dict[str, Any] | None = None
    ) -> None:
        status = resp.status_code
        if status < 400:
            return
        detail = ""
        if decoded_error:
            detail = str(decoded_error.get("message") or decoded_error.get("status"))
        if not detail:
            detail = _extract_grpc_detail(resp.text)
        retry_after = 60.0
        header = resp.headers.get("retry-after")
        if header:
            with contextlib.suppress(ValueError):
                retry_after = max(1.0, float(header.strip()))

        if status == 429:
            self.set_cooldown(retry_after, reason=f"AI Studio Web 429: {detail[:200]}")
            raise RateLimitError(
                f"AI Studio Web rate limited (429): {detail}",
                status_code=429,
                retry_after=retry_after,
            )
        if status in (401, 403):
            self.is_valid_session = False
            missing = self.missing_cookie_names()
            guidance = ""
            if missing:
                guidance = (
                    f" Missing cookies: {', '.join(missing)}. Copy the FULL "
                    "Cookie header from a live request."
                )
            else:
                guidance = (
                    " Cookies look complete; the session may have expired "
                    "(__Secure-1PSIDTS rotates frequently) or the egress IP "
                    "may be gated - re-copy a fresh Cookie header from an "
                    "active https://aistudio.google.com tab, optionally via "
                    "AISTUDIO_WEB_PROXY."
                )
            logger.warning(
                f"[AIStudioWeb] {status} PERMISSION/auth failure. Present "
                f"cookies: {sorted(self.present_cookie_names())}"
            )
            raise ValueError(
                f"AI Studio Web authentication failed ({status}): {detail}.{guidance}"
            )
        if status == 400:
            # Opaque Google HTML error pages usually mean a malformed payload
            # or unknown model; dump the request head at DEBUG for diagnosis.
            logger.debug(
                f"[AIStudioWeb] 400 rejected payload head: "
                f"{getattr(self, '_last_request_head', '')[:500]}"
            )
        raise ValueError(f"AI Studio Web Error ({status}): {detail}")

    async def generate_content(
        self,
        model: str,
        contents: list[dict[str, Any]],
        system_instruction: dict[str, Any] | None = None,
        generation_config: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        full_text = ""
        finish_reason = "STOP"
        usage: dict[str, Any] = {}
        async for chunk in self.stream_generate_content(
            model=model,
            contents=contents,
            system_instruction=system_instruction,
            generation_config=generation_config,
            tools=tools,
        ):
            for cand in chunk.get("candidates", []):
                for part in cand.get("content", {}).get("parts", []):
                    if part.get("text"):
                        full_text += part["text"]
                if cand.get("finishReason"):
                    finish_reason = cand["finishReason"]
            if chunk.get("usageMetadata"):
                usage = chunk["usageMetadata"]

        prompt_tokens = sum(
            len(p.get("text", "").split())
            for c in contents
            for p in c.get("parts", [])
            if isinstance(p, dict) and p.get("text")
        )
        completion_tokens = len(full_text.split()) if full_text else 0
        if not usage:
            usage = {
                "promptTokenCount": max(1, prompt_tokens),
                "candidatesTokenCount": completion_tokens,
                "totalTokenCount": max(1, prompt_tokens) + completion_tokens,
            }

        return {
            "responseId": f"aistudio-web-{uuid.uuid4().hex[:12]}",
            "modelVersion": model,
            "candidates": [
                {
                    "index": 0,
                    "text": full_text,
                    "thoughts": "",
                    "content": {"parts": [{"text": full_text}], "role": "model"},
                    "finishReason": finish_reason,
                    "thoughtSignature": None,
                }
            ],
            "text": full_text,
            "thoughts": "",
            "toolCalls": [],
            "finishReason": finish_reason,
            "usageMetadata": usage,
            "thoughtSignature": None,
        }

    async def stream_generate_content(
        self,
        model: str,
        contents: list[dict[str, Any]],
        system_instruction: dict[str, Any] | None = None,
        generation_config: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Generate content via the unary GenerateContent RPC.

        The AI Studio web client streams by calling GenerateContent and
        receiving a JSON array whose first element is the ordered list of
        partial-response chunk objects; deltas are extracted per chunk and
        yielded incrementally to preserve the streaming contract.
        """
        if not self.is_configured():
            raise ValueError(
                "AI Studio Web cookies are not configured (set AISTUDIO_WEB_COOKIES)."
            )

        await self.ensure_fresh_session()
        payload = await self.build_request_payload(
            model, contents, system_instruction, generation_config
        )
        url = f"{RPC_BASE}/GenerateContent"
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        self._last_request_head = body[:800]
        headers = self._get_headers()
        http = self.get_http_client()

        emitted = ""
        finish_reason: str | None = None
        stream_chunks: list[dict[str, Any]] = []

        def _process_node(node: Any) -> None:
            nonlocal emitted, finish_reason
            err = _extract_error_payload(node)
            if err:
                code = int(err.get("code") or 500)
                if code == 429:
                    self.set_cooldown(60.0, reason=str(err.get("message"))[:200])
                    raise RateLimitError(
                        f"AI Studio Web rate limited: {err.get('message')}",
                        status_code=429,
                        retry_after=60.0,
                    )
                raise ValueError(f"AI Studio Web Error ({code}): {err.get('message')}")

            delta_parts = [t for t in _iter_text_parts(node) if t and t.strip()]
            delta = "".join(delta_parts)

            reason = _find_finish_reason(node)
            if reason and not finish_reason:
                finish_reason = reason

            if not delta:
                return

            # Tolerate cumulative snapshots as well as true deltas.
            if delta.startswith(emitted):
                delta = delta[len(emitted) :]
            emitted += delta
            stream_chunks.append(
                {
                    "candidates": [
                        {
                            "content": {"parts": [{"text": delta}], "role": "model"},
                            "index": 0,
                        }
                    ],
                    "modelVersion": model,
                }
            )

        try:
            resp = await http.post(url, content=body.encode("utf-8"), headers=headers)
        except httpx.HTTPError as e:
            logger.warning(f"AI Studio Web request network error: {e}")
            self.set_cooldown(30.0, reason="network error")
            raise RateLimitError(f"AI Studio Web network error: {e}") from e

        try:
            await self._raise_for_response(resp)
            self.is_valid_session = True
            parsed = resp.json()
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(
                f"AI Studio Web returned an unparsable response: {e}"
            ) from e

        # Top-level dict => gRPC-style error payload; list => response whose
        # first element is the ordered chunk list (per live captures).
        if isinstance(parsed, dict):
            _process_node(parsed)
        else:
            outer = parsed if isinstance(parsed, list) else []
            nodes: list[Any] = (
                outer[0] if (outer and isinstance(outer[0], list)) else outer
            )
            for node in nodes:
                if node is None:
                    continue
                _process_node(node)
        while stream_chunks:
            yield stream_chunks.pop(0)

        if not emitted:
            raise ValueError(
                "AI Studio Web returned an empty response (no content chunks parsed)."
            )

        prompt_tokens = sum(
            len(p.get("text", "").split())
            for c in contents
            for p in c.get("parts", [])
            if isinstance(p, dict) and p.get("text")
        )
        completion_tokens = len(emitted.split())
        yield {
            "candidates": [
                {
                    "content": {"parts": [{"text": ""}], "role": "model"},
                    "finishReason": finish_reason or "STOP",
                    "index": 0,
                }
            ],
            "usageMetadata": {
                "promptTokenCount": max(1, prompt_tokens),
                "candidatesTokenCount": completion_tokens,
                "totalTokenCount": max(1, prompt_tokens) + completion_tokens,
            },
            "modelVersion": model,
        }

    async def fetch_available_models(
        self, force_refresh: bool = False
    ) -> dict[str, Any]:
        """Discover models via the MakerSuite ListModels RPC.

        Response entries carry name/displayName/limits/supported methods at
        proto positions 0/3/5/6/7. Falls back to the static catalog when the
        RPC fails or the backend is unconfigured.
        """
        now = time.time()
        if (
            self._discovered_models
            and not force_refresh
            and (now - getattr(self, "_models_fetched_at", 0.0) < self.model_cache_ttl)
        ):
            return {"models": self._discovered_models}

        if not self.is_configured():
            self._discovered_models = FALLBACK_MODELS
            self._models_fetched_at = now
            return {"models": self._discovered_models}

        http = self.get_http_client()
        try:
            resp = await http.post(
                f"{RPC_BASE}/ListModels", content="[]", headers=self._get_headers()
            )
            if resp.status_code == 200:
                parsed = resp.json()
                entries = (
                    parsed[0]
                    if isinstance(parsed, list)
                    and parsed
                    and isinstance(parsed[0], list)
                    else []
                )
                discovered: dict[str, dict[str, Any]] = {}
                for entry in entries:
                    if not isinstance(entry, list) or not entry:
                        continue
                    raw_name = entry[0] if isinstance(entry[0], str) else ""
                    clean_id = raw_name.replace("models/", "").strip()
                    if not clean_id:
                        continue
                    methods = (
                        [str(x) for x in entry[7]]
                        if len(entry) > 7 and isinstance(entry[7], list)
                        else []
                    )
                    # Only expose text-generation models through this gateway.
                    if "generateContent" not in methods:
                        continue

                    display = str(entry[3]) if len(entry) > 3 and entry[3] else clean_id
                    input_limit = (
                        int(entry[5])
                        if len(entry) > 5 and isinstance(entry[5], (int, float))
                        else 1048576
                    )
                    output_limit = (
                        int(entry[6])
                        if len(entry) > 6 and isinstance(entry[6], (int, float))
                        else 0
                    )
                    description = str(entry[4]) if len(entry) > 4 and entry[4] else ""
                    is_embedding = any("embed" in m for m in methods)

                    discovered[clean_id] = {
                        "displayName": f"{display} (AI Studio Web)",
                        "maxTokens": max(input_limit, output_limit, 1),
                        "maxOutputTokens": output_limit or None,
                        "supportsThinking": bool(
                            "thinking" in description.lower()
                            or "pro" in clean_id
                            or clean_id.startswith(("gemini-3",))
                        ),
                        "supportsTools": False,
                        "supportsVision": True,
                        "isEmbedding": is_embedding,
                        "supportedMethods": methods,
                    }

                if discovered:
                    self._discovered_models = discovered
                    self._models_fetched_at = now
                    logger.info(
                        f"[AIStudioWeb] Discovered {len(discovered)} models "
                        f"via ListModels"
                    )
                    return {"models": self._discovered_models}
        except Exception as e:
            logger.debug(f"[AIStudioWeb] ListModels discovery failed: {e}")

        self._discovered_models = FALLBACK_MODELS
        self._models_fetched_at = now
        return {"models": self._discovered_models}
