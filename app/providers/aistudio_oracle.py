"""Camofox persistent-login UI-oracle transport for AI Studio Web.

Architecture (single-login model, 2026-08-25):
- ONE persistent camofox session (`AISTUDIO_ORACLE_USER`, default `aistudio`)
  whose cookie store lives on the camofox-data disk volume.
- The user logs in ONCE via noVNC (https://camofox.luimu.dev) when needed;
  after that Google's own signed-in frontend rotates SIDCC/PSIDTS inside the
  live browser. No cookie files, no jar imports, no RotateCookies calls.
- Imported-cookie pipelines are dead by design: Google invalidates imported
  SIDCC generations server-side within the hour and RotateCookies returns 403
  for them (verified 2026-08-24). Login state can only come from a real login.

Design notes:
- One long-lived tab per gateway process; recreated lazily if it dies.
- Requests serialized with an asyncio.Lock: one conversation turn at a time.
- Each request starts a FRESH chat so DOM extraction has exactly one reply.
- On sign-out detection, raises OracleLoginRequired; callers surface it and a
  human re-authenticates over noVNC, then generation resumes automatically.
"""

import asyncio
import contextlib
import json
import logging
import os
import re
import time
import urllib.error
import urllib.request

logger = logging.getLogger("google_gate.providers.aistudio_oracle")

CAMOFOX_URL = os.getenv("CAMOFOX_URL", "http://camofox-browser:9377")
CAMOFOX_API_KEY = os.getenv("CAMOFOX_API_KEY", "")
ORACLE_USER = os.getenv("AISTUDIO_ORACLE_USER", "aistudio")
# Kept for backwards compatibility; UNUSED under the persistent-login model.
ORACLE_COOKIES_FILE = os.getenv(
    "AISTUDIO_ORACLE_COOKIES_FILE", "/opt/data/misc/ai-studio/cookies-fresh.txt"
)
ORACLE_TAB_TIMEOUT = float(os.getenv("AISTUDIO_ORACLE_TAB_TIMEOUT", "120"))
ORACLE_POLL_INTERVAL = float(os.getenv("AISTUDIO_ORACLE_POLL_INTERVAL", "1.5"))

NEW_CHAT_URL = "https://aistudio.google.com/prompts/new_chat"
NO_VNC_HINT = (
    "AI Studio login required: open https://camofox.luimu.dev and sign in to "
    "Google once; generation resumes automatically afterwards."
)


class OracleError(RuntimeError):
    """The camofox oracle could not complete a generation."""


class OracleLoginRequired(OracleError):
    """The persistent session is not signed in to AI Studio."""


def _bearer_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {CAMOFOX_API_KEY}",
        "Content-Type": "application/json",
    }


def _request(
    path: str,
    method: str = "GET",
    payload: dict | None = None,
    timeout: float = 90.0,
) -> dict:
    body = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        f"{CAMOFOX_URL}{path}",
        method=method,
        headers=_bearer_headers(),
        data=body,
    )
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raise OracleError(f"camofox {method} {path} -> {e.code}: {e.read().decode()[:200]}") from e
    except (urllib.error.URLError, TimeoutError) as e:
        raise OracleError(f"camofox unreachable at {CAMOFOX_URL}: {e}") from e


async def _async_request(
    path: str,
    method: str = "GET",
    payload: dict | None = None,
    timeout: float = 90.0,
) -> dict:
    return await asyncio.to_thread(_request, path, method, payload, timeout)


class AistudioOracle:
    """Serialised access to one AI Studio browser tab (persistent login)."""

    name = "aistudio_web"

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._tab_id: str | None = None

    # ------------------------------------------------------------------
    # Tab lifecycle
    # ------------------------------------------------------------------
    def is_available(self) -> bool:
        return bool(CAMOFOX_URL and CAMOFOX_API_KEY)

    def _ensure_tab(self) -> None:
        """Return a live tab on new_chat; create/recover one if needed."""
        if self._tab_id:
            alive = False
            try:
                result = _request(f"/tabs/{self._tab_id}/navigate", "POST", {
                    "userId": ORACLE_USER, "url": NEW_CHAT_URL,
                })
                alive = bool(result.get("tabId"))
            except OracleError:
                alive = False
            if alive:
                return
            # stale/dead tab: drop it and fall through to creation
            with contextlib.suppress(OracleError):
                _request(f"/tabs/{self._tab_id}?userId={ORACLE_USER}", "DELETE")
            self._tab_id = None

        # Prefer reusing an existing healthy signed-in tab in this session
        # before spawning a new one.
        try:
            listing = _request(f"/tabs?userId={ORACLE_USER}")
            for tab in listing.get("tabs", []):
                tid = tab.get("tabId")
                if not tid or "accounts.google" in str(tab.get("url", "")):
                    continue
                probe = _request(f"/tabs/{tid}/evaluate", "POST", {
                    "userId": ORACLE_USER,
                    "expression": (
                        '(() => ({ready: !!document.querySelector("textarea"),'
                        ' ok: location.host.includes("aistudio")}))()'
                    ),
                })
                state = probe.get("result") or {}
                if isinstance(state, str):
                    state = json.loads(state)
                if state.get("ready") and state.get("ok"):
                    self._tab_id = tid
                    _request(f"/tabs/{tid}/navigate", "POST", {
                        "userId": ORACLE_USER, "url": NEW_CHAT_URL,
                    })
                    return
        except (OracleError, KeyError, json.JSONDecodeError):
            pass

        created = _request("/tabs", "POST", {
            "userId": ORACLE_USER,
            "sessionKey": ORACLE_USER,
            "url": NEW_CHAT_URL,
        })
        self._tab_id = created["tabId"]

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + ORACLE_TAB_TIMEOUT
        signed_out_seen = 0
        while time.monotonic() < deadline:
            result = _request(f"/tabs/{self._tab_id}/evaluate", "POST", {
                "userId": ORACLE_USER,
                "expression": (
                    '(() => ({ready: !!document.querySelector("textarea"),'
                    ' onNewChat: location.href.includes("new_chat"),'
                    ' signedOut: location.host.includes("accounts.google")}))()'
                ),
            })
            state = result.get("result") or {}
            if isinstance(state, str):
                state = json.loads(state)
            if state.get("signedOut"):
                # Transient during redirect chains; only fatal if it persists.
                signed_out_seen += 1
                if signed_out_seen >= 3:
                    raise OracleLoginRequired(NO_VNC_HINT)
            elif state.get("ready") and state.get("onNewChat"):
                return
            time.sleep(1.5)
        raise OracleError("AI Studio tab never became ready (textarea timeout).")

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def _send_prompt(self, prompt: str) -> None:
        expr = (
            '(() => {'
            '  const ta=document.querySelector("textarea");'
            '  if(!ta) return "no-textarea";'
            '  const setter=Object.getOwnPropertyDescriptor('
            '      window.HTMLTextAreaElement.prototype,"value");'
            '  setter.set.call(ta, ' + json.dumps(prompt) + ');'
            '  ta.dispatchEvent(new Event("input",{bubbles:true}));'
            '  setTimeout(()=>{const btn=document.querySelector('
            '      \'button[aria-label*="Run"],button[type="submit"]\');'
            '      if(btn) btn.click();},400);'
            '  return "sent";})()'
        )
        result = _request(f"/tabs/{self._tab_id}/evaluate", "POST", {
            "userId": ORACLE_USER, "expression": expr,
        })
        if result.get("result") != "sent":
            raise OracleError(f"could not type prompt into AI Studio: {result}")

    def _extract_reply(self) -> tuple[str | None, str | None]:
        """Return (reply_text, error_text) - whichever is found first.

        Reads the last ms-chat-turn element: a fresh chat yields exactly two
        turns (user + model), so there is no history ambiguity.
        """
        expr = (
            '(() => {'
            '  const turns = document.querySelectorAll("ms-chat-turn");'
            '  if (turns.length < 2) return JSON.stringify({pending:true});'
            '  const last = turns[turns.length - 1];'
            '  const t = last.innerText;'
            '  // Feedback icons only render once generation completes.'
            '  if (!t.includes("thumb_up")) return JSON.stringify({pending:true});'
            '  return JSON.stringify({text: t});})()'
        )
        result = _request(f"/tabs/{self._tab_id}/evaluate", "POST", {
            "userId": ORACLE_USER, "expression": expr,
        })
        data = result.get("result") or {}
        if isinstance(data, str):
            data = json.loads(data)
        if data.get("pending"):
            return None, None
        text = data.get("text") or ""
        if "An internal error has occurred" in text:
            return None, "AI Studio internal error on generation"
        if "accounts.google" in text or text.strip() == "Sign in":
            return None, NO_VNC_HINT
        # Strip chrome: header lines up to the timestamp, trailing feedback icons.
        m = re.search(r"Model\s+[^\n]*\n(.*)", text, re.DOTALL)
        body = m.group(1) if m else text
        for stop in ("thumb_up", "thumb_down"):
            idx = body.find(stop)
            if idx >= 0:
                body = body[:idx]
        body = body.strip()
        return (body, None) if body else (None, None)

    def generate_once(self, prompt: str) -> str:
        """Send one prompt through the AI Studio UI and return the model's reply."""
        self._ensure_tab()
        self._wait_ready()
        self._send_prompt(prompt)

        deadline = time.monotonic() + ORACLE_TAB_TIMEOUT
        last_err: str | None = None
        while time.monotonic() < deadline:
            time.sleep(ORACLE_POLL_INTERVAL)
            try:
                reply, error = self._extract_reply()
            except OracleError as e:
                # Transient camofox hiccups during generation; keep polling.
                logger.debug(f"[oracle] transient poll error: {e}")
                continue
            if reply:
                return reply
            if error:
                last_err = error
                break
        raise OracleError(last_err or "timed out waiting for AI Studio response")

    def login_status(self) -> dict:
        """Report whether the persistent session currently reaches AI Studio."""
        try:
            self._ensure_tab()
            self._wait_ready()
            return {"signedIn": True, "tabId": self._tab_id}
        except OracleLoginRequired:
            return {"signedIn": False, "hint": NO_VNC_HINT}
        except OracleError as e:
            return {"signedIn": False, "error": str(e)}

    def open_login_tab(self) -> str:
        """Point a tab at the Google sign-in entry for the human via noVNC."""
        self._ensure_tab()
        with contextlib.suppress(OracleError):
            self._wait_ready()
        url = "https://aistudio.google.com/prompts/new_chat"
        _request(f"/tabs/{self._tab_id}/navigate", "POST", {
            "userId": ORACLE_USER, "url": url,
        })
        return (
            f"Tab {self._tab_id} navigated to {url}. Open https://camofox.luimu.dev "
            "and complete the Google sign-in in that tab."
        )

    # ------------------------------------------------------------------
    # Async facade used by the adapter
    # ------------------------------------------------------------------
    async def generate(self, prompt: str) -> str:
        async with self._lock:
            return await asyncio.to_thread(self.generate_once, prompt)
