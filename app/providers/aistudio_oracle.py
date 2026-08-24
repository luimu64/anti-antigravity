"""Playwright persistent-profile UI-oracle transport for AI Studio Web.

Architecture (2026-08-25, replaces the camofox transport):
- ONE Playwright Chromium instance owned by this process, launched against a
  PERSISTENT user-data dir (`AISTUDIO_ORACLE_PROFILE_DIR`). The Google login
  lives in that profile; afterwards Google's own signed-in frontend rotates
  SIDCC/PSIDTS inside the live browser.
- One-time login: run scripts/aistudio_login.py (headed) on any machine that
  shares the profile dir volume; sign in; close. Headless generation resumes
  automatically. No cookie files, no jar imports.
- No anti-bot measures needed: plain Chromium, default fingerprints.

Design notes:
- Requests serialized with an asyncio.Lock: one conversation turn at a time.
- Each request starts a FRESH chat so DOM extraction has exactly one reply.
- Persistent sign-out raises OracleLoginRequired so callers surface the
  re-login instruction.
"""

import asyncio
import contextlib
import logging
import os
import re
import time

logger = logging.getLogger("google_gate.providers.aistudio_oracle")

ORACLE_PROFILE_DIR = os.getenv(
    "AISTUDIO_ORACLE_PROFILE_DIR", "/opt/data/user-files/anti-antigravity/data/aistudio-profile"
)
ORACLE_HEADLESS = os.getenv("AISTUDIO_ORACLE_HEADLESS", "1") != "0"
ORACLE_TAB_TIMEOUT = float(os.getenv("AISTUDIO_ORACLE_TAB_TIMEOUT", "120"))
ORACLE_POLL_INTERVAL = float(os.getenv("AISTUDIO_ORACLE_POLL_INTERVAL", "1.5"))

NEW_CHAT_URL = "https://aistudio.google.com/prompts/new_chat"
LOGIN_HINT = (
    "AI Studio login required: run scripts/aistudio_login.py (headed) once, "
    "sign in to Google, then close the window."
)

RUN_BUTTON_SELECTOR = 'button[aria-label*="Run"], button[type="submit"]'
TEXTAREA_SELECTOR = "textarea"
TURN_SELECTOR = "ms-chat-turn"


class OracleError(RuntimeError):
    """The Playwright oracle could not complete a generation."""


class OracleLoginRequired(OracleError):
    """The persistent profile is not signed in to AI Studio."""


class _Browser:
    """Lazily-created persistent Chromium context + page (sync Playwright)."""

    def __init__(self) -> None:
        self._pw = None
        self._context = None
        self._page = None

    def page(self):
        if self._page is not None:
            try:
                _ = self._page.url  # liveness probe
                return self._page
            except Exception:
                self._teardown()
        if self._context is None:
            from playwright.sync_api import sync_playwright

            os.makedirs(ORACLE_PROFILE_DIR, exist_ok=True)
            self._pw = sync_playwright().start()
            self._context = self._pw.chromium.launch_persistent_context(
                ORACLE_PROFILE_DIR,
                headless=ORACLE_HEADLESS,
                args=["--disable-blink-features=AutomationControlled"],
            )
        pages = [p for p in self._context.pages if "aistudio" in p.url] or \
            self._context.pages
        self._page = pages[0] if pages else self._context.new_page()
        return self._page

    def _teardown(self) -> None:
        self._page = None
        if self._context:
            with contextlib.suppress(Exception):
                self._context.close()
            self._context = None
        if self._pw:
            with contextlib.suppress(Exception):
                self._pw.stop()
            self._pw = None

    def close(self) -> None:
        self._teardown()


class AistudioOracle:
    """Serialised access to one AI Studio tab (persistent Playwright profile)."""

    name = "aistudio_web"

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._browser = _Browser()

    def is_available(self) -> bool:
        return bool(ORACLE_PROFILE_DIR)

    # ------------------------------------------------------------------
    # Page lifecycle
    # ------------------------------------------------------------------
    def _ensure_page(self):
        page = self._browser.page()
        if not page.url.startswith("https://aistudio.google.com") or \
                "new_chat" not in page.url:
            page.goto(NEW_CHAT_URL, wait_until="domcontentloaded")
        return page

    def _wait_ready(self, page) -> None:
        deadline = time.monotonic() + ORACLE_TAB_TIMEOUT
        signed_out_seen = 0
        while time.monotonic() < deadline:
            state = page.evaluate(
                f"""() => ({{
                    ready: !!document.querySelector({TEXTAREA_SELECTOR!r}),
                    onNewChat: location.href.includes("new_chat"),
                    signedOut: location.host.includes("accounts.google"),
                }})"""
            )
            if state.get("signedOut"):
                # Transient during redirect chains; only fatal if it persists.
                signed_out_seen += 1
                if signed_out_seen >= 3:
                    raise OracleLoginRequired(LOGIN_HINT)
            elif state.get("ready") and state.get("onNewChat"):
                return
            time.sleep(1.5)
        raise OracleError("AI Studio page never became ready (textarea timeout).")

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def _send_prompt(self, page, prompt: str) -> None:
        sent = page.evaluate(
            f"""(prompt) => {{
                const ta = document.querySelector({TEXTAREA_SELECTOR!r});
                if (!ta) return false;
                const setter = Object.getOwnPropertyDescriptor(
                    window.HTMLTextAreaElement.prototype, "value");
                setter.set.call(ta, prompt);
                ta.dispatchEvent(new Event("input", {{bubbles: true}}));
                setTimeout(() => {{
                    const btn = document.querySelector({RUN_BUTTON_SELECTOR!r});
                    if (btn) btn.click();
                }}, 400);
                return true;
            }}""",
            prompt,
        )
        if not sent:
            raise OracleError("could not type prompt into AI Studio")

    def _extract_reply(self, page) -> tuple[str | None, str | None]:
        """Return (reply_text, error_text) - whichever is found first."""
        text = page.evaluate(
            f"""() => {{
                const turns = document.querySelectorAll({TURN_SELECTOR!r});
                if (turns.length < 2) return null;
                const t = turns[turns.length - 1].innerText;
                // Feedback icons only render once generation completes.
                return t.includes("thumb_up") ? t : null;
            }}"""
        )
        if text is None:
            return None, None
        if "An internal error has occurred" in text:
            return None, "AI Studio internal error on generation"
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
        page = self._ensure_page()
        try:
            self._wait_ready(page)
        except Exception:
            # Dead/crashed renderer: rebuild the page once before giving up.
            self._browser._teardown()
            page = self._ensure_page()
            self._wait_ready(page)
        self._send_prompt(page, prompt)

        deadline = time.monotonic() + ORACLE_TAB_TIMEOUT
        last_err: str | None = None
        while time.monotonic() < deadline:
            time.sleep(ORACLE_POLL_INTERVAL)
            try:
                reply, error = self._extract_reply(page)
            except OracleError:
                continue
            except Exception as e:  # transient evaluation errors mid-generation
                logger.debug(f"[oracle] transient poll error: {e}")
                continue
            if reply:
                return reply
            if error:
                last_err = error
                break
        raise OracleError(last_err or "timed out waiting for AI Studio response")

    def login_status(self) -> dict:
        """Report whether the persistent profile currently reaches AI Studio."""
        try:
            page = self._ensure_page()
            self._wait_ready(page)
            return {"signedIn": True}
        except OracleLoginRequired:
            return {"signedIn": False, "hint": LOGIN_HINT}
        except OracleError as e:
            return {"signedIn": False, "error": str(e)}

    async def generate(self, prompt: str) -> str:
        async with self._lock:
            return await asyncio.to_thread(self.generate_once, prompt)
