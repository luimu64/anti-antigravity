"""Cookie-lane live voice transport: drive the Gemini web app, bridge to Live.

Upstream reality this transport is built around (verified against the served
``BardChatUi`` frontend bundle, 2026-09):

* the web client exposes **no** bidirectional audio/video RPC — the only voice
  surfaces are ``GetTtsStream`` (rpcid ``JLpPJe``, voice out), ``ProcessFile``
  (rpcid ``LbusCb``, attachment upload) and ``StreamGenerate`` (rpcid
  ``RxAFq``, text generation);
* ``StreamGenerate`` is fronted by a reCAPTCHA Enterprise challenge that no
  headless client can satisfy (``INTERNAL_API.md`` §8), so raw RPC replay is a
  dead end on this lane;
* a real signed-in browser is *not* challenged. So the lane is driven the way
  the AI Studio oracle is driven — through the app's own UI in a persistent
  profile — and translated to the Live schema at the gateway edge:
  voice in = the utterance is attached to the conversation as an audio file,
  voice out = the app's own "listen" playback is captured from its response
  stream.

Everything above the browser is unit-tested; the browser driver itself is split
into pure helpers (:func:`clean_reply_text`, :func:`extract_dom_state`,
:func:`resolve_audio_out`) so a selector change is a one-line fix here, not a
rewrite. Selector sets are intentionally lists: the web app renames test ids
regularly and a tolerant probe beats a brittle exact match.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import os
import re
import tempfile
import threading
import time
from collections.abc import AsyncIterator
from pathlib import Path
from string import Template
from typing import Any

from app.config import DATA_DIR
from app.live.protocol import (
    DEFAULT_OUTPUT_RATE,
    LiveSetup,
    extract_audio_from_rpc_body,
    pcm16_to_wav,
    sniff_audio_mime,
)
from app.live.transport import LiveTransport, TransportEvent, TurnRequest

logger = logging.getLogger("google_gate.live.gemini_web")

APP_URL = "https://gemini.google.com/app"

# Persistent Google login lives here (same model as the AI Studio oracle):
# sign in ONCE headlessly via scripts/gemini_web_live_login.py, then the app's
# own frontend rotates the session cookies inside the live browser.
PROFILE_DIR = os.getenv(
    "GEMINI_WEB_LIVE_PROFILE_DIR", str(DATA_DIR / "gemini-live-profile")
)
HEADLESS = os.getenv("GEMINI_WEB_LIVE_HEADLESS", "1") != "0"
CHROMIUM_PATH = os.getenv("GEMINI_WEB_LIVE_CHROMIUM", "")
TURN_TIMEOUT = float(os.getenv("GEMINI_WEB_LIVE_TURN_TIMEOUT", "150"))
TTS_TIMEOUT = float(os.getenv("GEMINI_WEB_LIVE_TTS_TIMEOUT", "30"))
POLL_INTERVAL = float(os.getenv("GEMINI_WEB_LIVE_POLL", "1.0"))
# Audio frames handed to the client, in bytes (Live clients expect small deltas).
AUDIO_CHUNK_BYTES = int(os.getenv("GEMINI_WEB_LIVE_CHUNK_BYTES", "65536"))
# requestSource flag the web client stamps on ProcessFile/StreamGenerate.
REQUEST_SOURCE = int(os.getenv("GEMINI_WEB_LIVE_REQUEST_SOURCE", "4"))

# ---------------------------------------------------------------------------
# Tolerant selector sets (the only surface expected to need live confirmation)
# ---------------------------------------------------------------------------
PROMPT_SELECTORS = (
    'div[contenteditable="true"]',
    "rich-textarea .ql-editor",
    "rich-textarea div[role='textbox']",
    "textarea",
)
SEND_SELECTORS = (
    'button[aria-label*="Send" i]',
    "button.send-button",
    'button[type="submit"]',
)
STOP_SELECTORS = (
    'button[aria-label*="Stop" i]',
    'button[aria-label*="Cancel" i]',
)
LISTEN_SELECTORS = (
    'button[aria-label*="Listen" i]',
    'button[aria-label*="Read aloud" i]',
    'button[mattooltip*="Listen" i]',
    'button[mattooltip*="Read aloud" i]',
    '[data-test-id*="tts" i]',
)
REPLY_SELECTORS = (
    "model-response",
    "message-content.model-response-text",
    ".model-response-text",
    "response-container",
)
FILE_INPUT_SELECTORS = ('input[type="file"]',)
SIGNED_OUT_HOSTS = ("accounts.google.com", "consent.google.com")
RPC_URL_MARKERS = ("GetTtsStream", "batchexecute", "BardChatUi")

# Lines the web UI renders around a reply that are chrome, not content.
CHROME_LINES = {
    "gemini said",
    "gemini",
    "show drafts",
    "hide drafts",
    "copy",
    "copied",
    "more",
    "share",
    "edit",
    "listen",
    "read aloud",
    "sources",
    "google it",
    "thumb_up",
    "thumb_down",
    "stop",
    "regenerate",
    "export to docs",
    "drafts",
}
PROMPT_FALLBACK_TEXT = "Listen to this and reply."


class LiveLoginRequired(RuntimeError):
    """The persistent profile is not signed in to Gemini."""

    def __init__(self) -> None:
        super().__init__(
            "Gemini web live login required: run scripts/gemini_web_live_login.py "
            "(headless, reachable over noVNC at :6080) once, sign in to Google, "
            "then close the window."
        )


class LiveTurnAborted(RuntimeError):
    """The client interrupted the turn (barge-in)."""


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested without a browser)
# ---------------------------------------------------------------------------
def clean_reply_text(raw: str) -> str:
    """Strip the web app's reply chrome from a model turn's innerText."""
    if not raw:
        return ""
    lines: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower() in CHROME_LINES:
            continue
        lines.append(stripped)
    text = "\n".join(lines).strip()
    # Some builds prefix the bubble with a role label on the same line.
    text = re.sub(r"^(gemini|model)\s+said:?\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def extract_dom_state(payload: Any) -> dict[str, Any]:
    """Normalise whatever the injected DOM probe returned.

    ``evaluate`` results come back as objects, JSON strings or ``None``
    depending on the build, so shape-normalise here rather than at the call
    site.
    """
    import json

    if payload is None:
        return {}
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except (ValueError, TypeError):
            return {}
    if not isinstance(payload, dict):
        return {}
    state: dict[str, Any] = {
        "reply": clean_reply_text(str(payload.get("reply") or "")),
        "generating": bool(payload.get("generating")),
        "hasStop": bool(payload.get("hasStop")),
        "hasListen": bool(payload.get("hasListen")),
        "turns": int(payload.get("turns") or 0),
        "signedOut": bool(payload.get("signedOut")),
        "inputReady": bool(payload.get("inputReady")),
        "error": str(payload.get("error") or ""),
    }
    return state


def resolve_audio_out(
    raw: bytes, fallback_rate: int = DEFAULT_OUTPUT_RATE
) -> tuple[bytes, str]:
    """Return ``(audio_bytes, mime)`` for whatever the TTS stream produced."""
    if not raw:
        return b"", ""
    mime = sniff_audio_mime(raw)
    if mime.startswith("audio/pcm"):
        mime = f"audio/pcm;rate={fallback_rate}"
    return raw, mime


def chunk_audio(data: bytes, size: int = AUDIO_CHUNK_BYTES) -> list[bytes]:
    """Split a captured utterance into client-sized frames."""
    if size <= 0:
        return [data] if data else []
    return [data[i : i + size] for i in range(0, len(data), size)]


def estimate_tokens(text: str) -> int:
    """Whitespace heuristic, matching the web adapter's local accounting."""
    stripped = (text or "").strip()
    if not stripped:
        return 0
    return max(1, len(stripped.split()))


def turn_prompt(request: TurnRequest) -> str:
    """Text to submit for a turn: persona prefix on turn 1, paragraph breaks kept."""
    prompt = request.prompt.strip()
    if request.first_turn and request.system_instruction.strip():
        header = (
            f"[Instructions]\n{request.system_instruction.strip()}\n[End instructions]"
        )
        return f"{header}\n\n{prompt}".strip()
    return prompt


# ---------------------------------------------------------------------------
# Browser plumbing
# ---------------------------------------------------------------------------
def _probe_js() -> str:
    """DOM probe: reply text, generation state, listen-button availability."""
    from json import dumps

    template = Template("""
    () => {
      const replySelectors = $replies;
      const stopSelectors = $stops;
      const listenSelectors = $listens;
      const promptSelectors = $prompts;
      const listenScope = $scope;
      const pick = (sels) => {
        for (const sel of sels) {
          const nodes = Array.from(document.querySelectorAll(sel));
          if (nodes.length) return nodes;
        }
        return [];
      };
      const replies = pick(replySelectors);
      const last = replies.length ? replies[replies.length - 1] : null;
      const stops = pick(stopSelectors);
      const inputs = pick(promptSelectors);
      let hasStop = stops.length > 0;
      if (!hasStop) {
        hasStop = !!document.querySelector(
          '[aria-busy="true"],[data-test-id*="loading" i],mat-progress-spinner');
      }
      let hasListen = false;
      if (last) {
        const scope = last.closest(listenScope) || last.parentElement;
        hasListen = !!(scope && scope.querySelector(listenScope));
      }
      if (!hasListen) hasListen = pick(listenSelectors).length > 0;
      return {
        reply: last ? (last.innerText || '') : '',
        generating: hasStop,
        hasStop: hasStop,
        hasListen: hasListen,
        turns: replies.length,
        inputReady: inputs.length > 0,
        signedOut: false,
      };
    }
    """)
    return template.substitute(
        replies=dumps(list(REPLY_SELECTORS)),
        stops=dumps(list(STOP_SELECTORS)),
        listens=dumps(list(LISTEN_SELECTORS)),
        prompts=dumps(list(PROMPT_SELECTORS)),
        scope=dumps(", ".join(LISTEN_SELECTORS)),
    )


class _LiveBrowser:
    """Lazily-created persistent Chromium context for one Gemini web tab."""

    def __init__(self) -> None:
        self._pw = None
        self._context = None
        self._page = None
        self._signed_out_hits = 0
        # TTS capture: bodies of RPC responses seen while a capture is armed.
        self._capture_armed = threading.Event()
        self._captured: list[str] = []
        self._capture_lock = threading.Lock()

    # -- lifecycle -------------------------------------------------------
    def page(self):
        if self._page is not None:
            try:
                _ = self._page.url  # liveness probe
                return self._page
            except Exception:
                self._teardown()
        if self._context is None:
            from playwright.sync_api import sync_playwright

            Path(PROFILE_DIR).mkdir(parents=True, exist_ok=True)
            self._pw = sync_playwright().start()
            launch_kwargs: dict[str, Any] = {"headless": HEADLESS}
            if CHROMIUM_PATH:
                launch_kwargs["executable_path"] = CHROMIUM_PATH
            self._context = self._pw.chromium.launch_persistent_context(
                PROFILE_DIR,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--autoplay-policy=no-user-gesture-required",
                ],
                permissions=["microphone"],
                **launch_kwargs,
            )
            self._context.on("response", self._on_response)
        pages = [p for p in self._context.pages if "gemini.google.com" in p.url] or (
            self._context.pages
        )
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
        self.clear_capture()

    def close(self) -> None:
        self._teardown()

    # -- TTS response capture -------------------------------------------
    def _on_response(self, response) -> None:
        if not self._capture_armed.is_set():
            return
        url = response.url or ""
        if not any(marker in url for marker in RPC_URL_MARKERS):
            return
        try:
            body = response.text()
        except Exception as exc:  # response gone / not text
            logger.debug(f"[GeminiWebLive] capture skipped {url[-40:]}: {exc}")
            return
        with self._capture_lock:
            self._captured.append(body)

    def arm_capture(self) -> None:
        with self._capture_lock:
            self._captured = []
        self._capture_armed.set()

    def disarm_capture(self) -> None:
        self._capture_armed.clear()

    def captured_bodies(self) -> list[str]:
        with self._capture_lock:
            return list(self._captured)

    def clear_capture(self) -> None:
        self._capture_armed.clear()
        with self._capture_lock:
            self._captured = []


def _launch_error(exc: BaseException) -> str:
    """Turn a Playwright launch failure into a caller-safe one-liner.

    Playwright emits an ASCII banner plus install instructions; only the first
    line carries the diagnosis a gateway client needs.
    """
    first = (str(exc) or exc.__class__.__name__).strip().splitlines()[0].strip()
    if len(first) > 240:
        first = first[:237] + "..."
    hint = ""
    lowered = first.lower()
    if "executable doesn't exist" in lowered or "playwright install" in lowered:
        hint = " (install a browser, or set GEMINI_WEB_LIVE_CHROMIUM)"
    elif "display" in lowered and "not" in lowered:
        hint = " (a headed launch needs DISPLAY; use GEMINI_WEB_LIVE_HEADLESS=1)"
    return f"browser launch failed: {first}{hint}"


class _PageError(RuntimeError):
    """Page-level failure surfaced to the session with a caller-safe message."""


class GeminiWebLiveTransport(LiveTransport):
    """Live voice in / voice out over the Gemini web app (cookie lane)."""

    name = "gemini_web"

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._browser = _LiveBrowser()
        self._setup: LiveSetup | None = None
        self._interrupt = threading.Event()
        self._turns = 0
        self._last_reply = ""
        self._last_error = ""

    # -- availability ----------------------------------------------------
    def is_available(self) -> bool:
        try:
            import playwright  # noqa: F401
        except Exception:
            return False
        return bool(PROFILE_DIR)

    def status(self) -> dict[str, Any]:
        return {
            "transport": self.name,
            "available": self.is_available(),
            "profile_dir": PROFILE_DIR,
            "headless": HEADLESS,
            "turns": self._turns,
            "last_error": self._last_error,
        }

    # -- session lifecycle ----------------------------------------------
    async def open(
        self, setup: LiveSetup, raw_setup: dict[str, Any] | None = None
    ) -> None:
        # Turn-based lane: the raw setup frame carries nothing extra we can use.
        self._setup = setup
        self._interrupt.clear()
        await asyncio.to_thread(self._ensure_page)

    async def close(self) -> None:
        self._interrupt.set()
        await asyncio.to_thread(self._browser.clear_capture)

    async def abort(self) -> None:
        self._interrupt.set()
        await asyncio.to_thread(self._stop_generation)

    # -- browser steps (sync, always called via to_thread) ---------------
    def _ensure_page(self):
        try:
            page = self._browser.page()
        except Exception as exc:
            logger.warning(f"[GeminiWebLive] {_launch_error(exc)}")
            raise _PageError(_launch_error(exc)) from exc
        if "gemini.google.com" not in page.url:
            page.goto(APP_URL, wait_until="domcontentloaded")
        self._wait_ready(page)
        return page

    def _wait_ready(self, page) -> None:
        deadline = time.monotonic() + TURN_TIMEOUT
        signed_out = 0
        while time.monotonic() < deadline:
            host = ""
            with contextlib.suppress(Exception):
                host = page.url
            if any(bad in host for bad in SIGNED_OUT_HOSTS):
                signed_out += 1
                if signed_out >= 3:
                    raise LiveLoginRequired()
                time.sleep(POLL_INTERVAL)
                continue
            state = extract_dom_state(self._probe(page))
            if state.get("inputReady"):
                self._browser._signed_out_hits = 0
                return
            time.sleep(POLL_INTERVAL)
        raise _PageError("Gemini web input box never became ready (timeout).")

    def _probe(self, page) -> Any:
        try:
            return page.evaluate(_probe_js())
        except Exception as exc:
            logger.debug(f"[GeminiWebLive] probe failed: {exc}")
            return {}

    def _stop_generation(self) -> None:
        try:
            page = self._browser.page()
        except Exception:
            return
        for selector in STOP_SELECTORS:
            with contextlib.suppress(Exception):
                button = page.query_selector(selector)
                if button:
                    button.click(timeout=2000)
                    logger.info("[GeminiWebLive] aborted upstream generation")
                    return

    def _set_prompt(self, page, text: str) -> None:
        """Type into the contenteditable composer and submit."""
        setter = """
        (args) => {
          const {selectors, text} = args;
          let node = null;
          for (const sel of selectors) {
            node = document.querySelector(sel);
            if (node) break;
          }
          if (!node) return false;
          node.focus();
          if (node.tagName === 'TEXTAREA' || node.tagName === 'INPUT') {
            const proto = node.tagName === 'TEXTAREA'
              ? window.HTMLTextAreaElement.prototype
              : window.HTMLInputElement.prototype;
            Object.getOwnPropertyDescriptor(proto, 'value').set.call(node, text);
          } else {
            node.textContent = text;
          }
          node.dispatchEvent(new Event('input', {bubbles: true}));
          node.dispatchEvent(new Event('change', {bubbles: true}));
          return true;
        }
        """
        ok = page.evaluate(setter, {"selectors": list(PROMPT_SELECTORS), "text": text})
        if not ok:
            raise _PageError("could not find the Gemini prompt composer")
        page.keyboard.press("Enter")

    def _submit_text(self, page, text: str) -> None:
        """Submit a text-only turn, falling back to the send button."""
        before = int(extract_dom_state(self._probe(page)).get("turns") or 0)
        self._set_prompt(page, text)
        time.sleep(0.6)
        state = extract_dom_state(self._probe(page))
        if int(state.get("turns") or 0) > before or state.get("generating"):
            return
        for selector in SEND_SELECTORS:
            with contextlib.suppress(Exception):
                button = page.query_selector(selector)
                if button and button.is_enabled():
                    button.click(timeout=3000)
                    return
        logger.debug(
            "[GeminiWebLive] send button fallback found nothing; Enter already sent"
        )

    def _submit_turn(self, page, request: TurnRequest) -> None:
        """Attach the utterance audio (voice in) and submit the turn."""
        text = turn_prompt(request)
        tmp_path = ""
        if request.audio:
            wav = pcm16_to_wav(request.audio, rate=request.audio_rate or 16000)
            with tempfile.NamedTemporaryFile(
                suffix=".wav", dir=str(DATA_DIR), delete=False
            ) as handle:
                handle.write(wav)
                tmp_path = handle.name
            attached = False
            for selector in FILE_INPUT_SELECTORS:
                with contextlib.suppress(Exception):
                    element = page.query_selector(selector)
                    if element:
                        element.set_input_files(tmp_path)
                        attached = True
                        break
            if not attached:
                logger.warning(
                    "[GeminiWebLive] no file input found; sending the turn as text only"
                )
            else:
                time.sleep(0.8)  # let the upload chip render before submitting
        try:
            self._submit_text(
                page, text or (PROMPT_FALLBACK_TEXT if request.audio else "")
            )
        finally:
            if tmp_path:
                with contextlib.suppress(OSError):
                    os.unlink(tmp_path)

    def _await_reply(self, page, baseline_turns: int) -> str:
        """Poll the DOM until the model turn completes."""
        deadline = time.monotonic() + TURN_TIMEOUT
        last_reply = ""
        while time.monotonic() < deadline:
            if self._interrupt.is_set():
                raise LiveTurnAborted()
            state = extract_dom_state(self._probe(page))
            if state.get("signedOut"):
                raise LiveLoginRequired()
            reply = state.get("reply") or ""
            if reply:
                last_reply = reply
            done = int(state.get("turns") or 0) > baseline_turns and not state.get(
                "generating"
            )
            if done and reply and reply == last_reply:
                # Two consecutive identical reads after generation stopped.
                time.sleep(POLL_INTERVAL)
                confirm = extract_dom_state(self._probe(page)).get("reply") or ""
                if confirm == last_reply:
                    return last_reply
            time.sleep(POLL_INTERVAL)
        if last_reply:
            return last_reply
        raise _PageError("timed out waiting for a Gemini web reply")

    def _capture_tts(self, page) -> bytes:
        """Click the app's listen control and harvest the TTS audio bytes."""
        self._browser.arm_capture()
        try:
            clicked = False
            for selector in LISTEN_SELECTORS:
                with contextlib.suppress(Exception):
                    buttons = page.query_selector_all(selector)
                    if buttons:
                        buttons[-1].click(timeout=3000)
                        clicked = True
                        break
            if not clicked:
                logger.info(
                    "[GeminiWebLive] no listen/read-aloud control found; "
                    "voice-out requires the app's TTS control"
                )
                return b""
            deadline = time.monotonic() + TTS_TIMEOUT
            while time.monotonic() < deadline:
                audio = b""
                for body in self._browser.captured_bodies():
                    candidate = extract_audio_from_rpc_body(body)
                    if len(candidate) > len(audio):
                        audio = candidate
                if audio:
                    return audio
                time.sleep(0.5)
            return b""
        finally:
            self._browser.disarm_capture()

    # -- turn execution ---------------------------------------------------
    async def run_turn(self, request: TurnRequest) -> AsyncIterator[TransportEvent]:
        """Run one user turn against the web app, serialised per lane.

        The lane owns exactly one browser tab/conversation, so turns are
        serialised: a second concurrent session waits rather than interleaving
        two conversations into the same chat.
        """
        async with self._lock:
            async for event in self._run_turn_locked(request):
                yield event

    async def _run_turn_locked(
        self, request: TurnRequest
    ) -> AsyncIterator[TransportEvent]:
        if self._setup is None:
            yield TransportEvent.error("session not opened")
            return
        wants_audio = "AUDIO" in self._setup.response_modalities

        baseline = 0
        try:
            page = await asyncio.to_thread(self._ensure_page)
            baseline = int(
                extract_dom_state(await asyncio.to_thread(self._probe, page)).get(
                    "turns"
                )
                or 0
            )
            await asyncio.to_thread(self._submit_turn, page, request)
        except LiveLoginRequired as exc:
            self._last_error = str(exc)
            yield TransportEvent.error(str(exc))
            return
        except LiveTurnAborted:
            yield TransportEvent.done()
            return
        except Exception as exc:
            self._last_error = f"{exc}"
            logger.warning(f"[GeminiWebLive] turn submission failed: {exc}")
            yield TransportEvent.error(f"gemini web live turn failed: {exc}")
            return

        try:
            reply = await asyncio.to_thread(self._await_reply, page, baseline)
        except LiveTurnAborted:
            yield TransportEvent.done()
            return
        except LiveLoginRequired as exc:
            self._last_error = str(exc)
            yield TransportEvent.error(str(exc))
            return
        except Exception as exc:
            self._last_error = f"{exc}"
            logger.warning(f"[GeminiWebLive] reply wait failed: {exc}")
            yield TransportEvent.error(f"gemini web live reply failed: {exc}")
            return

        self._turns += 1
        self._last_reply = reply
        prompt_tokens = estimate_tokens(turn_prompt(request)) + math.ceil(
            len(request.audio) / 3200
        )  # ~100 ms of 16 kHz PCM16 per token-ish unit
        response_tokens = estimate_tokens(reply)

        if not wants_audio:
            yield TransportEvent.text_event(reply)
            yield TransportEvent.done(
                {
                    "promptTokenCount": prompt_tokens,
                    "responseTokenCount": response_tokens,
                    "totalTokenCount": prompt_tokens + response_tokens,
                }
            )
            return

        try:
            raw = await asyncio.to_thread(self._capture_tts, page)
        except Exception as exc:
            logger.warning(f"[GeminiWebLive] TTS capture failed: {exc}")
            raw = b""

        audio, mime = resolve_audio_out(raw, self._setup.output_audio_rate)
        if audio:
            for frame in chunk_audio(audio):
                yield TransportEvent.audio_event(frame, mime)
            # Text rides along as the transcript of what was spoken.
            yield TransportEvent.text_event(reply)
        else:
            self._last_error = "voice-out unavailable: no TTS audio captured"
            logger.warning(f"[GeminiWebLive] {self._last_error}")
            yield TransportEvent.error(self._last_error)

        yield TransportEvent.done(
            {
                "promptTokenCount": prompt_tokens,
                "responseTokenCount": response_tokens,
                "totalTokenCount": prompt_tokens + response_tokens,
            }
        )
