"""One-time headed login for the Gemini web live-voice profile.

The live lane drives gemini.google.com through a persistent Chromium profile
(``GEMINI_WEB_LIVE_PROFILE_DIR``). Google's anti-abuse pipeline challenges
headless RPC replay, so the login must happen **once in a real browser window**;
afterwards the app's own frontend rotates the session cookies inside that
profile and the lane runs headless.

Container (no display) — the image already ships Xvfb + x11vnc + novnc:

    Xvfb :99 -screen 0 1920x1080x24 &
    DISPLAY=:99 x11vnc -display :99 -forever -nopw -listen 0.0.0.0 -rfbport 5900 &
    websockify --web /usr/share/novnc 6080 localhost:5900 &
    DISPLAY=:99 GEMINI_WEB_LIVE_HEADLESS=0 python scripts/gemini_web_live_login.py

Then open ``http://<host>:6080/vnc.html``, sign in to the Google account whose
Gemini web session you want the lane to use, confirm
``https://gemini.google.com/app`` renders the signed-in composer, and answer the
prompt in the terminal. Ctrl+C (or an empty line) finishes without saving.

The profile directory must live on a persistent volume, or the login evaporates
on container recreate.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.live.gemini_web_transport import (  # noqa: E402
    APP_URL,
    CHROMIUM_PATH,
    PROFILE_DIR,
)


def main() -> int:
    from playwright.sync_api import sync_playwright

    os.makedirs(PROFILE_DIR, exist_ok=True)
    print(f"profile: {PROFILE_DIR}")
    print(f"target:  {APP_URL}")
    if not os.environ.get("DISPLAY"):
        print(
            "warning: DISPLAY is unset — start Xvfb (and x11vnc/websockify for "
            "remote viewing) first, see the module docstring",
            file=sys.stderr,
        )

    kwargs = {"headless": False}
    if CHROMIUM_PATH:
        kwargs["executable_path"] = CHROMIUM_PATH

    with sync_playwright() as pw:
        context = pw.chromium.launch_persistent_context(
            PROFILE_DIR,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--autoplay-policy=no-user-gesture-required",
            ],
            **kwargs,
        )
        page = context.pages[0] if context.pages else context.new_page()
        page.goto(APP_URL, wait_until="domcontentloaded")

        print("\nSign in to Google in the opened window.")
        print("Leave the tab on https://gemini.google.com/app (signed-in composer).")
        try:
            input("Press Enter once the composer is visible (Ctrl+C to abort)... ")
        except (EOFError, KeyboardInterrupt):
            print("\naborted; profile left as-is")
            context.close()
            return 1

        state = page.evaluate(
            """
            () => ({
              url: location.href,
              composer: !!document.querySelector(
                'div[contenteditable="true"],rich-textarea .ql-editor,textarea'),
              signedOut: location.host.includes('accounts.google.com'),
            })
            """
        )
        print(f"final page state: {state}")
        if not state.get("composer") or state.get("signedOut"):
            print(
                "not signed in yet — re-run and complete the Google sign-in",
                file=sys.stderr,
            )
        context.close()

    print("done. The live lane will reuse this profile headlessly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
