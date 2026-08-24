"""One-time headed login for the AI Studio persistent Playwright profile.

Run this once on a machine with a display (or via X forwarding / VNC):
    .venv/bin/python scripts/aistudio_login.py
Sign in to Google in the opened window, wait until AI Studio loads, then
press Enter here (or Ctrl+C). The profile dir is reused headlessly afterwards.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.providers.aistudio_oracle import (  # noqa: E402
    NEW_CHAT_URL,
    ORACLE_CHROMIUM,
    ORACLE_PROFILE_DIR,
)


def main() -> None:
    from playwright.sync_api import sync_playwright

    os.makedirs(ORACLE_PROFILE_DIR, exist_ok=True)
    print(f"profile: {ORACLE_PROFILE_DIR}")
    with sync_playwright() as pw:
        kwargs = {"headless": False}
        if ORACLE_CHROMIUM:
            kwargs["executable_path"] = ORACLE_CHROMIUM
        ctx = pw.chromium.launch_persistent_context(
            ORACLE_PROFILE_DIR,
            args=["--disable-blink-features=AutomationControlled"],
            **kwargs,
        )
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.goto(NEW_CHAT_URL)
        print("Sign in to Google in the opened window.")
        input("When AI Studio has loaded, press Enter to finish... ")
        print("signed in:", "aistudio.google.com" in page.url and
              "accounts.google" not in page.url)
        ctx.close()
    print("Profile saved. Headless generation can start.")


if __name__ == "__main__":
    main()
