#!/usr/bin/env python3
"""Probe the native Live API using *this deployment's own account credentials*.

Answers, without an API key: does the signed-in Google account authorize
``BidiGenerateContent``, and which native-audio models does it serve?

Run inside the gateway container (it needs the account's stored refresh token and
outbound access to Google). Nothing here prints the token.

    docker cp scripts/gemini_live_api_probe.py google-gate:/tmp/live_probe.py
    docker exec -e PYTHONPATH=/app google-gate python /tmp/live_probe.py

Exit code 0 means the account lane works; 1 means it does not.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import urllib.error
import urllib.request

LIST_MODELS = "https://generativelanguage.googleapis.com/v1beta/models?pageSize=200"
WS_BASE = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
)
CANDIDATES = [
    m
    for m in [
        os.getenv("GEMINI_LIVE_MODEL", ""),
        "models/gemini-2.5-flash-native-audio-preview-09-2025",
        "models/gemini-live-2.5-flash-preview",
        "models/gemini-2.0-flash-live-001",
        "models/gemini-3.7-flash-live",
    ]
    if m
]


def account_token() -> tuple[str, str]:
    """Load the account token and project from the gateway's own auth manager."""
    sys.path.insert(0, "/app")
    from app.providers.router import router_client

    auth = getattr(getattr(router_client, "antigravity", None), "auth", None)
    if auth is None:
        raise SystemExit("no antigravity auth manager: sign the account in first")
    auth.load_credentials()
    token = asyncio.get_event_loop().run_until_complete(auth.get_valid_access_token())
    project = getattr(auth, "project_id", "") or os.getenv("GOOGLE_PROJECT_ID", "")
    if not token:
        raise SystemExit("account has no access token: sign in via /auth/login")
    return token, project


def list_live_models(token: str, project: str) -> list[str]:
    req = urllib.request.Request(
        LIST_MODELS, headers={"Authorization": f"Bearer {token}"}
    )
    if project:
        req.add_header("x-goog-user-project", project)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        print(f"ListModels -> HTTP {exc.code}: {exc.read().decode()[:300]}")
        return []
    names = [m.get("name", "") for m in data.get("models", [])]
    live = [n for n in names if "native-audio" in n or "-live" in n]
    print(f"ListModels: {len(names)} models, {len(live)} live-capable")
    for n in live:
        print(f"  live: {n}")
    return live


async def probe_ws(token: str, project: str, model: str) -> bool:
    import websockets

    headers = {"Authorization": f"Bearer {token}"}
    if project:
        headers["x-goog-user-project"] = project
    setup = {
        "setup": {
            "model": model,
            "generationConfig": {"responseModalities": ["AUDIO"]},
            "inputAudioTranscription": {},
            "outputAudioTranscription": {},
        }
    }
    try:
        async with websockets.connect(
            WS_BASE, additional_headers=headers, open_timeout=25, max_size=None
        ) as ws:
            await ws.send(json.dumps(setup))
            frame = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
            if "setupComplete" in frame:
                print(f"OK   {model} -> setupComplete (duplex lane usable)")
                return True
            print(f"NO   {model} -> {json.dumps(frame)[:300]}")
            return False
    except Exception as exc:  # ugly upstream errors are the point of the probe
        print(f"NO   {model} -> {type(exc).__name__}: {str(exc)[:220]}")
        return False


async def main() -> int:
    token, project = account_token()
    print(f"account token: yes | project: {project or '(none)'}")
    live = list_live_models(token, project)
    models = [m for m in CANDIDATES if m] + [m for m in live if m not in CANDIDATES]
    seen: list[str] = []
    ok = False
    for model in models:
        if model in seen:
            continue
        seen.append(model)
        if await probe_ws(token, project, model):
            ok = True
            break
    if not ok:
        print("\nRESULT: the account token did NOT open a live socket.")
        print("The mobile app therefore uses something else (internal endpoint):")
        print("capture the app's live session to find it.")
        return 1
    print("\nRESULT: account OAuth works on the public Live API — no API key needed.")
    print("Set nothing; the lane picks the account token up automatically.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
