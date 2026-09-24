"""Live (bidiGenerateContent) voice sessions for google-gate.

The gateway exposes a WebSocket that speaks Google's **Live API**
(``BidiGenerateContent``) message schema so any Live-capable client can point at
``ws://<gateway>/v1/live`` instead of Google, while the upstream lane is a
browser-reverse-engineered web backend:

* :mod:`app.live.protocol` — message validation, audio helpers, frame builders.
* :mod:`app.live.session` — the transport-agnostic session state machine
  (turn assembly, barge-in, usage accounting).
* :mod:`app.live.gemini_web_transport` — the cookie-lane upstream: drives the
  real Gemini web app in a persistent signed-in browser profile.

Why a browser and not the RPC: ``gemini.google.com`` fronts ``StreamGenerate``
with a reCAPTCHA Enterprise challenge that no headless client can satisfy (see
``INTERNAL_API.md`` §8), and the web frontend has no bidirectional audio RPC at
all — its only voice endpoints are ``GetTtsStream`` (voice out) and file
attachments (voice in). The transport therefore drives the app's own UI, which
runs inside a browser context Google trusts, and bridges it to the Live schema.
"""

from app.live.session import LiveSession, LiveSessionError  # noqa: F401

__all__ = ["LiveSession", "LiveSessionError"]
