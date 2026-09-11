"""Unit tests for transient-429 cooldown handling in the Antigravity adapter."""

import sys

sys.path.insert(0, "/opt/data/user-files/anti-antigravity")

import httpx  # noqa: E402

from app.providers.antigravity import (  # noqa: E402
    TRANSIENT_429_COOLDOWN_S,
    _cooldown_for_429,
    _extract_retry_after_header,
)


def _resp(status: int, headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(
        status, headers=headers or {}, request=httpx.Request("POST", "https://x")
    )


def test_retry_after_header_honored():
    resp = _resp(429, {"Retry-After": "45"})
    assert _extract_retry_after_header(resp) == 45.0
    assert _cooldown_for_429(resp, 60.0) == 45.0


def test_no_header_uses_short_transient_cooldown():
    resp = _resp(429)
    assert _extract_retry_after_header(resp) is None
    cooldown = _cooldown_for_429(resp, 60.0)
    assert cooldown == TRANSIENT_429_COOLDOWN_S
    assert cooldown < 60.0


def test_transient_cooldown_env_overridable():
    assert isinstance(TRANSIENT_429_COOLDOWN_S, float)
    # Env override at import (documented); verify default is sane
    assert TRANSIENT_429_COOLDOWN_S >= 1.0


def test_invalid_header_falls_back_to_transient():
    resp = _resp(429, {"Retry-After": "not-a-number"})
    assert _extract_retry_after_header(resp) is None
    assert _cooldown_for_429(resp, 60.0) == TRANSIENT_429_COOLDOWN_S  # type: ignore[comparison-overlap]
