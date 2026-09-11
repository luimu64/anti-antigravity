"""Tests for consecutive-429 exponential backoff escalation."""

import httpx  # noqa: E402

from app.providers.antigravity import (  # noqa: E402
    TRANSIENT_429_COOLDOWN_S,
    AntigravityAdapter,
    _escalate_consecutive_429,
)


def _resp429(body: str) -> httpx.Response:
    return httpx.Response(
        429,
        headers={"Content-Type": "application/json"},
        request=httpx.Request("POST", "https://x"),
        content=body.encode(),
    )


BODY_SUBSECOND = '{"error":{"details":[{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"0.05s"}]}}'


def test_escalation_sequence():
    base = TRANSIENT_429_COOLDOWN_S
    assert _escalate_consecutive_429(0) == base  # first 429: base
    assert _escalate_consecutive_429(1) == base * 2
    assert _escalate_consecutive_429(2) == base * 4  # 12s
    assert _escalate_consecutive_429(5) == 60.0  # capped at default
    assert _escalate_consecutive_429(50) == 60.0  # stays capped


def test_consecutive_429s_escalate_through_adapter():
    adapter = AntigravityAdapter()
    resp = _resp429(BODY_SUBSECOND)
    cooldowns = []
    for _ in range(6):
        c = adapter._apply_429_cooldown(resp)
        cooldowns.append(c)
    # Even though body says 0.05s (floored at 0.5), consecutive strikes escalate
    assert cooldowns[0] <= 3.0
    assert cooldowns[1] > cooldowns[0]
    assert cooldowns[3] > cooldowns[2]
    assert cooldowns[-1] >= 30.0
    assert adapter._consecutive_429s == 6


def test_success_resets_strikes():
    adapter = AntigravityAdapter()
    resp = _resp429(BODY_SUBSECOND)
    for _ in range(4):
        adapter._apply_429_cooldown(resp)
    assert adapter._consecutive_429s == 4
    adapter._on_request_success()
    assert adapter._consecutive_429s == 0
    # Next cooldown back to base
    c = adapter._apply_429_cooldown(resp)
    assert c <= 3.0


def test_body_delay_floor_still_applies():
    adapter = AntigravityAdapter()
    c1 = adapter._apply_429_cooldown(_resp429(BODY_SUBSECOND))
    assert c1 >= 0.5  # body floor honored
