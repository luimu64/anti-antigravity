"""Tests for consecutive-429 escalation (quota-exhaustion bodies only)."""

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


TRANSIENT_BODY = '{"error":{"details":[{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"0.05s"}]}}'
QUOTA_BODY = (
    '{"error":{"message":"You have exhausted your capacity on this model. Resets in 0s.",'
    '"status":"RESOURCE_EXHAUSTED",'
    '"details":[{"@type":"type.googleapis.com/google.rpc.ErrorInfo","reason":"RATE_LIMIT_EXCEEDED",'
    '"metadata":{"quotaResetDelay":"312ms"}},'
    '{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"0.05s"}]}}'
)


def test_escalation_sequence():
    base = TRANSIENT_429_COOLDOWN_S
    assert _escalate_consecutive_429(0) == base
    assert _escalate_consecutive_429(1) == base * 2
    assert _escalate_consecutive_429(2) == base * 4
    assert _escalate_consecutive_429(5) == 60.0
    assert _escalate_consecutive_429(50) == 60.0


def test_transient_429_resets_strikes_never_escalates():
    adapter = AntigravityAdapter()
    c1 = adapter._apply_429_cooldown(_resp429(TRANSIENT_BODY))
    assert c1 <= TRANSIENT_429_COOLDOWN_S
    assert adapter._consecutive_429s == 0  # transient resets ladder
    for _ in range(5):
        c = adapter._apply_429_cooldown(_resp429(TRANSIENT_BODY))
        assert c <= TRANSIENT_429_COOLDOWN_S  # never climbs


def test_quota_429s_escalate():
    adapter = AntigravityAdapter()
    resp = _resp429(QUOTA_BODY)
    cooldowns = [adapter._apply_429_cooldown(resp) for _ in range(4)]
    assert cooldowns[1] > cooldowns[0]
    assert cooldowns[3] > cooldowns[2]
    assert cooldowns[-1] >= 24.0
    assert adapter._consecutive_429s == 4


def test_success_resets_strikes():
    adapter = AntigravityAdapter()
    for _ in range(4):
        adapter._apply_429_cooldown(_resp429(QUOTA_BODY))
    assert adapter._consecutive_429s == 4
    adapter._on_request_success()
    assert adapter._consecutive_429s == 0
    c = adapter._apply_429_cooldown(_resp429(QUOTA_BODY))
    assert c <= TRANSIENT_429_COOLDOWN_S


def test_body_delay_floor_still_applies():
    adapter = AntigravityAdapter()
    c1 = adapter._apply_429_cooldown(_resp429(TRANSIENT_BODY))
    assert c1 >= 0.5
