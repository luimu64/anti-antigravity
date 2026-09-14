"""Antigravity 429 policy: no cooldowns.

Transient 429s change nothing. Model-scoped quota-exhaustion 429s populate
the per-family deny map until the reset time parsed from the body.
"""

import time

from app.providers.antigravity import AntigravityAdapter

TRANSIENT_BODY = """
{
  "error": {"code": 429, "message": "Resource has been exhausted.",
  "status": "RESOURCE_EXHAUSTED",
  "details": [
    {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "0.05s"}
  ]}
}
"""

QUOTA_BODY = """
{
  "error": {"code": 429,
  "message": "You have exhausted your capacity on this model. Resets in 0s.",
  "status": "RESOURCE_EXHAUSTED",
  "details": [
    {"@type": "type.googleapis.com/google.rpc.ErrorInfo",
     "reason": "RATE_LIMIT_EXCEEDED",
     "domain": "cloudcode-pa.googleapis.com",
     "metadata": {"model": "gemini-3.8-flash-medium",
                  "quotaResetDelay": "312.252235ms"}}
  ]}
}
"""


def _resp429(body):
    import httpx

    return httpx.Response(429, text=body)


def test_transient_429_marks_nothing():
    adapter = AntigravityAdapter()
    adapter._apply_429(_resp429(TRANSIENT_BODY), "gemini-3.8-flash")
    assert adapter.get_cooldown_remaining() == 0.0
    assert not adapter._quota_exhausted_until


def test_quota_429_denies_family_only():
    adapter = AntigravityAdapter()
    now = time.time()
    adapter._apply_429(_resp429(QUOTA_BODY), "gemini-3.8-flash-medium")
    # Model family denied until the body's reset delay (~0.3s)
    assert adapter.quota_family_exhausted("gemini-3.8-flash-medium")
    assert adapter.quota_family_exhausted("gemini-3.8-flash")
    # Different family unaffected
    assert not adapter.quota_family_exhausted("gemini-3.6-flash")
    # Cooldown state untouched — other models route.
    assert adapter.get_cooldown_remaining() == 0.0
    assert 0.2 <= adapter._quota_exhausted_until["gemini-3.8-flash"] - now <= 1.0


def test_expiry_clears_deny():
    adapter = AntigravityAdapter()
    adapter.mark_quota_exhausted("gemini-3.8-flash", time.time() - 1)
    assert not adapter.quota_family_exhausted("gemini-3.8-flash")


def test_body_delay_floor_still_applies():
    adapter = AntigravityAdapter()
    now = time.time()
    adapter._apply_429(_resp429(QUOTA_BODY), "gemini-3.8-flash-medium")
    exp = adapter._quota_exhausted_until["gemini-3.8-flash"]
    assert exp - now >= 0.3
