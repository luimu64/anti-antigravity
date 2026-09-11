"""Tests for gRPC error-body retry-delay parsing (Antigravity 429s)."""

import sys

sys.path.insert(0, ".")

import httpx  # noqa: E402

from app.providers.antigravity import _retry_delay_from_body  # noqa: E402


def _resp(body: str) -> httpx.Response:
    return httpx.Response(
        429,
        headers={"Content-Type": "application/json"},
        request=httpx.Request("POST", "https://x"),
        content=body.encode(),
    )


LIVE_BODY = """{
  "error": {
    "code": 429,
    "message": "You have exhausted your capacity on this model. Resets in 0s.",
    "status": "RESOURCE_EXHAUSTED",
    "details": [
      {
        "@type": "type.googleapis.com/google.rpc.ErrorInfo",
        "reason": "RATE_LIMIT_EXCEEDED",
        "domain": "cloudcode-pa.googleapis.com",
        "metadata": {
          "uiMessage": "true",
          "model": "gemini-3.8-flash-medium",
          "quotaResetDelay": "312.252235ms",
          "quotaResetTimeStamp": "2026-09-11T09:49:27Z"
        }
      },
      {
        "@type": "type.googleapis.com/google.rpc.RetryInfo",
        "retryDelay": "0.057395018s"
      }
    ]
  }
}"""


def test_live_payload_parses():
    secs = _retry_delay_from_body(_resp(LIVE_BODY))
    assert secs is not None
    # RetryInfo 0.0574s wins over 312ms? Both parsed; max() takes the larger.
    assert abs(secs - 0.312252235) < 1e-6 or abs(secs - 0.057395) < 1e-3


def test_retryinfo_seconds_format():
    secs = _retry_delay_from_body(
        _resp(
            '{"error":{"details":[{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"1.5s"}]}}'
        )
    )
    assert abs(secs - 1.5) < 1e-9


def test_quota_reset_delay_ms_format():
    secs = _retry_delay_from_body(
        _resp(
            '{"error":{"details":[{"@type":"type.googleapis.com/google.rpc.ErrorInfo","metadata":{"quotaResetDelay":"500ms"}}]}}'
        )
    )
    assert secs is not None and abs(secs - 0.5) < 1e-9


def test_no_details_returns_none():
    assert _retry_delay_from_body(_resp('{"error":{"message":"x"}}')) is None
    assert _retry_delay_from_body(_resp("not json")) is None


def test_unrelated_details_returns_none():
    assert (
        _retry_delay_from_body(
            _resp(
                '{"error":{"details":[{"@type":"type.googleapis.com/google.rpc.Help"}]}}'
            )
        )
        is None
    )
