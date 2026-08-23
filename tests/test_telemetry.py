"""Comprehensive tests for the structured telemetry / routing logging system."""

import json
import logging
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.providers.antigravity import AntigravityAdapter
from app.providers.base import BaseAdapter, ModelNotFoundError, RateLimitError
from app.providers.gemini_api import GeminiApiAdapter
from app.providers.gemini_web import GeminiWebAdapter
from app.providers.router import MultiBackendRouter
from app.telemetry import (
    JsonFormatter,
    RequestIdFilter,
    TextFormatter,
    bind_request_id,
    get_request_id,
    log_event,
    new_request_id,
)


class RecordCollector(logging.Handler):
    """Capture LogRecords for assertions."""

    def __init__(self, level=logging.DEBUG):
        super().__init__(level)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord):
        self.records.append(record)

    def events(self, name: str) -> list[logging.LogRecord]:
        return [r for r in self.records if getattr(r, "event", None) == name]


@pytest.fixture
def routing_capture():
    collector = RecordCollector()
    logger = logging.getLogger("google_gate.routing")
    old_level = logger.level
    logger.addHandler(collector)
    logger.setLevel(logging.DEBUG)
    yield collector
    logger.removeHandler(collector)
    logger.setLevel(old_level)


@pytest.fixture
def base_capture():
    collector = RecordCollector()
    logger = logging.getLogger("google_gate.providers.base")
    old_level = logger.level
    logger.addHandler(collector)
    logger.setLevel(logging.DEBUG)
    yield collector
    logger.removeHandler(collector)
    logger.setLevel(old_level)


def make_adapter(
    cls,
    name: str,
    generate_result=None,
    generate_error: Exception | None = None,
):
    adapter = MagicMock(spec=cls)
    adapter.name = name
    adapter.enabled = True
    adapter.is_configured.return_value = True
    adapter.is_available.return_value = True
    adapter.cooldown_until = 0.0
    adapter.get_cooldown_remaining.return_value = 0.0
    if generate_error is not None:
        adapter.generate_content = AsyncMock(side_effect=generate_error)
    else:
        adapter.generate_content = AsyncMock(return_value=generate_result or {})
    return adapter


def make_capable_router(antigravity, gemini_api, gemini_web) -> MultiBackendRouter:
    """Router where every backend reports capability regardless of probe cache."""
    router = MultiBackendRouter(
        antigravity=antigravity, gemini_api=gemini_api, gemini_web=gemini_web
    )
    router.supports_model = lambda a, model=None, is_embedding=False: True
    return router


# ---------------------------------------------------------------------------
# Formatter & context plumbing
# ---------------------------------------------------------------------------


def test_new_request_id_format():
    rid = new_request_id()
    assert rid.startswith("req_")
    assert len(rid) == len("req_") + 12


def test_request_context_binding():
    assert get_request_id() is None
    bind_request_id("req_test123")
    assert get_request_id() == "req_test123"
    bind_request_id(None)
    assert get_request_id() is None


def test_json_formatter_output_structure():
    record = logging.LogRecord(
        name="google_gate.routing",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="test message",
        args=(),
        exc_info=None,
    )
    record.request_id = "req_abc123"
    record.event = "routing.evaluation"
    record.fields = {"model": "gpt-4o", "candidates": ["gemini_web"]}

    formatted = JsonFormatter().format(record)
    payload = json.loads(formatted)

    assert payload["level"] == "INFO"
    assert payload["logger"] == "google_gate.routing"
    assert payload["msg"] == "test message"
    assert payload["request_id"] == "req_abc123"
    assert payload["event"] == "routing.evaluation"
    assert payload["model"] == "gpt-4o"
    assert payload["candidates"] == ["gemini_web"]
    assert "ts" in payload


def test_json_formatter_truncates_long_strings():
    record = logging.LogRecord(
        name="google_gate.routing",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="x",
        args=(),
        exc_info=None,
    )
    record.fields = {"error": "E" * 5000}
    payload = json.loads(JsonFormatter().format(record))
    assert len(payload["error"]) < 600
    assert payload["error"].endswith("...(truncated)")


def test_text_formatter_appends_event_and_fields():
    record = logging.LogRecord(
        name="google_gate.routing",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )
    record.request_id = "req_xyz"
    record.event = "backend.attempt"
    record.fields = {"backend": "antigravity"}
    out = TextFormatter().format(record)
    assert "request_id=req_xyz" in out
    assert "event=backend.attempt" in out
    assert '"backend"' in out and '"antigravity"' in out


def test_log_event_attaches_fields(routing_capture):
    log_event(
        logging.getLogger("google_gate.routing"),
        logging.INFO,
        "routing.start",
        "starting",
        model="m1",
        candidates=["a", "b"],
    )
    recs = routing_capture.events("routing.start")
    assert len(recs) == 1
    assert recs[0].fields["model"] == "m1"
    assert recs[0].fields["candidates"] == ["a", "b"]


def test_request_id_filter_injects_context():
    filt = RequestIdFilter()
    record = logging.LogRecord(
        name="google_gate.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="m",
        args=(),
        exc_info=None,
    )
    bind_request_id("req_filter_check")
    assert filt.filter(record) is True
    assert record.request_id == "req_filter_check"
    bind_request_id(None)


# ---------------------------------------------------------------------------
# Router integration events
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_routing_events_on_429_fallback_success(routing_capture, base_capture):
    mock_agy = make_adapter(AntigravityAdapter, "antigravity")
    mock_api = make_adapter(
        GeminiApiAdapter,
        "gemini_api",
        generate_error=RateLimitError("API quota exhausted (429)", retry_after=30.0),
    )
    mock_web = make_adapter(
        GeminiWebAdapter,
        "gemini_web",
        generate_result={
            "text": "web response",
            "usageMetadata": {"totalTokenCount": 42},
        },
    )

    router = make_capable_router(mock_agy, mock_api, mock_web)

    result = await router.generate_content(model="gemini-3.7-flash-high", contents=[])

    assert result["text"] == "web response"

    # Full backend evaluation must be logged before any attempt
    eval_events = routing_capture.events("routing.evaluation")
    assert len(eval_events) == 1
    ev = eval_events[0].fields
    assert ev["model"] == "gemini-3.7-flash-high"
    assert set(ev["backends"].keys()) == {"antigravity", "gemini_api", "gemini_web"}
    for state in ev["backends"].values():
        assert {
            "enabled",
            "configured",
            "capable",
            "available",
            "cooldown_remaining_s",
        } <= set(state.keys())
    # free_first order: web -> api -> antigravity
    assert ev["candidates"] == ["gemini_web", "gemini_api", "antigravity"]

    # Attempts logged in execution order
    attempts = routing_capture.events("backend.attempt")
    assert [a.fields["backend"] for a in attempts] == ["gemini_web"]

    # Success outcome includes duration/tokens/attempts chain
    completed = routing_capture.events("routing.completed")
    assert len(completed) == 1
    cf = completed[0].fields
    assert cf["status"] == "success"
    assert cf["served_by"] == "gemini_web"
    assert cf["tokens"] == 42
    assert cf["attempts"][0]["status"] == "success"


@pytest.mark.asyncio
async def test_routing_events_fallback_chain_and_cooldown_reason(
    routing_capture, base_capture
):
    mock_agy = make_adapter(AntigravityAdapter, "antigravity")
    mock_api = make_adapter(
        GeminiApiAdapter,
        "gemini_api",
        generate_error=RateLimitError("quota exceeded (429)", retry_after=25.0),
    )
    mock_web = make_adapter(
        GeminiWebAdapter,
        "gemini_web",
        generate_error=RuntimeError("connection reset"),
    )

    router = make_capable_router(mock_agy, mock_api, mock_web)

    await router.generate_content(model="gpt-4o", contents=[])

    # Attempts recorded in order web -> api -> antigravity
    attempts = routing_capture.events("backend.attempt")
    assert [a.fields["backend"] for a in attempts] == [
        "gemini_web",
        "gemini_api",
        "antigravity",
    ]

    # Rate limited attempt emits backend.rate_limited with retry hint
    rl = routing_capture.events("backend.rate_limited")
    assert len(rl) == 1
    assert rl[0].fields["backend"] == "gemini_api"
    assert rl[0].fields["will_fallback"] is True

    # Non-rate-limit failure emits backend.failed with error class
    failed = routing_capture.events("backend.failed")
    assert len(failed) == 1
    assert failed[0].fields["backend"] == "gemini_web"
    assert failed[0].fields["error_type"] == "RuntimeError"

    # Cooldown on the rate-limited backend requested via adapter with upstream reason
    mock_api.set_cooldown.assert_called_once_with(25.0, reason="quota exceeded (429)")

    # Final completion attributes the winning backend and full chain
    completed = routing_capture.events("routing.completed")
    assert len(completed) == 1
    cf = completed[0].fields
    assert cf["served_by"] == "antigravity"
    assert cf["fallbacks_used"] == 2
    statuses = [a["status"] for a in cf["attempts"]]
    assert statuses == ["error", "rate_limited", "success"]
    assert cf["total_duration_ms"] >= 0


@pytest.mark.asyncio
async def test_model_not_found_logs_evaluation(routing_capture):
    mock_agy = make_adapter(AntigravityAdapter, "antigravity")
    mock_api = make_adapter(GeminiApiAdapter, "gemini_api")
    mock_web = make_adapter(GeminiWebAdapter, "gemini_web")

    router = MultiBackendRouter(
        antigravity=mock_agy, gemini_api=mock_api, gemini_web=mock_web
    )
    # No backend claims capability for this model
    with patch_supports(router, False), pytest.raises(ModelNotFoundError):
        router.check_availability(model="nonexistent-model-xyz")

    unsupported = routing_capture.events("routing.model_unsupported")
    assert len(unsupported) == 1
    fields = unsupported[0].fields
    assert fields["outcome"] == "model_not_found"
    assert fields["model"] == "nonexistent-model-xyz"
    assert all(not b["capable"] for b in fields["backends"].values())


def patch_supports(router: MultiBackendRouter, value: bool):
    import unittest.mock as um

    return um.patch.object(router, "supports_model", return_value=value)


@pytest.mark.asyncio
async def test_streaming_events_include_ttfb_and_chunks(routing_capture):
    async def stream_ok(*args, **kwargs):
        yield {"candidates": []}
        yield {
            "candidates": [],
            "usageMetadata": {"totalTokenCount": 7},
        }
        yield {"candidates": [], "finishReason": "STOP"}

    mock_agy = make_adapter(AntigravityAdapter, "antigravity")
    mock_api = make_adapter(GeminiApiAdapter, "gemini_api")
    mock_web = make_adapter(GeminiWebAdapter, "gemini_web")
    mock_web.stream_generate_content = stream_ok

    router = make_capable_router(mock_agy, mock_api, mock_web)

    chunks = [
        c
        async for c in router.stream_generate_content(
            model="gemini-3.6-flash-high", contents=[]
        )
    ]
    assert len(chunks) == 3

    connected = routing_capture.events("backend.stream_connected")
    assert len(connected) == 1
    assert connected[0].fields["ttfb_ms"] >= 0

    completed = routing_capture.events("routing.completed")
    assert completed[-1].fields["chunks"] == 3
    assert completed[-1].fields["tokens"] == 7


@pytest.mark.asyncio
async def test_all_exhausted_event_emitted(routing_capture):
    from app.providers.base import InMemoryRateTracker

    class ExhaustedAdapter(BaseAdapter):
        name = "exhausted_dummy"

        def __init__(self):
            super().__init__(enabled=True)
            self.rate_limiter = InMemoryRateTracker(rpm=0, tpm=0, rpd=10)
            self.rate_limiter.record_usage(tokens=1)
            self.rate_limiter._minute_requests.clear()

        def is_configured(self):
            return True

        async def generate_content(self, *a, **k):
            raise AssertionError("should never be called")

        async def stream_generate_content(self, *a, **k):
            yield {}

        async def fetch_available_models(self, force_refresh=False):
            return {"models": {}}

    exhausted = ExhaustedAdapter()
    # Fill RPD window completely -> no proactive capacity
    for _ in range(10):
        exhausted.rate_limiter.record_usage(tokens=1)
    exhausted.set_cooldown(45.0, reason="upstream quota")

    router = MultiBackendRouter(
        antigravity=exhausted,
        gemini_api=MagicMock(spec=GeminiApiAdapter),
        gemini_web=MagicMock(spec=GeminiWebAdapter),
    )
    # Disable other backends so only exhausted one is capable/available-ish
    router.gemini_api.enabled = False
    router.gemini_web.enabled = False

    with pytest.raises(RateLimitError):
        await router.generate_content(model="vision", contents=[])

    exhausted_ev = routing_capture.events("routing.all_exhausted")
    assert len(exhausted_ev) == 1
    assert exhausted_ev[0].fields["retry_after_s"] > 0
    agy_state = exhausted_ev[0].fields["backends"]["antigravity"]
    assert agy_state["available"] is False
    assert agy_state["capable"] is True


def test_cooldown_events_include_reason(base_capture):
    class Dummy(BaseAdapter):
        name = "dummy_clear"

        def is_configured(self):
            return True

        async def generate_content(self, *a, **k):
            return {}

        async def stream_generate_content(self, *a, **k):
            yield {}

        async def fetch_available_models(self, force_refresh=False):
            return {"models": {}}

    d = Dummy(enabled=True)
    d.set_cooldown(5.0, reason="upstream 429 quota")
    sets = base_capture.events("cooldown.set")
    assert len(sets) == 1
    sf = sets[0].fields
    assert sf["backend"] == "dummy_clear"
    assert sf["duration_s"] == pytest.approx(5.0)
    assert "429" in (sf["reason"] or "")
    assert sf["cooldown_until"] > time.time()

    d.clear_cooldown()
    clears = base_capture.events("cooldown.clear")
    assert len(clears) == 1
    assert clears[0].fields["backend"] == "dummy_clear"


def test_setup_logging_idempotent():
    from app.telemetry import setup_logging

    root = logging.getLogger("google_gate")
    setup_logging()
    handlers_after_first = len(root.handlers)
    assert handlers_after_first >= 1
    setup_logging()
    setup_logging()
    assert len(root.handlers) == handlers_after_first


def test_telemetry_no_sensitive_payload_in_events(routing_capture):
    """Events must carry metadata only — never raw message content."""
    mock_agy = make_adapter(AntigravityAdapter, "antigravity")
    mock_api = make_adapter(GeminiApiAdapter, "gemini_api")
    mock_web = make_adapter(
        GeminiWebAdapter,
        "gemini_web",
        generate_result={"text": "SECRET-MODEL-OUTPUT"},
    )
    router = make_capable_router(mock_agy, mock_api, mock_web)

    import asyncio

    result = asyncio.run(
        router.generate_content(
            model="m",
            contents=[{"role": "user", "parts": [{"text": "SECRET-USER-PROMPT"}]}],
        )
    )
    assert result["text"] == "SECRET-MODEL-OUTPUT"

    blob = json.dumps(
        [{k: str(v) for k, v in r.__dict__.items()} for r in routing_capture.records],
        default=str,
    )
    assert "SECRET-USER-PROMPT" not in blob
    assert "SECRET-MODEL-OUTPUT" not in blob
