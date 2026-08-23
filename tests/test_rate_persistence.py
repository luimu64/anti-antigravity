"""Tests for on-disk persistence of sliding-window rate limit counters."""

import json

import pytest

from app.providers.base import InMemoryRateTracker


def test_counters_survive_tracker_restart(tmp_path):
    """A freshly constructed tracker restores usage recorded by its predecessor."""
    store = tmp_path / "rate_limits.json"

    first = InMemoryRateTracker(
        rpm=3, tpm=100, rpd=5, persist_path=store, persist_key="gemini_api"
    )
    first.record_usage(tokens=30, now=1000.0)
    first.record_usage(tokens=40, now=1010.0)

    assert store.exists()

    second = InMemoryRateTracker(
        rpm=3, tpm=100, rpd=5, persist_path=store, persist_key="gemini_api"
    )
    stats = second.get_stats(now=1020.0)

    assert stats["rpm_used"] == 2
    assert stats["tpm_used"] == 70
    assert stats["rpd_used"] == 2
    # Capacity must still be enforced after the "reboot"
    assert second.has_capacity(estimated_tokens=10, now=1020.0) is True
    second.record_usage(tokens=30, now=1025.0)
    assert second.get_stats(now=1026.0)["rpm_used"] == 3
    assert second.has_capacity(estimated_tokens=1, now=1026.0) is False


def test_stale_entries_are_pruned_after_reload(tmp_path):
    """Entries that expired while the process was down are discarded on load."""
    store = tmp_path / "rate_limits.json"

    first = InMemoryRateTracker(persist_path=store, persist_key="antigravity")
    first.record_usage(tokens=10, now=1000.0)

    # Restart well beyond every window (minute + day)
    second = InMemoryRateTracker(persist_path=store, persist_key="antigravity")
    stats = second.get_stats(now=1000.0 + 90000.0)

    assert stats["rpm_used"] == 0
    assert stats["tpm_used"] == 0
    assert stats["rpd_used"] == 0


def test_reset_clears_persisted_state(tmp_path):
    """reset() also wipes the persisted counters so a reboot cannot resurrect them."""
    store = tmp_path / "rate_limits.json"

    first = InMemoryRateTracker(persist_path=store, persist_key="gemini_web")
    first.record_usage(tokens=10, now=1000.0)
    first.reset()

    second = InMemoryRateTracker(persist_path=store, persist_key="gemini_web")

    assert second.get_stats()["rpm_used"] == 0
    assert second.get_stats()["rpd_used"] == 0


def test_backends_are_isolated_in_shared_store(tmp_path):
    """Each backend key only touches its own section of the shared file."""
    store = tmp_path / "rate_limits.json"

    api = InMemoryRateTracker(persist_path=store, persist_key="gemini_api")
    web = InMemoryRateTracker(persist_path=store, persist_key="gemini_web")
    api.record_usage(tokens=10, now=1000.0)
    web.record_usage(tokens=20, now=1000.0)
    web.record_usage(tokens=20, now=1001.0)

    reloaded_api = InMemoryRateTracker(persist_path=store, persist_key="gemini_api")
    reloaded_web = InMemoryRateTracker(persist_path=store, persist_key="gemini_web")

    assert reloaded_api.get_stats(now=1002.0)["rpm_used"] == 1
    assert reloaded_web.get_stats(now=1002.0)["rpm_used"] == 2


def test_corrupted_store_is_tolerated(tmp_path):
    """A corrupt store file neither raises nor blocks fresh accounting."""
    store = tmp_path / "rate_limits.json"
    store.write_text("{not valid json!!")

    tracker = InMemoryRateTracker(persist_path=store, persist_key="gemini_api")
    tracker.record_usage(tokens=10, now=1000.0)

    # The write repaired the store with valid JSON
    data = json.loads(store.read_text())
    assert list(data["backends"]) == ["gemini_api"]

    reloaded = InMemoryRateTracker(persist_path=store, persist_key="gemini_api")
    assert reloaded.get_stats(now=1005.0)["rpm_used"] == 1


def test_no_persistence_by_default():
    """Trackers without a persist path behave exactly like before (no disk I/O)."""
    tracker = InMemoryRateTracker()
    assert tracker._persist_path is None
    tracker.record_usage(tokens=5)


def test_base_adapter_wires_persistence(monkeypatch, tmp_path):
    """BaseAdapter passes the configured store path and backend name to its tracker."""
    import app.providers.base as base_module
    from app.providers.gemini_api import GeminiApiAdapter

    store = tmp_path / "rate_limits.json"
    monkeypatch.setattr(base_module, "RATE_LIMITS_PERSIST", True)
    monkeypatch.setattr(base_module, "RATE_LIMITS_FILE", store)

    adapter = GeminiApiAdapter(api_key="test-key", enabled=True)
    assert adapter.rate_limiter._persist_path == store
    assert adapter.rate_limiter._persist_key == "gemini_api"

    adapter.record_usage(tokens=42)
    assert store.exists()

    reloaded_adapter = GeminiApiAdapter(api_key="test-key", enabled=True)
    assert reloaded_adapter.rate_limiter.get_stats()["rpm_used"] == 1
    assert reloaded_adapter.rate_limiter.get_stats()["tpm_used"] == 42


def test_base_adapter_persistence_disabled(monkeypatch, tmp_path):
    """With RATE_LIMITS_PERSIST=false adapters keep everything in memory."""
    import app.providers.base as base_module
    from app.providers.gemini_api import GeminiApiAdapter

    monkeypatch.setattr(base_module, "RATE_LIMITS_PERSIST", False)
    monkeypatch.setattr(base_module, "RATE_LIMITS_FILE", tmp_path / "unused.json")

    adapter = GeminiApiAdapter(api_key="test-key", enabled=True)
    adapter.record_usage(tokens=10)

    assert adapter.rate_limiter._persist_path is None
    assert not (tmp_path / "unused.json").exists()


@pytest.mark.parametrize(
    "bad_entry",
    [
        {"minute_requests": ["oops", 1.5]},
        {"minute_tokens": [["not-pair"], [1000.0, "many", 3]]},
    ],
)
def test_malformed_entries_are_skipped(tmp_path, bad_entry):
    """Invalid persisted records are ignored instead of crashing the load."""
    store = tmp_path / "rate_limits.json"
    store.write_text(json.dumps({"version": 1, "backends": {"gemini_api": bad_entry}}))

    tracker = InMemoryRateTracker(persist_path=store, persist_key="gemini_api")
    # Only the structurally valid entry (if any) survives; nothing raises.
    stats = tracker.get_stats()
    assert isinstance(stats["rpm_used"], int)
