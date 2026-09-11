"""Verify the bounded short-cooldown wait in router.check_availability."""

import asyncio
import sys
import time
from unittest.mock import patch

sys.path.insert(0, "/opt/data/user-files/anti-antigravity")

from app.providers.base import RateLimitError  # noqa: E402
from app.providers.router import MultiBackendRouter  # noqa: E402


class FakeAgy:
    name = "antigravity"
    enabled = True
    _cached_models = {"models": {"gemini-3.8-flash-medium": {}}}

    def is_configured(self):
        return True

    def get_cooldown_remaining(self):
        return 1.2


class LongCooldownAgy(FakeAgy):
    def get_cooldown_remaining(self):
        return 60.0


def _make_router(fake):
    router = MultiBackendRouter.__new__(MultiBackendRouter)
    router.antigravity = fake
    router.gemini_api = fake
    router.gemini_web = fake
    router.aistudio_web = fake
    router.adapters = {"antigravity": fake}
    router.routing_strategy = "free_first"
    return router


async def run_short():
    fake = FakeAgy()
    router = _make_router(fake)
    t0 = time.perf_counter()
    with (
        patch.object(router, "_snapshot_backends", lambda **kw: {}),
        patch.object(router, "get_capable_adapters", lambda **kw: [fake]),
    ):
        order = {"n": 0}

        def ordered(**kw):
            order["n"] += 1
            return [] if order["n"] == 1 else [fake]

        with patch.object(router, "get_ordered_adapters", ordered):
            cands = await router.check_availability(model="gemini-3.8-flash-medium")
    return time.perf_counter() - t0, cands


dt, cands = asyncio.run(run_short())
# First evaluation empty (cooldown) -> bounded wait -> second returns backend.
assert dt >= 1.1, f"no wait happened: {dt:.2f}s"
assert dt < 3.0, f"wait unbounded: {dt:.2f}s"
assert cands and cands[0].name == "antigravity", cands
print(f"OK bounded short-cooldown wait: waited {dt:.2f}s, then served")


async def run_long():
    fake = LongCooldownAgy()
    router = _make_router(fake)
    t0 = time.perf_counter()
    with (
        patch.object(router, "_snapshot_backends", lambda **kw: {}),
        patch.object(router, "get_capable_adapters", lambda **kw: [fake]),
        patch.object(router, "get_ordered_adapters", lambda **kw: []),
    ):
        try:
            await router.check_availability(model="gemini-3.8-flash-medium")
        except RateLimitError as e:
            return time.perf_counter() - t0, e.retry_after
    raise AssertionError("expected RateLimitError")


dt2, retry_after = asyncio.run(run_long())
assert dt2 < 0.5, f"long cooldown should fail fast, took {dt2:.2f}s"
assert retry_after > 1.0, retry_after
print(f"OK long cooldown rejected immediately (retry_after={retry_after}s)")
print("ALL OK")
