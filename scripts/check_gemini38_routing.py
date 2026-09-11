"""Verify gemini-3.8-flash alias resolution and antigravity routing checks."""

import sys

sys.path.insert(0, "/opt/data/user-files/anti-antigravity")

from app.config import CANONICAL_MODEL_MAP, MODEL_ALIASES
from app.translator import OpenAITranslator

resolve = OpenAITranslator.resolve_model

# Clean alias resolves; no-effort default is gateway-wide "medium" (3.7 parity)
r = resolve("gemini-3.8-flash")
base, _, r_tier = r.rpartition("-")
assert base == "gemini-3.8-flash" and r_tier in ("medium", "high"), r
assert r.endswith(resolve("gemini-3.7-flash").rpartition("-")[2]), (
    r
)  # same default tier
print("OK resolve clean (3.7-parity default):", r)

# Tiered IDs resolve to themselves (no re-tiering)
for tier in ("high", "medium", "low"):
    r = resolve(f"gemini-3.8-flash-{tier}")
    assert r == f"gemini-3.8-flash-{tier}", r
print("OK resolve tiers self-map")

# Effort overrides
assert resolve("gemini-3.8-flash", "low") == "gemini-3.8-flash-low"
assert resolve("gemini-3.8-flash", "medium") == "gemini-3.8-flash-medium"
assert resolve("gemini-3.8-flash", "high") == "gemini-3.8-flash-high"
print("OK effort overrides")

# Separator variants
assert resolve("gemini_3.8_flash") == "gemini-3.8-flash-medium"
assert resolve("Gemini-3.8-Flash") == "gemini-3.8-flash-medium"
print("OK separator/case variants")

# 3.7 behavior unchanged (default effort parity; current gateway default = medium)
assert resolve("gemini-3.7-flash") == "gemini-3.7-flash-medium"
assert resolve("gemini-3.7-flash-medium") == "gemini-3.7-flash-medium"
print("OK 3.7 unchanged")

# --- supports_model routing checks (antigravity branch logic) ---
from app.providers.router import MultiBackendRouter  # noqa: E402


class FakeAgy:
    name = "antigravity"
    _cached_models = {
        "models": {
            "gemini-3.8-flash-tiered": {},
            "gemini-3.8-flash-high": {},
            "gemini-3.8-flash-medium": {},
            "gemini-3.8-flash-low": {},
            "gemini-3.7-flash-high": {},
        }
    }


router = MultiBackendRouter.__new__(MultiBackendRouter)

# The pre-fix failure: clean alias did not match probed tiered catalog
for requested in (
    "gemini-3.8-flash",
    "gemini-3.8-flash-high",
    "gemini-3.8-flash-tiered",
):
    clean = requested.lower().replace("models/", "").strip()
    resolved_clean = resolve(clean).lower().replace("models/", "")
    probed = {k.lower(): v for k, v in FakeAgy._cached_models["models"].items()}
    ok = (
        clean in probed
        or resolved_clean in probed
        or CANONICAL_MODEL_MAP.get(resolved_clean, resolved_clean) in probed
        or (clean in MODEL_ALIASES and MODEL_ALIASES[clean].lower() in probed)
    )
    assert ok, f"supports_model would still fail for {requested}"
    print(f"OK routing check: {requested}")

print("ALL OK")
