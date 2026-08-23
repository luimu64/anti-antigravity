import os
import tempfile

# Isolate rate-limit counter persistence: keep test runs from reading or
# writing the real data/rate_limits.json store. Must be set before any
# `app.*` import because config values are resolved at import time.
os.environ.setdefault("RATE_LIMITS_PERSIST", "false")
os.environ.setdefault(
    "RATE_LIMITS_FILE",
    os.path.join(tempfile.gettempdir(), "google_gate_test_rate_limits.json"),
)

import pytest

from app.auth import auth_manager
from app.keys import api_key_manager


@pytest.fixture(autouse=True)
def restore_global_state():
    """
    Autouse fixture to snapshot and restore global singleton states
    (api_key_manager enforcement, auth_manager tokens) across test cases.
    """
    orig_enforce = api_key_manager.enforce_keys
    orig_access = auth_manager.access_token
    orig_refresh = auth_manager.refresh_token
    orig_project = auth_manager.project_id
    orig_tier = auth_manager.tier_name
    orig_email = auth_manager.user_email

    yield

    api_key_manager.enforce_keys = orig_enforce
    auth_manager.access_token = orig_access
    auth_manager.refresh_token = orig_refresh
    auth_manager.project_id = orig_project
    auth_manager.tier_name = orig_tier
    auth_manager.user_email = orig_email
