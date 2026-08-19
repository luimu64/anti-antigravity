import os
import pytest
from app.keys import api_key_manager
from app.auth import auth_manager

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
