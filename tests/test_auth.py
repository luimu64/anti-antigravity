import pytest
import time
from app.auth import OAuthManager


def test_oauth_manager_init(tmp_path):
    cred_file = tmp_path / "creds.json"
    mgr = OAuthManager(credentials_file=str(cred_file))
    assert mgr.credentials_file == str(cred_file)
    status = mgr.get_status()
    assert "authenticated" in status


def test_pkce_generation(tmp_path):
    cred_file = tmp_path / "creds.json"
    mgr = OAuthManager(credentials_file=str(cred_file))
    verifier, challenge = mgr.generate_pkce()
    assert len(verifier) > 40
    assert len(challenge) > 20


def test_authorization_url(tmp_path):
    cred_file = tmp_path / "creds.json"
    mgr = OAuthManager(credentials_file=str(cred_file))
    auth_url, state, verifier = mgr.get_authorization_url(
        redirect_uri="http://localhost:8000/auth/callback"
    )
    assert "accounts.google.com" in auth_url
    assert "client_id=" in auth_url
    assert "code_challenge=" in auth_url
    assert state in mgr._pkce_verifier_cache


def test_set_tokens_and_save(tmp_path):
    cred_file = tmp_path / "creds.json"
    mgr = OAuthManager(credentials_file=str(cred_file))
    mgr.set_tokens(
        access_token="test_access_token",
        refresh_token="test_refresh_token",
        project_id="test-project-123",
    )
    assert mgr.access_token == "test_access_token"
    assert mgr.refresh_token == "test_refresh_token"
    assert mgr.project_id == "test-project-123"
    assert cred_file.exists()

    # Create new manager instance and verify load
    mgr2 = OAuthManager(credentials_file=str(cred_file))
    assert mgr2.access_token == "test_access_token"
    assert mgr2.refresh_token == "test_refresh_token"
    assert mgr2.project_id == "test-project-123"
