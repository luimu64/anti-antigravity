from unittest.mock import AsyncMock, patch

import httpx
import pytest

from app.auth import auth_manager
from app.client import client
from main import app


@pytest.mark.asyncio
async def test_auth_login_localhost_redirect():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://localhost:8000"
    ) as ac:
        # Default localhost with redirect=True (default) -> RedirectResponse
        resp = await ac.get("/auth/login", follow_redirects=False)
        assert resp.status_code in (302, 307)
        assert "accounts.google.com" in resp.headers["location"]


@pytest.mark.asyncio
async def test_auth_login_remote_html():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://192.168.1.50:8000"
    ) as ac:
        # Remote IP or redirect=False -> HTML page with manual setup
        resp = await ac.get("/auth/login?redirect=false")
        assert resp.status_code == 200
        assert "Connect Google Account" in resp.text
        assert "Authorize Google Account" in resp.text


@pytest.mark.asyncio
async def test_auth_callback_success():
    transport = httpx.ASGITransport(app=app)
    with (
        patch.object(
            auth_manager,
            "exchange_code",
            new_callable=AsyncMock,
            return_value={"access_token": "ya29.test", "refresh_token": "1//test"},
        ),
        patch.object(
            client,
            "load_code_assist",
            new_callable=AsyncMock,
            return_value={"cloudaicompanionProject": "test-proj"},
        ),
    ):
        async with httpx.AsyncClient(
            transport=transport, base_url="http://localhost:8000"
        ) as ac:
            resp = await ac.get(
                "/auth/callback?code=test_code_123&state=test_state_456"
            )
            assert resp.status_code == 200
            assert "Authentication Successful!" in resp.text


@pytest.mark.asyncio
async def test_auth_callback_with_oauth_error():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://localhost:8000"
    ) as ac:
        resp = await ac.get("/auth/callback?error=access_denied")
        assert resp.status_code == 400
        assert "Authentication Failed" in resp.text
        assert "access_denied" in resp.text


@pytest.mark.asyncio
async def test_auth_callback_missing_code():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://localhost:8000"
    ) as ac:
        resp = await ac.get("/auth/callback")
        assert resp.status_code == 400


@pytest.mark.asyncio
async def test_auth_callback_exchange_exception():
    transport = httpx.ASGITransport(app=app)
    with patch.object(
        auth_manager,
        "exchange_code",
        new_callable=AsyncMock,
        side_effect=ValueError("Token exchange network error"),
    ):
        async with httpx.AsyncClient(
            transport=transport, base_url="http://localhost:8000"
        ) as ac:
            resp = await ac.get("/auth/callback?code=bad_code")
            assert resp.status_code == 500
            assert "Authentication Error" in resp.text


@pytest.mark.asyncio
async def test_auth_exchange_direct_refresh_or_access_token():
    transport = httpx.ASGITransport(app=app)
    with patch.object(
        client, "load_code_assist", new_callable=AsyncMock, return_value={}
    ):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            # 1. Refresh token starting with 1//
            resp_ref = await ac.post(
                "/auth/exchange", json={"code_or_url": "1//test_refresh_token_abc"}
            )
            assert resp_ref.status_code == 200
            assert resp_ref.json()["status"] == "ok"
            assert auth_manager.refresh_token == "1//test_refresh_token_abc"

            # 2. Access token starting with ya29.
            resp_acc = await ac.post(
                "/auth/exchange", json={"code_or_url": "ya29.test_access_token_xyz"}
            )
            assert resp_acc.status_code == 200
            assert resp_acc.json()["status"] == "ok"
            assert auth_manager.access_token == "ya29.test_access_token_xyz"


@pytest.mark.asyncio
async def test_auth_exchange_full_url_and_code():
    transport = httpx.ASGITransport(app=app)
    with (
        patch.object(
            auth_manager,
            "exchange_code",
            new_callable=AsyncMock,
            return_value={"access_token": "ya29.exchanged"},
        ),
        patch.object(
            client, "load_code_assist", new_callable=AsyncMock, return_value={}
        ),
    ):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            # Full URL
            full_url = "http://localhost:8085/?state=teststate123&code=4/0AeanS0..."
            resp = await ac.post("/auth/exchange", json={"code_or_url": full_url})
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"

            # Plain code
            resp2 = await ac.post(
                "/auth/exchange",
                json={"code_or_url": "4/0AeanS0...", "state": "teststate123"},
            )
            assert resp2.status_code == 200
            assert resp2.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_auth_exchange_failure():
    transport = httpx.ASGITransport(app=app)
    with patch.object(
        auth_manager,
        "exchange_code",
        new_callable=AsyncMock,
        side_effect=ValueError("Invalid code"),
    ):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/auth/exchange", json={"code_or_url": "invalid_code_here"}
            )
            assert resp.status_code == 400
            assert (
                "Authorization code exchange failed" in resp.json()["error"]["message"]
            )


@pytest.mark.asyncio
async def test_auth_token_set_and_refresh():
    transport = httpx.ASGITransport(app=app)
    with patch.object(
        client, "load_code_assist", new_callable=AsyncMock, return_value={}
    ):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            # Set tokens via /auth/token
            resp = await ac.post(
                "/auth/token",
                json={
                    "access_token": "ya29.manual_set",
                    "refresh_token": "1//manual_refresh",
                    "project_id": "manual-project-789",
                },
            )
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"
            assert auth_manager.access_token == "ya29.manual_set"
            assert auth_manager.refresh_token == "1//manual_refresh"
            assert auth_manager.project_id == "manual-project-789"

    # Token refresh via /auth/refresh
    with (
        patch.object(
            auth_manager,
            "refresh_access_token",
            new_callable=AsyncMock,
            return_value="ya29.new_refreshed_access_token",
        ),
        patch.object(
            client, "load_code_assist", new_callable=AsyncMock, return_value={}
        ),
    ):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp_ref = await ac.post("/auth/refresh")
            assert resp_ref.status_code == 200
            assert resp_ref.json()["status"] == "ok"
            assert "access_token" in resp_ref.json()

    # Token refresh failure
    with patch.object(
        auth_manager,
        "refresh_access_token",
        new_callable=AsyncMock,
        side_effect=ValueError("Token revoked"),
    ):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp_fail = await ac.post("/auth/refresh")
            assert resp_fail.status_code == 400
            assert "Token refresh failed" in resp_fail.json()["error"]["message"]


@pytest.mark.asyncio
async def test_auth_logout():
    transport = httpx.ASGITransport(app=app)
    auth_manager.access_token = "ya29.to_be_cleared"
    auth_manager.refresh_token = "1//to_be_cleared"

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post("/auth/logout")
        assert resp.status_code == 200
        assert resp.json()["status"] == "logged_out"
        assert auth_manager.access_token is None
        assert auth_manager.refresh_token is None
