import pytest
import httpx
from main import app

@pytest.mark.asyncio
async def test_health_endpoint():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "agy-to-api"

@pytest.mark.asyncio
async def test_auth_status_endpoint():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/auth/status")
        assert response.status_code == 200
        data = response.json()
        assert "authenticated" in data
        assert "project_id" in data

from app.keys import api_key_manager

@pytest.mark.asyncio
async def test_models_endpoint():
    transport = httpx.ASGITransport(app=app)
    key = api_key_manager.get_first_active_key() or "test-key"
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/v1/models", headers={"Authorization": f"Bearer {key}"})
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) > 0
        model_ids = [m["id"] for m in data["data"]]
        assert "gemini-3.7-flash-high" in model_ids or "gpt-4o" in model_ids

@pytest.mark.asyncio
async def test_retrieve_model_endpoint():
    transport = httpx.ASGITransport(app=app)
    key = api_key_manager.get_first_active_key() or "test-key"
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/v1/models/gpt-4o", headers={"Authorization": f"Bearer {key}"})
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "gpt-4o"
        assert data["root"] == "gemini-3.7-flash-high"

@pytest.mark.asyncio
async def test_dashboard_endpoint():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/")
        assert response.status_code == 200
        assert "Antigravity API" in response.text
