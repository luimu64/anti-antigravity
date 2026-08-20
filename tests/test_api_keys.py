import pytest
import httpx
from main import app
from app.keys import APIKeyManager, api_key_manager


def test_api_key_manager_crud(tmp_path):
    storage = tmp_path / "test_keys.json"
    mgr = APIKeyManager(storage_file=storage)

    # 1. Create key
    created = mgr.create_key(name="Test Client")
    assert created["name"] == "Test Client"
    assert created["key"].startswith("sk-agy-")
    assert storage.exists()

    # 2. Validate key
    assert mgr.validate_key(created["key"]) is True
    assert mgr.validate_key("invalid_key_123") is False

    # 3. List keys
    keys = mgr.list_keys()
    assert len(keys) >= 1
    assert any(k["name"] == "Test Client" for k in keys)
    assert any("..." in k["key_preview"] for k in keys)

    # 4. Revoke key
    assert mgr.revoke_key(created["id"]) is True
    assert mgr.validate_key(created["key"]) is False


@pytest.mark.asyncio
async def test_api_key_enforcement_endpoints():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        # Create a new test key
        res = await ac.post("/api/keys", json={"name": "Automated Test Key"})
        assert res.status_code == 200
        key_data = res.json()["key"]
        raw_key = key_data["key"]
        key_id = key_data["id"]

        # Ensure enforcement is enabled
        await ac.post("/api/keys/enforcement", json={"enforce": True})

        # Request with no key -> 401
        res_unauth = await ac.get("/v1/models")
        assert res_unauth.status_code == 401
        err_json = res_unauth.json()
        assert "error" in err_json or "detail" in err_json

        # Request with invalid key -> 401
        res_bad = await ac.get(
            "/v1/models", headers={"Authorization": "Bearer sk-invalid-key"}
        )
        assert res_bad.status_code == 401

        # Request with valid key -> 200
        res_valid = await ac.get(
            "/v1/models", headers={"Authorization": f"Bearer {raw_key}"}
        )
        assert res_valid.status_code == 200

        # Disable enforcement -> 200 with no key
        await ac.post("/api/keys/enforcement", json={"enforce": False})
        res_no_enforce = await ac.get("/v1/models")
        assert res_no_enforce.status_code == 200

        # Re-enable enforcement
        await ac.post("/api/keys/enforcement", json={"enforce": True})

        # Cleanup key
        del_res = await ac.delete(f"/api/keys/{key_id}")
        assert del_res.status_code == 200
