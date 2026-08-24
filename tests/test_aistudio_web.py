"""Tests for the AI Studio Web (MakerSuiteService RPC) backend adapter."""

import base64
import hashlib
import json
import re
import time
from unittest.mock import MagicMock

import httpx
import pytest

from app.providers.aistudio_web import (
    ORIGIN,
    SAFETY_CONFIG,
    AIStudioWebAdapter,
    _find_finish_reason,
    _iter_text_parts,
)
from app.providers.base import RateLimitError
from app.providers.router import MultiBackendRouter


def make_adapter(**kwargs) -> AIStudioWebAdapter:
    defaults = {
        "cookies": "SID=abc; SAPISID=test_sapisid_123; __Secure-1PAPISID=test_sapisid_123;",
        "session": "test_session_blob",
    }
    defaults.update(kwargs)
    return AIStudioWebAdapter(**defaults)


def mock_stream_adapter(respond) -> AIStudioWebAdapter:
    """Adapter whose HTTP client is backed by an httpx.MockTransport handler."""
    adapter = make_adapter(enabled=True)
    adapter._http_client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    return adapter


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------


def test_extract_sapisid():
    cookies = "NID=123; SID=xyz; SAPISID=AbCdEf/1234; __Secure-3PAPISID=other;"
    assert AIStudioWebAdapter.extract_sapisid(cookies) == "AbCdEf/1234"
    assert AIStudioWebAdapter.extract_sapisid("SID=only") == ""
    assert AIStudioWebAdapter.extract_sapisid("") == ""


def test_sapisidhash_deterministic(monkeypatch):
    monkeypatch.setattr(time, "time", lambda: 1787566395.0)
    core = AIStudioWebAdapter._sapisidhash("BVsbRnx0")
    expected = hashlib.sha1(f"1787566395 BVsbRnx0 {ORIGIN}".encode()).hexdigest()
    assert core == f"1787566395_{expected}"


def test_authorization_header_contains_all_hash_variants(monkeypatch):
    monkeypatch.setattr(time, "time", lambda: 1000.0)
    adapter = make_adapter()
    auth = adapter._authorization_header()
    digest = hashlib.sha1(f"1000 test_sapisid_123 {ORIGIN}".encode()).hexdigest()
    assert auth == (
        f"SAPISIDHASH 1000_{digest} "
        f"SAPISID1PHASH 1000_{digest} "
        f"SAPISID3PHASH 1000_{digest}"
    )


def test_configuration_checks():
    adapter = AIStudioWebAdapter(cookies="")
    assert adapter.enabled is False
    assert adapter.is_configured() is False

    # Cookie without SAPISID cannot produce authorization -> unconfigured
    adapter = AIStudioWebAdapter(cookies="SID=abc; HSID=def;")
    assert adapter.is_configured() is False

    adapter = make_adapter()
    assert adapter.is_configured() is True

    # Default web client API key is preloaded
    assert adapter.api_key.startswith("AIzaSy")


def test_headers_shape():
    adapter = make_adapter()
    headers = adapter._get_headers()
    assert headers["Content-Type"] == "application/json+protobuf"
    assert headers["Origin"] == ORIGIN
    assert headers["X-Goog-Api-Key"].startswith("AIzaSy")
    assert headers["X-Goog-Authuser"] == "0"
    assert headers["Authorization"].startswith("SAPISIDHASH ")
    assert "SAPISID=test_sapisid_123" in headers["Cookie"]
    assert headers["X-AiStudio-Visit-Id"].startswith("v1_")


# ---------------------------------------------------------------------------
# Wire format builders
# ---------------------------------------------------------------------------


def test_iter_text_parts_and_finish_reason():
    tree = ["x", [[None, "hello"], [None, "", None, "sig"], [None, " world"]]]
    texts = list(_iter_text_parts(tree))
    assert texts == ["hello", "", " world"]
    assert _find_finish_reason(tree) is None
    assert _find_finish_reason([["STOP"]]) == "STOP"


@pytest.mark.asyncio
async def test_build_request_payload_layout():
    adapter = make_adapter(session="sess_blob", enabled=True)
    contents = [{"role": "user", "parts": [{"text": "Hi there"}]}]
    payload = await adapter.build_request_payload(
        model="gemini-3.7-flash",
        contents=contents,
        system_instruction={"parts": [{"text": "Be terse."}]},
        generation_config={"maxOutputTokens": 8192, "temperature": 0.5},
    )

    assert payload[0] == "models/gemini-3.7-flash"
    # Contents slot: list of Content messages [parts, role]
    assert payload[1] == [[[[None, "Hi there"]], "user"]]
    # Tool/safety config slot matches live captures
    assert payload[2] == SAFETY_CONFIG
    # Generation config slot: maxOutputTokens at idx3, temperature idx4
    assert payload[3][3] == 8192
    assert payload[3][4] == 0.5
    # Session blob slot
    assert payload[4] == "sess_blob"
    # System instruction rides as a Content-shaped message at slot 5
    assert payload[5] == [[[None, "Be terse."]], "user"]
    # Trailing constants: unknown nulls, flag 1, visit id
    assert payload[10] == 1
    assert payload[11].startswith("v1_")


@pytest.mark.asyncio
async def test_convert_contents_roles_and_thought_signature():
    adapter = make_adapter()
    contents = [
        {"role": "assistant", "parts": [{"text": "Reply"}]},
        {"role": "user", "parts": [{"text": "Next"}]},
    ]
    # Assistant turn carrying a cached thoughtSignature (tool-calling turns)
    contents[0]["parts"][0]["thoughtSignature"] = "sig_blob=="

    wire, system_wire = await adapter._convert_contents(contents)
    assert system_wire is None
    model_turn, user_turn = wire
    assert model_turn[1] == "model"
    text_part, sig_part = model_turn[0]
    assert text_part == [None, "Reply"]
    # Signature carrier: empty text with blob at DataItem index 14
    assert len(sig_part) == 15
    assert sig_part[1] == ""
    assert sig_part[14] == "sig_blob=="
    assert user_turn == [[[None, "Next"]], "user"]


def test_generation_config_thinking_and_caps():
    adapter = make_adapter()

    # Thinking budget > 0 maps to extended mode config at slot 15
    cfg = adapter._build_generation_config(
        {
            "maxOutputTokens": 99999999,
            "thinkingConfig": {"includeThoughts": True, "thinkingBudget": 4096},
        }
    )
    assert cfg[3] == 65536  # capped to model limit
    assert cfg[15] == [1, None, None, 2]

    # Disabled thinking (budget 0) omits the thinking slot entirely
    cfg_off = adapter._build_generation_config(
        {"thinkingConfig": {"includeThoughts": True, "thinkingBudget": 0}}
    )
    assert cfg_off[15] is None


@pytest.mark.asyncio
async def test_models_catalog_static():
    adapter = make_adapter()
    res = await adapter.fetch_available_models()
    models = res["models"]
    assert "gemini-3.7-flash" in models
    assert all(not m["isEmbedding"] for m in models.values())


# ---------------------------------------------------------------------------
# Streaming / response parsing
# ---------------------------------------------------------------------------


def stream_response_body(chunks: list) -> bytes:
    """Unary GenerateContent response: outer[0] is the ordered chunk list."""
    return json.dumps([chunks]).encode()


@pytest.mark.asyncio
async def test_stream_generate_content_parses_chunks():
    body = stream_response_body(
        [
            [None, None, None, [[[None, "Hello"]], "model"]],
            [None, None, None, [[[None, " world"]], "model", None, None, [["STOP"]]]],
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "alkalimakersuite-pa.clients6.google.com"
        assert request.url.path.endswith("/GenerateContent")
        assert request.headers["authorization"].startswith("SAPISIDHASH")
        assert request.headers["x-goog-ext-519733851-bin"]
        sent = json.loads(request.content.decode())
        assert sent[0] == "models/gemini-2.0-flash"
        return httpx.Response(200, content=body)

    adapter = mock_stream_adapter(handler)
    chunks = []
    async for chunk in adapter.stream_generate_content(
        model="gemini-2.0-flash",
        contents=[{"role": "user", "parts": [{"text": "say hi"}]}],
    ):
        chunks.append(chunk)

    deltas = []
    usage = None
    finish = None
    for c in chunks:
        for cand in c.get("candidates", []):
            if cand.get("finishReason"):
                finish = cand["finishReason"]
            for part in cand.get("content", {}).get("parts", []):
                if part.get("text"):
                    deltas.append(part["text"])
        if c.get("usageMetadata"):
            usage = c["usageMetadata"]

    assert "".join(deltas) == "Hello world"
    assert finish == "STOP"
    assert usage["totalTokenCount"] >= usage["promptTokenCount"]


@pytest.mark.asyncio
async def test_stream_handles_cumulative_snapshots():
    body = stream_response_body(
        [
            [[[None, "Hel"]]],
            [[[None, "Hello wor"]]],
            [[[None, "Hello world"]], "model"],
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    adapter = mock_stream_adapter(handler)
    emitted = ""
    async for chunk in adapter.stream_generate_content(
        model="gemini-2.0-flash",
        contents=[{"role": "user", "parts": [{"text": "hi"}]}],
    ):
        for cand in chunk.get("candidates", []):
            for part in cand.get("content", {}).get("parts", []):
                if part.get("text"):
                    emitted += part["text"]

    assert emitted == "Hello world"


@pytest.mark.asyncio
async def test_rate_limit_error_from_error_object():
    err = {
        "error": {
            "code": 429,
            "message": "Resource exhausted",
            "status": "RESOURCE_EXHAUSTED",
        }
    }
    body = json.dumps(err).encode()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    adapter = mock_stream_adapter(handler)
    with pytest.raises(RateLimitError):
        async for _ in adapter.stream_generate_content(
            model="gemini-2.0-flash",
            contents=[{"role": "user", "parts": [{"text": "hi"}]}],
        ):
            pass


@pytest.mark.asyncio
async def test_http_401_raises_auth_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, content=b'{"error":{"message":"unauth"}}')

    adapter = mock_stream_adapter(handler)
    with pytest.raises(ValueError, match="authentication failed"):
        async for _ in adapter.stream_generate_content(
            model="gemini-2.0-flash",
            contents=[{"role": "user", "parts": [{"text": "hi"}]}],
        ):
            pass


@pytest.mark.asyncio
async def test_generate_content_merges_stream():
    body = stream_response_body(
        [
            [[[None, "One "]], "model"],
            [[[None, "two"]], "model", None, None, [["MAX_TOKENS"]]],
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    adapter = mock_stream_adapter(handler)
    result = await adapter.generate_content(
        model="gemini-2.0-flash",
        contents=[{"role": "user", "parts": [{"text": "count"}]}],
    )
    assert result["text"] == "One two"
    assert result["finishReason"] == "MAX_TOKENS"
    assert result["usageMetadata"]["totalTokenCount"] > 0


# ---------------------------------------------------------------------------
# Router integration
# ---------------------------------------------------------------------------


def test_router_registry_includes_aistudio_web():
    router = MultiBackendRouter(aistudio_web=make_adapter())
    assert "aistudio_web" in router.adapters
    status = router.get_status()
    backend = status["backends"]["aistudio_web"]
    assert backend["id"] == "aistudio_web"
    assert backend["configured"] is True
    assert backend["enabled"] is False  # disabled by default
    assert backend["masked_cookies"]


@pytest.mark.asyncio
async def test_free_first_prefers_aistudio_web_when_alone():
    mock_agy = MagicMock()
    mock_agy.name = "antigravity"
    mock_agy.enabled = False

    mock_api = MagicMock()
    mock_api.name = "gemini_api"
    mock_api.enabled = False

    mock_web = MagicMock()
    mock_web.name = "gemini_web"
    mock_web.enabled = False

    studio = make_adapter(enabled=True)
    studio.generate_content = None  # not used by availability check

    router = MultiBackendRouter(
        antigravity=mock_agy,
        gemini_api=mock_api,
        gemini_web=mock_web,
        aistudio_web=studio,
        routing_strategy="free_first",
    )
    candidates = router.check_availability(model="gemini-3.7-flash")
    assert [a.name for a in candidates] == ["aistudio_web"]


def test_supports_model_gate():
    adapter = make_adapter()
    router = MultiBackendRouter(aistudio_web=adapter)
    assert router.supports_model(adapter, model="gemini-3.7-flash") is True
    assert router.supports_model(adapter, model="gemini-3.1-pro") is True
    assert router.supports_model(adapter, model="models/gemini-2.0-flash") is True
    # Gateway-wide deprecated models are rejected before adapter checks
    assert router.supports_model(adapter, model="gemini-2.5-pro") is False
    # Non-Gemini models are out of scope for this backend
    assert router.supports_model(adapter, model="claude-sonnet-4-6") is False


def test_normalize_model_strips_tier_suffixes():
    # Antigravity tier variants map onto AI Studio base model names
    assert AIStudioWebAdapter.normalize_model("gemini-3.6-flash-medium") == (
        "gemini-3.6-flash"
    )
    assert AIStudioWebAdapter.normalize_model("gemini-3.7-flash-high") == (
        "gemini-3.7-flash"
    )
    assert AIStudioWebAdapter.normalize_model("gemini-3.1-pro-low") == "gemini-3.1-pro"
    assert AIStudioWebAdapter.normalize_model("Models/Gemini-2.0-Flash") == (
        "gemini-2.0-flash"
    )
    # Unknown gemini families degrade to the closest catalog entry
    assert AIStudioWebAdapter.normalize_model("gemini-9.9-ultra") == (
        "gemini-2.0-flash"
    )


# ---------------------------------------------------------------------------
# File attachment uploads (GetAppFolder -> token -> Drive multipart)
# ---------------------------------------------------------------------------


class UploadHarness:
    """MockTransport handler emulating the AI Studio upload RPC chain."""

    def __init__(self, fail_first_upload: bool = False):
        self.fail_first_upload = fail_first_upload
        self.calls: list[httpx.Request] = []
        self.upload_count = 0
        self.token_requests = 0
        self.folder_requests = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:

        url = str(request.url)
        self.calls.append(request)
        if url.endswith("/GetAppFolder"):
            self.folder_requests += 1
            return httpx.Response(200, content=b'["app_folder_123"]')
        if url.endswith("/GenerateAccessToken"):
            self.token_requests += 1
            return httpx.Response(
                200,
                content=json.dumps([f"ya29.token_v{self.token_requests}"]).encode(),
            )
        if "/upload/drive/v3/files" in url:
            self.upload_count += 1
            if self.fail_first_upload and self.upload_count == 1:
                return httpx.Response(401, content=b'{"error": "expired"}')
            body = request.content.decode("utf-8", errors="replace")
            assert "app_folder_123" in body  # parent folder metadata present
            assert "Content-Transfer-Encoding: base64" in body
            assert re.fullmatch(
                r"Bearer ya29\.token_v\d+", request.headers["authorization"]
            )
            return httpx.Response(
                200,
                content=json.dumps({"id": "DRIVEFILE123", "name": "x"}).encode(),
            )
        if url.endswith("/GenerateContent"):
            return httpx.Response(
                200, content=stream_response_body([[[[None, "ok"]], "model"]])
            )
        return httpx.Response(404)


@pytest.mark.asyncio
async def test_inline_data_part_uploads_and_references_file():
    harness = UploadHarness()
    adapter = make_adapter(enabled=True)
    adapter._http_client = httpx.AsyncClient(transport=httpx.MockTransport(harness))

    png_b64 = base64.b64encode(b"PNGDATA").decode()
    contents = [
        {
            "role": "user",
            "parts": [
                {"inlineData": {"mimeType": "image/png", "data": png_b64}},
                {"text": "what is this?"},
            ],
        }
    ]
    payload = await adapter.build_request_payload("gemini-2.0-flash", contents)

    user_turn = payload[1][0]
    file_part, text_part = user_turn[0]
    assert file_part == [None, None, None, None, None, ["DRIVEFILE123"]]
    assert text_part == [None, "what is this?"]
    assert harness.folder_requests == 1
    assert harness.token_requests == 1
    assert harness.upload_count == 1


@pytest.mark.asyncio
async def test_identical_attachment_uploaded_once():
    harness = UploadHarness()
    adapter = make_adapter(enabled=True)
    adapter._http_client = httpx.AsyncClient(transport=httpx.MockTransport(harness))

    data = base64.b64encode(b"SAMEDATA").decode()
    inline = {"inlineData": {"mimeType": "application/pdf", "data": data}}
    await adapter.build_request_payload(
        "gemini-2.0-flash",
        [{"role": "user", "parts": [inline]}],
    )
    await adapter.build_request_payload(
        "gemini-2.0-flash",
        [{"role": "user", "parts": [inline]}],
    )
    assert harness.upload_count == 1

    # Different bytes -> second upload
    other = base64.b64encode(b"OTHERDATA").decode()
    await adapter.build_request_payload(
        "gemini-2.0-flash",
        [
            {
                "role": "user",
                "parts": [
                    {"inlineData": {"mimeType": "application/pdf", "data": other}}
                ],
            }
        ],
    )
    assert harness.upload_count == 2


@pytest.mark.asyncio
async def test_drive_token_refresh_on_401():
    harness = UploadHarness(fail_first_upload=True)
    adapter = make_adapter(enabled=True)
    adapter._http_client = httpx.AsyncClient(transport=httpx.MockTransport(harness))

    data = base64.b64encode(b"TOKENDATA").decode()
    payload = await adapter.build_request_payload(
        "gemini-2.0-flash",
        [
            {
                "role": "user",
                "parts": [{"inlineData": {"mimeType": "text/plain", "data": data}}],
            }
        ],
    )
    assert payload[1][0][0][0] == [None, None, None, None, None, ["DRIVEFILE123"]]
    assert harness.upload_count == 2  # first attempt 401, retry succeeded
    assert harness.token_requests == 2  # forced refresh


@pytest.mark.asyncio
async def test_streaming_with_attachment_end_to_end():
    harness = UploadHarness()
    adapter = make_adapter(enabled=True)
    adapter._http_client = httpx.AsyncClient(transport=httpx.MockTransport(harness))

    data = base64.b64encode(b"E2EDATA").decode()
    chunks = []
    async for chunk in adapter.stream_generate_content(
        model="gemini-2.0-flash",
        contents=[
            {
                "role": "user",
                "parts": [
                    {"inlineData": {"mimeType": "text/markdown", "data": data}},
                    {"text": "summarize"},
                ],
            }
        ],
    ):
        chunks.append(chunk)
    text = "".join(
        p["text"]
        for c in chunks
        for cand in c.get("candidates", [])
        for p in cand.get("content", {}).get("parts", [])
        if p.get("text")
    )
    assert text == "ok"
    assert harness.upload_count == 1
