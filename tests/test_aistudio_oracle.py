"""Tests for the Playwright persistent-profile UI-oracle transport.

Playwright itself is faked: a FakePage/FakeContext records evaluate calls and
returns scripted results. Browser E2E against live AI Studio is a manual step
(scripts/aistudio_login.py + a gateway request).
"""

import pytest

import app.providers.aistudio_oracle as oracle_mod
from app.providers.aistudio_oracle import (
    AistudioOracle,
    OracleLoginRequired,
)


class FakePage:
    def __init__(self, url: str = "https://aistudio.google.com/prompts/new_chat"):
        self.url = url
        self.dead = False
        # Ordered list of (substring, result) matched against JS source.
        self.scripted: list[tuple[str, object]] = []
        self.goto_calls: list[str] = []
        self.prompts_sent: list[str] = []

    def goto(self, url, wait_until=None):
        self.goto_calls.append(url)
        self.url = url

    def evaluate(self, js, *args):
        if self.dead:
            raise RuntimeError("page closed")
        for pattern, result in self.scripted:
            if pattern in js:
                if callable(result):
                    return result(*args)
                return result
        return None


READY_JS = 'onNewChat: location.href.includes("new_chat")'
TURN_JS = "ms-chat-turn"


def ready_state():
    return {"ready": True, "onNewChat": True, "signedOut": False}


def signed_out_state():
    return {"ready": False, "onNewChat": False, "signedOut": True}


@pytest.fixture()
def oracle(monkeypatch):
    o = AistudioOracle()
    fake_page = FakePage()
    fake_ctx = type("Ctx", (), {"pages": [fake_page]})()
    monkeypatch.setattr(o._browser, "page", lambda: fake_page, raising=False)
    monkeypatch.setattr(o._browser, "_context", fake_ctx)
    return o, fake_page


def test_is_available_with_profile_dir():
    assert AistudioOracle().is_available() is True


def test_wait_ready_passes_when_signed_in(oracle):
    o, page = oracle
    page.scripted = [(READY_JS[:20], ready_state())]
    o._wait_ready(page)  # no raise


def test_wait_ready_raises_login_required_on_persistent_signout(oracle, monkeypatch):
    o, page = oracle
    page.scripted = [(READY_JS[:20], signed_out_state())]
    monkeypatch.setattr(oracle_mod.time, "sleep", lambda s: None)
    with pytest.raises(OracleLoginRequired, match="aistudio_login"):
        o._wait_ready(page)


def test_send_prompt_sets_textarea_and_clicks_run(oracle):
    o, page = oracle

    def eval_hook(js, args=None):
        page.prompts_sent.append(args)
        return True

    page.evaluate = eval_hook
    o._send_prompt(page, "hello world")
    assert page.prompts_sent == ["hello world"]


def test_extract_reply_none_while_pending(oracle):
    o, page = oracle
    page.scripted = [(TURN_JS, None)]
    assert o._extract_reply(page) == (None, None)


def test_extract_reply_between_markers(oracle):
    o, page = oracle
    page.scripted = [
        (
            TURN_JS,
            "Model 12:00\nthe reply text\nthumb_up thumb_down",
        )
    ]
    reply, err = o._extract_reply(page)
    assert reply == "the reply text"
    assert err is None


def test_extract_reply_surfaces_internal_error(oracle):
    o, page = oracle
    page.scripted = [
        (
            TURN_JS,
            "Model 12:00\nAn internal error has occurred. Retry.\nthumb_up",
        )
    ]
    reply, err = o._extract_reply(page)
    assert reply is None
    assert "internal error" in err


def test_generate_once_happy_path(oracle, monkeypatch):
    o, page = oracle
    page.scripted = [
        (READY_JS[:20], ready_state()),
        ("return true", True),  # send_prompt returns True
        (TURN_JS, "Model 12:00\nfinal answer\nthumb_up"),
    ]
    monkeypatch.setattr(oracle_mod.time, "sleep", lambda s: None)
    assert o.generate_once("hi") == "final answer"


def test_generate_once_rebuilds_dead_renderer(oracle, monkeypatch):
    o, page = oracle
    rebuilt = FakePage()

    calls = {"n": 0}

    def flaky_wait_ready(p):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("Target closed")

    monkeypatch.setattr(o, "_wait_ready", flaky_wait_ready)
    monkeypatch.setattr(o._browser, "_ensure_page", lambda: rebuilt, raising=False)
    monkeypatch.setattr(oracle_mod.time, "sleep", lambda s: None)

    sent = {}

    def fake_send(p, prompt):
        sent["prompt"] = prompt

    monkeypatch.setattr(o, "_send_prompt", fake_send)

    def extract(p):
        return "ok", None

    monkeypatch.setattr(o, "_extract_reply", extract)
    assert o.generate_once("hi") == "ok"


def test_login_status_reports_signout(oracle, monkeypatch):
    o, page = oracle
    page.scripted = [(READY_JS[:20], signed_out_state())]
    monkeypatch.setattr(oracle_mod.time, "sleep", lambda s: None)
    status = o.login_status()
    assert status["signedIn"] is False
    assert "aistudio_login" in status.get("hint", "")


def test_adapter_maps_login_required_to_clear_error(monkeypatch):
    """The adapter surfaces OracleLoginRequired as an actionable ValueError."""
    import asyncio

    from app.providers.aistudio_web import AIStudioWebAdapter

    adapter = AIStudioWebAdapter(cookies="SAPISID=x;", enabled=True)
    adapter.transport_mode = "oracle"

    class FailingOracle:
        name = "aistudio_web"

        async def generate(self, prompt):
            raise OracleLoginRequired("sign in once")

    adapter.oracle = FailingOracle()

    async def run():
        out = []
        with pytest.raises(ValueError, match="one-time manual sign-in"):
            async for chunk in adapter.stream_generate_content(
                model="gemini-2.0-flash",
                contents=[{"role": "user", "parts": [{"text": "hi"}]}],
            ):
                out.append(chunk)
        return out

    asyncio.get_event_loop_policy().new_event_loop().run_until_complete(run())
