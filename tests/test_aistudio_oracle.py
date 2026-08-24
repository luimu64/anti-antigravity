"""Tests for the persistent-login camofox UI-oracle transport.

The REST boundary is faked with a stub _request; the browser itself is not
mocked here (E2E against a live AI Studio tab is a manual step).
"""

import json

import pytest

import app.providers.aistudio_oracle as oracle_mod
from app.providers.aistudio_oracle import (
    AistudioOracle,
    OracleError,
    OracleLoginRequired,
)


class FakeCamofox:
    """In-memory camofox REST server: tabs + evaluate expressions."""

    def __init__(self):
        self.tabs: dict[str, str] = {}  # tabId -> url
        self.next_id = 0
        # Per-tab JS expression -> result. First match wins.
        self.scripted: list[tuple[str, object]] = []
        self.fail_paths: set[str] = set()
        self.deleted: list[str] = []

    def evaluate(self, url: str, expr: str):
        for pattern, result in self.scripted:
            if pattern in expr:
                return result
        if 'document.querySelector("textarea")' in expr and "setter" in expr:
            return {"result": "sent"}
        return {"ok": True, "result": None}

    def request(self, path: str, method: str = "GET", payload: dict | None = None,
                timeout: float = 90.0) -> dict:
        if any(p in path for p in self.fail_paths):
            raise OracleError(f"camofox {method} {path} -> 500: boom")
        if path == "/tabs" and method == "POST":
            self.next_id += 1
            tid = f"tab-{self.next_id}"
            self.tabs[tid] = payload.get("url", "")
            return {"tabId": tid}
        if path.startswith("/tabs?"):
            return {"tabs": [
                {"tabId": t, "url": u} for t, u in self.tabs.items()
            ]}
        if "/navigate" in path and method == "POST":
            tid = path.split("/")[2]
            if tid not in self.tabs:
                raise OracleError("camofox POST navigate -> 404: no tab")
            self.tabs[tid] = payload["url"]
            return {"ok": True, "tabId": tid}
        if "/evaluate" in path and method == "POST":
            return self.evaluate(path, payload.get("expression", ""))
        if method == "DELETE":
            tid = path.split("/")[2].split("?")[0]
            self.deleted.append(tid)
            self.tabs.pop(tid, None)
            return {"ok": True}
        return {}

    def install(self, monkeypatch):
        monkeypatch.setattr(oracle_mod, "_request", self.request)


STATE_READY = (
    '(() => ({ready: !!document.querySelector("textarea"),'
    ' onNewChat: location.href.includes("new_chat"),'
    ' signedOut: location.host.includes("accounts.google")}))()'
)


def ready_state() -> str:
    return json.dumps(
        {"ready": True, "onNewChat": True, "signedOut": False}
    )


def signed_out_state() -> str:
    return json.dumps(
        {"ready": False, "onNewChat": False, "signedOut": True}
    )


@pytest.fixture()
def fx(monkeypatch):
    fake = FakeCamofox()
    fake.install(monkeypatch)
    oracle = AistudioOracle()
    return fake, oracle


def test_ensure_tab_reuses_existing_healthy_tab(fx):
    fake, oracle = fx
    fake.tabs["t1"] = "https://aistudio.google.com/prompts/new_chat"
    fake.scripted = [
        ('ready: !!document.querySelector("textarea"), ok:',
         {"result": json.dumps({"ready": True, "ok": True})}),
    ]
    oracle._ensure_tab()
    assert oracle._tab_id == "t1"
    assert fake.tabs["t1"].endswith("new_chat")


def test_ensure_tab_skips_signin_tabs_and_creates(fx):
    fake, oracle = fx
    fake.tabs["corpse"] = "https://accounts.google.com/v3/signin"
    oracle._ensure_tab()
    # corpse skipped; a fresh tab was created
    assert oracle._tab_id != "corpse"
    assert oracle._tab_id in fake.tabs


def test_stale_tab_is_deleted_then_recreated(fx):
    fake, oracle = fx
    oracle._tab_id = "dead"
    fake.tabs["dead"] = "https://aistudio.google.com/prompts/new_chat"
    # navigate to the dead tab fails
    fake.fail_paths.add("/tabs/dead/navigate")
    oracle._ensure_tab()
    assert "dead" in fake.deleted
    assert oracle._tab_id != "dead"


def test_wait_ready_passes_on_signed_in_tab(fx):
    fake, oracle = fx
    oracle._tab_id = "t1"
    fake.scripted = [(STATE_READY[:40], {"result": ready_state()})]
    oracle._wait_ready()  # no raise


def test_wait_ready_raises_login_required_after_persistent_signout(fx, monkeypatch):
    fake, oracle = fx
    oracle._tab_id = "t1"
    fake.scripted = [(STATE_READY[:40], {"result": signed_out_state()})]
    monkeypatch.setattr(oracle_mod.time, "sleep", lambda s: None)
    with pytest.raises(OracleLoginRequired, match="camofox.luimu.dev"):
        oracle._wait_ready()


def test_login_status_reports_signout(fx, monkeypatch):
    fake, oracle = fx
    fake.scripted = [(STATE_READY[:40], {"result": signed_out_state()})]
    monkeypatch.setattr(oracle_mod.time, "sleep", lambda s: None)
    status = oracle.login_status()
    assert status["signedIn"] is False
    assert "camofox.luimu.dev" in status.get("hint", "")


def test_send_prompt_sets_textarea_and_clicks_run(fx):
    fake, oracle = fx
    oracle._tab_id = "t1"
    captured = {}

    def eval_hook(url, expr):
        captured["expr"] = expr
        return {"result": "sent"}

    fake.evaluate = eval_hook
    oracle._send_prompt("hello world")
    assert 'setter.set.call(ta, "hello world")' in captured["expr"]
    assert 'aria-label*="Run"' in captured["expr"]


def test_generate_once_extracts_reply_between_markers(fx, monkeypatch):
    fake, oracle = fx
    # Tab lifecycle: healthy reuse probe
    fake.tabs["t1"] = "https://aistudio.google.com/prompts/new_chat"
    fake.scripted = [
        ('ready: !!document.querySelector("textarea"), ok:',
         {"result": json.dumps({"ready": True, "ok": True})}),
        (STATE_READY[:40], {"result": ready_state()}),
        ("ms-chat-turn", {
            "result": json.dumps({
                "text": "Model 12:00\nthe reply text\nthumb_up thumb_down",
            })
        }),
    ]

    polls = {"n": 0}

    def fake_sleep(s):
        polls["n"] += 1

    monkeypatch.setattr(oracle_mod.time, "sleep", fake_sleep)
    reply = oracle.generate_once("hi")
    assert reply == "the reply text"


def test_generate_once_surfaces_error_text(fx, monkeypatch):
    fake, oracle = fx
    fake.tabs["t1"] = "https://aistudio.google.com/prompts/new_chat"
    fake.scripted = [
        ('ready: !!document.querySelector("textarea"), ok:',
         {"result": json.dumps({"ready": True, "ok": True})}),
        (STATE_READY[:40], {"result": ready_state()}),
        ("ms-chat-turn", {
            "result": json.dumps({
                "text": "Model 12:00\nAn internal error has occurred. Retry\nthumb_up",
            })
        }),
    ]
    monkeypatch.setattr(oracle_mod.time, "sleep", lambda s: None)
    with pytest.raises(OracleError, match="internal error"):
        oracle.generate_once("hi")
