"""Asking a vendor what it serves, against a mock transport rather than a vendor.

A routing file names identifiers somebody else owns and renames. Four of eleven routes were
wrong on 2026-09-12, and each one cost a paid call to discover. This is the cheaper check, so
the shapes of the three listing endpoints are pinned here.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from mselect.runner import catalogue

ANTHROPIC = {
    "kind": "anthropic",
    "base_url": "https://api.anthropic.test",
    "api_key_env": "TEST_ANTHROPIC_KEY",
    "api_version": "2023-06-01",
}
OPENAI = {
    "kind": "openai_compat",
    "base_url": "https://api.openai.test/v1",
    "api_key_env": "TEST_OPENAI_KEY",
}
GOOGLE = {
    "kind": "google",
    "base_url": "https://google.test",
    "api_key_env": "TEST_GOOGLE_KEY",
    "api_version": "v1beta",
}
LOCAL = {"kind": "openai_compat", "base_url": "http://127.0.0.1:11434/v1", "price_zero": True}


def _transport(routes: dict[str, Any]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        for path, payload in routes.items():
            if request.url.path == path:
                if isinstance(payload, int):
                    return httpx.Response(payload, text="nope")
                return httpx.Response(200, json=payload)
        return httpx.Response(404, text=f"no route for {request.url.path}")

    return httpx.MockTransport(handler)


def _client(routes: dict[str, Any]) -> httpx.Client:
    return httpx.Client(transport=_transport(routes))


def test_anthropic_ids_come_from_data(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_ANTHROPIC_KEY", "k")
    payload = {"data": [{"id": "claude-haiku-4-5-20251001"}, {"id": "claude-sonnet-4-5"}]}
    with _client({"/v1/models": payload}) as client:
        got = catalogue.fetch("anthropic", ANTHROPIC, client=client)
    assert got.ok
    assert got.models == ("claude-haiku-4-5-20251001", "claude-sonnet-4-5")


def test_google_ids_lose_the_models_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    """Google names a model `models/gemini-3.8-flash`; the routing file names the tail."""
    monkeypatch.setenv("TEST_GOOGLE_KEY", "k")
    payload = {"models": [{"name": "models/gemini-3.8-flash"}, {"name": "models/gemini-3.5-pro"}]}
    with _client({"/v1beta/models": payload}) as client:
        got = catalogue.fetch("google", GOOGLE, client=client)
    assert got.models == ("gemini-3.5-pro", "gemini-3.8-flash")


def test_an_openai_compatible_id_keeps_its_slash(monkeypatch: pytest.MonkeyPatch) -> None:
    """Together names a model `openai/gpt-oss-120b`, and the slash is part of the id."""
    monkeypatch.setenv("TEST_OPENAI_KEY", "k")
    payload = {"data": [{"id": "openai/gpt-oss-120b"}, {"id": "meta-llama/Llama-3.3-70B"}]}
    with _client({"/v1/models": payload}) as client:
        got = catalogue.fetch("openai", OPENAI, client=client)
    assert got.models == ("meta-llama/Llama-3.3-70B", "openai/gpt-oss-120b")


def test_a_missing_key_is_a_result_not_a_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TEST_ANTHROPIC_KEY", raising=False)
    got = catalogue.fetch("anthropic", ANTHROPIC)
    assert not got.ok and got.error is not None and "TEST_ANTHROPIC_KEY" in got.error


def test_a_refusal_is_a_result_not_a_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_ANTHROPIC_KEY", "k")
    with _client({"/v1/models": 401}) as client:
        got = catalogue.fetch("anthropic", ANTHROPIC, client=client)
    assert not got.ok and got.error is not None and "401" in got.error


def test_a_local_server_is_asked_the_same_way(monkeypatch: pytest.MonkeyPatch) -> None:
    """A route naming a model nobody pulled is the same mistake as one naming a retired model."""
    payload = {"data": [{"id": "llama3.2:3b"}]}
    with _client({"/v1/models": payload}) as client:
        got = catalogue.fetch("local", LOCAL, client=client)
    assert got.models == ("llama3.2:3b",)


def test_a_route_the_vendor_does_not_list_is_flagged_with_the_family() -> None:
    providers = {"anthropic": ANTHROPIC}
    routes = {
        "good": {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"},
        "bad": {"provider": "anthropic", "model": "claude-sonnet-5"},
    }
    catalogues = {
        "anthropic": catalogue.Catalogue(
            "anthropic",
            models=("claude-haiku-4-5-20251001", "claude-sonnet-4-5", "claude-sonnet-4-5-2026"),
        )
    }
    checks, _ = catalogue.check_routes(providers, routes, ["good", "bad"], catalogues=catalogues)
    by_alias = {c.alias: c for c in checks}
    assert by_alias["good"].listed and by_alias["good"].nearest == ()
    assert not by_alias["bad"].listed
    assert by_alias["bad"].known, "the provider answered; the model is simply absent"
    # Ranked, not filtered. Every Anthropic id starts "claude", so the test that matters is
    # that the family a reader asked for comes first rather than that nothing else appears.
    assert by_alias["bad"].nearest[:2] == ("claude-sonnet-4-5", "claude-sonnet-4-5-2026")


def test_a_provider_that_did_not_answer_is_unknown_rather_than_missing() -> None:
    """Never report a route as wrong because the listing call failed. Those are different."""
    providers = {"anthropic": ANTHROPIC}
    routes = {"a": {"provider": "anthropic", "model": "anything"}}
    catalogues = {"anthropic": catalogue.Catalogue("anthropic", error="TEST_KEY is not set")}
    checks, _ = catalogue.check_routes(providers, routes, ["a"], catalogues=catalogues)
    assert not checks[0].listed
    assert not checks[0].known
    assert checks[0].nearest == ()


def test_nearest_ranks_the_family_above_the_brand() -> None:
    """Anything keyed on the first word of a Claude id matches every Claude id."""
    listed = ("claude-haiku-4-5-20251001", "claude-opus-4-1", "claude-sonnet-4-5")
    assert catalogue._nearest("claude-sonnet-5", listed)[0] == "claude-sonnet-4-5"
    assert catalogue._nearest("claude-opus-5", listed)[0] == "claude-opus-4-1"


def test_nearest_is_empty_when_nothing_shares_a_word() -> None:
    assert catalogue._nearest("claude-sonnet-5", ("gemini-3.8-flash", "o3")) == ()
