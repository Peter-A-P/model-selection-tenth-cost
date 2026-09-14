"""The one place this project talks to a vendor, tested without talking to one.

What matters here is not that a request is well formed, which the gateway's own tests cover,
but the three decisions this layer makes: batching is a price and never a blocker, a failed
call is a fact about the call rather than about the item, and a spend cap stops the run
instead of being written into thousands of rows.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
import respx
from boundary import BatchNotReady, SpendCapExceeded
from boundary.config import CacheConfig, CapsConfig, ProjectCap, load_config
from boundary.gateway import Gateway

from mselect.runner import gateway
from mselect.runner.administer import Item, Prompt, administer, build_prompts
from mselect.runner.gateway import CONFIG, PROJECT, BoundaryCaller, describe

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
BATCHES_URL = "https://api.anthropic.com/v1/messages/batches"
BATCH_ID = "msgbatch_02test"
LOCAL_URL = "http://127.0.0.1:11434/v1/chat/completions"

MC = Item(
    item_id="mmlu:1",
    benchmark="mmlu",
    kind="multiple_choice",
    question="What is the capital of France?",
    options=("Lyon", "Paris", "Nice", "Lille"),
    answer="Paris",
)


@pytest.fixture
def keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("ANTHROPIC", "OPENAI", "GOOGLE", "OPENWEIGHTS"):
        monkeypatch.setenv(f"{name}_API_KEY", f"test-{name.lower()}-key-000000000000")


def _gateway(tmp_path: Path, *, caps: CapsConfig | None = None) -> Gateway:
    """This project's real configuration, with the development cache off and a throwaway
    ledger, so one test cannot answer another from cache or from a recorded spend."""
    config = load_config(CONFIG).model_copy(
        update={"cache": CacheConfig(enabled=False, path=tmp_path / "cache")}
    )
    return Gateway(
        config,
        project=PROJECT,
        ledger_path=tmp_path / "ledger.sqlite",
        caps=caps,
        sleep=lambda _s: None,
    )


@pytest.fixture
def gw(tmp_path: Path, keys: None) -> Iterator[Gateway]:
    g = _gateway(tmp_path)
    yield g
    g.close()


def _anthropic_ok(text: str = "B") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "msg_01",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-4-5-20251001",
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 40, "output_tokens": 1},
        },
    )


def _local_ok(text: str = "B") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "model": "llama3.2:3b",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 30, "completion_tokens": 1},
        },
    )


def _batch_submitted() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": BATCH_ID,
            "type": "message_batch",
            "processing_status": "in_progress",
            "request_counts": {"processing": 1},
            "results_url": None,
        },
    )


def _batch_ended() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": BATCH_ID,
            "type": "message_batch",
            "processing_status": "ended",
            "request_counts": {"succeeded": 2},
            "results_url": f"{BATCHES_URL}/{BATCH_ID}/results",
        },
    )


def _batch_results(custom_ids: list[str]) -> httpx.Response:
    import json

    lines = [
        json.dumps(
            {
                "custom_id": cid,
                "result": {"type": "succeeded", "message": _anthropic_ok().json()},
            }
        )
        for cid in custom_ids
    ]
    return httpx.Response(200, content="\n".join(lines).encode("utf-8"))


def test_the_alias_goes_to_the_vendor_unresolved_and_the_routes_file_decides(
    gw: Gateway,
) -> None:
    """The panel is named by alias so nothing in the code goes stale when a vendor renames a
    model. The request carries the alias; the gateway resolves it."""
    resolved = describe(gw, ["anthropic-haiku", "local-small-a"])
    assert resolved[0]["explicit"] == "anthropic/claude-haiku-4-5-20251001"
    assert resolved[1]["explicit"] == "local/llama3.2:3b"
    assert resolved[1]["price_zero"] is True


def test_a_local_call_is_answered_scored_and_costs_nothing(gw: Gateway) -> None:
    caller = BoundaryCaller(gw, use_batches=False, run_id="t")
    with respx.mock(assert_all_called=True) as mock:
        mock.post(LOCAL_URL).mock(return_value=_local_ok("B"))
        records = administer([MC], "local-small-a", caller)

    assert len(records) == 1
    record = records[0]
    assert record.correct == 1 and record.parsed == "B"
    assert record.cost_usd == pytest.approx(0.0), "a local model is free, and costed as free"
    assert record.input_tokens == 30 and record.output_tokens == 1
    assert record.error is None and record.ledger_id


def test_a_failed_call_is_recorded_and_the_run_carries_on(gw: Gateway) -> None:
    """A timeout says nothing about an item, so it is a row with an error and not a zero."""
    caller = BoundaryCaller(gw, use_batches=False)
    with respx.mock(assert_all_called=True) as mock:
        mock.post(LOCAL_URL).mock(return_value=httpx.Response(500, json={"error": "boom"}))
        replies = caller.ask(build_prompts([MC], "local-small-a"))

    assert len(replies) == 1
    assert replies[0].text is None
    assert replies[0].error is not None and "ProviderError" in replies[0].error


def test_a_spend_cap_stops_the_run_rather_than_filling_it_with_rows(
    tmp_path: Path, keys: None
) -> None:
    """The one error that must not become a row: carrying on would spend the rest of the
    panel's budget recording that there is no budget."""
    caps = CapsConfig(
        version=1,
        portfolio_monthly_usd=1000.0,
        projects={PROJECT: ProjectCap(monthly_usd=0.0000001)},
    )
    gw = _gateway(tmp_path, caps=caps)
    try:
        caller = BoundaryCaller(gw, use_batches=False)
        with respx.mock(assert_all_called=False) as mock:
            route = mock.post(ANTHROPIC_URL).mock(return_value=_anthropic_ok())
            with pytest.raises(SpendCapExceeded):
                caller.ask(build_prompts([MC], "anthropic-haiku"))
        assert route.call_count == 0, "a refused call reaches no vendor"
    finally:
        gw.close()


def test_anthropic_goes_through_the_batch_endpoint(gw: Gateway) -> None:
    """Half the price, and latency does not matter anywhere in this project."""
    caller = BoundaryCaller(gw, use_batches=True, wait_s=0.0)
    prompts = build_prompts([MC, MC], "anthropic-haiku")
    submitted: list[str] = []

    # Not assert_all_called: the single-message route is registered precisely to show that
    # nothing reaches it, so it is the one route that must stay uncalled.
    with respx.mock(assert_all_called=False) as mock:
        import json as _json

        def on_submit(request: httpx.Request) -> httpx.Response:
            submitted.extend(r["custom_id"] for r in _json.loads(request.content)["requests"])
            return _batch_submitted()

        single = mock.post(ANTHROPIC_URL).mock(return_value=_anthropic_ok())
        submit = mock.post(BATCHES_URL).mock(side_effect=on_submit)
        status = mock.get(f"{BATCHES_URL}/{BATCH_ID}").mock(return_value=_batch_ended())
        results = mock.get(f"{BATCHES_URL}/{BATCH_ID}/results").mock(
            side_effect=lambda request: _batch_results(submitted)
        )
        replies = caller.ask(prompts)

    assert single.call_count == 0, "batched work must not also go one at a time"
    assert submit.call_count == 1 and status.called and results.called
    assert len(replies) == 2 and all(r.text == "B" for r in replies)
    # The batch rate is half, and it came from the returned usage rather than an estimate.
    entry = gw.prices.lookup("anthropic", "claude-haiku-4-5-20251001")
    assert entry is not None
    full = 40 / 1e6 * entry.input + 1 / 1e6 * entry.output
    assert replies[0].cost_usd == pytest.approx(full * 0.5)


def test_a_provider_without_batches_falls_back_and_is_asked_only_once(gw: Gateway) -> None:
    """Batching is a price, not a method. A provider that has no batch endpoint is answered
    one call at a time, and the panel is not held up by it."""
    caller = BoundaryCaller(gw, use_batches=True)
    prompts = build_prompts([MC, MC], "local-small-a")

    with respx.mock(assert_all_called=True) as mock:
        route = mock.post(LOCAL_URL).mock(return_value=_local_ok())
        replies = caller.ask(prompts)

    assert len(replies) == 2 and route.call_count == 2
    assert "local-small-a" in caller._no_batches, "the answer is remembered, not re-asked"

    with respx.mock(assert_all_called=True) as mock:
        again = mock.post(LOCAL_URL).mock(return_value=_local_ok())
        caller.ask(build_prompts([MC], "local-small-a"))
    assert again.call_count == 1


def test_replies_come_back_in_the_order_the_prompts_went_out(gw: Gateway) -> None:
    """administer matches replies to prompts by position, so an adapter that reordered them
    would score one item against another item's answer."""
    caller = BoundaryCaller(gw, use_batches=False)
    items = [
        Item(
            item_id=f"mmlu:{i}",
            benchmark="mmlu",
            kind="multiple_choice",
            question=f"Q{i}",
            options=("a", "b", "c", "d"),
            answer="b",
        )
        for i in range(3)
    ]
    prompts = build_prompts(items, "local-small-a")

    def answer(request: httpx.Request) -> httpx.Response:
        import json as _json

        sent = _json.loads(request.content)["messages"][-1]["content"]
        return _local_ok("B" if "Q1" in sent else "A")

    with respx.mock(assert_all_called=True) as mock:
        mock.post(LOCAL_URL).mock(side_effect=answer)
        replies = caller.ask(prompts)

    assert [r.text for r in replies] == ["A", "B", "A"]


def test_every_call_leaves_a_ledger_row(gw: Gateway) -> None:
    caller = BoundaryCaller(gw, use_batches=False, run_id="panel-1")
    with respx.mock(assert_all_called=True) as mock:
        mock.post(LOCAL_URL).mock(return_value=_local_ok())
        caller.ask(build_prompts([MC], "local-small-a"))

    rows: list[dict[str, Any]] = gw.ledger.rows()
    assert len(rows) == 1
    assert rows[0]["project"] == PROJECT and rows[0]["run_id"] == "panel-1"
    assert rows[0]["purpose"] == "own-run" and rows[0]["costed"] == 1
    assert gw.ledger.uncosted_count() == 0


def test_replies_are_empty_for_no_prompts(gw: Gateway) -> None:
    assert BoundaryCaller(gw).ask([]) == []
    assert administer([], "local-small-a", BoundaryCaller(gw)) == []


def test_a_prompt_with_no_temperature_sends_no_temperature_field() -> None:
    """The last link: a Prompt carrying None must not become `temperature: 0` on the wire."""
    from mselect.runner.gateway import _request

    prompt = Prompt(
        item_id="i",
        alias="anthropic-opus",
        template="plain",
        rotation=0,
        system="s",
        user="u",
        max_tokens=16,
        temperature=None,
    )
    assert _request(prompt).temperature is None
    assert _request(replace(prompt, temperature=0.0)).temperature == 0.0


def test_a_cache_namespace_isolates_one_administration_from_another(tmp_path: Path) -> None:
    """The whole point of `--repeat`: the second administration must not read the first's cache.

    Checked at the gateway rather than through the CLI, because this is the line that does the
    work. A namespace puts the cache in a directory of its own, so an identical request built by
    a later administration misses, and the vendor is asked again. Within an administration the
    cache is untouched, which is what keeps section 3.3's "a rerun costs nothing" true.
    """
    from boundary.cache import ExactMatchCache

    root = tmp_path / "cache"
    base = ExactMatchCache(root)
    apart = ExactMatchCache(root / "r2")
    assert apart.root != base.root
    assert apart.root.parent == base.root, "a namespace lives under the cache, not beside it"


def test_missing_keys_names_the_variable_and_who_needs_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Naming the variable without naming what it costs you is half an error message."""
    config = {
        "providers": {
            "anthropic": {"api_key_env": "TEST_ANTHROPIC_KEY"},
            "local": {"price_zero": True},
        }
    }
    routes = {
        "paid-one": {"provider": "anthropic", "model": "m"},
        "paid-two": {"provider": "anthropic", "model": "m"},
        "free-one": {"provider": "local", "model": "llama"},
    }
    monkeypatch.delenv("TEST_ANTHROPIC_KEY", raising=False)
    absent = gateway.missing_keys(config, ["paid-one", "paid-two", "free-one"], routes)
    assert absent == {"TEST_ANTHROPIC_KEY": ["paid-one", "paid-two"]}

    monkeypatch.setenv("TEST_ANTHROPIC_KEY", "present")
    assert gateway.missing_keys(config, ["paid-one", "free-one"], routes) == {}


def test_a_provider_with_no_key_to_need_never_blocks_a_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The local server is the reason the panel reaches the bottom of the ability range."""
    config = {"providers": {"local": {"price_zero": True}}}
    routes = {"free-one": {"provider": "local", "model": "llama"}}
    assert gateway.missing_keys(config, ["free-one"], routes) == {}


def test_an_unfinished_batch_does_not_end_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """Found on 2026-09-14, an hour into the test-retest arm and with seven models still to go.

    `anthropic-haiku`'s first batch of 250 was still in_progress when the poll limit expired,
    `batch_results` raised `BatchNotReady`, and nothing caught it: the handler in this module
    guarded the one-at-a-time path and not the batch path. The traceback reached the command and
    the run ended having measured nothing.

    A batch still running says nothing about the other aliases, which is the same reasoning that
    already applied to every other failure here. It becomes retryable failures and the run goes
    on. The batch id is kept, because the vendor will finish and charge for that batch whether
    or not anyone waited, and a resume that submits a second one pays for the same answers twice.
    """

    class Stuck:
        """A gateway whose batches are accepted and never finish."""

        def batch_submit(self, requests: object, **kw: object) -> object:
            return SimpleNamespace(batch_id="msgbatch_test", provider="anthropic")

        def batch_results(self, handle: object, **kw: object) -> object:
            raise BatchNotReady("msgbatch_test", "in_progress", {"processing": 3})

    prompts = build_prompts([MC, MC, MC], "anthropic-haiku")
    caller = BoundaryCaller(Stuck(), use_batches=True)  # type: ignore[arg-type]
    replies = caller.ask(prompts)

    assert len(replies) == len(prompts), "every prompt still gets a reply, even a failed one"
    assert all(r.error is not None and r.retryable for r in replies)
    assert all("msgbatch_test" in str(r.error) for r in replies), "the bill has a name"
    assert caller.unfinished == ["msgbatch_test"]
