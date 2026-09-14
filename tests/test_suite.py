"""Choosing the own-run suite and pricing it, with no network and no money.

The suite is the thing the adaptive test will be measured against, so how it is drawn decides
whether the headline claim is tested or flattered. The two properties that matter are pinned
here: the mix follows the pool, and the draw does not look at any item's fitted parameters.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from mselect.runner import prompts, suite
from mselect.runner.administer import Item
from mselect.runner.items import Pool

PRICES = {
    "per_million_tokens": {
        "anthropic": {
            "cheap": {"input": 1.0, "output": 5.0, "batch_multiplier": 0.5},
        },
        "openai": {"unlisted-elsewhere": {"input": 2.0, "output": 8.0}},
    }
}
ROUTES = {
    "anthropic-haiku": {"provider": "anthropic", "model": "cheap"},
    "anthropic-opus": {"provider": "anthropic", "model": "cheap"},
    "openai-mid": {"provider": "openai", "model": "not-in-the-price-file"},
    "local-small-a": {"provider": "local", "model": "llama3.2:3b"},
}
PANEL = (
    prompts.PanelEntry("anthropic-haiku", "mid", "full suite", ""),
    prompts.PanelEntry("anthropic-opus", "frontier", "adaptive subset", ""),
    prompts.PanelEntry("openai-mid", "mid", "full suite", ""),
    prompts.PanelEntry("local-small-a", "local", "full suite", ""),
)


def _pool(counts: dict[str, int]) -> Pool:
    items = tuple(
        Item(
            item_id=f"{benchmark}{index:05d}",
            benchmark=benchmark,
            kind="multiple_choice",
            question=f"Question {index} of {benchmark}?",
            options=("a", "b", "c", "d"),
            answer="b",
        )
        for benchmark, n in counts.items()
        for index in range(n)
    )
    return Pool(version="v1", items=items, excluded=())


def test_the_mix_follows_the_pool() -> None:
    """The own-run suite is a miniature of the bank, not a rebalanced version of it."""
    pool = _pool({"mmlu": 7_000, "math": 2_000, "gpqa": 1_000})
    drawn = suite.choose(pool, 1_000, seed=1, today="2026-09-11")
    assert drawn.by_benchmark(pool) == {"gpqa": 100, "math": 200, "mmlu": 700}
    assert len(drawn.item_ids) == 1_000
    assert len(set(drawn.item_ids)) == 1_000, "no item is asked twice"


def test_the_counts_sum_to_the_size_asked_for() -> None:
    """Largest remainder, so rounding never loses or invents an item."""
    pool = _pool({"mmlu": 990, "math": 7, "gpqa": 3})
    drawn = suite.choose(pool, 100, seed=1, today="2026-09-11")
    assert sum(drawn.by_benchmark(pool).values()) == 100
    # A benchmark that is 0.3 percent of the pool gets 0.3 of a seat in a suite of 100, and
    # rounds to none. That is proportional selection doing what it says, not a bug: the suite
    # is a miniature of the bank, and a miniature of 0.3 percent is nothing. The guard against
    # it mattering is size, which the real suite has; see the test below.
    assert drawn.by_benchmark(pool)["mmlu"] == 99


def test_a_benchmark_is_never_asked_for_more_than_it_has() -> None:
    pool = _pool({"mmlu": 10, "math": 90})
    drawn = suite.choose(pool, 95, seed=1, today="2026-09-11")
    counts = drawn.by_benchmark(pool)
    assert counts["mmlu"] <= 10 and counts["math"] <= 90
    assert sum(counts.values()) == 95


def test_the_seed_is_the_draw() -> None:
    pool = _pool({"mmlu": 500, "math": 500})
    first = suite.choose(pool, 100, seed=7, today="2026-09-11")
    again = suite.choose(pool, 100, seed=7, today="2026-09-11")
    other = suite.choose(pool, 100, seed=8, today="2026-09-11")
    assert first.item_ids == again.item_ids, "the same seed is the same suite"
    assert first.item_ids != other.item_ids


def test_asking_for_more_than_the_pool_is_refused() -> None:
    with pytest.raises(ValueError, match="pool of"):
        suite.choose(_pool({"mmlu": 10}), 11)


def test_a_suite_survives_a_round_trip(tmp_path: Path) -> None:
    pool = _pool({"mmlu": 100})
    drawn = suite.choose(pool, 10, seed=3, today="2026-09-11")
    path = drawn.save(tmp_path / "suite.json")
    assert suite.Suite.load(path) == drawn


def test_a_model_with_no_listed_price_is_not_free() -> None:
    """The gateway's rule: an unknown rate is uncosted, and a total that hides it is a lie."""
    pool = _pool({"mmlu": 100})
    drawn = suite.choose(pool, 10, seed=1, today="2026-09-11")
    estimate = suite.estimate(drawn, pool, ROUTES, PRICES, panel=PANEL, adaptive_items=4)
    priced = {line.alias: line.usd for line in estimate.lines}
    assert priced["openai-mid"] is None, "no rate listed"
    assert estimate.unpriced == ("openai-mid",)
    assert priced["local-small-a"] == 0.0, "the laptop is free, and that is a rate"
    assert priced["anthropic-haiku"] is not None and priced["anthropic-haiku"] > 0


def test_the_frontier_tier_is_asked_the_subset_and_the_rest_the_suite() -> None:
    pool = _pool({"mmlu": 100})
    drawn = suite.choose(pool, 20, seed=1, today="2026-09-11")
    estimate = suite.estimate(drawn, pool, ROUTES, PRICES, panel=PANEL, adaptive_items=5)
    calls = {line.alias: line.calls for line in estimate.lines}
    assert calls == {
        "anthropic-haiku": 20,
        "anthropic-opus": 5,
        "openai-mid": 20,
        "local-small-a": 20,
    }


def test_a_batch_rate_is_only_claimed_where_a_batch_can_be_sent() -> None:
    """Found 2026-09-13, against the bill for the full-suite arm.

    The vendors publish batch rates and the price file records them, so the estimate halved
    everything whenever `batch=True`. The gateway can only send a batch to Anthropic: it tries,
    and falls back to standard calls where the provider has no batch endpoint. The ledger for
    the arm shows 9,045 Anthropic calls carrying a batch id and 21,000 others carrying none, and
    the estimate had priced all of them at half. OpenAI and Google came in at twice their line.

    A price list that is right can still be read wrongly, and this is where it was read.
    """
    prices = {
        "per_million_tokens": {
            "anthropic": {"cheap": {"input": 1.0, "output": 5.0, "batch_multiplier": 0.5}},
            # Publishes a batch rate that this project has no way to use.
            "google": {"flash": {"input": 1.0, "output": 5.0, "batch_multiplier": 0.5}},
        }
    }
    routes = {
        "anthropic-haiku": {"provider": "anthropic", "model": "cheap"},
        "google-mid": {"provider": "google", "model": "flash"},
    }
    panel = (
        prompts.PanelEntry("anthropic-haiku", "mid", "full suite", ""),
        prompts.PanelEntry("google-mid", "mid", "full suite", ""),
    )
    pool = _pool({"mmlu": 100})
    drawn = suite.choose(pool, 20, seed=1, today="2026-09-11")

    batched = suite.estimate(drawn, pool, routes, prices, panel=panel, batch=True)
    single = suite.estimate(drawn, pool, routes, prices, panel=panel, batch=False)
    # Both aliases have a listed rate, so neither is uncosted and both figures are real.
    by_alias = {line.alias: line.usd for line in batched.lines if line.usd is not None}
    alone = {line.alias: line.usd for line in single.lines if line.usd is not None}
    assert set(by_alias) == set(alone) == {"anthropic-haiku", "google-mid"}

    assert by_alias["anthropic-haiku"] == pytest.approx(alone["anthropic-haiku"] / 2)
    assert by_alias["google-mid"] == pytest.approx(alone["google-mid"]), (
        "a provider the gateway cannot batch to pays the standard rate, whatever the price "
        "file says the vendor would charge for a batch"
    )
    assert "google" not in suite.BATCHING_PROVIDERS


def test_batching_halves_the_rate_that_says_it_does() -> None:
    pool = _pool({"mmlu": 100})
    drawn = suite.choose(pool, 20, seed=1, today="2026-09-11")
    batched = suite.estimate(drawn, pool, ROUTES, PRICES, panel=PANEL, batch=True)
    single = suite.estimate(drawn, pool, ROUTES, PRICES, panel=PANEL, batch=False)
    assert batched.usd == pytest.approx(single.usd / 2)


def test_the_experiments_are_priced_on_the_full_suite_models_only() -> None:
    """A repeat of items a frontier model never saw is not a test-retest."""
    pool = _pool({"mmlu": 4_000})
    drawn = suite.choose(pool, 3_000, seed=1, today="2026-09-11")
    parts = suite.programme(drawn, pool, ROUTES, PRICES, panel=PANEL)
    assert set(parts) == {
        "full suite and frontier check",
        *(name for name, *_ in suite.EXPERIMENTS),
    }
    for name, *_ in suite.EXPERIMENTS:
        assert "anthropic-opus" not in {line.alias for line in parts[name].lines}
    retest = {line.alias: line.calls for line in parts["test-retest"].lines}
    assert retest["anthropic-haiku"] == 500, "one further administration of 500 items"
    rotations = {line.alias: line.calls for line in parts["position-bias"].lines}
    assert rotations["anthropic-haiku"] == 900, "three further rotations of 300 items"


def test_a_measured_output_beats_the_cap_and_says_so() -> None:
    """The estimate must not price a reasoning budget as though every model fills it.

    With a cap large enough for reasoning, assuming every model generates the maximum is a
    worst case multiplied by eleven models and 3,000 items, and it is the number that
    authorises the spend. A line built from real output tokens is marked measured; one built
    from the cap is marked as the upper bound it is.
    """
    pool = _pool({"mmlu": 4_000})
    drawn = suite.choose(pool, 3_000, seed=1, today="2026-09-11")
    capped = suite.estimate(drawn, pool, ROUTES, PRICES, panel=PANEL)
    measured = suite.estimate(
        drawn, pool, ROUTES, PRICES, panel=PANEL, observed={"anthropic-haiku": 6.0}
    )

    before = next(x for x in capped.lines if x.alias == "anthropic-haiku")
    after = next(x for x in measured.lines if x.alias == "anthropic-haiku")
    assert not before.measured and after.measured
    assert after.output_tokens < before.output_tokens / 10
    assert after.usd is not None and before.usd is not None and after.usd < before.usd
    assert "anthropic-haiku" in capped.capped
    assert "anthropic-haiku" not in measured.capped


def test_a_measurement_above_the_cap_is_held_to_the_cap() -> None:
    """A model cannot generate more than it is allowed, whatever an old record says."""
    pool = _pool({"mmlu": 100})
    drawn = suite.choose(pool, 10, seed=1, today="2026-09-11")
    huge = suite.estimate(
        drawn, pool, ROUTES, PRICES, panel=PANEL, observed={"anthropic-haiku": 10_000.0}
    )
    line = next(x for x in huge.lines if x.alias == "anthropic-haiku")
    assert line.output_tokens == prompts.Settings().tokens_for("plain") * line.calls


def test_the_committed_suite_is_the_one_the_code_draws() -> None:
    """The file in `config/` is a record of a decision, so it has to match the decision."""
    path = suite.default_path("v1")
    if not path.exists():  # pragma: no cover - present in the repository
        pytest.skip("no committed suite")
    committed = suite.Suite.load(path)
    assert committed.size == len(committed.item_ids) == suite.DEFAULT_SIZE
    assert committed.seed == suite.DEFAULT_SEED
    assert len(set(committed.item_ids)) == committed.size
    assert committed.item_ids == tuple(sorted(committed.item_ids))
    assert Counter(len(i) for i in committed.item_ids) == {16: committed.size}


def test_every_benchmark_of_the_real_pool_is_in_the_real_suite() -> None:
    """Proportional selection can round a tiny benchmark away, so check it does not here.

    The smallest benchmark that can be administered is MATH at 437 of 19,919 items, which is
    2.2 percent and 66 seats in a suite of 3,000. This is the assertion that the suite size
    chosen is large enough for the mix to survive it.
    """
    pool = _pool(
        {
            "mmlu": 13_937,
            "legalbench": 2_047,
            "gsm8k": 1_000,
            "med_qa": 1_000,
            "mmlu_pro": 998,
            "commonsense": 500,
            "math": 437,
        }
    )
    drawn = suite.choose(pool, suite.DEFAULT_SIZE, seed=suite.DEFAULT_SEED, today="2026-09-11")
    counts = drawn.by_benchmark(pool)
    assert set(counts) == {
        "mmlu",
        "legalbench",
        "gsm8k",
        "med_qa",
        "mmlu_pro",
        "commonsense",
        "math",
    }
    assert min(counts.values()) >= 60, counts
    assert sum(counts.values()) == suite.DEFAULT_SIZE


def test_the_real_panel_gives_every_model_a_full_suite_run() -> None:
    """Section 4.2 compares an adaptive ranking against the own-run full-suite ranking.

    A model asked only the adaptive subset cannot be in the second of those, so it widens the
    panel without widening the validation. Every alias runs the whole suite, which PLAN.md
    section 7 held budget for and the measured cost made affordable.
    """
    assert len(prompts.PANEL) >= 10, "section 7's first choice: ten full-suite models"
    subset_only = [e.alias for e in prompts.PANEL if e.coverage != "full suite"]
    assert not subset_only, f"asked the subset only, so absent from the ranking: {subset_only}"
    assert {e.tier for e in prompts.PANEL} == {"mid", "frontier", "open weights", "local"}


def test_every_panel_alias_has_a_route_and_a_price() -> None:
    """An alias with no route cannot be called; one with no price writes an uncosted row."""
    from mselect.runner import gateway

    config = gateway.load_config()
    routes = gateway.routes_of(config)
    providers = config.get("providers", {})
    prices = suite.load_prices(suite.latest_price_file(gateway.CONFIG.parent / "prices"))
    listed = prices.get("per_million_tokens", {})

    for entry in prompts.PANEL:
        route = routes.get(entry.alias)
        assert route is not None, f"{entry.alias} has no route"
        if providers.get(route["provider"], {}).get("price_zero"):
            continue
        assert route["model"] in listed.get(route["provider"], {}), (
            f"{entry.alias} -> {route['model']} has no rate in the price file"
        )


def _records(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def test_a_truncated_reply_is_not_counted_as_a_measurement(tmp_path: Path) -> None:
    """The trap this nearly walked into.

    A reply cut off by the token cap generated what it was allowed to, not what the model
    would have generated. Counting those would report a reasoning model at sixteen tokens an
    item because sixteen was the cap, which is the truncation reading itself back as evidence
    and understates the bill by an order of magnitude.
    """
    path = _records(
        tmp_path / "smoke.jsonl",
        [
            {
                "alias": "reasoner",
                "output_tokens": 16,
                "error": "the model returned no text (max_tokens), 16 output tokens spent",
                "correct": None,
            },
            {
                "alias": "reasoner",
                "output_tokens": 16,
                "error": "the model returned no text (max_tokens), 16 output tokens spent",
                "correct": None,
            },
            {"alias": "answerer", "output_tokens": 5, "error": None, "correct": 1},
            {"alias": "answerer", "output_tokens": 7, "error": None, "correct": 0},
        ],
    )
    observed = suite.observed_output(path)
    assert "reasoner" not in observed, "every one of its replies was cut off"
    assert observed["answerer"] == pytest.approx(6.0), "a wrong answer is still a finished one"


def test_an_unparsed_reply_is_not_counted_either(tmp_path: Path) -> None:
    """Unparsed usually means the reply stopped mid-sentence, which is truncation by another
    name. `google-mid` returned "To determine which developmental milestone is delayed, let's"
    and nothing more."""
    path = _records(
        tmp_path / "smoke.jsonl",
        [
            {"alias": "m", "output_tokens": 12, "error": None, "correct": None, "unparsed": True},
            {"alias": "m", "output_tokens": 4, "error": None, "correct": 1},
        ],
    )
    assert suite.observed_output(path)["m"] == pytest.approx(4.0)


def test_no_records_means_no_measurement_rather_than_zero(tmp_path: Path) -> None:
    assert suite.observed_output(tmp_path / "missing.jsonl") == {}
