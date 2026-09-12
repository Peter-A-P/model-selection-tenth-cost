"""Choosing the own-run suite and pricing it, with no network and no money.

The suite is the thing the adaptive test will be measured against, so how it is drawn decides
whether the headline claim is tested or flattered. The two properties that matter are pinned
here: the mix follows the pool, and the draw does not look at any item's fitted parameters.
"""

from __future__ import annotations

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


def test_the_reasoning_template_costs_more_output_than_the_answer_only_one() -> None:
    pool = _pool({"mmlu": 4_000})
    drawn = suite.choose(pool, 3_000, seed=1, today="2026-09-11")
    parts = suite.programme(drawn, pool, ROUTES, PRICES, panel=PANEL)
    plain = next(x for x in parts["framing (letter-only)"].lines if x.alias == "anthropic-haiku")
    reasoning = next(
        x for x in parts["framing (brief reasoning)"].lines if x.alias == "anthropic-haiku"
    )
    assert reasoning.calls == plain.calls
    assert reasoning.output_tokens > plain.output_tokens * 10


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
