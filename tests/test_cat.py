"""The adaptive estimator, the selector, and the stopping rules on simulated responses.

PLAN.md section 5: "the adaptive estimator converges to the true ability on simulated responses
with the expected standard error".
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mselect.cat import select, simulate, stop
from mselect.cat.estimate import Ability, score
from mselect.irt.model import Items, prob


def bank(n_items: int = 600, *, seed: int = 0) -> Items:
    rng = np.random.default_rng(seed)
    a = np.exp(rng.normal(0.0, 0.3, n_items))
    b = rng.normal(0.0, 1.3, n_items)
    return Items.twopl(a, b)


def answers(items: Items, theta: float, *, seed: int = 0) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    p = prob(np.array([theta]), items)[0]
    drawn: NDArray[np.float64] = (rng.random(p.size) < p).astype(float)
    return drawn


def selector_for(items: Items, *, balance: bool = True) -> select.Selector:
    labels = np.array(
        ["alpha"] * (items.n_items // 2) + ["beta"] * (items.n_items - items.n_items // 2)
    )
    strata, weights, _ = select.benchmark_strata(labels)
    return select.Selector(
        items=items,
        available=np.ones(items.n_items, dtype=bool),
        strata=strata,
        weights=weights,
        balance=balance,
    )


def test_adaptive_estimate_converges_on_true_ability() -> None:
    items = bank()
    errors = []
    reported = []
    for run, truth in enumerate(np.linspace(-1.5, 1.5, 13)):
        responses = answers(items, truth, seed=run)
        rng = np.random.default_rng(run)
        thetas, ses = simulate.adaptive_run(
            responses,
            items,
            selector_for(items),
            checkpoints=[60],
            max_items=60,
            rng=rng,
        )
        errors.append(thetas[60] - truth)
        reported.append(ses[60])

    assert float(np.abs(np.mean(errors))) < 0.15  # no systematic drift
    assert float(np.sqrt(np.mean(np.square(errors)))) < 0.35
    # The posterior standard error is the estimator's own claim about its accuracy; it has to
    # be in the same range as the error actually made, or every interval downstream is a lie.
    assert 0.5 < float(np.mean(reported)) / float(np.sqrt(np.mean(np.square(errors)))) < 2.0


def test_more_items_means_a_tighter_posterior() -> None:
    items = bank(seed=3)
    responses = answers(items, 0.4, seed=3)
    thetas, ses = simulate.adaptive_run(
        responses,
        items,
        selector_for(items),
        checkpoints=[10, 40, 160],
        max_items=160,
        rng=np.random.default_rng(0),
    )
    assert ses[10] > ses[40] > ses[160]
    assert abs(thetas[160] - 0.4) < 0.4


def test_adaptive_beats_random_selection_at_the_same_item_count() -> None:
    """The whole claim in one test: choosing items by information is worth something."""
    items = bank(n_items=800, seed=5)
    adaptive_err: list[float] = []
    random_err: list[float] = []
    for run, truth in enumerate(np.linspace(-1.2, 1.2, 9)):
        responses = answers(items, truth, seed=100 + run)
        rng = np.random.default_rng(run)
        thetas, _ = simulate.adaptive_run(
            responses, items, selector_for(items), checkpoints=[30], max_items=30, rng=rng
        )
        adaptive_err.append(thetas[30] - truth)
        picked = select.random_subset(np.ones(items.n_items, dtype=bool), 30, rng)
        random_err.append(score(responses, items, picked).theta - truth)

    assert np.sqrt(np.mean(np.square(adaptive_err))) < np.sqrt(np.mean(np.square(random_err)))


def test_content_balancing_keeps_the_benchmark_mix() -> None:
    items = bank(n_items=400, seed=7)
    selector = selector_for(items)
    responses = answers(items, 0.0, seed=7)
    used = np.zeros(items.n_items, dtype=bool)
    counts = np.zeros(2)
    ability = Ability()
    rng = np.random.default_rng(0)
    for _ in range(80):
        item = selector.next_item(ability.theta, used, counts, rng)
        used[item] = True
        counts[selector.strata[item]] += 1
        ability.update(item, int(responses[item]), items)

    assert abs(counts[0] - counts[1]) <= 2  # two benchmarks of equal weight, 80 items


def test_unbalanced_selection_drifts_off_the_mix() -> None:
    """The comparison that shows content balancing is doing something."""
    rng = np.random.default_rng(0)
    a = np.concatenate([np.full(200, 2.2), np.full(200, 0.4)])
    b = np.concatenate([rng.normal(0, 0.3, 200), rng.normal(0, 1.5, 200)])
    items = Items.twopl(a, b)
    responses = answers(items, 0.0, seed=11)
    selector = selector_for(items, balance=False)
    used = np.zeros(items.n_items, dtype=bool)
    counts = np.zeros(2)
    ability = Ability()
    for _ in range(60):
        item = selector.next_item(ability.theta, used, counts, rng)
        used[item] = True
        counts[selector.strata[item]] += 1
        ability.update(item, int(responses[item]), items)

    assert counts[0] > 3 * counts[1]


def test_mle_agrees_with_eap_once_there_are_enough_items() -> None:
    items = bank(seed=13)
    responses = answers(items, 0.8, seed=13)
    picked = select.random_subset(np.ones(items.n_items, dtype=bool), 200, np.random.default_rng(1))
    ability = score(responses, items, picked)
    assert abs(ability.mle(items) - ability.theta) < 0.2


def test_pairwise_rule_separates_a_clear_gap_and_refuses_a_small_one() -> None:
    items = bank(n_items=500, seed=17)
    strong = score(items=items, responses=answers(items, 1.2, seed=1), index=np.arange(200))
    weak = score(items=items, responses=answers(items, -1.2, seed=2), index=np.arange(200))
    close = score(items=items, responses=answers(items, 1.25, seed=3), index=np.arange(200))

    clear = stop.separated(strong, weak)
    assert clear.separated and clear.leader == "a"
    assert not stop.separated(strong, close).separated


def test_precision_rule_waits_for_the_prior_to_be_outvoted() -> None:
    items = bank(seed=19)
    ability = Ability()
    assert not stop.precision_reached(ability, 1.0)  # no items answered yet
    responses = answers(items, 0.0, seed=19)
    for item in range(40):
        ability.update(item, int(responses[item]), items)
    assert stop.precision_reached(ability, 0.6)


def test_dense_block_finds_the_rectangle_inside_a_ragged_matrix() -> None:
    rng = np.random.default_rng(0)
    x = np.full((60, 900), np.nan)
    x[:40, :600] = (rng.random((40, 600)) < 0.6).astype(float)  # the dense block
    x[40:, 600:] = (rng.random((20, 300)) < 0.6).astype(float)  # a disjoint corner
    block = simulate.dense_block(x, min_density=0.95, min_models=10, min_items=100)

    assert block.density >= 0.95
    assert block.rows.size >= 20
    assert set(block.rows.tolist()).issubset(set(range(40))) or set(block.rows.tolist()).issubset(
        set(range(40, 60))
    )
