"""The interface project 03 imports.

PLAN.md section 8 is the contract: the bank and its parameters, the adaptive estimator, the
power function, reliability, and which items are not independent evidence. These tests are the
contract's teeth, so a rename inside the package cannot quietly break the consumer.
"""

from __future__ import annotations

import numpy as np
import pytest

import mselect
from mselect.irt.q3 import Q3Result


def test_the_public_interface_is_importable_the_way_03_calls_it() -> None:
    answer = mselect.power.items_needed(3, 0.8, 0.0)
    assert answer.items > 0
    assert answer.effect_points == 3.0
    assert "80% power" in answer.describe()
    # And the same call through the top-level re-export.
    assert mselect.items_needed(3, 0.8, 0.0).items == answer.items


def test_the_power_function_needs_no_arguments_about_banks() -> None:
    """A consumer that has never fitted anything still gets a number from the packaged bank."""
    three = mselect.items_needed(3).items
    one = mselect.items_needed(1).items
    five = mselect.items_needed(5).items
    assert one > three > five  # a smaller drop needs more items
    assert mselect.power_at(three, 3) == pytest.approx(0.8, abs=0.05)
    assert 0.0 < mselect.detectable_effect(200) < 100.0


def test_items_needed_rejects_impossible_questions_rather_than_guessing() -> None:
    with pytest.raises(ValueError):
        mselect.items_needed(0.0)
    with pytest.raises(ValueError):
        mselect.items_needed(3, power=1.5)


def test_the_bank_and_its_parameters_ship_inside_the_package() -> None:
    bank = mselect.default_bank()
    items = mselect.default_items()
    assert bank.n_models > 100
    assert bank.n_items > 3000  # PLAN.md section 11's floor
    assert items.n_items == bank.n_items
    assert bank.bank_hash  # a bank without a hash is not frozen


def test_reliability_reports_its_own_caveat() -> None:
    figure = mselect.reliability()
    assert 0.5 < figure.agreement < 1.0
    assert figure.n_repeated_cells > 0
    assert "not the temperature-0 test-retest" in figure.caveat.lower()


def test_dependence_tells_a_consumer_how_much_to_widen_its_interval() -> None:
    table = mselect.dependence()
    assert table, "the packaged bank must carry its dependence table"
    for benchmark, record in table.items():
        assert record.benchmark == benchmark
        assert record.variance_inflation(1) == 1.0  # one item cannot be inflated
        assert record.variance_inflation(100) >= 1.0
        assert 0 < record.effective_items(100) <= 100
    # The worst benchmark is materially worse than the best: the whole point of publishing it.
    worst = min(record.effective_items(100) for record in table.values())
    best = max(record.effective_items(100) for record in table.values())
    assert worst < best / 2


def test_dependent_blocks_are_tight_clusters_not_a_whole_benchmark() -> None:
    blocks = mselect.dependent_blocks()
    index = mselect.dependent_block_index()
    bank = mselect.default_bank()
    for block in blocks:
        assert block.size >= 2
        assert block.max_q3 >= 0.5  # a block means near-duplicates, not diffuse dependence
        assert block.size < bank.n_items / 10  # never "the entire benchmark"
        for item in block.items:
            assert index[item] >= 0


def test_blocks_chain_through_a_shared_item() -> None:
    """A depends on B and B on C puts all three in one block, even if A and C look unrelated."""
    matrix = np.array(
        [
            [np.nan, 0.9, 0.0, 0.0],
            [0.9, np.nan, 0.9, 0.0],
            [0.0, 0.9, np.nan, 0.0],
            [0.0, 0.0, 0.0, np.nan],
        ]
    )
    result = Q3Result(
        index=np.arange(4, dtype=np.intp),
        matrix=matrix,
        n_models=50,
        expected_bias=-0.33,
        pairs=[],
    )
    blocks = mselect.handover.blocks_from_q3(result, ["a", "b", "c", "d"], "fixture")
    assert len(blocks) == 1
    assert blocks[0].items == ("a", "b", "c")


def test_the_adaptive_estimator_is_exported_and_runs() -> None:
    items = mselect.default_items()
    ability = mselect.Ability()
    responses = np.isfinite(mselect.default_bank().x[0])
    asked = 0
    for position in np.flatnonzero(responses)[:25]:
        value = mselect.default_bank().x[0, position]
        ability.update(int(position), int(value), items)
        asked += 1
    assert ability.n_answered == asked
    low, high = ability.interval()
    assert low < ability.theta < high
    assert ability.se > 0


def test_the_shipped_banks_are_discoverable_and_loadable() -> None:
    """A consumer should not have to guess which banks a release carries."""
    versions = mselect.banks()
    assert "v1" in versions
    for version in versions:
        bank = mselect.default_bank(version)
        assert bank.n_models > 0 and bank.n_items > 0
        assert bank.bank_hash


def test_a_bank_with_no_repeated_cells_reports_no_reliability_rather_than_perfect() -> None:
    """100 percent agreement over zero cells is a number that looks like evidence and is not."""
    for version in mselect.banks():
        reliability = mselect.reliability(version)
        if reliability.measured:
            assert 0.0 <= reliability.agreement <= 1.0
            assert f"{reliability.agreement:.1%}" in reliability.describe()
        else:
            assert np.isnan(reliability.agreement)
            assert "no repeated cells" in reliability.describe()
