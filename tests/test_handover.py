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
    """Whichever figure a consumer gets, it has to say what it is.

    Amended 2026-09-14. This used to assert the caveat said "not the temperature-0 test-retest",
    which was right while the figure was HELM's overlapping administrations standing in for one.
    The own-run test-retest is measured now and is returned in preference, so the assertion is
    about the caveat matching the source rather than about one particular source.
    """
    figure = mselect.reliability()
    assert 0.5 < figure.agreement < 1.0
    assert figure.n_repeated_cells > 0
    if figure.from_own_run:
        assert "temperature 0" in figure.source
        assert "worst hosted model" in figure.caveat.lower()
    else:
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


def test_both_banks_answer_the_whole_version_taking_interface() -> None:
    """A consumer that picks bank v2 must not fall off the interface halfway through."""
    for version in mselect.banks():
        assert mselect.load_bank(version).n_items > 0
        assert mselect.default_items(version).a.size > 0
        assert isinstance(mselect.dependence(version), dict)
        assert isinstance(mselect.dependent_blocks(version), tuple)
        assert isinstance(mselect.dependent_block_index(version), dict)
        assert mselect.reliability(version).describe()


def test_the_second_bank_is_dense_where_the_first_is_not() -> None:
    """The reason bank v2 exists: its headline needs no block peeled out of it."""
    density = {v: float(np.isfinite(mselect.load_bank(v).x).mean()) for v in mselect.banks()}
    assert density["v1"] < 0.7
    if "v2" in density:
        assert density["v2"] > 0.95


def test_reliability_prefers_the_measurement_over_the_stand_in() -> None:
    """Project 03 imports this to size its release gate, so it must get the real figure.

    Until 2026-09-14 it returned HELM's overlapping administrations, which differ in run date
    and release as well as in sampling and so bound benchmark noise from below rather than
    measuring it. The own-run test-retest measures it directly, and `from_own_run` is how a
    caller tells which of the two it is holding.
    """
    result = mselect.reliability()
    assert result.from_own_run, "the measured test-retest is committed and should be preferred"
    assert 0.9 < result.agreement < 1.0
    assert result.score_move_points is not None and result.score_move_points > 0

    # The sizing numbers a gate needs, and they must shrink as the test gets longer.
    assert result.points_sd(100) > result.points_sd(500) > 0
    assert "times fewer" in result.resampling_note()


def test_the_reliability_figure_is_the_worst_hosted_model_not_the_average() -> None:
    """A gate has to hold for the model it is watching. The two on a laptop are near-perfect
    and would flatter any average they were included in."""
    import json

    from mselect import paths

    payload = json.loads(
        (paths.ROOT / "mselect" / "config" / "own-run-retest-v1.json").read_text(encoding="utf-8")
    )
    hosted = {
        name: value for name, value in payload["agreement"].items() if not name.startswith("local-")
    }
    assert mselect.reliability().agreement == pytest.approx(min(hosted.values()))
    assert min(hosted.values()) < min(
        payload["agreement"][n] for n in payload["agreement"] if n.startswith("local-")
    )


def test_points_sd_is_sized_from_a_hosted_model_not_from_a_laptop() -> None:
    """The number a gate sizes itself with must not be flattered by near-deterministic models.

    Raised by project 03 on 2026-09-16. `agreement` had already been fixed to report the worst
    hosted model, and the caveat string said the laptop models were excluded, but `points_sd`
    was still derived from the pooled flip rate with those models in it. On bank v1 that
    understates a gate's noise floor by a factor of 1.45 against the worst hosted arm, which is
    the false-pass direction: a gate certifying it can detect drift it would in fact miss.
    """
    result = mselect.reliability()
    assert result.worst_hosted_flip_rate is not None
    assert result.observed_flip_rate is not None
    # The pooled rate is the smaller one, which is exactly why it was the wrong number.
    assert result.worst_hosted_flip_rate > result.observed_flip_rate
    assert result.points_sd(500) == pytest.approx(
        100.0 * (result.worst_hosted_flip_rate / 500) ** 0.5
    )
    # The panel-wide figure is still reachable, and saying which one you took is the point.
    assert result.points_sd(500, pooled=True) == pytest.approx(
        100.0 * (result.observed_flip_rate / 500) ** 0.5
    )
    assert result.points_sd(500) > result.points_sd(500, pooled=True)


def test_another_laptop_model_cannot_move_the_gates_noise_floor() -> None:
    """The regression 03 predicted, made impossible rather than merely unlikely.

    A third near-deterministic local model drags the pooled flip rate down and would have made
    the gate look more sensitive still. Sized from the worst hosted model, a laptop model cannot
    reach the figure at all, whatever its agreement and however many of them there are.
    """
    from mselect.handover import _worst_hosted_flip_rate

    payload = {
        "agreement": {"openai-mid": 0.936, "local-small-a": 0.998, "local-small-b": 1.000},
    }
    before = _worst_hosted_flip_rate(payload)
    payload["agreement"]["local-mid-a"] = 0.999
    assert _worst_hosted_flip_rate(payload) == before
    assert before == pytest.approx(1.0 - 0.936)


def test_an_artifact_written_before_the_fix_still_sizes_from_a_hosted_model() -> None:
    """The committed summary predates the new key, and a missing key must not silently
    fall back to a pooled figure. It is derived from the per-model map instead."""
    from mselect.handover import _worst_hosted_flip_rate

    assert _worst_hosted_flip_rate({"agreement": {"a": 0.9, "local-x": 1.0}}) == pytest.approx(0.1)
    assert _worst_hosted_flip_rate({"worst_hosted_flip_rate": 0.25}) == pytest.approx(0.25)
    assert _worst_hosted_flip_rate({}) is None
