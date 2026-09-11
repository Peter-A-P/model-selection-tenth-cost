"""The three experiment analyses, on fixtures with known answers.

The runs these consume need vendor calls and have not happened (PLAN.md section 13.4), so the
fixtures plant the effect and the test checks the analysis recovers it. That way the code is
right before it is ever pointed at data that costs money to collect.
"""

from __future__ import annotations

import numpy as np
import pytest

from mselect.experiments import analysis, crossbank


def test_retest_recovers_a_known_flip_rate() -> None:
    rng = np.random.default_rng(0)
    first = (rng.random(500) < 0.6).astype(float)
    second = first.copy()
    flip = rng.choice(500, size=50, replace=False)  # exactly 10 percent change
    second[flip] = 1.0 - second[flip]

    result = analysis.retest("fixture", first, second)

    assert abs(result.agreement.point - 0.9) < 0.01
    assert result.agreement.lo < 0.9 < result.agreement.hi
    assert result.flips_to_correct + result.flips_to_incorrect == 50
    assert 0.7 < result.phi < 0.85


def test_retest_of_an_identical_run_is_perfect_and_says_so() -> None:
    values = np.array([1.0, 0.0, 1.0, 1.0, 0.0] * 20)
    result = analysis.retest("fixture", values, values.copy())

    assert result.agreement.point == 1.0
    assert result.phi == pytest.approx(1.0)
    assert result.flips_to_correct == 0 and result.flips_to_incorrect == 0


def test_position_bias_finds_a_planted_preference_for_option_a() -> None:
    """A model that likes the first option is right more often when the answer is first."""
    rng = np.random.default_rng(1)
    n_items, rotations = 300, 4
    letters = np.array([["A", "B", "C", "D"]] * n_items)
    probability = np.where(letters == "A", 0.85, 0.55)
    correct = (rng.random((n_items, rotations)) < probability).astype(float)

    result = analysis.position_bias("fixture", correct, letters)

    assert result.accuracy_by_position["A"].point > result.accuracy_by_position["C"].point
    assert 0.2 < result.bias_index < 0.45
    assert result.share_order_dependent.point > 0.5
    assert result.accuracy_by_position["A"].lo > result.accuracy_by_position["B"].hi


def test_position_bias_is_near_zero_when_order_does_not_matter() -> None:
    rng = np.random.default_rng(2)
    letters = np.array([["A", "B", "C", "D"]] * 400)
    correct = (rng.random((400, 4)) < 0.6).astype(float)

    result = analysis.position_bias("fixture", correct, letters)

    assert result.bias_index < 0.12
    for letter in "ABCD":
        assert result.accuracy_by_position[letter].lo < 0.6 < result.accuracy_by_position[letter].hi


def test_framing_separates_item_variance_from_template_variance() -> None:
    """Items differ a lot, templates differ a little, and the decomposition should say so."""
    rng = np.random.default_rng(3)
    n_items = 400
    easy = rng.random(n_items) < 0.5  # half the items are much easier
    base = np.where(easy, 0.9, 0.2)
    shift = {"plain": 0.0, "letter_only": 0.02, "brief_reasoning": 0.08}
    templates = list(shift)
    correct = np.column_stack(
        [
            (rng.random(n_items) < np.clip(base + shift[name], 0, 1)).astype(float)
            for name in templates
        ]
    )

    result = analysis.framing("fixture", correct, templates)

    assert result.variance_item > result.variance_template
    assert result.variance_template < 0.05
    assert (
        result.accuracy_by_template["brief_reasoning"].point
        > result.accuracy_by_template["plain"].point
    )
    # The reasoning template's advantage is what the answer-only format costs.
    assert result.cost_of_answer_only.point > 0.0
    assert result.cost_of_answer_only.lo > 0.0
    assert (
        abs(result.variance_item + result.variance_template + result.variance_interaction - 1.0)
        < 1e-9
    )


def test_framing_refuses_a_mismatched_template_list() -> None:
    with pytest.raises(ValueError):
        analysis.framing("fixture", np.zeros((10, 2)), ["plain", "letter_only", "brief_reasoning"])


def test_every_reported_number_carries_an_interval() -> None:
    """CLAUDE.md: a bare number is a bug. The formatter always prints one."""
    interval = analysis.bootstrap([1.0, 0.0, 1.0, 1.0])
    assert "to" in interval.fmt() and "n = 4" in interval.fmt()
    assert analysis.bootstrap([]).fmt() == "n/a"


def test_cross_bank_correlations_recover_a_planted_relationship() -> None:
    """Two calibrations of the same items: the statistic must see a known amount of agreement."""
    rng = np.random.default_rng(4)
    truth = rng.normal(size=500)
    one = truth + rng.normal(scale=0.3, size=500)
    two = truth + rng.normal(scale=0.3, size=500)

    correlation = crossbank._bootstrap_statistic(one, two, crossbank._pearson, resamples=200)

    assert 0.8 < correlation.point < 0.95
    assert correlation.lo < correlation.point < correlation.hi
    assert correlation.n == 500


def test_cross_bank_correlation_of_unrelated_calibrations_is_zero() -> None:
    rng = np.random.default_rng(5)
    one, two = rng.normal(size=400), rng.normal(size=400)

    correlation = crossbank._bootstrap_statistic(one, two, crossbank._pearson, resamples=200)

    assert abs(correlation.point) < 0.12
    assert correlation.lo < 0.0 < correlation.hi


def test_spearman_sees_a_monotone_relationship_that_pearson_understates() -> None:
    x = np.linspace(0.1, 3.0, 200)
    y = np.exp(4.0 * x)  # perfectly ordered, wildly non-linear

    assert crossbank._spearman(x, y) == pytest.approx(1.0, abs=1e-9)
    assert crossbank._pearson(x, y) < 0.75


def test_the_decile_overlap_is_the_decision_a_consumer_actually_makes() -> None:
    """Identical orderings recover the whole decile; reversed orderings recover none of it."""
    values = np.arange(100.0)
    assert crossbank._decile_overlap(values, values.copy(), top=True) == 1.0
    assert crossbank._decile_overlap(values, -values, top=True) == 0.0
    assert crossbank._decile_overlap(values, -values, top=False) == 0.0


def test_a_bridge_with_too_few_items_reports_no_number_rather_than_a_wrong_one() -> None:
    short = np.array([1.0, 2.0])
    result = crossbank._bootstrap_statistic(short, short.copy(), crossbank._pearson)
    assert result.n == 2 and np.isnan(result.point)
