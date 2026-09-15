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


def test_an_unidentified_difficulty_destroys_a_correlation_that_the_filter_recovers() -> None:
    """Why the cross-bank result is reported twice.

    `b` is `-d/a`, so an item whose slope is near zero has a difficulty that is arithmetic
    rather than measurement. Mixing those in with real ones is what turns a correlation of 0.7
    into one of 0.0, which is exactly what the 998 shared MMLU-Pro items do.
    """
    rng = np.random.default_rng(11)
    truth = rng.normal(size=300)
    good_one = truth + rng.normal(scale=0.3, size=300)
    good_two = truth + rng.normal(scale=0.3, size=300)
    junk_one = rng.normal(scale=8.0, size=300)
    junk_two = rng.normal(scale=8.0, size=300)

    mixed_one = np.concatenate([good_one, junk_one])
    mixed_two = np.concatenate([good_two, junk_two])

    mixed = crossbank._bootstrap_statistic(mixed_one, mixed_two, crossbank._pearson, resamples=200)
    filtered = crossbank._bootstrap_statistic(good_one, good_two, crossbank._pearson, resamples=200)

    assert abs(mixed.point) < 0.3
    assert filtered.point > 0.8
    assert filtered.lo > mixed.hi, "the filter has to change the conclusion, not just the number"


def test_symmetric_flips_are_reported_as_noise_and_lopsided_ones_are_not() -> None:
    """Added 2026-09-14, after this project read its own noise as a signal.

    Section 15.28 first claimed the retest flips were "not symmetric" and that "something other
    than a coin is moving", on the strength of one model flipping 7 items right and 2 wrong.
    That is nine discordant pairs and McNemar's exact test puts it at p = 0.18. Pooled over the
    panel it was 83 against 85, p = 0.94: as symmetric as a coin.

    The distinction matters beyond the embarrassment. A random walk can only be measured and
    allowed for; a systematic shift can be corrected for. Reporting one as the other sends a
    release gate after the wrong problem.
    """
    n = 400
    first = np.zeros(n)
    second = np.zeros(n)
    # Twenty items move each way: the same agreement as a lopsided split, and a different fact.
    second[:20] = 1.0
    first[20:40] = 1.0
    even = analysis.retest("even", first, second)
    assert even.flips_to_correct == even.flips_to_incorrect == 20
    assert even.net_points == pytest.approx(0.0)
    assert even.symmetry_p > 0.05, "balanced flips are noise"

    # Everything moves one way, which is drift and must not read as noise.
    lopsided_second = np.zeros(n)
    lopsided_second[:30] = 1.0
    drifted = analysis.retest("drifted", np.zeros(n), lopsided_second)
    assert drifted.flips_to_correct == 30 and drifted.flips_to_incorrect == 0
    assert drifted.net_points == pytest.approx(7.5)
    assert drifted.symmetry_p < 0.01, "a one-way move is not a coin"


def test_a_model_that_never_changes_its_mind_has_no_symmetry_to_test() -> None:
    """No discordant pairs means the question does not arise, and nan says so."""
    same = np.array([1.0, 0.0, 1.0, 1.0])
    result = analysis.retest("steady", same, same)
    assert result.flips_to_correct == result.flips_to_incorrect == 0
    assert np.isnan(result.symmetry_p)


def test_a_re_administration_is_not_a_fresh_draw_from_the_response_model() -> None:
    """The finding of 2026-09-14, in miniature.

    Item response theory treats a response as Bernoulli(p), so the same model asked the same
    item twice should disagree with probability 2p(1-p). On the own-run panel it disagrees five
    times less often than that, on every model, because p describes how models at one ability
    differ from each other rather than how one model differs from itself.

    The consequence points the friendly way: a drift test compares a model with its own earlier
    self, so it sits in the small within-model variance and needs fewer items than the
    information function implies.
    """
    n = 1000
    # Every item a coin flip under the model, and the model in fact almost never changes.
    predicted = np.full(n, 0.5)
    first = np.zeros(n)
    second = np.zeros(n)
    second[:30] = 1.0

    result = analysis.resampling(first, second, predicted)
    assert result.n_pairs == n
    assert result.observed_flip_rate == pytest.approx(0.03)
    assert result.predicted_flip_rate == pytest.approx(0.5)
    assert result.ratio == pytest.approx(0.5 / 0.03)
    # The number a release gate sizes itself with, and the one it would have used instead.
    assert result.points_sd(500) < result.points_sd(500, predicted=True)
    assert result.points_sd(500) == pytest.approx(100 * (0.03 / 500) ** 0.5)


def test_resampling_says_nothing_when_there_is_nothing_to_compare() -> None:
    nothing = np.full(5, np.nan)
    result = analysis.resampling(nothing, nothing, np.full(5, 0.5))
    assert result.n_pairs == 0
    assert np.isnan(result.observed_flip_rate)


def test_a_thin_position_is_reported_but_does_not_set_the_bias_index() -> None:
    """Found on the first real run of the position-bias arm, 2026-09-14.

    `anthropic-haiku` came out at a bias index of 0.667, which would be an enormous effect. The
    suite mixes option counts, 253 four-option items against 13 with ten, so the later letters
    are reachable by a handful of items: 160 observations sat at A and 2 at I. The index was the
    spread between a position holding six items and one holding two.

    The function was tested against fixtures where every item had the same number of options,
    which is exactly the condition that makes every position equally observed. A real suite is
    not like that, and CLAUDE.md's rule that a bare number is a defect applies here.
    """
    # Forty items with the answer at A or B, and two lonely ones at Z that happen to be perfect.
    positions = np.array([["A", "B"]] * 40 + [["Z", "Z"]])
    correct = np.vstack([np.tile([1.0, 0.0], (40, 1)), np.array([[1.0, 1.0]])])

    result = analysis.position_bias("m", correct, positions, min_per_position=30)
    assert set(result.accuracy_by_position) == {"A", "B", "Z"}, "every position is still reported"
    assert result.accuracy_by_position["Z"].point == pytest.approx(1.0)
    assert result.positions_counted == ("A", "B"), "Z has two observations and cannot count"
    assert result.bias_index == pytest.approx(1.0), "A is always right and B always wrong"

    # With the floor lowered, Z joins in and the index changes. That is the knob doing its job.
    lenient = analysis.position_bias("m", correct, positions, min_per_position=1)
    assert lenient.positions_counted == ("A", "B", "Z")


def test_a_bias_index_with_nothing_thick_enough_is_not_invented() -> None:
    """Better no number than a number from three items."""
    positions = np.array([["A", "B"]] * 3)
    correct = np.tile([1.0, 0.0], (3, 1))
    result = analysis.position_bias("m", correct, positions, min_per_position=30)
    assert result.positions_counted == ()
    assert np.isnan(result.bias_index)
    assert result.accuracy_by_position["A"].point == pytest.approx(1.0)
