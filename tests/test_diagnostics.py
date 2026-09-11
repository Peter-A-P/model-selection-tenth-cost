"""Planted-signal tests for the diagnostics.

PLAN.md section 5: "Q3 flags planted dependent item pairs; DIF flags planted group
differences". A diagnostic that cannot find a violation put there on purpose is not evidence
when it reports none on the real bank.
"""

from __future__ import annotations

import numpy as np

from mselect.irt import dif, dimensionality, fitstats, q3
from mselect.irt import fit as fitting
from mselect.irt.model import Items, prob, sigmoid


def base_matrix(
    n_models: int = 200, n_items: int = 200, *, seed: int = 0
) -> tuple[np.ndarray, Items, np.ndarray]:
    rng = np.random.default_rng(seed)
    theta = rng.normal(0.0, 1.0, n_models)
    a = np.exp(rng.normal(0.0, 0.3, n_items))
    b = rng.normal(0.0, 1.0, n_items)
    items = Items.twopl(a, b)
    x = (rng.random((n_models, n_items)) < prob(theta, items)).astype(float)
    return x, items, theta


def test_q3_finds_pairs_that_share_a_nuisance_factor() -> None:
    """Items in the same block get a shared per-model disturbance: classic local dependence."""
    rng = np.random.default_rng(1)
    x, items, theta = base_matrix(n_models=250, n_items=120, seed=1)
    n_blocks = 10
    for block in range(n_blocks):  # the first 20 items are ten dependent pairs
        shared = rng.normal(0.0, 2.5, x.shape[0])
        for offset in (0, 1):
            j = 2 * block + offset
            z = items.a[j] * (theta - items.b[j]) + shared
            x[:, j] = (rng.random(x.shape[0]) < sigmoid(z)).astype(float)

    fit = fitting.fit_mml(x, kind="2pl", max_iter=200)
    result = q3.q3(x, fit.items, fit.theta, np.arange(x.shape[1], dtype=np.intp))

    planted = {(2 * block, 2 * block + 1) for block in range(n_blocks)}
    upper = np.triu_indices(result.matrix.shape[0], k=1)
    order = np.argsort(-np.nan_to_num(result.matrix[upper], nan=-np.inf))
    ranking = [(int(upper[0][k]), int(upper[1][k])) for k in order]

    # Nine of the ten planted pairs take the top nine places out of 7,140, and the pairs the
    # report would name are mostly planted rather than noise. The tenth pair is a fair miss:
    # both of its items are so easy that almost every model answers both correctly, leaving no
    # residual variance for any correlation to show up in. Q3 cannot see dependence between
    # items that carry no information, which is worth knowing before reading the real bank.
    assert len(planted & set(ranking[:10])) >= 9
    named = {(a, b) for a, b, _ in result.pairs}
    assert len(planted & named) / max(len(named), 1) > 0.5
    planted_values = [result.matrix[a, b] for a, b in planted]
    others = result.values[np.isfinite(result.values)]
    assert float(np.median(planted_values)) > float(np.quantile(others, 0.99))


def test_q3_on_clean_data_sits_near_its_expected_bias() -> None:
    x, _, _ = base_matrix(n_models=250, n_items=120, seed=2)
    fit = fitting.fit_mml(x, kind="2pl", max_iter=200)
    result = q3.q3(x, fit.items, fit.theta, np.arange(x.shape[1], dtype=np.intp))
    summary = result.summary()

    assert abs(summary["mean"] - result.expected_bias) < 0.03
    assert summary["share_above_flag"] < 0.02


def test_dif_finds_items_planted_to_favour_one_group() -> None:
    rng = np.random.default_rng(3)
    x, items, theta = base_matrix(n_models=300, n_items=150, seed=3)
    group = np.zeros(x.shape[0], dtype=bool)
    group[: x.shape[0] // 2] = True
    planted = np.arange(15)
    for j in planted:  # the focal group gets a 1.2 logit advantage on these items
        z = items.a[j] * (theta - items.b[j]) + 1.2 * group
        x[:, j] = (rng.random(x.shape[0]) < sigmoid(z)).astype(float)

    fit = fitting.fit_mml(x, kind="2pl", max_iter=200)
    result = dif.run_dif(
        x, fit.theta, group, grouping="planted", reference="reference", focal="focal"
    )
    flagged = np.flatnonzero(result.flagged())

    assert len(set(planted.tolist()) & set(flagged.tolist())) >= 13  # 13 of the 15 planted
    assert float((result.ets_delta[planted] > 0).mean()) == 1.0  # all favour the focal group
    clean = np.setdiff1d(np.arange(x.shape[1]), planted)
    assert float(result.flagged()[clean].mean()) < 0.05  # few false alarms
    assert float((result.uniform_p[planted] < 0.01).mean()) > 0.8


def test_dif_is_quiet_when_the_groups_really_are_the_same() -> None:
    x, _, _ = base_matrix(n_models=300, n_items=150, seed=4)
    rng = np.random.default_rng(4)
    group = np.asarray(rng.random(x.shape[0]) < 0.5, dtype=bool)
    fit = fitting.fit_mml(x, kind="2pl", max_iter=200)
    result = dif.run_dif(
        x, fit.theta, group, grouping="none", reference="a", focal="b", logistic=False
    )

    assert float(result.flagged().mean()) < 0.03


def test_parallel_analysis_separates_one_factor_from_two() -> None:
    rng = np.random.default_rng(5)
    n_models, n_items = 160, 80
    theta = rng.normal(0.0, 1.0, n_models)
    second = rng.normal(0.0, 1.0, n_models)
    b = rng.normal(0.0, 0.8, n_items)

    one = (rng.random((n_models, n_items)) < sigmoid(1.3 * (theta[:, None] - b[None, :]))).astype(
        float
    )
    loading = np.where(np.arange(n_items) < n_items // 2, 1.3, 0.0)
    other = np.where(np.arange(n_items) < n_items // 2, 0.0, 1.3)
    z = loading * (theta[:, None] - b[None, :]) + other * (second[:, None] - b[None, :])
    two = (rng.random((n_models, n_items)) < sigmoid(z)).astype(float)

    one_factor = dimensionality.parallel_analysis(one, draws=10, seed=0)
    two_factor = dimensionality.parallel_analysis(two, draws=10, seed=0)

    assert one_factor.ratio_first_to_second > two_factor.ratio_first_to_second
    assert one_factor.variance_first > two_factor.variance_first


def test_tetrachoric_approximation_tracks_the_exact_value() -> None:
    rng = np.random.default_rng(6)
    latent = rng.multivariate_normal([0, 0], [[1.0, 0.6], [0.6, 1.0]], size=4000)
    x = (latent > 0.3).astype(float)
    approx, _ = dimensionality.tetrachoric_matrix(x)
    both = float(((x[:, 0] == 1) & (x[:, 1] == 1)).sum())
    one = float(((x[:, 0] == 1) & (x[:, 1] == 0)).sum())
    other = float(((x[:, 0] == 0) & (x[:, 1] == 1)).sum())
    neither = float(((x[:, 0] == 0) & (x[:, 1] == 0)).sum())
    exact = dimensionality.tetrachoric_exact((both, one, other, neither))

    assert abs(approx[0, 1] - exact) < 0.05
    assert abs(exact - 0.6) < 0.08


def test_fit_statistics_flag_a_mis_keyed_item() -> None:
    x, _, _ = base_matrix(n_models=250, n_items=100, seed=7)
    x[:, :5] = 1.0 - x[:, :5]  # five items with a flipped key
    fit = fitting.fit_mml(x, kind="2pl", max_iter=200)
    stats = fitstats.item_fit(x, fit.items, fit.theta)
    flagged = fitstats.flags(stats, fit.items)

    assert flagged["negative_discrimination"][:5].all()
    assert not flagged["negative_discrimination"][5:].any()
    # The empirical curve falls as ability rises, which is the mis-keyed signature the report
    # shows as evidence; it fires on four of the five without needing the fitted slope at all.
    assert int(flagged["non_monotone"][:5].sum()) >= 4
    assert float(flagged["non_monotone"][5:].mean()) < 0.05
    # Outfit does NOT catch these, and that is the point of keeping both statistics: a 2PL
    # fits a flipped item comfortably by giving it a negative slope, so the residuals look
    # ordinary. Only the sign of the slope and the falling empirical curve give it away.
    assert abs(float(np.median(stats.outfit[:5])) - 1.0) < 0.2
    assert float(np.median(stats.point_biserial[:5])) < 0.0


def test_separation_auc_points_the_right_way() -> None:
    """An item the strong models get right scores 1; a flipped one scores 0."""
    theta = np.linspace(-2.0, 2.0, 40)
    perfect = np.where(theta > 0.0, 1.0, 0.0)
    flipped = 1.0 - perfect
    coin = np.tile([1.0, 0.0], 20)
    x = np.column_stack([perfect, flipped, coin])

    auc = fitstats.separation_auc(x, theta)
    assert auc[0] == 1.0
    assert auc[1] == 0.0
    assert 0.3 < auc[2] < 0.7


def test_perfect_separation_is_flagged_and_its_slope_is_prior_bound() -> None:
    """A perfectly ordered item has no likelihood maximum in the slope, so the prior sets it."""
    rng = np.random.default_rng(11)
    x, _, theta = base_matrix(n_models=120, n_items=60, seed=11)
    fit = fitting.fit_mml(x, kind="2pl", max_iter=200)
    order = np.argsort(fit.theta)
    for j in range(3):  # three items answered correctly by exactly the top half
        column = np.zeros(x.shape[0])
        column[order[x.shape[0] // 2 :]] = 1.0
        x[:, j] = column
    del rng, theta

    refit = fitting.fit_mml(x, kind="2pl", max_iter=200)
    stats = fitstats.item_fit(x, refit.items, refit.theta)
    flagged = fitstats.flags(stats, refit.items)

    assert flagged["perfect_separation"][:3].all()
    assert float(stats.separation_auc[:3].min()) == 1.0
    # All three land on nearly the same slope despite being independent items: that value is the
    # prior's, not the data's, which is why the report calls it a lower bound.
    assert float(np.ptp(refit.items.a[:3])) < 0.2
