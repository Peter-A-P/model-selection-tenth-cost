"""Parameter recovery on simulated matrices.

PLAN.md section 5: "2PL and 3PL recover known parameters on simulated matrices within
tolerance". A fitter that cannot recover parameters it generated itself cannot be trusted with
a benchmark, so this is the test that has to fail meaningfully.
"""

from __future__ import annotations

import numpy as np
import pytest

from mselect.irt import fit as fitting
from mselect.irt.model import Items, prob


def simulate(
    n_models: int = 400,
    n_items: int = 300,
    *,
    seed: int = 0,
    guessing: float = 0.0,
    missing: float = 0.0,
) -> tuple[np.ndarray, Items, np.ndarray]:
    """A 2PL or 3PL matrix with known parameters, plus the abilities that generated it."""
    rng = np.random.default_rng(seed)
    theta = rng.normal(0.0, 1.0, n_models)
    a = np.exp(rng.normal(0.0, 0.35, n_items))  # centred near 1, always positive
    b = rng.normal(0.0, 1.2, n_items)
    c = np.full(n_items, guessing)
    items = Items(a, b, c)
    p = prob(theta, items)
    x = (rng.random(p.shape) < p).astype(float)
    if missing > 0:
        holes = rng.random(p.shape) < missing
        x[holes] = np.nan
    return x, items, theta


def on_the_fitted_scale(truth: Items, theta: np.ndarray) -> Items:
    """Put the generating parameters on the scale the fit is identified against.

    Marginal maximum likelihood fixes the ability scale by declaring the panel standard
    normal, so a panel whose abilities happen to have standard deviation 0.95 comes back with
    difficulties stretched by 1/0.95 and discriminations shrunk by 0.95. That is the
    identification constraint working, not error, so recovery is asserted after applying it.
    """
    scale = float(theta.std())
    centre = float(theta.mean())
    return Items(truth.a * scale, (truth.b - centre) / scale, truth.c)


def test_2pl_recovers_its_own_parameters() -> None:
    x, truth, theta = simulate(n_models=600, n_items=300, seed=1)
    fit = fitting.fit_mml(x, kind="2pl", max_iter=300)
    scaled = on_the_fitted_scale(truth, theta)

    assert fit.converged
    assert np.corrcoef(fit.items.b, truth.b)[0, 1] > 0.98
    assert np.corrcoef(fit.items.a, truth.a)[0, 1] > 0.85
    assert np.sqrt(np.mean((fit.items.b - scaled.b) ** 2)) < 0.2
    assert np.sqrt(np.mean((fit.items.a - scaled.a) ** 2)) < 0.2
    # No systematic stretch left once the identification constraint is applied.
    assert 0.95 < float(np.polyfit(scaled.b, fit.items.b, 1)[0]) < 1.06
    assert np.corrcoef(fit.theta, theta)[0, 1] > 0.95


def test_2pl_survives_a_matrix_that_is_half_holes() -> None:
    """The real matrix is sparse: models are not all run on all benchmarks."""
    x, truth, theta = simulate(n_models=600, n_items=300, seed=2, missing=0.5)
    fit = fitting.fit_mml(x, kind="2pl", max_iter=300)
    scaled = on_the_fitted_scale(truth, theta)

    assert np.corrcoef(fit.items.b, truth.b)[0, 1] > 0.96
    assert np.sqrt(np.mean((fit.items.b - scaled.b) ** 2)) < 0.3


def test_3pl_recovers_difficulty_under_guessing() -> None:
    x, truth, theta = simulate(n_models=800, n_items=200, seed=3, guessing=0.25)
    mc = np.ones(x.shape[1], dtype=bool)
    three = fitting.fit_mml(x, kind="3pl", mc=mc, max_iter=300)
    two = fitting.fit_mml(x, kind="2pl", max_iter=300)
    scaled = on_the_fitted_scale(truth, theta)

    # The 3PL should be closer to the truth than a 2PL that has to absorb the guessing.
    assert np.sqrt(np.mean((three.items.b - scaled.b) ** 2)) < np.sqrt(
        np.mean((two.items.b - scaled.b) ** 2)
    )
    assert np.corrcoef(three.items.b, truth.b)[0, 1] > 0.95
    assert 0.1 < float(np.median(three.items.c)) < 0.4


def test_negative_discrimination_is_recovered_not_hidden() -> None:
    """A mis-keyed item has a negative slope, and the fit has to say so."""
    x, truth, _ = simulate(n_models=600, n_items=200, seed=4)
    x[:, :10] = 1.0 - x[:, :10]  # ten items with their key flipped
    fit = fitting.fit_mml(x, kind="2pl", max_iter=300)

    assert float(np.max(fit.items.a[:10])) < 0.0
    assert float(np.min(fit.items.a[10:])) > 0.0
    del truth


def test_standard_errors_are_in_the_right_ballpark() -> None:
    """Analytic standard errors against the spread of repeated fits of the same truth."""
    spreads: list[float] = []
    reported: list[float] = []
    for seed in range(8):
        x, _, _ = simulate(n_models=300, n_items=60, seed=100 + seed)
        fit = fitting.fit_mml(x, kind="2pl", max_iter=200)
        spreads.append(float(np.mean(fit.items.b)))
        reported.append(float(np.mean(fit.se_b)))
    assert 0.05 < float(np.mean(reported)) < 0.5
    assert np.isfinite(spreads).all()


def test_eap_scores_a_model_the_fit_never_saw() -> None:
    x, truth, theta = simulate(n_models=400, n_items=250, seed=5)
    fit = fitting.fit_mml(x[:-50], kind="2pl", max_iter=300)
    held_out, se = fitting.eap(x[-50:], fit.items)

    assert np.corrcoef(held_out, theta[-50:])[0, 1] > 0.93
    assert float(np.median(se)) < 0.35
    del truth


@pytest.mark.parametrize("kind", ["2pl", "3pl"])
def test_fit_rejects_a_matrix_that_is_not_binary(kind: str) -> None:
    x = np.full((5, 5), 0.5)
    with pytest.raises(ValueError):
        fitting.fit_mml(x, kind=kind)
