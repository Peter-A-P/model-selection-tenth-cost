"""Marginal maximum likelihood for the 2PL and 3PL, by expectation maximisation.

PLAN.md section 4.1 named `py-irt` (variational, PyTorch) for the full matrix and PyMC on a
subset. This module fits the same models by Bock-Aitkin EM in numpy instead. Why, written
down as PLAN.md asks: the matrix here is at most a few hundred models by some tens of
thousands of items, which EM with 61 quadrature points fits exactly in under a minute, so a
1.5 GB PyTorch dependency and a variational approximation would both cost more than they buy.
The Bayesian cross-check the plan wanted from PyMC is kept as a nonparametric bootstrap over
models (`bootstrap_items`), which needs no new dependency and is checked against the analytic
standard errors.

Priors, stated as the plan requires. The fit is a MAP fit, not a bare MLE:

* discrimination `a` ~ Normal(1, 1), which is weakly informative on the logistic scale and,
  importantly, does not forbid a negative slope: a mis-keyed item must be free to show one.
* intercept `d = -a b` ~ Normal(0, 4), which keeps items that every model got right (or every
  model got wrong) at a finite difficulty instead of running off to plus or minus infinity.
* guessing `c` ~ Beta(2, 8) for multiple-choice items, mean 0.2, and fixed at 0 for
  free-response items where it is not identified (PLAN.md section 4.1 and the third entry in
  section 10).

With 80 or more responses per item the likelihood dominates all three; they matter only for
items that carry almost no information, which are exactly the items the broken-item report is
looking for.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from mselect.irt.model import Floats, Items, sigmoid

EPS = 1e-10


@dataclass(frozen=True, slots=True)
class Priors:
    """The weakly informative priors above. Stored with a fit so a report can state them."""

    a_mean: float = 1.0
    a_sd: float = 1.0
    d_sd: float = 4.0
    c_alpha: float = 2.0
    c_beta: float = 8.0

    def describe(self) -> str:
        return (
            f"a ~ Normal({self.a_mean}, {self.a_sd}), d ~ Normal(0, {self.d_sd}), "
            f"c ~ Beta({self.c_alpha}, {self.c_beta}) on multiple choice and 0 elsewhere"
        )


@dataclass(frozen=True, slots=True)
class Quadrature:
    """Fixed-node Gauss-style quadrature for the standard normal ability prior."""

    nodes: Floats
    weights: Floats

    @classmethod
    def normal(cls, points: int = 61, span: float = 4.5) -> Quadrature:
        nodes = np.linspace(-span, span, points)
        w = np.exp(-0.5 * nodes**2)
        return cls(nodes, w / w.sum())


@dataclass(frozen=True, slots=True)
class Fit:
    """A fitted bank: item parameters, their standard errors, and the abilities that fell out."""

    items: Items
    se_a: Floats
    se_b: Floats
    se_c: Floats
    theta: Floats
    theta_se: Floats
    responses_per_item: NDArray[np.int64]
    items_per_model: NDArray[np.int64]
    loglik: float
    log_posterior: float
    iterations: int
    converged: bool
    kind: str
    priors: Priors

    def describe(self) -> str:
        return (
            f"{self.kind.upper()} fit: {self.items.n_items} items, {self.theta.size} models, "
            f"{self.iterations} EM iterations, "
            f"{'converged' if self.converged else 'DID NOT CONVERGE'}, "
            f"marginal log likelihood {self.loglik:,.1f}"
        )


def masked(x: Floats) -> tuple[Floats, Floats]:
    """Split a response matrix with NaN holes into (responses with holes as 0, observed mask)."""
    obs = np.isfinite(x).astype(np.float64)
    x0 = np.where(np.isfinite(x), x, 0.0)
    if not np.all((x0 == 0.0) | (x0 == 1.0)):
        raise ValueError("responses must be 0, 1 or NaN")
    return x0, obs


def initial_items(x: Floats, *, mc: NDArray[np.bool_] | None = None, kind: str = "2pl") -> Items:
    """Classical starting values: difficulty from proportion correct, discrimination at 1."""
    x0, obs = masked(x)
    n = np.maximum(obs.sum(axis=0), 1.0)
    p = np.clip(x0.sum(axis=0) / n, 0.02, 0.98)
    b = -np.log(p / (1.0 - p))  # logistic difficulty implied by proportion correct
    a = np.ones_like(b)
    c = np.zeros_like(b)
    if kind == "3pl" and mc is not None:
        c = np.where(mc, 0.15, 0.0)
    return Items(a, np.clip(b, -6.0, 6.0), c)


def _posterior(
    x0: Floats, obs: Floats, a: Floats, d: Floats, c: Floats, quad: Quadrature
) -> tuple[Floats, float]:
    """E-step: posterior over the ability grid for every model, and the marginal log likelihood."""
    p = np.clip(_prob_at_nodes(a, d, c, quad.nodes), EPS, 1 - EPS)  # (J, Q)
    ll = x0 @ np.log(p) + (obs - x0) @ np.log1p(-p)  # (N, Q)
    ll += np.log(quad.weights)[None, :]
    peak = ll.max(axis=1, keepdims=True)
    w = np.exp(ll - peak)
    total = w.sum(axis=1, keepdims=True)
    loglik = float((np.log(total) + peak).sum())
    return w / total, loglik


def _prob_at_nodes(a: Floats, d: Floats, c: Floats, nodes: Floats) -> Floats:
    """(J, Q) response probabilities in the slope-intercept form the fitter works in."""
    s = sigmoid(a[:, None] * nodes[None, :] + d[:, None])
    return c[:, None] + (1.0 - c[:, None]) * s


def _log_prior(a: Floats, d: Floats, c: Floats, priors: Priors, mc: NDArray[np.bool_]) -> float:
    """The log prior the fit is penalised by, summed over items.

    EM here is monotone in the log posterior, not in the marginal likelihood on its own, so
    this is the quantity convergence is judged on. Reporting only the likelihood and watching
    it wobble downward by a few units near the optimum is a trap worth not falling into twice.
    """
    value = -0.5 * (((a - priors.a_mean) / priors.a_sd) ** 2).sum()
    value -= 0.5 * ((d / priors.d_sd) ** 2).sum()
    safe_c = np.clip(c, 1e-6, 1 - 1e-6)
    prior_c = (priors.c_alpha - 1.0) * np.log(safe_c) + (priors.c_beta - 1.0) * np.log1p(-safe_c)
    return float(value + np.where(mc, prior_c, 0.0).sum())


def _to_intercept(items: Items) -> tuple[Floats, Floats, Floats]:
    return items.a.copy(), -items.a * items.b, items.c.copy()


def _to_difficulty(a: Floats, d: Floats, c: Floats) -> Items:
    """b = -d/a, with the slope floored away from zero.

    An item with no discrimination has no meaningful difficulty, and its b runs to thousands.
    That is arithmetically harmless, because every later use multiplies it back by a, but it is
    nonsense to read: `flag_unidentified` marks those items so no report quotes the number.
    """
    floor = np.where(np.abs(a) > 1e-3, a, np.where(a >= 0, 1e-3, -1e-3))
    return Items(a, -d / floor, c)


def flag_unidentified(items: Items, *, min_a: float = 0.05) -> NDArray[np.bool_]:
    """Items whose difficulty is not identified because their slope is indistinguishable from 0."""
    return np.abs(items.a) < min_a


def _objective(
    a: Floats,
    d: Floats,
    c: Floats,
    n: Floats,
    r: Floats,
    nodes: Floats,
    priors: Priors,
    mc: NDArray[np.bool_],
) -> Floats:
    """The M-step objective per item: expected complete-data log likelihood plus log prior."""
    s = sigmoid(a[:, None] * nodes[None, :] + d[:, None])
    p = np.clip(c[:, None] + (1.0 - c[:, None]) * s, EPS, 1 - EPS)
    value = (r * np.log(p) + (n - r) * np.log1p(-p)).sum(axis=1)
    value -= 0.5 * ((a - priors.a_mean) / priors.a_sd) ** 2
    value -= 0.5 * (d / priors.d_sd) ** 2
    safe_c = np.clip(c, 1e-6, 1 - 1e-6)
    prior_c = (priors.c_alpha - 1.0) * np.log(safe_c) + (priors.c_beta - 1.0) * np.log1p(-safe_c)
    return value + np.where(mc, prior_c, 0.0)


def _accept(
    current: tuple[Floats, Floats, Floats],
    step: tuple[Floats, Floats, Floats],
    bounds: tuple[tuple[float, float], ...],
    n: Floats,
    r: Floats,
    nodes: Floats,
    priors: Priors,
    mc: NDArray[np.bool_],
    *,
    halvings: int = 10,
) -> tuple[Floats, Floats, Floats]:
    """Take the largest fraction of the proposed step that does not make any item worse.

    EM only converges if every M-step increases the objective. Fisher scoring on its own
    overshoots badly on items that every model answered the same way, so each item keeps its
    own step size and halves it until its own objective stops falling.
    """
    a, d, c = current
    base = _objective(a, d, c, n, r, nodes, priors, mc)
    scale = np.ones_like(a)
    best = (a.copy(), d.copy(), c.copy())
    pending = np.ones_like(a, dtype=bool)
    for _ in range(halvings):
        if not pending.any():
            break
        trial_a = np.clip(a + scale * step[0], *bounds[0])
        trial_d = np.clip(d + scale * step[1], *bounds[1])
        trial_c = np.clip(c + scale * step[2], *bounds[2])
        trial_c = np.where(mc, trial_c, 0.0)
        value = _objective(trial_a, trial_d, trial_c, n, r, nodes, priors, mc)
        better = pending & (value >= base - 1e-9)
        best[0][better] = trial_a[better]
        best[1][better] = trial_d[better]
        best[2][better] = trial_c[better]
        pending &= ~better
        scale = np.where(pending, scale * 0.5, scale)
    return best


def _m_step(
    current: tuple[Floats, Floats, Floats],
    n: Floats,
    r: Floats,
    nodes: Floats,
    priors: Priors,
    mc: NDArray[np.bool_],
    kind: str,
    newton_steps: int = 2,
) -> tuple[Floats, Floats, Floats]:
    """Fisher scoring on (a, d) for every item at once, then a Newton step on c for the 3PL.

    Working in the slope-intercept form z = a*theta + d keeps the 2PL M-step a weighted
    logistic regression; b is recovered as -d/a once the fit has converged.
    """
    a, d, c = (array.copy() for array in current)
    zero = np.zeros_like(a)
    bounds = ((-6.0, 6.0), (-30.0, 30.0), (0.0, 0.45))

    for _ in range(newton_steps):
        s = sigmoid(a[:, None] * nodes[None, :] + d[:, None])
        p = np.clip(c[:, None] + (1.0 - c[:, None]) * s, EPS, 1 - EPS)
        dp_dz = (1.0 - c[:, None]) * s * (1.0 - s)
        var = p * (1.0 - p)
        score = (r - n * p) / var * dp_dz  # dL/dz at each node
        w = n * dp_dz**2 / var  # Fisher weight

        g_a = (score * nodes[None, :]).sum(axis=1) - (a - priors.a_mean) / priors.a_sd**2
        g_d = score.sum(axis=1) - d / priors.d_sd**2
        h_aa = (w * nodes[None, :] ** 2).sum(axis=1) + 1.0 / priors.a_sd**2
        h_ad = (w * nodes[None, :]).sum(axis=1)
        h_dd = w.sum(axis=1) + 1.0 / priors.d_sd**2

        det = np.maximum(h_aa * h_dd - h_ad**2, EPS)
        step_a = np.clip((h_dd * g_a - h_ad * g_d) / det, -2.0, 2.0)
        step_d = np.clip((h_aa * g_d - h_ad * g_a) / det, -4.0, 4.0)
        a, d, c = _accept((a, d, c), (step_a, step_d, zero), bounds, n, r, nodes, priors, mc)

        if kind == "3pl" and mc.any():
            s = sigmoid(a[:, None] * nodes[None, :] + d[:, None])
            p = np.clip(c[:, None] + (1.0 - c[:, None]) * s, EPS, 1 - EPS)
            dp_dc = 1.0 - s
            var = p * (1.0 - p)
            g_c = ((r - n * p) / var * dp_dc).sum(axis=1)
            h_c = (n * dp_dc**2 / var).sum(axis=1)
            safe_c = np.clip(c, 1e-3, 0.45)
            g_c += (priors.c_alpha - 1.0) / safe_c - (priors.c_beta - 1.0) / (1.0 - safe_c)
            h_c += (priors.c_alpha - 1.0) / safe_c**2 + (priors.c_beta - 1.0) / (1.0 - safe_c) ** 2
            step_c = np.where(mc, np.clip(g_c / np.maximum(h_c, EPS), -0.1, 0.1), 0.0)
            a, d, c = _accept((a, d, c), (zero, zero, step_c), bounds, n, r, nodes, priors, mc)

    return a, d, c


def _item_standard_errors(
    a: Floats,
    d: Floats,
    c: Floats,
    n: Floats,
    nodes: Floats,
    priors: Priors,
    mc: NDArray[np.bool_],
) -> tuple[Floats, Floats, Floats]:
    """Posterior standard deviations from the curvature at the solution, by the delta method.

    These are the usual MML standard errors: they condition on the expected counts from the
    final E-step and so ignore the uncertainty in ability, which makes them a little optimistic.
    `bootstrap_items` resamples models to show by how much.
    """
    s = sigmoid(a[:, None] * nodes[None, :] + d[:, None])
    p = np.clip(c[:, None] + (1.0 - c[:, None]) * s, EPS, 1 - EPS)
    dp_dz = (1.0 - c[:, None]) * s * (1.0 - s)
    var = p * (1.0 - p)
    w = n * dp_dz**2 / var

    h_aa = (w * nodes[None, :] ** 2).sum(axis=1) + 1.0 / priors.a_sd**2
    h_ad = (w * nodes[None, :]).sum(axis=1)
    h_dd = w.sum(axis=1) + 1.0 / priors.d_sd**2
    det = np.maximum(h_aa * h_dd - h_ad**2, EPS)
    var_a = h_dd / det
    var_d = h_aa / det
    cov_ad = -h_ad / det

    a_safe = np.where(np.abs(a) < 1e-3, np.sign(a) * 1e-3 + 1e-12, a)
    db_da = d / a_safe**2
    db_dd = -1.0 / a_safe
    var_b = db_da**2 * var_a + db_dd**2 * var_d + 2.0 * db_da * db_dd * cov_ad

    dp_dc = 1.0 - s
    h_c = (n * dp_dc**2 / var).sum(axis=1)
    safe_c = np.clip(c, 1e-3, 0.45)
    h_c += (priors.c_alpha - 1.0) / safe_c**2 + (priors.c_beta - 1.0) / (1.0 - safe_c) ** 2
    se_c = np.where(mc, 1.0 / np.sqrt(np.maximum(h_c, EPS)), 0.0)

    return np.sqrt(np.maximum(var_a, 0.0)), np.sqrt(np.maximum(var_b, 0.0)), se_c


def fit_mml(
    x: Floats,
    *,
    kind: str = "2pl",
    mc: NDArray[np.bool_] | None = None,
    quad: Quadrature | None = None,
    priors: Priors | None = None,
    max_iter: int = 200,
    tol: float = 1e-4,
    start: Items | None = None,
    progress: Callable[[str], None] | None = None,
) -> Fit:
    """Fit item parameters to a response matrix with NaN for "this model never saw this item".

    The ability scale is fixed by the standard normal prior on theta, which is the only
    identification constraint the 2PL needs.
    """
    if kind not in {"2pl", "3pl"}:
        raise ValueError("kind must be '2pl' or '3pl'")
    x0, obs = masked(x)
    n_items = x.shape[1]
    if mc is None:
        mc = np.zeros(n_items, dtype=bool)
    if kind == "2pl":
        mc = np.zeros(n_items, dtype=bool)
    quad = quad or Quadrature.normal()
    priors = priors or Priors()
    a, d, c = _to_intercept(start or initial_items(x, mc=mc, kind=kind))

    loglik = -np.inf
    converged = False
    iterations = 0

    previous = -np.inf
    posterior_value = -np.inf
    for iteration in range(1, max_iter + 1):
        iterations = iteration
        posterior, loglik = _posterior(x0, obs, a, d, c, quad)
        posterior_value = loglik + _log_prior(a, d, c, priors, mc)
        counts_n = obs.T @ posterior
        counts_r = x0.T @ posterior
        new_a, new_d, new_c = _m_step((a, d, c), counts_n, counts_r, quad.nodes, priors, mc, kind)
        # Convergence is judged on (a, d, c). Difficulty is a ratio and swings wildly for an
        # item whose discrimination is near zero, so judging on b would hide a converged fit
        # behind one dead item.
        shift = float(
            max(
                np.abs(new_a - a).max(),
                np.abs(new_d - d).max(),
                np.abs(new_c - c).max(),
            )
        )
        gain = posterior_value - previous
        previous = posterior_value
        a, d, c = new_a, new_d, new_c
        if progress and iteration % 25 == 0:
            progress(
                f"  EM {iteration}: log posterior {posterior_value:,.1f} (+{gain:,.3f}), "
                f"max shift {shift:.5f}"
            )
        if shift < tol or (0.0 <= gain < 1e-8 * abs(posterior_value)):
            converged = True
            break

    posterior, loglik = _posterior(x0, obs, a, d, c, quad)
    posterior_value = loglik + _log_prior(a, d, c, priors, mc)
    counts_n = obs.T @ posterior
    theta = posterior @ quad.nodes
    theta_se = np.sqrt(np.maximum(posterior @ quad.nodes**2 - theta**2, 0.0))
    se_a, se_b, se_c = _item_standard_errors(a, d, c, counts_n, quad.nodes, priors, mc)

    return Fit(
        items=_to_difficulty(a, d, c),
        se_a=se_a,
        se_b=se_b,
        se_c=se_c,
        theta=theta,
        theta_se=theta_se,
        responses_per_item=obs.sum(axis=0).astype(np.int64),
        items_per_model=obs.sum(axis=1).astype(np.int64),
        loglik=loglik,
        log_posterior=posterior_value,
        iterations=iterations,
        converged=converged,
        kind=kind,
        priors=priors,
    )


def eap(x: Floats, items: Items, quad: Quadrature | None = None) -> tuple[Floats, Floats]:
    """Expected a posteriori ability and its posterior standard deviation, item parameters fixed.

    This is how a model that was never in the calibration is scored, which is the whole point
    of freezing a bank.
    """
    quad = quad or Quadrature.normal()
    x0, obs = masked(np.atleast_2d(x))
    a, d, c = _to_intercept(items)
    posterior, _ = _posterior(x0, obs, a, d, c, quad)
    theta = posterior @ quad.nodes
    se = np.sqrt(np.maximum(posterior @ quad.nodes**2 - theta**2, 0.0))
    return theta, se


def bootstrap_items(
    x: Floats,
    *,
    kind: str = "2pl",
    mc: NDArray[np.bool_] | None = None,
    index: NDArray[np.intp] | None = None,
    draws: int = 200,
    seed: int = 0,
    priors: Priors | None = None,
    max_iter: int = 60,
) -> dict[str, Floats]:
    """Nonparametric bootstrap over models for a subset of items.

    Resampling models, not responses, is the right unit: the uncertainty being quantified is
    "would another panel of models have given these item parameters". Returns the 2.5th, 50th
    and 97.5th percentiles of `a` and `b` for the requested items.
    """
    rng = np.random.default_rng(seed)
    n_models = x.shape[0]
    index = np.arange(x.shape[1], dtype=np.intp) if index is None else index
    a_draws = np.empty((draws, index.size))
    b_draws = np.empty((draws, index.size))
    for draw in range(draws):
        rows = rng.integers(0, n_models, n_models)
        fit = fit_mml(x[rows], kind=kind, mc=mc, priors=priors, max_iter=max_iter)
        a_draws[draw] = fit.items.a[index]
        b_draws[draw] = fit.items.b[index]
    quantiles = (2.5, 50.0, 97.5)
    return {
        "a_lo": np.percentile(a_draws, quantiles[0], axis=0),
        "a_mid": np.percentile(a_draws, quantiles[1], axis=0),
        "a_hi": np.percentile(a_draws, quantiles[2], axis=0),
        "b_lo": np.percentile(b_draws, quantiles[0], axis=0),
        "b_mid": np.percentile(b_draws, quantiles[1], axis=0),
        "b_hi": np.percentile(b_draws, quantiles[2], axis=0),
    }


def refit_without(
    x: Floats,
    row: int,
    *,
    kind: str = "2pl",
    mc: NDArray[np.bool_] | None = None,
    start: Fit | None = None,
    max_iter: int = 40,
) -> Fit:
    """Refit the bank with one model held out, warm-started from the full fit.

    The leave-one-model-out simulation (PLAN.md section 4.2) needs the held-out model to have
    had no influence on the item parameters it is then scored with.
    """
    keep = np.ones(x.shape[0], dtype=bool)
    keep[row] = False
    return fit_mml(
        x[keep],
        kind=kind,
        mc=mc,
        start=replace(start.items) if start is not None else None,
        max_iter=max_iter,
    )
