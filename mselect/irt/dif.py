"""Differential item functioning: does an item behave differently for two kinds of model at
the same ability?

PLAN.md section 4.1 asks for two groupings. Open weights against API-only models says whether
the bank is fair across the panel. Models released before against after an item's benchmark was
published is the contamination probe: an item that is much easier for later models than their
ability predicts is a candidate for having leaked into training data.

Two tests, because they answer different questions:

* Mantel-Haenszel, stratified on ability, gives an odds ratio and the ETS delta scale that
  testing organisations use to triage items (|delta| above 1.5 is "large" by that convention).
* A logistic regression with an ability-by-group interaction separates uniform DIF (the item
  is simply easier for one group) from non-uniform DIF (it discriminates differently), which
  Mantel-Haenszel cannot see.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from mselect.irt.fit import masked
from mselect.irt.model import Floats, sigmoid

ETS_LARGE = 1.5  # |delta| above this is the "C" category in the ETS scheme
STRATA = 5


@dataclass(frozen=True, slots=True)
class DifResult:
    """One grouping, tested item by item. Arrays are aligned with the bank's item order."""

    grouping: str
    reference: str
    focal: str
    n_reference: NDArray[np.int64]
    n_focal: NDArray[np.int64]
    odds_ratio: Floats
    ets_delta: Floats
    mh_chi2: Floats
    mh_p: Floats
    uniform_chi2: Floats
    uniform_p: Floats
    nonuniform_chi2: Floats
    nonuniform_p: Floats

    def flagged(self, *, alpha: float = 0.01, delta: float = ETS_LARGE) -> NDArray[np.bool_]:
        """Large and significant: both, because with 14,000 items either alone is noise."""
        return (np.abs(self.ets_delta) > delta) & (self.mh_p < alpha) & np.isfinite(self.ets_delta)

    def summary(self, *, alpha: float = 0.01) -> dict[str, float]:
        testable = np.isfinite(self.ets_delta)
        flagged = self.flagged(alpha=alpha)
        favours_focal = flagged & (self.ets_delta > 0)
        return {
            "items_testable": float(testable.sum()),
            "items_flagged": float(flagged.sum()),
            "share_flagged": float(flagged.sum() / max(testable.sum(), 1)),
            "flagged_favouring_focal": float(favours_focal.sum()),
            "median_abs_delta": float(np.nanmedian(np.abs(self.ets_delta[testable]))),
            "uniform_significant": float((self.uniform_p < alpha).sum()),
            "nonuniform_significant": float((self.nonuniform_p < alpha).sum()),
        }


def mantel_haenszel(
    x: Floats,
    theta: Floats,
    group: NDArray[np.bool_],
    *,
    strata: int = STRATA,
    min_per_cell: int = 2,
) -> tuple[Floats, Floats, Floats, Floats, NDArray[np.int64], NDArray[np.int64]]:
    """Odds ratio, ETS delta, chi-square and p per item, matching on ability.

    `group` is True for the focal group. Ability is the matching variable rather than total
    score, which is what makes this work on a matrix where models answered different items.
    """
    x0, obs = masked(x)
    edges = np.quantile(theta, np.linspace(0.0, 1.0, strata + 1))
    edges[0] -= 1e-9
    which = np.clip(np.searchsorted(edges, theta, side="left") - 1, 0, strata - 1)

    num = np.zeros(x.shape[1])
    den = np.zeros(x.shape[1])
    expected = np.zeros(x.shape[1])
    variance = np.zeros(x.shape[1])
    observed = np.zeros(x.shape[1])
    usable = np.zeros(x.shape[1], dtype=bool)

    for stratum in range(strata):
        rows = which == stratum
        ref = rows & ~group
        foc = rows & group
        if not ref.any() or not foc.any():
            continue
        # The ETS convention: the odds ratio is reference over focal, so a positive delta means
        # the item favours the focal group. Getting this the wrong way round would invert every
        # contamination conclusion, so the naming follows the textbook 2x2 table exactly.
        a = (x0[ref] * obs[ref]).sum(axis=0)  # reference correct
        b = ((1.0 - x0[ref]) * obs[ref]).sum(axis=0)  # reference incorrect
        c = (x0[foc] * obs[foc]).sum(axis=0)  # focal correct
        d = ((1.0 - x0[foc]) * obs[foc]).sum(axis=0)  # focal incorrect
        n = a + b + c + d
        ok = (n > 0) & ((a + b) >= min_per_cell) & ((c + d) >= min_per_cell)
        safe_n = np.where(n > 0, n, 1.0)
        num += np.where(ok, a * d / safe_n, 0.0)
        den += np.where(ok, b * c / safe_n, 0.0)
        row1, row2 = a + b, c + d  # reference total, focal total
        col1, col2 = a + c, b + d  # correct total, incorrect total
        expected += np.where(ok, row1 * col1 / safe_n, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            var = row1 * row2 * col1 * col2 / (safe_n**2 * np.maximum(safe_n - 1.0, 1.0))
        variance += np.where(ok, var, 0.0)
        observed += np.where(ok, a, 0.0)
        usable |= ok

    with np.errstate(divide="ignore", invalid="ignore"):
        odds = np.where((den > 0) & (num > 0), num / den, np.nan)
        delta = -2.35 * np.log(odds)
        chi2 = np.where(
            variance > 0,
            (np.abs(observed - expected) - 0.5) ** 2 / np.maximum(variance, 1e-12),
            np.nan,
        )
    chi2 = np.where(usable, chi2, np.nan)
    p = np.where(np.isfinite(chi2), stats.chi2.sf(np.nan_to_num(chi2, nan=0.0), df=1), np.nan)

    n_focal = (obs[group] > 0).sum(axis=0).astype(np.int64)
    n_reference = (obs[~group] > 0).sum(axis=0).astype(np.int64)
    return odds, delta, chi2, p, n_reference, n_focal


def _irls(design: Floats, y: Floats, *, iterations: int = 30) -> tuple[Floats, float]:
    """Ridge-stabilised logistic regression. Returns coefficients and the log likelihood."""
    beta = np.zeros(design.shape[1])
    ridge = 1e-4 * np.eye(design.shape[1])
    for _ in range(iterations):
        p = np.clip(sigmoid(design @ beta), 1e-9, 1 - 1e-9)
        w = p * (1.0 - p)
        gradient = design.T @ (y - p) - ridge @ beta
        hessian = design.T @ (design * w[:, None]) + ridge
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:  # pragma: no cover - singular only on degenerate items
            break
        beta = beta + np.clip(step, -4.0, 4.0)
        if np.abs(step).max() < 1e-7:
            break
    p = np.clip(sigmoid(design @ beta), 1e-9, 1 - 1e-9)
    loglik = float((y * np.log(p) + (1.0 - y) * np.log1p(-p)).sum())
    return beta, loglik


def logistic_dif(
    x: Floats, theta: Floats, group: NDArray[np.bool_], *, min_models: int = 20
) -> tuple[Floats, Floats, Floats, Floats]:
    """Likelihood-ratio tests for uniform and non-uniform DIF, item by item."""
    n_items = x.shape[1]
    uniform_chi2 = np.full(n_items, np.nan)
    nonuniform_chi2 = np.full(n_items, np.nan)
    g = group.astype(float)
    for j in range(n_items):
        column = x[:, j]
        rows = np.isfinite(column)
        if rows.sum() < min_models:
            continue
        y = column[rows]
        if y.min() == y.max():
            continue
        gj = g[rows]
        if gj.min() == gj.max():
            continue
        t = theta[rows]
        base = np.column_stack([np.ones_like(t), t])
        _, ll_base = _irls(base, y)
        with_group = np.column_stack([base, gj])
        _, ll_group = _irls(with_group, y)
        full = np.column_stack([with_group, t * gj])
        _, ll_full = _irls(full, y)
        uniform_chi2[j] = max(2.0 * (ll_group - ll_base), 0.0)
        nonuniform_chi2[j] = max(2.0 * (ll_full - ll_group), 0.0)
    uniform_p = np.where(
        np.isfinite(uniform_chi2), stats.chi2.sf(np.nan_to_num(uniform_chi2), 1), np.nan
    )
    nonuniform_p = np.where(
        np.isfinite(nonuniform_chi2), stats.chi2.sf(np.nan_to_num(nonuniform_chi2), 1), np.nan
    )
    return uniform_chi2, uniform_p, nonuniform_chi2, nonuniform_p


def run_dif(
    x: Floats,
    theta: Floats,
    group: NDArray[np.bool_],
    *,
    grouping: str,
    reference: str,
    focal: str,
    strata: int = STRATA,
    logistic: bool = True,
) -> DifResult:
    """Both tests for one grouping, packaged for the report."""
    odds, delta, chi2, p, n_ref, n_foc = mantel_haenszel(x, theta, group, strata=strata)
    if logistic:
        u_chi2, u_p, n_chi2, n_p = logistic_dif(x, theta, group)
    else:
        empty = np.full(x.shape[1], np.nan)
        u_chi2, u_p, n_chi2, n_p = empty, empty, empty, empty
    return DifResult(
        grouping=grouping,
        reference=reference,
        focal=focal,
        n_reference=n_ref,
        n_focal=n_foc,
        odds_ratio=odds,
        ets_delta=delta,
        mh_chi2=chi2,
        mh_p=p,
        uniform_chi2=u_chi2,
        uniform_p=u_p,
        nonuniform_chi2=n_chi2,
        nonuniform_p=n_p,
    )
