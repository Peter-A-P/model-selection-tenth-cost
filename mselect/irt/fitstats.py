"""Item fit: infit and outfit, the empirical response curve, and the flags that make an item
a candidate for `docs/items-that-measure-nothing.md`.

PLAN.md section 4.1: "Items flagged when discrimination is below 0.3, when fit statistics fall
outside the usual bounds, or when the empirical response curve is non-monotone (the mis-keyed
signature)."
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from mselect.irt.fit import masked
from mselect.irt.model import Floats, Items, information, prob

LOW_DISCRIMINATION = 0.3  # PLAN.md section 4.1
MSQ_LOW, MSQ_HIGH = 0.7, 1.3  # the usual productive-for-measurement bounds
CURVE_BINS = 5


@dataclass(frozen=True, slots=True)
class ItemFit:
    """Per-item fit statistics, aligned with the bank's item order."""

    infit: Floats
    outfit: Floats
    point_biserial: Floats
    proportion_correct: Floats
    n_responses: NDArray[np.int64]
    curve: Floats  # (n_items, CURVE_BINS) proportion correct by ability bin
    curve_n: NDArray[np.int64]
    monotone_drop: Floats  # largest fall between consecutive ability bins
    information_at_zero: Floats
    separation_auc: Floats  # 1.0 when ability orders the responses perfectly


def item_fit(x: Floats, items: Items, theta: Floats, *, bins: int = CURVE_BINS) -> ItemFit:
    """Infit and outfit mean squares, the classical statistics, and the empirical curve.

    Outfit is the plain mean of squared standardised residuals, so a single surprising response
    from a far-away model moves it; infit weights by the information each response carries, so
    it is the one to read for items in the middle of the range.
    """
    x0, obs = masked(x)
    p = np.clip(prob(theta, items), 1e-6, 1 - 1e-6)
    variance = p * (1.0 - p)
    residual = (x0 - p) * obs
    z2 = np.where(obs > 0, residual**2 / variance, 0.0)

    n = obs.sum(axis=0)
    outfit = z2.sum(axis=0) / np.maximum(n, 1.0)
    infit = residual.__pow__(2).sum(axis=0) / np.maximum((variance * obs).sum(axis=0), 1e-12)

    proportion = np.where(n > 0, (x0 * obs).sum(axis=0) / np.maximum(n, 1.0), np.nan)
    biserial = _point_biserial(x0, obs, theta)
    curve, curve_n = empirical_curve(x, theta, bins=bins)
    auc = separation_auc(x, theta)
    drops = np.nanmax(np.diff(curve, axis=1) * -1.0, axis=1, initial=0.0)
    info_zero = information(np.zeros(1), items)[0]

    return ItemFit(
        infit=infit,
        outfit=outfit,
        point_biserial=biserial,
        proportion_correct=proportion,
        n_responses=n.astype(np.int64),
        curve=curve,
        curve_n=curve_n,
        monotone_drop=drops,
        information_at_zero=info_zero,
        separation_auc=auc,
    )


def separation_auc(x: Floats, theta: Floats) -> Floats:
    """Per item, the probability that a randomly chosen correct model outranks an incorrect one.

    This is the Mann-Whitney statistic, and it matters for reading the fitted slopes. An item
    with an AUC of exactly 1 is perfectly separated by ability: no model below the cut got it
    right and none above got it wrong. For such an item the likelihood has no maximum in `a`
    (larger is always better), so the value that comes out of the fit is set by the prior and
    the sample size, not by the data. That is why the discrimination histogram has a spike:
    those items all land on the same prior-determined value whatever their difficulty. Their
    fitted slope should be read as a lower bound.
    """
    x0, obs = masked(x)
    order = np.argsort(theta)
    correct = (x0 * obs)[order]
    incorrect = ((1.0 - x0) * obs)[order]
    # For each position, how many incorrect responses sit below it in ability. A correct answer
    # from a model that outranks an incorrect one is a concordant pair, which is the direction
    # the statistic counts.
    below = np.cumsum(incorrect, axis=0) - incorrect
    concordant = (correct * below).sum(axis=0)
    pairs = correct.sum(axis=0) * incorrect.sum(axis=0)
    return np.where(pairs > 0, concordant / np.maximum(pairs, 1.0), np.nan)


def _point_biserial(x0: Floats, obs: Floats, theta: Floats) -> Floats:
    """Correlation between the response and ability, over the models that answered the item."""
    n = obs.sum(axis=0)
    theta_col = theta[:, None]
    mean_theta = (theta_col * obs).sum(axis=0) / np.maximum(n, 1.0)
    mean_x = (x0 * obs).sum(axis=0) / np.maximum(n, 1.0)
    dt = (theta_col - mean_theta) * obs
    dx = (x0 - mean_x) * obs
    cov = (dt * dx).sum(axis=0)
    denom = np.sqrt((dt**2).sum(axis=0) * (dx**2).sum(axis=0))
    return np.where(denom > 0, cov / np.maximum(denom, 1e-12), np.nan)


def empirical_curve(
    x: Floats, theta: Floats, *, bins: int = CURVE_BINS
) -> tuple[Floats, NDArray[np.int64]]:
    """Proportion correct by ability bin: the picture that shows a mis-keyed item."""
    x0, obs = masked(x)
    edges = np.quantile(theta, np.linspace(0.0, 1.0, bins + 1))
    edges[0] -= 1e-9
    which = np.clip(np.searchsorted(edges, theta, side="left") - 1, 0, bins - 1)
    curve = np.full((x.shape[1], bins), np.nan)
    counts = np.zeros((x.shape[1], bins), dtype=np.int64)
    for b in range(bins):
        rows = which == b
        if not rows.any():
            continue
        n = obs[rows].sum(axis=0)
        correct = (x0[rows] * obs[rows]).sum(axis=0)
        counts[:, b] = n.astype(np.int64)
        curve[:, b] = np.where(n > 0, correct / np.maximum(n, 1.0), np.nan)
    return curve, counts


def flags(fit: ItemFit, items: Items, *, min_curve_n: int = 5) -> dict[str, NDArray[np.bool_]]:
    """The boolean columns that drive the broken-item report.

    `non_monotone` only fires where every ability bin has enough responses to mean anything,
    because a bin of two models can fall by half for no reason at all.
    """
    enough = fit.curve_n.min(axis=1) >= min_curve_n
    return {
        "perfect_separation": np.isfinite(fit.separation_auc) & (fit.separation_auc >= 1.0),
        "negative_discrimination": items.a < 0.0,
        "low_discrimination": (items.a >= 0.0) & (items.a < LOW_DISCRIMINATION),
        "misfit_high": fit.outfit > MSQ_HIGH,
        "misfit_low": fit.outfit < MSQ_LOW,
        "non_monotone": enough & (fit.monotone_drop > 0.15),
        "everyone_right": fit.proportion_correct > 0.99,
        "everyone_wrong": fit.proportion_correct < 0.01,
        "no_information": fit.information_at_zero < 0.01,
    }
