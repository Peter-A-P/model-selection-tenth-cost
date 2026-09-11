"""Yen's Q3: local dependence between items, measured rather than assumed away.

PLAN.md section 4.1 and the risk table: local independence is the assumption every standard
error in this project rests on, benchmark items violate it, and the honest thing is to report
the violation with a number. Q3 is the correlation between item residuals after ability is
partialled out. Under local independence it has a small negative expectation, about
-1/(n_items - 1), and that bias is reported next to the statistic rather than corrected away.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from mselect.irt.fit import masked
from mselect.irt.model import Floats, Items, prob

FLAG = 0.2  # PLAN.md section 4.1: report the pairs above this


@dataclass(frozen=True, slots=True)
class Q3Result:
    """Q3 over one set of items, and the pairs worth naming."""

    index: NDArray[np.intp]  # which items, as bank positions
    matrix: Floats  # (k, k) residual correlations, diagonal set to nan
    n_models: int
    expected_bias: float
    pairs: list[tuple[int, int, float]]  # bank positions and Q3, above FLAG, worst first

    @property
    def values(self) -> Floats:
        upper = np.triu_indices(self.matrix.shape[0], k=1)
        return self.matrix[upper]

    def summary(self) -> dict[str, float]:
        v = self.values
        v = v[np.isfinite(v)]
        if v.size == 0:
            return {"n_pairs": 0.0}
        return {
            "n_pairs": float(v.size),
            "mean": float(v.mean()),
            "expected_under_independence": self.expected_bias,
            "sd": float(v.std()),
            "p95": float(np.quantile(v, 0.95)),
            "max": float(v.max()),
            "share_above_flag": float((v > FLAG).mean()),
        }


def residuals(x: Floats, items: Items, theta: Floats) -> Floats:
    """Standardised residuals, with NaN where the model never saw the item."""
    x0, obs = masked(x)
    p = np.clip(prob(theta, items), 1e-6, 1 - 1e-6)
    z = (x0 - p) / np.sqrt(p * (1.0 - p))
    return np.where(obs > 0, z, np.nan)


def q3(
    x: Floats,
    items: Items,
    theta: Floats,
    index: NDArray[np.intp],
    *,
    min_models: int = 20,
    flag: float = FLAG,
    max_pairs: int = 200,
) -> Q3Result:
    """Q3 over the selected items, using the models that answered all of them.

    Complete cases, not pairwise deletion: a correlation computed on a different subset of
    models for every pair is not comparable across pairs, and the point here is a distribution.
    """
    z = residuals(x, items, theta)[:, index]
    complete = np.isfinite(z).all(axis=1)
    z = z[complete]
    if z.shape[0] < min_models:
        raise ValueError(
            f"only {z.shape[0]} models answered all {index.size} items; need {min_models}"
        )
    centred = z - z.mean(axis=0, keepdims=True)
    sd = centred.std(axis=0)
    safe = np.where(sd > 1e-9, sd, np.nan)
    corr = (centred.T @ centred) / z.shape[0] / np.outer(safe, safe)
    np.fill_diagonal(corr, np.nan)

    upper = np.triu_indices(corr.shape[0], k=1)
    values = corr[upper]
    order = np.argsort(-np.nan_to_num(values, nan=-np.inf))[:max_pairs]
    pairs = [
        (int(index[upper[0][k]]), int(index[upper[1][k]]), float(values[k]))
        for k in order
        if np.isfinite(values[k]) and values[k] > flag
    ]
    return Q3Result(
        index=index,
        matrix=corr,
        n_models=int(z.shape[0]),
        expected_bias=-1.0 / max(index.size - 1, 1),
        pairs=pairs,
    )


def sample_items(
    benchmarks: NDArray[np.str_],
    benchmark: str,
    *,
    size: int,
    seed: int = 0,
) -> NDArray[np.intp]:
    """A reproducible sample of one benchmark's items, because Q3 is quadratic in item count."""
    rng = np.random.default_rng(seed)
    pool = np.flatnonzero(benchmarks == benchmark).astype(np.intp)
    if pool.size <= size:
        return pool
    return np.sort(rng.choice(pool, size=size, replace=False))
