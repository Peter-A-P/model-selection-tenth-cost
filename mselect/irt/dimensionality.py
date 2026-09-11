"""Is the bank measuring one thing?

PLAN.md section 4.1: eigenvalues of the tetrachoric correlation matrix, parallel analysis, and
a comparison between a single ability and one ability per benchmark. If the benchmarks are not
one dimension, the composite is reported next to the per-benchmark abilities and the write-up
says so, rather than quietly reporting a single number.

Tetrachoric correlations use the Digby approximation,
rho ~= (u - 1) / (u + 1) with u = (ad/bc)^(3/4), which is within about 0.02 of the exact
maximum likelihood value across the range that matters here and costs one multiplication
instead of a two-dimensional numerical integral per pair. The exact version is available for
checking a sample of pairs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats

from mselect.irt.model import Floats


@dataclass(frozen=True, slots=True)
class Dimensionality:
    """Eigenvalues of the item correlation matrix against a parallel-analysis reference."""

    eigenvalues: Floats
    reference_p95: Floats  # 95th percentile of eigenvalues from column-permuted data
    n_above_reference: int
    variance_first: float
    variance_second: float
    ratio_first_to_second: float
    n_models: int
    n_items: int

    def summary(self) -> dict[str, float]:
        return {
            "n_items": float(self.n_items),
            "n_models": float(self.n_models),
            "first_eigenvalue_share": self.variance_first,
            "second_eigenvalue_share": self.variance_second,
            "first_to_second_ratio": self.ratio_first_to_second,
            "factors_above_parallel_reference": float(self.n_above_reference),
        }


def tetrachoric_matrix(x: Floats, *, min_models: int = 20) -> tuple[Floats, int]:
    """Tetrachoric correlations over complete cases, with the Digby approximation."""
    complete = np.isfinite(x).all(axis=1)
    data = x[complete]
    if data.shape[0] < min_models:
        raise ValueError(f"only {data.shape[0]} models answered every selected item")
    n = data.shape[0]
    ones = data
    zeros = 1.0 - data
    # a: both correct, d: both wrong, b and c: one each. 0.5 is Yates' continuity correction,
    # which also keeps the ratio finite when a cell is empty.
    both = ones.T @ ones + 0.5
    neither = zeros.T @ zeros + 0.5
    one_only = ones.T @ zeros + 0.5
    other_only = zeros.T @ ones + 0.5
    u = ((both * neither) / (one_only * other_only)) ** 0.75
    rho = (u - 1.0) / (u + 1.0)
    np.fill_diagonal(rho, 1.0)
    return np.clip(rho, -0.999, 0.999), n


def tetrachoric_exact(counts: tuple[float, float, float, float]) -> float:
    """Maximum likelihood tetrachoric for one pair, for checking the approximation."""
    both, one_only, other_only, neither = counts
    total = both + one_only + other_only + neither
    p1 = (both + one_only) / total
    p2 = (both + other_only) / total
    h1, h2 = (
        stats.norm.ppf(np.clip(p1, 1e-6, 1 - 1e-6)),
        stats.norm.ppf(np.clip(p2, 1e-6, 1 - 1e-6)),
    )

    def joint(rho: float) -> float:
        return float(
            stats.multivariate_normal.cdf(
                [h1, h2], mean=[0.0, 0.0], cov=[[1.0, rho], [rho, 1.0]], lower_limit=[-10, -10]
            )
        )

    target = both / total

    def gap(rho: float) -> float:
        return joint(rho) - target

    lo, hi = -0.999, 0.999
    if gap(lo) * gap(hi) > 0:
        return float(np.sign(gap(hi)) * 0.999)
    return float(optimize.brentq(gap, lo, hi, xtol=1e-4))


def parallel_analysis(
    x: Floats, *, draws: int = 25, seed: int = 0, min_models: int = 20
) -> Dimensionality:
    """Eigenvalues against a reference built by permuting each item independently.

    Permuting within a column keeps every item's difficulty and every sample size exactly as
    observed while destroying the correlation between items, which is the right null for
    "how large would this eigenvalue be if the bank measured nothing in common".
    """
    rho, n_models = tetrachoric_matrix(x, min_models=min_models)
    eigenvalues = np.sort(np.linalg.eigvalsh(rho))[::-1]

    rng = np.random.default_rng(seed)
    complete = x[np.isfinite(x).all(axis=1)]
    reference = np.empty((draws, eigenvalues.size))
    for draw in range(draws):
        shuffled = np.column_stack([rng.permutation(col) for col in complete.T])
        ref_rho, _ = tetrachoric_matrix(shuffled, min_models=min_models)
        reference[draw] = np.sort(np.linalg.eigvalsh(ref_rho))[::-1]
    p95 = np.percentile(reference, 95, axis=0)

    total = float(np.clip(eigenvalues.sum(), 1e-9, None))
    return Dimensionality(
        eigenvalues=eigenvalues,
        reference_p95=p95,
        n_above_reference=int((eigenvalues > p95).sum()),
        variance_first=float(eigenvalues[0] / total),
        variance_second=float(eigenvalues[1] / total) if eigenvalues.size > 1 else float("nan"),
        ratio_first_to_second=float(eigenvalues[0] / eigenvalues[1])
        if eigenvalues.size > 1
        else float("nan"),
        n_models=n_models,
        n_items=int(rho.shape[0]),
    )


def ability_correlations(thetas: dict[str, Floats]) -> dict[tuple[str, str], tuple[float, int]]:
    """Correlation between per-benchmark abilities, over the models that have both.

    Two benchmarks that measure the same ability agree on the models they share. Where they do
    not, a single composite ability is hiding something, and the README has to say so.
    """
    out: dict[tuple[str, str], tuple[float, int]] = {}
    names = sorted(thetas)
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            a, b = thetas[first], thetas[second]
            both = np.isfinite(a) & np.isfinite(b)
            n = int(both.sum())
            if n < 5:
                out[(first, second)] = (float("nan"), n)
                continue
            out[(first, second)] = (float(np.corrcoef(a[both], b[both])[0, 1]), n)
    return out
