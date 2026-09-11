"""The leave-one-model-out simulation that produces the headline curve.

PLAN.md section 4.2. For each held-out model: refit the item parameters without it, run the
adaptive test against its recorded responses, and record ability and standard error at every
item count. Rank the held-out models at each item count, compare with the full-suite ranking by
Kendall's tau, and bootstrap over models for the interval. Then do the same with random and
stratified subsampling at the same item counts, because a claim about efficiency is only worth
something against the thing it claims to beat.

One thing the plan did not say, and the data forced: the response matrix is not rectangular.
Models were run on different HELM projects, so "the full-suite ranking" is only defined over a
block of models and items that is nearly complete. `dense_block` peels the matrix down to that
block, and every number from this module is reported with the block it was measured on.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from mselect.cat import select
from mselect.cat.estimate import Ability, score
from mselect.irt import fit as fitting
from mselect.irt.model import Floats, Items

CHECKPOINTS: tuple[int, ...] = (10, 25, 50, 100, 150, 200, 300, 400, 600, 800)


@dataclass(frozen=True, slots=True)
class Block:
    """A nearly complete sub-matrix: the models and items the headline claim is measured on."""

    rows: NDArray[np.intp]
    cols: NDArray[np.intp]
    density: float

    def describe(self) -> str:
        return f"{self.rows.size} models x {self.cols.size:,} items, {self.density:.1%} complete"


def dense_block(
    x: Floats,
    *,
    min_density: float = 0.98,
    min_models: int = 40,
    min_items: int = 500,
    step: float = 0.05,
) -> Block:
    """Peel the sparsest models and items away until the block is nearly complete."""
    rows = np.arange(x.shape[0], dtype=np.intp)
    cols = np.arange(x.shape[1], dtype=np.intp)
    while True:
        observed = np.isfinite(x[np.ix_(rows, cols)])
        density = float(observed.mean())
        if density >= min_density:
            return Block(rows, cols, density)
        row_fill = observed.mean(axis=1)
        col_fill = observed.mean(axis=0)
        drop_rows = row_fill.min() <= col_fill.min()
        if drop_rows and rows.size > min_models:
            keep = row_fill > np.quantile(row_fill, step)
            rows = (
                rows[keep] if keep.sum() >= min_models else rows[np.argsort(-row_fill)[:min_models]]
            )
        elif cols.size > min_items:
            keep = col_fill > np.quantile(col_fill, step)
            cols = (
                cols[keep] if keep.sum() >= min_items else cols[np.argsort(-col_fill)[:min_items]]
            )
        else:
            return Block(rows, cols, density)


def full_suite_scores(x: Floats) -> Floats:
    """Proportion correct over the block: the ranking every efficiency claim is measured against."""
    observed = np.isfinite(x)
    totals: Floats = np.where(observed, x, 0.0).sum(axis=1)
    scores: Floats = totals / np.maximum(observed.sum(axis=1), 1)
    return scores


@dataclass(frozen=True, slots=True)
class Estimate:
    point: float
    lo: float
    hi: float
    n: int

    def fmt(self) -> str:
        return f"{self.point:.3f} ({self.lo:.3f} to {self.hi:.3f}, n = {self.n})"


def kendall_with_ci(
    estimates: Floats, truth: Floats, *, resamples: int = 2000, seed: int = 0
) -> Estimate:
    """Kendall's tau-b against the full-suite ranking, with a bootstrap interval over models.

    Models are the unit resampled, because the question is "would this ranking hold on another
    panel of models", not "on another draw of items".
    """
    both = np.isfinite(estimates) & np.isfinite(truth)
    a, b = estimates[both], truth[both]
    n = int(both.sum())
    if n < 3:
        return Estimate(float("nan"), float("nan"), float("nan"), n)
    point = float(stats.kendalltau(a, b).statistic)
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples)
    for i in range(resamples):
        take = rng.integers(0, n, n)
        if np.unique(b[take]).size < 3:  # pragma: no cover - degenerate resample
            draws[i] = np.nan
            continue
        draws[i] = stats.kendalltau(a[take], b[take]).statistic
    draws = draws[np.isfinite(draws)]
    return Estimate(point, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)), n)


def adaptive_run(
    responses: Floats,
    items: Items,
    selector: select.Selector,
    *,
    checkpoints: Sequence[int],
    max_items: int,
    rng: np.random.Generator,
) -> tuple[dict[int, float], dict[int, float]]:
    """One model's adaptive test. Returns ability and posterior standard error per checkpoint."""
    ability = Ability()
    used = np.zeros(items.n_items, dtype=bool)
    counts = np.zeros(selector.weights.size)
    thetas: dict[int, float] = {}
    errors: dict[int, float] = {}
    wanted = sorted(set(checkpoints))
    for step in range(1, max_items + 1):
        item = selector.next_item(ability.theta, used, counts, rng)
        if item < 0:
            break
        used[item] = True
        counts[selector.strata[item]] += 1.0
        ability.update(item, int(responses[item]), items)
        if step in wanted:
            thetas[step] = ability.theta
            errors[step] = ability.se
    for point in wanted:  # a model that ran out of items keeps its last estimate
        if point not in thetas and thetas:
            last = max(thetas)
            thetas[point], errors[point] = thetas[last], errors[last]
    return thetas, errors


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Everything the headline chart and the README table are drawn from."""

    block: Block
    checkpoints: tuple[int, ...]
    n_models_held_out: int  # how many models the curve is measured on; see `evaluate`
    n_block_items: int
    truth: Floats
    adaptive_theta: dict[int, Floats]
    adaptive_se: dict[int, Floats]
    random_theta: dict[int, Floats]
    random_raw: dict[int, Floats]
    stratified_theta: dict[int, Floats]
    tau_adaptive: dict[int, Estimate]
    tau_random: dict[int, Estimate]
    tau_random_raw: dict[int, Estimate]
    tau_stratified: dict[int, Estimate]
    tau_full_self: float

    def share_of_suite(self, n: int) -> float:
        return n / self.n_block_items

    def describe(self) -> str:
        header = f"{'items':>6}  {'adaptive':<30}{'stratified':<30}{'random (IRT)':<30}random (raw score)"
        lines = [f"block: {self.block.describe()}", header]
        for n in self.checkpoints:
            lines.append(
                f"{n:>6}  {self.tau_adaptive[n].fmt():<30}{self.tau_stratified[n].fmt():<30}"
                f"{self.tau_random[n].fmt():<30}{self.tau_random_raw[n].fmt()}"
            )
        return "\n".join(lines)


def leave_one_model_out(
    x: Floats,
    benchmarks: NDArray[np.str_],
    *,
    kind: str = "2pl",
    mc: NDArray[np.bool_] | None = None,
    checkpoints: Sequence[int] = CHECKPOINTS,
    max_items: int | None = None,
    seed: int = 0,
    refit_iterations: int = 15,
    block: Block | None = None,
    evaluate: int | None = None,
    progress: Callable[[str], None] = lambda _: None,
) -> SimulationResult:
    """Run the whole simulation on a bank matrix and return the curve with its intervals.

    `evaluate` caps how many models are held out one at a time. Every model is still in the
    calibration and in the truth, and the models that are held out are drawn with the seed, so
    the curve is measured on a random sample of the panel rather than on the easy part of it.
    The cap exists because each held-out model costs a warm-started refit of the whole matrix:
    on a 400-model bank that is hours, and the interval on Kendall's tau stops narrowing
    usefully well before then.
    """
    block = block or dense_block(x)
    sub = x[np.ix_(block.rows, block.cols)]
    sub_benchmarks = benchmarks[block.cols]
    sub_mc = None if mc is None else mc[block.cols]
    progress(f"block: {block.describe()}")

    truth = full_suite_scores(sub)
    strata, weights, _ = select.benchmark_strata(sub_benchmarks)
    points = tuple(n for n in sorted(set(checkpoints)) if n <= sub.shape[1])
    cap = max_items or max(points)

    base = fitting.fit_mml(sub, kind=kind, mc=sub_mc, max_iter=200, progress=None)
    progress(f"full-panel fit: {base.describe()}")

    adaptive_theta = {n: np.full(block.rows.size, np.nan) for n in points}
    adaptive_se = {n: np.full(block.rows.size, np.nan) for n in points}
    random_theta = {n: np.full(block.rows.size, np.nan) for n in points}
    random_raw = {n: np.full(block.rows.size, np.nan) for n in points}
    stratified_theta = {n: np.full(block.rows.size, np.nan) for n in points}

    rows_to_run = np.arange(sub.shape[0], dtype=np.intp)
    if evaluate is not None and evaluate < rows_to_run.size:
        rows_to_run = np.sort(
            np.random.default_rng(seed).choice(rows_to_run, size=evaluate, replace=False)
        )
        progress(f"holding out {rows_to_run.size} of {sub.shape[0]} models, drawn with seed {seed}")

    for done, row in enumerate(int(value) for value in rows_to_run):
        rng = np.random.default_rng(seed + row)
        held = fitting.refit_without(
            sub, row, kind=kind, mc=sub_mc, start=base, max_iter=refit_iterations
        )
        responses = sub[row]
        available = np.isfinite(responses)
        selector = select.Selector(
            items=held.items, available=available, strata=strata, weights=weights
        )
        thetas, errors = adaptive_run(
            responses, held.items, selector, checkpoints=points, max_items=cap, rng=rng
        )
        for n in points:
            if n in thetas:
                adaptive_theta[n][row] = thetas[n]
                adaptive_se[n][row] = errors[n]
            picked = select.random_subset(available, n, rng)
            random_theta[n][row] = score(responses, held.items, picked).theta
            random_raw[n][row] = float(np.nanmean(responses[picked])) if picked.size else np.nan
            spread = select.stratified_subset(
                available, n, strata, weights, rng, difficulty=held.items.b
            )
            stratified_theta[n][row] = score(responses, held.items, spread).theta
        if (done + 1) % 10 == 0:
            progress(f"  held out {done + 1}/{rows_to_run.size} models")

    tau = {
        name: {n: kendall_with_ci(series[n], truth, seed=seed) for n in points}
        for name, series in (
            ("adaptive", adaptive_theta),
            ("random", random_theta),
            ("random_raw", random_raw),
            ("stratified", stratified_theta),
        )
    }
    return SimulationResult(
        block=block,
        checkpoints=points,
        n_models_held_out=int(rows_to_run.size),
        n_block_items=int(sub.shape[1]),
        truth=truth,
        adaptive_theta=adaptive_theta,
        adaptive_se=adaptive_se,
        random_theta=random_theta,
        random_raw=random_raw,
        stratified_theta=stratified_theta,
        tau_adaptive=tau["adaptive"],
        tau_random=tau["random"],
        tau_random_raw=tau["random_raw"],
        tau_stratified=tau["stratified"],
        tau_full_self=float(stats.kendalltau(base.theta, truth).statistic),
    )
