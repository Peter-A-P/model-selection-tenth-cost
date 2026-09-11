"""Item selection and stopping rules for the adaptive test.

PLAN.md section 4.2: maximum Fisher information at the current ability estimate, with content
balancing so each benchmark contributes in proportion to its weight in the full suite, and
randomesque selection among the top five so the same items are not burned on every model.

The content balancing is not decoration. Without it, maximum information alone picks whichever
benchmark happens to have the sharpest items near the model's ability, and the adaptive score
then measures a different construct from the full-suite score it is being compared with, which
would make the headline number meaningless.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from mselect.irt.model import Floats, Items, information


@dataclass(frozen=True, slots=True)
class Selector:
    """Chooses the next item for one model, from the items that model can be asked."""

    items: Items
    available: NDArray[np.bool_]  # items this model has a recorded response to
    strata: NDArray[np.intp]  # benchmark index per item, for content balancing
    weights: Floats  # target share of the test per benchmark
    randomesque: int = 5
    balance: bool = True

    def next_item(
        self, theta: float, used: NDArray[np.bool_], counts: Floats, rng: np.random.Generator
    ) -> int:
        """The next item: most informative at `theta` inside the benchmark that is furthest behind.

        Returns -1 when nothing is left to ask.
        """
        pool = self.available & ~used
        if not pool.any():
            return -1
        if self.balance:
            target = self.weights * max(counts.sum(), 1.0)
            deficit = target - counts
            order = np.argsort(-deficit)
            for stratum in order:
                inside = pool & (self.strata == stratum)
                if inside.any():
                    pool = inside
                    break
        info = information(np.array([theta]), self.items)[0]
        info = np.where(pool, info, -np.inf)
        top = np.argpartition(-info, min(self.randomesque, info.size - 1))[: self.randomesque]
        top = top[np.isfinite(info[top])]
        if top.size == 0:  # pragma: no cover - only if every candidate has -inf information
            return int(np.flatnonzero(pool)[0])
        return int(rng.choice(top).item())


def benchmark_strata(benchmarks: NDArray[np.str_]) -> tuple[NDArray[np.intp], Floats, list[str]]:
    """Map items to benchmark indices and the share of the full suite each benchmark holds."""
    names = sorted(set(benchmarks.tolist()))
    index = {name: i for i, name in enumerate(names)}
    strata = np.array([index[name] for name in benchmarks], dtype=np.intp)
    weights = np.array([(strata == i).mean() for i in range(len(names))])
    return strata, weights, names


def random_subset(
    available: NDArray[np.bool_], n: int, rng: np.random.Generator
) -> NDArray[np.intp]:
    """The first baseline: n items drawn uniformly from what the model answered."""
    pool = np.flatnonzero(available)
    if pool.size <= n:
        return pool
    return rng.choice(pool, size=n, replace=False)


def stratified_subset(
    available: NDArray[np.bool_],
    n: int,
    strata: NDArray[np.intp],
    weights: Floats,
    rng: np.random.Generator,
    difficulty: Floats | None = None,
    difficulty_bins: int = 4,
) -> NDArray[np.intp]:
    """The second baseline: proportional by benchmark, and within benchmark by difficulty.

    This is the strongest thing a careful practitioner does without item response theory, which
    is what makes it the baseline worth beating.
    """
    chosen: list[int] = []
    for stratum in range(weights.size):
        want = round(n * float(weights[stratum]))
        pool = np.flatnonzero(available & (strata == stratum))
        if want <= 0 or pool.size == 0:
            continue
        if difficulty is None or pool.size <= want:
            take = rng.choice(pool, size=min(want, pool.size), replace=False)
            chosen.extend(take.tolist())
            continue
        order = pool[np.argsort(difficulty[pool])]
        bins = np.array_split(order, difficulty_bins)
        per_bin = max(want // max(len(bins), 1), 1)
        for chunk in bins:
            if chunk.size == 0:
                continue
            take = rng.choice(chunk, size=min(per_bin, chunk.size), replace=False)
            chosen.extend(take.tolist())
    return np.array(sorted(set(chosen)), dtype=np.intp)
