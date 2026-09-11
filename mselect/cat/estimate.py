"""Ability estimation during an adaptive test.

PLAN.md section 4.2: expected a posteriori with a standard normal prior, and maximum
likelihood as a check once ten or more items have been answered.

The estimator is incremental. An adaptive test asks for a new ability estimate after every
single item, so the posterior is carried as a log likelihood over the quadrature grid and one
item adds one vector of length Q to it. That makes a 400-item adaptive run cost the same as
400 additions rather than 400 full rescorings.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import optimize

from mselect.irt.fit import Quadrature
from mselect.irt.model import Floats, Items, sigmoid

EPS = 1e-10


@dataclass(slots=True)
class Ability:
    """A running posterior over ability, updated one item at a time."""

    quad: Quadrature = field(default_factory=Quadrature.normal)
    loglik: Floats = field(init=False)
    answered: list[int] = field(default_factory=list)
    responses: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.loglik = np.zeros_like(self.quad.nodes)

    def update(self, item: int, correct: int, items: Items) -> None:
        """Add one answered item to the posterior."""
        z = items.a[item] * (self.quad.nodes - items.b[item])
        p = np.clip(items.c[item] + (1.0 - items.c[item]) * sigmoid(z), EPS, 1 - EPS)
        self.loglik += np.log(p) if correct else np.log1p(-p)
        self.answered.append(item)
        self.responses.append(int(correct))

    @property
    def posterior(self) -> Floats:
        total = self.loglik + np.log(self.quad.weights)
        w: Floats = np.exp(total - total.max())
        return w / float(w.sum())

    @property
    def theta(self) -> float:
        """Expected a posteriori: the point estimate the adaptive test selects items against."""
        return float(self.posterior @ self.quad.nodes)

    @property
    def se(self) -> float:
        """Posterior standard deviation: the quantity the stopping rule watches."""
        post = self.posterior
        mean = float(post @ self.quad.nodes)
        second = float(post @ self.quad.nodes**2)
        return float(np.sqrt(max(second - mean**2, 0.0)))

    @property
    def n_answered(self) -> int:
        return len(self.answered)

    def interval(self, level: float = 0.95) -> tuple[float, float]:
        """Equal-tailed credible interval straight from the posterior, not a normal approximation.

        With ten items the posterior is visibly skewed, and a model near the top of the panel
        would get an interval that overshoots the range the bank can measure at all.
        """
        post = self.posterior
        cumulative = np.cumsum(post)
        tail = (1.0 - level) / 2.0
        lo = float(np.interp(tail, cumulative, self.quad.nodes))
        hi = float(np.interp(1.0 - tail, cumulative, self.quad.nodes))
        return lo, hi

    def mle(self, items: Items, *, bound: float = 5.0) -> float:
        """Maximum likelihood ability, as the check PLAN.md section 4.2 asks for.

        Undefined when every answer is right or every answer is wrong, which is exactly when
        EAP's prior is doing the work; that case returns the corresponding bound.
        """
        if not self.answered:
            return float("nan")
        index = np.array(self.answered, dtype=np.intp)
        y = np.array(self.responses, dtype=float)
        subset = items.subset(index)
        if y.min() == y.max():
            return bound if y[0] == 1 else -bound

        def negative(theta: float) -> float:
            z = subset.a * (theta - subset.b)
            p = np.clip(subset.c + (1.0 - subset.c) * sigmoid(np.asarray(z)), EPS, 1 - EPS)
            return float(-(y * np.log(p) + (1.0 - y) * np.log1p(-p)).sum())

        result = optimize.minimize_scalar(negative, bounds=(-bound, bound), method="bounded")
        return float(result.x)


def score(
    responses: Floats, items: Items, index: np.ndarray, quad: Quadrature | None = None
) -> Ability:
    """Score a fixed set of items in one go: the baselines and the validation runs use this."""
    ability = Ability(quad=quad or Quadrature.normal())
    for position in index:
        value = responses[position]
        if np.isfinite(value):
            ability.update(int(position), int(value), items)
    return ability
