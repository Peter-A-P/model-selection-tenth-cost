"""The three measurement experiments, analysed.

PLAN.md section 4.3 specifies test-retest reliability, position bias and prompt framing. The
runs that feed them need vendor calls and have not happened (PLAN.md section 13.4), but the
analysis does not depend on how the responses were obtained, so it is written and tested here
against synthetic fixtures. When the own-run panel exists these functions take its matrices
unchanged.

One rule holds across all three, from CLAUDE.md: every reported number carries an interval. The
intervals here are percentile bootstraps over items, because the question in each case is
"would another set of items have said the same", and items are the unit that varies.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from mselect.irt.model import Floats

RESAMPLES = 2000


@dataclass(frozen=True, slots=True)
class Interval:
    """A point estimate and its percentile bootstrap interval. Never a bare number."""

    point: float
    lo: float
    hi: float
    n: int

    def fmt(self, *, percent: bool = True) -> str:
        if self.n == 0:
            return "n/a"
        if percent:
            return f"{self.point:.1%} ({self.lo:.1%} to {self.hi:.1%}, n = {self.n})"
        return f"{self.point:.3f} ({self.lo:.3f} to {self.hi:.3f}, n = {self.n})"


def bootstrap(values: Sequence[float], *, resamples: int = RESAMPLES, seed: int = 0) -> Interval:
    """Percentile bootstrap of the mean over items."""
    data = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if data.size == 0:
        return Interval(float("nan"), float("nan"), float("nan"), 0)
    rng = np.random.default_rng(seed)
    draws = data[rng.integers(0, data.size, (resamples, data.size))].mean(axis=1)
    return Interval(
        float(data.mean()),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
        int(data.size),
    )


@dataclass(frozen=True, slots=True)
class Retest:
    """How much of a benchmark score is noise, for one model."""

    model: str
    agreement: Interval
    phi: float
    flips_to_correct: int
    flips_to_incorrect: int
    n_items: int
    # McNemar's exact test on the discordant pairs: could a coin have produced this split?
    # A small value means the second administration is systematically better or worse rather
    # than differently wrong, which is drift rather than noise and a different thing to report.
    symmetry_p: float

    @property
    def net_points(self) -> float:
        """Change in percentage points of score, which is what anyone claims dropped."""
        if not self.n_items:
            return float("nan")
        return 100.0 * (self.flips_to_correct - self.flips_to_incorrect) / self.n_items

    def describe(self) -> str:
        return (
            f"{self.model}: {self.agreement.fmt()} agreement, phi {self.phi:.3f}, "
            f"{self.flips_to_correct + self.flips_to_incorrect} of {self.n_items} items changed, "
            f"{self.net_points:+.1f} points, symmetry p {self.symmetry_p:.3f}"
        )


def retest(model: str, first: Floats, second: Floats, *, seed: int = 0) -> Retest:
    """Agreement and the phi coefficient between two runs of the same items at temperature 0.

    Named `retest` rather than `test_retest` so that pytest never mistakes the experiment for
    one of its own tests if a caller imports it by name.

    Temperature 0 is not determinism: batching, routing and silent model updates all move
    answers. The number that matters for a release gate is how many items change when nothing
    changed, because that is the floor below which a drift signal is noise.
    """
    both = np.isfinite(first) & np.isfinite(second)
    a, b = first[both], second[both]
    if a.size == 0:
        return Retest(model, bootstrap([]), float("nan"), 0, 0, 0, float("nan"))
    agreement = bootstrap((a == b).astype(float), seed=seed)
    phi = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
    up = int(((a == 0) & (b == 1)).sum())
    down = int(((a == 1) & (b == 0)).sum())
    # McNemar's exact test, which on two administrations of the same items is a binomial test
    # on the discordant pairs. Added 2026-09-14 after this module's own output was read as
    # showing systematic movement: `google-frontier` flipped 7 items right and 2 wrong, which
    # looks like drift and is p = 0.18. Pooled across the panel it was 83 against 85, p = 0.94.
    # An eyeball is not a test, and a claim about symmetry that nobody can regenerate is the
    # kind of assertion this repository treats as a defect.
    symmetry = float(stats.binomtest(up, up + down, 0.5).pvalue) if up + down else float("nan")
    return Retest(
        model=model,
        agreement=agreement,
        phi=phi,
        flips_to_correct=up,
        flips_to_incorrect=down,
        n_items=int(a.size),
        symmetry_p=symmetry,
    )


@dataclass(frozen=True, slots=True)
class PositionBias:
    """What a bare multiple-choice accuracy hides about option order."""

    model: str
    accuracy_by_position: dict[str, Interval]
    bias_index: float
    share_order_dependent: Interval
    n_items: int

    def describe(self) -> str:
        spread = ", ".join(
            f"{k} {v.point:.1%}" for k, v in sorted(self.accuracy_by_position.items())
        )
        return (
            f"{self.model}: bias index {self.bias_index:.3f} ({spread}); "
            f"{self.share_order_dependent.fmt()} of items change outcome with the order"
        )


def position_bias(
    model: str,
    correct: Floats,
    answer_position: NDArray[np.str_],
    *,
    seed: int = 0,
) -> PositionBias:
    """Accuracy by the letter the correct answer wore, over cyclic permutations of one item set.

    `correct` and `answer_position` are (items, rotations): the same item asked with the answer
    in a different position each time. The bias index is the spread of accuracy across positions
    (max minus min), and "order dependent" counts items that are not answered the same way under
    every rotation.
    """
    positions = sorted(set(answer_position.ravel().tolist()))
    by_position = {
        letter: bootstrap(correct[answer_position == letter].tolist(), seed=seed)
        for letter in positions
    }
    points = [interval.point for interval in by_position.values() if np.isfinite(interval.point)]
    index = float(max(points) - min(points)) if points else float("nan")
    per_item = np.array(
        [1.0 if len(set(row[np.isfinite(row)].tolist())) > 1 else 0.0 for row in correct]
    )
    return PositionBias(
        model=model,
        accuracy_by_position=by_position,
        bias_index=index,
        share_order_dependent=bootstrap(per_item.tolist(), seed=seed),
        n_items=int(correct.shape[0]),
    )


@dataclass(frozen=True, slots=True)
class Framing:
    """How much of the score the prompt template is responsible for."""

    model: str
    accuracy_by_template: dict[str, Interval]
    variance_item: float
    variance_template: float
    variance_interaction: float
    cost_of_answer_only: Interval

    def describe(self) -> str:
        parts = ", ".join(
            f"{k} {v.point:.1%}" for k, v in sorted(self.accuracy_by_template.items())
        )
        return (
            f"{self.model}: {parts}; variance item {self.variance_item:.1%}, "
            f"template {self.variance_template:.1%}, interaction {self.variance_interaction:.1%}"
        )


def framing(
    model: str,
    correct: Floats,
    templates: Sequence[str],
    *,
    baseline: str = "plain",
    reasoning: str = "brief_reasoning",
    seed: int = 0,
) -> Framing:
    """Variance decomposition of correctness into item, template and item-by-template.

    `correct` is (items, templates), aligned with `templates`. The decomposition is the usual
    two-way analysis of variance without replication, so the interaction and the residual are the
    same term and are reported as one; with a single observation per cell they cannot be
    separated, and pretending otherwise would invent precision.
    """
    if correct.shape[1] != len(templates):
        raise ValueError("correct must have one column per template")
    grand = float(np.nanmean(correct))
    item_means = np.nanmean(correct, axis=1)
    template_means = np.nanmean(correct, axis=0)
    n_items, n_templates = correct.shape

    ss_total = float(np.nansum((correct - grand) ** 2))
    ss_item = float(n_templates * np.nansum((item_means - grand) ** 2))
    ss_template = float(n_items * np.nansum((template_means - grand) ** 2))
    ss_interaction = max(ss_total - ss_item - ss_template, 0.0)
    scale = ss_total if ss_total > 0 else 1.0

    by_template = {
        name: bootstrap(correct[:, index].tolist(), seed=seed)
        for index, name in enumerate(templates)
    }
    cost = bootstrap([], seed=seed)
    if baseline in templates and reasoning in templates:
        difference = correct[:, templates.index(reasoning)] - correct[:, templates.index(baseline)]
        cost = bootstrap(difference.tolist(), seed=seed)
    return Framing(
        model=model,
        accuracy_by_template=by_template,
        variance_item=ss_item / scale,
        variance_template=ss_template / scale,
        variance_interaction=ss_interaction / scale,
        cost_of_answer_only=cost,
    )
