"""How many items are actually needed to see a difference.

PLAN.md section 4.4: `items_needed(effect, power, ability)` returns how many items, selected
adaptively at a given ability, are needed to detect a drop of `effect` points in full-suite
accuracy at the requested power. This is the interface project 03 imports to answer "how many
eval items do you need to detect a three-point regression", so it is deliberately small, typed,
and free of any dependency on how the bank was built.

The derivation, in three steps:

1. A drop in full-suite accuracy is converted to a drop in ability through the slope of the
   test characteristic curve at that ability. The curve is the bank's own expected score, so a
   bank of easy items (a high slope near the bottom, flat at the top) honestly reports that the
   same accuracy drop means a larger ability drop where the curve is flat.
2. The information an adaptive test collects per item is the mean information of the items it
   would actually choose at that ability, not the average item in the bank. That is the whole
   point of adaptive selection and it is where the saving comes from.
3. Two models each measured on n items have a difference with variance 2 / (n I), so
   n = 2 (z_alpha + z_beta)^2 / (delta_theta^2 I). The one-sample form, comparing against a
   known reference ability, drops the factor of two.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from mselect.irt.model import Floats, Items, information, prob


def _items_or_default(items: Items | None) -> Items:
    """The caller's bank, or the calibrated one this package ships with.

    Imported inside the function rather than at module scope: `power` is the interface other
    projects import, and it should not drag Polars and a Parquet read into every import of it
    when the caller is passing their own parameters anyway.
    """
    if items is not None:
        return items
    from mselect.data.bank import default_items

    return default_items()


@dataclass(frozen=True, slots=True)
class ItemsNeeded:
    """The answer, with everything that went into it, because a bare number is a bug."""

    items: int
    effect_points: float
    power: float
    alpha: float
    ability: float
    delta_theta: float
    curve_slope: float
    information_per_item: float
    paired: bool
    items_available: int
    capped: bool

    def describe(self) -> str:
        cap = " (capped by the bank)" if self.capped else ""
        return (
            f"{self.items} items{cap} to detect {self.effect_points:.1f} accuracy points at "
            f"ability {self.ability:+.1f} with {self.power:.0%} power "
            f"(delta theta {self.delta_theta:.3f}, {self.information_per_item:.2f} information "
            f"per adaptively chosen item)"
        )


def curve_slope(items: Items, ability: float, *, step: float = 0.05) -> float:
    """d(expected proportion correct)/d(ability) for the whole bank, by central difference."""
    grid = np.array([ability - step, ability + step])
    scores = prob(grid, items).mean(axis=1)
    return float((scores[1] - scores[0]) / (2.0 * step))


def adaptive_information(items: Items, ability: float, n: int) -> float:
    """Mean information of the n most informative items at this ability."""
    info = information(np.array([ability]), items)[0]
    info = info[np.isfinite(info)]
    if info.size == 0:  # pragma: no cover - an empty bank
        return float("nan")
    take = min(n, info.size)
    return float(np.sort(info)[::-1][:take].mean())


def items_needed(
    effect: float,
    power: float = 0.8,
    ability: float = 0.0,
    *,
    items: Items | None = None,
    alpha: float = 0.05,
    paired: bool = True,
    max_items: int | None = None,
    rounds: int = 12,
) -> ItemsNeeded:
    """Items per model to detect `effect` accuracy points at `ability`, at the requested power.

    `effect` is in percentage points of full-suite accuracy, the unit a benchmark table is read
    in: 3 means "a three point drop". `paired` is the two-model comparison (the release-gate
    question, old model against new); set it False when one side is a fixed reference.

    `items` defaults to the calibrated bank that ships with the package, so a caller who just
    wants the number writes `items_needed(3, 0.8, ability)` and gets it. Pass your own `Items`
    to ask the same question of a different bank.
    """
    items = _items_or_default(items)
    if not 0.0 < power < 1.0:
        raise ValueError("power must be between 0 and 1")
    if effect <= 0.0:
        raise ValueError("effect must be a positive number of accuracy points")

    slope = curve_slope(items, ability)
    if slope <= 1e-6:
        raise ValueError(
            f"the bank's expected score is flat at ability {ability:+.2f}: no number of items "
            "from this bank can resolve an accuracy difference there"
        )
    delta_theta = (effect / 100.0) / slope
    z = stats.norm.isf(alpha / 2.0) + stats.norm.isf(1.0 - power)
    factor = 2.0 if paired else 1.0

    n = 50
    info = adaptive_information(items, ability, n)
    for _ in range(rounds):
        info = adaptive_information(items, ability, n)
        proposed = int(np.ceil(factor * z**2 / (delta_theta**2 * max(info, 1e-9))))
        proposed = max(proposed, 1)
        if proposed == n:
            break
        n = proposed

    cap = max_items if max_items is not None else items.n_items
    capped = n > cap
    return ItemsNeeded(
        items=min(n, cap),
        effect_points=effect,
        power=power,
        alpha=alpha,
        ability=ability,
        delta_theta=delta_theta,
        curve_slope=slope,
        information_per_item=info,
        paired=paired,
        items_available=items.n_items,
        capped=capped,
    )


def detectable_effect(
    n_items: int,
    power: float = 0.8,
    ability: float = 0.0,
    *,
    items: Items | None = None,
    alpha: float = 0.05,
    paired: bool = True,
) -> float:
    """The inverse question: with this many items, how small a drop can be seen at all?"""
    items = _items_or_default(items)
    slope = curve_slope(items, ability)
    info = adaptive_information(items, ability, n_items)
    z = stats.norm.isf(alpha / 2.0) + stats.norm.isf(1.0 - power)
    factor = 2.0 if paired else 1.0
    delta_theta = z * np.sqrt(factor / (n_items * max(info, 1e-9)))
    return float(delta_theta * slope * 100.0)


def power_at(
    n_items: int,
    effect: float,
    ability: float = 0.0,
    *,
    items: Items | None = None,
    alpha: float = 0.05,
    paired: bool = True,
) -> float:
    """Power to detect `effect` accuracy points with `n_items` items: the simulation check."""
    items = _items_or_default(items)
    slope = curve_slope(items, ability)
    if slope <= 1e-6:
        return float("nan")
    delta_theta = (effect / 100.0) / slope
    info = adaptive_information(items, ability, n_items)
    factor = 2.0 if paired else 1.0
    se = np.sqrt(factor / (n_items * max(info, 1e-9)))
    critical = stats.norm.isf(alpha / 2.0)
    return float(
        stats.norm.sf(critical - delta_theta / se) + stats.norm.cdf(-critical - delta_theta / se)
    )


def expected_score_curve(items: Items, grid: Floats) -> Floats:
    """The test characteristic curve, exported for the report's ability-to-accuracy figure."""
    return prob(grid, items).mean(axis=1)
