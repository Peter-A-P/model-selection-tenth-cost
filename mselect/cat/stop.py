"""When to stop asking.

PLAN.md section 4.2 gives two rules:

* precision: stop when the posterior standard error is below a target;
* pairwise: stop when two models' intervals no longer overlap, or declare them
  indistinguishable at the budget when the item cap is reached.

The pairwise rule is the one a platform team actually uses ("is the new model worse?"), and it
is deliberately conservative: overlapping credible intervals are reported as "not separated at
this budget", never as "the same".
"""

from __future__ import annotations

from dataclasses import dataclass

from mselect.cat.estimate import Ability


@dataclass(frozen=True, slots=True)
class Decision:
    """The answer a pairwise adaptive comparison gives, with the evidence behind it."""

    separated: bool
    leader: str | None
    items_used: int
    theta_a: float
    theta_b: float
    interval_a: tuple[float, float]
    interval_b: tuple[float, float]

    def describe(self, name_a: str, name_b: str) -> str:
        if self.separated and self.leader is not None:
            winner = name_a if self.leader == "a" else name_b
            return (
                f"{winner} ahead after {self.items_used} items "
                f"({name_a} {self.theta_a:.2f} [{self.interval_a[0]:.2f}, {self.interval_a[1]:.2f}], "
                f"{name_b} {self.theta_b:.2f} [{self.interval_b[0]:.2f}, {self.interval_b[1]:.2f}])"
            )
        return (
            f"not separated after {self.items_used} items "
            f"({name_a} {self.theta_a:.2f}, {name_b} {self.theta_b:.2f}; intervals overlap)"
        )


def precision_reached(ability: Ability, target_se: float, *, min_items: int = 10) -> bool:
    """Stop once the posterior is tight enough, but never before the prior has been outvoted."""
    return ability.n_answered >= min_items and ability.se <= target_se


def separated(first: Ability, second: Ability, *, level: float = 0.95) -> Decision:
    """Are these two models separated at this level, on the evidence so far?"""
    lo_a, hi_a = first.interval(level)
    lo_b, hi_b = second.interval(level)
    apart = lo_a > hi_b or lo_b > hi_a
    leader = None
    if apart:
        leader = "a" if first.theta > second.theta else "b"
    return Decision(
        separated=apart,
        leader=leader,
        items_used=max(first.n_answered, second.n_answered),
        theta_a=first.theta,
        theta_b=second.theta,
        interval_a=(lo_a, hi_a),
        interval_b=(lo_b, hi_b),
    )
