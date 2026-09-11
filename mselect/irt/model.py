"""The 2PL and 3PL response models, their information functions, and the test characteristic
curve.

One convention throughout: `a` is discrimination, `b` difficulty, `c` the lower asymptote
(guessing), and ability is `theta` on a standard normal scale. The logistic metric is used
without the 1.702 scaling constant, so an `a` of 1 here is a normal-ogive `a` of about 0.59.

Discrimination is deliberately not constrained to be positive. An item whose fitted `a` is
near zero or negative is a finding, not a numerical failure: PLAN.md section 4.1 wants those
items listed, because a negative slope is the signature of a mis-keyed answer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Floats = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Items:
    """Item parameters for a bank, aligned with the bank's item order."""

    a: Floats
    b: Floats
    c: Floats

    def __post_init__(self) -> None:
        if not (self.a.shape == self.b.shape == self.c.shape):
            raise ValueError("a, b and c must have the same shape")

    @property
    def n_items(self) -> int:
        return int(self.a.shape[0])

    def subset(self, index: NDArray[np.intp]) -> Items:
        return Items(self.a[index], self.b[index], self.c[index])

    @classmethod
    def twopl(cls, a: Floats, b: Floats) -> Items:
        return cls(a, b, np.zeros_like(a))


def prob(theta: Floats, items: Items) -> Floats:
    """P(correct) with shape (len(theta), n_items)."""
    z = np.asarray(items.a)[None, :] * (theta[:, None] - np.asarray(items.b)[None, :])
    s = _sigmoid(z)
    c = np.asarray(items.c)[None, :]
    return c + (1.0 - c) * s


def information(theta: Floats, items: Items) -> Floats:
    """Fisher information per item at each ability, shape (len(theta), n_items).

    For the 3PL this is a^2 (1 - p)/p ((p - c)/(1 - c))^2, which collapses to the familiar
    a^2 p (1 - p) when c is zero.
    """
    p = prob(theta, items)
    p = np.clip(p, 1e-12, 1 - 1e-12)
    c = np.asarray(items.c)[None, :]
    a = np.asarray(items.a)[None, :]
    return a**2 * ((1.0 - p) / p) * ((p - c) / (1.0 - c)) ** 2


def test_information(theta: Floats, items: Items) -> Floats:
    """Total information of a whole set of items at each ability."""
    return information(theta, items).sum(axis=1)


def expected_score(theta: Floats, items: Items) -> Floats:
    """The test characteristic curve: expected proportion correct over the given items.

    This is what converts a difference in ability into a difference in benchmark accuracy,
    which is the currency `power.items_needed` is asked about.
    """
    return prob(theta, items).mean(axis=1)


def _sigmoid(z: Floats) -> Floats:
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def sigmoid(z: Floats) -> Floats:
    """Numerically safe logistic, exported because the fitter and the CAT both need it."""
    return _sigmoid(z)
