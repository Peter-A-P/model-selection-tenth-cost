"""The interval printed beside every accuracy in the decision-model table.

The failure worth catching is an interval that escapes [0, 1] or collapses to a point at the
extremes, which a normal approximation does: a model right on every item would be printed as
certain, and one on a handful of items as more certain than it is.
"""

from __future__ import annotations

import math

import pytest

from mselect.report.build_report import _wilson


def test_the_interval_contains_the_proportion_and_stays_inside_zero_and_one() -> None:
    for correct, n in [(0, 10), (10, 10), (83, 150), (1, 2454), (2453, 2454)]:
        lo, hi = _wilson(correct, n)
        p = correct / n
        assert 0.0 <= lo <= hi <= 1.0
        assert lo <= p + 1e-12 and p <= hi + 1e-12


def test_all_correct_is_not_printed_as_certain() -> None:
    lo, hi = _wilson(10, 10)
    assert lo < 0.75 and hi == pytest.approx(1.0)


def test_a_known_value() -> None:
    """Kev 4B on MMLU-Pro, as the README quotes it: 83 of 150, 47% to 63%."""
    lo, hi = _wilson(83, 150)
    assert (round(lo, 2), round(hi, 2)) == (0.47, 0.63)


def test_no_items_is_no_interval() -> None:
    assert all(math.isnan(v) for v in _wilson(0, 0))
