"""The own-run validation: reading a paid-for run back without spending anything.

Two of these are regressions for defects found on 2026-09-13 while running the real panel, and
both were silent. Neither raised anything; both produced a number that looked like an answer.
That is the failure mode worth testing for here, because this module's output is the project's
headline and nothing downstream would have questioned it.
"""

from __future__ import annotations

import numpy as np
import pytest

from mselect.experiments import ownrun
from mselect.irt.model import Items, prob


def _panel(
    *,
    n_models: int = 8,
    n_items: int = 300,
    seed: int = 3,
    holes: tuple[int, int] | None = None,
    unidentified: int = 0,
) -> ownrun.Panel:
    """A synthetic panel with known abilities, generated from the parameters it is scored with.

    The models are spread across the ability range on purpose: a panel that is all one strength
    has no ranking to reproduce, and the interesting failures here are about the extremes.
    """
    rng = np.random.default_rng(seed)
    theta = np.linspace(-2.0, 2.0, n_models)
    a = rng.uniform(0.6, 2.0, n_items)
    b = rng.normal(0.0, 1.2, n_items)
    if unidentified:
        b[:unidentified] = np.nan
    items = Items(a, np.nan_to_num(b, nan=0.0), np.zeros(n_items))
    p = prob(theta, items)
    responses = (rng.random(p.shape) < p).astype(float)
    if unidentified:  # the item still has a response, it just cannot measure anything
        items = Items(a, b, np.zeros(n_items))
    cost = np.full((n_models, n_items), 0.001)
    if holes is not None:
        row, keep = holes
        responses[row, keep:] = np.nan
        cost[row, keep:] = np.nan
    benchmarks = np.array(["mmlu" if i % 2 else "gsm8k" for i in range(n_items)])
    return ownrun.Panel(
        aliases=tuple(f"m{i}" for i in range(n_models)),
        item_ids=tuple(f"item-{i}" for i in range(n_items)),
        benchmarks=benchmarks,
        responses=responses,
        cost_usd=cost,
        items=items,
        unidentified=~np.isfinite(items.b),
    )


def test_an_item_the_bank_could_not_place_does_not_erase_the_model() -> None:
    """Regression, 2026-09-13. Found on the live panel, and it destroyed every number.

    68 of the 3,000 suite items have a non-finite fitted difficulty, which is the bank refusing
    to invent one for an item every model in its fit answered the same way. Scoring one of those
    makes log p NaN, and because the posterior sums over every answered item, one such item
    makes the model's whole ability NaN. The first run reported the transfer correlation on
    zero models and nobody had to be told: `n = 0` was printed and the run exited 0.
    """
    panel = _panel(unidentified=5)
    assert panel.usable.sum() == panel.n_items - 5
    assert np.isfinite(panel.accuracy()).all(), "an unmeasurable item must not erase an accuracy"

    result = ownrun.validate(panel, checkpoints=(25, 100), resamples=100)
    assert result.transfer.n == panel.n_models, "every model must still be estimated"
    assert np.isfinite(result.transfer.point)
    for point in result.points:
        assert point.tau_adaptive.n == panel.n_models
        assert point.tau_random.n == panel.n_models
        assert point.tau_stratified.n == panel.n_models


def test_a_model_with_a_short_row_is_ranked_on_the_same_items_as_the_rest() -> None:
    """Regression, 2026-09-13. The subtler of the two, and it inverted the conclusion.

    `google-frontier` was refused after 1,088 of 3,000 items, so its accuracy was computed over
    a third of the suite and every other model's over all of it. Comparing those ranks the item
    sets as much as the models, and the run reported that adaptive selection was beaten by
    random selection at every size. On a frame every model shares, the opposite holds.
    """
    panel = _panel(holes=(0, 60))
    assert panel.common().sum() == 60
    dense = panel.dense()
    assert dense.n_items == 60
    assert dense.measurable().all(), "the dense frame has no holes left in it"
    assert dense.n_models == panel.n_models

    # The frame changes the truth, which is the whole reason it has to be shared.
    assert not np.allclose(panel.accuracy(), dense.accuracy())
    assert ownrun.validate(panel, checkpoints=(25,), resamples=100).panel.n_items == 60


def test_the_bank_ranks_models_it_was_not_fitted_on() -> None:
    """The claim the project rests on, in miniature: parameters in, ranking out, nothing refitted."""
    panel = _panel(n_models=10, n_items=400)
    result = ownrun.validate(panel, checkpoints=(25, 100, 200), resamples=200)
    assert result.transfer.point > 0.8, "scoring every item must recover the ranking"
    best = max(point.tau_adaptive.point for point in result.points)
    assert best > 0.7, "a short adaptive test must get most of the way there"


def test_a_checkpoint_is_priced_from_the_cells_it_actually_used() -> None:
    """Not an item count times an average price.

    Adaptive selection asks different models different items, and on this panel the models
    differ in price by a factor of ten. "100 items" is not a cost; "these 100 items, at what
    they cost that model" is, and it is the number section 11 asks to be reported.
    """
    panel = _panel(n_models=4, n_items=200)
    result = ownrun.validate(panel, checkpoints=(50,), resamples=100)
    point = result.points[0]
    assert point.usd_adaptive == pytest.approx(4 * 50 * 0.001)
    assert result.full_usd == pytest.approx(4 * 200 * 0.001)


def test_an_uncosted_cell_counts_as_zero_rather_than_poisoning_the_total() -> None:
    """A local model records no price and a failed call records none either. Neither is NaN money."""
    panel = _panel(n_models=3, n_items=100, holes=(0, 40))
    assert np.isfinite(panel.spend()).all()
    assert panel.spend()[0] == pytest.approx(40 * 0.001)


def test_the_validation_round_trips_through_json() -> None:
    """The result is written to out/ and read by the report, so it has to survive the trip."""
    panel = _panel(n_models=5, n_items=150)
    payload = ownrun.validate(panel, checkpoints=(25, 50), resamples=100).to_json()
    assert payload["n_items"] == 150
    assert len(payload["checkpoints"]) == 2
    assert set(payload["accuracy"]) == set(panel.aliases)
    for row in payload["checkpoints"]:
        for key in ("tau_adaptive", "tau_random", "tau_stratified", "tau_random_raw"):
            assert set(row[key]) == {"point", "lo", "hi", "n"}
