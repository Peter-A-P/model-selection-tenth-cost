"""The own-run panel: does an item bank calibrated on other people's models rank ours?

This is the validation PLAN.md section 11 has carried unticked since the start, and the one
the whole project rests on. Every other result here is measured inside HELM: items calibrated
on 150 public models, adaptive selection checked by holding one of those models out. That is a
real test, but it is a test within one harness, and a bank that only works on the panel it was
fitted from is a description of that panel rather than an instrument.

So the eleven models in `prompts.PANEL` were run here, through this project's own prompts,
parser and gateway, on 3,000 items drawn from the bank. **None of them is in the bank's fit.**
The item parameters are read as given and nothing is refitted: the numbers below ask whether
`a` and `b` estimated from other people's models predict which items separate ours.

Three things come out of it, in the order they are worth reading:

**Does the bank transfer at all.** The correlation between ability estimated from the bank's
parameters and the accuracy actually observed over all 3,000 items. If that is weak nothing
else matters.

**How few items reproduce the ranking.** Kendall's tau against the full-suite ranking at each
checkpoint, for adaptive selection and for the two baselines, with bootstrap intervals over
models. The bootstrap is over models because the question is whether the ranking would hold on
another panel, and with eleven models those intervals are wide and are reported wide.

**What a ranking costs.** Not an item count converted to money by an average: the real dollars
recorded for the exact items the selector picked, model by model, against the real dollars the
full 3,000 cost. That is the line section 11 calls "cost per ranking decision reported in
dollars", and it is the number a team re-evaluating models monthly actually needs.

A note on holes. A model that was refused partway through has a shorter row, and its row is
scored on what it answered. `Selector` already only offers items a model has a response to,
so nothing here invents a response; what it does mean is that a model with a short row has a
noisier truth, and `Panel.describe` says which models those are rather than hiding them in an
average.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from numpy.typing import NDArray

from mselect import paths
from mselect.cat import select, simulate
from mselect.cat.estimate import Ability, score
from mselect.cat.simulate import Estimate, kendall_with_ci
from mselect.irt.model import Floats, Items

CHECKPOINTS: tuple[int, ...] = (10, 25, 50, 100, 150, 200, 300, 500, 750, 1000)


@dataclass(frozen=True, slots=True)
class Panel:
    """One administration of the own-run suite: models by items, with the bank's parameters.

    `responses` is 0 or 1 where a reply was scored and NaN everywhere else, which covers both
    a call that failed and a reply nothing could be read out of. `cost_usd` is what the ledger
    charged for that one cell, so a subset of items has an exact price rather than an average.
    """

    aliases: tuple[str, ...]
    item_ids: tuple[str, ...]
    benchmarks: NDArray[np.str_]
    responses: Floats
    cost_usd: Floats
    items: Items
    # Items whose fitted difficulty is not a number: every model in the bank's fit answered them
    # the same way, so the likelihood has no maximum and the bank records that rather than
    # guessing. They are excluded from measurement and counted in `describe`.
    unidentified: NDArray[np.bool_]

    @property
    def n_models(self) -> int:
        return len(self.aliases)

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    def answered(self) -> NDArray[np.bool_]:
        """Cells with a scored reply. A fact about the run, before any question of measurement."""
        return np.isfinite(self.responses)

    @property
    def usable(self) -> NDArray[np.bool_]:
        """Items the bank has finite parameters for, and so can measure a model with."""
        return np.isfinite(self.items.a) & np.isfinite(self.items.b) & np.isfinite(self.items.c)

    def measurable(self) -> NDArray[np.bool_]:
        """Cells that were answered on an item the bank can place. The frame for every number.

        Answered and usable are different things and both matter: a cell can be missing because
        a vendor refused the call, or because the item is one nobody was ever separated by. The
        first is an accident of the run and the second is a property of the bank.
        """
        return self.answered() & self.usable[None, :]

    def common(self) -> NDArray[np.bool_]:
        """Items every model answered and the bank can measure with: the comparable frame."""
        return self.measurable().all(axis=0)

    def dense(self) -> Panel:
        """The panel restricted to the items every model answered.

        A ranking is a comparison and a comparison needs a common frame. Without this, a model
        that was refused partway through is ranked on a different item set from the models it is
        being ranked against, and the difference between the item sets enters the answer. It is
        not a small effect: the first run of `validate` scored `google-frontier` on the 1,088
        items it reached and everything else on 3,000, and reported that adaptive selection was
        beaten by random selection at every size. On a common frame the opposite is true.

        The cost is real. One model refused after a third of the suite costs every model the
        other two thirds, which is why filling that row is worth a second run rather than an
        asterisk.
        """
        keep = np.flatnonzero(self.common())
        return Panel(
            aliases=self.aliases,
            item_ids=tuple(self.item_ids[j] for j in keep),
            benchmarks=self.benchmarks[keep],
            responses=self.responses[:, keep],
            cost_usd=self.cost_usd[:, keep],
            items=self.items.subset(keep),
            unidentified=self.unidentified[keep],
        )

    def accuracy(self) -> Floats:
        """Observed proportion correct per model, over the items the bank can measure with.

        Restricted to usable items so that the ranking being reproduced and the ranking doing
        the reproducing are computed over the same columns. Over 2,932 of 3,000 items the
        difference to plain accuracy is in the third decimal, and the alternative is comparing
        a short test against a truth it could never have reached.
        """
        return simulate.full_suite_scores(np.where(self.measurable(), self.responses, np.nan))

    def spend(self) -> Floats:
        """What each model's full row cost, treating an uncosted cell as zero and saying so."""
        return np.nan_to_num(self.cost_usd, nan=0.0).sum(axis=1)

    def describe(self) -> str:
        seen = self.measurable().sum(axis=1)
        accuracy = self.accuracy()
        spend = self.spend()
        dropped = int((~self.usable).sum())
        lines = [
            f"{self.n_models} models by {self.n_items:,} items, "
            f"{int(self.answered().sum()):,} scored cells, US${spend.sum():,.2f}",
            f"{dropped} items have no fitted difficulty and cannot measure anything; "
            f"{self.n_items - dropped:,} remain, "
            f"{int(self.common().sum()):,} of them answered by every model",
            f"{'alias':<18}{'usable':>8}{'accuracy':>10}{'US$':>9}",
        ]
        for i in np.argsort(-accuracy):
            lines.append(
                f"{self.aliases[i]:<18}{int(seen[i]):>8,}{accuracy[i]:>10.3f}{spend[i]:>9.4f}"
            )
        return "\n".join(lines)


def _bank_items(
    item_ids: Sequence[str], version: str, kind: str
) -> tuple[Items, NDArray[np.bool_]]:
    """Item parameters for the suite, in the suite's order, from the bank's own fit.

    Read and never refitted. The point of the exercise is that these numbers were estimated
    without seeing any of the models they are about to be used on.
    """
    table = pq.read_table(paths.BANK / version / f"params-{kind}.parquet")
    columns = {name: table.column(name).to_pylist() for name in table.column_names}
    by_id = {
        item: (a, b, c, bool(flag))
        for item, a, b, c, flag in zip(
            columns["item_id"],
            columns["a"],
            columns["b"],
            columns["c"],
            columns["difficulty_unidentified"],
            strict=True,
        )
    }
    missing = [item for item in item_ids if item not in by_id]
    if missing:
        raise ValueError(
            f"{len(missing)} suite items have no fitted parameters, first {missing[0]}"
        )
    rows = [by_id[item] for item in item_ids]
    a = np.array([r[0] for r in rows], dtype=float)
    b = np.array([r[1] for r in rows], dtype=float)
    c = np.array([r[2] for r in rows], dtype=float)
    unidentified = np.array([r[3] for r in rows], dtype=bool)
    return Items(a, b, c), unidentified


def load(path: Path, *, version: str = "v1", kind: str = "2pl") -> Panel:
    """Read a record file written by `mselect run` into a panel. Calls nothing.

    Item order is the order the suite was committed in, not the order the file happens to be
    in, so two runs of the same suite line up column for column whatever order they were asked.
    """
    from mselect.runner import suite as suite_mod

    chosen = suite_mod.Suite.load(suite_mod.default_path(version))
    item_ids = tuple(chosen.item_ids)
    column = {item: i for i, item in enumerate(item_ids)}

    aliases: list[str] = []
    row_of: dict[str, int] = {}
    benchmarks = np.array([""] * len(item_ids), dtype=object)
    grid: list[list[float]] = []
    money: list[list[float]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError:
                continue
            item = record.get("item_id")
            j = column.get(str(item))
            if j is None:
                continue
            alias = str(record.get("alias"))
            if alias not in row_of:
                row_of[alias] = len(aliases)
                aliases.append(alias)
                grid.append([float("nan")] * len(item_ids))
                money.append([float("nan")] * len(item_ids))
            i = row_of[alias]
            correct = record.get("correct")
            if correct is not None:
                grid[i][j] = float(correct)
            cost = record.get("cost_usd")
            if isinstance(cost, int | float):
                money[i][j] = float(cost)
            mark = record.get("benchmark")
            if mark:
                benchmarks[j] = str(mark)

    items, unidentified = _bank_items(item_ids, version, kind)
    return Panel(
        aliases=tuple(aliases),
        item_ids=item_ids,
        benchmarks=benchmarks.astype(str),
        responses=np.array(grid, dtype=float),
        cost_usd=np.array(money, dtype=float),
        items=items,
        unidentified=unidentified,
    )


@dataclass(frozen=True, slots=True)
class Point:
    """One checkpoint: how well each method ranked the panel, and what it cost to find out."""

    n_items: int
    tau_adaptive: Estimate
    tau_random: Estimate
    tau_random_raw: Estimate
    tau_stratified: Estimate
    usd_adaptive: float
    usd_random: float
    mean_se: float


@dataclass(frozen=True, slots=True)
class Validation:
    """Does a bank fitted on other people's models rank ours, how few items, and for how much."""

    panel: Panel
    checkpoints: tuple[int, ...]
    truth: Floats
    transfer: Estimate
    full_usd: float
    points: tuple[Point, ...]

    def share(self, n: int) -> float:
        return n / self.panel.n_items

    def describe(self) -> str:
        lines = [
            self.panel.describe(),
            "",
            f"bank ability against observed accuracy, every item: tau {self.transfer.fmt()}",
            f"the full suite cost US${self.full_usd:,.2f}",
            "",
            f"{'items':>6}{'share':>7}  {'adaptive':<30}{'stratified':<30}"
            f"{'random (IRT)':<30}{'random (raw)':<30}{'US$':>8}{'of full':>9}",
        ]
        for point in self.points:
            saving = point.usd_adaptive / self.full_usd if self.full_usd else float("nan")
            lines.append(
                f"{point.n_items:>6}{self.share(point.n_items):>7.1%}  "
                f"{point.tau_adaptive.fmt():<30}{point.tau_stratified.fmt():<30}"
                f"{point.tau_random.fmt():<30}{point.tau_random_raw.fmt():<30}"
                f"{point.usd_adaptive:>8.2f}{saving:>9.1%}"
            )
        return "\n".join(lines)

    def to_json(self) -> dict[str, Any]:
        def estimate(e: Estimate) -> dict[str, float | int]:
            return {"point": e.point, "lo": e.lo, "hi": e.hi, "n": e.n}

        return {
            "aliases": list(self.panel.aliases),
            "n_items": self.panel.n_items,
            "scored_cells": int(self.panel.answered().sum()),
            "accuracy": dict(zip(self.panel.aliases, self.truth.tolist(), strict=True)),
            "transfer_tau": estimate(self.transfer),
            "full_suite_usd": self.full_usd,
            "checkpoints": [
                {
                    "n_items": p.n_items,
                    "share": self.share(p.n_items),
                    "tau_adaptive": estimate(p.tau_adaptive),
                    "tau_stratified": estimate(p.tau_stratified),
                    "tau_random": estimate(p.tau_random),
                    "tau_random_raw": estimate(p.tau_random_raw),
                    "usd_adaptive": p.usd_adaptive,
                    "usd_random": p.usd_random,
                    "mean_posterior_se": p.mean_se,
                }
                for p in self.points
            ],
        }


def _cost_of(panel: Panel, row: int, picked: NDArray[np.intp]) -> float:
    """What those exact items cost that exact model. An uncosted cell counts as zero."""
    if picked.size == 0:
        return 0.0
    return float(np.nan_to_num(panel.cost_usd[row, picked], nan=0.0).sum())


def _adaptive(
    panel: Panel,
    row: int,
    selector: select.Selector,
    *,
    checkpoints: Sequence[int],
    rng: np.random.Generator,
) -> tuple[dict[int, float], dict[int, float], NDArray[np.intp]]:
    """One model's adaptive test, keeping the order it asked in.

    `cat.simulate.adaptive_run` does the same walk and returns ability and standard error only.
    The order is thrown away there because a simulation over a public bank has no bill; here the
    bill is the deliverable, and "n items" priced at an average is not the same claim as "these
    n items, at what they actually cost". A reasoning model's items are not interchangeable with
    a small model's, and the whole point of selecting adaptively is that they are not.
    """
    ability = Ability()
    used = np.zeros(panel.n_items, dtype=bool)
    counts = np.zeros(selector.weights.size)
    responses = panel.responses[row]
    wanted = sorted(set(checkpoints))
    cap = max(wanted)
    thetas: dict[int, float] = {}
    errors: dict[int, float] = {}
    order: list[int] = []
    for step in range(1, cap + 1):
        item = selector.next_item(ability.theta, used, counts, rng)
        if item < 0:
            break
        used[item] = True
        counts[selector.strata[item]] += 1.0
        ability.update(item, int(responses[item]), panel.items)
        order.append(item)
        if step in wanted:
            thetas[step] = ability.theta
            errors[step] = ability.se
    for point in wanted:  # a model that ran out of items keeps its last estimate
        if point not in thetas and thetas:
            last = max(thetas)
            thetas[point], errors[point] = thetas[last], errors[last]
    return thetas, errors, np.array(order, dtype=np.intp)


def validate(
    panel: Panel,
    *,
    checkpoints: Sequence[int] = CHECKPOINTS,
    seed: int = 0,
    resamples: int = 2000,
    common_frame: bool = True,
) -> Validation:
    """Rank the panel from a few items and compare with the ranking from all of them.

    Nothing is fitted. The item parameters come from the bank, the responses come from the
    record file, and the only thing computed is what a short test would have concluded had it
    stopped early. That is the whole design: if anything were refitted here the models would
    no longer be unseen and the result would be worth much less.
    """
    if common_frame:
        panel = panel.dense()
    truth = panel.accuracy()
    available = panel.measurable()
    strata, weights, _ = select.benchmark_strata(panel.benchmarks)
    points = tuple(n for n in sorted(set(checkpoints)) if n <= panel.n_items)

    n_models = panel.n_models
    theta_all = np.full(n_models, np.nan)
    adaptive = {n: np.full(n_models, np.nan) for n in points}
    adaptive_se = {n: np.full(n_models, np.nan) for n in points}
    random_theta = {n: np.full(n_models, np.nan) for n in points}
    random_raw = {n: np.full(n_models, np.nan) for n in points}
    stratified = {n: np.full(n_models, np.nan) for n in points}
    usd_adaptive = dict.fromkeys(points, 0.0)
    usd_random = dict.fromkeys(points, 0.0)

    for row in range(n_models):
        rng = np.random.default_rng(seed + row)
        responses = panel.responses[row]
        here = available[row]
        theta_all[row] = score(responses, panel.items, np.flatnonzero(here)).theta

        selector = select.Selector(
            items=panel.items, available=here, strata=strata, weights=weights
        )
        thetas, errors, order = _adaptive(panel, row, selector, checkpoints=points, rng=rng)
        for n in points:
            if n in thetas:
                adaptive[n][row] = thetas[n]
                adaptive_se[n][row] = errors[n]
            usd_adaptive[n] += _cost_of(panel, row, order[:n])

            picked = select.random_subset(here, n, rng)
            random_theta[n][row] = score(responses, panel.items, picked).theta
            random_raw[n][row] = float(np.nanmean(responses[picked])) if picked.size else np.nan
            usd_random[n] += _cost_of(panel, row, picked)

            spread = select.stratified_subset(
                here, n, strata, weights, rng, difficulty=panel.items.b
            )
            stratified[n][row] = score(responses, panel.items, spread).theta

    transfer = kendall_with_ci(theta_all, truth, resamples=resamples, seed=seed)
    out = tuple(
        Point(
            n_items=n,
            tau_adaptive=kendall_with_ci(adaptive[n], truth, resamples=resamples, seed=seed),
            tau_random=kendall_with_ci(random_theta[n], truth, resamples=resamples, seed=seed),
            tau_random_raw=kendall_with_ci(random_raw[n], truth, resamples=resamples, seed=seed),
            tau_stratified=kendall_with_ci(stratified[n], truth, resamples=resamples, seed=seed),
            usd_adaptive=usd_adaptive[n],
            usd_random=usd_random[n],
            mean_se=float(np.nanmean(adaptive_se[n])),
        )
        for n in points
    )
    return Validation(
        panel=panel,
        checkpoints=points,
        truth=truth,
        transfer=transfer,
        full_usd=float(panel.spend().sum()),
        points=out,
    )


def default_path(version: str = "v1", template: str = "plain", rotation: int = 0) -> Path:
    """Where `mselect run` leaves the records this reads."""
    return paths.OUT / version / f"own-run-{template}-{rotation}.jsonl"
