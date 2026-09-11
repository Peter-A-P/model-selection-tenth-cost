"""The simulation command: the headline curve, the baselines, and the power-function check.

PLAN.md sections 4.2 and 4.4. Everything here writes to `out/` as JSON and Parquet, and the
report module turns those files into the README table and the figures. Nothing in the report
recomputes a number, so the table and the chart cannot disagree with each other.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict
from typing import Any

import numpy as np
import polars as pl

from mselect import paths, power
from mselect.cat import simulate
from mselect.data import bank as bank_io
from mselect.irt import run_fit
from mselect.irt.model import Items

GAP_BANDS: tuple[tuple[float, float], ...] = ((0.5, 1.5), (1.5, 3.0), (3.0, 5.0), (5.0, 10.0))


def run(
    *,
    version: str = "v1",
    kind: str = "2pl",
    seed: int = 0,
    max_items: int = 800,
    checkpoints: Sequence[int] = simulate.CHECKPOINTS,
    progress: Callable[[str], None] = lambda _: None,
) -> str:
    """Leave-one-model-out over the dense block, with the two subsampling baselines."""
    bank = bank_io.load(version)
    items, _, _ = run_fit.load_params(bank, kind)
    progress(bank.describe())

    result = simulate.leave_one_model_out(
        bank.x,
        bank.benchmarks,
        kind=kind,
        mc=bank.multiple_choice if kind == "3pl" else None,
        checkpoints=checkpoints,
        max_items=max_items,
        seed=seed,
        progress=progress,
    )

    rows: list[dict[str, object]] = []
    for n in result.checkpoints:
        for method, table in (
            ("adaptive", result.tau_adaptive),
            ("stratified", result.tau_stratified),
            ("random", result.tau_random),
            ("random_raw_score", result.tau_random_raw),
        ):
            estimate = table[n]
            rows.append(
                {
                    "items": n,
                    "share_of_suite": result.share_of_suite(n),
                    "method": method,
                    "tau": estimate.point,
                    "tau_lo": estimate.lo,
                    "tau_hi": estimate.hi,
                    "n_models": estimate.n,
                }
            )
    curve = pl.DataFrame(rows)
    curve.write_parquet(paths.out_for(version) / f"simulation-curve-{kind}.parquet")

    paths_frame = pl.DataFrame(
        {
            "model_id": [bank.model_ids[i] for i in result.block.rows],
            "full_suite_score": result.truth,
            **{f"theta_{n}": result.adaptive_theta[n] for n in result.checkpoints},
            **{f"se_{n}": result.adaptive_se[n] for n in result.checkpoints},
        }
    )
    paths_frame.write_parquet(paths.out_for(version) / f"simulation-models-{kind}.parquet")

    validation = validate_power(result, items, seed=seed)
    pl.DataFrame(validation).write_parquet(
        paths.out_for(version) / f"power-validation-{kind}.parquet"
    )

    summary = {
        "bank_version": bank.version,
        "bank_hash": bank.bank_hash,
        "kind": kind,
        "seed": seed,
        "block": {
            "models": int(result.block.rows.size),
            "items": int(result.block.cols.size),
            "density": result.block.density,
        },
        "full_suite_items": result.n_block_items,
        "tau_of_the_full_fit": result.tau_full_self,
        "curve": rows,
        "target_reached": _target_reached(result),
        "equivalent_budget": equivalent_budget(rows),
        "power_function": _power_table(items),
        "power_validation": validation,
    }
    out = paths.out_for(version) / f"simulation-{kind}.json"
    out.write_text(json.dumps(summary, indent=2, default=float) + "\n", encoding="utf-8")
    progress(result.describe())
    return f"simulation written to {out}"


def _target_reached(result: simulate.SimulationResult) -> dict[str, object]:
    """The first item count where the adaptive ranking's interval covers the full-fit tau.

    This is the honest form of "the same ranking at a tenth of the items": the point where the
    adaptive ranking stops being distinguishable from ranking on everything.
    """
    best = result.tau_full_self
    for n in result.checkpoints:
        estimate = result.tau_adaptive[n]
        if estimate.hi >= best:
            return {
                "items": n,
                "share_of_suite": result.share_of_suite(n),
                "tau": estimate.point,
                "tau_lo": estimate.lo,
                "tau_hi": estimate.hi,
                "compared_with_full_fit_tau": best,
            }
    return {"items": None, "note": "the adaptive curve never reaches the full fit's tau"}


def equivalent_budget(curve: list[dict[str, Any]]) -> list[dict[str, object]]:
    """How many items each baseline needs to match what the adaptive test gets at this budget.

    This is the efficiency claim in the unit a practitioner spends: calls. For each adaptive
    checkpoint, the number of items at which each baseline first reaches the same tau, found by
    linear interpolation between the checkpoints either side. A baseline that never reaches it
    within the simulated budget gets None, and a baseline that is already above it at its first
    checkpoint gets that checkpoint.
    """
    counts = sorted({int(row["items"]) for row in curve})
    by_method: dict[str, dict[int, float]] = {}
    for row in curve:
        by_method.setdefault(str(row["method"]), {})[int(row["items"])] = float(row["tau"])
    adaptive = by_method.get("adaptive", {})
    out: list[dict[str, object]] = []
    for n in counts:
        target = adaptive.get(n)
        if target is None:
            continue
        entry: dict[str, object] = {"adaptive_items": n, "adaptive_tau": target}
        for method, series in by_method.items():
            if method == "adaptive":
                continue
            entry[method] = _first_crossing(counts, series, target)
        out.append(entry)
    return out


def _first_crossing(counts: list[int], series: dict[int, float], target: float) -> float | None:
    """The item count at which a series first reaches `target`, interpolating between points."""
    previous_count: int | None = None
    previous_value = 0.0
    for n in counts:
        value = series.get(n)
        if value is None:
            continue
        if value >= target:
            if previous_count is None or value == previous_value:
                return float(n)
            span = value - previous_value
            return float(previous_count + (n - previous_count) * (target - previous_value) / span)
        previous_count, previous_value = n, value
    return None


def _power_table(items: Items) -> list[dict[str, object]]:
    """`items_needed` across the effects and abilities project 03 will ask about."""
    out: list[dict[str, object]] = []
    for effect in (1.0, 3.0, 5.0):
        for ability in (-1.0, 0.0, 1.0):
            try:
                answer = power.items_needed(effect, 0.8, ability, items=items)
            except ValueError as exc:
                out.append({"effect_points": effect, "ability": ability, "error": str(exc)})
                continue
            out.append(asdict(answer))
    return out


def validate_power(
    result: simulate.SimulationResult, items: Items, *, seed: int = 0
) -> list[dict[str, object]]:
    """Does the power function predict the separation rate the simulation actually produced?

    For every pair of held-out models, the full-suite accuracy gap is known. At each item count
    the simulation says whether the pair separated (a two-sided z test on the difference of the
    two ability estimates, using the posterior standard errors the adaptive test reported). The
    power function predicts that rate from the item bank alone. They should agree.
    """
    del seed
    truth = result.truth
    rows: list[dict[str, object]] = []
    n_models = truth.size
    pairs = [(i, j) for i in range(n_models) for j in range(i + 1, n_models)]
    for n in result.checkpoints:
        theta = result.adaptive_theta[n]
        se = result.adaptive_se[n]
        for low, high in GAP_BANDS:
            selected = [
                (i, j)
                for i, j in pairs
                if low <= abs(truth[i] - truth[j]) * 100.0 < high
                and np.isfinite(theta[i])
                and np.isfinite(theta[j])
            ]
            if len(selected) < 20:
                continue
            z = np.array(
                [abs(theta[i] - theta[j]) / np.sqrt(se[i] ** 2 + se[j] ** 2) for i, j in selected]
            )
            correct_direction = np.array(
                [np.sign(theta[i] - theta[j]) == np.sign(truth[i] - truth[j]) for i, j in selected]
            )
            observed = float(((z > 1.96) & correct_direction).mean())
            mid_gap = float(np.mean([abs(truth[i] - truth[j]) * 100.0 for i, j in selected]))
            ability = float(np.mean([(theta[i] + theta[j]) / 2.0 for i, j in selected]))
            predicted = power.power_at(n, mid_gap, ability, items=items)
            rows.append(
                {
                    "items": n,
                    "gap_band_points": f"{low:g} to {high:g}",
                    "mean_gap_points": mid_gap,
                    "mean_ability": ability,
                    "n_pairs": len(selected),
                    "observed_separation_rate": observed,
                    "predicted_power": predicted,
                }
            )
    return rows
