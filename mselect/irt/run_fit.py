"""Fitting a frozen bank and diagnosing it: the two commands that produce every number.

Item parameters are written next to the bank they were fitted to, carrying the bank version and
the bank's content hash, so a parameter file can never be read against the wrong bank without
it being obvious (PLAN.md, "the item bank is versioned and content-hashed").
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray

from mselect import handover, paths
from mselect.data import bank as bank_io
from mselect.data import benchmarks as benchmark_meta
from mselect.data.bank import load_params
from mselect.irt import dif, dimensionality, fitstats, q3
from mselect.irt import fit as fitting

# `load_params` lives in `data.bank` so that the packaged bank can be loaded without importing
# the fitter. It is re-exported here because that is where callers first looked for it.
__all__ = ["FIT_VERSION", "diagnose_bank", "fit_bank", "load_params"]

FIT_VERSION = "1"  # bumped when the estimator changes in a way that moves parameters


def _params_path(bank: bank_io.Bank, kind: str) -> Path:
    return bank.path / f"params-{kind}.parquet"


def fit_bank(
    *,
    version: str = "v1",
    kind: str = "2pl",
    max_iter: int = 200,
    progress: Callable[[str], None] = lambda _: None,
) -> str:
    """Fit item parameters and abilities, and write both beside the bank."""
    bank = bank_io.load(version)
    progress(bank.describe())
    mc = bank.multiple_choice if kind == "3pl" else None
    fit = fitting.fit_mml(bank.x, kind=kind, mc=mc, max_iter=max_iter, progress=progress)

    unidentified = fitting.flag_unidentified(fit.items)
    pl.DataFrame(
        {
            "item_id": bank.item_ids,
            "benchmark": bank.benchmarks,
            "a": fit.items.a,
            "b": np.where(unidentified, np.nan, fit.items.b),
            "c": fit.items.c,
            "se_a": fit.se_a,
            "se_b": np.where(unidentified, np.nan, fit.se_b),
            "se_c": fit.se_c,
            "n_models": fit.responses_per_item,
            "difficulty_unidentified": unidentified,
        }
    ).write_parquet(_params_path(bank, kind))

    pl.DataFrame(
        {
            "model_id": bank.model_ids,
            "theta": fit.theta,
            "theta_se": fit.theta_se,
            "n_items": fit.items_per_model,
            "full_suite_score": bank_io.full_suite_scores(bank),
        }
    ).write_parquet(bank.path / f"abilities-{kind}.parquet")

    meta = {
        "fit_version": FIT_VERSION,
        "kind": kind,
        "bank_version": bank.version,
        "bank_hash": bank.bank_hash,
        "fitted": datetime.now(UTC).date().isoformat(),
        "estimator": "marginal maximum a posteriori by Bock-Aitkin EM, 61-point normal quadrature",
        "priors": fit.priors.describe(),
        "priors_values": asdict(fit.priors),
        "iterations": fit.iterations,
        "converged": fit.converged,
        "marginal_loglik": fit.loglik,
        "log_posterior": fit.log_posterior,
        "n_items": bank.n_items,
        "n_models": bank.n_models,
        "items_with_unidentified_difficulty": int(unidentified.sum()),
    }
    (bank.path / f"params-{kind}.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return fit.describe()


def diagnose_bank(
    *,
    version: str = "v1",
    kind: str = "2pl",
    q3_sample: int = 400,
    dimension_sample: int = 300,
    seed: int = 0,
    progress: Callable[[str], None] = lambda _: None,
) -> str:
    """Fit statistics, local dependence, dimensionality and DIF, written to out/ as JSON."""
    bank = bank_io.load(version)
    items, params, meta = load_params(bank, kind)
    abilities = pl.read_parquet(bank.path / f"abilities-{kind}.parquet")
    theta = abilities["theta"].to_numpy()
    progress(f"{bank.describe()}, parameters from fit {meta['fit_version']} ({kind})")

    stats = fitstats.item_fit(bank.x, items, theta)
    flags = fitstats.flags(stats, items)
    progress("item fit done")

    item_table = params.with_columns(
        infit=pl.Series(stats.infit),
        outfit=pl.Series(stats.outfit),
        point_biserial=pl.Series(stats.point_biserial),
        proportion_correct=pl.Series(stats.proportion_correct),
        monotone_drop=pl.Series(stats.monotone_drop),
        separation_auc=pl.Series(stats.separation_auc),
        information_at_zero=pl.Series(stats.information_at_zero),
        **{name: pl.Series(values) for name, values in flags.items()},
    )
    item_table.write_parquet(paths.out_for(bank.version) / f"item-diagnostics-{kind}.parquet")

    benchmarks = sorted(set(bank.benchmarks.tolist()))
    local_dependence: dict[str, object] = {}
    blocks: dict[str, list[handover.DependentBlock]] = {}
    dependence_by_benchmark: dict[str, handover.Dependence] = {}
    for name in benchmarks:
        index = q3.sample_items(bank.benchmarks, name, size=q3_sample, seed=seed)
        if index.size < 20:
            continue
        try:
            dependence = q3.q3(bank.x, items, theta, index)
        except ValueError as exc:
            local_dependence[name] = {"skipped": str(exc)}
            continue
        blocks[name] = handover.blocks_from_q3(dependence, bank.item_ids, name)
        stats_q3 = dependence.summary()
        dependence_by_benchmark[name] = handover.Dependence(
            benchmark=name,
            mean_q3=float(stats_q3["mean"]),
            expected_under_independence=float(stats_q3["expected_under_independence"]),
            share_above_flag=float(stats_q3["share_above_flag"]),
            n_models=int(dependence.n_models),
            items_sampled=int(index.size),
        )
        summary = dependence.summary()
        summary["n_models"] = float(dependence.n_models)
        top = [
            {"item_a": bank.item_ids[a], "item_b": bank.item_ids[b], "q3": round(value, 3)}
            for a, b, value in dependence.pairs[:20]
        ]
        local_dependence[name] = {"summary": summary, "worst_pairs": top}
        progress(f"Q3 {name}: {summary.get('share_above_flag', 0):.1%} of pairs above 0.2")

    handover.write_blocks(
        bank.path / handover.BLOCKS_FILE,
        blocks,
        dependence_by_benchmark,
        bank_version=bank.version,
        bank_hash=bank.bank_hash,
        seed=seed,
    )
    total_blocks = sum(len(per_benchmark) for per_benchmark in blocks.values())
    worst = min(
        (record for record in dependence_by_benchmark.values()),
        key=lambda record: record.effective_items(100),
        default=None,
    )
    progress(
        f"handover: {total_blocks} near-duplicate item blocks, and 100 items are worth as few as "
        f"{worst.effective_items(100):.0f} independent ones ({worst.benchmark})"
        if worst is not None
        else f"handover: {total_blocks} near-duplicate item blocks"
    )

    dimensions: dict[str, object] = {}
    for name in benchmarks:
        index = q3.sample_items(bank.benchmarks, name, size=dimension_sample, seed=seed + 1)
        subset = bank.x[:, index]
        keep = np.isfinite(subset).all(axis=1)
        if keep.sum() < 25 or index.size < 50:
            continue
        try:
            structure = dimensionality.parallel_analysis(subset, draws=15, seed=seed)
        except (ValueError, np.linalg.LinAlgError) as exc:  # pragma: no cover
            dimensions[name] = {"skipped": str(exc)}
            continue
        dimensions[name] = structure.summary()
        progress(f"dimensionality {name}: first eigenvalue share {structure.variance_first:.2f}")

    per_benchmark_theta: dict[str, np.ndarray] = {}
    for name in benchmarks:
        mask = bank.benchmark_mask(name)
        if mask.sum() < 50:
            continue
        sub_fit = fitting.fit_mml(bank.x[:, mask], kind="2pl", max_iter=120)
        answered = np.isfinite(bank.x[:, mask]).sum(axis=1)
        per_benchmark_theta[name] = np.where(answered >= 25, sub_fit.theta, np.nan)
    correlations = {
        f"{a} vs {b}": {"correlation": round(value, 3), "n_models": n}
        for (a, b), (value, n) in dimensionality.ability_correlations(per_benchmark_theta).items()
    }
    progress("per-benchmark abilities done")

    dif_results: dict[str, object] = {}
    for split in _panel_splits(bank):
        if split.degenerate:
            dif_results[split.key] = {"skipped": split.why}
            progress(f"DIF {split.key} skipped: {split.why}")
            continue
        result = dif.run_dif(
            bank.x,
            theta,
            split.focal_mask,
            grouping=split.grouping,
            reference=split.reference,
            focal=split.focal,
        )
        dif_results[split.key] = {"summary": result.summary(), "grouping": result.grouping}
        _write_dif(bank, result, kind, split.label)
        progress(f"DIF {split.key} done")

    released = _release_months(bank)
    for name in benchmarks:
        mask = bank.benchmark_mask(name)
        published = benchmark_meta.published(name)
        if not published or mask.sum() < 50:
            continue
        after = released > published
        usable = np.isfinite(bank.x[:, mask]).any(axis=1) & (released != "")
        if not (after & usable).any() or not (~after & usable).any():
            dif_results[f"contamination_{name}"] = {
                "skipped": (
                    f"every model in the panel that answered {name} was released after it was "
                    f"published ({published}); the before/after contrast does not exist"
                )
            }
            continue
        sub = dif.run_dif(
            bank.x[:, mask],
            theta,
            after,
            grouping=f"{name}: released after publication ({published})",
            reference="released before",
            focal="released after",
            logistic=False,
        )
        dif_results[f"contamination_{name}"] = {"summary": sub.summary(), "grouping": sub.grouping}

    cohort = _cohort_split(released)
    cohort_result = dif.run_dif(
        bank.x,
        theta,
        cohort,
        grouping="newer model generation vs older, at matched ability",
        reference="older half of the panel",
        focal="newer half of the panel",
        logistic=False,
    )
    dif_results["generation_drift"] = {
        "summary": cohort_result.summary(),
        "grouping": cohort_result.grouping,
        "note": (
            "a median split on release date. Items that are relatively easier for newer models "
            "at the same ability are contamination candidates, not proof of contamination."
        ),
    }
    _write_dif(bank, cohort_result, kind, "generation")
    progress("DIF by generation done")

    report = {
        "bank_version": bank.version,
        "bank_hash": bank.bank_hash,
        "fit": meta,
        "item_flags": {name: int(values.sum()) for name, values in flags.items()},
        "n_items": bank.n_items,
        "n_models": bank.n_models,
        "discrimination": {
            "median": float(np.median(items.a)),
            "share_below_0.3": float((items.a < 0.3).mean()),
            "share_negative": float((items.a < 0.0).mean()),
        },
        "local_dependence": local_dependence,
        "dependent_blocks": {
            "count": sum(len(per_benchmark) for per_benchmark in blocks.values()),
            "largest": max(
                (block.size for per_benchmark in blocks.values() for block in per_benchmark),
                default=0,
            ),
            "items_in_a_block": sum(
                block.size for per_benchmark in blocks.values() for block in per_benchmark
            ),
        },
        "dimensionality": dimensions,
        "benchmark_ability_correlations": correlations,
        "dif": dif_results,
    }
    out = paths.out_for(bank.version) / f"diagnostics-{kind}.json"
    out.write_text(json.dumps(report, indent=2, default=float) + "\n", encoding="utf-8")
    return f"diagnostics written to {out}"


@dataclass(frozen=True, slots=True)
class _Split:
    """One differential-item-functioning contrast this panel can or cannot draw."""

    key: str
    label: str
    grouping: str
    reference: str
    focal: str
    focal_mask: NDArray[np.bool_]

    @property
    def degenerate(self) -> bool:
        return not self.focal_mask.any() or bool(self.focal_mask.all())

    @property
    def why(self) -> str:
        side = "in the focal group" if not self.focal_mask.any() else "outside it"
        return f"{self.grouping}: every model in the panel is {side}, so there is no contrast"


def _panel_splits(bank: bank_io.Bank) -> list[_Split]:
    """The model groupings worth testing, for whichever panel this bank was built from.

    Bank v1's panel mixes API-only and open-weight models, so "does an item behave differently
    for models you can download" is answerable. Bank v2's panel is the Open LLM Leaderboard,
    where every model is open by construction, so that contrast has one side; what exists there
    is a third of the panel being weight merges rather than models trained directly, which is a
    fair question to ask of an item bank calibrated on them. A split with everyone on one side
    is recorded as skipped, with the reason, rather than run on a grouping that says nothing.
    """
    splits: list[_Split] = [
        _Split(
            key="open_vs_api",
            label="open-vs-api",
            grouping="open weights vs API only",
            reference="API only",
            focal="open weights",
            focal_mask=(bank.models["access"].to_numpy() == "open"),
        )
    ]
    if "model_type" in bank.models.columns:
        splits.append(
            _Split(
                key="merge_vs_trained",
                label="merge-vs-trained",
                grouping="merged models vs models trained directly, at matched ability",
                reference="trained directly",
                focal="merged models",
                focal_mask=(bank.models["model_type"].to_numpy() == "merge"),
            )
        )
    return splits


def _write_dif(bank: bank_io.Bank, result: dif.DifResult, kind: str, label: str) -> None:
    frame = pl.DataFrame(
        {
            "item_id": bank.item_ids if result.ets_delta.size == bank.n_items else None,
            "ets_delta": result.ets_delta,
            "mh_p": result.mh_p,
            "n_reference": result.n_reference,
            "n_focal": result.n_focal,
            "uniform_p": result.uniform_p,
            "nonuniform_p": result.nonuniform_p,
        }
    )
    frame.write_parquet(paths.out_for(bank.version) / f"dif-{label}-{kind}.parquet")


def _release_months(bank: bank_io.Bank) -> NDArray[np.str_]:
    dates = bank.models["release_date"].fill_null("").to_list()
    months: NDArray[np.str_] = np.array([value[:7] if value else "" for value in dates])
    return months


def _cohort_split(released: NDArray[np.str_]) -> NDArray[np.bool_]:
    known = sorted(value for value in released.tolist() if value)
    if not known:  # pragma: no cover - the schema always carries dates
        return np.zeros(released.size, dtype=bool)
    median = known[len(known) // 2]
    newer: NDArray[np.bool_] = released > median
    return newer
