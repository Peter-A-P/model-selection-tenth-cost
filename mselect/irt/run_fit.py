"""Fitting a frozen bank and diagnosing it: the two commands that produce every number.

Item parameters are written next to the bank they were fitted to, carrying the bank version and
the bank's content hash, so a parameter file can never be read against the wrong bank without
it being obvious (PLAN.md, "the item bank is versioned and content-hashed").
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray

from mselect import paths
from mselect.data import bank as bank_io
from mselect.data import benchmarks as benchmark_meta
from mselect.irt import dif, dimensionality, fitstats, q3
from mselect.irt import fit as fitting
from mselect.irt.model import Items

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


def load_params(
    bank: bank_io.Bank, kind: str = "2pl"
) -> tuple[Items, pl.DataFrame, dict[str, object]]:
    """Item parameters for a bank, refusing to load parameters fitted to different bytes."""
    frame = pl.read_parquet(_params_path(bank, kind))
    meta = json.loads((bank.path / f"params-{kind}.json").read_text(encoding="utf-8"))
    if meta["bank_hash"] != bank.bank_hash:
        raise ValueError(
            f"parameters were fitted to bank {meta['bank_hash']}, not {bank.bank_hash}: refit"
        )
    if frame["item_id"].to_list() != bank.item_ids:
        raise ValueError("parameter file is not aligned with the bank's item order")
    items = Items(
        frame["a"].to_numpy(),
        np.nan_to_num(frame["b"].to_numpy(), nan=0.0),
        frame["c"].to_numpy(),
    )
    return items, frame, meta


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
    item_table.write_parquet(paths.ensure(paths.OUT) / f"item-diagnostics-{kind}.parquet")

    benchmarks = sorted(set(bank.benchmarks.tolist()))
    local_dependence: dict[str, object] = {}
    for name in benchmarks:
        index = q3.sample_items(bank.benchmarks, name, size=q3_sample, seed=seed)
        if index.size < 20:
            continue
        try:
            dependence = q3.q3(bank.x, items, theta, index)
        except ValueError as exc:
            local_dependence[name] = {"skipped": str(exc)}
            continue
        summary = dependence.summary()
        summary["n_models"] = float(dependence.n_models)
        top = [
            {"item_a": bank.item_ids[a], "item_b": bank.item_ids[b], "q3": round(value, 3)}
            for a, b, value in dependence.pairs[:20]
        ]
        local_dependence[name] = {"summary": summary, "worst_pairs": top}
        progress(f"Q3 {name}: {summary.get('share_above_flag', 0):.1%} of pairs above 0.2")

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

    access = bank.models["access"].to_numpy()
    open_weights = access == "open"
    dif_results: dict[str, object] = {}
    access_dif = dif.run_dif(
        bank.x,
        theta,
        open_weights,
        grouping="open weights vs API only",
        reference="API only",
        focal="open weights",
    )
    dif_results["open_vs_api"] = {"summary": access_dif.summary(), "grouping": access_dif.grouping}
    _write_dif(bank, access_dif, kind, "open-vs-api")
    progress("DIF by access done")

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
        "dimensionality": dimensions,
        "benchmark_ability_correlations": correlations,
        "dif": dif_results,
    }
    out = paths.ensure(paths.OUT) / f"diagnostics-{kind}.json"
    out.write_text(json.dumps(report, indent=2, default=float) + "\n", encoding="utf-8")
    return f"diagnostics written to {out}"


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
    frame.write_parquet(paths.ensure(paths.OUT) / f"dif-{label}-{kind}.parquet")


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
