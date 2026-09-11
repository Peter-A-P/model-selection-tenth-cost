"""`mselect report`: regenerate the README table, the two evidence documents and the figures.

Nothing here computes a statistic. Every number is read from the files that `mselect fit`,
`mselect diagnose` and `mselect simulate` wrote, so the README, the documents and the charts
cannot disagree with one another, and a number in the README can always be traced to the run
that produced it.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from mselect import handover, paths
from mselect.data import bank as bank_io
from mselect.data import benchmarks as benchmark_meta
from mselect.irt import run_fit
from mselect.report import figures

START = "<!-- mselect:results:start -->"
END = "<!-- mselect:results:end -->"
# Bank v1 keeps the unsuffixed names it has always had, so its links and figures do not move.
# Any other bank gets its version appended to every file it writes and its own pair of README
# markers, because two banks writing to one filename is two banks describing each other.
HEADLINE_VERSION = "v1"
PENDING = "_pending the own-run panel (needs the gateway's batch support; see PLAN.md section 3.3)_"


def _load_json(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def write_all(
    *, version: str = "v1", kind: str = "2pl", progress: Callable[[str], None] = lambda _: None
) -> str:
    bank = bank_io.load(version)
    items, params, fit_meta = run_fit.load_params(bank, kind)
    diagnostics = _load_json(paths.out_for(version) / f"diagnostics-{kind}.json")
    item_diagnostics = pl.read_parquet(paths.out_for(version) / f"item-diagnostics-{kind}.parquet")
    simulation_path = paths.out_for(version) / f"simulation-{kind}.json"
    simulation = _load_json(simulation_path) if simulation_path.exists() else None

    # Figures live under docs/, not out/: the README links them, so they are a committed
    # deliverable rather than a run artefact.
    figure_dir = paths.ensure(paths.ROOT / "docs" / "figures")
    tag = _suffix(version)
    written = [
        figures.item_parameters(params, figure_dir / f"item-parameters{tag}.png"),
        figures.information_curve(items, figure_dir / f"information{tag}.png"),
        figures.local_dependence(
            diagnostics["local_dependence"], figure_dir / f"local-dependence{tag}.png"
        ),
    ]
    if simulation is not None:
        curve = pl.read_parquet(paths.out_for(version) / f"simulation-curve-{kind}.parquet")
        written.append(
            figures.headline_curve(
                curve,
                float(simulation["tau_of_the_full_fit"]),
                int(simulation["full_suite_items"]),
                figure_dir / f"headline-curve{tag}.png",
            )
        )
    progress(f"{len(written)} figures written to {figure_dir}")

    table = results_table(bank, diagnostics, simulation, fit_meta)
    readme = paths.ROOT / "README.md"
    _replace_between(readme, table, _markers(version))
    progress(f"README results table for {version} regenerated")

    broken = broken_items_doc(bank, item_diagnostics, diagnostics)
    broken_path = paths.ROOT / "docs" / f"items-that-measure-nothing{tag}.md"
    broken_path.write_text(broken, encoding="utf-8")
    diagnostics_doc = diagnostics_document(bank, diagnostics, fit_meta)
    diagnostics_path = paths.ROOT / "docs" / f"diagnostics{tag}.md"
    diagnostics_path.write_text(diagnostics_doc, encoding="utf-8")
    progress(f"{broken_path.name} and {diagnostics_path.name} regenerated")
    return "report complete"


def _markers(version: str) -> tuple[str, str]:
    if version == HEADLINE_VERSION:
        return START, END
    return (
        f"<!-- mselect:results:{version}:start -->",
        f"<!-- mselect:results:{version}:end -->",
    )


def _suffix(version: str) -> str:
    return "" if version == HEADLINE_VERSION else f"-{version}"


def _replace_between(path: Path, block: str, markers: tuple[str, str]) -> None:
    start, end = markers
    text = path.read_text(encoding="utf-8")
    if start not in text or end not in text:
        raise ValueError(f"{path} has no {start} marker to write into")
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    path.write_text(f"{head}{start}\n{block}\n{end}{tail}", encoding="utf-8")


def _source_name(bank: bank_io.Bank) -> str:
    """Where a bank's responses came from, in words, from its own manifest.

    Hard-coding "the public HELM releases" was right while there was one bank and became a lie
    the moment there were two.
    """
    name = bank.manifest.get("source_name")
    return str(name) if name else str(bank.manifest.get("source_bucket", "an unnamed source"))


def _tau_row(curve: list[dict[str, Any]], items: int, method: str) -> dict[str, Any] | None:
    for row in curve:
        if row["items"] == items and row["method"] == method:
            return row
    return None


def results_table(
    bank: bank_io.Bank,
    diagnostics: dict[str, Any],
    simulation: dict[str, Any] | None,
    fit_meta: dict[str, Any],
) -> str:
    """The README's results table. Every row is a measured number or an honest 'not yet'."""
    manifest: dict[str, Any] = dict(bank.manifest)
    flags = diagnostics["item_flags"]
    discrimination = diagnostics["discrimination"]
    lines = [
        "| Measure | Result |",
        "|---|---|",
    ]

    if simulation is not None:
        curve = simulation["curve"]
        suite = int(simulation["full_suite_items"])
        block = simulation["block"]
        reached = simulation["target_reached"]
        for entry in simulation.get("equivalent_budget", []):
            if entry["adaptive_items"] != 10:
                continue
            raw = entry.get("random_raw_score")
            if raw:
                factor = float(raw) / float(entry["adaptive_items"])
                lines.append(
                    f"| **Calls saved at the screening budget**: items an ordinary random sample needs "
                    f"to rank as well as 10 adaptive items | {float(raw):.0f} items, "
                    f"**{factor:.1f} times** the adaptive budget (tau {float(entry['adaptive_tau']):.3f}) |"
                )
        for count in (10, 50, 200, 800):
            row = _tau_row(curve, count, "adaptive")
            if row is None:
                continue
            baseline = _tau_row(curve, count, "stratified")
            share = f"{row['share_of_suite']:.1%}"
            cell = f"**{row['tau']:.3f}** (95% CI {row['tau_lo']:.3f} to {row['tau_hi']:.3f})"
            if baseline is not None:
                cell += f"; best baseline {baseline['tau']:.3f} ({baseline['tau_lo']:.3f} to {baseline['tau_hi']:.3f})"
            lines.append(
                f"| Kendall's tau against the full {suite:,}-item ranking, {count} adaptive items ({share} of the suite) | {cell} |"
            )
        if reached.get("items"):
            lines.append(
                f"| Items at which the adaptive ranking becomes indistinguishable from ranking on everything "
                f"| {reached['items']} items, {reached['share_of_suite']:.1%} of the suite "
                f"(tau {reached['tau']:.3f}, upper bound {reached['tau_hi']:.3f} vs {reached['compared_with_full_fit_tau']:.3f}) |"
            )
        lines.append(
            f"| Ceiling on any ability-based ranking: the fit on all {suite:,} items, against the "
            f"suite average | tau {float(simulation['tau_of_the_full_fit']):.3f} "
            f"(ability and suite average are not the same construct) |"
        )
        calibrated = int(block.get("models_calibrated_on", block["models"]))
        held_out = int(block["models"])
        # The two differ only when the simulation held out a sample of a large panel; saying
        # "150 models (calibrated on 400)" is the honest form and "400 models" would not be.
        panel = f"{held_out} models" + (
            f" held out of {calibrated}" if calibrated != held_out else ""
        )
        lines.append(
            f"| Panel the ranking claim is measured on | {panel} x "
            f"{block['items']:,} items, {block['density']:.1%} complete |"
        )
        for entry in simulation["power_function"]:
            if (
                entry.get("effect_points") == 3.0
                and entry.get("ability") == 0.0
                and "items" in entry
            ):
                lines.append(
                    f"| Items needed to detect a 3-point accuracy drop at 80% power (mid-panel ability) "
                    f"| {entry['items']:,} items per model |"
                )
        for entry in simulation["power_function"]:
            if (
                entry.get("effect_points") == 1.0
                and entry.get("ability") == 0.0
                and "items" in entry
            ):
                lines.append(
                    f"| Items needed to detect a 1-point drop at 80% power (mid-panel ability) "
                    f"| {entry['items']:,} items per model |"
                )
    else:
        lines.append(f"| Kendall's tau against the full-suite ranking | {PENDING} |")

    total = bank.n_items
    lines.append(
        f"| Items whose discrimination is below 0.3, out of {total:,} "
        f"| {flags['low_discrimination'] + flags['negative_discrimination']:,} "
        f"({discrimination['share_below_0.3']:.1%}) |"
    )
    lines.append(
        f"| Items whose fitted slope is negative (the mis-keyed signature) "
        f"| {flags['negative_discrimination']:,} ({discrimination['share_negative']:.1%}) |"
    )
    lines.append(
        f"| Items carrying no measurable information at mid-panel ability | {flags['no_information']:,} "
        f"({flags['no_information'] / total:.1%}) |"
    )

    shares = [
        float(payload["summary"]["share_above_flag"])
        for payload in diagnostics["local_dependence"].values()
        if isinstance(payload, dict) and "summary" in payload
    ]
    if shares:
        lines.append(
            f"| Local dependence: item pairs with Q3 above 0.2 | {min(shares):.0%} to {max(shares):.0%} "
            f"of pairs, depending on the benchmark |"
        )
    correlations = [
        entry["correlation"] for entry in diagnostics["benchmark_ability_correlations"].values()
    ]
    if correlations:
        lines.append(
            f"| Dimensionality: correlation between per-benchmark abilities | {min(correlations):.2f} to "
            f"{max(correlations):.2f} across benchmark pairs |"
        )
    reliability = handover.reliability(bank.version)
    if reliability.measured:
        lines.append(
            f"| Reliability: the same model answering the same item twice "
            f"| {reliability.agreement:.1%} agreement "
            f"(n = {reliability.n_repeated_cells:,} repeated cells) |"
        )
    else:
        lines.append(
            "| Reliability: the same model answering the same item twice "
            "| no repeated administrations in this bank, so no figure |"
        )
    # Whichever model groupings this panel could support. A grouping with everyone on one side
    # is recorded as skipped by `diagnose`, and printing its reason is more use than omitting it.
    for key, payload in sorted(diagnostics["dif"].items()):
        if key.startswith("contamination_") or key == "generation_drift":
            continue
        label = key.replace("_", " ")
        if "summary" not in payload:
            lines.append(
                f"| Differential item functioning, {label} | not measurable on this panel |"
            )
            continue
        lines.append(
            f"| Differential item functioning, {payload['grouping']} "
            f"| {int(payload['summary']['items_flagged']):,} items flagged "
            f"({payload['summary']['share_flagged']:.1%}) |"
        )
    lines.append(f"| Position bias and prompt-framing effects | {PENDING} |")
    lines.append(f"| Cost per ranking decision, in dollars | {PENDING} |")
    lines.append("")
    lines.append(
        f"Bank `{bank.version}` (`{bank.bank_hash}`): {bank.n_models} models x {bank.n_items:,} items, "
        f"{int(manifest['n_responses']):,} recorded responses from {_source_name(bank)}. "
        f"Fitted with {fit_meta['estimator']}. "
        f"Regenerate with `mselect report --version {bank.version}`."
    )
    return "\n".join(lines)


def broken_items_doc(
    bank: bank_io.Bank, item_diagnostics: pl.DataFrame, diagnostics: dict[str, Any]
) -> str:
    """`docs/items-that-measure-nothing.md`: the broken-item report, with evidence per item."""
    frame = item_diagnostics.join(
        bank.items.select(
            "item_id", "benchmark", "preview", "instance_id", "scenario_key", "n_options"
        ),
        on="item_id",
        how="left",
    )
    total = frame.height
    flags = diagnostics["item_flags"]

    lines = [
        "# Items that measure nothing",
        "",
        "Generated by `mselect report`. Do not hand-edit.",
        "",
        f"Bank `{bank.version}` (`{bank.bank_hash}`): {bank.n_models} models by {total:,} items from "
        f"{_source_name(bank)}. Every number below comes from the 2PL fit recorded in "
        f"`mselect/bank/{bank.version}/params-2pl.json`.",
        "",
        "## What is wrong, and how often",
        "",
        "| Flag | Items | Share | What it means |",
        "|---|---:|---:|---|",
    ]
    meanings = {
        "negative_discrimination": "stronger models get it wrong more often: the signature of a mis-keyed answer or a trick item",
        "low_discrimination": "the item barely separates strong models from weak ones",
        "no_information": "the item contributes essentially nothing at mid-panel ability",
        "everyone_right": "every model answers it correctly: a free mark",
        "everyone_wrong": "no model answers it correctly: unreachable, or the key is wrong",
        "non_monotone": "the empirical curve falls as ability rises, in a bin with enough models to mean it",
        "misfit_high": "responses are more erratic than the model expects (outfit above 1.3)",
        "misfit_low": "responses are more predictable than the model expects (outfit below 0.7)",
        "perfect_separation": "ability orders every response correctly, so the likelihood has no maximum in the slope and the fitted value is a lower bound set by the prior, not by the data",
    }
    for name, count in sorted(flags.items(), key=lambda kv: -kv[1]):
        lines.append(f"| `{name}` | {count:,} | {count / total:.1%} | {meanings.get(name, '')} |")

    lines += [
        "",
        "## By benchmark",
        "",
        "| Benchmark | Items | a < 0.3 | a < 0 | no information |",
        "|---|---:|---:|---:|---:|",
    ]
    by_benchmark = (
        frame.group_by("benchmark")
        .agg(
            pl.len().alias("items"),
            (pl.col("a") < 0.3).mean().alias("weak"),
            (pl.col("a") < 0.0).mean().alias("negative"),
            pl.col("no_information").mean().alias("dead"),
        )
        # Benchmark name breaks the tie, so two benchmarks of the same size do not swap
        # places between runs and put a spurious diff in a committed document.
        .sort(["items", "benchmark"], descending=[True, False])
    )
    for row in by_benchmark.iter_rows(named=True):
        lines.append(
            f"| {benchmark_meta.title(row['benchmark'])} | {row['items']:,} | {row['weak']:.1%} | "
            f"{row['negative']:.1%} | {row['dead']:.1%} |"
        )

    lines += [
        "",
        "## The twenty worst, with their evidence",
        "",
        "Sorted by fitted discrimination, lowest first, among items at least 60 models answered. "
        "The last column is the proportion correct by ability quintile: for a working item it rises "
        "from left to right, and for these it does not.",
        "",
        "| Item | Benchmark | a (SE) | models | proportion correct | by ability quintile |",
        "|---|---|---|---:|---:|---|",
    ]
    worst = frame.filter(pl.col("n_models") >= 60).sort(["a", "item_id"]).head(20)
    curves = _quintile_curves(bank, worst["item_id"].to_list())
    for row in worst.iter_rows(named=True):
        curve = curves.get(row["item_id"], [])
        drawn = " ".join(f"{value:.2f}" if np.isfinite(value) else "n/a" for value in curve)
        lines.append(
            f"| `{row['item_id']}` | {row['benchmark']} | {row['a']:.2f} ({row['se_a']:.2f}) | "
            f"{row['n_models']} | {row['proportion_correct']:.2f} | {drawn} |"
        )

    lines += [
        "",
        "Item text is not reproduced here. The bank stores a content hash, the HELM instance id and "
        "the scenario it came from, which is enough to look any item up in the cached HELM release "
        "without republishing benchmark questions in a public repository. GPQA items carry no "
        "preview at all, at its authors' request.",
        "",
        "## What this does not claim",
        "",
        "A negative slope is evidence that something is wrong with an item, not proof that the answer "
        "key is wrong. Three other things produce the same signature: an item where the stronger "
        "model reasons past a deliberately simple answer, an item whose correct answer is ambiguous, "
        "and a grader that marks a correct answer wrong. Separating those needs the item text and a "
        "human, and PLAN.md section 9 says a sample is hand-checked before any item is named as "
        "mis-keyed. That hand-check has not been done yet, so no item here is called mis-keyed.",
        "",
    ]
    return "\n".join(lines)


def _quintile_curves(bank: bank_io.Bank, item_ids: list[str]) -> dict[str, list[float]]:
    """Proportion correct by ability quintile for a handful of items: the evidence column."""
    abilities = pl.read_parquet(bank.path / "abilities-2pl.parquet")["theta"].to_numpy()
    index = {key: i for i, key in enumerate(bank.item_ids)}
    edges = np.quantile(abilities, np.linspace(0, 1, 6))
    edges[0] -= 1e-9
    which = np.clip(np.searchsorted(edges, abilities, side="left") - 1, 0, 4)
    out: dict[str, list[float]] = {}
    for item_id in item_ids:
        column = bank.x[:, index[item_id]]
        curve = []
        for bin_index in range(5):
            rows = (which == bin_index) & np.isfinite(column)
            curve.append(float(column[rows].mean()) if rows.sum() else float("nan"))
        out[item_id] = curve
    return out


def diagnostics_document(
    bank: bank_io.Bank, diagnostics: dict[str, Any], fit_meta: dict[str, Any]
) -> str:
    """`docs/diagnostics.md`: local dependence, dimensionality and DIF, in full."""
    lines = [
        "# Local dependence, dimensionality and differential item functioning",
        "",
        "Generated by `mselect report`. Do not hand-edit.",
        "",
        f"Bank `{bank.version}` (`{bank.bank_hash}`), {fit_meta['kind'].upper()} fit, "
        f"priors {fit_meta['priors']}.",
        "",
        "## Local dependence (Yen's Q3)",
        "",
        "Q3 is the correlation between item residuals once ability is accounted for. Under local "
        "independence it has a small negative expectation, about -1/(k-1) for k items, which is "
        "printed next to each row rather than corrected away. Computed on a reproducible sample of "
        "items per benchmark, using the models that answered all of them.",
        "",
        "| Benchmark | Models | Pairs | Mean Q3 | Expected | 95th pct | Max | Share above 0.2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, payload in sorted(diagnostics["local_dependence"].items()):
        if not isinstance(payload, dict) or "summary" not in payload:
            lines.append(f"| {name} | - | - | - | - | - | - | skipped |")
            continue
        s = payload["summary"]
        lines.append(
            f"| {benchmark_meta.title(name)} | {int(s['n_models'])} | {int(s['n_pairs']):,} | "
            f"{s['mean']:+.3f} | {s['expected_under_independence']:+.3f} | {s['p95']:.3f} | "
            f"{s['max']:.3f} | {s['share_above_flag']:.1%} |"
        )

    lines += [
        "",
        "### What that means for anyone using this bank",
        "",
        "Diffuse dependence of this size changes confidence intervals, so the handover file "
        "`mselect/bank/v1/dependent-blocks.json` states it in the unit a consumer can act on. "
        "The design effect is the standard one for equicorrelated units, 1 + (n - 1) r, with r "
        "the mean residual correlation measured above; `mselect.dependence()` returns it.",
        "",
        "| Benchmark | Mean Q3 | 100 items are worth this many independent ones | Variance inflation |",
        "|---|---:|---:|---:|",
    ]
    for name, record in sorted(handover.dependence().items()):
        lines.append(
            f"| {benchmark_meta.title(name)} | {record.mean_q3:+.3f} | "
            f"{record.effective_items(100):.0f} | x{record.variance_inflation(100):.1f} |"
        )
    blocks = handover.dependent_blocks()
    if blocks:
        involved = sum(block.size for block in blocks)
        lines += [
            "",
            f"Separately, {len(blocks)} tight blocks covering {involved} items have a Q3 above "
            f"{handover.BLOCK_THRESHOLD} inside the block: items whose residuals move together so "
            "closely that they are effectively the same question asked twice. The largest holds "
            f"{max(block.size for block in blocks)} items. Those are listed in full in the same file.",
        ]

    lines += [
        "",
        "This is the honest limitation of the whole project. Benchmark items are not independent "
        "given ability: they share passages, templates, subject matter and formats. Every standard "
        "error from a fixed-length test is therefore optimistic, the more so the more items come "
        "from the same block. Project 03 is told which blocks these are so that it does not treat "
        "them as independent evidence.",
        "",
        "## Dimensionality",
        "",
        "Eigenvalues of the tetrachoric correlation matrix against a parallel-analysis reference "
        "built by permuting each item independently, on a sample of each benchmark's items.",
        "",
        "| Benchmark | Items | Models | First eigenvalue share | Second | Ratio | Factors above reference |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, payload in sorted(diagnostics["dimensionality"].items()):
        if not isinstance(payload, dict) or "first_eigenvalue_share" not in payload:
            continue
        lines.append(
            f"| {benchmark_meta.title(name)} | {int(payload['n_items'])} | {int(payload['n_models'])} | "
            f"{payload['first_eigenvalue_share']:.2f} | {payload['second_eigenvalue_share']:.2f} | "
            f"{payload['first_to_second_ratio']:.1f} | {int(payload['factors_above_parallel_reference'])} |"
        )

    lines += [
        "",
        "## Do the benchmarks measure the same ability?",
        "",
        "Each benchmark was fitted on its own and the resulting abilities correlated across the "
        "models that have both. A composite ability is only honest where these are high.",
        "",
        "| Pair | Correlation | Models |",
        "|---|---:|---:|",
    ]
    for pair, payload in sorted(
        diagnostics["benchmark_ability_correlations"].items(),
        key=lambda kv: kv[1]["correlation"],
    ):
        lines.append(f"| {pair} | {payload['correlation']:.2f} | {payload['n_models']} |")

    lines += ["", "## Differential item functioning", ""]
    for key, payload in diagnostics["dif"].items():
        if not isinstance(payload, dict):
            continue
        if "skipped" in payload:
            lines += [f"**{key}**: not testable. {payload['skipped']}", ""]
            continue
        summary = payload["summary"]
        lines += [
            f"**{payload['grouping']}**",
            "",
            f"- items testable: {int(summary['items_testable']):,}",
            f"- flagged (ETS delta above 1.5 and Mantel-Haenszel p below 0.01): "
            f"{int(summary['items_flagged']):,} ({summary['share_flagged']:.1%})",
            f"- of those, favouring the focal group: {int(summary['flagged_favouring_focal']):,}",
            f"- median absolute delta across testable items: {summary['median_abs_delta']:.2f}",
        ]
        if payload.get("note"):
            lines.append(f"- note: {payload['note']}")
        lines.append("")
    return "\n".join(lines)
