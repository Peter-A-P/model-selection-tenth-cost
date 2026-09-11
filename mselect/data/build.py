"""Assemble the response matrix from the cache and freeze a versioned, content-hashed bank.

PLAN.md section 3.2. Items are keyed by a content hash of the question, its options and its
answer key, so the same item is the same row wherever it came from; an item is kept only when
at least `min_models` models have a recorded response to it; and the frozen bank carries a
manifest with the hash of every file in it, so a fit can name the bytes it was fitted to.

Two things the plan did not anticipate, both recorded in the manifest:

* The same model sometimes answers the same item in two different runs (HELM Lite's MMLU
  subjects overlap the standalone MMLU project). Those repeated cells are free test-retest
  evidence across administrations, so the builder counts them and their agreement rate before
  collapsing them by majority.
* Item text is not committed. The bank stores the content hash, the benchmark, the HELM
  instance id and a short preview, which is enough to look an item up in the cache and enough
  evidence for the broken-item report, without republishing benchmark text in a public
  repository. GPQA gets no preview at all: its authors ask that it not be reproduced in
  scrapeable form.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from mselect import paths
from mselect.data import helm

PREVIEW_CHARS = 160
NO_PREVIEW = frozenset({"gpqa"})  # not reproduced: see the module docstring


def normalise(text: str) -> str:
    """Unicode NFC and collapsed whitespace, so trivial formatting is not a different item."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def item_hash(benchmark: str, text: str, options: Iterable[str], answer: str) -> str:
    """The item's identity: benchmark, question, options and answer key.

    The key is part of the identity on purpose. The same question with a different answer key
    is a different measurement, and telling those apart is what the mis-keyed-item check needs.
    """
    digest = hashlib.sha256()
    digest.update(benchmark.encode())
    digest.update(b"\x00")
    digest.update(normalise(text).encode())
    for option in options:
        digest.update(b"\x01")
        digest.update(normalise(option).encode())
    digest.update(b"\x02")
    digest.update(normalise(answer).encode())
    return digest.hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class BuildSummary:
    version: str
    n_models: int
    n_items: int
    n_responses: int
    n_dropped_items: int
    repeated_cells: int
    repeated_agreement: float
    per_benchmark: dict[str, int]
    bank_hash: str

    def describe(self) -> str:
        rows = ", ".join(f"{k} {v}" for k, v in sorted(self.per_benchmark.items()))
        agreement = (
            f"{self.repeated_agreement:.1%} agreement on {self.repeated_cells:,} repeated cells"
            if self.repeated_cells
            else "no repeated cells"
        )
        return (
            f"bank {self.version} ({self.bank_hash}): {self.n_models} models, "
            f"{self.n_items:,} items, {self.n_responses:,} responses; {rows}; "
            f"{self.n_dropped_items:,} items dropped below the response floor; {agreement}"
        )


def _collect(
    client: helm.Client, scenario: helm.Scenario, progress: Callable[[str], None]
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    """Long-format responses and the item registry for one scenario, read from the cache."""
    runs = helm.runs_for(client, scenario)
    instances: dict[tuple[str, str], dict[str, helm.Instance]] = {}
    for run in helm.instance_runs(runs):
        instances[(run.scenario_key, run.suite)] = helm.instances(client, run)

    rows: list[dict[str, object]] = []
    items: dict[str, dict[str, object]] = {}
    for done, run in enumerate(runs, start=1):
        by_id = instances.get((run.scenario_key, run.suite), {})
        for instance_id, correct in helm.responses(client, run).items():
            instance = by_id.get(instance_id)
            if instance is None or not instance.text:
                continue
            key = item_hash(scenario.benchmark, instance.text, instance.options, instance.answer)
            if key not in items:
                preview = "" if scenario.benchmark in NO_PREVIEW else normalise(instance.text)
                items[key] = {
                    "item_id": key,
                    "benchmark": scenario.benchmark,
                    "kind": scenario.kind,
                    "scenario_key": run.scenario_key,
                    "instance_id": instance_id,
                    "n_options": len(instance.options),
                    "has_key": bool(instance.answer),
                    "preview": preview[:PREVIEW_CHARS],
                    "source_project": scenario.project,
                    "source_release": scenario.release,
                }
            rows.append({"model_id": run.model, "item_id": key, "correct": correct})
        if done % 500 == 0:
            progress(f"  {scenario.project}/{scenario.prefix}: read {done}/{len(runs)} runs")
    progress(
        f"  {scenario.project}/{scenario.prefix}: {len(items):,} items, {len(rows):,} responses"
    )
    return rows, items


def _collapse_repeats(frame: pl.DataFrame) -> tuple[pl.DataFrame, int, float]:
    """One response per (model, item). Repeated cells are counted, then resolved by majority.

    A tie (one correct, one incorrect, no majority) is dropped rather than guessed: an item a
    model gets right half the time is not evidence about its ability at that item.
    """
    grouped = frame.group_by(["model_id", "item_id"]).agg(
        pl.col("correct").mean().alias("mean"), pl.len().alias("times")
    )
    repeats = grouped.filter(pl.col("times") > 1)
    repeated_cells = int(repeats.height)
    agreement = (
        float(repeats.select((pl.col("mean").is_in([0.0, 1.0])).mean()).item())
        if repeated_cells
        else 1.0
    )
    resolved = (
        grouped.filter(pl.col("mean") != 0.5)
        .with_columns((pl.col("mean") > 0.5).cast(pl.Int8).alias("correct"))
        .select("model_id", "item_id", "correct")
    )
    return resolved, repeated_cells, agreement


def build_bank(
    *,
    version: str = "v1",
    min_models: int = 40,
    scenarios: Iterable[helm.Scenario] = helm.SCENARIOS,
    progress: Callable[[str], None] = lambda _: None,
    root: Path | None = None,
) -> BuildSummary:
    """Read the cache, assemble the matrix, drop thin items, and write the frozen bank."""
    cache = helm.Cache(paths.ensure(paths.HELM_CACHE))
    out_dir = paths.ensure((root or paths.BANK) / version)

    all_rows: list[dict[str, object]] = []
    all_items: dict[str, dict[str, object]] = {}
    model_meta: dict[str, helm.ModelMeta] = {}
    releases: dict[str, str] = {}
    with helm.Client(cache) as client:
        for scenario in scenarios:
            releases[scenario.project] = scenario.release
            rows, items = _collect(client, scenario, progress)
            all_rows.extend(rows)
            for key, record in items.items():
                all_items.setdefault(key, record)
        for project, release in releases.items():
            for model, meta in helm.models_for(client, project, release).items():
                model_meta.setdefault(model, meta)

    frame = pl.DataFrame(
        all_rows, schema={"model_id": pl.Utf8, "item_id": pl.Utf8, "correct": pl.Int8}
    )
    responses, repeated_cells, agreement = _collapse_repeats(frame)

    counts = responses.group_by("item_id").len().rename({"len": "n_models"})
    items_frame = (
        pl.DataFrame(list(all_items.values()))
        .join(counts, on="item_id", how="inner")
        .filter(pl.col("n_models") >= min_models)
        .sort("item_id")
    )
    dropped = len(all_items) - items_frame.height
    responses = responses.join(items_frame.select("item_id"), on="item_id", how="semi")

    per_model = responses.group_by("model_id").len().rename({"len": "n_items"})
    models_frame = (
        pl.DataFrame(
            [
                {
                    "model_id": meta.model,
                    "display_name": meta.display_name,
                    "organisation": meta.organisation,
                    "access": meta.access,
                    "release_date": meta.release_date,
                    "num_parameters": meta.num_parameters,
                }
                for meta in model_meta.values()
            ]
        )
        .join(per_model, on="model_id", how="inner")
        .sort("model_id")
    )
    responses = responses.join(models_frame.select("model_id"), on="model_id", how="semi").sort(
        ["model_id", "item_id"]
    )

    items_frame.write_parquet(out_dir / "items.parquet")
    models_frame.write_parquet(out_dir / "models.parquet")
    responses.write_parquet(out_dir / "responses.parquet")

    per_benchmark = {
        str(row["benchmark"]): int(row["len"])
        for row in items_frame.group_by("benchmark").len().iter_rows(named=True)
    }
    bank_hash = _hash_files(
        [out_dir / n for n in ("items.parquet", "models.parquet", "responses.parquet")]
    )
    manifest = {
        "bank_version": version,
        "built": datetime.now(UTC).date().isoformat(),
        "bank_hash": bank_hash,
        "min_models_per_item": min_models,
        "n_models": models_frame.height,
        "n_items": items_frame.height,
        "n_responses": responses.height,
        "items_dropped_below_floor": dropped,
        "repeated_cells": repeated_cells,
        "repeated_cell_agreement": agreement,
        "items_per_benchmark": per_benchmark,
        "sources": [
            {
                "project": s.project,
                "release": s.release,
                "scenario": s.prefix,
                "metric": s.metric,
                "benchmark": s.benchmark,
                "kind": s.kind,
            }
            for s in scenarios
        ],
        "source_bucket": helm.BUCKET,
    }
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return BuildSummary(
        version=version,
        n_models=models_frame.height,
        n_items=items_frame.height,
        n_responses=responses.height,
        n_dropped_items=dropped,
        repeated_cells=repeated_cells,
        repeated_agreement=agreement,
        per_benchmark=per_benchmark,
        bank_hash=bank_hash,
    )


def _hash_files(files: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()[:16]
