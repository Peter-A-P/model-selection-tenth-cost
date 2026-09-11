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

import numpy as np
import polars as pl
from numpy.typing import NDArray

from mselect import paths
from mselect.data import bank as bank_io
from mselect.data import helm, ollm

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
        "source_name": "the public HELM per-item releases",
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


def _ollm_family(model_type: str) -> str:
    """The leaderboard's own type label, reduced to base, merge, tuned or other.

    The labels arrive with emoji and free text ("pretrained", "fine-tuned on domain-specific
    datasets", "base merges and moerges"). Merges are kept apart from models trained directly
    because they are a third of this panel, and whether a merge answers items differently at
    matched ability is the differential-item-functioning question this panel can actually put.
    Anything the leaderboard does not classify is left as "other" rather than guessed into a
    group.
    """
    text = model_type.lower()
    if "merge" in text or "moerge" in text:
        return "merge"
    if "continuously pretrained" in text:
        return "tuned"
    if "pretrained" in text:
        return "base"
    if any(word in text for word in ("fine-tuned", "chat", "instruction")):
        return "tuned"
    return "other"


def _ollm_models(panel: ollm.Panel, answered: dict[str, int]) -> pl.DataFrame:
    """The model table.

    `access` is "open" for every row, because the leaderboard only evaluates models whose
    weights are on the hub. That removes the open-weights-versus-API contrast bank v1 uses for
    differential item functioning, and `model_type` replaces it: whether a submission is a base
    model or has been tuned is the contrast this panel can actually draw.
    """
    return pl.DataFrame(
        [
            {
                "model_id": member.fullname,
                "display_name": member.fullname.split("/")[-1],
                "organisation": member.organisation,
                "access": "open",
                "release_date": member.upload_date,
                "num_parameters": (
                    None
                    if member.params_b is None or member.params_b <= 0
                    else int(member.params_b * 1e9)
                ),
                "model_type": _ollm_family(member.model_type),
                "leaderboard_average": member.average,
                "precision": member.precision,
                "n_items": answered.get(member.fullname, 0),
            }
            for member in panel.members
        ]
    ).sort("model_id")


def _dense_matrix(
    responses: pl.DataFrame, models_frame: pl.DataFrame, items_frame: pl.DataFrame
) -> NDArray[np.float64]:
    """The long responses as a dense matrix in the bank's own model and item order."""
    model_index = {key: i for i, key in enumerate(models_frame["model_id"].to_list())}
    item_index = {key: i for i, key in enumerate(items_frame["item_id"].to_list())}
    x = np.full((len(model_index), len(item_index)), np.nan)
    rows = np.fromiter(
        (model_index[m] for m in responses["model_id"].to_list()),
        dtype=np.intp,
        count=responses.height,
    )
    cols = np.fromiter(
        (item_index[i] for i in responses["item_id"].to_list()),
        dtype=np.intp,
        count=responses.height,
    )
    x[rows, cols] = responses["correct"].to_numpy().astype(float)
    return x


def _ollm_task(
    client: ollm.Client,
    task: ollm.Task,
    panel: ollm.Panel,
    progress: Callable[[str], None],
) -> tuple[pl.DataFrame, list[dict[str, object]], dict[str, object]]:
    """One task's responses and items, taken from each model's own view of the documents.

    There is no reference model here, and that is the point. Each model's `doc_id` is mapped to
    an item through that model's own document text, so a dataset revision that changes a question
    produces two items rather than a disagreement: the models that saw the old wording answer one,
    the models that saw the new wording answer the other, and the response floor decides whether
    either is worth keeping. MMLU-Pro shipped exactly such a revision partway through the
    leaderboard's life.
    """
    frames: list[pl.DataFrame] = []
    registry: dict[str, dict[str, object]] = {}
    keys: dict[str, str] = {}
    drifted: set[str] = set()
    missing: list[str] = []

    for member in panel.members:
        try:
            identity = ollm.read_identity(client, member, task)
            scores = ollm.read_task(client, member, task)
        except ollm.FetchError as exc:
            missing.append(f"{member.fullname}: {exc}")
            continue
        joined = identity.join(scores, on="doc_id", how="inner")
        correct = ollm.binary(joined, f"{member.fullname}/{task.suffix}")
        frames.append(
            pl.DataFrame(
                {
                    "model_id": pl.Series([member.fullname] * joined.height, dtype=pl.Utf8),
                    "item_id": joined["item_id"],
                    "correct": correct,
                }
            )
        )
        for item, doc, key in zip(
            joined["item_id"].to_list(),
            joined["doc_id"].to_list(),
            joined["answer_key"].to_list(),
            strict=True,
        ):
            if item not in registry:
                registry[item] = {
                    "item_id": item,
                    "benchmark": task.benchmark,
                    "kind": task.kind,
                    "scenario_key": task.suffix,
                    "instance_id": f"{task.label}#{doc}",
                    "n_options": None,
                    "has_key": bool(key),
                    "preview": "",
                    "source_project": ollm.ORG,
                    "source_release": "latest",
                }
                keys[item] = str(key)
            elif keys[item] != key:
                drifted.add(item)

    responses = (
        pl.concat(frames)
        if frames
        else pl.DataFrame(
            {"model_id": pl.Series([], dtype=pl.Utf8), "item_id": pl.Series([], dtype=pl.Utf8)}
        ).with_columns(pl.Series("correct", [], dtype=pl.Int8))
    )
    per_item = responses.group_by("item_id").len()
    models = responses["model_id"].n_unique()
    everywhere = int((per_item["len"] == models).sum()) if models else 0
    summary: dict[str, object] = {
        "task": task.suffix,
        "benchmark": task.benchmark,
        "metric": task.metric,
        "content_fields": list(task.content),
        "models_read": models,
        "models_missing": missing,
        "items": len(registry),
        "items_every_model_answered": everywhere,
        "items_under_half_the_panel": int((per_item["len"] < models / 2).sum()) if models else 0,
        "answer_key_drift_items": len(drifted),
    }
    progress(
        f"{task.suffix}: {len(registry):,} items, {models} models, "
        f"{everywhere:,} items answered by all of them"
    )
    return responses, list(registry.values()), summary


def build_ollm_bank(
    *,
    version: str = "v2",
    min_models: int = 100,
    tasks: Iterable[ollm.Task] | None = None,
    progress: Callable[[str], None] = lambda _: None,
    root: Path | None = None,
) -> BuildSummary:
    """Freeze a bank from the Open LLM Leaderboard cache.

    The output has the same shape as `build_bank`, so the fit, the diagnostics, the simulation
    and the report are unchanged. The floor is higher than bank v1's because this panel is
    larger and because a dataset revision leaves a tail of items that only the models on one side
    of it ever saw; an item answered by fewer than a quarter of 400 models has parameters too
    loose to be worth publishing.
    """
    cache = ollm.Cache(paths.ensure(paths.OLLM_CACHE))
    out_dir = paths.ensure((root or paths.BANK) / version)
    panel = ollm.load_panel(paths.OLLM_CACHE)
    selected = list(tasks) if tasks is not None else list(ollm.TASKS)

    all_items: list[dict[str, object]] = []
    frames: list[pl.DataFrame] = []
    summaries: list[dict[str, object]] = []

    with ollm.Client(cache, ollm.token_from_env()) as client:
        for index, task in enumerate(selected, start=1):
            progress(f"[{index}/{len(selected)}] {task.suffix}")
            responses, items, summary = _ollm_task(client, task, panel, progress)
            frames.append(responses)
            all_items.extend(items)
            summaries.append(summary)

    frame = pl.concat(frames) if frames else pl.DataFrame()
    responses, repeated_cells, agreement = _collapse_repeats(frame)

    counts = responses.group_by("item_id").len().rename({"len": "n_models"})
    items_frame = (
        pl.DataFrame(all_items)
        .unique(subset=["item_id"], keep="first")
        .join(counts, on="item_id", how="inner")
        .filter(pl.col("n_models") >= min_models)
        .sort("item_id")
    )
    dropped_items = len(all_items) - items_frame.height
    responses = responses.join(items_frame.select("item_id"), on="item_id", how="semi")
    answered = {
        str(row["model_id"]): int(row["len"])
        for row in responses.group_by("model_id").len().iter_rows(named=True)
    }
    models_frame = _ollm_models(panel, answered).filter(pl.col("n_items") > 0)
    responses = responses.join(models_frame.select("model_id"), on="model_id", how="semi").sort(
        ["model_id", "item_id"]
    )

    items_frame.write_parquet(out_dir / "items.parquet")
    models_frame.write_parquet(out_dir / "models.parquet")
    # The panel file travels with the bank, not just with the gitignored cache: it is the only
    # record of which candidates were probed and rejected, and the bank is what gets published.
    (out_dir / "panel.json").write_text(
        json.dumps(panel.to_json(), indent=2) + "\n", encoding="utf-8"
    )
    matrix = _dense_matrix(responses, models_frame, items_frame)
    bank_io.write_matrix(out_dir, matrix)
    density = float(np.isfinite(matrix).mean())

    per_benchmark = {
        str(row["benchmark"]): int(row["len"])
        for row in items_frame.group_by("benchmark").len().iter_rows(named=True)
    }
    bank_hash = _hash_files(
        [out_dir / n for n in ("items.parquet", "models.parquet", bank_io.MATRIX_FILE)]
    )
    averages = [member.average for member in panel.members]
    manifest = {
        "bank_version": version,
        "built": datetime.now(UTC).date().isoformat(),
        "bank_hash": bank_hash,
        "min_models_per_item": min_models,
        "n_models": models_frame.height,
        "n_items": items_frame.height,
        "n_responses": responses.height,
        "items_dropped_below_floor": dropped_items,
        "repeated_cells": repeated_cells,
        "repeated_cell_agreement": agreement,
        "items_per_benchmark": per_benchmark,
        "density": density,
        "panel": {
            "requested": panel.size,
            "seed": panel.seed,
            "strata": panel.strata,
            "per_organisation": panel.per_organisation,
            "leaderboard_average_low": min(averages),
            "leaderboard_average_high": max(averages),
        },
        "per_task": summaries,
        "sources": [
            {
                "task": task.suffix,
                "benchmark": task.benchmark,
                "metric": task.metric,
                "kind": task.kind,
            }
            for task in selected
        ],
        "source_name": "the Open LLM Leaderboard v2 per-item details",
        "source_bucket": f"{ollm.HUB}/datasets/{ollm.ORG}",
    }
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return BuildSummary(
        version=version,
        n_models=models_frame.height,
        n_items=items_frame.height,
        n_responses=responses.height,
        n_dropped_items=dropped_items,
        repeated_cells=repeated_cells,
        repeated_agreement=agreement,
        per_benchmark=per_benchmark,
        bank_hash=bank_hash,
    )
