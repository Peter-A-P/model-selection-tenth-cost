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


def _ollm_items(task_items: ollm.TaskItems) -> list[dict[str, object]]:
    """The item table rows for one task.

    No preview and no option count. The leaderboard's per-item files carry the full question,
    but this bank does not fetch that column and would not commit it if it did: bank v1 already
    withholds item text, and here the identity is lm-eval-harness's own hash of the document, so
    there is nothing to preview. `instance_id` is the task and the document's position in it,
    which is what a reader needs in order to find the item in the source dataset.
    """
    task = task_items.task
    return [
        {
            "item_id": item,
            "benchmark": task.benchmark,
            "kind": task.kind,
            "scenario_key": task.suffix,
            "instance_id": f"{task.label}#{doc}",
            "n_options": None,
            "has_key": True,
            "preview": "",
            "source_project": ollm.ORG,
            "source_release": "latest",
        }
        for doc, item in zip(task_items.doc_ids, task_items.item_ids, strict=True)
    ]


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


def build_ollm_bank(
    *,
    version: str = "v2",
    tasks: Iterable[ollm.Task] | None = None,
    audit: int | None = None,
    progress: Callable[[str], None] = lambda _: None,
    root: Path | None = None,
) -> BuildSummary:
    """Freeze a bank from the Open LLM Leaderboard cache.

    The output has the same shape as `build_bank`, so the fit, the diagnostics, the simulation
    and the report are unchanged. Two things the HELM build must handle do not arise here, and
    two new ones do. Every model in the panel answered every item, so there are no repeated
    cells to collapse by majority and no response floor to apply: an item either has the whole
    panel or the build has a bug. What is new is the alignment audit, because item identity
    comes from a column this build reads for only some models, and the row-count check, because
    a model whose task file has a different number of rows was evaluated against a different
    version of the dataset and does not belong in the same matrix.
    """
    cache = ollm.Cache(paths.ensure(paths.OLLM_CACHE))
    out_dir = paths.ensure((root or paths.BANK) / version)
    panel = ollm.load_panel(paths.OLLM_CACHE)
    selected = list(tasks) if tasks is not None else list(ollm.TASKS)
    # `audit=None` verifies every model. The sample default lives in `ollm.AUDIT_MODELS` for a
    # quick look; a bank that gets published is worth checking in full, and the cache makes the
    # second run of it free.
    audit_models = len(panel.members) if audit is None else audit

    all_items: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    dropped_models: dict[str, list[str]] = {}

    with ollm.Client(cache, ollm.token_from_env()) as client:
        for index, task in enumerate(selected, start=1):
            aligned = ollm.audit_alignment(client, task, panel.members, audit=audit_models)
            all_items.extend(_ollm_items(aligned))
            by_doc = dict(zip(aligned.doc_ids, aligned.item_ids, strict=True))
            expected = len(aligned.doc_ids)

            misaligned = set(aligned.mismatched)
            kept = 0
            for member in panel.members:
                if member.fullname in misaligned:
                    # The audit found this model's doc_id to item mapping differs from the
                    # reference. Its answers cannot be matched to items, so this task is a hole
                    # for it rather than a guess.
                    dropped_models.setdefault(member.fullname, []).append(
                        f"{task.suffix}: item alignment differs from {aligned.reference}"
                    )
                    continue
                try:
                    frame = ollm.read_task(client, member, task)
                except ollm.FetchError as exc:
                    # A submission whose run of one task did not survive the Parquet conversion.
                    # It keeps its other 35 tasks and leaves a hole in this one, which is what
                    # the density figure in the manifest is for.
                    dropped_models.setdefault(member.fullname, []).append(f"{task.suffix}: {exc}")
                    continue
                if frame.height != expected:
                    dropped_models.setdefault(member.fullname, []).append(
                        f"{task.suffix}: {frame.height} rows, expected {expected}"
                    )
                    continue
                scores = ollm.binary(frame, f"{member.fullname}/{task.suffix}")
                for doc, score in zip(frame["doc_id"].to_list(), scores.to_list(), strict=True):
                    item = by_doc.get(int(doc))
                    if item is None:  # pragma: no cover - the row-count check gets here first
                        continue
                    rows.append({"model_id": member.fullname, "item_id": item, "correct": score})
                kept += 1
            audits.append(
                {
                    "task": task.suffix,
                    "benchmark": task.benchmark,
                    "metric": task.metric,
                    "items": expected,
                    "reference_model": aligned.reference,
                    "models_audited": aligned.audited,
                    "models_misaligned": aligned.mismatched,
                    "models_kept": kept,
                    "answer_key_drift_items": aligned.key_drift_items,
                    "answer_key_drift_models": aligned.key_drift_models,
                }
            )
            if aligned.mismatched:
                progress(f"  {task.suffix}: MISALIGNED {aligned.mismatched}")
            progress(
                f"[{index}/{len(selected)}] {task.suffix}: {expected:,} items, "
                f"{kept}/{len(panel.members)} models"
            )

    frame = pl.DataFrame(rows, schema={"model_id": pl.Utf8, "item_id": pl.Utf8, "correct": pl.Int8})
    responses, repeated_cells, agreement = _collapse_repeats(frame)

    counts = responses.group_by("item_id").len().rename({"len": "n_models"})
    items_frame = (
        pl.DataFrame(all_items)
        .unique(subset=["item_id"], keep="first")
        .join(counts, on="item_id", how="inner")
        .sort("item_id")
    )
    dropped_items = len(all_items) - items_frame.height
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
        "min_models_per_item": len(panel.members),
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
        "alignment_audit": audits,
        "models_with_unexpected_row_counts": dropped_models,
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
