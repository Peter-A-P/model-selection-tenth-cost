"""Loading a frozen bank: the response matrix, the item and model tables, and the manifest.

Everything downstream (the fit, the diagnostics, the adaptive simulation, the power function)
takes a `Bank`, so there is exactly one definition of "which items, in which order".
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray

from mselect import paths
from mselect.irt.model import Floats


@dataclass(frozen=True, slots=True)
class Bank:
    """A frozen item bank and the responses that calibrate it."""

    version: str
    path: Path
    items: pl.DataFrame
    models: pl.DataFrame
    x: Floats  # (n_models, n_items), 0, 1 or NaN
    manifest: dict[str, object]

    @property
    def item_ids(self) -> list[str]:
        return self.items["item_id"].to_list()

    @property
    def model_ids(self) -> list[str]:
        return self.models["model_id"].to_list()

    @property
    def benchmarks(self) -> NDArray[np.str_]:
        return self.items["benchmark"].to_numpy()

    @property
    def multiple_choice(self) -> NDArray[np.bool_]:
        """Which items get a guessing parameter: PLAN.md section 4.1."""
        return (self.items["kind"] == "multiple_choice").to_numpy()

    @property
    def n_models(self) -> int:
        return int(self.x.shape[0])

    @property
    def n_items(self) -> int:
        return int(self.x.shape[1])

    @property
    def bank_hash(self) -> str:
        return str(self.manifest["bank_hash"])

    def observed(self) -> NDArray[np.bool_]:
        observed: NDArray[np.bool_] = np.isfinite(self.x)
        return observed

    def benchmark_mask(self, benchmark: str) -> NDArray[np.bool_]:
        mask: NDArray[np.bool_] = self.benchmarks == benchmark
        return mask

    def describe(self) -> str:
        density = float(np.isfinite(self.x).mean())
        return (
            f"bank {self.version} ({self.bank_hash}): {self.n_models} models x "
            f"{self.n_items:,} items, {density:.1%} of cells observed"
        )


def load(version: str = "v1", root: Path | None = None) -> Bank:
    """Read a frozen bank and pivot the long-format responses into a dense matrix with holes."""
    path = (root or paths.BANK) / version
    items = pl.read_parquet(path / "items.parquet").sort("item_id")
    models = pl.read_parquet(path / "models.parquet").sort("model_id")
    responses = pl.read_parquet(path / "responses.parquet")
    manifest = json.loads((path / "MANIFEST.json").read_text(encoding="utf-8"))

    item_index = {key: i for i, key in enumerate(items["item_id"].to_list())}
    model_index = {key: i for i, key in enumerate(models["model_id"].to_list())}
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
    return Bank(version=version, path=path, items=items, models=models, x=x, manifest=manifest)


def full_suite_scores(bank: Bank) -> Floats:
    """Each model's proportion correct over the items it actually answered.

    This is the "full-suite score" every claim in the README is measured against, and it is
    the thing a benchmark leaderboard reports.
    """
    x = bank.x
    observed = np.isfinite(x)
    counts = observed.sum(axis=1)
    totals = np.where(observed, x, 0.0).sum(axis=1)
    return np.where(counts > 0, totals / np.maximum(counts, 1), np.nan)
