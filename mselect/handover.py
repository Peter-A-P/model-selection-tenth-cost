"""What project 03 takes from this project, in the shape it takes it.

PLAN.md section 8 lists the handover: the frozen bank with its parameters and fit flags, the
adaptive estimator, `power.items_needed`, reliability figures, and a note on which item blocks
are locally dependent so that 03 does not treat them as independent evidence.

Everything here reads files that ship inside the package, so an importing project needs no
network, no data directory and no fitting run. It gets the bank this project published, at the
version it published.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from mselect.data import bank as bank_io
from mselect.irt.q3 import FLAG, Q3Result

BLOCKS_FILE = "dependent-blocks.json"

# Blocks are formed at a much higher Q3 than the reporting flag. At 0.2 the chaining swallows a
# whole benchmark into one component (11 percent of MMLU pairs are above 0.2, which is more than
# enough to connect everything to everything), and "treat all 13,937 MMLU items as one piece of
# evidence" is true in a useless way. At 0.8 a block is what a consumer can act on: items whose
# residuals move together so closely that they are effectively the same question asked twice.
# The diffuse, benchmark-wide dependence is handed over separately, as a variance inflation.
BLOCK_THRESHOLD = 0.8


@dataclass(frozen=True, slots=True)
class DependentBlock:
    """A set of items that move together beyond what ability explains."""

    benchmark: str
    items: tuple[str, ...]
    max_q3: float

    @property
    def size(self) -> int:
        return len(self.items)


@dataclass(frozen=True, slots=True)
class Dependence:
    """How far one benchmark's items are from being independent evidence."""

    benchmark: str
    mean_q3: float
    expected_under_independence: float
    share_above_flag: float
    n_models: int
    items_sampled: int

    def variance_inflation(self, n_items: int) -> float:
        """How much wider a confidence interval over `n_items` of this benchmark should be.

        The standard design effect for equicorrelated units, 1 + (n - 1) * r, with r the mean
        residual correlation measured here. A consumer multiplies its variance by this, or
        equivalently its interval width by the square root of it.
        """
        correlation = max(self.mean_q3, 0.0)
        return 1.0 + max(n_items - 1, 0) * correlation

    def effective_items(self, n_items: int) -> float:
        """How many independent items `n_items` of this benchmark are actually worth."""
        return n_items / self.variance_inflation(n_items)


@dataclass(frozen=True, slots=True)
class Reliability:
    """How much of a benchmark score is noise, as far as this bank can say."""

    agreement: float
    n_repeated_cells: int
    source: str
    caveat: str

    def describe(self) -> str:
        return (
            f"{self.agreement:.1%} agreement over {self.n_repeated_cells:,} repeated cells "
            f"({self.source}). {self.caveat}"
        )


def blocks_from_q3(
    result: Q3Result, item_ids: list[str], benchmark: str, *, threshold: float = BLOCK_THRESHOLD
) -> list[DependentBlock]:
    """Group items into blocks by chaining together every pair whose Q3 is above the threshold.

    Chaining, rather than reporting pairs, is what a consumer needs: if A depends on B and B on
    C, then treating all three as independent evidence is wrong even when A and C look
    uncorrelated on their own.
    """
    parent = {int(position): int(position) for position in result.index}

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a

    strongest: dict[int, float] = {}
    size = result.matrix.shape[0]
    for row in range(size):
        for column in range(row + 1, size):
            value = float(result.matrix[row, column])
            if value <= threshold:
                continue
            left, right = int(result.index[row]), int(result.index[column])
            union(left, right)
            root = find(left)
            strongest[root] = max(strongest.get(root, value), value)

    grouped: dict[int, list[int]] = {}
    for position in result.index:
        grouped.setdefault(find(int(position)), []).append(int(position))

    blocks = [
        DependentBlock(
            benchmark=benchmark,
            items=tuple(sorted(item_ids[position] for position in members)),
            max_q3=round(strongest.get(root, 0.0), 3),
        )
        for root, members in grouped.items()
        if len(members) > 1
    ]
    return sorted(blocks, key=lambda block: (-block.size, -block.max_q3))


def write_blocks(
    path: Path,
    blocks: dict[str, list[DependentBlock]],
    dependence_by_benchmark: dict[str, Dependence],
    *,
    bank_version: str,
    bank_hash: str,
    seed: int,
    threshold: float = BLOCK_THRESHOLD,
) -> Path:
    """Write the handover's dependence file beside the bank, in two parts.

    The blocks say "these particular items are the same question asked twice". The per-benchmark
    record says "and beyond those, this whole benchmark's items move together by this much",
    which is the part that actually changes a confidence interval.
    """
    payload: dict[str, Any] = {
        "bank_version": bank_version,
        "bank_hash": bank_hash,
        "block_threshold": threshold,
        "reporting_flag": FLAG,
        "sample_seed": seed,
        "caveat": (
            "Q3 is computed on a reproducible sample of each benchmark's items, not on all of "
            "them. A block between two items that were not both sampled cannot appear here, so "
            "the block list is a lower bound, and the mean is a sample estimate."
        ),
        "dependence": [
            {
                "benchmark": record.benchmark,
                "mean_q3": round(record.mean_q3, 4),
                "expected_under_independence": round(record.expected_under_independence, 4),
                "share_above_flag": round(record.share_above_flag, 4),
                "n_models": record.n_models,
                "items_sampled": record.items_sampled,
                "variance_inflation_at_100_items": round(record.variance_inflation(100), 2),
                "effective_items_per_100": round(record.effective_items(100), 1),
            }
            for record in sorted(dependence_by_benchmark.values(), key=lambda d: d.benchmark)
        ],
        "blocks": [
            {
                "benchmark": block.benchmark,
                "size": block.size,
                "max_q3": block.max_q3,
                "items": list(block.items),
            }
            for per_benchmark in blocks.values()
            for block in per_benchmark
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


@cache
def dependent_blocks(version: str = bank_io.DEFAULT_VERSION) -> tuple[DependentBlock, ...]:
    """The locally dependent item blocks in the packaged bank.

    A consumer that builds a confidence interval over items should not treat two items from the
    same block as two pieces of evidence. PLAN.md section 8 makes this part of the handover for
    exactly that reason.
    """
    path = bank_io.default_bank(version).path / BLOCKS_FILE
    if not path.exists():
        return ()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return tuple(
        DependentBlock(
            benchmark=str(entry["benchmark"]),
            items=tuple(str(item) for item in entry["items"]),
            max_q3=float(entry["max_q3"]),
        )
        for entry in payload["blocks"]
    )


@cache
def dependence(version: str = bank_io.DEFAULT_VERSION) -> dict[str, Dependence]:
    """Per-benchmark local dependence, keyed by benchmark.

    `dependence()["math"].effective_items(100)` answers the question a release gate actually has:
    a hundred MATH items are worth how many independent ones. On this bank the answer is about
    seven, and that is the single most consequential number in the handover.
    """
    path = bank_io.default_bank(version).path / BLOCKS_FILE
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(entry["benchmark"]): Dependence(
            benchmark=str(entry["benchmark"]),
            mean_q3=float(entry["mean_q3"]),
            expected_under_independence=float(entry["expected_under_independence"]),
            share_above_flag=float(entry["share_above_flag"]),
            n_models=int(entry["n_models"]),
            items_sampled=int(entry["items_sampled"]),
        )
        for entry in payload.get("dependence", [])
    }


def dependent_block_index(version: str = bank_io.DEFAULT_VERSION) -> dict[str, int]:
    """Item id to block number, for a consumer that wants to look items up one at a time."""
    return {
        item: number
        for number, block in enumerate(dependent_blocks(version))
        for item in block.items
    }


@cache
def reliability(version: str = bank_io.DEFAULT_VERSION) -> Reliability:
    """The reliability figure this bank can support today.

    PLAN.md section 4.3 specifies a test-retest experiment at temperature 0 on an own-run panel,
    which is blocked (section 13.4). What exists instead is free and real: HELM re-ran some
    models across releases, so some model-item cells appear twice, and they do not always agree.
    That is a floor on benchmark noise, not the planned experiment, and it says so.
    """
    manifest: dict[str, Any] = dict(bank_io.default_bank(version).manifest)
    return Reliability(
        agreement=float(manifest["repeated_cell_agreement"]),
        n_repeated_cells=int(manifest["repeated_cells"]),
        source="the same model answering the same item in two HELM administrations",
        caveat=(
            "Not the temperature-0 test-retest of PLAN.md section 4.3, which needs the own-run "
            "panel: these repeats differ in run date and HELM release as well as in sampling, so "
            "they bound benchmark noise from below rather than measuring it directly."
        ),
    )
