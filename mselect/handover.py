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

import numpy as np

from mselect import paths
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
    # The direct measurement, present once the own-run test-retest has been run. None means
    # the figure above is HELM's overlapping administrations standing in for it.
    score_move_points: float | None = None
    observed_flip_rate: float | None = None
    predicted_flip_rate: float | None = None
    symmetry_p: float | None = None
    # The same quantity as `observed_flip_rate` for the worst-agreeing hosted model rather than
    # for the panel. `points_sd` sizes from this one; see its docstring for why the pooled figure
    # is the wrong number for a gate and the right one for `resampling_note`.
    worst_hosted_flip_rate: float | None = None

    @property
    def measured(self) -> bool:
        """False when the bank has no repeated administrations to measure agreement over."""
        return self.n_repeated_cells > 0

    @property
    def from_own_run(self) -> bool:
        """True when this is the temperature-0 test-retest rather than the HELM stand-in."""
        return self.score_move_points is not None

    def points_sd(self, n_items: int, *, pooled: bool = False) -> float:
        """Standard deviation of a score change, in points, on a test of this many items.

        The number a release gate sizes itself with. Derived from the measured flip rate rather
        than from the information function, and the two differ by a factor of five: see
        `resampling_note`.

        **Sized from the worst hosted model by default, corrected 2026-09-16.** It used to use
        the pooled rate over the whole panel, which `agreement` had already been fixed not to do
        and for the same reason: a gate has to hold for the model it is watching rather than for
        the average one, and the models on a laptop agree with themselves almost perfectly at
        temperature 0. Pooling them in understated the noise by 10% in points, and against the
        worst hosted arm by a factor of 1.45. That is the false-pass direction, a gate certifying
        it can detect drift it would in fact miss. Project 03 found it.

        `pooled=True` returns the old panel-wide figure, which is the right one for describing
        the panel and the wrong one for sizing a gate. Whichever a caller takes, the report says
        which, because these two numbers differ by nearly half and a bare one is not a result.
        """
        rate = self.observed_flip_rate if pooled else self.worst_hosted_flip_rate
        if rate is None or n_items <= 0:
            return float("nan")
        return 100.0 * float(np.sqrt(rate / n_items))

    def resampling_note(self) -> str:
        """Why a drift test needs fewer items than the information function says it does."""
        if self.observed_flip_rate is None or self.predicted_flip_rate is None:
            return "Not measured on this bank."
        ratio = self.predicted_flip_rate / self.observed_flip_rate
        return (
            f"A model re-asked the same item changes its answer {self.observed_flip_rate:.4f} of "
            f"the time, against {self.predicted_flip_rate:.4f} predicted by 2p(1-p), "
            f"{ratio:.1f} times fewer. The response model's p describes how models at one "
            "ability differ from each other, not how one model differs from itself, so a test "
            "comparing a model with its own earlier self on the same items sits in the smaller "
            "variance and needs fewer items than the information function implies."
        )

    def describe(self) -> str:
        if not self.measured:
            return f"no repeated cells in this bank ({self.source}). {self.caveat}"
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


def _worst_hosted_flip_rate(payload: dict[str, Any]) -> float | None:
    """How often the least self-consistent hosted model changed its own answer.

    Written out by `mselect retest` since 2026-09-16. Derived from the per-model agreement map
    when it is absent, so that an artifact generated before that date still sizes a gate from a
    hosted model rather than from a laptop. The two routes differ only in how they weight models
    with unequal item counts, which on this design is nothing.
    """
    direct = payload.get("worst_hosted_flip_rate")
    if direct is not None:
        return float(direct)
    agreement: dict[str, float] = payload.get("agreement") or {}
    hosted = [v for k, v in agreement.items() if not k.startswith("local-")]
    return 1.0 - min(hosted) if hosted else None


def _own_run_retest(version: str) -> Reliability | None:
    """The committed summary of the own-run test-retest, if this bank has one.

    Committed rather than left in `out/` on purpose: `out/` is gitignored, and this function is
    what project 03 imports, so a result that lives only in a run directory is a result the
    consumer can never see. The file holds agreement, flip counts and the derived floor, and no
    item text or replies, so PLAN.md section 13.5 is untouched.
    """
    path = paths.ROOT / "mselect" / "config" / f"own-run-retest-{version}.json"
    if not path.is_file():
        return None
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return Reliability(
        agreement=float(payload["worst_hosted_agreement"]),
        n_repeated_cells=int(payload["n_models"]) * int(payload["n_items"]),
        source=(
            f"{payload['n_models']} current models answering the same {payload['n_items']} items "
            f"twice at temperature 0, a day apart, measured {payload['measured']}"
        ),
        caveat=(
            "The worst hosted model is reported, for `agreement` and for the flip rate "
            "`points_sd` sizes from, because a gate has to hold for the model it is watching "
            "rather than for the average one. The models on a laptop agree almost perfectly and "
            "are excluded from both figures for that reason. `observed_flip_rate` is the "
            "panel-wide number and stays panel-wide, because `resampling_note` compares it "
            "against a panel-wide prediction and the two have to be the same population."
        ),
        score_move_points=float(payload["largest_score_move_points"]),
        observed_flip_rate=float(payload["observed_flip_rate"]),
        predicted_flip_rate=float(payload["predicted_flip_rate"]),
        symmetry_p=float(payload["symmetry_p_pooled"]),
        worst_hosted_flip_rate=_worst_hosted_flip_rate(payload),
    )


@cache
def reliability(version: str = bank_io.DEFAULT_VERSION) -> Reliability:
    """The reliability figure this bank can support today.

    **Measured 2026-09-14** for bank v1: eleven current models answered the same 500 items twice
    at temperature 0 a day apart, through this project's own runner. Where that result is
    present it is returned, because it is the experiment PLAN.md section 4.3 asks for rather
    than a stand-in for it.

    Where it is absent, the fallback is free and real but weaker: HELM re-ran some models across
    releases, so some model-item cells appear twice and do not always agree. Those repeats differ
    in run date and release as well as in sampling, so they bound benchmark noise from below
    rather than measuring it. `from_own_run` says which of the two a caller is holding.
    """
    own_run = _own_run_retest(version)
    if own_run is not None:
        return own_run
    manifest: dict[str, Any] = dict(bank_io.default_bank(version).manifest)
    cells = int(manifest["repeated_cells"])
    if cells == 0:
        # Bank v2 takes the latest run of each task and nothing else, so no model answers any
        # item twice in it. Returning 100 percent agreement over zero cells would be a number
        # that looks like evidence and is not one.
        return Reliability(
            agreement=float("nan"),
            n_repeated_cells=0,
            source=f"bank {version}: one administration per model and task",
            caveat=(
                "This bank has no repeated model-item cells, so it carries no reliability "
                "figure at all. Bank v1 does, from HELM's overlapping administrations."
            ),
        )
    return Reliability(
        agreement=float(manifest["repeated_cell_agreement"]),
        n_repeated_cells=cells,
        source="the same model answering the same item in two HELM administrations",
        caveat=(
            "Not the temperature-0 test-retest of PLAN.md section 4.3, which needs the own-run "
            "panel: these repeats differ in run date and HELM release as well as in sampling, so "
            "they bound benchmark noise from below rather than measuring it directly."
        ),
    )
