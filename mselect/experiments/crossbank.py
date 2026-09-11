"""Does an item bank transfer? The same questions, calibrated twice.

Bank v1 and bank v2 were built from different sources, and by luck they overlap exactly: every
one of the 998 MMLU-Pro items HELM sampled is also in the Open LLM Leaderboard's 12,032. The two
calibrations of those items have almost nothing else in common.

|  | bank v1 | bank v2 |
|---|---|---|
| panel | 150 models, mostly frontier APIs | 400 open-weight leaderboard submissions |
| harness | HELM, chain-of-thought prompting | lm-eval-harness, 5-shot log-likelihood |
| scoring | the model's stated final answer | the highest-likelihood option |

So the question a release gate has to ask before importing anyone's item parameters gets a
number here: if you calibrate difficulty on one panel and one harness, does it still mean
anything on another? The answer is a correlation with an interval, not a yes.

Ability scales are not comparable across the two banks. Each is identified against a standard
normal prior over its own panel, so a difficulty of 1.2 in v1 and 1.2 in v2 are not the same
quantity, and nothing here compares them as levels. What is comparable is the ordering: the
correlation between the two difficulty vectors, and between the two discrimination vectors, over
the items both banks contain.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

import numpy as np
import polars as pl
from numpy.typing import NDArray

from mselect import paths
from mselect.data import bank as bank_io
from mselect.data import ollm
from mselect.experiments.analysis import Interval

BRIDGE_TASK = "leaderboard_mmlu_pro"
BRIDGE_BENCHMARK = "mmlu_pro"
BRIDGE_FIELD = "question_id"
RESAMPLES = 2000


@dataclass(frozen=True, slots=True)
class Agreement:
    """How two calibrations of the same items agree, with an interval on every number."""

    n_items: int
    difficulty_pearson: Interval
    difficulty_spearman: Interval
    discrimination_pearson: Interval
    discrimination_spearman: Interval
    hardest_decile_recovered: Interval
    weakest_decile_recovered: Interval

    def describe(self) -> str:
        return (
            f"{self.n_items} items in both banks: difficulty correlates "
            f"{self.difficulty_pearson.fmt(percent=False)}, discrimination "
            f"{self.discrimination_pearson.fmt(percent=False)}; "
            f"{self.hardest_decile_recovered.fmt()} of one bank's hardest tenth is in the "
            f"other's hardest tenth"
        )


def _bootstrap_statistic(
    left: NDArray[np.float64],
    right: NDArray[np.float64],
    statistic: Callable[[NDArray[np.float64], NDArray[np.float64]], float],
    *,
    resamples: int = RESAMPLES,
    seed: int = 0,
) -> Interval:
    """A percentile bootstrap over items, because items are what would vary in another sample."""
    keep = np.isfinite(left) & np.isfinite(right)
    a, b = left[keep], right[keep]
    if a.size < 3:
        return Interval(float("nan"), float("nan"), float("nan"), int(a.size))
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples)
    for index in range(resamples):
        pick = rng.integers(0, a.size, a.size)
        draws[index] = statistic(a[pick], b[pick])
    return Interval(
        statistic(a, b),
        float(np.nanquantile(draws, 0.025)),
        float(np.nanquantile(draws, 0.975)),
        int(a.size),
    )


def _pearson(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    order_a = np.argsort(np.argsort(a)).astype(float)
    order_b = np.argsort(np.argsort(b)).astype(float)
    return _pearson(order_a, order_b)


def _decile_overlap(left: NDArray[np.float64], right: NDArray[np.float64], *, top: bool) -> float:
    """The share of one bank's extreme tenth that the other bank also puts in its extreme tenth.

    A correlation can be respectable while the decision a consumer actually makes, "which items
    are the hard ones", still moves. This is that decision, measured directly.
    """
    size = max(1, left.size // 10)
    pick_left = np.argsort(left)[-size:] if top else np.argsort(left)[:size]
    pick_right = np.argsort(right)[-size:] if top else np.argsort(right)[:size]
    return float(len(set(pick_left.tolist()) & set(pick_right.tolist())) / size)


def bridge(
    *,
    first: str = "v1",
    second: str = "v2",
    progress: Callable[[str], None] = lambda _: None,
) -> pl.DataFrame:
    """The item pairs the two banks share, matched on the source dataset's own question id.

    HELM keeps MMLU-Pro's `question_id` as its instance id, prefixed with "id". The leaderboard
    keeps it inside the document column, which is read one field at a time so that matching the
    banks costs 1.9 MB rather than the 72 MB file. Nothing here needs the question text, and no
    question text is written anywhere.
    """
    bank_one = bank_io.load(first)
    bank_two = bank_io.load(second)

    left = (
        bank_one.items.filter(pl.col("benchmark") == BRIDGE_BENCHMARK)
        .select("item_id", "instance_id")
        .with_columns(pl.col("instance_id").str.strip_prefix("id").alias("question_id"))
        .rename({"item_id": f"item_{first}"})
        .drop("instance_id")
    )
    if left.height == 0:
        raise ValueError(f"bank {first} has no {BRIDGE_BENCHMARK} items to bridge with")

    task = next(t for t in ollm.TASKS if t.suffix == BRIDGE_TASK)
    panel = ollm.load_panel(paths.OLLM_CACHE)
    cache = ollm.Cache(paths.ensure(paths.OLLM_CACHE))
    with ollm.Client(cache, ollm.token_from_env()) as client:
        keys = ollm.read_doc_field(client, panel.members[0], task, BRIDGE_FIELD)
    progress(f"{keys.height:,} {BRIDGE_TASK} documents keyed by {BRIDGE_FIELD}")

    right = (
        bank_two.items.filter(pl.col("benchmark") == BRIDGE_BENCHMARK)
        .select("item_id", "instance_id")
        .with_columns(
            pl.col("instance_id").str.split("#").list.last().cast(pl.Int64).alias("doc_id")
        )
        .join(keys, on="doc_id", how="inner")
        .rename({"item_id": f"item_{second}", BRIDGE_FIELD: "question_id"})
        .select(f"item_{second}", "question_id")
    )
    matched = left.join(right, on="question_id", how="inner")
    progress(f"{matched.height:,} items are in both banks")
    return matched


def compare(
    *,
    first: str = "v1",
    second: str = "v2",
    kind: str = "2pl",
    seed: int = 0,
    progress: Callable[[str], None] = lambda _: None,
) -> Agreement:
    """Correlate the two calibrations of the shared items, with bootstrap intervals."""
    matched = bridge(first=first, second=second, progress=progress)
    bank_one = bank_io.load(first)
    bank_two = bank_io.load(second)
    _, params_one, _ = bank_io.load_params(bank_one, kind)
    _, params_two, _ = bank_io.load_params(bank_two, kind)

    joined = (
        matched.join(
            params_one.select("item_id", "a", "b").rename({"item_id": f"item_{first}"}),
            on=f"item_{first}",
            how="inner",
        )
        .rename({"a": "a_one", "b": "b_one"})
        .join(
            params_two.select("item_id", "a", "b").rename({"item_id": f"item_{second}"}),
            on=f"item_{second}",
            how="inner",
        )
        .rename({"a": "a_two", "b": "b_two"})
    )
    b_one = joined["b_one"].to_numpy()
    b_two = joined["b_two"].to_numpy()
    a_one = joined["a_one"].to_numpy()
    a_two = joined["a_two"].to_numpy()

    return Agreement(
        n_items=joined.height,
        difficulty_pearson=_bootstrap_statistic(b_one, b_two, _pearson, seed=seed),
        difficulty_spearman=_bootstrap_statistic(b_one, b_two, _spearman, seed=seed),
        discrimination_pearson=_bootstrap_statistic(a_one, a_two, _pearson, seed=seed),
        discrimination_spearman=_bootstrap_statistic(a_one, a_two, _spearman, seed=seed),
        hardest_decile_recovered=_bootstrap_statistic(
            b_one, b_two, lambda x, y: _decile_overlap(x, y, top=True), seed=seed
        ),
        weakest_decile_recovered=_bootstrap_statistic(
            a_one, a_two, lambda x, y: _decile_overlap(x, y, top=False), seed=seed
        ),
    )


def run(
    *,
    first: str = "v1",
    second: str = "v2",
    kind: str = "2pl",
    seed: int = 0,
    progress: Callable[[str], None] = lambda _: None,
) -> str:
    """Run the comparison and write it where the report can read it."""
    agreement = compare(first=first, second=second, kind=kind, seed=seed, progress=progress)
    payload = {
        "run": datetime.now(UTC).date().isoformat(),
        "first_bank": first,
        "second_bank": second,
        "fit": kind,
        "seed": seed,
        "agreement": asdict(agreement),
    }
    out = paths.out_for(second) / f"crossbank-{first}-{kind}.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return agreement.describe()
