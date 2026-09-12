"""Rebuilding administrable items from the HELM cache.

The bank stores no item text (PLAN.md section 13.5), so a run that asks a model one of these
questions has to reconstruct it here, from the cached release the bank was built from. That is
a lookup and not a second source: the item's identity is still `build.item_hash` over exactly
the bytes HELM published, so an item administered here is the same item the bank calibrated.

Reconstructing is not copying, and this module is where that shows. A public per-instance
release is built to let you check a score, not to let you re-ask the question, and three of the
eight benchmarks in bank v1 will not survive a naive round trip. Each is handled once, here,
rather than in the runner, because getting any of them wrong is not a crash: it is a plausible
number that is wrong, paid for at full price.

* **GPQA is withheld.** Every instance's text is a placeholder, `[encrypted_text_N]`, because
  its authors ask that the questions not be published in scrapeable form. HELM honours that and
  so does this: those items are excluded with the reason recorded. They stay in the bank, which
  needs only the 0 or 1, and they can never be in an own run.
* **A LegalBench item carries one option, and it is the answer.** HELM stores the reference it
  tagged correct and not the distractors, so administering an item as it arrives would offer
  the model a single choice which is the right one. The real task is a classification over a
  small fixed label set, so the options are rebuilt from the labels that task actually uses,
  which is the set of distinct answers across its own instances.
* **A free-response answer key is a worked solution.** GSM8K's reference ends "The answer is
  120." and MATH's ends in a `\\boxed{}`, and neither is something to compare a reply against.
  The final value is extracted, and the check that it is the right value is that the reference
  solution itself grades correct against it, which it does for all 1,437 of them.

What is not fixable is recorded rather than dropped. `Pool.excluded` names every item that
cannot be administered and why, so the suite that gets run is a decision with a reason behind
it instead of whatever happened to load.
"""

from __future__ import annotations

import collections
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import polars as pl

from mselect import paths
from mselect.data import build, helm
from mselect.runner import parse
from mselect.runner.administer import MULTIPLE_CHOICE, Item

# HELM's stand-in for text its source asks it not to republish. Matched as a prefix because the
# placeholders are numbered per instance.
WITHHELD: Final = "[encrypted_text"

# A choice needs something to choose between. One option is not a question.
MIN_OPTIONS: Final = 2

WITHHELD_REASON: Final = "the source withholds the question text"
NO_KEY_REASON: Final = "no answer key"
UNGRADED_REASON: Final = "no final answer in the reference solution"
NOT_AN_OPTION_REASON: Final = "the answer key is not one of the options"
ONE_LABEL_REASON: Final = "the task has fewer than two labels to choose between"


@dataclass(frozen=True, slots=True)
class Excluded:
    """An item in the bank that cannot be asked of a model, and the reason it cannot."""

    item_id: str
    benchmark: str
    reason: str


@dataclass(frozen=True, slots=True)
class Pool:
    """Every bank item that can be administered, and every one that cannot."""

    version: str
    items: tuple[Item, ...]
    excluded: tuple[Excluded, ...]

    def by_benchmark(self) -> dict[str, int]:
        return dict(sorted(collections.Counter(i.benchmark for i in self.items).items()))

    def by_reason(self) -> dict[str, int]:
        return dict(sorted(collections.Counter(e.reason for e in self.excluded).items()))

    def index(self) -> dict[str, Item]:
        return {item.item_id: item for item in self.items}

    def summary(self) -> str:
        counts = ", ".join(f"{name} {n:,}" for name, n in self.by_benchmark().items())
        return (
            f"bank {self.version}: {len(self.items):,} items can be administered "
            f"({counts}); {len(self.excluded):,} cannot"
        )


def final_answer(solution: str) -> str | None:
    """The value a free-response reference settles on, or None when it does not settle on one.

    The reference is a worked solution, so this is the same extraction the runner applies to a
    model's reply. Using one function for both is the point: a key recovered by a rule the
    grader does not share would be a key the grader cannot match.
    """
    answer = parse.parse_math(solution)
    return answer if answer and answer.strip() else None


def _label_sets(
    seen: dict[tuple[str, str], set[str]], benchmark: str, scenario_key: str
) -> list[str]:
    """The labels one classification task chooses between, sorted so a letter is stable."""
    return sorted(seen.get((benchmark, scenario_key), set()))


def _scan(
    client: helm.Client, wanted: set[str]
) -> Iterator[tuple[str, helm.Scenario, str, helm.Instance]]:
    """Every cached instance that is an item of the bank, with the scenario that produced it."""
    for scenario in helm.SCENARIOS:
        for run in helm.instance_runs(helm.runs_for(client, scenario)):
            for instance in helm.instances(client, run).values():
                if not instance.text:
                    continue
                item_id = build.item_hash(
                    scenario.benchmark, instance.text, instance.options, instance.answer
                )
                if item_id in wanted:
                    yield item_id, scenario, run.scenario_key, instance


def administrable(
    version: str = "v1", *, cache_root: Path | None = None, bank_root: Path | None = None
) -> Pool:
    """Rebuild every item of a bank that a model can actually be asked.

    Reads the cache only. Nothing here fetches, because an own run should not discover halfway
    through that a release moved: `mselect bank-fetch` is what fills the cache.
    """
    items_table = pl.read_parquet((bank_root or paths.BANK) / version / "items.parquet")
    wanted = set(items_table["item_id"].to_list())

    collected: dict[str, tuple[helm.Scenario, str, helm.Instance]] = {}
    labels: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    with helm.Client(helm.Cache(cache_root or paths.HELM_CACHE)) as client:
        for item_id, scenario, scenario_key, instance in _scan(client, wanted):
            if item_id in collected:
                continue
            collected[item_id] = (scenario, scenario_key, instance)
            if instance.answer:
                labels[(scenario.benchmark, scenario_key)].add(instance.answer)

    built: list[Item] = []
    excluded: list[Excluded] = []

    def refuse(item_id: str, benchmark: str, reason: str) -> None:
        excluded.append(Excluded(item_id=item_id, benchmark=benchmark, reason=reason))

    for item_id in items_table["item_id"].to_list():
        entry = collected.get(item_id)
        if entry is None:
            continue  # not in the cache at all; `bank-fetch` decides that, not this
        scenario, scenario_key, instance = entry
        benchmark = scenario.benchmark

        if instance.text.startswith(WITHHELD):
            refuse(item_id, benchmark, WITHHELD_REASON)
            continue
        if not instance.answer:
            refuse(item_id, benchmark, NO_KEY_REASON)
            continue

        if scenario.kind != MULTIPLE_CHOICE:
            answer = final_answer(instance.answer)
            if answer is None:
                refuse(item_id, benchmark, UNGRADED_REASON)
                continue
            built.append(
                Item(
                    item_id=item_id,
                    benchmark=benchmark,
                    kind=scenario.kind,
                    question=instance.text,
                    answer=answer,
                )
            )
            continue

        options = list(instance.options)
        if len(options) < MIN_OPTIONS:
            options = _label_sets(labels, benchmark, scenario_key)
            if len(options) < MIN_OPTIONS:
                refuse(item_id, benchmark, ONE_LABEL_REASON)
                continue
        if instance.answer not in options:
            refuse(item_id, benchmark, NOT_AN_OPTION_REASON)
            continue
        built.append(
            Item(
                item_id=item_id,
                benchmark=benchmark,
                kind=scenario.kind,
                question=instance.text,
                options=tuple(options),
                answer=instance.answer,
            )
        )

    return Pool(version=version, items=tuple(built), excluded=tuple(excluded))
