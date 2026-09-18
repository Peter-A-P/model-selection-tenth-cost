"""Items that are the same question twice, under two item ids.

Found on 2026-09-17 by accident, which is worth saying because it is the reason this module
exists rather than a paragraph in a notebook. The `local-mid-a` run came back with 2,999 records
on an arm where every other model has 3,000, and re-running it said "nothing to do". Both facts
have one cause: `mmlu_pro` instance `id3054` and `mmlu` `high_school_biology` instance `id252`
are the same question, so they produce one `request_sha256`, and `records.done` keys on the
request hash rather than on the cell. One twin's record was lost in a killed chunk and the other
settled the hash for both. PLAN.md section 15.37.

MMLU-Pro was built partly out of MMLU, which is not a secret. What was not known here is how much
of that overlap is inside this bank, and a bank that measures local dependence cannot leave the
question open: **a duplicated item is local dependence by construction.** Two labels for one
question load on whatever they both load on, twice, and no Q3 threshold is needed to see it.

What this can and cannot see. The key below is the question text and the set of option texts,
both whitespace-normalised, with the options sorted so that two publications of one question are
still one question when the options are listed in a different order. It is exact-match only: a
question reworded by a comma is two questions here, so every number this produces is a floor. It
also covers only the administrable pool, because an item with no text cannot be compared against
anything, and 446 items of bank v1 have none (GPQA is withheld at its authors' request).

A group whose members disagree about the answer is reported separately and is a different kind of
problem: the same question with two different keys is not redundancy, it is a contradiction, and
one of the two is scoring models wrongly.
"""

from __future__ import annotations

import collections
import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from mselect.runner.administer import Item

# The separator is a unit separator rather than a space or a newline, so that a question ending
# in the text of its first option cannot collide with the same bytes split the other way.
_UNIT: Final = "\x1f"
_WHITESPACE: Final = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Collapse runs of whitespace and strip, and change nothing else.

    Deliberately conservative. Case, punctuation and LaTeX are all left alone: two questions that
    differ only in capitalisation are probably the same question, but "probably" is how a
    duplicate count becomes an argument rather than a measurement.
    """
    return _WHITESPACE.sub(" ", text).strip()


def question_key(item: Item) -> str:
    """A content hash over what a model is actually asked, ignoring the order of the options.

    The options are sorted rather than kept in their published order because the position-bias
    experiment already establishes that order is a presentation choice and not part of the
    question. Two items that differ only in option order are the same item asked twice.
    """
    parts = [normalise(item.question), *sorted(normalise(option) for option in item.options)]
    return hashlib.sha256(_UNIT.join(parts).encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class Duplicate:
    """One question that appears in the bank more than once."""

    key: str
    item_ids: tuple[str, ...]
    benchmarks: tuple[str, ...]
    answers: tuple[str, ...]
    n_options: int
    question: str

    @property
    def size(self) -> int:
        return len(self.item_ids)

    @property
    def cross_benchmark(self) -> bool:
        return len(set(self.benchmarks)) > 1

    @property
    def keys_disagree(self) -> bool:
        """Whether the copies are scored against different answers.

        An empty answer is not a disagreement: free-response keys are extracted elsewhere and an
        item that carries none is uninformative here rather than contradictory.
        """
        answered = {a for a in self.answers if a}
        return len(answered) > 1


def find(items: Sequence[Item]) -> tuple[Duplicate, ...]:
    """Every question in `items` that appears more than once, worst first.

    Ordered by group size and then by item id, so the output is stable across runs and a diff of
    two scans shows what changed rather than what moved.
    """
    groups: dict[str, list[Item]] = collections.defaultdict(list)
    for item in items:
        groups[question_key(item)].append(item)

    found = [
        Duplicate(
            key=key,
            item_ids=tuple(sorted(i.item_id for i in group)),
            benchmarks=tuple(sorted(i.benchmark for i in group)),
            answers=tuple(normalise(i.answer) for i in sorted(group, key=lambda i: i.item_id)),
            n_options=len(group[0].options),
            question=normalise(group[0].question),
        )
        for key, group in groups.items()
        if len(group) > 1
    ]
    return tuple(sorted(found, key=lambda d: (-d.size, d.item_ids)))


@dataclass(frozen=True, slots=True)
class Scan:
    """What a whole-bank scan found, with the denominators it was measured against."""

    duplicates: tuple[Duplicate, ...]
    n_scanned: int
    n_unscannable: int

    @property
    def n_items_involved(self) -> int:
        return sum(d.size for d in self.duplicates)

    @property
    def n_distinct_questions(self) -> int:
        return self.n_scanned - (self.n_items_involved - len(self.duplicates))

    @property
    def contradictions(self) -> tuple[Duplicate, ...]:
        return tuple(d for d in self.duplicates if d.keys_disagree)

    def summary(self) -> str:
        pairs = len(self.duplicates)
        cross = sum(1 for d in self.duplicates if d.cross_benchmark)
        return (
            f"{self.n_scanned:,} items carry {self.n_distinct_questions:,} distinct questions: "
            f"{pairs} question{'' if pairs == 1 else 's'} appear more than once, covering "
            f"{self.n_items_involved} items, {cross} of them across two benchmarks. "
            f"{self.n_unscannable:,} items have no text and could not be compared. "
            f"{len(self.contradictions)} group{'' if len(self.contradictions) == 1 else 's'} "
            f"disagree about the answer."
        )


def scan(items: Sequence[Item], *, unscannable: int = 0) -> Scan:
    return Scan(duplicates=find(items), n_scanned=len(items), n_unscannable=unscannable)
