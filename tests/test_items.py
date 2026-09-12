"""Rebuilding administrable items from the cache, with no network and no money.

Three of bank v1's eight benchmarks do not survive a naive round trip from a public release
back into a question, and each failure is quiet: a withheld question that looks like a question,
a single option that is the answer, a key that is a worked solution. None of them raises. All
of them would produce a number, at full price, that means nothing. So each is pinned here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import pytest

from mselect import paths
from mselect.data import build, helm
from mselect.runner import administer, items

BUCKET = helm.BUCKET
SCENARIOS = {s.benchmark: s for s in helm.SCENARIOS}


@dataclass(frozen=True)
class Spec:
    """One instance to seed, in the terms the test cares about rather than HELM's JSON."""

    instance_id: str
    text: str
    options: list[str]
    answer: str

    def as_helm(self) -> dict[str, object]:
        """The same instance shaped the way HELM's instances.json shapes it."""
        return {
            "id": self.instance_id,
            "input": {"text": self.text},
            "references": [
                {"output": {"text": option}, "tags": ["correct"] if option == self.answer else []}
                for option in self.options
            ],
        }


def _seed(cache_root: Path, bank_root: Path, plan: dict[str, list[Spec]]) -> None:
    """Write a cache and a bank holding exactly the instances in `plan`, keyed by benchmark."""
    cache = helm.Cache(cache_root)
    rows: list[dict[str, object]] = []
    for benchmark, specs in plan.items():
        scenario = SCENARIOS[benchmark]
        run_name = f"{scenario.prefix}:subject=x,model=org_m"
        suite = "suite"
        cache.put(
            f"{BUCKET}/{scenario.project}/benchmark_output/releases/{scenario.release}"
            f"/runs_to_run_suites.json",
            json.dumps({run_name: suite}).encode(),
        )
        run = helm.Run(scenario, run_name, suite, "org/m", f"{scenario.prefix}:subject=x")
        cache.put(f"{run.url}/instances.json", json.dumps([s.as_helm() for s in specs]).encode())
        rows.extend(
            {
                "item_id": build.item_hash(benchmark, s.text, s.options, s.answer),
                "benchmark": benchmark,
                "kind": scenario.kind,
            }
            for s in specs
        )
    # Every scenario the loader scans must have a run map, even an empty one, or the client
    # raises on a cache miss the way it would for a release that never downloaded.
    for scenario in helm.SCENARIOS:
        url = (
            f"{BUCKET}/{scenario.project}/benchmark_output/releases/{scenario.release}"
            f"/runs_to_run_suites.json"
        )
        if cache.get(url) is None:
            cache.put(url, b"{}")
    (bank_root / "v1").mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(bank_root / "v1" / "items.parquet")


def test_a_withheld_question_is_excluded_and_says_so(tmp_path: Path) -> None:
    """GPQA's text is a placeholder. An own run cannot ask a question it was not given."""
    _seed(
        tmp_path / "cache",
        tmp_path / "bank",
        {
            "gpqa": [
                Spec("id0", "[encrypted_text_0]", ["[encrypted_text_1]"], "[encrypted_text_1]")
            ],
            "commonsense": [Spec("id1", "Real question?", ["a", "b", "c", "d"], "b")],
        },
    )
    pool = items.administrable("v1", cache_root=tmp_path / "cache", bank_root=tmp_path / "bank")
    assert [i.benchmark for i in pool.items] == ["commonsense"]
    assert [(e.benchmark, e.reason) for e in pool.excluded] == [("gpqa", items.WITHHELD_REASON)]


def test_a_one_option_item_takes_its_tasks_labels(tmp_path: Path) -> None:
    """LegalBench stores only the correct reference. The choice is the task's label set."""
    _seed(
        tmp_path / "cache",
        tmp_path / "bank",
        {
            "legalbench": [
                Spec("id0", "Is this generic?", ["Yes"], "Yes"),
                Spec("id1", "Is that generic?", ["No"], "No"),
            ]
        },
    )
    pool = items.administrable("v1", cache_root=tmp_path / "cache", bank_root=tmp_path / "bank")
    assert len(pool.items) == 2
    for item in pool.items:
        assert item.options == ("No", "Yes"), "both labels, sorted so the letter is stable"
        assert administer.expected_key(item) is not None
    assert {i.answer for i in pool.items} == {"Yes", "No"}


def test_a_task_with_one_label_is_refused(tmp_path: Path) -> None:
    """One label is not a choice, and offering it would score every model 100 percent."""
    _seed(
        tmp_path / "cache",
        tmp_path / "bank",
        {"legalbench": [Spec("id0", "Only ever yes?", ["Yes"], "Yes")]},
    )
    pool = items.administrable("v1", cache_root=tmp_path / "cache", bank_root=tmp_path / "bank")
    assert pool.items == ()
    assert [e.reason for e in pool.excluded] == [items.ONE_LABEL_REASON]


def test_a_free_response_key_is_the_final_answer_not_the_working(tmp_path: Path) -> None:
    """The reference is a worked solution. Grading a reply against all of it grades it wrong."""
    solution = "Rice took 30 minutes, pork took 20+30 = 50. The answer is 120."
    _seed(
        tmp_path / "cache",
        tmp_path / "bank",
        {"gsm8k": [Spec("id0", "How many minutes?", [solution], solution)]},
    )
    pool = items.administrable("v1", cache_root=tmp_path / "cache", bank_root=tmp_path / "bank")
    assert len(pool.items) == 1
    item = pool.items[0]
    assert item.answer == "120", "the value, not the derivation"
    assert administer.score(item, administer.Reply(text="\\boxed{120}"))[1] == 1
    assert administer.score(item, administer.Reply(text="\\boxed{119}"))[1] == 0


def test_a_reference_with_no_final_answer_is_refused(tmp_path: Path) -> None:
    _seed(
        tmp_path / "cache",
        tmp_path / "bank",
        {"gsm8k": [Spec("id0", "How many?", ["no number here"], "no number here")]},
    )
    pool = items.administrable("v1", cache_root=tmp_path / "cache", bank_root=tmp_path / "bank")
    assert pool.items == ()
    assert [e.reason for e in pool.excluded] == [items.UNGRADED_REASON]


def test_final_answer_reads_both_reference_styles() -> None:
    assert items.final_answer("so it is 40+50+30 = 120 minutes. The answer is 120.") == "120"
    assert items.final_answer("so $x \\cdot y = \\boxed{364}$.") == "364"
    assert items.final_answer("\\boxed{\\frac{\\pi}{3}}") == "\\frac{\\pi}{3}"
    assert items.final_answer("no value at all") is None


CACHE_PRESENT = (paths.HELM_CACHE / "blobs").exists()


@pytest.mark.skipif(not CACHE_PRESENT, reason="needs the HELM cache from `mselect bank-fetch`")
def test_every_administrable_item_scores_one_against_its_own_key() -> None:
    """The check worth having before spending: can each item grade its own reference right?

    An item that cannot is one whose key, prompt and grader disagree, and every model would be
    marked wrong on it. This runs over the whole bank because 105 of the 19,919 failed it the
    first time it ran, all of them symbolic MATH answers sent bare rather than boxed, which was
    the check being wrong rather than the pipeline. The reply here is shaped the way the
    template asks for it, which is the only shape the grader will ever see.
    """
    pool = items.administrable("v1")
    assert len(pool.items) > 19_000, "the cache is present but nearly empty"

    wrong: list[str] = []
    for item in pool.items:
        key = administer.expected_key(item)
        assert key is not None, f"{item.item_id} is administrable but has no key"
        text = "\\boxed{" + item.answer + "}" if item.free_response else key
        if administer.score(item, administer.Reply(text=text))[1] != 1:
            wrong.append(f"{item.benchmark} {item.item_id}")
    assert not wrong, f"{len(wrong)} items do not grade their own reference: {wrong[:5]}"


@pytest.mark.skipif(not CACHE_PRESENT, reason="needs the HELM cache from `mselect bank-fetch`")
def test_gpqa_is_the_only_benchmark_that_cannot_be_administered() -> None:
    pool = items.administrable("v1")
    assert pool.by_reason() == {items.WITHHELD_REASON: 446}
    assert {e.benchmark for e in pool.excluded} == {"gpqa"}
    assert "gpqa" not in pool.by_benchmark()
