"""One question under two item ids: the exact end of the local dependence scale.

Found on 2026-09-17 because a killed run came back one record short and a rerun said there was
nothing to do. PLAN.md section 15.37.
"""

from __future__ import annotations

from mselect.experiments import duplicates
from mselect.runner.administer import Item


def _item(item_id: str, question: str, options: tuple[str, ...], **kw: str) -> Item:
    return Item(
        item_id=item_id,
        benchmark=kw.get("benchmark", "mmlu"),
        kind="multiple_choice",
        question=question,
        options=options,
        answer=kw.get("answer", options[0] if options else ""),
    )


def test_the_same_question_under_two_ids_is_one_question() -> None:
    found = duplicates.find(
        [
            _item(
                "aaa", "Which organelle makes ATP?", ("mitochondrion", "ribosome"), benchmark="mmlu"
            ),
            _item(
                "bbb",
                "Which organelle makes ATP?",
                ("mitochondrion", "ribosome"),
                benchmark="mmlu_pro",
            ),
            _item("ccc", "Which organelle stores DNA?", ("nucleus", "ribosome")),
        ]
    )
    assert len(found) == 1
    assert found[0].item_ids == ("aaa", "bbb")
    assert found[0].benchmarks == ("mmlu", "mmlu_pro")
    assert found[0].cross_benchmark
    assert not found[0].keys_disagree


def test_option_order_is_presentation_and_not_part_of_the_question() -> None:
    """Finding 7 is the whole argument: option order is a presentation choice, so two items that
    differ only in it are the same item asked twice."""
    found = duplicates.find(
        [
            _item("aaa", "Pick one.", ("alpha", "beta", "gamma")),
            _item("bbb", "Pick one.", ("gamma", "alpha", "beta")),
        ]
    )
    assert len(found) == 1 and found[0].size == 2


def test_whitespace_is_normalised_and_wording_is_not() -> None:
    same = duplicates.find(
        [
            _item("aaa", "Which  organelle\nmakes ATP?", ("x", "y")),
            _item("bbb", "Which organelle makes ATP?", ("x", "y")),
        ]
    )
    assert len(same) == 1
    # A comma is a different question here, on purpose: "probably the same" is how a duplicate
    # count stops being a measurement.
    assert not duplicates.find(
        [
            _item("aaa", "Which organelle makes ATP?", ("x", "y")),
            _item("bbb", "Which organelle, makes ATP?", ("x", "y")),
        ]
    )


def test_two_copies_with_different_answers_are_flagged_as_a_contradiction() -> None:
    found = duplicates.find(
        [
            _item("aaa", "Pick one.", ("alpha", "beta"), answer="alpha"),
            _item("bbb", "Pick one.", ("alpha", "beta"), answer="beta"),
        ]
    )
    assert len(found) == 1 and found[0].keys_disagree


def test_a_missing_answer_key_is_not_a_contradiction() -> None:
    found = duplicates.find(
        [
            _item("aaa", "Pick one.", ("alpha", "beta"), answer="alpha"),
            _item("bbb", "Pick one.", ("alpha", "beta"), answer=""),
        ]
    )
    assert len(found) == 1 and not found[0].keys_disagree


def test_the_scan_counts_distinct_questions_rather_than_items() -> None:
    items = [
        _item("aaa", "Q1", ("x", "y")),
        _item("bbb", "Q1", ("x", "y")),
        _item("ccc", "Q2", ("x", "y")),
        _item("ddd", "Q3", ("x", "y")),
    ]
    scan = duplicates.scan(items, unscannable=7)
    assert scan.n_scanned == 4
    assert scan.n_items_involved == 2
    # Four items, one of which is a second copy, so three questions.
    assert scan.n_distinct_questions == 3
    assert scan.n_unscannable == 7
    assert "3 distinct questions" in scan.summary()


def test_a_bank_with_no_duplicates_says_so_without_dividing_by_anything() -> None:
    scan = duplicates.scan([_item("aaa", "Q1", ("x", "y"))])
    assert scan.duplicates == ()
    assert scan.n_distinct_questions == 1
    assert scan.contradictions == ()


def test_groups_come_out_worst_first_and_stable() -> None:
    items = [
        _item("b2", "Q1", ("x", "y")),
        _item("b1", "Q1", ("x", "y")),
        _item("b3", "Q1", ("x", "y")),
        _item("a2", "Q2", ("x", "y")),
        _item("a1", "Q2", ("x", "y")),
    ]
    found = duplicates.find(items)
    assert [d.size for d in found] == [3, 2]
    assert found[0].item_ids == ("b1", "b2", "b3")
    assert duplicates.find(items) == duplicates.find(list(reversed(items)))
