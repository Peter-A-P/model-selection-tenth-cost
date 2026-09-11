"""Administering items and scoring replies, with no vendor and no money.

The scoring rules are the part that has to be right: a parser mistake recorded as a wrong
answer becomes an item statistic, and the broken-item report would then be describing this
code rather than the benchmark. Every path through it is exercised against a fake caller.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from mselect.runner import records
from mselect.runner.administer import (
    Administration,
    Item,
    Prompt,
    Reply,
    administer,
    build_prompts,
    cell_key,
    expected_key,
    score,
)
from mselect.runner.prompts import Settings

MC = Item(
    item_id="mmlu:42",
    benchmark="mmlu",
    kind="multiple_choice",
    question="What is the capital of France?",
    options=("Lyon", "Paris", "Nice", "Lille"),
    answer="Paris",
)
MATH = Item(
    item_id="math:7",
    benchmark="math",
    kind="free_response",
    question="What is 6 times 7?",
    answer="42",
)


class FakeCaller:
    """Answers from a function of the prompt, and remembers what it was asked."""

    def __init__(self, reply_for: Callable[[Prompt], Reply]) -> None:
        self._reply_for = reply_for
        self.seen: list[Prompt] = []

    def ask(self, prompts: Sequence[Prompt]) -> list[Reply]:
        self.seen.extend(prompts)
        return [self._reply_for(prompt) for prompt in prompts]


def _always(text: str | None, **kw: object) -> FakeCaller:
    return FakeCaller(lambda _p: Reply(text=text, **kw))  # type: ignore[arg-type]


# -- prompts -------------------------------------------------------------------------------


def test_build_prompts_renders_the_item_and_carries_the_fixed_settings() -> None:
    prompts = build_prompts([MC], "anthropic-haiku")
    assert len(prompts) == 1
    prompt = prompts[0]
    assert "What is the capital of France?" in prompt.user
    assert "B. Paris" in prompt.user, "options are lettered in their given order"
    assert prompt.temperature == 0.0
    assert prompt.max_tokens == Settings().max_tokens
    assert prompt.cell == cell_key("mmlu:42", "anthropic-haiku", "plain", 0)


def test_the_reasoning_template_gets_its_own_token_budget() -> None:
    settings = Settings()
    plain = build_prompts([MC], "m", template="plain")[0]
    reasoning = build_prompts([MC], "m", template="brief_reasoning")[0]
    assert plain.max_tokens == settings.max_tokens
    assert reasoning.max_tokens == settings.reasoning_max_tokens
    assert reasoning.max_tokens > plain.max_tokens


def test_a_free_response_item_cannot_be_rotated() -> None:
    """There is no option order to rotate, and accepting one silently would make the
    position-bias experiment look as though it covered items it never touched."""
    with pytest.raises(ValueError, match="free response"):
        build_prompts([MATH], "m", rotation=1)


def test_the_request_hash_changes_with_anything_that_changes_the_reply() -> None:
    base = build_prompts([MC], "m")[0]
    same = build_prompts([MC], "m")[0]
    other_model = build_prompts([MC], "other")[0]
    rotated = build_prompts([MC], "m", rotation=1)[0]
    assert base.request_sha256 == same.request_sha256
    assert base.request_sha256 != other_model.request_sha256
    assert base.request_sha256 != rotated.request_sha256


# -- scoring -------------------------------------------------------------------------------


def test_rotation_moves_the_key_without_changing_the_wording() -> None:
    assert expected_key(MC) == "B"
    assert expected_key(MC, rotation=1) == "C"
    rotated = build_prompts([MC], "m", rotation=1)[0]
    assert "C. Paris" in rotated.user
    assert "Paris" in rotated.user and "Lyon" in rotated.user


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("B", 1),
        ("The answer is B.", 1),
        ("**B**", 1),
        ("Answer: (B)", 1),
        ("A", 0),
        ("I think it is Paris, so A", 0),
    ],
)
def test_multiple_choice_replies_are_graded(reply: str, expected: int) -> None:
    parsed, correct = score(MC, Reply(text=reply))
    assert correct == expected
    assert parsed is not None


def test_an_unparseable_reply_is_unparsed_and_never_wrong() -> None:
    parsed, correct = score(MC, Reply(text="I cannot answer that."))
    assert parsed is None and correct is None, "an unreadable reply is not a wrong answer"


def test_a_failed_call_is_not_a_wrong_answer() -> None:
    parsed, correct = score(MC, Reply(text=None, error="ReadTimeout"))
    assert (parsed, correct) == (None, None)


def test_an_item_whose_key_is_not_among_its_options_cannot_be_scored() -> None:
    """A broken item, not a wrong model. Scoring it would mark every model wrong on it and
    the broken-item report would then be describing a real item as impossible."""
    broken = Item(
        item_id="mmlu:99",
        benchmark="mmlu",
        kind="multiple_choice",
        question="Q",
        options=("a", "b"),
        answer="not an option",
    )
    assert expected_key(broken) is None
    assert score(broken, Reply(text="A")) == (None, None)


def test_free_response_grading_uses_the_boxed_answer() -> None:
    assert score(MATH, Reply(text="The product is \\boxed{42}."))[1] == 1
    assert score(MATH, Reply(text="\\boxed{41}"))[1] == 0
    assert score(MATH, Reply(text="I am not sure."))[1] is None


# -- administering -------------------------------------------------------------------------


def test_administer_scores_and_records_every_cell() -> None:
    caller = _always("B", cost_usd=0.0001, input_tokens=30, output_tokens=1)
    out = administer([MC], "anthropic-haiku", caller)
    assert len(out) == 1
    record = out[0]
    assert record.correct == 1 and record.parsed == "B" and record.key == "B"
    assert record.unparsed is False and record.error is None and record.scored
    assert record.alias == "anthropic-haiku" and record.benchmark == "mmlu"
    assert record.cost_usd == 0.0001 and record.input_tokens == 30
    assert record.settings["temperature"] == 0.0


def test_administer_skips_cells_already_done(tmp_path: Path) -> None:
    """Resuming is the answer to an interrupted run, and it must not cost anything for the
    part that already happened."""
    caller = _always("B")
    first = administer([MC, MATH], "m", caller)
    assert len(first) == 2 and len(caller.seen) == 2

    again = FakeCaller(lambda _p: Reply(text="B"))
    out = administer([MC, MATH], "m", again, done=frozenset({first[0].cell}))
    assert len(out) == 1, "only the cell that was not done"
    assert out[0].item_id == "math:7"
    assert [p.item_id for p in again.seen] == ["math:7"], "nothing was sent for the done cell"


def test_administer_sends_nothing_when_everything_is_done() -> None:
    caller = _always("B")
    every = frozenset(p.cell for p in build_prompts([MC, MATH], "m"))
    assert administer([MC, MATH], "m", caller, done=every) == []
    assert caller.seen == []


def test_a_caller_that_loses_a_reply_is_an_error_not_a_misalignment() -> None:
    """Replies are matched to prompts by position. A caller that returns a different number
    of them would silently score one item against another item's answer."""

    class Dropping:
        def ask(self, prompts: Sequence[Prompt]) -> list[Reply]:
            return [Reply(text="B") for _ in prompts][:-1]

    with pytest.raises(RuntimeError, match=r"misaligned"):
        administer([MC, MATH], "m", Dropping())


def test_an_error_from_the_caller_is_recorded_rather_than_scored() -> None:
    caller = _always(None, error="ReadTimeout")
    record = administer([MC], "m", caller)[0]
    assert record.error == "ReadTimeout"
    assert record.correct is None and record.unparsed is False
    assert record.scored is False


# -- records -------------------------------------------------------------------------------


def _record(cell: str, **kw: object) -> Administration:
    base: dict[str, object] = {
        "cell": cell,
        "item_id": "mmlu:1",
        "alias": "m",
        "benchmark": "mmlu",
        "kind": "multiple_choice",
        "template": "plain",
        "rotation": 0,
        "key": "B",
        "parsed": "B",
        "correct": 1,
        "unparsed": False,
        "reply": "B",
        "cost_usd": 0.001,
        "input_tokens": 10,
        "output_tokens": 1,
        "model_returned": "claude-haiku-4-5-20251001",
        "ledger_id": 1,
        "error": None,
        "request_sha256": "abc",
    }
    base.update(kw)
    return Administration(**base)  # type: ignore[arg-type]


def test_records_round_trip_and_resume(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    assert records.done(path) == frozenset(), "a missing file is simply nothing done"
    assert records.append(path, [_record("a"), _record("b")]) == 2
    assert records.done(path) == frozenset({"a", "b"})
    records.append(path, [_record("c")])
    assert records.done(path) == frozenset({"a", "b", "c"})
    assert records.spend_usd(path) == pytest.approx(0.003)


def test_a_failed_cell_is_not_done_but_an_unparsed_one_is(tmp_path: Path) -> None:
    """Resuming should pick up a timeout. It should not re-ask a question the model answered
    unreadably at temperature zero: the unparsed share is a number this project reports."""
    path = tmp_path / "run.jsonl"
    records.append(
        path,
        [
            _record("ok"),
            _record("failed", error="ReadTimeout", correct=None, reply=None),
            _record("unreadable", correct=None, parsed=None, unparsed=True, reply="hmm"),
        ],
    )
    assert records.done(path) == frozenset({"ok", "unreadable"})
    assert records.done(path, include_errors=True) == frozenset({"ok", "failed", "unreadable"})


def test_a_half_written_last_line_is_skipped_not_raised_on(tmp_path: Path) -> None:
    """The normal shape of a file whose process was killed mid-write. The cell it belongs to
    is simply not done and gets asked again."""
    path = tmp_path / "run.jsonl"
    records.append(path, [_record("a")])
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"cell": "b", "correct"')
    assert records.done(path) == frozenset({"a"})
    assert len(list(records.read(path))) == 1
