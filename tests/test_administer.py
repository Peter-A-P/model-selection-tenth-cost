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


def test_every_template_gets_the_same_token_budget() -> None:
    """Changed 2026-09-12, when reasoning models joined the panel.

    The budgets differed when the answer-only cap was 16, which was a cost decision that saved
    nothing (output is billed on what is generated, not on the cap) and cost four models their
    answers. Giving the reasoning template more room than the others would have measured the
    room alongside the framing, and the framing is the thing the experiment is about.
    """
    settings = Settings()
    plain = build_prompts([MC], "m", template="plain")[0]
    reasoning = build_prompts([MC], "m", template="brief_reasoning")[0]
    assert plain.max_tokens == settings.max_tokens == reasoning.max_tokens
    assert plain.max_tokens >= 512, "room for a model that reasons before it answers"


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
    out = administer([MC, MATH], "m", again, done=frozenset({first[0].request_sha256}))
    assert len(out) == 1, "only the request that was not already made"
    assert out[0].item_id == "math:7"
    assert [p.item_id for p in again.seen] == ["math:7"], "nothing was sent for the done cell"


def test_administer_sends_nothing_when_everything_is_done() -> None:
    caller = _always("B")
    every = frozenset(p.request_sha256 for p in build_prompts([MC, MATH], "m"))
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
        # Distinct per record, because resume keys on the request rather than the cell.
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
        "request_sha256": f"hash-of-{cell}",
    }
    base.update(kw)
    return Administration(**base)  # type: ignore[arg-type]


def test_records_round_trip_and_resume(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    assert records.done(path) == frozenset(), "a missing file is simply nothing done"
    assert records.append(path, [_record("a"), _record("b")]) == 2
    assert records.done(path) == frozenset({"hash-of-a", "hash-of-b"})
    records.append(path, [_record("c")])
    assert records.done(path) == frozenset({"hash-of-a", "hash-of-b", "hash-of-c"})
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
    assert records.done(path) == frozenset({"hash-of-ok", "hash-of-unreadable"})
    assert records.done(path, include_errors=True) == frozenset(
        {"hash-of-ok", "hash-of-failed", "hash-of-unreadable"}
    )


def test_a_half_written_last_line_is_skipped_not_raised_on(tmp_path: Path) -> None:
    """The normal shape of a file whose process was killed mid-write. The cell it belongs to
    is simply not done and gets asked again."""
    path = tmp_path / "run.jsonl"
    records.append(path, [_record("a")])
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"cell": "b", "request_sha256": "hash-of-b", "correct"')
    assert records.done(path) == frozenset({"hash-of-a"})
    assert len(list(records.read(path))) == 1


def test_a_reply_with_no_text_is_a_finding_not_a_silence() -> None:
    """A 200 with an empty completion used to count as neither unparsed nor failed.

    It is what a model does when it spends its whole token budget reasoning before answering,
    and it is the failure this check exists to catch. `google-frontier` reported "0 scored,
    0 correct, 0 unparsed, 0 failed" on 2026-09-12 and had cost real money.
    """
    empty = Reply(text=None, output_tokens=13, finish_reason="max_tokens", cost_usd=0.0002)
    written = administer([MC], "some-alias", FakeCaller(lambda _p: empty))
    assert len(written) == 1
    record = written[0]
    assert record.correct is None and record.unparsed is False
    assert record.error is not None, "an empty reply has to say something"
    assert "no text" in record.error
    assert "max_tokens" in record.error, "the vendor's reason is the useful part"
    assert "13 output tokens" in record.error
    assert record.finish_reason == "max_tokens"


def test_whitespace_only_is_no_text_too() -> None:
    written = administer([MC], "some-alias", _always("   " + chr(10)))
    assert written[0].error is not None and "no text" in written[0].error


def test_a_real_answer_is_not_mistaken_for_an_empty_one() -> None:
    written = administer([MC], "some-alias", _always("B", finish_reason="stop"))
    assert written[0].error is None
    assert written[0].correct == 1
    assert written[0].finish_reason == "stop"


def test_a_model_that_refuses_temperature_is_sent_none_not_zero() -> None:
    """Anthropic's 5 family returns 400 for `temperature` at all, including zero.

    Sending 0 and sending nothing are different requests, and only the second is accepted.
    """
    prompts = build_prompts([MC], "anthropic-opus", omit_temperature=True)
    assert prompts[0].temperature is None
    assert build_prompts([MC], "anthropic-haiku")[0].temperature == 0.0


def test_omitting_temperature_is_a_different_request_hash() -> None:
    """Otherwise a cached reply from before the fix would answer for one made after it."""
    with_temp = build_prompts([MC], "a")[0]
    without = build_prompts([MC], "a", omit_temperature=True)[0]
    assert with_temp.request_sha256 != without.request_sha256


def test_the_record_says_what_was_sent_not_what_was_configured() -> None:
    """The settings block is evidence. A record claiming temperature 0 that was never sent
    would make every one of those 3,000 rows say something untrue."""
    written = administer([MC], "anthropic-opus", _always("B"), omit_temperature=True)
    assert written[0].settings["temperature"] is None
    normal = administer([MC], "anthropic-haiku", _always("B"))
    assert normal[0].settings["temperature"] == 0.0


def test_repointing_an_alias_at_another_model_re_asks_its_items() -> None:
    """The bug this contract exists for.

    `openai-frontier` moved from gpt-5.4 to gpt-5.6-sol on 2026-09-12 and a resume reported
    "every cell already recorded; nothing called". The cell was the same; the measurement was
    not. A run that inherited those answers would have a column of one model's replies
    labelled with another model's name and no way to tell.
    """
    before = build_prompts([MC], "openai-frontier", route="openai/gpt-5.4")[0]
    after = build_prompts([MC], "openai-frontier", route="openai/gpt-5.6-sol")[0]
    assert before.cell == after.cell, "same model slot, same item: one cell"
    assert before.request_sha256 != after.request_sha256, "different model: different request"

    caller = _always("B")
    out = administer(
        [MC],
        "openai-frontier",
        caller,
        done=frozenset({before.request_sha256}),
        route="openai/gpt-5.6-sol",
    )
    assert len(out) == 1, "the new model is asked even though the cell was recorded"


def test_changing_a_vendor_field_re_asks_too() -> None:
    """Disabling a model's reasoning changes what it answers, so old answers do not count."""
    plain = build_prompts([MC], "google-mid", route="google/gemini-3.5-flash-lite")[0]
    fixed = build_prompts([MC], "google-mid", route="google/gemini-3.5-flash-lite+abc123")[0]
    assert plain.request_sha256 != fixed.request_sha256


def test_an_unchanged_route_still_resumes_for_free() -> None:
    """The whole point of resume: an interrupted run must not pay twice for the same work."""
    route = "anthropic/claude-haiku-4-5-20251001"
    first = administer([MC, MATH], "anthropic-haiku", _always("B"), route=route)
    again = _always("B")
    out = administer(
        [MC, MATH],
        "anthropic-haiku",
        again,
        done=frozenset(r.request_sha256 for r in first),
        route=route,
    )
    assert out == [] and again.seen == []
