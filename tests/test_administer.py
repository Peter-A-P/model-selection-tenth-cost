"""Administering items and scoring replies, with no vendor and no money.

The scoring rules are the part that has to be right: a parser mistake recorded as a wrong
answer becomes an item statistic, and the broken-item report would then be describing this
code rather than the benchmark. Every path through it is exercised against a fake caller.
"""

from __future__ import annotations

import json
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


def test_only_a_retryable_failure_comes_back_around(tmp_path: Path) -> None:
    """Resuming should pick up a timeout and settle everything else.

    Amended 2026-09-12 against the live run, which produced 33 failures in its first four
    models and not one of them was transient: 25 models reasoning past the token budget, 4
    refusals, 1 stopping early. Re-asking those buys the identical failure at the identical
    price, and 25 truncations at 1024 output tokens each is real money per resume.
    """
    path = tmp_path / "run.jsonl"
    records.append(
        path,
        [
            _record("ok"),
            _record("timeout", error="ReadTimeout", retryable=True, correct=None, reply=None),
            _record("refused", error="the model returned no text (refusal)", correct=None),
            _record("truncated", error="no text (max_tokens), 1024 spent", correct=None),
            _record("unreadable", correct=None, parsed=None, unparsed=True, reply="hmm"),
        ],
    )
    assert records.done(path) == frozenset(
        {"hash-of-ok", "hash-of-refused", "hash-of-truncated", "hash-of-unreadable"}
    ), "only the timeout is asked again"
    assert records.done(path, include_errors=True) == frozenset(
        {
            "hash-of-ok",
            "hash-of-timeout",
            "hash-of-refused",
            "hash-of-truncated",
            "hash-of-unreadable",
        }
    )


def test_a_failure_with_no_verdict_is_read_from_what_it_says(tmp_path: Path) -> None:
    """A record written before `retryable` existed must not be mistaken for a settled one.

    Found 2026-09-13 in the live panel run. The run began at 18:39 on 2026-09-12 and
    `retryable` was added to the code at 21:29, so the running process kept writing records
    without the field for another nine hours. At 00:23 Google's daily free-tier quota for
    `gemini-3.8-flash` ran out and refused the remaining 1,910 calls with a 429, and every one
    of those records reached `done` with no verdict in it. Defaulting that to "not retryable"
    settled all 1,910: resuming would have skipped them in silence and left that model scored
    on 1,088 items of 3,000.

    So the three cases are distinct, and stay distinct: a verdict of True is asked again, a
    verdict of False is settled, and no verdict at all is decided from the recorded error.
    """
    path = tmp_path / "run.jsonl"
    quota = (
        "ProviderError: google returned 429 after 0 retries: RESOURCE_EXHAUSTED: You exceeded "
        "your current quota. Quota exceeded for metric: generate_requests_per_model_per_day, "
        "limit: 10000, model: gemini-3.8-flash. Please retry in 21h6m2.949697714s."
    )
    legacy = [
        (quota, "quota"),
        ("ProviderError: google returned 503 after 0 retries: UNAVAILABLE", "unavailable"),
        ("the model returned no text (max_tokens), 1024 output tokens spent", "truncated"),
        ("the model returned no text (refusal)", "refused"),
        ("ProviderError: openai returned 400 after 0 retries: bad request", "rejected"),
    ]
    with path.open("w", encoding="utf-8") as handle:
        for error, cell in legacy:
            handle.write(
                json.dumps(
                    {
                        "cell": cell,
                        "request_sha256": f"hash-of-{cell}",
                        "error": error,
                        "correct": None,
                    }
                )
                + "\n"
            )

    settled = records.done(path)
    assert "hash-of-quota" not in settled, "a daily quota is the most repeatable failure there is"
    assert "hash-of-unavailable" not in settled
    assert "hash-of-truncated" in settled, "a model that overran its budget will overrun it again"
    assert "hash-of-refused" in settled
    assert "hash-of-rejected" in settled, "a 400 describes the request, and will describe it again"


def test_a_recorded_verdict_beats_what_the_error_string_looks_like(tmp_path: Path) -> None:
    """The error text is consulted only when there is no verdict, never instead of one.

    The adapter holds the status and the exception type; a sentence is a lossy copy of both.
    A record that carries `retryable` carries the adapter's answer, and that is the one that
    counts even where the words point the other way.
    """
    path = tmp_path / "run.jsonl"
    records.append(
        path,
        [
            _record("says-429", error="returned 429 but settled", retryable=False, correct=None),
            _record("says-400", error="returned 400 but transient", retryable=True, correct=None),
        ],
    )
    assert records.done(path) == frozenset({"hash-of-says-429"})


def test_raising_the_budget_re_asks_a_truncated_item_without_being_asked_to(
    tmp_path: Path,
) -> None:
    """Settling a deterministic failure costs nothing, because the request hash is the key.

    A model that reasoned past 1024 tokens will do it again at 1024. At 4096 it is a different
    request, so it comes back on its own and no special case is needed anywhere.
    """
    small = build_prompts([MC], "m")[0]
    big = build_prompts([MC], "m", settings=Settings(max_tokens=4096))[0]
    assert small.request_sha256 != big.request_sha256


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


def test_a_cached_reply_is_recorded_as_cached() -> None:
    """Section 3.3 says a rerun costs nothing, and this is what makes that checkable.

    A paid alias reporting no spend is either a cache hit or a costing failure, and those look
    identical in a total. On 2026-09-12 four of eleven aliases reported US$0.00000 and it took
    opening the ledger to find out which it was.
    """
    written = administer([MC], "m", _always("B", cached=True, cost_usd=0.0))
    assert written[0].cached is True
    assert written[0].correct == 1, "a cached reply is still a reply"
    assert administer([MC], "m", _always("B"))[0].cached is False


def test_a_reply_the_gateway_could_not_price_keeps_a_null_cost() -> None:
    """Uncosted is not free, and a total that adds it as zero is wrong by an unknown amount.

    together-open-b wrote three uncosted rows on 2026-09-12: Together reported prompt-cache
    tokens and the price file had no rate for them, so the gateway refused to guess. That
    refusal is correct; what was missing was anything saying so above the ledger.
    """
    written = administer([MC], "m", _always("B", cost_usd=None, input_tokens=200))
    assert written[0].cost_usd is None, "never coerced to zero"
    assert written[0].error is None and written[0].correct == 1, "the call worked"


def test_a_scoring_fix_is_recoverable_from_the_stored_reply() -> None:
    """The claim that makes a parser defect cheap, tested rather than asserted.

    Three of this project's defects on 2026-09-12 were in scoring, found after the calls were
    paid for. Because the reply text is kept, the repair is rescoring rather than re-asking:
    the panel costs US$13.40 and six hours to ask, and a parser fix should cost neither.

    The reply below is the one that exposed the case-sensitivity defect, from a 3B model on a
    MedQA item. Scoring it now, from text alone, gets the answer the model actually gave.
    """
    stored = (
        "The infant's inability to pull himself to stand suggests a delay in social "
        "development, because these behaviors indicate attachment anxiety.\n\nAnswer: B"
    )
    item = Item(
        item_id="mmlu:1",
        benchmark="mmlu",
        kind="multiple_choice",
        question="Which milestone is delayed?",
        options=("Fine motor", "Social", "Gross motor", "Speech"),
        answer="Social",
    )
    parsed, correct = score(item, Reply(text=stored))
    assert parsed == "B" and correct == 1, "scored from the reply alone, with no vendor call"
