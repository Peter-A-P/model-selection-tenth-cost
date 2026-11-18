"""Decision models asked the bank's questions: the request they get and how a reply is scored.

The failures worth catching are quiet ones. A tie broken toward A would plant a position bias
the model never had. An unreadable reply scored as wrong would turn a host's malformed answer
into an item statistic. And a text model's request hash that moved when this was added would
make every record on disk look unasked, and the next run would pay for all of them again.
"""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from typing import Any

import pytest

from mselect.runner import decision
from mselect.runner.administer import Item, Reply, administer, build_prompts
from mselect.runner.gateway import BoundaryCaller

MC = Item(
    item_id="mmlu:1",
    benchmark="mmlu",
    kind="multiple_choice",
    question="What is the capital of France?",
    options=("Lyon", "Paris", "Nice", "Lille"),
    answer="Paris",
)
MATH = Item(item_id="math:1", benchmark="math", kind="free_response", question="1+1?", answer="2")


def _answers(probs: dict[str, Any] | None = None, **extra: Any) -> str:
    answer: dict[str, Any] = {"type": "choice", **extra}
    if probs is not None:
        answer["probabilities"] = probs
    return json.dumps({decision.QUESTION_ID: answer})


def test_the_question_is_the_state_and_the_options_are_one_choice() -> None:
    (prompt,) = build_prompts([MC], "decision-jev-1.14", template=decision.DECISION)
    assert prompt.user == MC.question
    assert prompt.system == "" and prompt.temperature is None
    questions = json.loads(prompt.questions)
    assert questions == {
        "answer": {
            "type": "choice",
            "instructions": decision.INSTRUCTIONS,
            "criteria": {"A": "Lyon", "B": "Paris", "C": "Nice", "D": "Lille"},
        }
    }


def test_a_free_response_item_is_refused_rather_than_asked() -> None:
    with pytest.raises(ValueError, match="no options"):
        build_prompts([MATH], "decision-jev-1.14", template=decision.DECISION)


def test_a_text_models_request_hash_is_what_it_was_before_decision_models() -> None:
    """Every record on disk is keyed on this hash. Recomputed here by the formula as it stood
    before `questions` existed, so a change that moved it fails rather than re-asking 60,000
    cached calls."""
    (prompt,) = build_prompts([MC], "anthropic-haiku", route="anthropic/claude-haiku-4-5")
    body = json.dumps(
        [
            prompt.alias,
            prompt.route,
            prompt.system,
            prompt.user,
            prompt.max_tokens,
            None if prompt.temperature is None else round(prompt.temperature, 6),
        ],
        separators=(",", ":"),
        ensure_ascii=False,
    )
    assert prompt.request_sha256 == hashlib.sha256(body.encode("utf-8")).hexdigest()


def test_the_highest_probability_is_the_answer() -> None:
    reply = _answers({"A": 0.1, "B": 0.7, "C": 0.15, "D": 0.05})
    assert decision.chosen(reply, 4) == "B"


def test_a_tie_at_the_top_is_unparsed_not_broken_toward_a() -> None:
    assert decision.chosen(_answers({"A": 0.4, "B": 0.4, "C": 0.2}), 4) is None


def test_an_option_that_was_not_offered_is_ignored() -> None:
    assert decision.chosen(_answers({"E": 0.9, "C": 0.1}), 4) == "C"


@pytest.mark.parametrize(
    "reply",
    [None, "", "not json", json.dumps({"other": {}}), _answers({}), _answers({"A": "high"})],
)
def test_an_unreadable_reply_is_none(reply: str | None) -> None:
    assert decision.chosen(reply, 4) is None


def test_a_bare_choice_still_scores() -> None:
    assert decision.chosen(_answers(None, choice="D"), 4) == "D"


class Recorder:
    def __init__(self, text: str) -> None:
        self.text = text
        self.requests: list[Any] = []

    def chat(self, request: Any, **kw: object) -> Any:
        self.requests.append(request)
        return SimpleNamespace(
            text=self.text,
            cost_usd=0.0,
            usage=SimpleNamespace(input_tokens=40, output_tokens=0),
            model_returned="typesafe/jev-1.14",
            ledger_id="1",
            ok=True,
            finish_reason=None,
            cached=False,
            status="ok",
            raw={},
        )


def test_a_decision_is_sent_without_system_or_temperature_and_scored() -> None:
    recorder = Recorder(_answers({"A": 0.05, "B": 0.9, "C": 0.03, "D": 0.02}))
    caller = BoundaryCaller(recorder, use_batches=False)  # type: ignore[arg-type]
    (record,) = administer([MC], "decision-jev-1.14", caller, template=decision.DECISION)

    (request,) = recorder.requests
    assert request.system is None and request.temperature is None
    assert request.messages == [{"role": "user", "content": MC.question}]
    assert request.extra["questions"]["answer"]["criteria"]["B"] == "Paris"
    assert record.parsed == "B" and record.correct == 1 and not record.unparsed


def test_an_unreadable_decision_is_unparsed_never_wrong() -> None:
    caller = BoundaryCaller(Recorder("{}"), use_batches=False)  # type: ignore[arg-type]
    (record,) = administer([MC], "decision-jev-1.14", caller, template=decision.DECISION)
    assert record.correct is None and record.unparsed


def test_reply_type_is_unchanged() -> None:
    """The record keeps the whole answers object as the reply, so calibration can be read back."""
    text = _answers({"A": 0.2, "B": 0.8})
    assert Reply(text=text).text == text
