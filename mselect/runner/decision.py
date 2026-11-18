"""Asking a decision model the bank's questions, and scoring what comes back.

A decision model takes a state and typed questions with named options and returns one
probability per option, and no text. A multiple-choice benchmark item is that shape already:
the question is the state, and its lettered options are the options of one `choice` question.
So the bank is asked exactly as it is, with the same calibrated items and the same answer key,
and a decision model lands on the same ability scale as the text models. That is the point:
not a separate benchmark for a new kind of model, the same ruler.

What is the same as a text model's administration:

  * **The items and the key.** The question text the text models see, the options in the
    order they see them at rotation 0, the letter that is correct.
  * **The score.** Right or wrong, from the option given the highest probability. The bank's
    item parameters are fitted on right and wrong, so that is what goes into the matrix.

What differs, and why:

  * **Multiple choice only.** A free-response item (MATH, GSM8K) has no options to choose
    between, so a decision model is not asked it. About 7% of the suite.
  * **No system prompt, no temperature, no answer format.** The interface has no place for
    any of them and the gateway refuses a request that carries one. The instruction to answer
    lives in the question's `instructions`, worded once, here.
  * **A probability for every option.** Kept in the record as the reply, so calibration can be
    measured later from the same file: whether a 0.8 is right eight times in ten. A text model
    cannot be asked that.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Final

from mselect.runner.prompts import LETTERS

DECISION: Final = "decision"
QUESTION_ID: Final = "answer"

# One instruction for every item, so the comparison between decision models is the models and
# not the wording, and as close as the interface allows to what `plain` asks a text model.
INSTRUCTIONS: Final = "The state is a benchmark question. Which option is the correct answer to it?"


def questions_for(options: Sequence[str]) -> dict[str, Any]:
    """The decision request's `questions` for one item: one choice, keyed by letter.

    Letters rather than the option text as keys, because two options can share a text after
    normalisation and a key must be unique, and because the answer key is a letter.
    """
    if len(options) > len(LETTERS):
        raise ValueError(f"{len(options)} options is more than the {len(LETTERS)} letters")
    return {
        QUESTION_ID: {
            "type": "choice",
            "instructions": INSTRUCTIONS,
            "criteria": {LETTERS[i]: text for i, text in enumerate(options)},
        }
    }


def canonical(questions: dict[str, Any]) -> str:
    """The questions as the string the request hash and the prompt carry."""
    return json.dumps(questions, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def probabilities(reply: str | None) -> dict[str, float] | None:
    """The probability the model gave each option, or None when the reply is not readable.

    The gateway returns the `answers` object as JSON text. A reply that is not JSON, has no
    answer to this question, or has no probabilities is unreadable: recorded as unparsed, never
    as wrong, by the same rule as a text reply the parser cannot read.
    """
    if not reply:
        return None
    try:
        answer = json.loads(reply)[QUESTION_ID]
    except (ValueError, KeyError, TypeError):
        return None
    if not isinstance(answer, dict):
        return None
    raw = answer.get("probabilities")
    if isinstance(raw, dict) and raw:
        try:
            return {str(k): float(v) for k, v in raw.items()}
        except (TypeError, ValueError):
            return None
    # Some hosts mark `probabilities` optional. A bare choice is still an answer, at an
    # unknown probability, so it scores but carries nothing for calibration.
    choice = answer.get("choice")
    return {str(choice): 1.0} if isinstance(choice, str) and choice else None


def chosen(reply: str | None, n_options: int) -> str | None:
    """The letter given the highest probability among the options offered, or None.

    A tie at the top is unparsed rather than broken by letter order: breaking it toward A
    would hand the position-bias finding a bias the model never had.
    """
    probs = probabilities(reply)
    if probs is None:
        return None
    offered = set(LETTERS[:n_options])
    ranked = sorted(((p, k) for k, p in probs.items() if k in offered), reverse=True)
    if not ranked:
        return None
    if len(ranked) > 1 and ranked[0][0] == ranked[1][0]:
        return None
    return ranked[0][1]
