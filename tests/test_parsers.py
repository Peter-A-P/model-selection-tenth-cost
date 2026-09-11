"""Adversarial fixtures for the answer parsers.

PLAN.md section 5: "answer parsers survive markdown, whitespace, and 'The answer is (B)'".
Every fixture here is a shape a real model reply takes, including the ones that should be
refused rather than guessed at.
"""

from __future__ import annotations

import pytest

from mselect.runner import parse

CHOICES: list[tuple[str, str | None]] = [
    ("B", "B"),
    ("(B)", "B"),
    (" b ", "B"),
    ("**B**", "B"),
    ("B.", "B"),
    ("B)", "B"),
    ("The answer is (B).", "B"),
    ("Answer: B", "B"),
    ("answer = b", "B"),
    ("Final answer: **C**", "C"),
    ("```\nC\n```", "C"),
    ("\n\n  D  \n", "D"),
    ("Option C", "C"),
    ("The correct option is D.", "D"),
    ("I said A earlier, but the answer is C.", "C"),
    ("Ｂ", "B"),  # full width, as some models emit
    ("​B", "B"),  # zero width space
    ("Hmm, A or B, hard to say.", None),  # genuinely ambiguous: refuse
    ("E", None),  # out of range for a four-option item
    ("I cannot answer that.", None),
    ("", None),
    ("42", None),
]


@pytest.mark.parametrize(("reply", "expected"), CHOICES)
def test_choice_parser_on_adversarial_replies(reply: str, expected: str | None) -> None:
    assert parse.parse_choice(reply, n_options=4) == expected


def test_choice_parser_respects_the_option_count() -> None:
    assert parse.parse_choice("The answer is F.", n_options=10) == "F"
    assert parse.parse_choice("The answer is F.", n_options=4) is None
    with pytest.raises(ValueError):
        parse.parse_choice("A", n_options=0)


def test_a_word_starting_with_a_letter_is_not_an_answer() -> None:
    """ "A model that answers in a sentence should not be read as choosing option A."""
    assert parse.parse_choice("Because the compound is acidic, the answer is C.", 4) == "C"
    assert parse.parse_choice("Cannot determine from the information given.", 4) is None


MATHS: list[tuple[str, str | None]] = [
    (r"\boxed{42}", "42"),
    (r"The answer is $\boxed{42}$.", "42"),
    (r"$\boxed{\frac{1}{2}}$", "1/2"),
    (r"\boxed{\dfrac{3}{4}}", "3/4"),
    (r"\boxed{0.5}", "1/2"),
    (r"\boxed{1,234}", "1234"),
    (r"\boxed{-3}", "-3"),
    (r"\boxed{x = 5}", "5"),
    (r"\boxed{5\%}", "5"),
    (r"\boxed{2\sqrt{3}}", r"2\sqrt{3}"),
    (r"\boxed{\text{none}}", "none"),
    ("The final answer is 17.", "17"),
    ("So we get 3 apples and 4 pears, giving 7.", "7"),
    ("", None),
    ("I am not able to work this out.", None),
    (r"first \boxed{3} then corrected: \boxed{4}", "4"),  # the last box wins
    (r"\boxed{\frac{10}{4}}", "5/2"),  # reduced
]


@pytest.mark.parametrize(("reply", "expected"), MATHS)
def test_math_parser_on_adversarial_replies(reply: str, expected: str | None) -> None:
    assert parse.parse_math(reply) == expected


def test_math_equivalence_accepts_the_same_value_written_differently() -> None:
    assert parse.math_equivalent("0.5", r"\frac{1}{2}")
    assert parse.math_equivalent("1,000", "1000")
    assert parse.math_equivalent("42.0", "42")
    assert parse.math_equivalent("10/4", "2.5")
    assert not parse.math_equivalent("0.5", "0.6")
    # No algebra is attempted, and the parser does not pretend otherwise.
    assert not parse.math_equivalent(r"2\sqrt{3}", "3.4641")


def test_grading_returns_none_rather_than_scoring_an_unparsed_reply_wrong() -> None:
    assert parse.grade_choice("The answer is B.", "B") == 1
    assert parse.grade_choice("The answer is C.", "B") == 0
    assert parse.grade_choice("I refuse.", "B") is None
    assert parse.grade_math(r"\boxed{7}", "7") == 1
    assert parse.grade_math(r"\boxed{8}", "7") == 0
    assert parse.grade_math("no idea", "7") is None
