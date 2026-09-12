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


# -- prose replies, which the panel now produces on purpose ---------------------------------
#
# Every case below is a shape a reasoning model actually produces. The first two are the
# defects found on 2026-09-12 by a 3B model on a MedQA item; both were invisible while every
# reply was a single letter.


def test_a_capital_answer_line_is_read() -> None:
    """`prompts.TEMPLATES["plain"]` asks for exactly this, and the parser could not read it.

    The explicit pattern was case sensitive, so "Answer: B" fell through to the fallback and
    the fallback got it wrong. Every compliant reply from a model that capitalises was at risk.
    """
    reply = (
        "The infant's inability to pull himself to stand, grasp his rattle, and transfer it "
        "from one hand to the other suggest a delay in social development. This is because "
        "these behaviors are indicative of attachment anxiety.\n\nAnswer: B"
    )
    assert parse.parse_choice(reply, 4) == "B"
    assert parse.grade_choice(reply, "B", 4) == 1
    assert parse.grade_choice(reply, "A", 4) == 0


def test_prose_that_never_answers_is_unparsed_rather_than_an_invented_letter() -> None:
    """The worst failure available to this module: producing a score instead of an absence.

    A standalone letter A to J counted as a candidate, and "a" is the indefinite article. A
    reply that reasoned and never answered scored as answering A, with a letter the model
    never chose, and `docs/items-that-measure-nothing.md` would have been describing this
    function rather than the benchmark.
    """
    assert parse.parse_choice("This is a question about a delay in development.", 4) is None
    assert parse.grade_choice("This is a question about a delay.", "A", 4) is None
    assert parse.parse_choice("I think this needs a moment of thought.", 4) is None


def test_a_lone_letter_is_still_an_answer_in_any_case() -> None:
    """The exclusion is about prose. A reply that is nothing but a letter is an answer."""
    assert parse.parse_choice("a", 4) == "A"
    assert parse.parse_choice("A.", 4) == "A"
    assert parse.parse_choice(" (a) ", 4) == "A"
    assert parse.parse_choice("I", 10) == "I"


def test_an_explicit_statement_beats_the_word_exclusion() -> None:
    """Excluding the article must not cost a model that explicitly answers A or I."""
    assert parse.parse_choice("After a long look at a hard question, the answer is A.", 4) == "A"
    assert parse.parse_choice("Answer: A", 4) == "A"
    assert parse.parse_choice("Answer: I", 10) == "I"


def test_reasoning_that_names_options_then_answers() -> None:
    assert parse.parse_choice("Rule out A since it is too small. Answer: D", 4) == "D"
    assert parse.parse_choice("B looks right at first. The answer is C.", 4) == "C"


def test_reasoning_that_names_options_and_never_settles_is_refused() -> None:
    """Taking the last letter mentioned would be a guess, and would be wrong as often as not:
    "so it is not D" ends on D."""
    assert parse.parse_choice("B says one thing and C says another, so it is not D", 4) is None


def test_a_model_that_corrects_itself_is_taken_at_its_last_word() -> None:
    assert parse.parse_choice("Answer: B. Wait, rereading it, Answer: D", 4) == "D"


def test_free_response_reasoning_ends_in_the_box() -> None:
    reply = "First 2 plus 2 is 4, and 4 times 3 is 12. So the total is \boxed{12}."
    assert parse.parse_math(reply) == "12"
    assert parse.grade_math(reply, "12") == 1


# -- naming an option instead of lettering it -----------------------------------------------
#
# LegalBench offers Yes, No, Analysis and Rule rather than lettered alternatives, and it is
# 2,047 items of bank v1. `openai-mid` answered "Answer: No" correctly on 2026-09-12 and lost
# the item, which is format compliance scored as ability.

YES_NO = ["No", "Yes"]


def test_an_option_named_in_an_explicit_statement_is_the_answer() -> None:
    assert parse.parse_choice("Answer: No", 2, YES_NO) == "A"
    assert parse.parse_choice("Answer: Yes", 2, YES_NO) == "B"
    assert parse.parse_choice("The answer is yes.", 2, YES_NO) == "B"
    assert parse.grade_choice("Answer: No", "A", 2, YES_NO) == 1
    assert parse.grade_choice("Answer: No", "B", 2, YES_NO) == 0


def test_a_reply_that_is_only_the_option_text_is_the_answer() -> None:
    assert parse.parse_choice("No", 2, YES_NO) == "A"
    assert parse.parse_choice("  Yes.  ", 2, YES_NO) == "B"


def test_an_option_mentioned_inside_prose_is_not_a_choice() -> None:
    """The restraint that makes the rest safe.

    "Yes" and "No" are ordinary English words. `together-open-a` wrote "there is no clear
    connection" in a reply whose answer was A, and reading that "no" as the answer would be a
    coin toss dressed as a measurement.
    """
    prose = "Based on the information provided, there is no clear connection between them."
    assert parse.parse_choice(prose, 2, YES_NO) is None
    assert parse.grade_choice(prose, "A", 2, YES_NO) is None


def test_a_letter_still_wins_over_the_option_text() -> None:
    """A model that letters its answer has said which one, and that is the stronger signal."""
    assert parse.parse_choice("Yes, the answer is A.", 2, YES_NO) == "A"


def test_option_text_respects_the_rotation_the_model_was_shown() -> None:
    """The position-bias experiment permutes the display order, so "No" is not always A."""
    from mselect.runner.prompts import rotate

    shown = rotate(YES_NO, 1)
    assert shown == ["Yes", "No"], "rotation moves each option one place later"
    assert parse.parse_choice("Answer: No", 2, shown) == "B"
    assert parse.parse_choice("Answer: Yes", 2, shown) == "A"


def test_without_options_nothing_changes() -> None:
    """Callers that pass no options get exactly the old behaviour."""
    assert parse.parse_choice("Answer: No", 2) is None
    assert parse.parse_choice("Answer: B", 2) == "B"


def test_an_ambiguous_name_is_refused() -> None:
    """Two options that normalise the same cannot be told apart, so neither is chosen."""
    assert parse.parse_choice("Answer: yes", 2, ["Yes", "yes."]) is None


def test_a_bare_letter_is_a_label_before_it_is_content() -> None:
    """MMLU has items whose options are themselves letters, and they collide.

    A logic item offers "A", "~A", "B", "~B"; a physics item offers "2c", "c", "0.8c", "0.5c".
    A reply of "B" there is ambiguous between the label and the content, and the label is what
    the model was asked for. Reading option text first got all four of these wrong, which the
    self-administration check over the whole bank caught.
    """
    logic = ["A", "~A", "B", "~B"]
    assert parse.parse_choice("B", 4, logic) == "B", "the second option, not the one reading B"
    assert parse.parse_choice("Answer: B", 4, logic) == "B"

    physics = ["2c", "c", "0.8c", "0.5c"]
    assert parse.parse_choice("C", 4, physics) == "C", "not the option whose text is c"


def test_option_text_still_wins_when_it_is_not_a_letter() -> None:
    """The collision rule must not cost LegalBench the fix it exists for."""
    assert parse.parse_choice("No", 2, YES_NO) == "A"
    assert parse.parse_choice("Answer: No", 2, YES_NO) == "A"
