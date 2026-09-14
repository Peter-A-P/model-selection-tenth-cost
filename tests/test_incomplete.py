"""Finding items that ask about statements they do not contain.

This flags items as broken and the flag goes in a published document, so a false positive is
worse than a miss: it accuses a benchmark of carrying a defect it does not have. Every case
below that must not be flagged is one an earlier version of the rule did flag, against the real
bank, and each is a different way of being wrong about the same thing.
"""

from __future__ import annotations

from mselect.experiments import incomplete
from mselect.runner.administer import Item


def _item(question: str, options: tuple[str, ...], answer: str = "1,2,3") -> Item:
    return Item(
        item_id="0" * 16,
        benchmark="mmlu",
        kind="multiple_choice",
        question=question,
        options=options,
        answer=answer,
    )


def test_an_item_whose_statements_are_missing_is_flagged() -> None:
    """The shape of the thing: a stem that is only a title, and options that index a list."""
    found = incomplete.find([_item("Demand reduction.", ("1,3,4", "2,3,4", "1,2,3", "1,2,4"))])
    assert len(found) == 1
    assert found[0].quote() == "Demand reduction."


def test_roman_numbered_statements_count_too() -> None:
    found = incomplete.find(
        [_item("Which of these scans can image brain function?", ("I only", "II only", "I and II"))]
    )
    assert len(found) == 1


def test_an_item_that_carries_its_statements_is_left_alone() -> None:
    """The same options, with the list present. Nothing is wrong with this item."""
    question = (
        "Triacylglycerides consist of I. a ribose backbone II. a glycerol backbone "
        "III. three phosphodiester linkages IV. three ester linkages"
    )
    assert incomplete.find([_item(question, ("I and III", "II only", "II and IV"))]) == ()


def test_a_numeric_answer_is_not_a_statement_reference() -> None:
    """Flagged by the first version of this rule, which read a bare "4" as naming statement 4."""
    assert (
        incomplete.find([_item("How many sides does a rhombus have?", ("4", "6", "8", "10"))]) == ()
    )


def test_prices_are_not_statement_lists() -> None:
    """Flagged by the second version, which looked for digits with a comma between them.

    A comma between digits is a thousands separator far more often than it is a list, and no
    amount of staring at the digits distinguishes the two. What does is that a price carries a
    currency symbol, which a reference to statements never would.
    """
    assert incomplete.find([_item("What is the cost?", ("$1,000", "$1,500", "$2,005,000"))]) == ()
    assert incomplete.find([_item("What is 32 x 67?", ("1,824", "1,934", "2,044", "2,144"))]) == ()


def test_a_chemical_name_is_not_a_statement_list() -> None:
    options = ("1,2-dichlorobenzene", "1,3-dichlorobenzene", "1,4-dichlorobenzene")
    assert incomplete.find([_item("para-dichlorobenzene is also called", options)]) == ()


def test_pairs_of_quantities_are_not_statement_lists() -> None:
    """Triangle side lengths: separated in every option, and still not a reference to anything.

    Two things give them away. The numbers are not 1..k, so they do not label a list, and "9, 8"
    descends where a reference to statements 8 and 9 would not.
    """
    options = ("4, 7", "5, 5", "3, 9", "9, 8")
    question = "The longest side of a triangle is 10. Which could NOT be the other two sides?"
    assert incomplete.find([_item(question, options)]) == ()


def test_an_option_that_is_not_a_reference_at_all_disqualifies_the_item() -> None:
    """One prose option among three lists means the options are answers, not indices."""
    options = ("1,2,3", "1,3,4", "the study was never replicated")
    assert incomplete.find([_item("Which statements hold?", options)]) == ()


def test_the_rule_needs_more_than_one_option_that_joins_or_qualifies() -> None:
    """Two bare single numbers are two answers, however contiguous they look."""
    assert incomplete.find([_item("How many?", ("1", "2"))]) == ()
