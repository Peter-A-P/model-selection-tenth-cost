"""Items that cannot be answered because the question is not all there.

`docs/items-that-measure-nothing.md` has documented two kinds of broken item, and both are found
by statistics. An item with near-zero discrimination is found by fitting the bank. An item whose
answer key is wrong is found by stronger models disagreeing with it consistently. Both need 150
models and a calibrated bank before they can be seen at all.

This is a third kind and it needs neither. Some questions are simply incomplete: they ask which
of several numbered statements are correct, and the statements are not in the item. The options
are `1,2,3` and `1,3,4` and `2,3,4`, and there is nothing anywhere in the question numbered 1.
Nobody can answer that, and a model that says so is describing the item rather than failing it.

**Found by reading what a model said when it refused**, which is the part of this project's own
output that was being thrown away as "unparsed". `anthropic-haiku` produced eleven unparsed
replies on 2026-09-12 and several were well-formed complaints:

    "I need to see the numbered statements to evaluate which ones are correct. However, the
     statements (1, 2, 3, 4) are not provided in your question."

    "The question appears incomplete as only the title "Demand reduction" and multiple choice
     options are provided."

The model is right both times. But a complaint is not evidence on its own, because a model that
cannot answer has every reason to blame the question, so the rule here is structural and the
complaints are only what pointed at it. An item is flagged when its options are lists of
statement numbers and the question carries no numbered statements at all. That can be checked
without asking any model anything, which is what makes it a property of the bank rather than an
opinion about it.

A note on what this does not flag. MATH carries 44 items whose diagram arrives as Asymptote
source rather than as a picture, and the obvious guess is that those are unanswerable too. The
panel says otherwise: 0.733 (0.631 to 0.815) against 0.808 (0.776 to 0.837) for plain MATH
items, intervals overlapping on 86 observations. Models read the source. The guess was wrong and
it is recorded in `docs/rejected.md` rather than quietly dropped.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from mselect.runner.administer import Item

ROMAN: Final = {
    "i": 1,
    "ii": 2,
    "iii": 3,
    "iv": 4,
    "v": 5,
    "vi": 6,
    "vii": 7,
    "viii": 8,
    "ix": 9,
}

# What an option is allowed to be made of, if it refers to statements: single digits, roman
# numerals, the separators that join them, and "only". Anything else at all, a currency symbol,
# a letter, a bracket, a decimal point, means the option is an answer rather than a reference.
SEPARATORS: Final = re.compile(r"\s*(?:,|&|\band\b)\s*|\s+")
ONLY: Final = re.compile(r"\s*\bonly\b\s*$", re.IGNORECASE)
EDGES: Final = "()[]. "

# A numbered statement inside the question: "1. something", "(ii) something", "III. something".
ARABIC_STATEMENT: Final = re.compile(r"(?:^|\s)\(?[1-9]\)?[.)]\s+\S", re.MULTILINE)
ROMAN_STATEMENT: Final = re.compile(
    r"(?:^|\s)\(?(?:i{1,3}v?|iv|vi{0,3})\)?[.)]\s+\S", re.MULTILINE | re.IGNORECASE
)

# Two, because one numbered thing is a list of one and this is about a list the item refers to.
ENOUGH: Final = 2
# Statement lists are short. Beyond this the numbers are far more likely to be quantities.
MOST: Final = 9


def statement_numbers(option: str) -> list[int] | None:
    """The statements an option names, or None when the option is not a reference at all.

    The whole option has to be statement numbers, separators and "only". This is what keeps
    "$1,000" and "1,824" and "1,2-dichlorobenzene" out: each has a comma between digits, and
    each also has something that a reference to statements would never contain.
    """
    text = ONLY.sub("", option.strip().strip(EDGES))
    if not text:
        return None
    numbers: list[int] = []
    for piece in SEPARATORS.split(text):
        token = piece.strip().strip(EDGES).lower()
        if not token:
            continue
        if token in ROMAN:
            numbers.append(ROMAN[token])
        elif len(token) == 1 and token.isdigit() and token != "0":
            numbers.append(int(token))
        else:
            return None  # a word, a decimal, a multi-digit quantity: not a statement number
    return numbers or None


@dataclass(frozen=True, slots=True)
class Incomplete:
    """One item that asks about statements it does not contain."""

    item_id: str
    benchmark: str
    question: str
    options: tuple[str, ...]
    answer: str

    def quote(self, width: int = 90) -> str:
        text = " ".join(self.question.split())
        return text if len(text) <= width else text[: width - 3] + "..."


def refers_to_numbered_statements(options: Sequence[str]) -> bool:
    """Whether the options name statements from a numbered list rather than answer the question.

    Three properties, and each one was learned by watching an earlier version get it wrong.

    **Every option is nothing but a reference.** Not "contains digits": a price and a chemical
    name both contain a comma between digits.

    **At least two of them join or qualify.** "1,2,3" and "I only" refer to statements; a lone
    "4" is an answer, and reading it as a reference flagged "How many sides does a rhombus have?"

    **The numbers label a list.** They run 1..k with nothing skipped and ascend within an option,
    because they are labels on something the question was meant to carry. This is what rules out
    options like "4, 7" / "5, 5" / "3, 9" / "9, 8", which are pairs of triangle side lengths.
    """
    cleaned = [option.strip() for option in options if option.strip()]
    if len(cleaned) < ENOUGH:
        return False

    seen: set[int] = set()
    referring = 0
    for option in cleaned:
        numbers = statement_numbers(option)
        if numbers is None:
            return False
        if numbers != sorted(numbers) or len(numbers) != len(set(numbers)):
            return False
        if len(numbers) > 1 or ONLY.search(option):
            referring += 1
        seen.update(numbers)

    if referring < ENOUGH or not seen:
        return False
    return max(seen) <= MOST and seen == set(range(1, max(seen) + 1))


def carries_the_statements(question: str) -> bool:
    """Whether the question actually contains a numbered list, in either numbering style.

    Deliberately generous. The statements are often run together inline rather than set out on
    their own lines, and counting a run-on list as present is the safe direction to be wrong in:
    this flags items as broken, so it should under-flag rather than over-flag.
    """
    return (
        len(ARABIC_STATEMENT.findall(question)) >= ENOUGH
        or len(ROMAN_STATEMENT.findall(question)) >= ENOUGH
    )


def find(items: Sequence[Item]) -> tuple[Incomplete, ...]:
    """Every item that asks about numbered statements it does not carry."""
    out = [
        Incomplete(
            item_id=item.item_id,
            benchmark=item.benchmark,
            question=item.question,
            options=tuple(item.options),
            answer=item.answer,
        )
        for item in items
        if refers_to_numbered_statements(item.options) and not carries_the_statements(item.question)
    ]
    return tuple(sorted(out, key=lambda i: (i.benchmark, i.item_id)))
