"""Turning a model's reply into a 0 or a 1.

PLAN.md section 5 lists this among the tests that have to fail meaningfully: "answer parsers
survive markdown, whitespace, and 'The answer is (B)'". A parser that silently scores a correct
answer as wrong does not look like a bug in the results, it looks like an item that measures
nothing, and the whole broken-item report would then be full of the parser's own mistakes.

Two rules run through everything here:

* Never guess. A reply that does not contain an identifiable answer returns None, and the
  caller records it as unparsed rather than as incorrect. The share of unparsed replies is a
  number worth reporting on its own.
* Prefer the explicit. "The answer is B" beats a stray "A" earlier in the sentence, and the
  last explicit statement beats an earlier one, because models correct themselves.

Both rules were tested against answer-only replies and both broke the first time a model
replied in prose, which is now the normal case: the panel contains reasoning models on purpose
(PLAN.md section 15.14). A parser that reads the article "a" as option A does not fail loudly,
it produces a score, and that is worse than producing nothing.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from fractions import Fraction

LETTERS = "ABCDEFGHIJ"

# Case-insensitive since 2026-09-12. It was not, so "Answer: B" did not match while
# "answer: B" did, and "Answer: X" is precisely the form `prompts.TEMPLATES["plain"]` asks
# every model to end with. A compliant reply fell through to the fallback below, which is
# where the damage was.
_EXPLICIT = re.compile(
    r"(?:final\s+answer|answer|option|choice|response)\s*(?:is|:|=)?\s*"
    r"[\*\s\(\[\{\"']*([A-Ja-j])(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_BARE_LETTER = re.compile(
    r"(?<![A-Za-z0-9])[\*\(\[\{\"']*([A-Ja-j])[\*\)\]\}\"'\.,:]*(?![A-Za-z0-9])"
)
# A reply that is nothing but a letter, which is what the answer-only format asks for. Matched
# separately so that a bare "a" on its own is read as option A while the same letter inside a
# sentence is read as the article it almost always is.
_SOLE_LETTER = re.compile(r"^[\*\s\(\[\{\"']*([A-Ja-j])[\*\)\]\}\"'\.,:\s]*$")

# Letters that are also English words. Inside prose they are the article and the pronoun far
# more often than they are an option, and reading one as an answer invents a response the model
# never gave. Excluded from the prose fallback only: "Answer: A" still reads as A, and a reply
# of "a" on its own still reads as A.
_WORD_LETTERS = frozenset({"a", "i", "I"})

# The tail of the last explicit answer statement, whatever it says. Used to read a reply that
# names an option rather than lettering it: "Answer: No" on an item whose options are words.
_ANSWER_MARK = re.compile(
    r"(?:final\s+answer|answer|option|choice|response)\s*(?:is|:|=)\s*",
    re.IGNORECASE,
)

_BOXED = re.compile(r"\\boxed\s*\{")
_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


def clean(text: str) -> str:
    """Normalise unicode, drop code fences and bold markers, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("```", " ").replace("`", " ")
    text = text.replace("**", " ").replace("__", " ")
    return " ".join(text.split())


_EDGE_PUNCTUATION = ".,:;!?\"'()[]{} "


def _normalise_option(text: str) -> str:
    """Case and edge punctuation removed, from both sides of a comparison.

    Symmetric on purpose. Stripping only the reply would let two options that differ by a full
    stop look like different answers, and then a reply matching both would look like a reply
    matching one.
    """
    return text.strip().strip(_EDGE_PUNCTUATION).casefold()


def _same(first: str, second: str) -> bool:
    """Two short answers as the same answer, ignoring case and surrounding punctuation."""
    return _normalise_option(first) == _normalise_option(second)


def stated_answer(body: str) -> str | None:
    """Whatever the last explicit answer statement names, or None if there is none."""
    marks = list(_ANSWER_MARK.finditer(body))
    if not marks:
        return None
    tail = body[marks[-1].end() :].strip()
    return tail or None


def _by_option_text(body: str, options: Sequence[str] | None) -> str | None:
    """The letter of the option a reply names, when it names exactly one and means it.

    Only an explicit statement or a reply that is nothing but the option text. Never a mention
    inside prose: "Yes" and "No" are ordinary English words, and a reply reading "there is no
    clear connection" names no option at all.
    """
    if not options:
        return None
    candidates = [stated_answer(body), body]
    for candidate in candidates:
        if candidate is None:
            continue
        hits = [index for index, option in enumerate(options) if _same(candidate, option)]
        if len(hits) == 1:
            return LETTERS[hits[0]]
    return None


def parse_choice(text: str, n_options: int = 4, options: Sequence[str] | None = None) -> str | None:
    """The chosen option letter, or None when the reply does not identify one.

    `n_options` matters: a reply of "E" to a four-option item is not an answer, it is a model
    that has lost the plot, and scoring it as incorrect would blame the item.

    `options` are the option texts **in the order the model was shown them**, which is what
    `prompts.rotate` produces. Given them, a reply that names an option rather than lettering
    it is read as the answer it plainly is: LegalBench offers Yes, No, Analysis and Rule rather
    than lettered alternatives, and a model answering "No" there should not lose the item.
    """
    if n_options < 1 or n_options > len(LETTERS):
        raise ValueError(f"n_options must be between 1 and {len(LETTERS)}")
    allowed = set(LETTERS[:n_options])
    body = clean(text)
    if not body:
        return None

    explicit = [match.group(1).upper() for match in _EXPLICIT.finditer(body)]
    for letter in reversed(explicit):
        if letter in allowed:
            return letter

    # A reply that is nothing but a letter is read as that option's letter, and this has to
    # come before reading option text or it collides with items whose options are themselves
    # letters. MMLU carries logic items offering "A", "~A", "B", "~B", and physics items
    # offering "2c", "c", "0.8c": there, a reply of "B" or "c" is ambiguous between the label
    # and the content, and the label is what the model was asked for.
    sole = _SOLE_LETTER.match(body)
    if sole is not None:
        letter = sole.group(1).upper()
        return letter if letter in allowed else None

    named = _by_option_text(body, options)
    if named is not None and named in allowed:
        return named

    # Prose, with no explicit statement. Letters that are also English words are dropped
    # first: without that, a reply that reasons and never answers scores as answering A
    # because it contained the word "a", which invents a response rather than recording that
    # none was given.
    bare = [
        match.group(1).upper()
        for match in _BARE_LETTER.finditer(body)
        if match.group(1) not in _WORD_LETTERS
    ]
    bare = [letter for letter in bare if letter in allowed]
    if not bare:
        return None
    if len(set(bare)) > 1:
        # Two different letters and no explicit statement: ambiguous, and answer-only output
        # was asked for. Refusing to choose is the honest result.
        return None
    return bare[0]


def extract_boxed(text: str) -> str | None:
    """The contents of the last \\boxed{...}, brace-matched so nested braces survive."""
    body = unicodedata.normalize("NFKC", text)
    starts = [match.end() for match in _BOXED.finditer(body)]
    if not starts:
        return None
    start = starts[-1]
    depth = 1
    out: list[str] = []
    for char in body[start:]:
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(out).strip()
        out.append(char)
    return "".join(out).strip() or None


def normalise_math(answer: str) -> str:
    """A canonical form for a short mathematical answer.

    Deliberately conservative: it strips presentation (dollar signs, \\left, \\!, trailing
    punctuation, thousands separators, units of the "\\%" kind) and normalises simple fractions
    and decimals, but it does not attempt algebra. Two answers that differ by a factorisation
    are left as different, and that shows up as item misfit rather than as a silent mark.
    """
    text = unicodedata.normalize("NFKC", answer).strip()
    text = text.replace("$", "").replace("\\!", "").replace("\\,", "").replace("\\;", "")
    text = text.replace("\\left", "").replace("\\right", "").replace("\\%", "").replace("%", "")
    text = text.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    text = re.sub(r"\\text\s*\{([^}]*)\}", r"\1", text)
    text = re.sub(r"^[a-zA-Z]\s*=\s*", "", text.strip())
    text = text.strip().rstrip(".").strip()
    text = re.sub(r"\s+", "", text)
    fraction = re.fullmatch(r"\\frac\{(-?[\d.]+)\}\{(-?[\d.]+)\}", text)
    if fraction:
        text = f"{fraction.group(1)}/{fraction.group(2)}"
    if re.fullmatch(r"-?\d[\d,]*(\.\d+)?", text):
        text = text.replace(",", "")
    value = as_number(text)
    if value is not None:
        return _format_number(value)
    return text


def as_number(text: str) -> Fraction | None:
    """A Fraction when the answer is a plain number or a simple fraction, else None."""
    body = text.replace(",", "").strip()
    try:
        return Fraction(body)
    except (ValueError, ZeroDivisionError):
        pass
    try:
        return Fraction(float(body)).limit_denominator(10**6)
    except (ValueError, OverflowError):
        return None


def _format_number(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def parse_math(text: str) -> str | None:
    """The final answer of a free-response reply: the boxed value, or the last number."""
    boxed = extract_boxed(text)
    if boxed:
        return normalise_math(boxed)
    body = clean(text)
    if not body:
        return None
    explicit = re.findall(r"(?:final\s+answer|answer)\s*(?:is|:|=)?\s*(-?[\d,\.\/]+)", body, re.I)
    if explicit:
        return normalise_math(explicit[-1])
    numbers = _NUMBER.findall(body)
    if not numbers:
        return None
    return normalise_math(numbers[-1])


def math_equivalent(first: str, second: str) -> bool:
    """Are two short mathematical answers the same value?"""
    a, b = normalise_math(first), normalise_math(second)
    if a == b:
        return True
    value_a, value_b = as_number(a), as_number(b)
    if value_a is None or value_b is None:
        return False
    return value_a == value_b


def grade_choice(
    text: str, key: str, n_options: int = 4, options: Sequence[str] | None = None
) -> int | None:
    """1, 0, or None when the reply cannot be parsed at all."""
    choice = parse_choice(text, n_options, options)
    if choice is None:
        return None
    return int(choice == key.strip().upper())


def grade_math(text: str, key: str) -> int | None:
    answer = parse_math(text)
    if answer is None:
        return None
    return int(math_equivalent(answer, key))
