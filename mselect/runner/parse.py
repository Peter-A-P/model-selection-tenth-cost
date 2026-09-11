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
"""

from __future__ import annotations

import re
import unicodedata
from fractions import Fraction

LETTERS = "ABCDEFGHIJ"

_EXPLICIT = re.compile(
    r"(?:final\s+answer|answer|option|choice|response)\s*(?:is|:|=)?\s*"
    r"[\*\s\(\[\{\"']*([A-Ja-j])(?![A-Za-z0-9])",
)
_BARE_LETTER = re.compile(
    r"(?<![A-Za-z0-9])[\*\(\[\{\"']*([A-Ja-j])[\*\)\]\}\"'\.,:]*(?![A-Za-z0-9])"
)
_BOXED = re.compile(r"\\boxed\s*\{")
_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


def clean(text: str) -> str:
    """Normalise unicode, drop code fences and bold markers, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("```", " ").replace("`", " ")
    text = text.replace("**", " ").replace("__", " ")
    return " ".join(text.split())


def parse_choice(text: str, n_options: int = 4) -> str | None:
    """The chosen option letter, or None when the reply does not identify one.

    `n_options` matters: a reply of "E" to a four-option item is not an answer, it is a model
    that has lost the plot, and scoring it as incorrect would blame the item.
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

    bare = [match.group(1).upper() for match in _BARE_LETTER.finditer(body)]
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


def grade_choice(text: str, key: str, n_options: int = 4) -> int | None:
    """1, 0, or None when the reply cannot be parsed at all."""
    choice = parse_choice(text, n_options)
    if choice is None:
        return None
    return int(choice == key.strip().upper())


def grade_math(text: str, key: str) -> int | None:
    answer = parse_math(text)
    if answer is None:
        return None
    return int(math_equivalent(answer, key))
