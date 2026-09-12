"""The prompt templates the own-run panel will use, and the panel itself.

No vendor call lives here, and none can: the runner that sends these goes through the portfolio
gateway (project 04), whose Message Batches support lands in its v0.2, and it needs this
project's spend caps set first. PLAN.md section 3.3 and section 13.4. What is here is everything
that can be settled without spending anything: the exact strings, the panel, and the settings,
so that when the runner is wired up there is nothing left to decide and the framing experiment
already has its three templates written down.

PLAN.md section 3.3: answer-only output, temperature 0, fixed system prompt. **Amended
2026-09-12**: `max_tokens` is no longer small, and temperature 0 holds for nine of eleven
models. Both because the panel contains reasoning models, deliberately: a frontier tier without
them would not be a frontier tier. See PLAN.md sections 15.11 and 15.13.

PLAN.md section 4.3: the framing experiment compares the answer-only template against a
letter-only instruction and a brief-reasoning template, on the same items. What that comparison
means has narrowed, because a model that reasons internally reasons under all three.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

# Rewritten 2026-09-12 for a panel that contains reasoning models. The old wording forbade
# explanation outright, which a model that reasons internally cannot obey and which made
# compliance a proxy for ability. What is required now is the thing the score depends on: the
# answer, in the format asked for, as the last thing written.
SYSTEM: Final = (
    "You are answering benchmark questions. Give the answer in the format asked for, and make "
    "it the last thing you write. Do not restate the question."
)

# The framing experiment's three templates (PLAN.md section 4.3). `plain` is the one the bank
# runs use; the other two exist to measure what that choice costs.
TEMPLATES: Final[dict[str, str]] = {
    # `plain` ends by naming the exact form the answer must take. It used to end with a bare
    # "Answer:" completion prompt, which is fine for a model that answers immediately and
    # useless for one that reasons first: the reply then names several options and the parser
    # refuses to guess between them, correctly. An explicit final line makes the answer
    # findable without the parser ever having to guess.
    "plain": '{question}\n\n{options}\n\nGive the answer on a final line as "Answer: X".',
    "letter_only": (
        "{question}\n\n{options}\n\nReply with a single letter and nothing else. Answer:"
    ),
    "brief_reasoning": (
        "{question}\n\n{options}\n\nThink in at most two short sentences, then give the answer "
        'on a final line in the form "Answer: X".'
    ),
}

MATH_TEMPLATES: Final[dict[str, str]] = {
    "plain": "{question}\n\nGive the final answer inside \\boxed{{}}.",
    "letter_only": "{question}\n\nReply with the final answer inside \\boxed{{}} and nothing else.",
    "brief_reasoning": (
        "{question}\n\nWork in at most three short sentences, then give the final answer inside "
        "\\boxed{{}}."
    ),
}

LETTERS: Final = "ABCDEFGHIJ"


# Enough room for a model that reasons before it answers, and free for one that does not.
#
# A token cap is a ceiling, not a bill: output is billed on what a model generates, so a model
# that replies "B" costs two tokens whether the cap is 16 or 1024. The old cap of 16 therefore
# saved nothing and cost four of the eleven models their answers entirely, because they spent
# the whole budget reasoning and had none left to answer with.
#
# Uniform across models on purpose. A per-model budget is a per-model tuning decision inside a
# comparison between models, and there is no version of that which is not a thumb on the scale.
# `request-extras.yaml` can still override one, for a model that needs more than this, and any
# such override is a fact about that model that gets reported.
ANSWER_BUDGET: Final = 1024


@dataclass(frozen=True, slots=True)
class Settings:
    """Run settings, fixed for every bank run so that a rerun is the same experiment."""

    temperature: float = 0.0
    max_tokens: int = ANSWER_BUDGET
    # Equal to `max_tokens` since 2026-09-12. They differed when the answer-only cap was 16,
    # which was a cost decision that turned out to cost nothing and lose data. The framing
    # experiment's templates differ in what they ask for, which is the thing being measured;
    # giving one of them more room than the others would have measured the room as well.
    reasoning_max_tokens: int = ANSWER_BUDGET
    system: str = SYSTEM

    def tokens_for(self, template: str) -> int:
        return self.reasoning_max_tokens if template == "brief_reasoning" else self.max_tokens


@dataclass(frozen=True, slots=True)
class PanelEntry:
    """One model in the own-run panel, by gateway alias rather than by vendor model id."""

    alias: str  # resolved by the gateway's routing table, not here
    tier: str  # "frontier", "mid", "open weights", "local"
    coverage: str  # "full suite" or "adaptive subset"
    note: str


# PLAN.md section 3.3. Aliases, not vendor model identifiers: the gateway owns the mapping, so
# this list does not go stale when a vendor renames a model, and nothing here can accidentally
# become a call.
#
# Widened 2026-09-11: every model runs the full suite, where the three frontier models used to
# run the adaptive subset only. That was a cost compromise and the cost turned out not to exist
# (section 7: the whole programme is about US$5 measured, against US$60 assumed). It was not
# free: section 4.2 validates an adaptive ranking against the own-run full-suite ranking, and a
# model with no full-suite run cannot appear in the second one, so the panel had eleven models
# and the validation had eight, two of them a 3B and a 7B on the laptop. The frontier check is
# unchanged, because the adaptive subset is chosen from what was administered.
PANEL: Final[tuple[PanelEntry, ...]] = (
    PanelEntry("anthropic-haiku", "mid", "full suite", "full-suite anchor"),
    PanelEntry("anthropic-sonnet", "mid", "full suite", "full-suite anchor"),
    PanelEntry("anthropic-opus", "frontier", "full suite", "frontier check, and the top anchor"),
    PanelEntry("openai-mid", "mid", "full suite", "full-suite anchor"),
    PanelEntry("openai-frontier", "frontier", "full suite", "frontier check, and the top anchor"),
    PanelEntry("google-mid", "mid", "full suite", "full-suite anchor"),
    PanelEntry("google-frontier", "frontier", "full suite", "frontier check, and the top anchor"),
    PanelEntry("together-open-a", "open weights", "full suite", "cheap full-suite anchor"),
    PanelEntry("together-open-b", "open weights", "full suite", "cheap full-suite anchor"),
    PanelEntry("local-small-a", "local", "full suite", "extends the ability range downward"),
    PanelEntry("local-small-b", "local", "full suite", "extends the ability range downward"),
)


def rotate(options: list[str], rotation: int = 0) -> list[str]:
    """The options in the order the model is shown them.

    A rotation of k moves every option k places later in the list, so the correct answer takes a
    different letter without the wording of any option changing. PLAN.md section 4.3 asks for
    four cyclic permutations, which is `rotation` 0 to 3.

    One function for this, used to write the prompt and to read the reply. A parser that maps
    option text back to a letter has to agree with the prompt about which letter that was, and
    two implementations of a rotation come apart the first time one of them changes.
    """
    if not options:
        return []
    size = len(options)
    if size > len(LETTERS):
        raise ValueError(f"{size} options is more than the {len(LETTERS)} letters available")
    return [options[(index - rotation) % size] for index in range(size)]


def format_options(options: list[str], *, rotation: int = 0) -> str:
    """Lettered options, in the order `rotate` puts them."""
    shown = rotate(options, rotation)
    return "\n".join(f"{LETTERS[index]}. {text}" for index, text in enumerate(shown))


def answer_letter(options: list[str], answer: str, *, rotation: int = 0) -> str:
    """Which letter the correct answer wears under this rotation."""
    try:
        original = options.index(answer)
    except ValueError as exc:
        raise ValueError("the answer is not one of the options") from exc
    return LETTERS[(original + rotation) % len(options)]


def render(
    question: str,
    options: list[str],
    *,
    template: str = "plain",
    rotation: int = 0,
    free_response: bool = False,
) -> str:
    """The exact user message for one item."""
    templates = MATH_TEMPLATES if free_response else TEMPLATES
    if template not in templates:
        raise ValueError(f"unknown template {template!r}")
    return (
        templates[template]
        .format(question=question.strip(), options=format_options(options, rotation=rotation))
        .strip()
    )
