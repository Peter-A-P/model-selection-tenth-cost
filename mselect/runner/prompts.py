"""The prompt templates the own-run panel will use, and the panel itself.

No vendor call lives here, and none can: the runner that sends these goes through the portfolio
gateway (project 04), whose Message Batches support lands in its v0.2, and it needs this
project's spend caps set first. PLAN.md section 3.3 and section 13.4. What is here is everything
that can be settled without spending anything: the exact strings, the panel, and the settings,
so that when the runner is wired up there is nothing left to decide and the framing experiment
already has its three templates written down.

PLAN.md section 3.3: answer-only output, temperature 0, fixed system prompt, small `max_tokens`.
PLAN.md section 4.3: the framing experiment compares the answer-only template against a
letter-only instruction and a brief-reasoning template, on the same items.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

SYSTEM: Final = (
    "You are answering benchmark questions. Reply with the answer only, in the format asked "
    "for. Do not explain, restate the question, or add anything else."
)

# The framing experiment's three templates (PLAN.md section 4.3). `plain` is the one the bank
# runs use; the other two exist to measure what that choice costs.
TEMPLATES: Final[dict[str, str]] = {
    "plain": "{question}\n\n{options}\n\nAnswer:",
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


@dataclass(frozen=True, slots=True)
class Settings:
    """Run settings, fixed for every bank run so that a rerun is the same experiment."""

    temperature: float = 0.0
    max_tokens: int = 16  # answer-only; the reasoning template overrides this
    reasoning_max_tokens: int = 256
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
PANEL: Final[tuple[PanelEntry, ...]] = (
    PanelEntry("anthropic-haiku", "mid", "full suite", "full-suite anchor"),
    PanelEntry("anthropic-sonnet", "mid", "full suite", "full-suite anchor"),
    PanelEntry("anthropic-opus", "frontier", "adaptive subset", "frontier check"),
    PanelEntry("openai-mid", "mid", "full suite", "full-suite anchor"),
    PanelEntry("openai-frontier", "frontier", "adaptive subset", "frontier check"),
    PanelEntry("google-mid", "mid", "full suite", "full-suite anchor"),
    PanelEntry("google-frontier", "frontier", "adaptive subset", "frontier check"),
    PanelEntry("together-open-a", "open weights", "full suite", "cheap full-suite anchor"),
    PanelEntry("together-open-b", "open weights", "full suite", "cheap full-suite anchor"),
    PanelEntry("local-small-a", "local", "full suite", "extends the ability range downward"),
    PanelEntry("local-small-b", "local", "full suite", "extends the ability range downward"),
)


def format_options(options: list[str], *, rotation: int = 0) -> str:
    """Lettered options, optionally cyclically rotated for the position-bias experiment.

    A rotation of k moves every option k places later in the list, so the correct answer takes a
    different letter without the wording of any option changing. PLAN.md section 4.3 asks for
    four cyclic permutations, which is `rotation` 0 to 3.
    """
    if not options:
        return ""
    size = len(options)
    if size > len(LETTERS):
        raise ValueError(f"{size} options is more than the {len(LETTERS)} letters available")
    rotated = [options[(index - rotation) % size] for index in range(size)]
    return "\n".join(f"{LETTERS[index]}. {text}" for index, text in enumerate(rotated))


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
