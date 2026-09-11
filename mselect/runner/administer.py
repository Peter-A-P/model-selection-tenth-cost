"""Administering items to a model and scoring what comes back.

This is the own-run panel's core (PLAN.md section 3.3). It holds no vendor call and no
credential: what to ask, how to score it, and what to record. The thing that actually talks
to a vendor is a `Caller`, which the runner is handed, and the only implementation that will
ever make a request goes through the portfolio gateway (project 04).

That seam is deliberate and is the reason this module exists separately from the gateway
adapter. The scoring rules are the part that has to be right, they are the part worth
testing against adversarial replies, and none of that testing should need a network, a key
or a dollar. A fake caller exercises every path here.

Two rules carry over from `parse.py` and are enforced here rather than assumed:

* An unparseable reply is recorded as unparsed, never as incorrect. Scoring a reply the
  parser could not read as a wrong answer would turn parser mistakes into item statistics,
  and the broken-item report would then be a report about this code.
* An item whose answer key is not among its own options cannot be scored at all. It is
  recorded with the reason and no score, because that is a broken item and the report
  should say so rather than quietly mark every model wrong on it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from mselect.runner import parse
from mselect.runner.prompts import Settings, answer_letter, render

MULTIPLE_CHOICE = "multiple_choice"


@dataclass(frozen=True, slots=True)
class Item:
    """One item to administer: what the bank knows, plus the text from the cache.

    The bank itself stores no item text (PLAN.md section 13.5), so the text arrives here
    from the HELM cache and is never written back into the bank or committed.
    """

    item_id: str
    benchmark: str
    kind: str
    question: str
    options: tuple[str, ...] = ()
    answer: str = ""

    @property
    def free_response(self) -> bool:
        return self.kind != MULTIPLE_CHOICE


@dataclass(frozen=True, slots=True)
class Prompt:
    """One administration before it is sent.

    `cell` is what makes a run resumable: it names the model, the item and the two things
    the experiments vary, so a record already on disk can be skipped without asking what it
    contained.
    """

    item_id: str
    alias: str
    template: str
    rotation: int
    system: str
    user: str
    max_tokens: int
    temperature: float

    @property
    def cell(self) -> str:
        return cell_key(self.item_id, self.alias, self.template, self.rotation)

    @property
    def request_sha256(self) -> str:
        """Content hash of everything that determines the reply. The gateway caches on its
        own request bytes; this is the same idea one level up, and it is what a record
        carries so that a changed prompt can never be mistaken for a cached old one."""
        body = json.dumps(
            [self.alias, self.system, self.user, self.max_tokens, round(self.temperature, 6)],
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return hashlib.sha256(body.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class Reply:
    """What a caller got back. A caller never scores; scoring happens here, once."""

    text: str | None
    cost_usd: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    model_returned: str | None = None
    ledger_id: int | None = None
    error: str | None = None


class Caller(Protocol):
    """Something that can answer prompts. One call, one reply, in the same order.

    Whether the implementation sends them one at a time or as a vendor batch is its own
    business: that choice changes the price and not the experiment, which is why it is not
    visible here.
    """

    def ask(self, prompts: Sequence[Prompt]) -> list[Reply]: ...


@dataclass(frozen=True, slots=True)
class Administration:
    """One scored cell, as it is written to the record file."""

    cell: str
    item_id: str
    alias: str
    benchmark: str
    kind: str
    template: str
    rotation: int
    key: str | None
    parsed: str | None
    correct: int | None
    unparsed: bool
    reply: str | None
    cost_usd: float | None
    input_tokens: int
    output_tokens: int
    model_returned: str | None
    ledger_id: int | None
    error: str | None
    request_sha256: str
    settings: dict[str, Any] = field(default_factory=dict)

    @property
    def scored(self) -> bool:
        return self.correct is not None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def cell_key(item_id: str, alias: str, template: str, rotation: int) -> str:
    return f"{alias}|{item_id}|{template}|{rotation}"


def build_prompts(
    items: Sequence[Item],
    alias: str,
    *,
    template: str = "plain",
    rotation: int = 0,
    settings: Settings | None = None,
) -> list[Prompt]:
    """The exact messages for one model over a set of items.

    Rotation applies only to multiple-choice items: there is no option order to rotate on a
    free-response item, and silently accepting one would make the position-bias experiment
    look as though it covered items it never touched.
    """
    fixed = settings or Settings()
    out: list[Prompt] = []
    for item in items:
        if item.free_response and rotation:
            raise ValueError(
                f"item {item.item_id} is free response and cannot be rotated; "
                "rotation belongs to the position-bias experiment on multiple choice"
            )
        out.append(
            Prompt(
                item_id=item.item_id,
                alias=alias,
                template=template,
                rotation=rotation,
                system=fixed.system,
                user=render(
                    item.question,
                    list(item.options),
                    template=template,
                    rotation=rotation,
                    free_response=item.free_response,
                ),
                max_tokens=fixed.tokens_for(template),
                temperature=fixed.temperature,
            )
        )
    return out


def expected_key(item: Item, *, rotation: int = 0) -> str | None:
    """The answer this item is scored against, or None when it cannot be scored.

    None means a broken item, not a wrong model: a multiple-choice item whose answer key is
    not one of its own options, or one with no key at all.
    """
    if item.free_response:
        return item.answer.strip() or None
    if not item.options or not item.answer:
        return None
    try:
        return answer_letter(list(item.options), item.answer, rotation=rotation)
    except ValueError:
        return None


def score(item: Item, reply: Reply, *, rotation: int = 0) -> tuple[str | None, int | None]:
    """(what the model answered, whether it was right).

    A None score is never a zero. It means one of: the call failed, the reply could not be
    parsed, or the item has no usable key.
    """
    if reply.error is not None or reply.text is None:
        return None, None
    key = expected_key(item, rotation=rotation)
    if key is None:
        return None, None
    if item.free_response:
        return parse.parse_math(reply.text), parse.grade_math(reply.text, key)
    n_options = len(item.options)
    return (
        parse.parse_choice(reply.text, n_options),
        parse.grade_choice(reply.text, key, n_options),
    )


def administer(
    items: Sequence[Item],
    alias: str,
    caller: Caller,
    *,
    template: str = "plain",
    rotation: int = 0,
    settings: Settings | None = None,
    done: frozenset[str] = frozenset(),
) -> list[Administration]:
    """Ask one model every item it has not already been asked, and score the replies.

    `done` is the set of cells already on disk, so a run that stopped half way is resumed by
    reading its own record file rather than by paying for the first half again. Nothing is
    sent for a cell in `done`, and nothing is returned for it either: the record that exists
    is the record.
    """
    fixed = settings or Settings()
    prompts = [
        p
        for p in build_prompts(items, alias, template=template, rotation=rotation, settings=fixed)
        if p.cell not in done
    ]
    if not prompts:
        return []
    by_id = {item.item_id: item for item in items}
    replies = caller.ask(prompts)
    if len(replies) != len(prompts):
        raise RuntimeError(
            f"caller returned {len(replies)} replies for {len(prompts)} prompts; a reply has "
            "to correspond to the prompt in the same position or the scores are misaligned"
        )

    records: list[Administration] = []
    for prompt, reply in zip(prompts, replies, strict=True):
        item = by_id[prompt.item_id]
        key = expected_key(item, rotation=rotation)
        parsed, correct = score(item, reply, rotation=rotation)
        error = reply.error
        if error is None and key is None:
            error = "item has no usable answer key"
        records.append(
            Administration(
                cell=prompt.cell,
                item_id=item.item_id,
                alias=alias,
                benchmark=item.benchmark,
                kind=item.kind,
                template=template,
                rotation=rotation,
                key=key,
                parsed=parsed,
                correct=correct,
                unparsed=error is None and reply.text is not None and correct is None,
                reply=reply.text,
                cost_usd=reply.cost_usd,
                input_tokens=reply.input_tokens,
                output_tokens=reply.output_tokens,
                model_returned=reply.model_returned,
                ledger_id=reply.ledger_id,
                error=error,
                request_sha256=prompt.request_sha256,
                settings={
                    "temperature": fixed.temperature,
                    "max_tokens": prompt.max_tokens,
                    "system_sha256": hashlib.sha256(fixed.system.encode("utf-8")).hexdigest()[:16],
                },
            )
        )
    return records
