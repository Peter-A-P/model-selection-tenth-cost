"""The one place this project talks to a vendor, through the portfolio gateway.

`administer.py` knows what to ask and how to score it and nothing else; this turns its
prompts into gateway calls and the answers back into replies. Keeping the two apart is what
lets every scoring rule be tested against a fake caller with no network, no key and no
dollar, and it is why this module holds no scoring logic at all.

Three things are decided here rather than in the experiment:

* **Batching is a price, not a method.** Anthropic's Message Batches cost half as much and
  answer later, which is fine everywhere in this project. It is a flag, and a provider
  without batch support falls back to one call at a time automatically, so a batch problem
  can delay the bill but never the panel.
* **A spend cap is a stop, not a result.** A failed call is recorded and the run carries on,
  because a timeout says nothing about an item. A cap being hit says the run should not
  continue, so it is raised rather than written into thousands of rows.
* **The gateway owns the money and the record.** Every call here writes a ledger row and a
  cost computed from returned usage, and this project's cap is enforced before the request
  leaves. Nothing in this file adds a price or a retry of its own.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from boundary import (
    BatchNotReady,
    BoundaryError,
    ChatRequest,
    ChatResponse,
    ConfigError,
    Gateway,
    SpendCapExceeded,
)

from mselect.runner.administer import Prompt, Reply

CONFIG = Path(__file__).resolve().parent.parent / "config" / "boundary.yaml"
PROJECT = "model-selection-tenth-cost"

# Anthropic accepts far more than this in one batch. The limit here is about what a failure
# costs: a batch that cannot be collected leaves its rows in flight, and a smaller batch
# means fewer of them and a shorter wait before anything at all is known.
DEFAULT_BATCH_SIZE = 250


def open_gateway(
    *,
    config: Path | str = CONFIG,
    project: str = PROJECT,
    ledger_path: Path | None = None,
) -> Gateway:
    """A gateway configured for this project's panel. The caller closes it."""
    return Gateway.from_config(config, project=project, ledger_path=ledger_path)


def _request(prompt: Prompt) -> ChatRequest:
    """One prompt as a vendor-neutral request. The alias goes through unresolved: the
    routes file decides what it means, which is the reason the panel is named by alias."""
    return ChatRequest(
        model=prompt.alias,
        system=prompt.system,
        messages=[{"role": "user", "content": prompt.user}],
        max_tokens=prompt.max_tokens,
        temperature=prompt.temperature,
    )


def _reply(response: ChatResponse) -> Reply:
    return Reply(
        text=response.text,
        cost_usd=response.cost_usd,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        model_returned=response.model_returned,
        ledger_id=response.ledger_id,
        error=None if response.ok else str(response.status),
    )


def _chunks(prompts: Sequence[Prompt], size: int) -> Iterator[Sequence[Prompt]]:
    for start in range(0, len(prompts), size):
        yield prompts[start : start + size]


class BoundaryCaller:
    """Answers prompts through the gateway, batching where the provider allows it."""

    def __init__(
        self,
        gateway: Gateway,
        *,
        purpose: str = "own-run",
        run_id: str | None = None,
        use_batches: bool = True,
        batch_size: int = DEFAULT_BATCH_SIZE,
        wait_s: float = 3600.0,
        poll_s: float = 30.0,
    ) -> None:
        self.gateway = gateway
        self.purpose = purpose
        self.run_id = run_id
        self.use_batches = use_batches
        self.batch_size = batch_size
        self.wait_s = wait_s
        self.poll_s = poll_s
        # Providers that turned out to have no batch endpoint. Asked once, not once per
        # chunk, so a panel of thousands of items does not repeat a known answer.
        self._no_batches: set[str] = set()

    def ask(self, prompts: Sequence[Prompt]) -> list[Reply]:
        if not prompts:
            return []
        replies: list[Reply] = []
        for alias, group in _by_alias(prompts):
            if self.use_batches and alias not in self._no_batches:
                batched = self._ask_batched(alias, group)
                if batched is not None:
                    replies.extend(batched)
                    continue
            replies.extend(self._ask_one_at_a_time(group))
        return replies

    def _ask_batched(self, alias: str, prompts: Sequence[Prompt]) -> list[Reply] | None:
        """None when this provider has no batch endpoint, so the caller falls back."""
        out: list[Reply] = []
        for chunk in _chunks(prompts, self.batch_size):
            try:
                handle = self.gateway.batch_submit(
                    [_request(p) for p in chunk], purpose=self.purpose, run_id=self.run_id
                )
            except ConfigError:
                # No batch support for this provider. Remembered, and the whole group falls
                # back rather than half of it, so one alias is measured one way.
                self._no_batches.add(alias)
                return None
            responses = self.gateway.batch_results(handle, wait_s=self.wait_s, poll_s=self.poll_s)
            out.extend(_reply(r) for r in responses)
        return out

    def _ask_one_at_a_time(self, prompts: Sequence[Prompt]) -> list[Reply]:
        out: list[Reply] = []
        for prompt in prompts:
            try:
                response = self.gateway.chat(
                    _request(prompt), purpose=self.purpose, run_id=self.run_id
                )
            except SpendCapExceeded:
                # The one error that must not become a row. Carrying on would spend the rest
                # of the panel's budget recording that there is no budget.
                raise
            except BatchNotReady as e:
                out.append(Reply(text=None, error=str(e)))
            except BoundaryError as e:
                # A refused or failed call is a fact about the call, not about the item. It
                # is recorded, the run continues, and `records.done` will offer it again.
                out.append(Reply(text=None, error=f"{type(e).__name__}: {e}"))
            else:
                out.append(_reply(response))
        return out


def _by_alias(prompts: Sequence[Prompt]) -> list[tuple[str, list[Prompt]]]:
    """Group while keeping order. A batch goes to one provider, and an alias is the only
    thing here that knows which provider that is."""
    groups: dict[str, list[Prompt]] = {}
    for prompt in prompts:
        groups.setdefault(prompt.alias, []).append(prompt)
    return list(groups.items())


def describe(gateway: Gateway, aliases: Sequence[str]) -> list[dict[str, Any]]:
    """What each alias points at right now, for a runner to print before it spends
    anything. The panel is a decision, so a run should say out loud what it is about to
    administer rather than leave it in a configuration file."""
    out: list[dict[str, Any]] = []
    for alias in aliases:
        ref = gateway.resolve(alias)
        out.append(
            {
                "alias": alias,
                "provider": ref.provider,
                "model": ref.model,
                "explicit": ref.explicit,
                "price_zero": ref.provider_config.price_zero,
            }
        )
    return out
