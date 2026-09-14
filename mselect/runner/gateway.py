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

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import yaml
from boundary import (
    BatchNotReady,
    BoundaryError,
    ChatRequest,
    ChatResponse,
    ConfigError,
    Gateway,
    ProviderError,
    SpendCapExceeded,
)
from boundary.cache import ExactMatchCache

from mselect.runner.administer import Prompt, Reply

CONFIG = Path(__file__).resolve().parent.parent / "config" / "boundary.yaml"
EXTRAS = CONFIG.parent / "request-extras.yaml"
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
    cache_namespace: str | None = None,
) -> Gateway:
    """A gateway configured for this project's panel. The caller closes it.

    `cache_namespace` gives this administration a cache of its own, and exists for exactly one
    reason: test-retest. That experiment asks the same model the same items at the same settings
    twice, so the request bytes are identical by design, so the content-hash cache answers the
    second administration out of the first one's replies. The agreement it then measures is the
    cache's, not the model's.

    Section 3.3's rule is not weakened by this and is the reason it is a namespace rather than a
    switch. Inside one administration a rerun still costs nothing, which is what that rule is
    for. Across administrations nothing is shared, which is what a repeated measurement needs.
    A flag that simply turned the cache off would make resuming a half-finished repeat cost full
    price, and resuming is not re-measuring.
    """
    gateway = Gateway.from_config(config, project=project, ledger_path=ledger_path)
    if cache_namespace and gateway.cache is not None:
        gateway.cache = ExactMatchCache(gateway.cache.root / cache_namespace)
    return gateway


def load_config(path: Path = CONFIG) -> dict[str, Any]:
    """The routing file as plain data, for reading rather than for calling.

    Opening a gateway constructs transports and a ledger. Pricing a run that has not happened
    yet needs neither, and should not be able to make a request by accident.
    """
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return dict(loaded) if isinstance(loaded, dict) else {}


def routes_of(config: dict[str, Any]) -> dict[str, dict[str, str]]:
    """alias -> {provider, model}, as the routing file states it."""
    routes: dict[str, dict[str, str]] = {}
    for alias, route in config.get("routes", {}).items():
        if isinstance(route, dict) and "provider" in route and "model" in route:
            routes[str(alias)] = {
                "provider": str(route["provider"]),
                "model": str(route["model"]),
            }
    return routes


def extras_of(path: Path = EXTRAS) -> dict[str, dict[str, Any]]:
    """alias -> vendor fields merged into that model's request body.

    Kept out of the gateway's routing file, whose schema forbids unknown route keys and is
    right to. A property of the model rather than of the provider: two Gemini models in one
    panel do not need the same thing, and a model that reasons by default has to be told not
    to or it spends an answer-only budget on thinking and returns nothing.
    """
    if not path.is_file():
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    aliases = loaded.get("aliases") if isinstance(loaded, dict) else None
    if not isinstance(aliases, dict):
        return {}
    return {str(a): dict(v) for a, v in aliases.items() if isinstance(v, dict)}


def omits_of(path: Path = EXTRAS) -> dict[str, frozenset[str]]:
    """alias -> request fields that must not be sent to that model at all.

    Distinct from `extras_of`, which adds fields. Sending `temperature: 0` and sending no
    temperature are different requests, and Anthropic's 5 family accepts only the second.
    """
    if not path.is_file():
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    omit = loaded.get("omit") if isinstance(loaded, dict) else None
    if not isinstance(omit, dict):
        return {}
    return {
        str(a): frozenset(str(f) for f in fields)
        for a, fields in omit.items()
        if isinstance(fields, list)
    }


def omits_temperature(alias: str, omits: dict[str, frozenset[str]] | None = None) -> bool:
    return "temperature" in (omits if omits is not None else omits_of()).get(alias, frozenset())


def tokens_of(path: Path = EXTRAS) -> dict[str, int]:
    """alias -> answer-only token budget, where the default is not enough.

    Section 3.3 gives every reply a small budget because the format is answer-only. A model
    that reasons before answering needs room for the reasoning as well, and giving it none
    does not produce a short answer, it produces no answer. Per alias and recorded, because
    a model that needs more room is a fact about that model worth reporting.
    """
    if not path.is_file():
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    tokens = loaded.get("tokens") if isinstance(loaded, dict) else None
    if not isinstance(tokens, dict):
        return {}
    return {str(a): int(n) for a, n in tokens.items() if isinstance(n, int)}


def route_key(
    alias: str,
    routes: dict[str, dict[str, str]],
    extras: dict[str, dict[str, Any]] | None = None,
) -> str:
    """What an alias resolves to right now, as one short string for the request hash.

    Everything outside the prompt that changes the reply: the provider, the model and any
    vendor fields sent with it. A run that is edited and resumed compares this, so repointing
    an alias at another model re-asks its items instead of inheriting the old model's answers.
    """
    route = routes.get(alias)
    if route is None:
        return ""
    key = f"{route['provider']}/{route['model']}"
    extra = (extras or {}).get(alias)
    if extra:
        digest = hashlib.sha256(
            json.dumps(extra, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:12]
        key = f"{key}+{digest}"
    return key


def _request(prompt: Prompt, extra: Mapping[str, Any] | None = None) -> ChatRequest:
    """One prompt as a vendor-neutral request. The alias goes through unresolved: the
    routes file decides what it means, which is the reason the panel is named by alias."""
    return ChatRequest(
        model=prompt.alias,
        system=prompt.system,
        messages=[{"role": "user", "content": prompt.user}],
        max_tokens=prompt.max_tokens,
        temperature=prompt.temperature,
        extra=dict(extra) if extra else {},
    )


def _why(response: ChatResponse) -> str:
    """A failed call, in the vendor's own words where it gave any.

    `str(response.status)` alone is "errored" or "batch_errored", which names the shape of the
    failure and not the failure. The body is already parsed on the response; six Anthropic
    failures reported nothing at all before this read it.
    """
    detail = ""
    raw = response.raw
    if isinstance(raw, dict):
        error = raw.get("error")
        if isinstance(error, dict):
            parts = [str(error[k]) for k in ("type", "message") if error.get(k)]
            detail = ": ".join(parts)
        elif isinstance(error, str):
            detail = error
        if not detail and raw.get("message"):
            detail = str(raw["message"])
    return f"{response.status}: {detail}" if detail else str(response.status)


# Statuses where the same request could succeed next time: the vendor was busy, slow, or
# briefly broken. Everything else a vendor says with a 4xx is a statement about the request,
# and repeating it repeats the answer and the bill.
_TRANSIENT_STATUS: Final = frozenset({408, 409, 429, 500, 502, 503, 504, 529})


def _retryable(response: ChatResponse) -> bool:
    if response.ok:
        return False
    status = response.status
    if isinstance(status, int):
        return status in _TRANSIENT_STATUS
    # A non-numeric status is a transport failure or a batch outcome word. A timeout deserves
    # another go; a batch item that errored does not, because the gateway drops the vendor's
    # reason (section 15.9) and asking again would only lose it again at the same price.
    return str(status).lower() in {"timeout", "connect_error", "read_error"}


def _reply(response: ChatResponse) -> Reply:
    return Reply(
        text=response.text,
        cost_usd=response.cost_usd,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        model_returned=response.model_returned,
        ledger_id=response.ledger_id,
        error=None if response.ok else _why(response),
        finish_reason=response.finish_reason,
        cached=response.cached,
        retryable=_retryable(response),
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
        extras: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self.gateway = gateway
        self.purpose = purpose
        self.run_id = run_id
        # Per-alias vendor fields from the routing file. Empty for most models; the ones that
        # reason by default need to be told not to, or an answer-only budget buys no answer.
        self.extras: Mapping[str, Mapping[str, Any]] = extras or {}
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
                    [_request(p, self.extras.get(p.alias)) for p in chunk],
                    purpose=self.purpose,
                    run_id=self.run_id,
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
                    _request(prompt, self.extras.get(prompt.alias)),
                    purpose=self.purpose,
                    run_id=self.run_id,
                )
            except SpendCapExceeded:
                # The one error that must not become a row. Carrying on would spend the rest
                # of the panel's budget recording that there is no budget.
                raise
            except BatchNotReady as e:
                # The results are not in yet, which says nothing at all about the item.
                out.append(Reply(text=None, error=str(e), retryable=True))
            except ProviderError as e:
                # The vendor answered and refused. A 429 or a 503 is worth asking again; a 400
                # is the vendor describing the request, and asking again gets the same 400.
                out.append(
                    Reply(
                        text=None,
                        error=f"{type(e).__name__}: {e}",
                        retryable=getattr(e, "status", None) in _TRANSIENT_STATUS,
                    )
                )
            except BoundaryError as e:
                # Transport: a timeout, a dropped connection, a proxy. Worth another go.
                out.append(Reply(text=None, error=f"{type(e).__name__}: {e}", retryable=True))
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
