"""Choosing the own-run suite, and pricing it before a call is made.

PLAN.md section 3.3 puts a panel of eight to eleven models against about 3,000 items. Which
3,000 is a decision about the experiment, so it is made once, seeded, written to a file and
committed, the way project 03 recorded its panel rather than leaving it to whatever the code
happened to pick on the day.

**Proportional to the pool, at random within each benchmark.** Two properties matter and they
pull the same way. The own-run "full suite" is the thing the adaptive test is measured against,
so it has to be a fair miniature of the bank the items were calibrated on: MMLU is 70 percent of
bank v1 and it is 70 percent of this suite. And the items have to be drawn without regard to
their fitted parameters, because choosing informative items would tilt the suite toward the ones
adaptive selection is good at and flatter the headline claim. A seeded uniform draw inside each
benchmark does both, and the seed is in the file so the draw can be repeated.

The estimate here is a planning number and not a safety mechanism. The cap in
`mselect/config/caps.yaml` is what actually stops a run, enforced by the gateway before a
request leaves; this only answers "can this fit inside that" before anyone finds out the
expensive way. Token counts are estimated from characters, so they are approximate by
construction and the ratio is stated rather than buried.
"""

from __future__ import annotations

import datetime as dt
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import yaml

from mselect import paths
from mselect.runner import prompts
from mselect.runner.items import Pool

# Characters per token. Four is the usual rule of thumb for English prose and it is optimistic
# for LaTeX and for legal text, both of which are in here, so the estimate carries a margin
# below rather than pretending to a precision it does not have.
CHARS_PER_TOKEN: Final = 4.0

# The estimate is multiplied by this before it is compared with a cap. A planning number that
# is exactly right is a planning number that is wrong half the time.
MARGIN: Final = 1.25

SUITE_FILE: Final = "own-run-suite.json"
DEFAULT_SIZE: Final = 3_000
DEFAULT_SEED: Final = 20_260_911


@dataclass(frozen=True, slots=True)
class Suite:
    """The items every full-suite model of the panel is asked, and how they were chosen."""

    version: str  # the bank the items belong to
    size: int
    seed: int
    chosen: str  # ISO date, so a rerun that draws differently is visible
    item_ids: tuple[str, ...]

    def by_benchmark(self, pool: Pool) -> dict[str, int]:
        benchmarks = {item.item_id: item.benchmark for item in pool.items}
        return dict(sorted(Counter(benchmarks[i] for i in self.item_ids).items()))

    def to_json(self) -> dict[str, Any]:
        return {
            "bank_version": self.version,
            "size": self.size,
            "seed": self.seed,
            "chosen": self.chosen,
            "item_ids": list(self.item_ids),
        }

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> Suite:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            version=str(raw["bank_version"]),
            size=int(raw["size"]),
            seed=int(raw["seed"]),
            chosen=str(raw["chosen"]),
            item_ids=tuple(str(i) for i in raw["item_ids"]),
        )


def choose(
    pool: Pool,
    size: int = DEFAULT_SIZE,
    *,
    seed: int = DEFAULT_SEED,
    today: str | None = None,
) -> Suite:
    """Draw a suite proportional to the pool's benchmark mix, at random inside each benchmark.

    Largest-remainder apportionment, so the counts sum to `size` exactly and a small benchmark
    is not rounded out of existence. A benchmark can never be given more items than it has.
    """
    if size > len(pool.items):
        raise ValueError(f"{size:,} items asked of a pool of {len(pool.items):,}")

    grouped: dict[str, list[str]] = defaultdict(list)
    for item in pool.items:
        grouped[item.benchmark].append(item.item_id)

    total = len(pool.items)
    exact = {name: size * len(ids) / total for name, ids in grouped.items()}
    counts = {name: min(int(value), len(grouped[name])) for name, value in exact.items()}
    # Hand out what rounding down left over, to the largest remainders first, skipping any
    # benchmark already giving everything it has.
    remaining = size - sum(counts.values())
    order = sorted(exact, key=lambda name: (-(exact[name] - int(exact[name])), name))
    while remaining > 0:
        progressed = False
        for name in order:
            if remaining == 0:
                break
            if counts[name] < len(grouped[name]):
                counts[name] += 1
                remaining -= 1
                progressed = True
        if not progressed:  # pragma: no cover - only if size > pool, already refused above
            raise ValueError("the pool cannot fill the suite")

    rng = random.Random(seed)
    picked: list[str] = []
    for name in sorted(grouped):
        ids = sorted(grouped[name])
        picked.extend(rng.sample(ids, counts[name]))

    return Suite(
        version=pool.version,
        size=size,
        seed=seed,
        chosen=today or dt.date.today().isoformat(),
        item_ids=tuple(sorted(picked)),
    )


@dataclass(frozen=True, slots=True)
class Line:
    """One panel member's share of the bill."""

    alias: str
    model: str
    calls: int
    input_tokens: int
    output_tokens: int
    usd: float | None  # None when no price is listed, which is never treated as zero
    # Where the output-token figure came from. A measured line rests on what this model
    # actually generated in a smoke run; a capped line assumes it generates the maximum it is
    # allowed, which for a reasoning model is a worst case rather than an expectation.
    measured: bool = False

    @property
    def priced(self) -> bool:
        return self.usd is not None


@dataclass(frozen=True, slots=True)
class Estimate:
    """What a run would cost, per alias and in total, before anything is sent."""

    lines: tuple[Line, ...]
    margin: float

    @property
    def usd(self) -> float:
        return sum(line.usd or 0.0 for line in self.lines)

    @property
    def with_margin(self) -> float:
        return self.usd * self.margin

    @property
    def unpriced(self) -> tuple[str, ...]:
        return tuple(line.alias for line in self.lines if not line.priced)

    @property
    def capped(self) -> tuple[str, ...]:
        """Aliases whose output is the cap rather than a measurement: an upper bound."""
        return tuple(line.alias for line in self.lines if not line.measured)

    def table(self) -> str:
        width = max((len(line.alias) for line in self.lines), default=5)
        rows = [
            f"  {line.alias:<{width}}  {line.calls:>7,} calls  "
            f"{line.input_tokens:>10,} in  {line.output_tokens:>7,} out"
            f"{'' if line.measured else ' (cap)'}  "
            + ("no price listed" if line.usd is None else f"US${line.usd:>8,.2f}")
            for line in self.lines
        ]
        rows.append(
            f"  {'total':<{width}}  US${self.usd:,.2f}, "
            f"US${self.with_margin:,.2f} with the {self.margin:g}x margin"
        )
        return "\n".join(rows)


def _tokens(characters: int) -> int:
    return int(characters / CHARS_PER_TOKEN + 0.5)


def observed_output(path: Path) -> dict[str, float]:
    """Mean output tokens per alias over replies that actually finished. Empty when there are none.

    **Only scored replies count, and the distinction is the whole point.** A reply that was cut
    off by the token cap did not generate what the model would have generated; it generated what
    it was allowed to. Averaging those in would report a reasoning model as costing sixteen
    tokens an item because sixteen was all it could have, which is the truncation reading itself
    back as a measurement and would understate the bill by an order of magnitude.

    A scored reply reached an identifiable answer, so the generation ran to the end of the thing
    being measured. An alias with none of those gets no entry, and the estimate falls back to the
    cap and says it is doing so, which is the right answer to "we have not measured this yet".
    """
    if not path.is_file():
        return {}
    totals: dict[str, list[int]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        alias, out = record.get("alias"), record.get("output_tokens")
        finished = record.get("error") is None and record.get("correct") is not None
        if finished and isinstance(alias, str) and isinstance(out, int) and out > 0:
            totals[alias].append(out)
    return {alias: sum(seen) / len(seen) for alias, seen in totals.items() if seen}


def _rate(prices: dict[str, Any], provider: str, model: str) -> dict[str, Any] | None:
    table = prices.get("per_million_tokens", {}).get(provider, {})
    entry = table.get(model)
    return dict(entry) if isinstance(entry, dict) else None


def load_prices(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return dict(loaded) if isinstance(loaded, dict) else {}


def latest_price_file(directory: Path) -> Path:
    """The newest dated price file. Repricing adds a file; it never edits an old one."""
    files = sorted(directory.glob("*.yaml"))
    if not files:
        raise FileNotFoundError(f"no price file in {directory}")
    return files[-1]


def estimate(
    suite: Suite,
    pool: Pool,
    routes: dict[str, dict[str, str]],
    prices: dict[str, Any],
    *,
    panel: tuple[prompts.PanelEntry, ...] = prompts.PANEL,
    settings: prompts.Settings | None = None,
    adaptive_items: int = 300,
    template: str = "plain",
    batch: bool = True,
    margin: float = MARGIN,
    observed: dict[str, float] | None = None,
) -> Estimate:
    """Price a run of `suite` against `panel`, from the committed price file.

    A model with no rate in the price file is reported as unpriced rather than as free, which
    is the same rule the gateway's ledger follows: the library never fills a rate in.

    `observed` gives mean output tokens per alias from a real run. Without it a line assumes
    the model generates its whole token budget, which since that budget became large enough
    for reasoning is a worst case and not an expectation. Lines built from the cap are marked,
    so a total can be read for what it is.
    """
    settings = settings or prompts.Settings()
    index = pool.index()
    chosen = [index[i] for i in suite.item_ids if i in index]

    characters = 0
    for item in chosen:
        body = prompts.render(
            item.question,
            list(item.options),
            template=template,
            free_response=item.free_response,
        )
        characters += len(settings.system) + len(body)
    per_item_in = _tokens(characters) / max(len(chosen), 1)
    cap = settings.tokens_for(template)
    seen = observed or {}

    lines: list[Line] = []
    for entry in panel:
        route = routes.get(entry.alias)
        if route is None:
            continue
        calls = len(chosen) if entry.coverage == "full suite" else adaptive_items
        provider = route["provider"]
        model = route["model"]
        input_tokens = int(per_item_in * calls + 0.5)
        per_item_out = seen.get(entry.alias)
        measured = per_item_out is not None
        output_tokens = int(min(per_item_out or cap, cap) * calls + 0.5)

        rate = _rate(prices, provider, model)
        usd: float | None
        if provider == "local":
            usd = 0.0  # the laptop; electricity is not billed per token
        elif rate is None:
            usd = None
        else:
            batched = batch and provider in BATCHING_PROVIDERS
            multiplier = float(rate.get("batch_multiplier", 1.0)) if batched else 1.0
            usd = multiplier * (
                input_tokens * float(rate["input"]) / 1e6
                + output_tokens * float(rate["output"]) / 1e6
            )
        lines.append(
            Line(
                alias=entry.alias,
                model=model,
                calls=calls,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                usd=usd,
                measured=measured,
            )
        )
    return Estimate(lines=tuple(lines), margin=margin)


def default_path(version: str = "v1") -> Path:
    """Beside the routes and the caps, and committed.

    Not under `out/`, which is gitignored. Which items the panel was asked is a decision about
    the experiment and part of the record of it, the same way the routing table is; a reader
    checking a published own-run number has to be able to see the list it was measured on.
    """
    return paths.ROOT / "mselect" / "config" / f"own-run-suite-{version}.json"


# PLAN.md section 4.3. Each experiment is priced as the calls it adds on top of the full suite,
# because the first administration of every item is already paid for above.
#   test-retest      500 items asked a second time
#   position bias    300 multiple-choice items, three further rotations
#   framing          300 items, two further templates, one of them the reasoning one
# Providers the gateway can actually send a batch to. A vendor publishing a batch rate is not
# the same as this project being able to claim it: `BoundaryCaller` tries a batch, and falls
# back to standard calls where the provider has no batch endpoint, which on 2026-09-13 was
# every provider except Anthropic. The ledger for the full-suite arm is the evidence: 9,045
# Anthropic calls carried a batch id and the other 21,000 carried none.
#
# Pricing the discount anyway understated OpenAI and Google by about half. When boundary grows
# another batch adapter, this is the line that changes.
BATCHING_PROVIDERS: Final[frozenset[str]] = frozenset({"anthropic"})

EXPERIMENTS: Final[tuple[tuple[str, int, int, str], ...]] = (
    ("test-retest", 500, 1, "plain"),
    ("position-bias", 300, 3, "plain"),
    ("framing (letter-only)", 300, 1, "letter_only"),
    ("framing (brief reasoning)", 300, 1, "brief_reasoning"),
)


def programme(
    suite: Suite,
    pool: Pool,
    routes: dict[str, dict[str, str]],
    prices: dict[str, Any],
    *,
    panel: tuple[prompts.PanelEntry, ...] = prompts.PANEL,
    settings: prompts.Settings | None = None,
    adaptive_items: int = 300,
    batch: bool = True,
    margin: float = MARGIN,
    observed: dict[str, float] | None = None,
) -> dict[str, Estimate]:
    """Every line of PLAN.md section 7, priced from the items rather than from an assumption.

    The section 7 table was written before the items existed, so its per-call token figure was
    a guess. These are the same lines costed against the suite that will actually be sent.
    """
    settings = settings or prompts.Settings()
    out = {
        "full suite and frontier check": estimate(
            suite,
            pool,
            routes,
            prices,
            panel=panel,
            settings=settings,
            adaptive_items=adaptive_items,
            batch=batch,
            margin=margin,
            observed=observed,
        )
    }
    # The experiments run on the full-suite models only: a frontier model is asked the adaptive
    # subset and nothing else, and a repeat of 500 items on a model that never saw them is not
    # a test-retest.
    repeated = tuple(e for e in panel if e.coverage == "full suite")
    for name, size, repeats, template in EXPERIMENTS:
        smaller = Suite(
            version=suite.version,
            size=min(size, len(suite.item_ids)),
            seed=suite.seed,
            chosen=suite.chosen,
            item_ids=suite.item_ids[: min(size, len(suite.item_ids))],
        )
        one = estimate(
            smaller,
            pool,
            routes,
            prices,
            panel=repeated,
            settings=settings,
            adaptive_items=0,
            template=template,
            batch=batch,
            margin=margin,
            observed=observed,
        )
        out[name] = Estimate(
            lines=tuple(
                Line(
                    alias=line.alias,
                    model=line.model,
                    calls=line.calls * repeats,
                    input_tokens=line.input_tokens * repeats,
                    output_tokens=line.output_tokens * repeats,
                    usd=None if line.usd is None else line.usd * repeats,
                    measured=line.measured,
                )
                for line in one.lines
            ),
            margin=margin,
        )
    return out
