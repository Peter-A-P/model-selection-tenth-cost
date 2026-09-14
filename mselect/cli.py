"""The mselect command line.

PLAN.md section 5: `mselect bank build`, `mselect fit`, `mselect simulate`, `mselect run`,
`mselect report`. `mselect run` needs the own-run panel and is added with the runner.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import typer

from mselect import paths

if TYPE_CHECKING:
    from mselect.runner.prompts import PanelEntry, Settings

app = typer.Typer(add_completion=False, help=__doc__)
bank_app = typer.Typer(
    add_completion=False, help="Fetch public per-item results and freeze a bank."
)
app.add_typer(bank_app, name="bank")


def _say(message: str) -> None:
    typer.echo(message)


def prompts_settings(max_tokens: int) -> Settings:
    """Run settings with a bigger budget, to smoke-test a model that reasons before answering.

    Both budgets move together: the point of the override is to find out how much room a
    model needs before it produces any text at all, and the reasoning template is not a
    separate question at that stage.
    """
    from mselect.runner.prompts import Settings as RunSettings

    return RunSettings(max_tokens=max_tokens, reasoning_max_tokens=max_tokens)


def prompts_panel() -> tuple[PanelEntry, ...]:
    """The panel, imported late so the command line starts without pulling in the runner."""
    from mselect.runner.prompts import PANEL

    return PANEL


@bank_app.command("fetch")
def bank_fetch() -> None:
    """Warm the local cache of public per-item results. Free, resumable, cached forever."""
    from mselect.data import helm

    cache = helm.Cache(paths.ensure(paths.HELM_CACHE))
    with helm.Client(cache) as client:
        helm.fetch_all(client, progress=_say)


@bank_app.command("panel")
def bank_panel(
    size: int = typer.Option(400, help="How many leaderboard models to calibrate on."),
    strata: int = typer.Option(40, help="Equal-width bands of leaderboard average to span."),
    per_organisation: int = typer.Option(8, help="Cap per hub organisation."),
    seed: int = typer.Option(0),
) -> None:
    """Choose the Open LLM Leaderboard panel, prove it is readable, and write panel.json.

    Reads only. The details datasets are gated, so this needs HF_TOKEN in the environment, but
    a token is all the gate asks for: no request is filed and nothing is written to the account
    the token belongs to.
    """
    import json

    from mselect.data import ollm

    cache = ollm.Cache(paths.ensure(paths.OLLM_CACHE))
    with ollm.Client(cache, ollm.token_from_env()) as client:
        panel = ollm.assemble_panel(
            client,
            size=size,
            strata=strata,
            per_organisation=per_organisation,
            seed=seed,
            progress=_say,
        )
    path = ollm.panel_path()
    path.write_text(json.dumps(panel.to_json(), indent=2) + "\n", encoding="utf-8")
    _say(panel.describe())
    _say(f"wrote {path}")


@bank_app.command("fetch-ollm")
def bank_fetch_ollm() -> None:
    """Warm the cache of Open LLM Leaderboard per-item results for the chosen panel."""
    from mselect.data import ollm

    cache = ollm.Cache(paths.ensure(paths.OLLM_CACHE))
    panel = ollm.load_panel()
    with ollm.Client(cache, ollm.token_from_env()) as client:
        ollm.fetch_all(client, panel.members, progress=_say)


@bank_app.command("build")
def bank_build(
    version: str = typer.Option("v1", help="Bank version to write under mselect/bank/."),
    min_models: int = typer.Option(40, help="Keep an item only with at least this many responses."),
    source: str = typer.Option("helm", help="helm (bank v1) or ollm (bank v2)."),
) -> None:
    """Assemble the response matrix from the cache and freeze the bank."""
    from mselect.data import build

    if source == "ollm":
        summary = build.build_ollm_bank(version=version, min_models=min_models, progress=_say)
    else:
        summary = build.build_bank(version=version, min_models=min_models, progress=_say)
    _say(summary.describe())


@app.command("fit")
def fit(
    version: str = typer.Option("v1", help="Bank version to fit."),
    model: str = typer.Option("2pl", help="2pl or 3pl."),
    max_iter: int = typer.Option(200),
) -> None:
    """Fit item parameters by marginal maximum likelihood and write them beside the bank."""
    from mselect.irt import run_fit

    _say(run_fit.fit_bank(version=version, kind=model, max_iter=max_iter, progress=_say))


@app.command("diagnose")
def diagnose(version: str = typer.Option("v1")) -> None:
    """Fit statistics, Q3 local dependence, dimensionality and DIF for a fitted bank."""
    from mselect.irt import run_fit

    _say(run_fit.diagnose_bank(version=version, progress=_say))


@app.command("simulate")
def simulate(
    version: str = typer.Option("v1"),
    seed: int = typer.Option(0),
    max_items: int = typer.Option(400),
    evaluate: int = typer.Option(0, help="Hold out this many models (0 means every model)."),
) -> None:
    """Leave-one-model-out adaptive simulation against random and stratified baselines."""
    from mselect.cat import run_sim

    _say(
        run_sim.run(
            version=version,
            seed=seed,
            max_items=max_items,
            evaluate=evaluate or None,
            progress=_say,
        )
    )


@app.command("crossbank")
def crossbank(
    first: str = typer.Option("v1", help="The bank whose items are matched from."),
    second: str = typer.Option("v2", help="The bank matched into."),
    model: str = typer.Option("2pl", help="2pl or 3pl."),
    seed: int = typer.Option(0),
) -> None:
    """Do two banks agree about which shared items are hard? The portability number."""
    from mselect.experiments import crossbank as experiment

    _say(experiment.run(first=first, second=second, kind=model, seed=seed, progress=_say))


@app.command("routes")
def routes() -> None:
    """Check the panel without calling anything: identifiers, prices, keys and the local server.

    Everything here is free and reads only files and environment variables. Key values are
    never printed, only whether the variable is set. What this cannot check is the one thing
    that matters most, and it says so at the end.
    """
    import os
    import socket
    import ssl
    import urllib.error
    import urllib.parse
    import urllib.request

    import truststore

    from mselect.runner import gateway, prompts
    from mselect.runner import suite as suite_mod

    def reachable(base_url: str) -> str:
        """A TLS handshake and nothing else. Proves verification, spends nothing."""
        host = urllib.parse.urlparse(base_url).hostname
        if host is None:
            return "no host in the base url"
        context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        try:
            with (
                socket.create_connection((host, 443), timeout=8) as raw,
                context.wrap_socket(raw, server_hostname=host) as tls,
            ):
                certificate = tls.getpeercert() or {}
                # getpeercert types the issuer loosely: a tuple of relative distinguished
                # names, each a tuple of (key, value) pairs. Flattened rather than trusted.
                issuer: dict[str, str] = {}
                for rdn in certificate.get("issuer", ()):
                    if isinstance(rdn, tuple):
                        for pair in rdn:
                            if isinstance(pair, tuple) and len(pair) == 2:
                                issuer[str(pair[0])] = str(pair[1])
                who = issuer.get("organizationName") or issuer.get("commonName") or "?"
                return f"verified, issued by {who}"
        except ssl.SSLCertVerificationError as e:
            return f"CERTIFICATE NOT VERIFIED ({e.verify_message or e})"
        except (OSError, TimeoutError) as e:
            return f"UNREACHABLE ({type(e).__name__})"

    config = gateway.load_config()
    all_routes = gateway.routes_of(config)
    providers = config.get("providers", {})
    prices = suite_mod.load_prices(
        suite_mod.latest_price_file(gateway.CONFIG.parent / str(config.get("prices", "prices")))
    )
    listed = prices.get("per_million_tokens", {})

    _say(f"panel of {len(prompts.PANEL)}, from {gateway.CONFIG.name}")
    _say(f"prices from {suite_mod.latest_price_file(gateway.CONFIG.parent / 'prices').name}")
    _say("")
    problems: list[str] = []
    for entry in prompts.PANEL:
        route = all_routes.get(entry.alias)
        if route is None:
            problems.append(f"{entry.alias} has no route in {gateway.CONFIG.name}")
            _say(f"  {entry.alias:<18} NO ROUTE")
            continue
        provider = providers.get(route["provider"], {})
        free = bool(provider.get("price_zero"))
        priced = route["model"] in listed.get(route["provider"], {})
        if not priced and not free:
            problems.append(f"{entry.alias} ({route['model']}) has no rate in the price file")
        key_env = provider.get("api_key_env")
        if key_env is None:
            key = "not needed"
        elif os.environ.get(str(key_env)):
            key = f"{key_env} set"
        else:
            key = f"{key_env} MISSING"
            problems.append(f"{entry.alias} needs {key_env}, which is not set")
        money = "free" if free else ("priced" if priced else "NO PRICE")
        _say(f"  {entry.alias:<18} {route['model']:<40} {entry.coverage:<15} {money:<9} {key}")

    # Where the run can happen. A network that inspects TLS re-signs certificates with its own
    # authority, which shows up here as an issuer that is not the vendor's, or as a refusal.
    _say("\nTLS to each vendor, verified against the OS trust store, sending nothing:")
    for name, spec in providers.items():
        if not isinstance(spec, dict) or spec.get("price_zero"):
            continue
        base = str(spec.get("base_url", ""))
        verdict = reachable(base)
        _say(f"  {name:<14} {urllib.parse.urlparse(base).hostname:<36} {verdict}")
        if not verdict.startswith("verified"):
            problems.append(f"{name} cannot be reached from this network: {verdict}")

    local = [
        name
        for name, spec in providers.items()
        if isinstance(spec, dict) and spec.get("price_zero")
    ]
    for name in local:
        base = str(providers[name].get("base_url", ""))
        try:
            with urllib.request.urlopen(f"{base}/models", timeout=3) as response:
                ok = response.status == 200
        except (urllib.error.URLError, OSError, ValueError):
            ok = False
        _say(f"\nlocal provider {name!r} at {base}: {'answering' if ok else 'not answering'}")
        if not ok:
            problems.append(f"the local server for {name!r} is not answering at {base}")

    _say("")
    if problems:
        _say(f"{len(problems)} thing(s) to fix before a run:")
        for problem in problems:
            _say(f"  ! {problem}")
    else:
        _say("every checkable thing checks out: routes, prices, keys, local server.")
    _say(
        "\nWhat this cannot check without spending: whether each model answers in the"
        "\nanswer-only format the run assumes. Google's Flash returned no text inside a"
        "\nsmall token budget on 2026-09-10 because it reasons by default, which is the"
        "\nkind of thing only a real call finds. `boundary smoke <provider>` is that call."
    )


@app.command("suite")
def suite(
    version: str = typer.Option("v1", help="Which bank the items come from."),
    size: int = typer.Option(3_000, help="How many items every full-suite model is asked."),
    seed: int = typer.Option(20_260_911, help="The draw. Recorded in the file."),
    write: bool = typer.Option(False, "--write", help="Save the suite. Without this, print only."),
) -> None:
    """Choose the own-run item set and price it. Sends nothing and costs nothing.

    Run this before `mselect run`. It prints what each panel member would be asked, what the
    bill would be against the committed price file, and how that compares with the cap. The
    cap is enforced by the gateway regardless; this is so nobody finds out the expensive way.
    """
    from mselect.runner import gateway, items
    from mselect.runner import suite as suite_mod

    pool = items.administrable(version)
    _say(pool.summary())
    for reason, n in pool.by_reason().items():
        _say(f"  excluded {n:,}: {reason}")

    chosen = suite_mod.choose(pool, size, seed=seed)
    counts = ", ".join(f"{k} {v:,}" for k, v in chosen.by_benchmark(pool).items())
    _say(f"\nsuite: {chosen.size:,} items, seed {chosen.seed}, drawn {chosen.chosen}")
    _say(f"  {counts}")

    config = gateway.load_config()
    prices = suite_mod.load_prices(
        suite_mod.latest_price_file(gateway.CONFIG.parent / str(config.get("prices", "prices")))
    )
    routes = gateway.routes_of(config)
    # What the models actually generated, where a smoke run has measured it. Without this the
    # output figure is the token cap, which since the cap became big enough for reasoning is a
    # worst case rather than an expectation.
    observed = suite_mod.observed_output(paths.OUT / version / "smoke.jsonl")
    lines = suite_mod.programme(chosen, pool, routes, prices, observed=observed)
    if observed:
        _say(f"\noutput tokens measured for {len(observed)} alias(es) from the smoke records")
    _say("\nestimated at the batch rate, before any call:")
    total = 0.0
    unpriced: set[str] = set()
    for name, part in lines.items():
        _say(f"\n  {name}")
        for line in part.lines:
            money = "no price listed" if line.usd is None else f"US${line.usd:>8,.2f}"
            _say(
                f"    {line.alias:<18} {line.calls:>7,} calls  "
                f"{line.input_tokens:>10,} in  {line.output_tokens:>8,} out  {money}"
            )
        total += part.usd
        unpriced |= set(part.unpriced)
        _say(f"    {'':<18} {'':>7}         subtotal US${part.usd:,.2f}")
    _say(
        f"\nwhole programme: US${total:,.2f}, "
        f"US${total * suite_mod.MARGIN:,.2f} with the {suite_mod.MARGIN:g}x margin"
    )
    for alias in sorted(unpriced):
        _say(f"  ! {alias} has no rate in the price file and is not counted as free")
    capped = sorted({a for part in lines.values() for a in part.capped})
    if capped:
        _say(
            f"  (cap) on {len(capped)} alias(es) means the output figure is the token budget "
            "rather than a measurement, so those lines are an upper bound: " + ", ".join(capped)
        )
    estimate = lines["full suite and frontier check"]

    caps = suite_mod.load_prices(gateway.CONFIG.parent / str(config.get("caps", "caps.yaml")))
    per_run = caps.get("projects", {}).get("model-selection-tenth-cost", {}).get("per_run_usd")
    if per_run is not None:
        verdict = "fits" if estimate.with_margin <= float(per_run) else "DOES NOT FIT"
        _say(
            f"\nper-run cap US${float(per_run):,.2f}: the largest single run "
            f"(US${estimate.with_margin:,.2f} with margin) {verdict}"
        )
    monthly = caps.get("projects", {}).get("model-selection-tenth-cost", {}).get("monthly_usd")
    if monthly is not None:
        fits = total * suite_mod.MARGIN <= float(monthly)
        verdict = "fits" if fits else "DOES NOT FIT"
        _say(f"monthly cap US${float(monthly):,.2f}: the whole programme {verdict}")

    if write:
        _say(f"\nwrote {chosen.save(suite_mod.default_path(version))}")
    else:
        _say("\nnothing written; pass --write to save the suite")


@app.command("models")
def models(
    show: str = typer.Option(
        "", "--show", help="Also print every listed model for these providers, comma-separated."
    ),
    like: str = typer.Option("", "--like", help="With --show, only ids containing this."),
) -> None:
    """Ask each vendor what it serves today and check the panel against the answer.

    Listing models generates no tokens, so it is billed nowhere and cannot spend. It needs the
    keys, which is the only reason it is not part of `mselect routes`.

    A model the vendor does not list is reported as "not listed" rather than wrong, because
    some accounts can call models that do not appear. It tells you which routes to doubt;
    `mselect smoke` is what settles them.
    """
    from mselect.runner import catalogue, gateway

    config = gateway.load_config()
    all_routes = gateway.routes_of(config)
    providers = config.get("providers", {})
    aliases = [e.alias for e in prompts_panel()]

    checks, catalogues = catalogue.check_routes(providers, all_routes, aliases)

    for name, found in sorted(catalogues.items()):
        if found.ok:
            _say(f"{name}: {len(found.models)} models listed")
        else:
            _say(f"{name}: could not ask ({found.error})")

    _say("")
    missing: list[catalogue.RouteCheck] = []
    for check in checks:
        if check.listed:
            mark, note = "ok", ""
        elif not check.known:
            mark, note = "?", "provider did not answer"
        else:
            mark, note = "NOT LISTED", ""
            missing.append(check)
        _say(f"  {mark:<10} {check.alias:<18} {check.provider}/{check.model} {note}")

    if missing:
        _say(f"\n{len(missing)} route(s) name something the vendor does not list:")
        for check in missing:
            near = ", ".join(check.nearest) if check.nearest else "nothing similar listed"
            _say(f"  {check.alias} -> {check.model}")
            _say(f"      the vendor does list: {near}")

    for name in [s.strip() for s in show.split(",") if s.strip()]:
        found = catalogues.get(name) or catalogue.fetch(name, providers.get(name, {}))
        if not found.ok:
            _say(f"\n{name}: {found.error}")
            continue
        wanted = [m for m in found.models if not like or like.lower() in m.lower()]
        _say(f"\n{name}, {len(wanted)} of {len(found.models)} models:")
        for model in wanted:
            _say(f"  {model}")


@app.command("smoke")
def smoke(
    alias: str = typer.Option(
        "", "--alias", help="Comma-separated aliases. Default: the price-zero local ones only."
    ),
    items_per_alias: int = typer.Option(3, "--items", help="Items per alias. Keep it small."),
    version: str = typer.Option("v1", help="Which bank the items come from."),
    yes: bool = typer.Option(
        False, "--yes", help="Required before any alias that costs money is called."
    ),
    batch: bool = typer.Option(
        True, "--batch/--no-batch", help="Use vendor batches. --no-batch to see a real error."
    ),
    max_tokens: int = typer.Option(
        0, "--max-tokens", help="Override the answer-only budget, to test a model that reasons."
    ),
) -> None:
    """Ask a few real items and report what came back. The only check that needs a real call.

    Reports, per alias: how many replies parsed, how many scored, how many came back
    unreadable, and what it cost. An unparsed reply is the finding, not a wrong answer: it
    means the answer-only format does not hold for that model and the run would record noise.

    Costs nothing by default. Naming a vendor alias needs `--yes`, and `mselect routes` is
    what to run first.

    Use `--no-batch` when a call failed and the reason is not in the output. A batch reports
    one outcome word per request and the gateway has nowhere to put the vendor's message, so
    "batch_errored" is all a batched failure can ever say. The same request sent on its own
    returns an error body, which is the thing worth reading. A failed call is billed nowhere,
    so the diagnosis is free either way.
    """
    from mselect.runner import administer, gateway, items, records
    from mselect.runner import suite as suite_mod

    config = gateway.load_config()
    all_routes = gateway.routes_of(config)
    providers = config.get("providers", {})

    def is_free(name: str) -> bool:
        route = all_routes.get(name)
        if route is None:
            return False
        return bool(providers.get(route["provider"], {}).get("price_zero"))

    named = [a.strip() for a in alias.split(",") if a.strip()]
    wanted = named or [e.alias for e in prompts_panel() if is_free(e.alias)]
    unknown = [a for a in wanted if a not in all_routes]
    if unknown:
        raise typer.BadParameter(f"no route for {', '.join(unknown)}")
    paid = [a for a in wanted if not is_free(a)]
    if paid and not yes:
        _say(f"{', '.join(paid)} would cost money. Nothing was called.")
        _say("Run `mselect routes` first, then add --yes when you mean it.")
        raise typer.Exit(code=1)

    pool = items.administrable(version)
    index = pool.index()
    chosen_suite = suite_mod.Suite.load(suite_mod.default_path(version))
    # The first few items of the committed suite, so a smoke run asks what the panel will ask
    # rather than something easier. Spread across benchmarks: a model can parse a multiple
    # choice and still not box a MATH answer.
    picked: list[str] = []
    seen: set[str] = set()
    for item_id in chosen_suite.item_ids:
        item = index.get(item_id)
        if item is None or item.benchmark in seen:
            continue
        seen.add(item.benchmark)
        picked.append(item_id)
        if len(picked) >= items_per_alias:
            break
    for item_id in chosen_suite.item_ids:
        if len(picked) >= items_per_alias:
            break
        if item_id in index and item_id not in picked:
            picked.append(item_id)
    asking = [index[i] for i in picked]

    _say(f"asking {len(asking)} items of {len(wanted)} alias(es): {', '.join(wanted)}")
    _say(f"items: {', '.join(sorted({i.benchmark for i in asking}))}")

    path = paths.ensure(paths.OUT / version) / "smoke.jsonl"
    already = records.done(path)
    total = 0.0
    with gateway.open_gateway() as gw:
        caller = gateway.BoundaryCaller(
            gw,
            purpose="smoke",
            run_id="smoke",
            extras=gateway.extras_of(),
            use_batches=batch,
        )
        omits = gateway.omits_of()
        extras = gateway.extras_of()
        budgets = gateway.tokens_of()
        for name in wanted:
            # --max-tokens overrides everything, for exploring; otherwise a model that needs
            # more room than the answer-only default takes it from the configuration.
            budget = max_tokens or budgets.get(name, 0)
            written = administer.administer(
                asking,
                name,
                caller,
                done=already,
                settings=prompts_settings(budget) if budget else None,
                omit_temperature=gateway.omits_temperature(name, omits),
                route=gateway.route_key(name, all_routes, extras),
            )
            if not written:
                _say(f"  {name:<18} every cell already recorded; nothing called")
                continue
            records.append(path, written)
            scored = sum(1 for r in written if r.correct is not None)
            right = sum(r.correct or 0 for r in written)
            unparsed = sum(1 for r in written if r.unparsed)
            failed = sum(1 for r in written if r.error is not None)
            cached = sum(1 for r in written if r.cached)
            # A reply the gateway could not price. It is not free and it is not an error: the
            # call happened and the bill for it is unknown, which is the one outcome a cost
            # table cannot represent. together-open-b wrote three of these on 2026-09-12
            # because Together reported prompt-cache tokens and the price file had no rate
            # for them, and it took reading the ledger to notice.
            uncosted = sum(
                1 for r in written if r.cost_usd is None and r.error is None and not r.cached
            )
            cost = sum(r.cost_usd or 0.0 for r in written)
            total += cost
            # A paid alias reporting nothing is either a cache hit or a costing failure, and
            # those look identical in a total. Saying which turns a puzzle into a fact.
            note = f" ({cached} from cache)" if cached else ""
            if uncosted:
                note += f" ({uncosted} UNCOSTED: the gateway has no rate for this call)"
            _say(
                f"  {name:<18} {scored}/{len(written)} scored, {right} correct, "
                f"{unparsed} unparsed, {failed} failed, US${cost:.5f}{note}"
            )
            for record in written:
                if record.unparsed or record.error is not None:
                    reply = (record.reply or "")[:60].replace("\n", " ")
                    why = record.error or "unparsed"
                    _say(f"      ! {record.benchmark} {why}: {reply!r}")
    _say(f"\ntotal US${total:.5f}; records in {path}")
    if total == 0.0 and paid:
        _say(
            "no cost recorded for a paid alias. A cache hit is the usual reason and says so "
            "above; anything else means checking the ledger before trusting it."
        )


@app.command("run")
def run(
    version: str = typer.Option("v1", help="Which bank the suite and items come from."),
    alias: str = typer.Option("", "--alias", help="Comma-separated. Default: the whole panel."),
    template: str = typer.Option("plain", help="plain, letter_only or brief_reasoning."),
    rotation: int = typer.Option(0, help="Option rotation, for the position-bias experiment."),
    limit: int = typer.Option(0, help="Stop after this many items per alias. 0 means all."),
    chunk: int = typer.Option(250, help="Items per write. Smaller loses less to a crash."),
    batch: bool = typer.Option(True, "--batch/--no-batch", help="Use vendor batches."),
    yes: bool = typer.Option(False, "--yes", help="Required. Without it, prints the plan only."),
) -> None:
    """Administer the committed suite to the panel. This is the run that spends the budget.

    Resumable: the record file is the state, keyed by request hash, so stopping and starting
    again costs nothing for the part already done and re-asks anything whose request changed.
    Without --yes it prints the plan and sends nothing.
    """
    from mselect.runner import administer, gateway, items, records
    from mselect.runner import suite as suite_mod

    pool = items.administrable(version)
    index = pool.index()
    chosen = suite_mod.Suite.load(suite_mod.default_path(version))
    ordered = [index[i] for i in chosen.item_ids if i in index]
    if limit:
        ordered = ordered[:limit]

    config = gateway.load_config()
    all_routes = gateway.routes_of(config)
    extras = gateway.extras_of()
    omits = gateway.omits_of()
    budgets = gateway.tokens_of()

    named = [a.strip() for a in alias.split(",") if a.strip()]
    wanted = named or [e.alias for e in prompts_panel() if e.coverage == "full suite"]
    unknown = [a for a in wanted if a not in all_routes]
    if unknown:
        raise typer.BadParameter(f"no route for {', '.join(unknown)}")

    path = paths.ensure(paths.OUT / version) / f"own-run-{template}-{rotation}.jsonl"
    already = records.done(path)

    _say(f"suite {chosen.size:,} items, seed {chosen.seed}, drawn {chosen.chosen}")
    _say(f"template {template!r}, rotation {rotation}, {len(ordered):,} items per alias")
    _say(f"records {path}")

    plan: list[tuple[str, int]] = []
    for name in wanted:
        prompts = administer.build_prompts(
            ordered,
            name,
            template=template,
            rotation=rotation,
            settings=prompts_settings(budgets[name]) if name in budgets else None,
            omit_temperature=gateway.omits_temperature(name, omits),
            route=gateway.route_key(name, all_routes, extras),
        )
        plan.append((name, sum(1 for p in prompts if p.request_sha256 not in already)))
    outstanding = sum(n for _, n in plan)

    _say("")
    for name, n in plan:
        state = "nothing to do" if n == 0 else f"{n:,} to ask"
        _say(f"  {name:<18} {all_routes[name]['model']:<40} {state}")
    _say(f"\n{outstanding:,} calls outstanding of {len(ordered) * len(wanted):,}")
    if already:
        _say(f"{len(already):,} already recorded and will not be asked again")

    if not outstanding:
        _say("\nnothing to do.")
        return
    if not yes:
        _say("\nnothing was sent. Add --yes to run it.")
        raise typer.Exit(code=1)

    spent = records.spend_usd(path)
    totals = {"scored": 0, "correct": 0, "unparsed": 0, "failed": 0, "uncosted": 0}
    with gateway.open_gateway() as gw:
        caller = gateway.BoundaryCaller(
            gw, purpose="own-run", run_id=f"{version}-{template}-{rotation}", use_batches=batch
        )
        for name in wanted:
            budget = budgets.get(name, 0)
            asked = 0
            for start in range(0, len(ordered), chunk):
                piece = ordered[start : start + chunk]
                written = administer.administer(
                    piece,
                    name,
                    caller,
                    template=template,
                    rotation=rotation,
                    settings=prompts_settings(budget) if budget else None,
                    done=already,
                    omit_temperature=gateway.omits_temperature(name, omits),
                    route=gateway.route_key(name, all_routes, extras),
                )
                if not written:
                    continue
                records.append(path, written)
                asked += len(written)
                totals["scored"] += sum(1 for r in written if r.correct is not None)
                totals["correct"] += sum(r.correct or 0 for r in written)
                totals["unparsed"] += sum(1 for r in written if r.unparsed)
                totals["failed"] += sum(1 for r in written if r.error is not None)
                totals["uncosted"] += sum(
                    1 for r in written if r.cost_usd is None and r.error is None and not r.cached
                )
                spent += sum(r.cost_usd or 0.0 for r in written)
                _say(f"  {name:<18} {asked:>6,} asked, US${spent:,.4f} so far")
    _say(
        f"\n{totals['scored']:,} scored, {totals['correct']:,} correct, "
        f"{totals['unparsed']:,} unparsed, {totals['failed']:,} failed, "
        f"{totals['uncosted']:,} uncosted"
    )
    _say(f"US${spent:,.4f} recorded in {path}")
    if totals["failed"]:
        _say("failed calls are not done: running this again picks them up and costs only those.")


@app.command("rescore")
def rescore(
    records_path: str = typer.Argument(..., help="A record file written by `mselect run`."),
    version: str = typer.Option("v1", help="Which bank the items come from."),
    write: bool = typer.Option(False, "--write", help="Replace the file. Without this, report."),
) -> None:
    """Score stored replies again with the current parser. Calls nothing and costs nothing.

    Three of this project's scoring defects were found after the calls were paid for. The reply
    text is kept precisely so that fixing one of those never means asking again: a run costs
    US$13.40 and six hours, and a parser fix should cost neither.

    Prints what moved and why. `--write` replaces the file, keeping the previous one beside it.
    """
    import json
    import pathlib
    import shutil

    from mselect.runner import administer, items

    path = pathlib.Path(records_path)
    if not path.is_file():
        raise typer.BadParameter(f"no record file at {path}")

    pool = items.administrable(version)
    index = pool.index()

    rows: list[dict[str, object]] = []
    moved: list[tuple[str, object, object]] = []
    counts = {"read": 0, "rescored": 0, "changed": 0, "no reply": 0, "unknown item": 0}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        counts["read"] += 1
        rows.append(row)

        reply_text = row.get("reply")
        if not isinstance(reply_text, str) or not reply_text.strip():
            counts["no reply"] += 1
            continue
        item = index.get(str(row.get("item_id")))
        if item is None:
            counts["unknown item"] += 1
            continue

        rotation = int(row.get("rotation") or 0)
        key = administer.expected_key(item, rotation=rotation)
        parsed, correct = administer.score(
            item, administer.Reply(text=reply_text), rotation=rotation
        )
        counts["rescored"] += 1
        before = (row.get("parsed"), row.get("correct"))
        if before != (parsed, correct):
            counts["changed"] += 1
            moved.append((str(row.get("item_id")), before, (parsed, correct)))
        row["key"] = key
        row["parsed"] = parsed
        row["correct"] = correct
        row["unparsed"] = correct is None and row.get("error") is None

    _say(f"{counts['read']:,} records, {counts['rescored']:,} rescored")
    _say(f"  {counts['no reply']:,} had no reply to score, and keep what they said")
    if counts["unknown item"]:
        _say(f"  {counts['unknown item']:,} name an item not in the pool and were left alone")
    _say(f"  {counts['changed']:,} changed")
    for item_id, was, now in moved[:10]:
        _say(f"      {item_id}  {was} -> {now}")
    if len(moved) > 10:
        _say(f"      and {len(moved) - 10:,} more")

    if not counts["changed"]:
        _say("\nthe current parser agrees with every stored score. Nothing to write.")
        return
    if not write:
        _say("\nnothing written; pass --write to apply this.")
        return

    backup = path.with_suffix(path.suffix + ".before-rescore")
    shutil.copy2(path, backup)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )
    _say(f"\nwrote {path}")
    _say(f"previous version kept at {backup}")


@app.command("validate")
def validate(
    version: str = typer.Option("v1", help="Which bank the parameters and suite come from."),
    template: str = typer.Option("plain", help="Which record file: plain, letter_only, ..."),
    rotation: int = typer.Option(0, help="Which option rotation."),
    kind: str = typer.Option("2pl", help="2pl or 3pl parameters."),
    resamples: int = typer.Option(2000, help="Bootstrap resamples for every interval."),
    seed: int = typer.Option(0, help="Recorded, so the numbers can be reproduced."),
    every_item: bool = typer.Option(
        False,
        "--every-item",
        help="Rank on each model's own items instead of the items all of them answered.",
    ),
    write: bool = typer.Option(True, "--write/--no-write", help="Save the result under out/."),
) -> None:
    """Does a bank fitted on other people's models rank ours, from how few items, for how much?

    Calls nothing and costs nothing: the replies were paid for once and are read from the record
    file. Nothing is refitted either, which is the point. The item parameters were estimated
    without seeing any model in this panel, so what comes out is a transfer result rather than a
    description of the panel it was fitted on.

    --every-item drops the common frame and ranks each model on whatever it answered. That is
    almost always the wrong comparison and it is offered because seeing the difference is the
    fastest way to understand why: a model refused partway through is otherwise ranked on a
    different item set from the models it is being ranked against.
    """
    import json

    from mselect.experiments import ownrun

    path = ownrun.default_path(version, template, rotation)
    if not path.is_file():
        raise typer.BadParameter(f"no record file at {path}; run `mselect run` first")

    panel = ownrun.load(path, version=version, kind=kind)
    result = ownrun.validate(panel, seed=seed, resamples=resamples, common_frame=not every_item)
    _say(f"records {path}")
    _say(f"bank {version}, {kind} parameters, fitted without any model in this panel")
    if every_item:
        _say("ranking each model on its own items: not a like-for-like comparison")
    _say("")
    _say(result.describe())

    if not write:
        return
    out = paths.ensure(paths.out_for(version)) / f"own-run-validation-{template}-{rotation}.json"
    payload = result.to_json()
    payload["bank_version"] = version
    payload["kind"] = kind
    payload["seed"] = seed
    payload["resamples"] = resamples
    payload["common_frame"] = not every_item
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _say(f"\nwrote {out}")


@app.command("report")
def report(version: str = typer.Option("v1")) -> None:
    """Regenerate the README results table and the figures from the saved outputs."""
    from mselect.report import build_report

    _say(build_report.write_all(version=version, progress=_say))


if __name__ == "__main__":  # pragma: no cover
    app()
