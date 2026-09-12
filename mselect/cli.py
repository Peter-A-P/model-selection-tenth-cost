"""The mselect command line.

PLAN.md section 5: `mselect bank build`, `mselect fit`, `mselect simulate`, `mselect run`,
`mselect report`. `mselect run` needs the own-run panel and is added with the runner.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import typer

from mselect import paths

if TYPE_CHECKING:
    from mselect.runner.prompts import PanelEntry

app = typer.Typer(add_completion=False, help=__doc__)
bank_app = typer.Typer(
    add_completion=False, help="Fetch public per-item results and freeze a bank."
)
app.add_typer(bank_app, name="bank")


def _say(message: str) -> None:
    typer.echo(message)


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
    lines = suite_mod.programme(chosen, pool, routes, prices)
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
) -> None:
    """Ask a few real items and report what came back. The only check that needs a real call.

    Reports, per alias: how many replies parsed, how many scored, how many came back
    unreadable, and what it cost. An unparsed reply is the finding, not a wrong answer: it
    means the answer-only format does not hold for that model and the run would record noise.

    Costs nothing by default. Naming a vendor alias needs `--yes`, and `mselect routes` is
    what to run first.
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
        caller = gateway.BoundaryCaller(gw, purpose="smoke", run_id="smoke")
        for name in wanted:
            written = administer.administer(asking, name, caller, done=already)
            if not written:
                _say(f"  {name:<18} every cell already recorded; nothing called")
                continue
            records.append(path, written)
            scored = sum(1 for r in written if r.correct is not None)
            right = sum(r.correct or 0 for r in written)
            unparsed = sum(1 for r in written if r.unparsed)
            failed = sum(1 for r in written if r.error is not None)
            cost = sum(r.cost_usd or 0.0 for r in written)
            total += cost
            _say(
                f"  {name:<18} {scored}/{len(written)} scored, {right} correct, "
                f"{unparsed} unparsed, {failed} failed, US${cost:.5f}"
            )
            for record in written:
                if record.unparsed or record.error is not None:
                    reply = (record.reply or "")[:60].replace("\n", " ")
                    why = record.error or "unparsed"
                    _say(f"      ! {record.benchmark} {why}: {reply!r}")
    _say(f"\ntotal US${total:.5f}; records in {path}")
    if total == 0.0 and paid:
        _say("no cost recorded for a paid alias: check the ledger before trusting that.")


@app.command("report")
def report(version: str = typer.Option("v1")) -> None:
    """Regenerate the README results table and the figures from the saved outputs."""
    from mselect.report import build_report

    _say(build_report.write_all(version=version, progress=_say))


if __name__ == "__main__":  # pragma: no cover
    app()
