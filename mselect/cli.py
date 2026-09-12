"""The mselect command line.

PLAN.md section 5: `mselect bank build`, `mselect fit`, `mselect simulate`, `mselect run`,
`mselect report`. `mselect run` needs the own-run panel and is added with the runner.
"""

from __future__ import annotations

import typer

from mselect import paths

app = typer.Typer(add_completion=False, help=__doc__)
bank_app = typer.Typer(
    add_completion=False, help="Fetch public per-item results and freeze a bank."
)
app.add_typer(bank_app, name="bank")


def _say(message: str) -> None:
    typer.echo(message)


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


@app.command("report")
def report(version: str = typer.Option("v1")) -> None:
    """Regenerate the README results table and the figures from the saved outputs."""
    from mselect.report import build_report

    _say(build_report.write_all(version=version, progress=_say))


if __name__ == "__main__":  # pragma: no cover
    app()
