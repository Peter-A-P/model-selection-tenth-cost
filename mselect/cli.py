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
        summary = build.build_ollm_bank(version=version, progress=_say)
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


@app.command("report")
def report(version: str = typer.Option("v1")) -> None:
    """Regenerate the README results table and the figures from the saved outputs."""
    from mselect.report import build_report

    _say(build_report.write_all(version=version, progress=_say))


if __name__ == "__main__":  # pragma: no cover
    app()
