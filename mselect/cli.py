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


@bank_app.command("build")
def bank_build(
    version: str = typer.Option("v1", help="Bank version to write under mselect/bank/."),
    min_models: int = typer.Option(40, help="Keep an item only with at least this many responses."),
) -> None:
    """Assemble the response matrix from the cache and freeze the bank."""
    from mselect.data import build

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
) -> None:
    """Leave-one-model-out adaptive simulation against random and stratified baselines."""
    from mselect.cat import run_sim

    _say(run_sim.run(version=version, seed=seed, max_items=max_items, progress=_say))


@app.command("report")
def report(version: str = typer.Option("v1")) -> None:
    """Regenerate the README results table and the figures from the saved outputs."""
    from mselect.report import build_report

    _say(build_report.write_all(version=version, progress=_say))


if __name__ == "__main__":  # pragma: no cover
    app()
