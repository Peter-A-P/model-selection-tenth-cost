"""The command line must not be able to spend money by accident.

Everything else in this repository is free to run. `mselect smoke` is the one command that can
reach a vendor, so the guard in front of it is worth a test rather than a careful habit: naming
an alias that costs money without saying so is refused before a gateway is even opened.
"""

from __future__ import annotations

from typer.testing import CliRunner

from mselect.cli import app

runner = CliRunner()


def test_naming_a_paid_alias_without_yes_calls_nothing() -> None:
    result = runner.invoke(app, ["smoke", "--alias", "anthropic-haiku"])
    assert result.exit_code == 1
    assert "would cost money" in result.output
    assert "Nothing was called" in result.output


def test_a_paid_alias_is_refused_even_alongside_a_free_one() -> None:
    """A free alias in the list does not buy the paid one a pass."""
    result = runner.invoke(app, ["smoke", "--alias", "local-small-a,openai-mid"])
    assert result.exit_code == 1
    assert "openai-mid" in result.output
    assert "local-small-a would cost" not in result.output


def test_an_unknown_alias_is_refused_before_anything_opens() -> None:
    result = runner.invoke(app, ["smoke", "--alias", "not-a-model"])
    assert result.exit_code != 0
    assert "no route for not-a-model" in result.output


def test_routes_and_suite_never_take_a_yes_flag() -> None:
    """The two free commands stay free: nothing about them should be gated on spending."""
    for command in ("routes", "suite"):
        help_text = runner.invoke(app, [command, "--help"]).output
        assert "--yes" not in help_text, f"{command} should not be able to spend"


def test_run_without_yes_sends_nothing_and_says_so() -> None:
    """The run that spends the budget must not start because someone typed the command."""
    result = runner.invoke(app, ["run", "--alias", "local-small-a", "--limit", "1"])
    assert result.exit_code == 1
    assert "nothing was sent" in result.output
    assert "--yes" in result.output


def test_run_prints_the_plan_before_it_asks_for_permission() -> None:
    """A plan nobody can read is not a confirmation step."""
    result = runner.invoke(app, ["run", "--alias", "local-small-a", "--limit", "5"])
    assert "calls outstanding" in result.output
    assert "local-small-a" in result.output
    assert "llama3.2:3b" in result.output, "the model, not just the alias it hides behind"


def test_run_refuses_an_alias_with_no_route() -> None:
    result = runner.invoke(app, ["run", "--alias", "not-a-model", "--limit", "1", "--yes"])
    assert result.exit_code != 0
    assert "no route for not-a-model" in result.output
