"""The command line must not be able to spend money by accident.

Everything else in this repository is free to run. `mselect smoke` is the one command that can
reach a vendor, so the guard in front of it is worth a test rather than a careful habit: naming
an alias that costs money without saying so is refused before a gateway is even opened.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from mselect import paths
from mselect.cli import app

runner = CliRunner()


@pytest.fixture
def elsewhere(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Send `mselect run`'s records somewhere with nothing in it.

    Without this a guard test reads whatever the real panel run has recorded, and a run that
    has already covered the item under test leaves the command with nothing outstanding. It
    then exits 0 for a reason that has nothing to do with the guard, which is how this was
    found: green all evening, red at 07:30 the next morning, with no code changed between.
    """
    monkeypatch.setattr(paths, "OUT", tmp_path)
    return tmp_path


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


def test_run_without_yes_sends_nothing_and_says_so(elsewhere: Path) -> None:
    """The run that spends the budget must not start because someone typed the command."""
    result = runner.invoke(app, ["run", "--alias", "local-small-a", "--limit", "1"])
    assert "nothing to do" not in result.output, "the guard needs something to guard"
    assert result.exit_code == 1
    assert "nothing was sent" in result.output
    assert "--yes" in result.output


def test_run_prints_the_plan_before_it_asks_for_permission(elsewhere: Path) -> None:
    """A plan nobody can read is not a confirmation step."""
    result = runner.invoke(app, ["run", "--alias", "local-small-a", "--limit", "5"])
    assert "calls outstanding" in result.output
    assert "local-small-a" in result.output
    assert "llama3.2:3b" in result.output, "the model, not just the alias it hides behind"


def test_run_refuses_an_alias_with_no_route() -> None:
    result = runner.invoke(app, ["run", "--alias", "not-a-model", "--limit", "1", "--yes"])
    assert result.exit_code != 0
    assert "no route for not-a-model" in result.output


def test_rescore_reports_without_writing_by_default(tmp_path: Path) -> None:
    """A record file is evidence. Rewriting it should take saying so."""
    path = tmp_path / "run.jsonl"
    path.write_text("", encoding="utf-8")
    result = runner.invoke(app, ["rescore", str(path)])
    assert result.exit_code == 0
    assert "--write" not in result.output or "nothing written" in result.output


def test_rescore_refuses_a_file_that_is_not_there(tmp_path: Path) -> None:
    result = runner.invoke(app, ["rescore", str(tmp_path / "nope.jsonl")])
    assert result.exit_code != 0
    assert "no record file" in result.output


def test_a_repeat_administration_writes_somewhere_else(elsewhere: Path) -> None:
    """Test-retest compares two administrations, so merging them into one file loses the arm."""
    plain = runner.invoke(app, ["run", "--alias", "local-small-a", "--limit", "1"])
    again = runner.invoke(app, ["run", "--alias", "local-small-a", "--limit", "1", "--repeat", "2"])
    assert "own-run-plain-0.jsonl" in plain.output
    assert "own-run-plain-0-r2.jsonl" in again.output
    assert "administration 2" in again.output


def test_a_repeat_administration_says_it_will_not_read_the_first_one_s_cache(
    elsewhere: Path,
) -> None:
    """The defect this flag exists for, found 2026-09-14 before the arm was paid for.

    Test-retest asks the same model the same items at the same settings a day apart, so the
    request bytes are identical by design and this project caches every vendor response by the
    hash of exactly those bytes. `out/own-run-cache` held 24,053 replies at the time. The second
    administration would have been answered from the first one's replies for every model not
    going through a batch: free, instant, and in perfect agreement, which is the cache reading
    itself back as a reliability coefficient.

    So the plan a repeat prints has to say that it is not doing that, because a run that quietly
    costs nothing is exactly what a correct resume looks like.
    """
    result = runner.invoke(
        app, ["run", "--alias", "local-small-a", "--limit", "1", "--repeat", "2"]
    )
    assert "cache namespace 'r2'" in result.output
    assert "made again rather than answered from administration 1" in result.output


def test_repeat_zero_is_refused(elsewhere: Path) -> None:
    result = runner.invoke(
        app, ["run", "--alias", "local-small-a", "--limit", "1", "--repeat", "0"]
    )
    assert result.exit_code != 0
    assert "repeat starts at 1" in result.output


def test_a_run_without_the_keys_refuses_before_it_calls_anything(
    elsewhere: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Found the expensive way on 2026-09-14, in a shell that had no vendor keys.

    The run printed its plan, said 5,500 calls outstanding, and then spent ninety minutes
    writing 4,000 records that were all the same ConfigError. Nothing was spent and nothing was
    lost, because that error is transient by classification and a resume re-asks all of it, but
    for ninety minutes it looked exactly like a run that was working.

    A missing key cannot be true for one call and false for the next, so discovering it per call
    is discovering it thousands of times. It is checked once, before the first call, and it is
    checked even with --yes, because --yes is permission to spend rather than an instruction to
    proceed regardless.
    """
    for variable in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(variable, raising=False)
    result = runner.invoke(
        app,
        ["run", "--alias", "anthropic-haiku,local-small-a", "--limit", "2", "--yes"],
    )
    assert result.exit_code == 1
    assert "ANTHROPIC_API_KEY is not set" in result.output
    assert "anthropic-haiku cannot run without it" in result.output
    assert "nothing was sent" in result.output
    assert "local-small-a" not in result.output.split("is not set")[-1], (
        "the laptop needs no key and must not be named as blocked by one"
    )


def test_retest_refuses_when_there_is_no_second_administration(elsewhere: Path) -> None:
    """Comparing one administration against itself would report perfect reliability."""
    result = runner.invoke(app, ["retest"])
    assert result.exit_code != 0
    assert "no record file" in result.output


def test_retest_never_takes_a_yes_flag() -> None:
    """Both administrations were paid for once; the arithmetic is free forever."""
    assert "--yes" not in runner.invoke(app, ["retest", "--help"]).output


def test_a_rotated_run_sets_aside_the_items_a_rotation_cannot_touch(elsewhere: Path) -> None:
    """Found 2026-09-14, dry-running the position-bias arm before paying for it.

    `mselect run --rotation 1` raised on the first free-response item it met, because there is
    no option order to rotate in one. The guard is right and stays: accepting a rotation there
    would make the position-bias result cover items whose options never moved. What was missing
    is that the command never applied the experiment's own design, which PLAN.md section 4.3
    states as multiple choice. On this suite that is 2,783 items of 3,000.
    """
    result = runner.invoke(app, ["run", "--alias", "local-small-a", "--rotation", "1"])
    assert "free-response items set aside" in result.output
    assert "multiple-choice items remain" in result.output
    assert "cannot be rotated" not in result.output, "it must not reach the guard at all"


def test_an_unrotated_run_keeps_every_item(elsewhere: Path) -> None:
    """The filter belongs to the position-bias arm and must not quietly shrink the others."""
    result = runner.invoke(app, ["run", "--alias", "local-small-a"])
    assert "set aside" not in result.output
    assert "3,000 items per alias" in result.output
