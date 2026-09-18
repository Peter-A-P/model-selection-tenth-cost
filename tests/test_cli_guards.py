"""The command line must not be able to spend money by accident.

Everything else in this repository is free to run. `mselect smoke` is the one command that can
reach a vendor, so the guard in front of it is worth a test rather than a careful habit: naming
an alias that costs money without saying so is refused before a gateway is even opened.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mselect import paths
from mselect.cli import app

runner = CliRunner()

_STYLING = re.compile(r"\x1b\[[0-9;]*m")


def unstyled(output: str) -> str:
    """What the command said, with the escape sequences rich wraps it in taken out.

    Typer forces colour on whenever `GITHUB_ACTIONS` is set, and rich's option highlighter then
    writes `--rotation` as a styled `-` followed by a styled `-rotation`, so the flag is no
    longer in the output as a literal string. Nothing sets that variable on a laptop, the same
    output arrives unstyled, and the same assertion passes: that is why
    `test_the_experiment_readers_refuse_a_missing_arm` was green on Windows and red on the
    runner for six builds in a row. Every assertion about output goes through here, the ones
    that check a flag is absent above all, because under styling those could not have failed.
    """
    return _STYLING.sub("", output)


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
    assert "would cost money" in unstyled(result.output)
    assert "Nothing was called" in unstyled(result.output)


def test_a_paid_alias_is_refused_even_alongside_a_free_one() -> None:
    """A free alias in the list does not buy the paid one a pass."""
    result = runner.invoke(app, ["smoke", "--alias", "local-small-a,openai-mid"])
    assert result.exit_code == 1
    assert "openai-mid" in unstyled(result.output)
    assert "local-small-a would cost" not in unstyled(result.output)


def test_an_unknown_alias_is_refused_before_anything_opens() -> None:
    result = runner.invoke(app, ["smoke", "--alias", "not-a-model"])
    assert result.exit_code != 0
    assert "no route for not-a-model" in unstyled(result.output)


def test_routes_and_suite_never_take_a_yes_flag() -> None:
    """The two free commands stay free: nothing about them should be gated on spending."""
    for command in ("routes", "suite"):
        help_text = unstyled(runner.invoke(app, [command, "--help"]).output)
        assert "--yes" not in help_text, f"{command} should not be able to spend"


def test_run_without_yes_sends_nothing_and_says_so(elsewhere: Path) -> None:
    """The run that spends the budget must not start because someone typed the command."""
    result = runner.invoke(app, ["run", "--alias", "local-small-a", "--limit", "1"])
    assert "nothing to do" not in unstyled(result.output), "the guard needs something to guard"
    assert result.exit_code == 1
    assert "nothing was sent" in unstyled(result.output)
    assert "--yes" in unstyled(result.output)


def test_run_prints_the_plan_before_it_asks_for_permission(elsewhere: Path) -> None:
    """A plan nobody can read is not a confirmation step."""
    result = runner.invoke(app, ["run", "--alias", "local-small-a", "--limit", "5"])
    assert "calls outstanding" in unstyled(result.output)
    assert "local-small-a" in unstyled(result.output)
    assert "llama3.2:3b" in unstyled(result.output), "the model, not just the alias it hides behind"


def test_run_refuses_an_alias_with_no_route() -> None:
    result = runner.invoke(app, ["run", "--alias", "not-a-model", "--limit", "1", "--yes"])
    assert result.exit_code != 0
    assert "no route for not-a-model" in unstyled(result.output)


def test_rescore_reports_without_writing_by_default(tmp_path: Path) -> None:
    """A record file is evidence. Rewriting it should take saying so."""
    path = tmp_path / "run.jsonl"
    path.write_text("", encoding="utf-8")
    result = runner.invoke(app, ["rescore", str(path)])
    assert result.exit_code == 0
    said = unstyled(result.output)
    assert "--write" not in said or "nothing written" in said


def test_rescore_refuses_a_file_that_is_not_there(tmp_path: Path) -> None:
    result = runner.invoke(app, ["rescore", str(tmp_path / "nope.jsonl")])
    assert result.exit_code != 0
    assert "no record file" in unstyled(result.output)


def test_a_repeat_administration_writes_somewhere_else(elsewhere: Path) -> None:
    """Test-retest compares two administrations, so merging them into one file loses the arm."""
    plain = runner.invoke(app, ["run", "--alias", "local-small-a", "--limit", "1"])
    again = runner.invoke(app, ["run", "--alias", "local-small-a", "--limit", "1", "--repeat", "2"])
    assert "own-run-plain-0.jsonl" in unstyled(plain.output)
    assert "own-run-plain-0-r2.jsonl" in unstyled(again.output)
    assert "administration 2" in unstyled(again.output)


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
    assert "cache namespace 'r2'" in unstyled(result.output)
    assert "made again rather than answered from administration 1" in unstyled(result.output)


def test_repeat_zero_is_refused(elsewhere: Path) -> None:
    result = runner.invoke(
        app, ["run", "--alias", "local-small-a", "--limit", "1", "--repeat", "0"]
    )
    assert result.exit_code != 0
    assert "repeat starts at 1" in unstyled(result.output)


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
    assert "ANTHROPIC_API_KEY is not set" in unstyled(result.output)
    assert "anthropic-haiku cannot run without it" in unstyled(result.output)
    assert "nothing was sent" in unstyled(result.output)
    assert "local-small-a" not in unstyled(result.output).split("is not set")[-1], (
        "the laptop needs no key and must not be named as blocked by one"
    )


def test_retest_refuses_when_there_is_no_second_administration(elsewhere: Path) -> None:
    """Comparing one administration against itself would report perfect reliability."""
    result = runner.invoke(app, ["retest"])
    assert result.exit_code != 0
    assert "no record file" in unstyled(result.output)


def test_retest_never_takes_a_yes_flag() -> None:
    """Both administrations were paid for once; the arithmetic is free forever."""
    assert "--yes" not in unstyled(runner.invoke(app, ["retest", "--help"]).output)


def test_a_rotated_run_sets_aside_the_items_a_rotation_cannot_touch(elsewhere: Path) -> None:
    """Found 2026-09-14, dry-running the position-bias arm before paying for it.

    `mselect run --rotation 1` raised on the first free-response item it met, because there is
    no option order to rotate in one. The guard is right and stays: accepting a rotation there
    would make the position-bias result cover items whose options never moved. What was missing
    is that the command never applied the experiment's own design, which PLAN.md section 4.3
    states as multiple choice. On this suite that is 2,783 items of 3,000.
    """
    result = runner.invoke(app, ["run", "--alias", "local-small-a", "--rotation", "1"])
    assert "free-response items set aside" in unstyled(result.output)
    assert "multiple-choice items remain" in unstyled(result.output)
    assert "cannot be rotated" not in unstyled(result.output), "it must not reach the guard at all"


def test_an_unrotated_run_keeps_every_item(elsewhere: Path) -> None:
    """The filter belongs to the position-bias arm and must not quietly shrink the others."""
    result = runner.invoke(app, ["run", "--alias", "local-small-a"])
    assert "set aside" not in unstyled(result.output)
    assert "3,000 items per alias" in unstyled(result.output)


def test_the_experiment_readers_refuse_a_missing_arm(elsewhere: Path) -> None:
    """Both commands name the run that would produce what they are missing.

    They exist because the analyses did and nothing read a record file into them: the arms could
    have been paid for and left as two JSONL files with no way to turn them into a number.
    """
    bias = runner.invoke(app, ["position-bias", "--rotations", "0,1"])
    assert bias.exit_code != 0
    said = unstyled(bias.output)
    assert "no record file" in said and "--rotation" in said

    frame = runner.invoke(app, ["framing", "--templates", "plain,letter_only"])
    assert frame.exit_code != 0
    said = unstyled(frame.output)
    assert "no record file" in said and "--template" in said


def test_comparing_needs_something_to_compare_against(elsewhere: Path) -> None:
    for command, flag in (("position-bias", "--rotations"), ("framing", "--templates")):
        result = runner.invoke(app, [command, flag, "0" if "rot" in flag else "plain"])
        assert result.exit_code != 0
        assert "at least two" in unstyled(result.output)


def test_neither_reader_can_spend_anything() -> None:
    """The arms are paid for once; the arithmetic is free forever."""
    for command in ("position-bias", "framing"):
        assert "--yes" not in unstyled(runner.invoke(app, [command, "--help"]).output)


def test_a_resume_asks_only_for_the_keys_its_remaining_work_needs(
    elsewhere: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A key check is about the calls that are left, not about the aliases that were named.

    Found on the restart of 2026-09-14: the run was refused for want of an Anthropic key while
    five of the eleven models on the command line had nothing left to do. The refusal is right
    when there is work behind it and wrong when there is not, and the difference is the plan the
    command has already printed two lines above.
    """
    from mselect.cli import prompts_settings
    from mselect.runner import administer, gateway, items, records
    from mselect.runner import suite as suite_mod

    settled = "anthropic-haiku"
    pool = items.administrable("v1")
    index = pool.index()
    chosen = suite_mod.Suite.load(suite_mod.default_path("v1"))
    ordered = [index[i] for i in chosen.item_ids if i in index][:2]
    config = gateway.load_config()
    all_routes = gateway.routes_of(config)
    budgets = gateway.tokens_of()
    asked = administer.build_prompts(
        ordered,
        settled,
        settings=prompts_settings(budgets[settled]) if settled in budgets else None,
        omit_temperature=gateway.omits_temperature(settled, gateway.omits_of()),
        route=gateway.route_key(settled, all_routes, gateway.extras_of()),
    )
    hashes = frozenset(prompt.request_sha256 for prompt in asked)

    def recorded(path: Path, *, include_errors: bool = False) -> frozenset[str]:
        return hashes

    monkeypatch.setattr(records, "done", recorded)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = runner.invoke(app, ["run", "--alias", f"{settled},local-small-a", "--limit", "2"])
    assert "ANTHROPIC_API_KEY" not in unstyled(result.output), (
        "the one alias with no work left must not hold up the laptop, which needs no key"
    )
    assert result.exit_code == 1
    assert "Add --yes to run it" in unstyled(result.output)


# The retest summary is written per template, and only `plain` may write the file the handover
# reads. Project 03 sizes a release gate from that file, so a second administration of another
# template landing on it would replace the noise floor with one measured under a prompt nobody
# gates on. Added 2026-09-18 with the second administration that made the collision possible.


def test_only_the_plain_template_writes_the_file_the_handover_reads() -> None:
    from mselect import cli

    canonical = cli._retest_summary_path("v1", "plain")
    assert canonical.name == "own-run-retest-v1.json"
    for template in ("letter_only", "brief_reasoning"):
        other = cli._retest_summary_path("v1", template)
        assert other != canonical
        assert template in other.name
    # The handover reads the canonical name and nothing else.
    from mselect import handover

    assert handover.reliability().worst_hosted_flip_rate is not None


def test_a_missing_template_summary_is_missing_rather_than_the_plain_one() -> None:
    """No fallback, on purpose: substituting `plain` is the defect the parameter exists to fix."""
    from mselect import cli

    assert cli._measured_flip_rates("v1", "no-such-template") == {}
    assert cli._measured_flip_rates("v1", "plain")


def test_every_template_is_measured_separately_and_they_disagree() -> None:
    """If the three templates had the same flip rate, charging them separately would be pointless.

    They do not. `local-mid-a` flips 15 times more often when asked to reason than when asked for
    an answer, and `together-open-a` flips 5 times less often under `letter_only` than under
    `plain`, so the correction goes in opposite directions for different models.
    """
    from mselect import cli

    rates = {
        t: cli._measured_flip_rates("v1", t) for t in ("plain", "letter_only", "brief_reasoning")
    }
    assert all(rates.values()), "all three templates have a second administration"
    assert rates["brief_reasoning"]["local-mid-a"] > 10 * rates["plain"]["local-mid-a"]
    assert rates["plain"]["together-open-a"] > 4 * rates["letter_only"]["together-open-a"]
