"""The push to the portfolio dashboard, tested without pushing.

What matters is the three promises in `central.py`: the key reaches the tool only through its
environment and is never printed, the source name is the one the first push used, and a push
that fails is a warning rather than a failed run. A real push is checked by hand, against the
dashboard's own count; nothing here talks to the network.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from mselect import paths
from mselect.runner import central

# Shaped like an ingest key without matching the credential scanner's patterns.
KEY = "bnd" + "_test0000"


class FakeRun:
    """Stands in for `subprocess.run` and remembers how it was called."""

    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.calls: list[tuple[list[str], dict[str, str]]] = []
        self.result = subprocess.CompletedProcess([], returncode, stdout, stderr)

    def __call__(self, args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append((list(args), dict(kwargs["env"])))
        return self.result


@pytest.fixture
def ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(paths, "OUT", tmp_path)
    monkeypatch.setattr(paths, "ROOT", tmp_path)
    monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/uvx")
    monkeypatch.delenv(central.KEY_ENV, raising=False)
    path = tmp_path / "own-run-ledger.sqlite"
    path.write_bytes(b"rows")
    return path


def test_the_source_is_the_name_the_first_push_used() -> None:
    """A different name would start a second source and count every row twice."""
    assert central.SOURCE == "model-selection-tenth-cost:out/own-run-ledger.sqlite"
    args = central.command(Path("out/own-run-ledger.sqlite"))
    assert args[args.index("--source") + 1] == central.SOURCE
    assert args[args.index("--project") + 1] == "model-selection-tenth-cost"
    assert args[args.index("--url") + 1] == "https://gateway.peterparker.ca"


def test_the_key_comes_from_the_environment_before_the_file(
    ledger: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (ledger.parent / ".env").write_text(f"{central.KEY_ENV}=from-file\n", encoding="utf-8")
    monkeypatch.setenv(central.KEY_ENV, KEY)
    assert central.ingest_key() == KEY


def test_the_key_is_read_from_env_file_when_the_environment_lacks_it(ledger: Path) -> None:
    (ledger.parent / ".env").write_text(
        f'HF_TOKEN=\nexport {central.KEY_ENV}="{KEY}"\nOTHER=x\n', encoding="utf-8"
    )
    assert central.ingest_key() == KEY


def test_the_key_goes_in_the_environment_and_never_in_the_arguments_or_the_output(
    ledger: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(central.KEY_ENV, KEY)
    fake = FakeRun(stdout=f"authenticated with {KEY}\nthe central ledger holds 5 of 5\n")
    monkeypatch.setattr(subprocess, "run", fake)
    said: list[str] = []
    assert central.push(say=said.append)
    args, env = fake.calls[0]
    assert env[central.KEY_ENV] == KEY
    assert not [a for a in args if KEY in a]
    assert not [line for line in said if KEY in line]
    assert any("holds 5 of 5" in line for line in said)


@pytest.mark.parametrize(
    "outcome",
    ["no key", "no ledger", "tool fails", "tool cannot start"],
)
def test_a_failed_push_is_a_warning_and_never_raises(
    ledger: Path, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    if outcome != "no key":
        monkeypatch.setenv(central.KEY_ENV, KEY)
    if outcome == "no ledger":
        ledger.unlink()
    fake = FakeRun(returncode=2, stderr="401 unauthorised")
    if outcome == "tool cannot start":

        def broken(*_: Any, **__: Any) -> subprocess.CompletedProcess[str]:
            raise subprocess.TimeoutExpired("uvx", 600)

        monkeypatch.setattr(subprocess, "run", broken)
    else:
        monkeypatch.setattr(subprocess, "run", fake)
    said: list[str] = []
    assert not central.push(say=said.append)
    assert any(line.startswith("warning:") for line in said)


def test_a_command_that_wrote_to_the_ledger_pushes_even_when_it_raises(
    ledger: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run stopped by a spend cap has still spent up to it."""
    pushed: list[bool] = []

    def recorded(**_: Any) -> bool:
        pushed.append(True)
        return True

    monkeypatch.setattr(central, "push", recorded)

    @central.push_after
    def spends() -> None:
        ledger.write_bytes(b"rows and more rows")
        raise RuntimeError("spend cap")

    with pytest.raises(RuntimeError):
        spends()
    assert pushed == [True]


def test_a_command_that_left_the_ledger_alone_pushes_nothing(
    ledger: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A plan printed without --yes sent nothing, so there is nothing new to push."""
    pushed: list[bool] = []

    def recorded(**_: Any) -> bool:
        pushed.append(True)
        return True

    monkeypatch.setattr(central, "push", recorded)

    @central.push_after
    def plans() -> int:
        return 1

    assert plans() == 1
    assert pushed == []
