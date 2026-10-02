"""Pushing this project's ledger to the portfolio dashboard after a run that spent.

The ledger the gateway writes, `out/own-run-ledger.sqlite`, is the record of every vendor call
this project has made. The portfolio dashboard at gateway.peterparker.ca shows it, and until
this existed it showed it as of project 04's last push from the laptop rather than as of this
project's last run. Project 04's `docs/central.md` ("From another project's own runs") is the
protocol: run a current `boundary` as a tool beside this project's own pin, so pushing needs no
upgrade of the library the runner calls through. That is why this shells out to `uvx` rather
than importing anything: the pin in `pyproject.toml` stays where it is.

Three rules decide the shape of this file:

* **The source name is fixed.** The first push, on 2026-09-30, named the source
  `model-selection-tenth-cost:out/own-run-ledger.sqlite` (71,725 rows), and the same name adds
  to the same source. A name built from a path that moved would start a second source and
  double-count every row on the dashboard, so it is a constant rather than derived.
* **The key never leaves the environment.** It is read from `BOUNDARY_INGEST_KEY`, or from that
  one line of `.env` when the variable is not set, and handed to the tool in its environment.
  It is never an argument, so it is never in a process listing, and anything the tool prints
  is scrubbed of it before it is echoed.
* **A failed push is a warning.** The run it follows has already spent the money and written
  the rows, and the push can be repeated at any time with `mselect push-ledger`. Failing the
  run would report a failure the vendor calls did not have.
"""

from __future__ import annotations

import functools
import os
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Final

from mselect import paths

URL: Final = "https://gateway.peterparker.ca"
# The released tool, not the pin this project calls through. `push` reads every ledger schema
# from v1 on (04 docs/central.md), so it reads this ledger whichever pin last wrote it, and the
# two move independently.
TOOL: Final = "git+https://github.com/Peter-A-P/compliant-ai-gateway@v0.34.1"
PROJECT: Final = "model-selection-tenth-cost"
SOURCE: Final = f"{PROJECT}:out/own-run-ledger.sqlite"
KEY_ENV: Final = "BOUNDARY_INGEST_KEY"
# The first `uvx` run builds the tool from the tag, which takes longer than any push after it.
TIMEOUT_S: Final = 600.0


def ledger_path() -> Path:
    """Where the gateway writes this project's ledger (`mselect/config/boundary.yaml`)."""
    return paths.OUT / "own-run-ledger.sqlite"


def ingest_key(env_file: Path | None = None) -> str:
    """The ingest key from the environment, or from `.env` when the environment lacks it.

    Only this one variable is read from the file, and nothing is put into `os.environ`: the
    vendor keys keep coming from the environment alone, as they always have.
    """
    key = os.environ.get(KEY_ENV, "").strip()
    if key:
        return key
    path = env_file if env_file is not None else paths.ROOT / ".env"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    for line in lines:
        name, sep, value = line.strip().removeprefix("export ").partition("=")
        if sep and name.strip() == KEY_ENV:
            return value.strip().strip("'\"")
    return ""


def command(ledger: Path) -> list[str]:
    """The push, exactly as 04 documents it. The key is not in it, and must never be."""
    return [
        "uvx",
        "--from",
        TOOL,
        "boundary",
        "ledger",
        "push",
        "--url",
        URL,
        "--ledger",
        str(ledger),
        "--source",
        SOURCE,
        "--project",
        PROJECT,
    ]


def push(
    *,
    say: Callable[[str], None] = print,
    ledger: Path | None = None,
    env_file: Path | None = None,
) -> bool:
    """Push the ledger. True when the tool said it succeeded; every failure is a warning."""
    ledger = ledger if ledger is not None else ledger_path()
    if not ledger.is_file():
        say(f"warning: ledger not pushed: there is no ledger at {ledger}")
        return False
    key = ingest_key(env_file)
    if not key:
        say(f"warning: ledger not pushed: {KEY_ENV} is not set and is not in .env")
        return False
    if shutil.which("uvx") is None:
        say("warning: ledger not pushed: uvx is not on PATH (it comes with uv)")
        return False

    def scrub(text: str) -> str:
        return text.replace(key, "[redacted]")

    say(f"pushing {ledger.name} to {URL} as source {SOURCE}")
    try:
        done = subprocess.run(
            command(ledger),
            env={**os.environ, KEY_ENV: key},
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        say(f"warning: ledger not pushed: {scrub(str(exc))}")
        return False
    for stream in (done.stdout, done.stderr):
        for line in scrub(stream).splitlines():
            say(f"  {line}")
    if done.returncode != 0:
        say(
            f"warning: ledger push exited {done.returncode}. The run itself is unaffected; "
            "`mselect push-ledger` tries again."
        )
        return False
    return True


def _signature(ledger: Path) -> tuple[tuple[int, int], ...]:
    """Size and modification time of the ledger and its write-ahead log, for change detection."""
    marks: list[tuple[int, int]] = []
    for path in (ledger, ledger.with_name(ledger.name + "-wal")):
        try:
            stat = path.stat()
        except OSError:
            marks.append((-1, -1))
            continue
        marks.append((stat.st_size, stat.st_mtime_ns))
    return tuple(marks)


def push_after[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    """Push the ledger when the wrapped command ends, if it wrote to the ledger.

    "Wrote to the ledger" is the test because it is the one fact that means there is something
    new to push, and it is true of exactly the runs that called through the gateway: a plan
    printed without `--yes`, a resume with nothing outstanding and a refused paid alias all leave
    the file alone and push nothing. It runs on the way out of an exception too, because a run
    stopped by a spend cap has still spent up to it, and those rows belong on the dashboard.
    """

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        before = _signature(ledger_path())
        try:
            return func(*args, **kwargs)
        finally:
            if _signature(ledger_path()) != before:
                print()
                push()

    return wrapper
