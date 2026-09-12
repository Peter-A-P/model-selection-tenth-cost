"""The own-run record file: append one line per administration, resume from what is there.

One JSON object per line, appended and never rewritten, so a run that is killed loses at
most the call it was in the middle of. The file is the resume state: `done` reads the cells
already recorded, and `administer` skips them, which is what makes "run it again" the answer
to an interrupted run rather than a second bill.

Records live under `out/`, which is gitignored. What this project commits is the 0 or 1 per
cell, not the replies: a reply can quote the question back, and benchmark item text is not
republished here (PLAN.md section 13.5).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from mselect import paths
from mselect.runner.administer import Administration


def records_path(bank_version: str, run_id: str) -> Path:
    """Where one run's records live. Scoped by bank version for the same reason fits are:
    a run against one bank must not be read as a run against another."""
    directory = paths.ensure(paths.out_for(bank_version) / "own-run")
    return directory / f"{run_id}.jsonl"


def append(path: Path, records: Iterable[Administration]) -> int:
    """Add records to the file, creating it if needed. Returns how many were written.

    Flushed per call rather than per line: a killed process loses the batch it was writing,
    and the cells in it are simply asked again on the next run, because they were never
    recorded as done.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_json(), ensure_ascii=False) + "\n")
            written += 1
    return written


def read(path: Path) -> Iterator[dict[str, object]]:
    """Every record in the file, skipping a trailing partial line.

    A partial last line is the normal shape of a file whose process was killed mid-write. It
    is skipped rather than raised on, because the cell it belongs to is simply not done and
    will be asked again.
    """
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                yield record


def done(path: Path, *, include_errors: bool = False) -> frozenset[str]:
    """The request hashes already recorded, for `administer` to skip.

    Hashes and not cell names. A cell is a model, an item, a template and a rotation, which
    is what to call a result; it is not what to key "have we asked this on", because it says
    nothing about what was asked. A request hash covers the prompt, the token budget, the
    temperature, the vendor fields and the model the alias resolved to, so an edited run
    re-asks what changed and inherits only what did not.

    A call that failed is not done by default: an error is usually the network or a rate
    limit, and the point of resuming is to pick those up. An unparsed reply is done, because
    asking the same question the same way again will not read any better, and the unparsed
    share is a number this project reports rather than retries away.
    """
    out: set[str] = set()
    for record in read(path):
        request = record.get("request_sha256")
        if not isinstance(request, str):
            continue
        if not include_errors and record.get("error") is not None:
            continue
        out.add(request)
    return frozenset(out)


def spend_usd(path: Path) -> float:
    """What this run has cost so far, from the records. The ledger is the authority; this is
    the quick answer without opening it."""
    total = 0.0
    for record in read(path):
        cost = record.get("cost_usd")
        if isinstance(cost, int | float):
            total += float(cost)
    return total
