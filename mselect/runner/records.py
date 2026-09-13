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
import re
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Final

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


# Statuses that mean the vendor was busy, slow or briefly broken, as they appear in an error
# string rather than as a number the adapter still has. Used only for records written before
# `retryable` existed; see `_legacy_retryable` for why that is not the same as parsing errors.
_LEGACY_STATUS: Final = re.compile(r"\breturned (?:408|409|429|500|502|503|504|529)\b")
_LEGACY_TRANSPORT: Final = re.compile(r"timeout|connect_?error|read_?error|not ready", re.I)


def _legacy_retryable(record: Mapping[str, object]) -> bool:
    """Whether a failure recorded before `retryable` existed is worth asking again.

    Reading a verdict back out of an error string is precisely what `gateway._retryable`
    refuses to do, and for the right reason: the adapter holds the HTTP status and the
    exception type, and a sentence is a lossy copy of both. This is not that. A record written
    on 2026-09-12 carries no verdict at all, and the sentence is the only thing left of it.

    It exists because the panel run was started at 18:39 on 2026-09-12 and `retryable` was
    added at 21:29, so the process kept writing verdict-less records for another nine hours,
    including the 1,910 that matter. Anything written since carries the field, and the field
    wins; this is consulted only when it is absent.
    """
    error = record.get("error")
    if not isinstance(error, str):
        return False
    return bool(_LEGACY_STATUS.search(error) or _LEGACY_TRANSPORT.search(error))


def done(path: Path, *, include_errors: bool = False) -> frozenset[str]:
    """The request hashes already recorded, for `administer` to skip.

    Hashes and not cell names. A cell is a model, an item, a template and a rotation, which
    is what to call a result; it is not what to key "have we asked this on", because it says
    nothing about what was asked. A request hash covers the prompt, the token budget, the
    temperature, the vendor fields and the model the alias resolved to, so an edited run
    re-asks what changed and inherits only what did not.

    A failed call is done unless the failure was worth repeating, which the record says.
    **Amended 2026-09-12, against the live run.** This used to re-ask every failure, on the
    assumption that an error is usually a timeout or a rate limit. The first four models of the
    panel produced 33 failures and not one was: 25 were models reasoning past their token
    budget, 4 were refusals, and the rest a model stopping early. Every one of those repeats
    identically at identical cost, and 25 truncations at 1024 output tokens each is about
    US$0.16 of nothing per resume, with five experiment arms still to run.

    Nothing is lost by settling them. The request hash covers everything that determines the
    reply, so raising the token budget asks all 25 again by itself, because that makes them
    different requests.

    **Amended again 2026-09-13, against the same run.** The paragraph above is right about
    the failures it was written from and wrong about the ones that came after it. Defaulting a
    missing `retryable` to False reads "no verdict" as "settled", and the run went on to write
    1,910 verdict-less records: Google's daily free-tier quota for `gemini-3.8-flash` ran out
    at 00:23 and refused every remaining call with a 429. Those are the most repeatable failure
    there is, and settling them would have left that model measured on 1,088 items of 3,000
    with nothing in the run's own output to say so. A missing verdict is now read as missing.

    An unparsed reply is done for the same reason it always was: asking the same question the
    same way will not read any better, and the unparsed share is a number this project reports
    rather than retries away.
    """
    out: set[str] = set()
    for record in read(path):
        request = record.get("request_sha256")
        if not isinstance(request, str):
            continue
        failed = record.get("error") is not None
        if failed and not include_errors:
            verdict = record.get("retryable")
            if _legacy_retryable(record) if verdict is None else bool(verdict):
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
