"""Set aside every own-run record for the named aliases, so `mselect run` asks them again.

Written 2026-11-17 for the defect in `mselect run` that sent no request extras from 2026-09-12
to that day. The request hash counted the extras and the request body did not carry them, so
the records of the two models that have extras say they were asked one way and were asked
another. A resume keys on the request hash, so it would call those records done; the only way
to ask again is to take them out of the live files.

Nothing is deleted. Each live file is first copied whole into `out/<version>/superseded-<date>/`,
then rewritten without the named aliases' rows, so the original state is one copy away and the
old answers stay readable beside the new ones.

    python scripts/supersede.py google-frontier together-open-b
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def supersede(aliases: set[str], version: str = "v1") -> None:
    out = ROOT / "out" / version
    keep_dir = out / f"superseded-{date.today().isoformat()}"
    keep_dir.mkdir(exist_ok=True)
    for path in sorted(out.glob("own-run-*.jsonl")):
        backup = keep_dir / path.name
        if not backup.exists():
            shutil.copy2(path, backup)
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        kept = [line for line in lines if json.loads(line).get("alias") not in aliases]
        removed = len(lines) - len(kept)
        tmp = path.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(kept), encoding="utf-8", newline="")
        tmp.replace(path)
        print(f"{path.name}: {removed:,} rows set aside, {len(kept):,} kept")


if __name__ == "__main__":
    names = set(sys.argv[1:])
    if not names:
        raise SystemExit("name at least one alias")
    supersede(names)
