"""Where things live. One place, so a run from anywhere writes to the same directories."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

# The repository root when running from a checkout; overridable so a run can point at another
# working directory (CI, a scratch copy) without editing code.
ROOT: Final = Path(os.environ.get("MSELECT_ROOT", Path(__file__).resolve().parent.parent))

RAW: Final = ROOT / "data" / "raw"  # fetched public bytes, gitignored
HELM_CACHE: Final = RAW / "helm"
OLLM_CACHE: Final = RAW / "ollm"  # Open LLM Leaderboard projected columns
BANK: Final = ROOT / "mselect" / "bank"  # frozen, versioned, committed
OUT: Final = ROOT / "out"  # fits, simulations, figures; gitignored


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def out_for(version: str) -> Path:
    """Run artefacts for one bank version.

    Scoped by version because there is more than one bank now, and a fit of v2 that overwrote
    v1's diagnostics would make the README describe one bank with another bank's numbers.
    """
    return ensure(OUT / version)
