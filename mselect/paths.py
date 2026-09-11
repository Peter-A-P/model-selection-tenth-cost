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
BANK: Final = ROOT / "mselect" / "bank"  # frozen, versioned, committed
OUT: Final = ROOT / "out"  # fits, simulations, figures; gitignored


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
