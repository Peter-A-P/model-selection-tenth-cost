"""No credential is ever committed.

CLAUDE.md: never commit credentials or `.env`; vendor keys come from the environment only. That
is a rule, and a rule without a test is a hope. This one scans every file git is tracking for
the shapes real credentials take, so a key that lands in the working tree by accident fails the
build instead of reaching a remote.

It exists because on 2026-09-11 a Hugging Face token, pasted into a stray "New Text Document.txt"
in the repository root, was swept into a commit by `git add -A`. It was caught and removed
before any push. This test is what would have caught it first.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Built from parts so that this file does not match its own patterns.
PREFIXES: dict[str, str] = {
    "Hugging Face token": "hf" + "_",
    "OpenAI key": "sk" + "-",
    "Anthropic key": "sk-ant" + "-",
    "GitHub personal access token": "ghp" + "_",
    "GitHub fine-grained token": "github" + "_pat_",
    "Google API key": "AIza" + "Sy",
    "AWS access key id": "AKIA" + "IOSFODNN",
    "Slack token": "xox" + "b-",
}
TAIL = r"[A-Za-z0-9_\-]{16,}"
SKIP_SUFFIXES = {".parquet", ".png", ".gz", ".ico", ".woff2", ".pyc"}


def tracked_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return [ROOT / name for name in out.stdout.split("\0") if name]


def test_no_tracked_file_contains_anything_shaped_like_a_credential() -> None:
    patterns = {name: re.compile(re.escape(prefix) + TAIL) for name, prefix in PREFIXES.items()}
    offenders: list[str] = []
    for path in tracked_files():
        if path.suffix in SKIP_SUFFIXES or not path.is_file():
            continue
        if path.name == Path(__file__).name:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for name, pattern in patterns.items():
            found = pattern.search(text)
            if found:
                offenders.append(f"{path.relative_to(ROOT)}: {name} at offset {found.start()}")
    assert not offenders, "credential-shaped strings in tracked files: " + "; ".join(offenders)


def test_no_env_file_is_tracked() -> None:
    names = {path.name for path in tracked_files()}
    assert not [name for name in names if name == ".env" or name.startswith(".env.")]


def test_the_scanner_would_actually_catch_one(tmp_path: Path) -> None:
    """The test above is only worth having if its pattern matches a real token shape."""
    pattern = re.compile(re.escape(PREFIXES["Hugging Face token"]) + TAIL)
    assert pattern.search("HF_TOKEN=" + "hf" + "_" + "A" * 34)
    assert pattern.search("key: " + "hf" + "_" + "aB3" * 12)
    assert not pattern.search("the hf_ prefix on its own is not a token")
    del tmp_path


def test_no_benchmark_text_is_committed() -> None:
    """The repository claims to carry no benchmark question text. This is that claim, enforced.

    An earlier bank stored a 160-character excerpt of each question as evidence for the
    broken-item report. The report never rendered it, so it was 2.3 MB of other people's text
    under seven licences, one of them per-task, serving nothing. It was removed in v0.3.0 and
    this test is what stops it coming back.

    The check is an allowlist of column names rather than a length limit, because a length limit
    is a guess about what text looks like. Every string column a bank carries has to be named
    here, which means adding one is a decision somebody makes on purpose.
    """
    import polars as pl

    from mselect import paths
    from mselect.data import bank as bank_io

    allowed = {
        "item_id",  # a content hash, 16 hex characters
        "benchmark",  # "mmlu", "bbh"
        "kind",  # "multiple_choice", "free_response"
        "scenario_key",  # the source's own run or task name, a configuration string
        "instance_id",  # the source's own id for the item, which is how to look it up
        "source_project",
        "source_release",
    }
    for version in bank_io.available():
        items = pl.read_parquet(paths.BANK / version / "items.parquet")
        text_columns = {name for name, dtype in items.schema.items() if dtype == pl.Utf8}
        unexpected = text_columns - allowed
        assert not unexpected, (
            f"bank {version} carries unlisted text columns {unexpected}; if one of them holds "
            f"benchmark content it must not be committed, and if it does not, add it above"
        )
