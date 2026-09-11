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
