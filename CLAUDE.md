# Working notes for Claude Code

This repository is Model Selection at a Tenth of the Cost: item response theory applied
to language-model benchmarks, with computerised adaptive testing that reproduces a full
benchmark ranking at a fraction of the calls. The build plan is [PLAN.md](PLAN.md).

## Read first

- [README.md](README.md): what this is and the current result table.
- [PLAN.md](PLAN.md): data, methods, package design, week-by-week schedule, cost. Do not
  deviate from it silently; if something turns out wrong, change the plan in the same
  commit as the code and say why.

## Engineering standard

- Python 3.13, managed with `uv`. Typed throughout; `mypy --strict` and `ruff` clean in CI.
- Tests that fail meaningfully: parameter recovery on simulated matrices, adaptive
  estimator convergence, planted local dependence and DIF detected, answer parsers
  against adversarial fixtures.
- `pyproject.toml` with pinned major versions and a comment saying why for each pin.
- Docs ship in the same commit as the change.
- Never commit credentials or `.env`. Vendor keys come from the environment only.

## Rules specific to this repository

- **Every reported number carries an interval.** A bare Kendall's tau or accuracy is a bug.
- **The item bank is versioned and content-hashed.** Parameters are stored with the bank
  version and fit version that produced them.
- **Cache every vendor response** by request content hash. Reruns must cost nothing. (The
  opposite rule applies in the release-gate repository's drift runs; do not confuse them.)
- **Anthropic calls use the Message Batches endpoint** unless latency matters, which it
  does not here.
- **Answer-only output format** for the bank runs; reasoning-style prompts belong to the
  framing experiment only.
- **Report local dependence and dimensionality**, never hide them. They are the
  interesting section.
- **Plain punctuation** in everything written here: no em-dashes or other typographic
  dashes, straight quotes only.

## What goes in the README

The README opens with the one-liner, the results table, and the honest limitation, before
any installation instructions. `mselect report` regenerates the table; do not hand-edit
it. `docs/rejected.md` records one approach tried and rejected, with evidence.
