#!/usr/bin/env bash
#
# The second administration of the two framing templates, 300 items x 12 models x 2 templates.
#
# Findings 7 and 8 both subtract each model's own instability before reporting an effect, and the
# flip rate they subtract was measured under the answer-only template alone. A template with more
# room to wander is therefore charged too little noise, which makes every net figure in finding 8
# an upper bound rather than an estimate. This arm measures the missing flip rates.
#
# `--repeat 2` is what makes it a measurement rather than a reread: it uses its own cache
# namespace, so every call is actually sent instead of being answered out of administration 1's
# records. That is also why this arm costs money where a rerun of anything else costs nothing.
#
# Usage:   scripts/framing-second-administration.sh [alias,alias,...]
#
# With no argument every full-suite panel member runs, which needs all four vendor keys in the
# environment. Pass the three laptop aliases to run the free half on its own.
#
set -u
cd "$(dirname "$0")/.." || exit 1

M=".venv/Scripts/mselect.exe"
[ -x "$M" ] || M="mselect"
LOG="out/framing-r2.log"
mkdir -p out

ALIAS_ARG=()
[ "$#" -gt 0 ] && ALIAS_ARG=(--alias "$1")

step () {
  echo "" >> "$LOG"
  echo "=== $* | started $(date -u '+%Y-%m-%d %H:%M:%SZ') ===" >> "$LOG"
  "$M" run "${ALIAS_ARG[@]}" --chunk 50 --yes --limit 300 --repeat 2 "$@" >> "$LOG" 2>&1
  echo "--- exit $? at $(date -u '+%Y-%m-%d %H:%M:%SZ') ---" >> "$LOG"
}

echo "" >> "$LOG"
echo "framing second administration, started $(date -u '+%Y-%m-%d %H:%M:%SZ')" >> "$LOG"

step --template letter_only
step --template brief_reasoning

echo "" >> "$LOG"
echo "=== both arms finished $(date -u '+%Y-%m-%d %H:%M:%SZ') ===" >> "$LOG"
