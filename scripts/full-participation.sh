#!/usr/bin/env bash
#
# Every call one panel member has to make to be comparable with the other eleven.
#
# This exists because "run the suite" is not enough to put a model in the tables. A model that
# has the 3,000-item suite and none of the experiment arms appears in the transfer tau and is
# missing from findings 6, 7 and 8, and the arms are not interchangeable: the position-bias
# rotations must be the same 300 items, the retest must be the same 500 asked twice, and the
# framing arms must be the same 300 under the other two templates. The counts below are what
# `local-small-a` and `local-small-b` were administered, and they are why `local-mid-a` could be
# put beside them on 2026-09-17 without an asterisk.
#
# Resumable and free to repeat. Every call is keyed by request content hash, so re-running this
# after an interruption asks only what is outstanding and costs nothing for what is not. That is
# not a nicety: the run this was written for was interrupted four times, three by a session
# restart and once by the laptop flattening its battery, and lost nothing.
#
# Usage:   scripts/full-participation.sh [alias]            (default: local-mid-a)
#
# On Windows, launch it detached rather than from a tool that owns the process, or the parent
# dying takes a nineteen-hour run with it:
#
#   Start-Process -FilePath "C:\Program Files\Git\bin\bash.exe" `
#     -ArgumentList "scripts/full-participation.sh" -WindowStyle Hidden
#
set -u
cd "$(dirname "$0")/.." || exit 1

A="${1:-local-mid-a}"
M=".venv/Scripts/mselect.exe"
[ -x "$M" ] || M="mselect"
LOG="out/$A.log"
mkdir -p out

# Appends rather than truncates. Three restarts in, a log that starts again each time is a log
# that has thrown away the evidence for why it restarted.
step () {
  echo "" >> "$LOG"
  echo "=== $* | started $(date -u '+%Y-%m-%d %H:%M:%SZ') ===" >> "$LOG"
  "$M" run --alias "$A" --chunk 50 --yes "$@" >> "$LOG" 2>&1
  echo "--- exit $? at $(date -u '+%Y-%m-%d %H:%M:%SZ') ---" >> "$LOG"
}

echo "" >> "$LOG"
echo "$A full participation, started $(date -u '+%Y-%m-%d %H:%M:%SZ')" >> "$LOG"

step --template plain --rotation 0                           # 3,000  the suite itself
step --template plain --rotation 1 --limit 300               #   300  position bias, finding 7
step --template plain --rotation 2 --limit 300               #   300
step --template plain --rotation 3 --limit 300               #   300
step --template plain --rotation 0 --repeat 2 --limit 500    #   500  test-retest, finding 6
step --template letter_only --limit 300                      #   300  framing, finding 8
step --template brief_reasoning --limit 300                  #   300  framing

echo "" >> "$LOG"
echo "=== all arms finished $(date -u '+%Y-%m-%d %H:%M:%SZ') ===" >> "$LOG"
