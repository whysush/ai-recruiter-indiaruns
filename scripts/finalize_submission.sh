#!/usr/bin/env bash
# Produce the correctly-named submission file and re-validate it.
#
# Usage:
#   bash scripts/finalize_submission.sh <participant_id>
# Example:
#   bash scripts/finalize_submission.sh team_redrob
#   -> creates team_redrob.csv (copy of submission.csv) and runs the official validator.
#
# The challenge requires the CSV to be named after your REGISTERED participant ID
# (spec section 2). This script just renames + validates; it does not re-rank.
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: bash scripts/finalize_submission.sh <participant_id>   (e.g. team_redrob)"
  exit 1
fi

PID="$1"
SRC="submission.csv"
DST="${PID}.csv"

if [ ! -f "$SRC" ]; then
  echo "ERROR: $SRC not found. Generate it first:"
  echo "  python scripts/run_submission.py --candidates data/candidates.jsonl --out submission.csv"
  exit 1
fi

cp "$SRC" "$DST"
echo "Wrote $DST"
python3 scripts/validate_submission.py "$DST"
