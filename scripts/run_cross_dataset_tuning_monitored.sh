#!/usr/bin/env bash
set -uo pipefail

active=/tmp/cross_dataset_tuning.active
exit_file=/tmp/cross_dataset_tuning.exit
log=/tmp/cross_dataset_tuning.log

if [[ -e "$active" ]]; then
  echo "cross-dataset tuning monitor already active" >&2
  exit 2
fi
rm -f "$exit_file"
touch "$active"
trap 'rm -f "$active"' EXIT

bash scripts/run_cross_dataset_tuning.sh >"$log" 2>&1
status=$?
printf '%s\n' "$status" >"$exit_file"
exit "$status"
