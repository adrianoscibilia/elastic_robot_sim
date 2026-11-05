#!/usr/bin/env bash
# RR_23/RR_24 real P0/P1/P2 campaign: shared helpers for scripts/real_campaign_{iiwa,ur10}.sh.
# Sourced, not executed. Needs a shell set up with `source workspace_setup.sh --real`.
#
# One step = one --run-id = several stages run in order, each logged to the console and to
# $ERD_DATA_ROOT/campaign_logs/<date>/<run_id>.log. The script stops at the first stage that exits
# non-zero, and after any `quicklook` whose quicklook.json verdict is "stop" (the quicklook stage
# itself always exits 0, RR_24 S1). Built on data/rr23_session/run.sh, which ran the same stages
# green on mock/emulator/URSim (RR_24 S6).

set -uo pipefail
export PYTHONUNBUFFERED=1

: "${ERD_DATA_ROOT:?source workspace_setup.sh --real first}"
: "${ERD_LAB:?source workspace_setup.sh --real first}"

CAMPAIGN_DATE=$(date +%Y%m%d)
LOGDIR="$ERD_DATA_ROOT/campaign_logs/$CAMPAIGN_DATE"
mkdir -p "$LOGDIR"

FROM=1          # --from N: start at step N
PLAN_ONLY=0     # --plan-only: only the plan stage of every step (Q = 10, no --previous/--modal-previous)
CONFIRM=0       # --confirm: keep the per-trajectory confirmation prompt (default: --no-confirm, as on 2026-10-14/21)

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --from) FROM=$2; shift 2 ;;
      --plan-only) PLAN_ONLY=1; shift ;;
      --confirm) CONFIRM=1; shift ;;
      -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
      *) echo "unknown option $1"; exit 2 ;;
    esac
  done
}

run_dir_of() {  # newest run folder with this id
  ls -d "$ERD_DATA_ROOT"/"$ROBOT_DIR"/*/"$1" 2>/dev/null | sort | tail -1
}

check_quicklook() {
  local id=$1 dir verdict
  dir=$(run_dir_of "$id")
  if [[ -z "$dir" || ! -f "$dir/quicklook.json" ]]; then
    echo "[$id] quicklook.json not found under $ERD_DATA_ROOT/$ROBOT_DIR/*/$id"; return 4
  fi
  verdict=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('verdict'))" "$dir/quicklook.json")
  echo "[$id] quick-look verdict: $verdict  ($dir/quicklook.md)"
  [[ "$verdict" == "stop" ]] && return 3
  return 0
}

# step N RECORDER CONFIG RUN_ID "<stage + options>" ...
step() {
  local n=$1 rec=$2 cfg=$3 id=$4; shift 4
  [[ $n -lt $FROM ]] && { echo "== step $n ($id): skipped (--from $FROM)"; return 0; }
  echo; echo "================ step $n: $id  ($(basename "$cfg"))  $(date +%H:%M:%S) ================"
  local st rc conf
  conf=$([[ $CONFIRM -eq 1 ]] && echo "" || echo "--no-confirm")
  for st in "$@"; do
    if [[ $PLAN_ONLY -eq 1 ]]; then
      [[ "$st" == plan* ]] || continue
      st=$(echo "$st" | sed -E 's/ --modal-previous [^ ]+//')
    fi
    echo "---- $id: $st"
    # shellcheck disable=SC2086
    ros2 run erd_recording "$rec" --config "$cfg" $st --run-id "$id" $([[ "$st" == plan* ]] || echo "$conf") \
      2>&1 | tee -a "$LOGDIR/$id.log"
    rc=${PIPESTATUS[0]}
    if [[ $rc -ne 0 ]]; then
      echo; echo "STOPPED at step $n ($id), stage '$st': exit $rc. Log: $LOGDIR/$id.log"
      echo "Do not re-run this id: tell the architect, then resume from the next step with --from $((n + 1)) or re-run step $n under a new id."
      exit $rc
    fi
    if [[ "$st" == quicklook* ]]; then
      if ! check_quicklook "$id"; then
        echo; echo "STOPPED by the quick-look of step $n ($id). Read $(run_dir_of "$id")/quicklook.md before anything else."
        exit 3
      fi
    fi
  done
}

finish() {
  echo; echo "================ campaign done $(date +%H:%M:%S) ================"
  [[ $PLAN_ONLY -eq 1 ]] && return 0
  echo "Hand-off entries (dry; add --write yourself for the files you accept):"
  local entry id split
  for entry in "$@"; do
    id=${entry%%:*}; split=${entry##*:}
    echo "  ros2 run erd_recording $RECORDER --config <its lab file> handoff --run-id $id --split $split"
  done
}
