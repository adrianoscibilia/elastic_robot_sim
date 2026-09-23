#!/usr/bin/env bash
# Unattended round-5 run: matched controller datasets -> Optuna -> training.
#
#   bash scripts/weekend_run.sh preflight   # ~15 min, proves every stage runs
#   bash scripts/weekend_run.sh full        # the real thing, days
#
# Resumable: every stage writes runs/<id>/<stage>.done and is skipped if present.
# Delete a marker to force that stage to re-run.  Everything is logged under
# runs/<id>/logs/ and summarised live in runs/<id>/STATUS.md.
set -uo pipefail

MODE="${1:-preflight}"
SIM_DIR="${SIM_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
NN_DIR="${NN_DIR:-$(cd "$SIM_DIR/../dynamic_model_nn" && pwd)}"
# The two repos have *separate* environments: the sim needs pinocchio/mujoco,
# dynamic_model_nn needs torch/optuna.  Resolve a launcher per repo rather than
# assuming a bare `python` exists (Ubuntu ships python3 only -> exit 127).
# Override with SIM_PY_CMD / NN_PY_CMD if neither guess is right.
pick_py() {
  local d="$1"
  [ -x "$d/.venv/bin/python" ] && { echo "$d/.venv/bin/python"; return; }
  if command -v uv >/dev/null 2>&1 && [ -f "$d/pyproject.toml" ]; then
    echo "uv run --project $d python"; return
  fi
  command -v python3 >/dev/null 2>&1 && { echo python3; return; }
  echo python
}
read -r -a SIM_PY <<< "${SIM_PY_CMD:-$(pick_py "$SIM_DIR")}"
read -r -a NN_PY  <<< "${NN_PY_CMD:-$(pick_py "$NN_DIR")}"

RUN_ID="${RUN_ID:-$MODE-$(date +%Y%m%d-%H%M)}"
RUN="$SIM_DIR/runs/$RUN_ID"
LOGS="$RUN/logs"; mkdir -p "$LOGS"
STATUS="$RUN/STATUS.md"

# Stop cleanly instead of half-finishing: nothing new starts past the deadline.
DEADLINE_HOURS="${DEADLINE_HOURS:-84}"
STARTED=$(date +%s)
MIN_FREE_GB="${MIN_FREE_GB:-40}"
# A hung stage is worse than a failed one when nobody is watching.
STAGE_TIMEOUT="${STAGE_TIMEOUT:-}"

PLATFORMS=(kuka_lbr_iiwa_14_r820_table_round5 ur10_table_round5)
MODES=(exact_ct pd velocity_pi)

if [ "$MODE" = preflight ]; then
  ROBOTS=2; TRAJ=1; QC_FLAGS=""; TRIALS=4; SCREEN_EPOCHS=2; REFINE_TOP=1; REFINE_EPOCHS=3
  STAGE_TIMEOUT="${STAGE_TIMEOUT:-20m}"
else
  ROBOTS="${ROBOTS:-20}"; TRAJ="${TRAJ:-6}"; QC_FLAGS="--full"; TRIALS=48
  STAGE_TIMEOUT="${STAGE_TIMEOUT:-18h}"
  SCREEN_EPOCHS=20; REFINE_TOP=5; REFINE_EPOCHS=300
fi

say() { printf '%s | %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$STATUS"; }

budget_left() {
  local used=$(( ($(date +%s) - STARTED) / 3600 ))
  [ "$used" -lt "$DEADLINE_HOURS" ]
}

disk_ok() {
  local free; free=$(df -BG --output=avail "$SIM_DIR" | tail -1 | tr -dc '0-9')
  [ "${free:-0}" -ge "$MIN_FREE_GB" ] || { say "STOP: only ${free}G free, need ${MIN_FREE_GB}G"; return 1; }
}

# stage <name> <logfile> <command...>
stage() {
  local name="$1" log="$2"; shift 2
  if [ -f "$RUN/$name.done" ]; then say "skip  $name (done)"; return 0; fi
  budget_left || { say "SKIP  $name -- past the ${DEADLINE_HOURS}h deadline"; return 0; }
  disk_ok    || return 1
  say "START $name"
  local runner=(); [ -n "$STAGE_TIMEOUT" ] && runner=(timeout --signal=INT --kill-after=120 "$STAGE_TIMEOUT")
  if "${runner[@]}" "$@" >>"$LOGS/$log" 2>&1; then
    touch "$RUN/$name.done"; say "OK    $name"
  else
    local rc=$?; say "FAIL  $name (exit $rc) -- see logs/$log; continuing"
    touch "$RUN/$name.failed"
  fi
}

cd "$SIM_DIR"
say "run $RUN_ID  mode=$MODE  robots=$ROBOTS  deadline=${DEADLINE_HOURS}h"
say "sim=$SIM_DIR  [${SIM_PY[*]}]"; say "nn=$NN_DIR  [${NN_PY[*]}]"
say "git $(git rev-parse --short HEAD 2>/dev/null) $(git status --porcelain 2>/dev/null | wc -l) dirty files"

# ---- 0. environment gate: fail here, not in three hours --------------------
check_env() {
  local rc=0
  echo "sim launcher: ${SIM_PY[*]}"; echo "nn  launcher: ${NN_PY[*]}"
  # Run all four and report each, so one log tells the whole story.
  _chk() {
    local name="$1"; shift
    echo "--- $name"
    if "$@"; then echo "PASS $name"; else echo "FAIL $name (exit $?)"; rc=1; fi
  }
  _chk sim-imports "${SIM_PY[@]}" -c 'import pinocchio, mujoco, numpy, pandas; print("sim deps ok")'
  _chk nn-imports  "${NN_PY[@]}"  -c 'import torch, optuna; print("torch", torch.__version__, "cuda", torch.cuda.is_available())'
  _chk qc-cli      "${SIM_PY[@]}" scripts/diagnose_controller_modes.py --help
  _chk optuna-cli  "${NN_PY[@]}"  "$NN_DIR/optuna_search_lite.py" --help
  echo "=== configs present ==="
  for c in "${PLATFORMS[@]}"; do
    if [ -f "config/identification/$c.yaml" ]; then echo "PASS config $c"
    else echo "FAIL config $c missing"; rc=1; fi
  done
  return $rc
}

stage 00-env env.log check_env
[ -f "$RUN/00-env.failed" ] && { say "ABORT: environment gate failed"; exit 1; }

# ---- 1. matched controller datasets + Q-C statistics -----------------------
# diagnose_controller_modes builds one dataset per controller from the same
# config and seed, which is exactly the matched set the A->B experiment needs.
for cfg in "${PLATFORMS[@]}"; do
  stage "10-qc-$cfg" "qc-$cfg.log" \
    "${SIM_PY[@]}" scripts/diagnose_controller_modes.py \
      --config "config/identification/$cfg.yaml" \
      --generate $QC_FLAGS --modes "${MODES[@]}" \
      --robots "$ROBOTS" --trajectories "$TRAJ" \
      --out "reports/$RUN_ID/qc-$cfg" --data-dir "data/identification/$RUN_ID/$cfg"
done

# ---- 2. Optuna, one study per (platform, controller) -----------------------
for cfg in "${PLATFORMS[@]}"; do
  for m in "${MODES[@]}"; do
    ds=$(ls -1 "data/identification/$RUN_ID/$cfg"/*"$m"*.csv 2>/dev/null | head -1)
    [ -z "$ds" ] && { say "SKIP  20-opt-$cfg-$m -- no dataset built"; continue; }
    stage "20-opt-$cfg-$m" "opt-$cfg-$m.log" \
      "${NN_PY[@]}" "$NN_DIR/optuna_search_lite.py" \
        --dataset "$ds" --label "${cfg%%_*}-$m" \
        --n-trials "$TRIALS" --screen-epochs "$SCREEN_EPOCHS" \
        --refine-top "$REFINE_TOP" --refine-epochs "$REFINE_EPOCHS" \
        --db "$RUN/optuna.db"
  done
done

# ---- 3. cross-controller evaluation (train A -> test B) --------------------
# Runs only if the agent has landed it; the run is still useful without it.
if [ -f "$NN_DIR/crosseval.py" ]; then
  stage 30-crosseval crosseval.log \
    "${NN_PY[@]}" "$NN_DIR/crosseval.py" --data-root "data/identification/$RUN_ID" \
      --study-db "$RUN/optuna.db" --out "reports/$RUN_ID/crosseval.csv"
else
  say "SKIP  30-crosseval -- crosseval.py not present (T-19 not landed)"
fi

# ---- 4. collect ------------------------------------------------------------
{
  echo; echo "## Result files"
  find "reports/$RUN_ID" -type f 2>/dev/null | sed 's/^/- /'
  echo; echo "## Datasets"
  du -sh "data/identification/$RUN_ID"/* 2>/dev/null | sed 's/^/- /'
  echo; echo "## Stages"
  ls "$RUN" | grep -E '\.(done|failed)$' | sed 's/^/- /'
  echo; echo "finished $(date)"
} >> "$STATUS"
say "DONE -- read $STATUS"
