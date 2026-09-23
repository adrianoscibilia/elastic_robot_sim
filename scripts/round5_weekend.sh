#!/usr/bin/env bash
# Round 5, unattended: datasets -> Optuna -> real trainings with checkpoints.
#
#   bash scripts/round5_weekend.sh doctor     # GO / NO-GO, changes nothing
#   bash scripts/round5_weekend.sh preflight  # ~20 min, runs every stage tiny
#   bash scripts/round5_weekend.sh launch     # detached, sleep-inhibited, supervised
#
# Everything is resumable: each stage drops runs/<id>/<stage>.done and is
# skipped on re-run, so a crash, an OOM kill or a power cut costs one stage.
# After a power cut, run `launch` again with the same RUN_ID to resume.
set -uo pipefail

SIM_DIR="${SIM_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
NN_DIR="${NN_DIR:-$(cd "$SIM_DIR/../dynamic_model_nn" && pwd)}"
CMD="${1:-doctor}"

pick_py() {
  local d="$1"
  [ -x "$d/.venv/bin/python" ] && { echo "$d/.venv/bin/python"; return; }
  if command -v uv >/dev/null 2>&1 && [ -f "$d/pyproject.toml" ]; then echo "uv run --project $d python"; return; fi
  command -v python3 >/dev/null 2>&1 && { echo python3; return; }
  echo python
}
read -r -a SIM_PY <<< "${SIM_PY_CMD:-$(pick_py "$SIM_DIR")}"
read -r -a NN_PY  <<< "${NN_PY_CMD:-$(pick_py "$NN_DIR")}"

IIWA="config/identification/kuka_lbr_iiwa_14_r820_table_round5.yaml"
UR10="config/identification/ur10_table_round5.yaml"
MODES=(exact_ct pd velocity_pi)
MIN_FREE_GB="${MIN_FREE_GB:-60}"

# ---------------------------------------------------------------- doctor ----
doctor() {
  local rc=0
  ok()   { echo "  PASS  $*"; }
  bad()  { echo "  FAIL  $*"; rc=1; }
  warn() { echo "  warn  $*"; }

  echo "== interpreters"
  echo "  sim: ${SIM_PY[*]}"; echo "  nn : ${NN_PY[*]}"
  "${SIM_PY[@]}" -c 'import pinocchio,mujoco,numpy,pandas' 2>/dev/null \
    && ok "sim imports (pinocchio, mujoco)" || bad "sim imports"
  "${NN_PY[@]}" -c 'import torch,optuna,yaml' 2>/dev/null \
    && ok "nn imports (torch, optuna, yaml)" || bad "nn imports"
  "${NN_PY[@]}" -c 'import torch;print("  info  cuda:",torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")' 2>/dev/null

  echo "== entry points"
  ( cd "$SIM_DIR" && "${SIM_PY[@]}" scripts/diagnose_controller_modes.py --help >/dev/null 2>&1 ) \
    && ok "diagnose_controller_modes --help" || bad "diagnose_controller_modes --help"
  for e in optuna_search_lite.py export_best_to_chain.py train_chain_from_config.py; do
    [ -f "$NN_DIR/$e" ] && ok "$e present" || bad "$e MISSING in $NN_DIR"
  done
  ( cd "$NN_DIR" && "${NN_PY[@]}" optuna_search_lite.py --help >/dev/null 2>&1 ) \
    && ok "optuna_search_lite --help" || bad "optuna_search_lite --help"
  ( cd "$NN_DIR" && "${NN_PY[@]}" export_best_to_chain.py --help >/dev/null 2>&1 ) \
    && ok "export_best_to_chain --help" || bad "export_best_to_chain --help"

  echo "== configs"
  for c in "$IIWA" "$UR10"; do
    [ -f "$SIM_DIR/$c" ] && ok "$(basename "$c")" || bad "$c MISSING (pass 3 renamed these)"
  done
  if [ -f "$SIM_DIR/$IIWA" ]; then
    local top
    top=$(grep -oE 'probe_top[_a-z]*: *[0-9.]+' "$SIM_DIR/$IIWA" | head -1 | grep -oE '[0-9.]+$')
    if [ -n "$top" ]; then
      awk -v t="$top" 'BEGIN{exit !(t<=90.001)}' \
        && ok "iiwa probe top ${top} Hz (<= 90, Q-14)" \
        || warn "iiwa probe top ${top} Hz -- Q-14 says cap at 90 BEFORE building"
    else
      warn "could not read the iiwa probe top; check Q-14 by hand"
    fi
  fi

  echo "== machine"
  local free; free=$(df -BG --output=avail "$SIM_DIR" | tail -1 | tr -dc '0-9')
  [ "${free:-0}" -ge "$MIN_FREE_GB" ] && ok "${free}G free (need ${MIN_FREE_GB})" || bad "only ${free}G free, need ${MIN_FREE_GB}"
  command -v systemd-inhibit >/dev/null && ok "systemd-inhibit available (blocks sleep)" \
    || warn "no systemd-inhibit: make sure this machine will not suspend"
  command -v setsid >/dev/null && ok "setsid available (survives logout)" || bad "setsid missing"
  echo "  info  git $(cd "$SIM_DIR" && git rev-parse --short HEAD 2>/dev/null), $(cd "$SIM_DIR" && git status --porcelain 2>/dev/null | wc -l) dirty"

  echo
  if [ $rc -eq 0 ]; then echo "GO -- next:  bash scripts/round5_weekend.sh preflight"
  else echo "NO-GO -- fix the FAIL lines above"; fi
  return $rc
}

# ------------------------------------------------------------- the stages ---
work() {
  local MODE="$1"
  RUN_ID="${RUN_ID:-$MODE-$(date +%Y%m%d-%H%M)}"
  RUN="$SIM_DIR/runs/$RUN_ID"; LOGS="$RUN/logs"; mkdir -p "$LOGS"
  STATUS="$RUN/STATUS.md"; DB="$RUN/optuna.db"
  DATA="data/identification/$RUN_ID"

  if [ "$MODE" = preflight ]; then
    ROBOTS=2; TRAJ=1; FULL=""; TRIALS=4; SCREEN=2; TOP=1; REFINE=3; EPOCHS=3
    STAGE_TIMEOUT="${STAGE_TIMEOUT:-25m}"; DEADLINE_HOURS="${DEADLINE_HOURS:-2}"
  else
    ROBOTS="${ROBOTS:-20}"; TRAJ="${TRAJ:-6}"; FULL="--full"; TRIALS=48
    SCREEN=20; TOP=5; REFINE=300; EPOCHS=300
    STAGE_TIMEOUT="${STAGE_TIMEOUT:-18h}"; DEADLINE_HOURS="${DEADLINE_HOURS:-80}"
  fi
  STARTED=$(date +%s)

  say() { printf '%s | %s\n' "$(date '+%m-%d %H:%M:%S')" "$*" | tee -a "$STATUS"; }
  stage() {
    local name="$1" log="$2"; shift 2
    [ -f "$RUN/$name.done" ] && { say "skip  $name"; return 0; }
    local used=$(( ($(date +%s) - STARTED) / 3600 ))
    if [ "$used" -ge "$DEADLINE_HOURS" ]; then say "SKIP  $name (past ${DEADLINE_HOURS}h)"; return 0; fi
    local free; free=$(df -BG --output=avail "$SIM_DIR" | tail -1 | tr -dc '0-9')
    if [ "${free:-0}" -lt 10 ]; then say "STOP  $name -- only ${free}G free"; return 1; fi
    say "START $name"
    if timeout --signal=INT --kill-after=180 "$STAGE_TIMEOUT" "$@" >>"$LOGS/$log" 2>&1; then
      touch "$RUN/$name.done"; say "OK    $name"
    else
      say "FAIL  $name (exit $?) -- logs/$log"; touch "$RUN/$name.failed"
    fi
    return 0
  }

  cd "$SIM_DIR"
  say "=== $RUN_ID  robots=$ROBOTS traj=$TRAJ trials=$TRIALS deadline=${DEADLINE_HOURS}h pid=$$"
  say "sim [${SIM_PY[*]}]   nn [${NN_PY[*]}]"

  # 1. Datasets first: they are the irreplaceable artifact, and the iiwa before
  #    the UR10 because it is the only platform with a true link-side target.
  local pair tag cfg
  for pair in "iiwa:$IIWA" "ur10:$UR10"; do
    tag="${pair%%:*}"; cfg="${pair#*:}"
    if [ ! -f "$cfg" ]; then say "SKIP  datasets $tag -- $cfg missing"; continue; fi
    stage "10-data-$tag" "data-$tag.log" \
      "${SIM_PY[@]}" scripts/diagnose_controller_modes.py \
        --config "$cfg" --generate $FULL --modes "${MODES[@]}" \
        --robots "$ROBOTS" --trajectories "$TRAJ" \
        --out "reports/$RUN_ID/qc-$tag" --data-dir "$DATA/$tag"
  done

  # 2. Optuna searches, 3. export the winners, 4. train them for real.
  #    optuna_search_lite writes no weights, so the chain step is what produces
  #    the checkpoints the ordinary training path saves.
  local m ds label
  for tag in iiwa ur10; do
    for m in "${MODES[@]}"; do
      ds=$(ls -1 "$DATA/$tag"/*"$m"*.csv 2>/dev/null | head -1)
      if [ -z "$ds" ]; then say "SKIP  $tag-$m -- no dataset"; continue; fi
      label="$tag-$m"
      stage "20-optuna-$label" "optuna-$label.log" \
        "${NN_PY[@]}" "$NN_DIR/optuna_search_lite.py" \
          --dataset "$SIM_DIR/$ds" --label "$label" --n-trials "$TRIALS" \
          --screen-epochs "$SCREEN" --refine-top "$TOP" --refine-epochs "$REFINE" \
          --db "$DB"
      stage "30-export-$label" "export-$label.log" \
        "${NN_PY[@]}" "$NN_DIR/export_best_to_chain.py" \
          --db "$DB" --label "$label" --dataset "$SIM_DIR/$ds" \
          --epochs "$EPOCHS" --out "$RUN/chain-$label.yaml"
      if [ -f "$RUN/chain-$label.yaml" ]; then
        stage "40-train-$label" "train-$label.log" \
          env -C "$NN_DIR" "${NN_PY[@]}" "$NN_DIR/train_chain_from_config.py" \
            --config "$RUN/chain-$label.yaml"
      fi
    done
  done

  # 5. Collect: hashes of every dataset, the chosen hyperparameters, and every
  #    checkpoint written since this run started.
  {
    echo; echo "## datasets"; find "$DATA" -name '*.csv' -exec sha256sum {} \; 2>/dev/null
    echo; echo "## chain configs (chosen hyperparameters)"; ls "$RUN"/chain-*.yaml 2>/dev/null
    echo; echo "## artefacts written during this run"
    find "$NN_DIR/models" "$NN_DIR/log" -newermt "@$STARTED" -type f 2>/dev/null | sed 's/^/- /'
    echo; echo "## stages"; ls "$RUN" | grep -E '\.(done|failed)$' | sed 's/^/- /'
    echo; echo "finished $(date)"
  } >> "$STATUS"
  cp -f "$DB" "$RUN/optuna-final.db" 2>/dev/null
  say "DONE -- read $STATUS"
  return 0
}

# ------------------------------------------------------------- supervisor ---
supervise() {
  local mode="$1" n=0
  # A crash or an OOM kill costs one stage, not the weekend: stages are
  # idempotent, so re-entering work() simply resumes.
  while [ $n -lt 6 ]; do
    n=$((n+1))
    echo "--- supervisor attempt $n $(date)"
    work "$mode" && break
    sleep 60
  done
}

case "$CMD" in
  doctor)    doctor ;;
  preflight) doctor && work preflight ;;
  run)       supervise full ;;
  launch)
    if ! doctor; then echo "refusing to launch: doctor says NO-GO"; exit 1; fi
    mkdir -p "$SIM_DIR/runs"
    export RUN_ID="${RUN_ID:-full-$(date +%Y%m%d-%H%M)}"
    boot="$SIM_DIR/runs/$RUN_ID.boot.log"
    inhibit=()
    command -v systemd-inhibit >/dev/null && \
      inhibit=(systemd-inhibit --what=sleep:idle:shutdown:handle-lid-switch \
               --who=round5 --why="round-5 unattended run" --mode=block)
    setsid nohup nice -n -5 ionice -c2 -n0 "${inhibit[@]}" bash -c \
      'echo -500 > /proc/self/oom_score_adj 2>/dev/null; exec bash "$0" run' \
      "$SIM_DIR/scripts/round5_weekend.sh" \
      </dev/null >>"$boot" 2>&1 &
    disown
    echo "launched RUN_ID=$RUN_ID"
    echo "watch:  tail -f $SIM_DIR/runs/$RUN_ID/STATUS.md"
    echo "boot :  $boot"
    ;;
  *) echo "usage: $0 {doctor|preflight|launch}"; exit 2 ;;
esac
