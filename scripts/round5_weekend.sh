#!/usr/bin/env bash
# Round 5, unattended: datasets -> Optuna -> real trainings -> evaluation.
#
#   bash scripts/round5_weekend.sh doctor     # GO / NO-GO, builds one tiny dataset in a temp dir
#   bash scripts/round5_weekend.sh preflight  # every stage, tiny; writes the launch ticket
#   bash scripts/round5_weekend.sh launch     # detached, sleep-inhibited, supervised
#
# Scope (env): PLATFORMS="iiwa ur10"  MODES="exact_ct pd velocity_pi"
#   e.g. PLATFORMS=ur10 MODES=velocity_pi bash scripts/round5_weekend.sh preflight
#
# Everything is resumable: each stage drops runs/<id>/<stage>.done and is
# skipped on re-run, so a crash, an OOM kill or a power cut costs one stage.
# After a power cut, run `launch` again with the same RUN_ID to resume.
#
# R5_10 T-8: one run at a time (flock on runs/.launch.lock, held by preflight
# and by the detached run); `launch` needs a green preflight ticket for the
# same git hashes and dirty-tree digest; stage timeouts come from that
# preflight's measured rates x 3.
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
HELP=("${SIM_PY[@]}" "$SIM_DIR/scripts/launch_helpers.py")

declare -A CONFIG=(
  [iiwa]="config/identification/kuka_lbr_iiwa_14_r820_table_round5.yaml"
  [ur10]="config/identification/ur10_table_round5.yaml"
)
read -r -a PLATFORM_LIST <<< "${PLATFORMS:-iiwa ur10}"
read -r -a MODES <<< "${MODES:-exact_ct pd velocity_pi}"
MODELS=(lnn_tau_res lnn_tau_elastic kalnn_tau_res kalnn_tau_elastic)
# Platforms whose unresolvable probe an architect decision accepted (R5_09 D-2).
PROBE_ACCEPTED="${PROBE_ACCEPTED:-ur10}"
MIN_FREE_GB="${MIN_FREE_GB:-60}"
JOBS="${JOBS:-$(( $(nproc) > 1 ? $(nproc) - 1 : 1 ))}"
SCREEN_ROWS="${SCREEN_ROWS:-100000}"
LOCK="$SIM_DIR/runs/.launch.lock"

# The tree a preflight vouches for: both repos' HEAD, tracked diff and
# untracked *source* files (outputs such as models/ or runs/ must not count,
# or a preflight would invalidate its own ticket).
tree_digest() {
  local d
  for d in "$SIM_DIR" "$NN_DIR"; do
    git -C "$d" rev-parse HEAD
    git -C "$d" diff HEAD --binary
    git -C "$d" ls-files -o --exclude-standard -z -- '*.py' '*.sh' '*.yaml' '*.yml' '*.toml' \
      | (cd "$d" && xargs -0 -r sha256sum)
  done 2>/dev/null | sha256sum | cut -c1-16
}
git_heads() { echo "sim=$(git -C "$SIM_DIR" rev-parse --short HEAD) nn=$(git -C "$NN_DIR" rev-parse --short HEAD)"; }

# ---------------------------------------------------------------- doctor ----
doctor() {
  # $1 = launch: the sleep check is a FAIL (the run lasts days); a preflight
  # only warns, since its own run takes minutes and holds no inhibitor.
  local context="${1:-check}" rc=0 tag
  ok()   { echo "  PASS  $*"; }
  bad()  { echo "  FAIL  $*"; rc=1; }
  warn() { echo "  warn  $*"; }

  echo "== interpreters"
  echo "  sim: ${SIM_PY[*]}"; echo "  nn : ${NN_PY[*]}"
  "${SIM_PY[@]}" -c 'import pinocchio,mujoco,numpy,pandas' 2>/dev/null \
    && ok "sim imports (pinocchio, mujoco)" || bad "sim imports"
  "${NN_PY[@]}" -c 'import torch,optuna,yaml,pyarrow' 2>/dev/null \
    && ok "nn imports (torch, optuna, yaml, pyarrow)" || bad "nn imports"
  "${NN_PY[@]}" -c 'import torch;print("  info  cuda:",torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")' 2>/dev/null

  echo "== entry points"
  ( cd "$SIM_DIR" && "${SIM_PY[@]}" scripts/diagnose_controller_modes.py --help >/dev/null 2>&1 ) \
    && ok "diagnose_controller_modes --help" || bad "diagnose_controller_modes --help"
  local e
  for e in optuna_search_lite.py export_best_to_chain.py train_chain_from_config.py evaluate_on.py; do
    ( cd "$NN_DIR" && "${NN_PY[@]}" "$e" --help >/dev/null 2>&1 ) && ok "$e --help" || bad "$e --help"
  done

  echo "== configs"
  for tag in "${PLATFORM_LIST[@]}"; do
    local cfg="${CONFIG[$tag]:-}"
    if [ -z "$cfg" ] || [ ! -f "$SIM_DIR/$cfg" ]; then bad "$tag: config ${cfg:-unknown} missing"; continue; fi
    ok "$tag: $(basename "$cfg")"
    # probe_harmonics x base_frequency against the Savitzky-Golay floor 0.09 x rate:
    # a FAIL, not a warning (R5_10 T-8).
    local line; line=$( cd "$SIM_DIR" && "${HELP[@]}" probe "$cfg" "$tag" "$PROBE_ACCEPTED" 2>&1 )
    case "$line" in PASS*) ok "${line#* }";; ACCEPTED*) echo "  ACCEPTED  ${line#* }";; *) bad "${line#* }";; esac
  done

  echo "== fresh build -> CustomDataset"
  # One tiny real build, read back by the consumer: the handoff the weekend
  # never exercised (R5_09 F-2).
  local tmp; tmp=$(mktemp -d)
  tag="${PLATFORM_LIST[0]}"
  if ( cd "$SIM_DIR" && timeout 20m "${SIM_PY[@]}" -W ignore scripts/diagnose_controller_modes.py \
         --config "${CONFIG[$tag]}" --generate --modes "${MODES[0]}" --robots 1 --trajectories 1 \
         --no-extras-control --backends mujoco --jobs "$JOBS" --quiet \
         --out "$tmp/qc" --data-dir "$tmp/data" >"$tmp/build.log" 2>&1 ); then
    local stem; stem=$( cd "$SIM_DIR" && "${HELP[@]}" stem "${CONFIG[$tag]}" 2>/dev/null )
    local ds="$tmp/data/${stem}_${MODES[0]}.parquet"
    if [ -f "$ds" ] && ( cd "$NN_DIR" && "${NN_PY[@]}" -c "
import sys; from dataset import CustomDataset
d = CustomDataset(sys.argv[1]); assert len(d) and d.target_is_per_joint, 'bad load'
print(f'  info  {len(d)} rows, dof {d.dof}, sg_window {d.sg_window}')" "$ds" 2>"$tmp/load.log" ); then
      ok "CustomDataset loads a fresh $tag build ($(basename "$ds"))"
    else
      bad "CustomDataset could not load $ds: $(tail -1 "$tmp/load.log" 2>/dev/null)"
    fi
  else
    bad "tiny $tag build failed: $(tail -2 "$tmp/build.log")"
  fi
  rm -rf "$tmp"

  echo "== machine"
  local free; free=$(df -BG --output=avail "$SIM_DIR" | tail -1 | tr -dc '0-9')
  [ "${free:-0}" -ge "$MIN_FREE_GB" ] && ok "${free}G free (need ${MIN_FREE_GB})" || bad "only ${free}G free, need ${MIN_FREE_GB}"
  local masked=1 t
  for t in sleep.target suspend.target hibernate.target hybrid-sleep.target; do
    [ "$(systemctl is-enabled "$t" 2>/dev/null)" = masked ] || masked=0
  done
  if command -v systemd-inhibit >/dev/null && systemd-inhibit --what=sleep:idle --who=doctor --why=check true 2>/dev/null; then
    ok "systemd-inhibit can block sleep"
  elif [ $masked -eq 1 ]; then
    ok "sleep/suspend/hibernate targets are masked"
  elif [ "$context" = launch ]; then
    bad "cannot inhibit sleep (systemd-inhibit refused, sleep targets not masked): the machine may suspend mid-run"
  else
    warn "cannot inhibit sleep from this session (launch will FAIL on this unless run where polkit allows it, or the sleep targets are masked)"
  fi
  command -v gsettings >/dev/null && echo "  info  GNOME idle sleep on AC: $(gsettings get org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 2>/dev/null)"
  command -v setsid >/dev/null && ok "setsid available (survives logout)" || bad "setsid missing"
  command -v flock >/dev/null && ok "flock available (one run at a time)" || bad "flock missing"
  if [ -z "${HOLDING_LOCK:-}" ] && [ -e "$LOCK" ] && ! flock -n "$LOCK" true; then bad "another run holds $LOCK"; fi
  echo "  info  $(git_heads), tree digest $(tree_digest), jobs $JOBS"

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
  TIMEOUTS="$RUN/timeouts.tsv"

  if [ "$MODE" = preflight ]; then
    # One Optuna trial per model, one epoch of everything: a pipeline check.
    ROBOTS=2; TRAJ=1; FULL=""; TRIALS=1; SCREEN=1; TOP=1; REFINE=1; EPOCHS=1
    DEADLINE_HOURS="${DEADLINE_HOURS:-3}"
  else
    ROBOTS="${ROBOTS:-20}"; TRAJ="${TRAJ:-3}"; FULL="--full"; TRIALS=48
    SCREEN=20; TOP=5; REFINE=300; EPOCHS=300
    DEADLINE_HOURS="${DEADLINE_HOURS:-80}"
  fi
  STARTED=$(date +%s)

  say() { printf '%s | %s\n' "$(date '+%m-%d %H:%M:%S')" "$*" | tee -a "$STATUS"; }
  stage_timeout() {
    local t=""
    [ -f "$TIMEOUTS" ] && t=$(awk -F'\t' -v s="$1" '$1==s{print $2}' "$TIMEOUTS")
    if [ -n "$t" ]; then echo "${t}s"; elif [ "$MODE" = preflight ]; then echo "${PREFLIGHT_STAGE_TIMEOUT:-45m}"; else echo "${DEFAULT_STAGE_TIMEOUT:-2h}"; fi
  }
  stage() {
    local name="$1" log="$2"; shift 2
    [ -f "$RUN/$name.done" ] && { say "skip  $name"; return 0; }
    local used=$(( ($(date +%s) - STARTED) / 3600 ))
    if [ "$used" -ge "$DEADLINE_HOURS" ]; then say "SKIP  $name (past ${DEADLINE_HOURS}h)"; touch "$RUN/$name.failed"; return 1; fi
    local free; free=$(df -BG --output=avail "$SIM_DIR" | tail -1 | tr -dc '0-9')
    if [ "${free:-0}" -lt 10 ]; then say "STOP  $name -- only ${free}G free"; return 1; fi
    local limit; limit=$(stage_timeout "$name")
    say "START $name (timeout $limit)"
    local t0; t0=$(date +%s)
    rm -f "$RUN/$name.failed"
    if timeout --signal=INT --kill-after=180 "$limit" "$@" >>"$LOGS/$log" 2>&1; then
      printf '%s\t%s\n' "$name" "$(( $(date +%s) - t0 ))" >> "$RUN/timing.tsv"
      touch "$RUN/$name.done"; say "OK    $name ($(( $(date +%s) - t0 ))s)"; return 0
    fi
    local code=$?
    say "FAIL  $name (exit $code) -- logs/$log"; touch "$RUN/$name.failed"; return 1
  }

  cd "$SIM_DIR"
  say "=== $RUN_ID  platforms=${PLATFORM_LIST[*]} modes=${MODES[*]} robots=$ROBOTS traj=$TRAJ trials=$TRIALS jobs=$JOBS pid=$$"
  say "sim [${SIM_PY[*]}]   nn [${NN_PY[*]}]   $(git_heads) digest $(tree_digest)"

  # 1. Datasets first: they are the irreplaceable artifact.  MuJoCo only
  #    (R5_10 T-3.2); Newton is scripts/audit_backends.py.
  local tag cfg
  for tag in "${PLATFORM_LIST[@]}"; do
    cfg="${CONFIG[$tag]}"
    stage "10-data-$tag" "data-$tag.log" \
      "${SIM_PY[@]}" scripts/diagnose_controller_modes.py \
        --config "$cfg" --generate $FULL --modes "${MODES[@]}" \
        --robots "$ROBOTS" --trajectories "$TRAJ" --backends mujoco --jobs "$JOBS" \
        --out "reports/$RUN_ID/qc-$tag" --data-dir "$DATA/$tag"
  done

  # 2. Optuna, 3. export the winners, 4. train them for real, 5. evaluate
  #    every trained model on the test split.  Datasets are named exactly
  #    (R5_10 T-1): a missing file fails its stage, it is never skipped or
  #    guessed with a glob.
  local m ds label suffix stem
  for tag in "${PLATFORM_LIST[@]}"; do
    stem=$("${HELP[@]}" stem "${CONFIG[$tag]}" 2>/dev/null)
    for m in "${MODES[@]}"; do
      for suffix in "" "_noextras"; do
        ds="$SIM_DIR/$DATA/$tag/${stem}_${m}${suffix}.parquet"
        label="$tag-$m${suffix//_/-}"
        if [ ! -f "$ds" ]; then
          say "FAIL  20-optuna-$label -- expected dataset missing: $ds"
          touch "$RUN/20-optuna-$label.failed"; continue
        fi
        stage "20-optuna-$label" "optuna-$label.log" \
          env -C "$NN_DIR" "${NN_PY[@]}" "$NN_DIR/optuna_search_lite.py" \
            --dataset "$ds" --label "$label" --n-trials "$TRIALS" --screen-rows "$SCREEN_ROWS" \
            --screen-epochs "$SCREEN" --refine-top "$TOP" --refine-epochs "$REFINE" \
            --db "$DB" --prior-db "${PRIOR_DB:-data/optuna_search_fmrr.db}" || continue
        stage "30-export-$label" "export-$label.log" \
          env -C "$NN_DIR" "${NN_PY[@]}" "$NN_DIR/export_best_to_chain.py" \
            --db "$DB" --label "$label" --dataset "$ds" \
            --epochs "$EPOCHS" --out "$RUN/chain-$label.yaml" || continue
        stage "40-train-$label" "train-$label.log" \
          env -C "$NN_DIR" "${NN_PY[@]}" "$NN_DIR/train_chain_from_config.py" \
            --config "$RUN/chain-$label.yaml" --summary-out "$RUN/chain-$label.summary.csv" || continue
        stage "50-eval-$label" "eval-$label.log" \
          env -C "$NN_DIR" "${NN_PY[@]}" "$NN_DIR/evaluate_on.py" \
            --chain-summary "$RUN/chain-$label.summary.csv" --dataset "$ds" --split test \
            --out "$RUN/eval-$label.csv"
      done
    done
  done

  # 6. Collect: hashes of every dataset, the chosen hyperparameters, and every
  #    checkpoint written since this run started.
  {
    echo; echo "## datasets"; find "$SIM_DIR/$DATA" -name '*.parquet' -exec sha256sum {} \; 2>/dev/null
    echo; echo "## chain configs (chosen hyperparameters)"; ls "$RUN"/chain-*.yaml 2>/dev/null
    echo; echo "## evaluations"; ls "$RUN"/eval-*.csv 2>/dev/null
    echo; echo "## artefacts written during this run"
    find "$NN_DIR/models" "$NN_DIR/log" -newermt "@$STARTED" -type f 2>/dev/null | sed 's/^/- /'
    echo; echo "## stages"; ls "$RUN" | grep -E '\.(done|failed)$' | sed 's/^/- /'
    echo; echo "finished $(date)"
  } >> "$STATUS"
  cp -f "$DB" "$RUN/optuna-final.db" 2>/dev/null

  local failed; failed=$(ls "$RUN" | grep -c '\.failed$')
  if [ "$MODE" = preflight ]; then
    if [ "$failed" -eq 0 ]; then
      # The launch ticket: which tree, which scope, and what it needs to size
      # the full run's stage timeouts.
      {
        echo "heads=$(git_heads | tr ' ' ',')"
        echo "digest=$(tree_digest)"
        echo "platforms=${PLATFORM_LIST[*]}"
        echo "modes=${MODES[*]}"
        echo "finished=$(date -Is)"
      } > "$RUN/PREFLIGHT_OK"
      say "PREFLIGHT GREEN -- ticket $RUN/PREFLIGHT_OK"
    else
      say "PREFLIGHT RED -- $failed stage(s) failed; no launch ticket"
    fi
  fi
  say "DONE -- read $STATUS"
  [ "$failed" -eq 0 ]
}

# ----------------------------------------------------------- launch ticket ---
find_ticket() {
  # A green preflight for this exact tree and at least this scope.
  local heads digest t
  heads="heads=$(git_heads | tr ' ' ',')"; digest="digest=$(tree_digest)"
  for t in $(ls -1t "$SIM_DIR"/runs/*/PREFLIGHT_OK 2>/dev/null); do
    grep -qx "$heads" "$t" && grep -qx "$digest" "$t" \
      && grep -qx "platforms=${PLATFORM_LIST[*]}" "$t" && grep -qx "modes=${MODES[*]}" "$t" \
      && { dirname "$t"; return 0; }
  done
  return 1
}

write_timeouts() {
  # Stage timeouts for the full run from the ticket's measured rates x 3.
  local pre="$1" out="$2" tag plan
  plan=$(mktemp)
  {
    echo '{"preflight": {"trials":1,"screen":1,"top":1,"refine":1,"epochs":1,"models":4,"robots":2,"traj":1},'
    echo " \"full\": {\"trials\":48,\"screen\":20,\"top\":5,\"refine\":300,\"epochs\":300,\"models\":4,\"robots\":${ROBOTS:-20},\"traj\":${TRAJ:-3}},"
    echo " \"screen_rows\": $SCREEN_ROWS, \"cap_s\": $(( ${DEADLINE_HOURS:-80} * 3600 )), \"platforms\": {"
    local first=1 m suffix stem pre_id; pre_id=$(basename "$pre")
    for tag in "${PLATFORM_LIST[@]}"; do
      stem=$("${HELP[@]}" stem "${CONFIG[$tag]}" 2>/dev/null)
      [ $first -eq 1 ] || echo ","; first=0
      printf '  "%s": {"data_dir": "%s", "datasets": {' "$tag" "$SIM_DIR/data/identification/$pre_id/$tag"
      local f2=1
      for m in "${MODES[@]}"; do for suffix in "" "_noextras"; do
        [ $f2 -eq 1 ] || printf ','; f2=0
        printf '"%s": "%s"' "$tag-$m${suffix//_/-}" "$SIM_DIR/data/identification/$pre_id/$tag/${stem}_${m}${suffix}.parquet"
      done; done
      printf '}}'
    done
    echo '}}'
  } > "$plan"
  "${HELP[@]}" timeouts "$pre" "$plan" > "$out"; local rc=$?
  rm -f "$plan"; return $rc
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

mkdir -p "$SIM_DIR/runs"
case "$CMD" in
  doctor)    doctor ;;
  preflight)
    exec 9>"$LOCK"
    flock -n 9 || { echo "refusing: another preflight/run holds $LOCK"; exit 1; }
    HOLDING_LOCK=1
    doctor && work preflight ;;
  run)
    # The detached process holds the lock for its whole life.
    exec 9>"$LOCK"
    flock -n 9 || { echo "refusing: another run holds $LOCK"; exit 1; }
    supervise full ;;
  launch)
    if ! flock -n "$LOCK" true; then echo "refusing to launch: another preflight/run holds $LOCK"; exit 1; fi
    if ! doctor launch; then echo "refusing to launch: doctor says NO-GO"; exit 1; fi
    ticket=$(find_ticket) || {
      echo "refusing to launch: no green preflight for $(git_heads), digest $(tree_digest),"
      echo "platforms '${PLATFORM_LIST[*]}', modes '${MODES[*]}'. Run: bash scripts/round5_weekend.sh preflight"
      exit 1; }
    export RUN_ID="${RUN_ID:-full-$(date +%Y%m%d-%H%M)}"
    mkdir -p "$SIM_DIR/runs/$RUN_ID"
    write_timeouts "$ticket" "$SIM_DIR/runs/$RUN_ID/timeouts.tsv" \
      || { echo "refusing to launch: could not size stage timeouts from $ticket"; exit 1; }
    echo "ticket $ticket; stage timeouts:"; sed 's/^/  /' "$SIM_DIR/runs/$RUN_ID/timeouts.tsv"
    boot="$SIM_DIR/runs/$RUN_ID.boot.log"
    inhibit=(systemd-inhibit --what=sleep:idle:shutdown:handle-lid-switch
             --who=round5 --why="round-5 unattended run" --mode=block)
    # No `nice -n -5`: it needs root and only printed "cannot set niceness".
    export PLATFORMS="${PLATFORM_LIST[*]}" MODES="${MODES[*]}"
    setsid nohup ionice -c2 -n0 "${inhibit[@]}" bash -c \
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
