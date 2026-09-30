#!/usr/bin/env bash
# Round 6, unattended (R6_04 Sec 4 steps 7-9): datasets -> Optuna -> trainings -> evaluation.
#
#   bash scripts/round6_run.sh doctor     # GO / NO-GO: suites, the Sec 2.2/3 bounds, a tiny build read back
#   bash scripts/round6_run.sh preflight  # every stage tiny, NN stages at two epoch counts; writes the ticket
#   bash scripts/round6_run.sh launch     # detached, sleep-inhibited, supervised full run
#
# Scope (env): VARIANTS="iiwa_drive iiwa_bus fmrr" (default; ur10 is dropped, R6_05)
#   e.g. VARIANTS="fmrr" bash scripts/round6_run.sh preflight
#
# Per variant: <stem>.parquet + <stem>_gainshift.parquet (iiwa_drive also
# _ablation_off and _ablation_clean), each held to the Sec 9 hard checks; a
# refused file is written *.refused.parquet, listed in STATUS.md and skipped by
# the NN stages -- not fatal to the others.  Then, per variant, the four
# models (lnn_tau_elastic, kalnn_tau_elastic, lnn_tau_res, kalnn_tau_res):
# Optuna -> export -> train -> evaluate_on on the test split and on the
# gain-shift file.
#
# Hard failures stop the run (R6_04 Sec 4 step 10): red test suites or a
# doctor FAIL (launch refuses), every variant of a platform refused, or less
# than 10 GB free.
#
# Everything is resumable: each stage drops runs/<id>/<stage>.done and is
# skipped on re-run; after a power cut, run `launch` again with the same
# RUN_ID.  One run at a time (flock on runs/.launch.lock); `launch` needs a
# green preflight ticket for the same git hashes and dirty-tree digest, and
# sizes its stage timeouts from that preflight x 3.
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
  [iiwa_drive]="config/identification/kuka_lbr_iiwa_14_r820_table_round6_drive.yaml"
  [iiwa_bus]="config/identification/kuka_lbr_iiwa_14_r820_table_round6_bus.yaml"
  [ur10]="config/identification/ur10_table_round6.yaml"
  [fmrr]="config/identification/fmrr_tecnobody_round6.yaml"
)
declare -A PLATFORM=([iiwa_drive]=iiwa [iiwa_bus]=iiwa [ur10]=ur10 [fmrr]=fmrr)
# UR10 `drive` failed budget gate 1b (loop noise > elastic / 5) and is dropped
# from production (R6_04 Sec 4 step 6, R6_05); VARIANTS="... ur10" still builds it.
read -r -a VARIANT_LIST <<< "${VARIANTS:-iiwa_drive iiwa_bus fmrr}"
ABLATE="${ABLATE:-iiwa_drive}"
MODELS=(lnn_tau_elastic kalnn_tau_elastic lnn_tau_res kalnn_tau_res)
MIN_FREE_GB="${MIN_FREE_GB:-60}"
JOBS="${JOBS:-$(( $(nproc) > 1 ? $(nproc) - 1 : 1 ))}"
SCREEN_ROWS="${SCREEN_ROWS:-100000}"
PRE_EPOCHS=(1 3)                      # the preflight's two epoch counts (fixed cost vs per-epoch rate)
LOCK="$SIM_DIR/runs/.launch.lock"

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
stem_of() { ( cd "$SIM_DIR" && "${HELP[@]}" stem "${CONFIG[$1]}" 2>/dev/null ); }

# ---------------------------------------------------------------- doctor ----
doctor() {
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

  echo "== test suites (fast)"
  if [ "$context" != quick ]; then
    # pytest is a dev tool the project venvs may not carry: `uv run --with pytest` when uv is there.
    local sim_test=("${SIM_PY[@]}") nn_test=("${NN_PY[@]}")
    if command -v uv >/dev/null 2>&1; then
      sim_test=(uv run --project "$SIM_DIR" --with pytest python); nn_test=(uv run --project "$NN_DIR" --with pytest python)
    fi
    ( cd "$SIM_DIR" && timeout 30m "${sim_test[@]}" -m pytest -q -m "not slow" -p no:cacheprovider >/tmp/r6_sim_tests.log 2>&1 ) \
      && ok "elastic_robot_sim: $(tail -1 /tmp/r6_sim_tests.log)" || bad "elastic_robot_sim suite red: $(tail -1 /tmp/r6_sim_tests.log)"
    ( cd "$NN_DIR" && timeout 30m "${nn_test[@]}" -m pytest -q -p no:cacheprovider test >/tmp/r6_nn_tests.log 2>&1 ) \
      && ok "dynamic_model_nn: $(tail -1 /tmp/r6_nn_tests.log)" || bad "dynamic_model_nn suite red: $(tail -1 /tmp/r6_nn_tests.log)"
  fi

  echo "== entry points"
  ( cd "$SIM_DIR" && "${SIM_PY[@]}" scripts/build_round6.py --help >/dev/null 2>&1 ) \
    && ok "build_round6 --help" || bad "build_round6 --help"
  local e
  for e in optuna_search_lite.py export_best_to_chain.py train_chain_from_config.py evaluate_on.py; do
    ( cd "$NN_DIR" && "${NN_PY[@]}" "$e" --help >/dev/null 2>&1 ) && ok "$e --help" || bad "$e --help"
  done

  echo "== configs: the Sec 2.2 gain bound, the Sec 3 explicit-term bound, the probe"
  for tag in "${VARIANT_LIST[@]}"; do
    local cfg="${CONFIG[$tag]:-}"
    if [ -z "$cfg" ] || [ ! -f "$SIM_DIR/$cfg" ]; then bad "$tag: config ${cfg:-unknown} missing"; continue; fi
    local line; line=$( cd "$SIM_DIR" && "${HELP[@]}" bounds "$cfg" "$tag" 2>&1 | tail -1 )
    case "$line" in PASS*) ok "${line#* }";; *) bad "${line#* }";; esac
  done

  echo "== fresh build -> CustomDataset (noise-free tau contract)"
  local tmp; tmp=$(mktemp -d)
  tag="${VARIANT_LIST[0]}"
  if ( cd "$SIM_DIR" && timeout 30m "${SIM_PY[@]}" -W ignore scripts/build_round6.py --config "${CONFIG[$tag]}" \
         --variants production --robots 1 --trajectories 1 --jobs "$JOBS" --quiet --refusal-ok \
         --out-dir "$tmp/data" >"$tmp/build.log" 2>&1 ); then
    local stem; stem=$(stem_of "$tag")
    local ds; ds=$(ls "$tmp/data/${stem}".parquet "$tmp/data/${stem}".refused.parquet 2>/dev/null | head -1)
    if [ -n "$ds" ] && ( cd "$NN_DIR" && "${NN_PY[@]}" -c "
import sys; from dataset import CustomDataset
d = CustomDataset(sys.argv[1], require_noise_free_tau=True); assert len(d) and d.target_is_per_joint, 'bad load'
print(f'  info  {len(d)} rows, dof {d.dof}, sg_window {d.sg_window}')" "$ds" 2>"$tmp/load.log" ); then
      ok "CustomDataset loads a fresh $tag build ($(basename "$ds"))"
    else
      bad "CustomDataset could not load ${ds:-the build}: $(tail -1 "$tmp/load.log" 2>/dev/null)"
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
  if [ $rc -eq 0 ]; then echo "GO"; else echo "NO-GO -- fix the FAIL lines above"; fi
  return $rc
}

# ------------------------------------------------------------- the stages ---
work() {
  local MODE="$1"
  RUN_ID="${RUN_ID:-r6-$MODE-$(date +%Y%m%d-%H%M)}"
  RUN="$SIM_DIR/runs/$RUN_ID"; LOGS="$RUN/logs"; mkdir -p "$LOGS"
  STATUS="$RUN/STATUS.md"; DB="$RUN/optuna.db"
  DATA="data/identification/$RUN_ID"
  TIMEOUTS="$RUN/timeouts.tsv"
  HARD_STOP=0

  local SHRINK=() EPOCH_SET
  if [ "$MODE" = preflight ]; then
    SHRINK=(--robots 3 --trajectories 1)
    TRIALS=1; TOP=1; EPOCH_SET=("${PRE_EPOCHS[@]}")
    DEADLINE_HOURS="${DEADLINE_HOURS:-4}"
  else
    TRIALS=48; SCREEN=20; TOP=5; REFINE=300; EPOCHS=300; EPOCH_SET=(full)
    DEADLINE_HOURS="${DEADLINE_HOURS:-96}"
  fi
  STARTED=$(date +%s)

  say() { printf '%s | %s\n' "$(date '+%m-%d %H:%M:%S')" "$*" | tee -a "$STATUS"; }
  stage_timeout() {
    local t="" base="${1%-e[0-9]*}"
    [ -f "$TIMEOUTS" ] && t=$(awk -F'\t' -v s="$1" -v b="$base" '$1==s||$1==b{print $2; exit}' "$TIMEOUTS")
    if [ -n "$t" ]; then echo "${t}s"; elif [ "$MODE" = preflight ]; then echo "${PREFLIGHT_STAGE_TIMEOUT:-60m}"; else echo "${DEFAULT_STAGE_TIMEOUT:-4h}"; fi
  }
  stage() {
    local name="$1" log="$2"; shift 2
    [ -f "$RUN/$name.done" ] && { say "skip  $name"; return 0; }
    local used=$(( ($(date +%s) - STARTED) / 3600 ))
    if [ "$used" -ge "$DEADLINE_HOURS" ]; then say "SKIP  $name (past ${DEADLINE_HOURS}h)"; touch "$RUN/$name.failed"; return 1; fi
    local free; free=$(df -BG --output=avail "$SIM_DIR" | tail -1 | tr -dc '0-9')
    if [ "${free:-0}" -lt 10 ]; then say "HARD STOP before $name -- only ${free}G free (< 10 GB)"; HARD_STOP=1; return 1; fi
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
  say "=== $RUN_ID  variants=${VARIANT_LIST[*]} mode=$MODE trials=$TRIALS jobs=$JOBS pid=$$"
  say "sim [${SIM_PY[*]}]   nn [${NN_PY[*]}]   $(git_heads) digest $(tree_digest)"

  # 1. Datasets first: the irreplaceable artifact.  Refused files are listed, not fatal.
  local tag cfg extra stem
  for tag in "${VARIANT_LIST[@]}"; do
    cfg="${CONFIG[$tag]}"; extra=()
    [[ " $ABLATE " == *" $tag "* ]] && extra=(--ablations)
    stage "10-data-$tag" "data-$tag.log" \
      "${SIM_PY[@]}" -W ignore scripts/build_round6.py --config "$cfg" "${SHRINK[@]}" "${extra[@]}" \
        --jobs "$JOBS" --quiet --refusal-ok --out-dir "$DATA/$tag"
    [ "$HARD_STOP" -eq 1 ] && break
    stem=$(stem_of "$tag")
    local refused; refused=$(ls "$DATA/$tag"/*.refused.parquet 2>/dev/null)
    [ -n "$refused" ] && say "REFUSED ($tag): $(echo $refused | xargs -n1 basename | tr '\n' ' ')"
  done
  # Hard failure: every variant of a platform refused (no production file).
  local platform any v
  for platform in $(printf '%s\n' "${VARIANT_LIST[@]}" | while read -r v; do echo "${PLATFORM[$v]}"; done | sort -u); do
    any=0
    for v in "${VARIANT_LIST[@]}"; do
      [ "${PLATFORM[$v]}" = "$platform" ] && [ -f "$DATA/$v/$(stem_of "$v").parquet" ] && any=1
    done
    [ $any -eq 0 ] && { say "HARD STOP -- every $platform variant refused or missing"; HARD_STOP=1; }
  done

  # 2-5. Per variant: Optuna, export, train, evaluate on test and gain-shift.
  local e ds gs label sfx
  if [ "$HARD_STOP" -eq 0 ]; then
    for tag in "${VARIANT_LIST[@]}"; do
      stem=$(stem_of "$tag")
      ds="$SIM_DIR/$DATA/$tag/${stem}.parquet"; gs="$SIM_DIR/$DATA/$tag/${stem}_gainshift.parquet"
      if [ ! -f "$ds" ]; then say "SKIP  NN stages of $tag -- production file refused or missing"; continue; fi
      for e in "${EPOCH_SET[@]}"; do
        if [ "$e" = full ]; then sfx=""; S=$SCREEN; R=$REFINE; E=$EPOCHS
        else sfx="-e$e"; S=$e; R=$e; E=$e; fi
        label="r6-$tag$sfx"
        stage "20-optuna-$tag$sfx" "optuna-$tag$sfx.log" \
          env -C "$NN_DIR" "${NN_PY[@]}" "$NN_DIR/optuna_search_lite.py" \
            --dataset "$ds" --label "$label" --models "${MODELS[@]}" --n-trials "$TRIALS" \
            --screen-rows "$SCREEN_ROWS" --screen-epochs "$S" --refine-top "$TOP" --refine-epochs "$R" \
            --db "$DB" --prior-db "${PRIOR_DB:-data/optuna_search_fmrr.db}" || continue
        [ "$HARD_STOP" -eq 1 ] && break 2
        stage "30-export-$tag$sfx" "export-$tag$sfx.log" \
          env -C "$NN_DIR" "${NN_PY[@]}" "$NN_DIR/export_best_to_chain.py" \
            --db "$DB" --label "$label" --dataset "$ds" --models "${MODELS[@]}" \
            --epochs "$E" --out "$RUN/chain-$tag$sfx.yaml" || continue
        stage "40-train-$tag$sfx" "train-$tag$sfx.log" \
          env -C "$NN_DIR" "${NN_PY[@]}" "$NN_DIR/train_chain_from_config.py" \
            --config "$RUN/chain-$tag$sfx.yaml" --summary-out "$RUN/chain-$tag$sfx.summary.csv" || continue
        stage "50-eval-$tag-test$sfx" "eval-$tag$sfx.log" \
          env -C "$NN_DIR" "${NN_PY[@]}" "$NN_DIR/evaluate_on.py" \
            --chain-summary "$RUN/chain-$tag$sfx.summary.csv" --dataset "$ds" --split test \
            --out "$RUN/eval-$tag-test$sfx.csv"
        if [ -f "$gs" ]; then
          stage "50-eval-$tag-gainshift$sfx" "eval-$tag$sfx.log" \
            env -C "$NN_DIR" "${NN_PY[@]}" "$NN_DIR/evaluate_on.py" \
              --chain-summary "$RUN/chain-$tag$sfx.summary.csv" --dataset "$gs" --split test \
              --out "$RUN/eval-$tag-gainshift$sfx.csv"
        else
          say "SKIP  gain-shift evaluation of $tag -- file refused or missing"
        fi
      done
    done
  fi

  # 6. Collect.
  {
    echo; echo "## datasets"; find "$SIM_DIR/$DATA" -name '*.parquet' -exec sha256sum {} \; 2>/dev/null
    echo; echo "## refused files"; find "$SIM_DIR/$DATA" -name '*.refused.parquet' 2>/dev/null | sed 's/^/- /'
    echo; echo "## hard checks and gates"; ls "$SIM_DIR/$DATA"/*/*.round6_checks.json 2>/dev/null | sed 's/^/- /'
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
    if [ "$failed" -eq 0 ] && [ "$HARD_STOP" -eq 0 ]; then
      {
        echo "heads=$(git_heads | tr ' ' ',')"
        echo "digest=$(tree_digest)"
        echo "variants=${VARIANT_LIST[*]}"
        echo "finished=$(date -Is)"
      } > "$RUN/PREFLIGHT_OK"
      say "PREFLIGHT GREEN -- ticket $RUN/PREFLIGHT_OK"
    else
      say "PREFLIGHT RED -- $failed stage(s) failed, hard stop $HARD_STOP; no launch ticket"
    fi
  fi
  [ "$HARD_STOP" -eq 1 ] && { say "HARD STOP -- see above"; say "DONE -- read $STATUS"; return 2; }
  say "DONE -- read $STATUS"
  [ "$failed" -eq 0 ]
}

# ----------------------------------------------------------- launch ticket ---
find_ticket() {
  local heads digest t
  heads="heads=$(git_heads | tr ' ' ',')"; digest="digest=$(tree_digest)"
  for t in $(ls -1t "$SIM_DIR"/runs/*/PREFLIGHT_OK 2>/dev/null); do
    grep -qx "$heads" "$t" && grep -qx "$digest" "$t" && grep -qx "variants=${VARIANT_LIST[*]}" "$t" \
      && { dirname "$t"; return 0; }
  done
  return 1
}

write_timeouts() {
  local pre="$1" out="$2" tag plan pre_id stem first=1
  plan=$(mktemp); pre_id=$(basename "$pre")
  {
    echo "{\"preflight\": {\"trials\": 1, \"top\": 1, \"models\": ${#MODELS[@]}},"
    echo " \"epochs\": [${PRE_EPOCHS[0]}, ${PRE_EPOCHS[1]}],"
    echo " \"full\": {\"trials\": 48, \"screen\": 20, \"top\": 5, \"refine\": 300, \"epochs\": 300, \"models\": ${#MODELS[@]}},"
    echo " \"screen_rows\": $SCREEN_ROWS, \"cap_s\": $(( ${DEADLINE_HOURS:-96} * 3600 )), \"variants\": {"
    for tag in "${VARIANT_LIST[@]}"; do
      stem=$(stem_of "$tag")
      local abl=0; [[ " $ABLATE " == *" $tag "* ]] && abl=1
      [ $first -eq 1 ] || echo ","; first=0
      printf '  "%s": {"data_dir": "%s", "full_bags": %s, "dataset": "%s"}' "$tag" \
        "$SIM_DIR/data/identification/$pre_id/$tag" \
        "$( cd "$SIM_DIR" && "${HELP[@]}" fullbags "${CONFIG[$tag]}" "$abl" 2>/dev/null )" \
        "$SIM_DIR/data/identification/$pre_id/$tag/$stem.parquet"
    done
    echo '}}'
  } > "$plan"
  ( cd "$SIM_DIR" && "${HELP[@]}" timeouts6 "$pre" "$plan" ) > "$out"; local rc=$?
  rm -f "$plan"; return $rc
}

# ------------------------------------------------------------- supervisor ---
supervise() {
  local mode="$1" n=0 rc
  while [ $n -lt 6 ]; do
    n=$((n+1))
    echo "--- supervisor attempt $n $(date)"
    work "$mode"; rc=$?
    [ $rc -eq 0 ] && break
    [ $rc -eq 2 ] && { echo "hard stop, not retrying"; break; }
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
    exec 9>"$LOCK"
    flock -n 9 || { echo "refusing: another run holds $LOCK"; exit 1; }
    supervise full ;;
  launch)
    if ! flock -n "$LOCK" true; then echo "refusing to launch: another preflight/run holds $LOCK"; exit 1; fi
    if ! doctor launch; then echo "refusing to launch: doctor says NO-GO"; exit 1; fi
    ticket=$(find_ticket) || {
      echo "refusing to launch: no green preflight for $(git_heads), digest $(tree_digest),"
      echo "variants '${VARIANT_LIST[*]}'. Run: bash scripts/round6_run.sh preflight"
      exit 1; }
    export RUN_ID="${RUN_ID:-r6-full-$(date +%Y%m%d-%H%M)}"
    mkdir -p "$SIM_DIR/runs/$RUN_ID"
    write_timeouts "$ticket" "$SIM_DIR/runs/$RUN_ID/timeouts.tsv" \
      || { echo "refusing to launch: could not size stage timeouts from $ticket"; exit 1; }
    echo "ticket $ticket; stage timeouts:"; sed 's/^/  /' "$SIM_DIR/runs/$RUN_ID/timeouts.tsv"
    boot="$SIM_DIR/runs/$RUN_ID.boot.log"
    inhibit=(systemd-inhibit --what=sleep:idle:shutdown:handle-lid-switch
             --who=round6 --why="round-6 unattended run" --mode=block)
    export VARIANTS="${VARIANT_LIST[*]}"
    setsid nohup ionice -c2 -n0 "${inhibit[@]}" bash -c \
      'echo -500 > /proc/self/oom_score_adj 2>/dev/null; exec bash "$0" run' \
      "$SIM_DIR/scripts/round6_run.sh" \
      </dev/null >>"$boot" 2>&1 &
    disown
    echo "launched RUN_ID=$RUN_ID"
    echo "watch:  tail -f $SIM_DIR/runs/$RUN_ID/STATUS.md"
    echo "boot :  $boot"
    ;;
  *) echo "usage: $0 {doctor|preflight|launch}"; exit 2 ;;
esac
