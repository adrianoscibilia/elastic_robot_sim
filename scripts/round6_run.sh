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

# pick_py, SIM_PY/NN_PY/HELP, tree_digest, git_heads, stage, doctor_machine,
# the ticket, supervise and launch_detached (shared with round6_retrain.sh).
LOCK="$SIM_DIR/runs/.launch.lock"
source "$SIM_DIR/scripts/launch_lib.sh"

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
  doctor_machine "$context" "$MIN_FREE_GB"
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
      write_ticket "$RUN/PREFLIGHT_OK" "variants=${VARIANT_LIST[*]}"
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
    ticket=$(find_ticket "variants=${VARIANT_LIST[*]}") || {
      echo "refusing to launch: no green preflight for $(git_heads), digest $(tree_digest),"
      echo "variants '${VARIANT_LIST[*]}'. Run: bash scripts/round6_run.sh preflight"
      exit 1; }
    export RUN_ID="${RUN_ID:-r6-full-$(date +%Y%m%d-%H%M)}"
    mkdir -p "$SIM_DIR/runs/$RUN_ID"
    write_timeouts "$ticket" "$SIM_DIR/runs/$RUN_ID/timeouts.tsv" \
      || { echo "refusing to launch: could not size stage timeouts from $ticket"; exit 1; }
    echo "ticket $ticket; stage timeouts:"; sed 's/^/  /' "$SIM_DIR/runs/$RUN_ID/timeouts.tsv"
    export VARIANTS="${VARIANT_LIST[*]}"
    launch_detached "$SIM_DIR/scripts/round6_run.sh" round6 "round-6 unattended run"
    ;;
  *) echo "usage: $0 {doctor|preflight|launch}"; exit 2 ;;
esac
