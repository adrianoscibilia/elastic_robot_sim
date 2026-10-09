#!/usr/bin/env bash
# Round-6 retrain pass (FR_03 T-5): training only, on the round-6 datasets and hyperparameters.
#
#   bash scripts/round6_retrain.sh doctor     # GO / NO-GO: suites, inputs (FR_01 hashes), GPU fit, machine
#   bash scripts/round6_retrain.sh preflight  # every stage tiny (fmrr + ur10, 2 epochs, 1 seed, both
#                                             # PARALLEL_SEEDS values); writes the ticket
#   bash scripts/round6_retrain.sh launch     # detached, sleep-inhibited, supervised full run
#
# Env: VARIANTS="fmrr ur10 iiwa_drive iiwa_bus" (default, fast first)
#      PARALLEL_SEEDS=1 (default) or 3: the three seeds of a variant as concurrent processes;
#        doctor refuses 3 unless 3 x the largest measured model peak fits in free GPU memory
#      RUN_ID (default r6-retrain-YYYYMMDD-HHMM), DEADLINE_HOURS (default 120)
#
# Per variant:
#   15-relabel-<v>         scripts/relabel_val.py: val 2 -> 4 robots (FMRR 2 -> 3), test unchanged (T-3)
#   35-chain-<v>           runs/<id>/chain-<v>-retrain.yaml: the round-6 chain's 4 classes + 2 residual
#                          + tau_cmd twins (T-4), seeds [0, 1, 2], selection val, 300 epochs, patience 30
#   40-train-<v>-s<seed>   one stage per seed (a crash costs one seed, not the variant)
#   45-collect-<v>         the seeds' summaries -> chain-<v>-retrain.summary.csv
#   50-eval-<v>-test       evaluate_on, ORIGINAL production file, --split test
#   50-eval-<v>-gainshift  ... and its gain-shift twin (iiwa_drive also -ablation_off, -ablation_clean)
# then once: 60-report     scripts/round6_retrain_report.py -> REFACTOR_SPECS/final_results/{data,figures}
#
# Everything is resumable: runs/<id>/<stage>.done is skipped on re-run; `launch`
# again with the same RUN_ID after a power cut.  A seed that finishes on a later
# attempt re-opens its variant's collect/eval stages and the report.  One run at a
# time (flock on runs/.launch.lock); `launch` needs a green retrain preflight
# ticket for the current git heads and tree digest, and sizes its stage timeouts
# from that preflight x 3.  Machinery shared with round6_run.sh: scripts/launch_lib.sh.
set -uo pipefail

SIM_DIR="${SIM_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
NN_DIR="${NN_DIR:-$(cd "$SIM_DIR/../dynamic_model_nn" && pwd)}"
CMD="${1:-doctor}"
LOCK="$SIM_DIR/runs/.launch.lock"
source "$SIM_DIR/scripts/launch_lib.sh"
# Training does not catch SIGINT, and a backgrounded process may ignore it: stop over-time stages with TERM.
STAGE_SIGNAL=TERM

R6_FULL="r6-full-20260929-0049"; R6_UR10="r6-ur10-20261005-1126"
declare -A SRC_RUN=([iiwa_drive]=$R6_FULL [iiwa_bus]=$R6_FULL [fmrr]=$R6_FULL [ur10]=$R6_UR10)
declare -A STEM=([iiwa_drive]=kuka_lbr_iiwa_14_r820_table_round6_drive [iiwa_bus]=kuka_lbr_iiwa_14_r820_table_round6_bus
                 [ur10]=ur10_table_round6 [fmrr]=fmrr_tecnobody_round6)
# FR_01 Sec 2: the production files this pass must train on.
declare -A SHA=(
  [iiwa_drive]=e98b7a0b1a6bc95b9e36916ab501f40137d0a33766451b29cd51f0a2bdc6bad2
  [iiwa_bus]=1d2f0c2c6c23989bebc78a5e67c8490051af8f4ed661b05c61ca7eda9308aa2f
  [ur10]=4e39c90784b02f2e1d1393af72f671cf01f4356f7cfb54c7e6bfed8ad98ba33e
  [fmrr]=51ca1ae3dead141145829c4b782ea75e1f81e7afcd09c794ec32238f0873d481)
declare -A VAL_ROBOTS=([iiwa_drive]=4 [iiwa_bus]=4 [ur10]=4 [fmrr]=3)
ABLATE="iiwa_drive"
read -r -a VARIANT_LIST <<< "${VARIANTS:-fmrr ur10 iiwa_drive iiwa_bus}"
PARALLEL_SEEDS="${PARALLEL_SEEDS:-1}"
SEEDS=(0 1 2); EPOCHS=300; PATIENCE=30
PRE_VARIANTS=(fmrr ur10); PRE_EPOCHS=2
MIN_FREE_GB="${MIN_FREE_GB:-20}"
FULL_DEADLINE_HOURS="${FULL_DEADLINE_HOURS:-120}"

src_file()  { echo "$SIM_DIR/data/identification/${SRC_RUN[$1]}/$1/${STEM[$1]}$2.parquet"; }
src_chain() { echo "$SIM_DIR/runs/${SRC_RUN[$1]}/chain-$1.yaml"; }
latest_preflight() { ls -1td "$SIM_DIR"/runs/r6-retrain-preflight-*/ 2>/dev/null | head -1 | sed 's:/$::'; }

# ---------------------------------------------------------------- doctor ----
doctor() {
  local context="${1:-check}" rc=0 tag
  ok()   { echo "  PASS  $*"; }
  bad()  { echo "  FAIL  $*"; rc=1; }
  warn() { echo "  warn  $*"; }

  echo "== interpreters"
  echo "  sim: ${SIM_PY[*]}"; echo "  nn : ${NN_PY[*]}"
  "${SIM_PY[@]}" -c 'import pinocchio,numpy,pandas,pyarrow,yaml,matplotlib' 2>/dev/null \
    && ok "sim imports (pinocchio, pyarrow, yaml, matplotlib)" || bad "sim imports"
  "${NN_PY[@]}" -c 'import torch,yaml,pyarrow' 2>/dev/null && ok "nn imports (torch, yaml, pyarrow)" || bad "nn imports"
  "${NN_PY[@]}" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null \
    && ok "cuda: $("${NN_PY[@]}" -c 'import torch;print(torch.cuda.get_device_name(0))' 2>/dev/null)" \
    || bad "no CUDA device: the budget (FR_03 Sec 3) assumes the GPU"

  echo "== test suites (fast)"
  if [ "$context" != quick ]; then
    local sim_test=("${SIM_PY[@]}") nn_test=("${NN_PY[@]}")
    if command -v uv >/dev/null 2>&1; then
      sim_test=(uv run --project "$SIM_DIR" --with pytest python); nn_test=(uv run --project "$NN_DIR" --with pytest python)
    fi
    ( cd "$SIM_DIR" && timeout 30m "${sim_test[@]}" -m pytest -q -m "not slow" -p no:cacheprovider >/tmp/r6r_sim_tests.log 2>&1 ) \
      && ok "elastic_robot_sim: $(tail -1 /tmp/r6r_sim_tests.log)" || bad "elastic_robot_sim suite red: $(tail -1 /tmp/r6r_sim_tests.log)"
    ( cd "$NN_DIR" && timeout 30m "${nn_test[@]}" -m pytest -q -p no:cacheprovider test >/tmp/r6r_nn_tests.log 2>&1 ) \
      && ok "dynamic_model_nn: $(tail -1 /tmp/r6r_nn_tests.log)" || bad "dynamic_model_nn suite red: $(tail -1 /tmp/r6r_nn_tests.log)"
  fi

  echo "== entry points"
  local e
  for e in train_chain_from_config.py evaluate_on.py; do
    ( cd "$NN_DIR" && "${NN_PY[@]}" "$e" --help >/dev/null 2>&1 ) && ok "$e --help" || bad "$e --help"
  done
  for e in relabel_val.py round6_retrain_report.py; do
    ( cd "$SIM_DIR" && "${SIM_PY[@]}" "scripts/$e" --help >/dev/null 2>&1 ) && ok "$e --help" || bad "$e --help"
  done

  echo "== inputs: round-6 files (FR_01 Sec 2 hashes) and chains"
  for tag in "${VARIANT_LIST[@]}"; do
    if [ -z "${STEM[$tag]:-}" ]; then bad "$tag: unknown variant"; continue; fi
    local f; f=$(src_file "$tag" "")
    if [ ! -f "$f" ]; then bad "$tag: $f missing"; continue; fi
    [ "$(sha256sum "$f" | cut -c1-64)" = "${SHA[$tag]}" ] && ok "$tag: $(basename "$f") sha256 ${SHA[$tag]:0:12} (FR_01)" \
      || bad "$tag: $(basename "$f") sha256 differs from FR_01"
    local extra=(_gainshift); [[ " $ABLATE " == *" $tag "* ]] && extra+=(_ablation_off _ablation_clean)
    local x; for x in "${extra[@]}"; do [ -f "$(src_file "$tag" "$x")" ] || bad "$tag: ${STEM[$tag]}$x.parquet missing"; done
    [ -f "$(src_chain "$tag")" ] && ok "$tag: $(basename "$(src_chain "$tag")")" || bad "$tag: $(src_chain "$tag") missing"
  done

  echo "== parallel seeds"
  case "$PARALLEL_SEEDS" in
    1) ok "PARALLEL_SEEDS=1 (sequential)";;
    3) local pre; pre=$(latest_preflight)
       if [ "$context" = preflight ]; then
         ok "PARALLEL_SEEDS=3 (the preflight measures it)"
       elif [ -z "$pre" ]; then
         bad "PARALLEL_SEEDS=3 needs a retrain preflight's measured GPU peak"
       else
         local line; line=$( cd "$SIM_DIR" && "${HELP[@]}" gpu_fit "$pre" 3 2>&1 | tail -1 )
         case "$line" in PASS*) ok "${line#* }";; *) bad "${line#* }";; esac
       fi;;
    *) bad "PARALLEL_SEEDS=$PARALLEL_SEEDS: use 1 or 3";;
  esac

  echo "== machine"
  doctor_machine "$context" "$MIN_FREE_GB"
  echo "  info  $(git_heads), tree digest $(tree_digest)"

  echo
  if [ $rc -eq 0 ]; then echo "GO"; else echo "NO-GO -- fix the FAIL lines above"; fi
  return $rc
}

# ------------------------------------------------------------- the stages ---
# train_seeds VARIANT CHAIN STAGE_PREFIX SUMMARY_PREFIX PARALLEL EXTRA_ARGS... -- SEEDS...
train_seeds() {
  local tag="$1" chain="$2" prefix="$3" sprefix="$4" parallel="$5"; shift 5
  local extra=() seeds=() s rc=0 pids=()
  while [ $# -gt 0 ] && [ "$1" != -- ]; do extra+=("$1"); shift; done
  shift; seeds=("$@")
  for s in "${seeds[@]}"; do
    local cmd=(env -C "$NN_DIR" "${NN_PY[@]}" "$NN_DIR/train_chain_from_config.py" --config "$chain"
               --seeds "$s" --summary-out "${sprefix}-s$s.summary.csv" "${extra[@]}")
    if [ "$parallel" -gt 1 ]; then
      stage "$prefix-s$s" "${prefix#40-}-s$s.log" "${cmd[@]}" &
      pids+=($!)
    else
      stage "$prefix-s$s" "${prefix#40-}-s$s.log" "${cmd[@]}" || rc=1
    fi
  done
  for s in "${pids[@]}"; do wait "$s" || rc=1; done
  return $rc
}

evaluate() {  # evaluate STAGE SUMMARY DATASET OUT
  local name="$1" summary="$2" dataset="$3" out="$4"
  [ -f "$RUN/$name.done" ] || rm -f "$out"   # a retried stage must not append twice
  stage "$name" "eval-${name#50-eval-}.log" \
    env -C "$NN_DIR" "${NN_PY[@]}" "$NN_DIR/evaluate_on.py" --chain-summary "$summary" --dataset "$dataset" \
      --split test --out "$out"
}

work() {
  local MODE="$1"
  if [ "$MODE" = preflight ]; then
    RUN_ID="${RUN_ID:-r6-retrain-preflight-$(date +%Y%m%d-%H%M)}"
  else
    RUN_ID="${RUN_ID:-r6-retrain-$(date +%Y%m%d-%H%M)}"
  fi
  RUN="$SIM_DIR/runs/$RUN_ID"; LOGS="$RUN/logs"; mkdir -p "$LOGS"
  STATUS="$RUN/STATUS.md"; TIMEOUTS="$RUN/timeouts.tsv"; HARD_STOP=0
  local variants seeds epochs passes data_root
  if [ "$MODE" = preflight ]; then
    variants=("${PRE_VARIANTS[@]}"); seeds=(0); epochs=$PRE_EPOCHS; passes=(1 3)
    data_root="data/identification/$RUN_ID"
    DEADLINE_HOURS="${DEADLINE_HOURS:-4}"
  else
    variants=("${VARIANT_LIST[@]}"); seeds=("${SEEDS[@]}"); epochs=$EPOCHS; passes=("$PARALLEL_SEEDS")
    data_root="data/identification/r6-retrain"
    DEADLINE_HOURS="${DEADLINE_HOURS:-$FULL_DEADLINE_HOURS}"
  fi
  STARTED=$(date +%s)

  cd "$SIM_DIR"
  say "=== $RUN_ID  variants=${variants[*]} mode=$MODE seeds=${seeds[*]} epochs=$epochs parallel=${passes[*]} pid=$$"
  say "sim [${SIM_PY[*]}]   nn [${NN_PY[*]}]   $(git_heads) digest $(tree_digest)"

  local tag k relabelled chain p s done_seeds
  for tag in "${variants[@]}"; do
    k=${VAL_ROBOTS[$tag]}
    relabelled="$SIM_DIR/$data_root/$tag/${STEM[$tag]}_val$k.parquet"
    chain="$RUN/chain-$tag-retrain.yaml"
    stage "15-relabel-$tag" "relabel-$tag.log" \
      "${SIM_PY[@]}" scripts/relabel_val.py --dataset "$(src_file "$tag" "")" --variant "$tag" \
        --val-robots "$k" --out-root "$data_root" || continue
    [ "$HARD_STOP" -eq 1 ] && break
    stage "35-chain-$tag" "chain-$tag.log" \
      "${HELP[@]}" retrain_chain "$(src_chain "$tag")" "$chain" "$relabelled" "$EPOCHS" "$PATIENCE" "${SEEDS[@]}" || continue

    for p in "${passes[@]}"; do
      if [ "$MODE" = preflight ] && [ "$p" -gt 1 ]; then
        # Three concurrent seeds, to measure the parallel slowdown and the GPU peak.
        train_seeds "$tag" "$chain" "40-train-$tag-p3" "$RUN/chain-$tag-retrain-p3" 3 --epochs "$epochs" -- 0 1 2
      else
        local extra=(); [ "$MODE" = preflight ] && extra=(--epochs "$epochs")
        train_seeds "$tag" "$chain" "40-train-$tag" "$RUN/chain-$tag-retrain" "$p" "${extra[@]}" -- "${seeds[@]}"
      fi
    done
    [ "$HARD_STOP" -eq 1 ] && break

    # Collect and evaluate the seeds that finished; a seed finishing later re-opens these stages.
    done_seeds=""
    for s in "${seeds[@]}"; do [ -f "$RUN/40-train-$tag-s$s.done" ] && done_seeds+="$s "; done
    if [ -z "$done_seeds" ]; then say "SKIP  collect/eval of $tag -- no seed trained"; continue; fi
    if [ "$(cat "$RUN/collect-$tag.seeds" 2>/dev/null)" != "$done_seeds" ]; then
      rm -f "$RUN/45-collect-$tag.done" "$RUN"/50-eval-"$tag"-*.done "$RUN/60-report.done"
      echo "$done_seeds" > "$RUN/collect-$tag.seeds"
    fi
    local inputs=(); for s in $done_seeds; do inputs+=("$RUN/chain-$tag-retrain-s$s.summary.csv"); done
    stage "45-collect-$tag" "collect-$tag.log" \
      "${HELP[@]}" merge_summaries "$RUN/chain-$tag-retrain.summary.csv" "${inputs[@]}" || continue
    local summary="$RUN/chain-$tag-retrain.summary.csv" x
    local evals=(test gainshift); [[ " $ABLATE " == *" $tag "* ]] && evals+=(ablation_off ablation_clean)
    for x in "${evals[@]}"; do
      local file; if [ "$x" = test ]; then file=$(src_file "$tag" ""); else file=$(src_file "$tag" "_$x"); fi
      if [ ! -f "$file" ]; then say "SKIP  50-eval-$tag-$x -- $(basename "$file") missing"; continue; fi
      evaluate "50-eval-$tag-$x" "$summary" "$file" "$RUN/eval-$tag-$x.csv"
    done
  done

  if [ "$HARD_STOP" -eq 0 ]; then
    local report=(); [ "$MODE" = preflight ] && report=(--out-dir "$RUN/report")
    stage "60-report" "report.log" "${SIM_PY[@]}" scripts/round6_retrain_report.py --run "$RUN" "${report[@]}"
  fi

  if [ "$MODE" = preflight ] && [ "$HARD_STOP" -eq 0 ]; then
    # Sizing for launch: projection.json (per (variant, class) s/epoch, wall time for PARALLEL_SEEDS 1 and 3).
    write_plan "$RUN" "$RUN/plan.json" "${VARIANT_LIST[@]}"
    if ( cd "$SIM_DIR" && "${HELP[@]}" retrain_plan "$RUN" "$RUN/plan.json" 1 ) > "$RUN/timeouts-p1.preview.tsv" 2>>"$LOGS/plan.log" \
       && ( cd "$SIM_DIR" && "${HELP[@]}" retrain_plan "$RUN" "$RUN/plan.json" 3 ) > "$RUN/timeouts-p3.preview.tsv" 2>>"$LOGS/plan.log"; then
      say "projection: $RUN/projection.json"
    else
      say "FAIL  projection -- logs/plan.log"; touch "$RUN/70-projection.failed"
    fi
  fi

  {
    echo; echo "## relabelled datasets"; find "$SIM_DIR/$data_root" -name '*.parquet' -newermt "@$STARTED" -exec sha256sum {} \; 2>/dev/null
    echo; echo "## chain configs"; ls "$RUN"/chain-*-retrain.yaml 2>/dev/null
    echo; echo "## summaries and evaluations"; ls "$RUN"/chain-*.summary.csv "$RUN"/eval-*.csv 2>/dev/null
    echo; echo "## models written during this attempt"
    find "$NN_DIR/models" -newermt "@$STARTED" -type f 2>/dev/null | sed 's/^/- /'
    echo; echo "## stages"; ls "$RUN" | grep -E '\.(done|failed)$' | sed 's/^/- /'
    echo; echo "finished $(date)"
  } >> "$STATUS"

  local failed; failed=$(ls "$RUN" | grep -c '\.failed$')
  if [ "$MODE" = preflight ]; then
    if [ "$failed" -eq 0 ] && [ "$HARD_STOP" -eq 0 ]; then
      write_ticket "$RUN/PREFLIGHT_OK" "kind=retrain" "preflight_variants=${PRE_VARIANTS[*]}" "parallel_seeds_tested=1 3"
      say "PREFLIGHT GREEN -- ticket $RUN/PREFLIGHT_OK"
    else
      say "PREFLIGHT RED -- $failed stage(s) failed, hard stop $HARD_STOP; no launch ticket"
    fi
  fi
  [ "$HARD_STOP" -eq 1 ] && { say "HARD STOP -- see above"; say "DONE -- read $STATUS"; return 2; }
  say "DONE -- read $STATUS"
  [ "$failed" -eq 0 ]
}

# write_plan PREFLIGHT_RUN OUT VARIANT...: what launch_helpers retrain_plan sizes the full run from.
write_plan() {
  local pre="$1" out="$2"; shift 2
  local tag first=1
  {
    echo "{\"epochs\": $EPOCHS, \"seeds\": [$(IFS=,; echo "${SEEDS[*]}")], \"cap_s\": $(( ${DEADLINE_HOURS_FULL_RUN:-$FULL_DEADLINE_HOURS} * 3600 )),"
    printf ' "variants": [%s],\n' "$(printf '"%s",' "$@" | sed 's/,$//')"
    printf ' "preflight_variants": [%s],\n' "$(printf '"%s",' "${PRE_VARIANTS[@]}" | sed 's/,$//')"
    echo " \"reference\": {"
    for tag in "$@"; do
      [[ " ${PRE_VARIANTS[*]} " == *" $tag "* ]] && continue
      [ $first -eq 1 ] || echo ","; first=0
      # The arms the preflight does not train take the UR10's rates x round 6's per-epoch ratio.
      printf '  "%s": {"from": "ur10", "round6_run": "%s", "round6_from_run": "%s"}' \
        "$tag" "$SIM_DIR/runs/${SRC_RUN[$tag]}" "$SIM_DIR/runs/${SRC_RUN[ur10]}"
    done
    echo "},"
    first=1; echo " \"datasets\": {"
    for tag in "$@" "${PRE_VARIANTS[@]}"; do
      [ $first -eq 1 ] || echo ","; first=0
      printf '  "%s": "%s"' "$tag" "$(src_file "$tag" "")"
    done
    echo "}}"
  } > "$out"
}

mkdir -p "$SIM_DIR/runs"
case "$CMD" in
  doctor)    doctor ;;
  preflight)
    exec 9>"$LOCK"
    flock -n 9 || { echo "refusing: another preflight/run holds $LOCK"; exit 1; }
    HOLDING_LOCK=1
    doctor preflight && work preflight ;;
  run)
    exec 9>"$LOCK"
    flock -n 9 || { echo "refusing: another run holds $LOCK"; exit 1; }
    supervise full ;;
  launch)
    if ! flock -n "$LOCK" true; then echo "refusing to launch: another preflight/run holds $LOCK"; exit 1; fi
    if ! doctor launch; then echo "refusing to launch: doctor says NO-GO"; exit 1; fi
    ticket=$(find_ticket "kind=retrain") || {
      echo "refusing to launch: no green retrain preflight for $(git_heads), digest $(tree_digest)."
      echo "Run: bash scripts/round6_retrain.sh preflight"
      exit 1; }
    export RUN_ID="${RUN_ID:-r6-retrain-$(date +%Y%m%d-%H%M)}"
    mkdir -p "$SIM_DIR/runs/$RUN_ID"
    write_plan "$ticket" "$SIM_DIR/runs/$RUN_ID/plan.json" "${VARIANT_LIST[@]}"
    ( cd "$SIM_DIR" && "${HELP[@]}" retrain_plan "$ticket" "$SIM_DIR/runs/$RUN_ID/plan.json" "$PARALLEL_SEEDS" ) \
      > "$SIM_DIR/runs/$RUN_ID/timeouts.tsv" \
      || { echo "refusing to launch: could not size stage timeouts from $ticket"; exit 1; }
    cp -f "$ticket/projection.json" "$SIM_DIR/runs/$RUN_ID/projection.json" 2>/dev/null
    echo "ticket $ticket; PARALLEL_SEEDS=$PARALLEL_SEEDS; stage timeouts:"; sed 's/^/  /' "$SIM_DIR/runs/$RUN_ID/timeouts.tsv"
    export VARIANTS="${VARIANT_LIST[*]}" PARALLEL_SEEDS
    launch_detached "$SIM_DIR/scripts/round6_retrain.sh" round6_retrain "round-6 retrain pass (FR_03)"
    ;;
  *) echo "usage: $0 {doctor|preflight|launch}"; exit 2 ;;
esac
