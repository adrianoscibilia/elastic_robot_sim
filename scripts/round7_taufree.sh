#!/usr/bin/env bash
# Round-7 torque-free comparison (FR_04 S-5): no-payload data, the tau-free classes, the tau-informed reference.
#
#   bash scripts/round7_taufree.sh doctor                     # GO / NO-GO: suites, round-6 inputs, GPU fit, machine
#   bash scripts/round7_taufree.sh preflight                  # every stage on fmrr + ur10, 2 epochs, 1 seed, both
#                                                             # PARALLEL_SEEDS values, every class; writes the ticket
#   PARALLEL_SEEDS=3 bash scripts/round7_taufree.sh launch    # detached, sleep-inhibited, supervised full run
#
# Env: VARIANTS="fmrr ur10 iiwa_drive iiwa_bus" (default), PARALLEL_SEEDS=1 (default) or 3,
#      RUN_ID (default r7-taufree-YYYYMMDD-HHMM), DEADLINE_HOURS (default 120)
#      INTEGRATOR=trapezoidal (default) or backward_euler: the two-mass filter's discretisation (FR_04
#        implementation report); the preflight records it in the ticket and launch needs the same value
#
# Per variant:
#   10-data-<v>            build_round6.py --taufree on config/identification/*_round7_nopayload.yaml (production +
#                          gain shift, every round-6 hard check and gate), round7_checks.py against the round-6
#                          manifests (S-1 identity), round6_budget.py for documentation (no gate)
#   15-relabel-<v>         relabel_val.py, unchanged (the FR_03 val rule)
#   30-coupling            (preflight only) benchmark_twomass_coupling.py: full vs diagonal on every variant
#   35-chain-<v>           runs/<id>/chain-<v>-taufree.yaml: FR_04 Sec 3.5's classes x seeds [0, 1, 2],
#                          selection val, 300 epochs, patience 30, coupling from the preflight
#   40-train-<v>-s<seed>   one stage per seed
#   45-collect-<v>         the seeds' summaries -> chain-<v>-taufree.summary.csv
#   50-eval-<v>-test       evaluate_on, r7 production file, --split test --exclude-head-s <burn-in>
#   50-eval-<v>-gainshift  ... and its r7 gain-shift twin
# then once: 60-report     scripts/round7_taufree_report.py -> REFACTOR_SPECS/final_results/{data,figures}
#
# Resumable, one run at a time, ticket on heads + tree digest, timeouts from the preflight x 3: launch_lib.sh.
# launch also applies the 72 h rule (FR_04 Sec 3.5): above 72 h projected at PARALLEL_SEEDS=3 it drops the
# optional classes, then classes 7-8, and writes the decision to runs/<id>/classes.txt.
set -uo pipefail

SIM_DIR="${SIM_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
NN_DIR="${NN_DIR:-$(cd "$SIM_DIR/../dynamic_model_nn" && pwd)}"
CMD="${1:-doctor}"
LOCK="$SIM_DIR/runs/.launch.lock"
source "$SIM_DIR/scripts/launch_lib.sh"
STAGE_SIGNAL=TERM

R6_FULL="r6-full-20260929-0049"; R6_UR10="r6-ur10-20261005-1126"
declare -A SRC_RUN=([iiwa_drive]=$R6_FULL [iiwa_bus]=$R6_FULL [fmrr]=$R6_FULL [ur10]=$R6_UR10)
declare -A R6_STEM=([iiwa_drive]=kuka_lbr_iiwa_14_r820_table_round6_drive [iiwa_bus]=kuka_lbr_iiwa_14_r820_table_round6_bus
                    [ur10]=ur10_table_round6 [fmrr]=fmrr_tecnobody_round6)
declare -A STEM=([iiwa_drive]=kuka_lbr_iiwa_14_r820_table_drive_round7_nopayload
                 [iiwa_bus]=kuka_lbr_iiwa_14_r820_table_bus_round7_nopayload
                 [ur10]=ur10_table_round7_nopayload [fmrr]=fmrr_tecnobody_round7_nopayload)
declare -A VAL_ROBOTS=([iiwa_drive]=4 [iiwa_bus]=4 [ur10]=4 [fmrr]=3)
read -r -a VARIANT_LIST <<< "${VARIANTS:-fmrr ur10 iiwa_drive iiwa_bus}"
PARALLEL_SEEDS="${PARALLEL_SEEDS:-1}"
INTEGRATOR="${INTEGRATOR:-trapezoidal}"
SEEDS=(0 1 2); EPOCHS=300; PATIENCE=30
CLASSES_ALL="lnn_tau_res kalnn_tau_res lnn_twomass kalnn_twomass lnn_res_hist kalnn_res_hist lnn_elastic_qdq kalnn_elastic_qdq lnn_tau_elastic kalnn_tau_elastic"
PRE_VARIANTS=(fmrr ur10); PRE_EPOCHS=2
MIN_FREE_GB="${MIN_FREE_GB:-25}"
FULL_DEADLINE_HOURS="${FULL_DEADLINE_HOURS:-120}"
FULL_DATA_ROOT="data/identification/r7-nopayload"

config_of()  { echo "$SIM_DIR/config/identification/${STEM[$1]}.yaml"; }
r6_manifest() { echo "$SIM_DIR/data/identification/${SRC_RUN[$1]}/$1/${R6_STEM[$1]}$2.manifest.json"; }
src_chain()  { echo "$SIM_DIR/runs/${SRC_RUN[$1]}/chain-$1.yaml"; }
r7_file()    { echo "$SIM_DIR/$2/$1/${STEM[$1]}$3.parquet"; }   # r7_file VARIANT DATA_ROOT SUFFIX
r7_contract() { echo "$SIM_DIR/$2/$1/${STEM[$1]}.contract.json"; }   # r7_contract VARIANT DATA_ROOT
latest_preflight() { ls -1td "$SIM_DIR"/runs/r7-taufree-preflight-*/ 2>/dev/null | head -1 | sed 's:/$::'; }

# ---------------------------------------------------------------- doctor ----
doctor() {
  local context="${1:-check}" rc=0 tag
  ok()   { echo "  PASS  $*"; }
  bad()  { echo "  FAIL  $*"; rc=1; }
  warn() { echo "  warn  $*"; }

  echo "== interpreters"
  echo "  sim: ${SIM_PY[*]}"; echo "  nn : ${NN_PY[*]}"
  "${SIM_PY[@]}" -c 'import pinocchio,numpy,pandas,pyarrow,yaml,matplotlib,scipy' 2>/dev/null \
    && ok "sim imports (pinocchio, pyarrow, yaml, matplotlib, scipy)" || bad "sim imports"
  "${NN_PY[@]}" -c 'import torch,yaml,pyarrow,scipy' 2>/dev/null && ok "nn imports (torch, yaml, pyarrow, scipy)" || bad "nn imports"
  "${NN_PY[@]}" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null \
    && ok "cuda: $("${NN_PY[@]}" -c 'import torch;print(torch.cuda.get_device_name(0))' 2>/dev/null)" \
    || bad "no CUDA device"

  echo "== test suites (fast)"
  if [ "$context" != quick ]; then
    local sim_test=("${SIM_PY[@]}") nn_test=("${NN_PY[@]}")
    if command -v uv >/dev/null 2>&1; then
      sim_test=(uv run --project "$SIM_DIR" --with pytest python); nn_test=(uv run --project "$NN_DIR" --with pytest python)
    fi
    ( cd "$SIM_DIR" && timeout 30m "${sim_test[@]}" -m pytest -q -m "not slow" -p no:cacheprovider >/tmp/r7t_sim_tests.log 2>&1 ) \
      && ok "elastic_robot_sim: $(tail -1 /tmp/r7t_sim_tests.log)" || bad "elastic_robot_sim suite red: $(tail -1 /tmp/r7t_sim_tests.log)"
    ( cd "$NN_DIR" && timeout 30m "${nn_test[@]}" -m pytest -q -p no:cacheprovider test >/tmp/r7t_nn_tests.log 2>&1 ) \
      && ok "dynamic_model_nn: $(tail -1 /tmp/r7t_nn_tests.log)" || bad "dynamic_model_nn suite red: $(tail -1 /tmp/r7t_nn_tests.log)"
  fi

  echo "== entry points"
  local e
  for e in train_chain_from_config.py evaluate_on.py benchmark_twomass_coupling.py; do
    ( cd "$NN_DIR" && "${NN_PY[@]}" "$e" --help >/dev/null 2>&1 ) && ok "$e --help" || bad "$e --help"
  done
  for e in build_round6.py round7_checks.py relabel_val.py round6_budget.py round7_taufree_report.py; do
    ( cd "$SIM_DIR" && "${SIM_PY[@]}" "scripts/$e" --help >/dev/null 2>&1 ) && ok "$e --help" || bad "$e --help"
  done

  echo "== inputs: r7 configs, round-6 manifests (the S-1 reference) and chains"
  for tag in "${VARIANT_LIST[@]}"; do
    if [ -z "${STEM[$tag]:-}" ]; then bad "$tag: unknown variant"; continue; fi
    [ -f "$(config_of "$tag")" ] && ok "$tag: $(basename "$(config_of "$tag")")" || bad "$tag: $(config_of "$tag") missing"
    local x; for x in "" _gainshift; do
      [ -f "$(r6_manifest "$tag" "$x")" ] || bad "$tag: $(r6_manifest "$tag" "$x") missing"
    done
    [ -f "$(src_chain "$tag")" ] && ok "$tag: $(basename "$(src_chain "$tag")")" || bad "$tag: $(src_chain "$tag") missing"
  done

  echo "== parallel seeds"
  case "$PARALLEL_SEEDS" in
    1) ok "PARALLEL_SEEDS=1 (sequential)";;
    3) local pre; pre=$(latest_preflight)
       if [ "$context" = preflight ]; then
         ok "PARALLEL_SEEDS=3 (the preflight measures it)"
       elif [ -z "$pre" ]; then
         bad "PARALLEL_SEEDS=3 needs a taufree preflight's measured GPU peak"
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

# ---------------------------------------------------------- the data stage --
# data_stage VARIANT DATA_ROOT RUN: S-1 build (+ S-2..S-4 contract blocks), S-1 identity checks, r7 budget.
data_stage() {
  local tag="$1" root="$2" run="$3" x rc=0
  mkdir -p "$SIM_DIR/$root/$tag"
  "${SIM_PY[@]}" "$SIM_DIR/scripts/build_round6.py" --config "$(config_of "$tag")" --out-dir "$SIM_DIR/$root/$tag" \
    --taufree --quiet || return 1
  for x in "" _gainshift; do
    "${SIM_PY[@]}" "$SIM_DIR/scripts/round7_checks.py" --r6 "$(r6_manifest "$tag" "$x")" \
      --r7 "$SIM_DIR/$root/$tag/${STEM[$tag]}$x.manifest.json" --out "$run/s1-$tag$x.json" || rc=1
  done
  [ $rc -eq 0 ] || return 1
  # Documentation only (no gate change): the budget may report FAILS and still pass this stage.
  "${SIM_PY[@]}" "$SIM_DIR/scripts/round6_budget.py" --config "$(config_of "$tag")" --quiet \
    --out "$SIM_DIR/reports/round7/budget_${STEM[$tag]}" > "$run/budget-$tag.txt" 2>&1
  echo "round6_budget.py exit $? (documentation; not gated)" >> "$run/budget-$tag.txt"
  return 0
}

# ------------------------------------------------------------- the stages ---
train_seeds() {  # train_seeds VARIANT CHAIN STAGE_PREFIX SUMMARY_PREFIX PARALLEL EXTRA_ARGS... -- SEEDS...
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

evaluate() {  # evaluate STAGE SUMMARY DATASET OUT HEAD_S
  local name="$1" summary="$2" dataset="$3" out="$4" head="$5"
  [ -f "$RUN/$name.done" ] || rm -f "$out"
  stage "$name" "eval-${name#50-eval-}.log" \
    env -C "$NN_DIR" "${NN_PY[@]}" "$NN_DIR/evaluate_on.py" --chain-summary "$summary" --dataset "$dataset" \
      --split test --exclude-head-s "$head" --out "$out"
}

# every_class_trained SUMMARY: a chain stage is green only if every class of its chain produced a row.
every_class_trained() {
  local chain="$1" summary="$2"
  "${SIM_PY[@]}" - "$chain" "$summary" <<'PY'
import sys, yaml, pandas as pd
chain, summary = sys.argv[1:3]
wanted = {r["base_name"] for r in yaml.safe_load(open(chain))["runs"]}
got = set(pd.read_csv(summary)["base_name"]) if summary else set()
missing = sorted(wanted - got)
print(f"classes trained {len(wanted) - len(missing)}/{len(wanted)}" + (f"; missing {missing}" if missing else ""))
sys.exit(1 if missing else 0)
PY
}

work() {
  local MODE="$1"
  if [ "$MODE" = preflight ]; then
    RUN_ID="${RUN_ID:-r7-taufree-preflight-$(date +%Y%m%d-%H%M)}"
  else
    RUN_ID="${RUN_ID:-r7-taufree-$(date +%Y%m%d-%H%M)}"
  fi
  RUN="$SIM_DIR/runs/$RUN_ID"; LOGS="$RUN/logs"; mkdir -p "$LOGS"
  STATUS="$RUN/STATUS.md"; TIMEOUTS="$RUN/timeouts.tsv"; HARD_STOP=0
  local variants seeds epochs passes data_root classes coupling
  if [ "$MODE" = preflight ]; then
    variants=("${PRE_VARIANTS[@]}"); seeds=(0); epochs=$PRE_EPOCHS; passes=(1 3)
    data_root="data/identification/$RUN_ID"; classes="$CLASSES_ALL"
    DEADLINE_HOURS="${DEADLINE_HOURS:-6}"
  else
    variants=("${VARIANT_LIST[@]}"); seeds=("${SEEDS[@]}"); epochs=$EPOCHS; passes=("$PARALLEL_SEEDS")
    data_root="$FULL_DATA_ROOT"; classes="$(head -1 "$RUN/classes.txt" 2>/dev/null || echo "$CLASSES_ALL")"
    DEADLINE_HOURS="${DEADLINE_HOURS:-$FULL_DEADLINE_HOURS}"
  fi
  coupling="$(sed -n 's/^coupling=//p' "$RUN/coupling.env" 2>/dev/null)"
  STARTED=$(date +%s)

  cd "$SIM_DIR"
  say "=== $RUN_ID  variants=${variants[*]} mode=$MODE seeds=${seeds[*]} epochs=$epochs parallel=${passes[*]} pid=$$"
  say "sim [${SIM_PY[*]}]   nn [${NN_PY[*]}]   $(git_heads) digest $(tree_digest)"
  say "classes: $classes"

  local tag
  for tag in "${variants[@]}"; do
    stage "10-data-$tag" "data-$tag.log" bash "$SIM_DIR/scripts/round7_taufree.sh" data "$tag" "$data_root" "$RUN" || continue
    [ "$HARD_STOP" -eq 1 ] && break
  done

  if [ "$MODE" = preflight ]; then
    # FR_04 Sec 3.3: full vs diagonal on every variant (built ones from their contract, the others from round 6).
    local specs=() src
    for tag in "${VARIANT_LIST[@]}"; do
      src="$SIM_DIR/$data_root/$tag/${STEM[$tag]}.contract.json"; [ -f "$src" ] || src="$(r6_manifest "$tag" "")"
      specs+=(--spec "$tag:$(cd "$SIM_DIR" && "${HELP[@]}" taufree_spec "$src")")
    done
    stage "30-coupling" "coupling.log" env -C "$NN_DIR" "${NN_PY[@]}" "$NN_DIR/benchmark_twomass_coupling.py" \
      "${specs[@]}" --out "$RUN/coupling.json"
    coupling="$("${SIM_PY[@]}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["coupling"])' "$RUN/coupling.json" 2>/dev/null || echo diagonal)"
    echo "coupling=$coupling" > "$RUN/coupling.env"
  fi
  coupling="${coupling:-diagonal}"
  say "coupling: $coupling, integrator: $INTEGRATOR"

  local k relabelled chain p s done_seeds
  for tag in "${variants[@]}"; do
    [ -f "$RUN/10-data-$tag.done" ] || { say "SKIP  $tag -- no data"; continue; }
    k=${VAL_ROBOTS[$tag]}
    relabelled="$(r7_file "$tag" "$data_root" "_val$k")"
    chain="$RUN/chain-$tag-taufree.yaml"
    stage "15-relabel-$tag" "relabel-$tag.log" \
      "${SIM_PY[@]}" scripts/relabel_val.py --dataset "$(r7_file "$tag" "$data_root" "")" --variant "$tag" \
        --val-robots "$k" --out-root "$data_root" || continue
    [ "$HARD_STOP" -eq 1 ] && break
    stage "35-chain-$tag" "chain-$tag.log" \
      "${HELP[@]}" taufree_chain "$(src_chain "$tag")" "$chain" "$relabelled" \
        "$(r7_contract "$tag" "$data_root")" "$EPOCHS" "$PATIENCE" "$coupling:$INTEGRATOR" "$classes" "${SEEDS[@]}" || continue

    for p in "${passes[@]}"; do
      if [ "$MODE" = preflight ] && [ "$p" -gt 1 ]; then
        train_seeds "$tag" "$chain" "40-train-$tag-p3" "$RUN/chain-$tag-taufree-p3" 3 --epochs "$epochs" -- 0 1 2
      else
        local extra=(); [ "$MODE" = preflight ] && extra=(--epochs "$epochs")
        train_seeds "$tag" "$chain" "40-train-$tag" "$RUN/chain-$tag-taufree" "$p" "${extra[@]}" -- "${seeds[@]}"
      fi
    done
    [ "$HARD_STOP" -eq 1 ] && break
    if [ "$MODE" = preflight ]; then
      # "every class": a 2-epoch chain that lost a class is a red preflight, not a quiet gap.
      for s in "$RUN"/chain-"$tag"-taufree*-s*.summary.csv; do
        every_class_trained "$chain" "$s" >> "$LOGS/classes-$tag.log" 2>&1 \
          || { say "FAIL  $(basename "$s"): $(tail -1 "$LOGS/classes-$tag.log")"; touch "$RUN/41-classes-$tag.failed"; }
      done
    fi

    done_seeds=""
    for s in "${seeds[@]}"; do [ -f "$RUN/40-train-$tag-s$s.done" ] && done_seeds+="$s "; done
    if [ -z "$done_seeds" ]; then say "SKIP  collect/eval of $tag -- no seed trained"; continue; fi
    if [ "$(cat "$RUN/collect-$tag.seeds" 2>/dev/null)" != "$done_seeds" ]; then
      rm -f "$RUN/45-collect-$tag.done" "$RUN"/50-eval-"$tag"-*.done "$RUN/60-report.done"
      echo "$done_seeds" > "$RUN/collect-$tag.seeds"
    fi
    local inputs=(); for s in $done_seeds; do inputs+=("$RUN/chain-$tag-taufree-s$s.summary.csv"); done
    stage "45-collect-$tag" "collect-$tag.log" \
      "${HELP[@]}" merge_summaries "$RUN/chain-$tag-taufree.summary.csv" "${inputs[@]}" || continue
    local summary="$RUN/chain-$tag-taufree.summary.csv" x head
    head="$("${SIM_PY[@]}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["transmission_priors"]["burn_in_s"])' \
            "$(r7_contract "$tag" "$data_root")")"
    for x in test gainshift; do
      local file; if [ "$x" = test ]; then file=$(r7_file "$tag" "$data_root" ""); else file=$(r7_file "$tag" "$data_root" "_$x"); fi
      if [ ! -f "$file" ]; then say "SKIP  50-eval-$tag-$x -- $(basename "$file") missing"; continue; fi
      evaluate "50-eval-$tag-$x" "$summary" "$file" "$RUN/eval-$tag-$x.csv" "$head"
    done
  done

  if [ "$HARD_STOP" -eq 0 ]; then
    local report=(); [ "$MODE" = preflight ] && report=(--out-dir "$RUN/report")
    stage "60-report" "report.log" "${SIM_PY[@]}" scripts/round7_taufree_report.py --run "$RUN" --data-root "$data_root" "${report[@]}"
  fi

  if [ "$MODE" = preflight ] && [ "$HARD_STOP" -eq 0 ]; then
    write_plan "$RUN" "$RUN/plan.json" "${VARIANT_LIST[@]}"
    if ( cd "$SIM_DIR" && "${HELP[@]}" retrain_plan "$RUN" "$RUN/plan.json" 1 ) > "$RUN/timeouts-p1.preview.tsv" 2>>"$LOGS/plan.log" \
       && cp "$RUN/projection.json" "$RUN/projection-p1.json" \
       && ( cd "$SIM_DIR" && "${HELP[@]}" retrain_plan "$RUN" "$RUN/plan.json" 3 ) > "$RUN/timeouts-p3.preview.tsv" 2>>"$LOGS/plan.log"; then
      say "projection: $RUN/projection.json"
    else
      say "FAIL  projection -- logs/plan.log"; touch "$RUN/70-projection.failed"
    fi
  fi

  {
    echo; echo "## datasets written"; find "$SIM_DIR/$data_root" -name '*.parquet' -newermt "@$STARTED" -exec sha256sum {} \; 2>/dev/null
    echo; echo "## S-1 identity checks"; ls "$RUN"/s1-*.json 2>/dev/null
    echo; echo "## chain configs"; ls "$RUN"/chain-*-taufree.yaml 2>/dev/null
    echo; echo "## summaries and evaluations"; ls "$RUN"/chain-*.summary.csv "$RUN"/eval-*.csv 2>/dev/null
    echo; echo "## models written during this attempt"
    find "$NN_DIR/models" -newermt "@$STARTED" -type f 2>/dev/null | sed 's/^/- /'
    echo; echo "## stages"; ls "$RUN" | grep -E '\.(done|failed)$' | sed 's/^/- /'
    echo; echo "finished $(date)"
  } >> "$STATUS"

  local failed; failed=$(ls "$RUN" | grep -c '\.failed$')
  if [ "$MODE" = preflight ]; then
    if [ "$failed" -eq 0 ] && [ "$HARD_STOP" -eq 0 ]; then
      write_ticket "$RUN/PREFLIGHT_OK" "kind=taufree" "preflight_variants=${PRE_VARIANTS[*]}" "parallel_seeds_tested=1 3" \
        "coupling=$coupling" "integrator=$INTEGRATOR"
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
    echo "{\"summary_stem\": \"taufree\", \"epochs\": $EPOCHS, \"seeds\": [$(IFS=,; echo "${SEEDS[*]}")],"
    echo " \"cap_s\": $(( FULL_DEADLINE_HOURS * 3600 )),"
    printf ' "classes": [%s],\n' "$(printf '"%s",' ${CLASSES:-$CLASSES_ALL} | sed 's/,$//')"
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
      # Test rows of the round-6 production files (the r7 files have the same bags).
      printf '  "%s": "%s"' "$tag" "$SIM_DIR/data/identification/${SRC_RUN[$tag]}/$tag/${R6_STEM[$tag]}.parquet"
    done
    echo "}}"
  } > "$out"
}

mkdir -p "$SIM_DIR/runs"
case "$CMD" in
  doctor)    doctor ;;
  data)      shift; data_stage "$@" ;;
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
    ticket=$(find_ticket "kind=taufree" "integrator=$INTEGRATOR") || {
      echo "refusing to launch: no green taufree preflight (integrator=$INTEGRATOR) for $(git_heads), digest $(tree_digest)."
      echo "Run: bash scripts/round7_taufree.sh preflight"
      exit 1; }
    export RUN_ID="${RUN_ID:-r7-taufree-$(date +%Y%m%d-%H%M)}"
    mkdir -p "$SIM_DIR/runs/$RUN_ID"
    cp -f "$ticket/coupling.env" "$ticket/coupling.json" "$SIM_DIR/runs/$RUN_ID/" 2>/dev/null
    write_plan "$ticket" "$SIM_DIR/runs/$RUN_ID/plan.json" "${VARIANT_LIST[@]}"
    # FR_04 Sec 3.5: above 72 h projected at PARALLEL_SEEDS=3, drop the optional classes, then classes 7-8.
    ( cd "$SIM_DIR" && "${HELP[@]}" taufree_classes "$ticket" "$SIM_DIR/runs/$RUN_ID/plan.json" ) \
      > "$SIM_DIR/runs/$RUN_ID/classes.txt" || { echo "refusing to launch: the class plan failed"; exit 1; }
    ( cd "$SIM_DIR" && "${HELP[@]}" retrain_plan "$ticket" "$SIM_DIR/runs/$RUN_ID/plan.json" "$PARALLEL_SEEDS" ) \
      > "$SIM_DIR/runs/$RUN_ID/timeouts.tsv" \
      || { echo "refusing to launch: could not size stage timeouts from $ticket"; exit 1; }
    cp -f "$ticket/projection.json" "$SIM_DIR/runs/$RUN_ID/projection.json" 2>/dev/null
    echo "ticket $ticket; PARALLEL_SEEDS=$PARALLEL_SEEDS; $(cat "$SIM_DIR/runs/$RUN_ID/coupling.env" 2>/dev/null)"
    echo "classes: $(head -1 "$SIM_DIR/runs/$RUN_ID/classes.txt")"; sed -n '2,$p' "$SIM_DIR/runs/$RUN_ID/classes.txt"
    echo "stage timeouts:"; sed 's/^/  /' "$SIM_DIR/runs/$RUN_ID/timeouts.tsv"
    export VARIANTS="${VARIANT_LIST[*]}" PARALLEL_SEEDS INTEGRATOR
    launch_detached "$SIM_DIR/scripts/round7_taufree.sh" round7_taufree "round-7 torque-free comparison (FR_04)"
    ;;
  *) echo "usage: $0 {doctor|preflight|launch}"; exit 2 ;;
esac
