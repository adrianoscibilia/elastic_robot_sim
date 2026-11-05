#!/usr/bin/env bash
# identified_filter/10 (2026-11-04): complete the r9 simulated table before any real run.
#
#   bash scripts/r9_complete.sh --launch   # the owner's one line: checks, then runs this script detached
#                                          # as the systemd --user unit r9-complete (survives the terminal)
#   bash scripts/r9_complete.sh            # the work itself, in the foreground (what the unit runs)
#
# 1. The training queue of config/campaigns/r9_complete.yaml (run_id r9-campaign): the finished r9 jobs are
#    skipped by their done.json; J12, J13, J14, J4, J15 train; then the campaign's own final sweep (real-file
#    eval of whatever exists, r8_summary.py -> results/r9/SUMMARY.md).
# 2. Zollo et al. A-pc on r9 (CPU, after the GPU queue so the two never compete for RAM), both platforms:
#    identification on the r9 train robots, then the bounded ID-elastic / ID-rigid on the r9 test robots.
#    Fallbacks, each logged: identification on the first 40 train robots if the full one fails; the evaluation
#    with --skip-gate if the harness gate fails (the gate result stays in the log).
# Progress: runs/r9-campaign/r9_complete.log; the end marker runs/r9-campaign/r9_complete.done.
# No commits. Writes only under runs/, reports/zollo/r9/ and REFACTOR_SPECS/identified_filter/results/r9/.
set -u
SIM="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SIM"
UNIT=r9-complete
CONFIG=config/campaigns/r9_complete.yaml
RUN=runs/r9-campaign
LOG="$RUN/r9_complete.log"
mkdir -p "$RUN"

if [ "${1:-}" = "--launch" ]; then
  for target in sleep.target suspend.target hibernate.target hybrid-sleep.target; do
    state="$(systemctl show -p LoadState --value "$target" 2>/dev/null)"
    if [ "$state" != "masked" ]; then
      echo "refusing to launch: $target is '$state', the machine may suspend mid-run. Mask them with:"
      echo "  sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target"
      exit 1
    fi
  done
  echo "sleep: the sleep targets are masked: OK"
  loginctl show-user "${USER:-$(id -un)}" -p Linger 2>/dev/null || true
  if systemctl --user is-active --quiet "$UNIT"; then
    echo "$UNIT is already running (journalctl --user -u $UNIT -f)"; exit 1
  fi
  rm -f "$RUN/r9_complete.done"
  systemd-run --user --unit="$UNIT" --collect --working-directory="$SIM" --property=KillSignal=SIGTERM \
    --setenv=PATH="$PATH" bash "$SIM/scripts/r9_complete.sh"
  code=$?
  echo "unit: $UNIT   progress: tail -f $SIM/$LOG   status: uv run python scripts/r8_campaign.py status --config $CONFIG"
  exit $code
fi

UV="$(command -v uv || echo uv)"
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

say "r9_complete start${INVOCATION_ID:+ (systemd unit $UNIT)}"

# 1. the training queue (+ the campaign's final sweep and summary)
"$UV" run python scripts/r8_campaign.py run --config "$CONFIG" >>"$RUN/r9_complete.queue.log" 2>&1
say "training queue exit $?"

# 2. Zollo A-pc on r9, simulation only
for platform in iiwa_drive ur10; do
  cfg="config/zollo/r9/$platform.yaml"
  out="reports/zollo/r9/$platform"
  mkdir -p "$out"
  ident="$out/identification_Apc_pooled.json"
  timeout 6h "$UV" run python scripts/zollo_identify.py --config "$cfg" --variant A \
      --transmission per_robot_transmission >"$out/identify.log" 2>&1
  code=$?
  say "zollo identify $platform (all train robots): exit $code"
  if [ $code -ne 0 ] || [ ! -f "$ident" ]; then
    timeout 4h "$UV" run python scripts/zollo_identify.py --config "$cfg" --variant A \
        --transmission per_robot_transmission --robots 40 --tag r40 >"$out/identify_r40.log" 2>&1
    code=$?
    ident="$out/identification_Apc_pooled_r40.json"
    say "zollo identify $platform (fallback, first 40 train robots): exit $code"
  fi
  if [ ! -f "$ident" ]; then
    say "zollo $platform: no identification, evaluation skipped"; continue
  fi
  timeout 6h "$UV" run python scripts/zollo_evaluate.py --config "$cfg" --identification "$ident" \
      --files sim >"$out/evaluate.log" 2>&1
  code=$?
  say "zollo evaluate $platform: exit $code"
  if [ $code -ne 0 ] && grep -q "does not reproduce the harness" "$out/evaluate.log"; then
    timeout 6h "$UV" run python scripts/zollo_evaluate.py --config "$cfg" --identification "$ident" \
        --files sim --skip-gate >"$out/evaluate_skipgate.log" 2>&1
    say "zollo evaluate $platform (--skip-gate, harness gate FAILED, see evaluate.log): exit $?"
  fi
done

say "r9_complete finished"
date '+%F %T' >"$RUN/r9_complete.done"
