#!/usr/bin/env bash
# identified_filter/01 Sec 6.2-6.3: every final r8 checkpoint on the old real files and on the new ones of
# REFACTOR_SPECS/identified_filter/real_files.yaml (test: scored zero-shot; modal_reference: tools/modal_fit.py
# vs the models' zero-shot K_hat, D_hat and diag M_phi).  Idempotent: re-run whenever a file is added.
#
#   bash scripts/run_real_eval.sh                 # every final checkpoint of runs/r8-campaign
#   RUN_ID=r8-campaign bash scripts/run_real_eval.sh --dry-run
set -euo pipefail
SIM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec uv run --project "$SIM_DIR" python "$SIM_DIR/scripts/r8_real_eval.py" --run-id "${RUN_ID:-r8-campaign}" "$@"
