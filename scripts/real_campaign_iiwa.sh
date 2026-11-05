#!/usr/bin/env bash
# RR_23/RR_24: the whole real iiwa campaign (P0, P1 j100, P2, TE sweeps, P1 j150) in one go, on peepo.
# Driver already running in another terminal with lab_config:=$ERD_LAB/iiwa_probe.yaml
# (iiwa_probe_j150.yaml has the same abort.tracking_rad, so the same driver serves both).
#   scripts/real_campaign_iiwa.sh [--from N] [--plan-only] [--confirm]
# Stops at the first failing stage or quick-look STOP; resume with --from N.
H="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/real_campaign_lib.sh
source "$H/real_campaign_lib.sh"
parse_args "$@"

ROBOT_DIR=kuka_lbr_iiwa_14_r820; RECORDER=record_iiwa
P=$ERD_LAB/iiwa_probe.yaml
J=$ERD_LAB/iiwa_probe_j150.yaml
R=$RECORDER

step 1 $R $P iiwa_standstill_02   plan preflight standstill quicklook
step 2 $R $P iiwa_probe_l025      "plan --ladder 0.25" preflight "run --ladder 0.25" convert report quicklook
step 3 $R $P iiwa_probe_all_01    plan preflight run convert report "quicklook --previous iiwa_probe_l025"
step 4 $R $P iiwa_modal_r050      "plan --modal --rung 0.5 --repeats 1" preflight run convert report quicklook
step 5 $R $P iiwa_modal_01        "plan --modal --rung 1.0 --modal-previous iiwa_modal_r050" preflight run convert report "quicklook --previous iiwa_modal_r050"
step 6 $R $P iiwa_te_sweeps_01    "plan --te-sweeps" preflight run convert report quicklook
step 7 $R $J iiwa_probe_j150_l050 "plan --ladder 0.5" preflight "run --ladder 0.5" convert report "quicklook --previous iiwa_probe_all_01"
step 8 $R $J iiwa_probe_j150_01   plan preflight run convert report "quicklook --previous iiwa_probe_j150_l050"

finish iiwa_standstill_02:calibration iiwa_probe_all_01:test iiwa_modal_01:modal_reference \
       iiwa_te_sweeps_01:calibration iiwa_probe_j150_01:test
