#!/usr/bin/env bash
# RR_23/RR_24: the whole real UR10 campaign (mount check, P0, P1, P2, lean_fwd, TE sweeps) in one go, on the laptop.
# Driver already running in another terminal with lab_config:=$ERD_LAB/ur10_probe.yaml, pendant program on Play,
# speed slider 100 %, normal mode, robot at home. Every converting run does `identify` first (RR_24 S1).
#   scripts/real_campaign_ur10.sh [--from N] [--plan-only] [--confirm]
# Needs data/real_robot/ur10_cb3/20261021/ur_B_all_02 on this machine (the mount-check reference).
# Stops at the first failing stage or quick-look STOP; resume with --from N.
H="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/real_campaign_lib.sh
source "$H/real_campaign_lib.sh"
parse_args "$@"

ROBOT_DIR=ur10_cb3; RECORDER=record_ur10
C=$ERD_LAB/ur10_probe.yaml
R=$RECORDER
M=ur_mount_check_01

step 1 $R $C $M                  plan preflight identify "quicklook --mount-check --reference ur_B_all_02"
step 2 $R $C ur_standstill_02    plan preflight standstill quicklook
step 3 $R $C ur_probe_l025       "plan --ladder 0.25" preflight identify "run --ladder 0.25" convert report "quicklook --reference $M"
step 4 $R $C ur_probe_all_01     plan preflight identify run convert report "quicklook --previous ur_probe_l025 --reference $M"
step 5 $R $C ur_modal_r050       "plan --modal --rung 0.5 --repeats 1" preflight identify run convert report "quicklook --reference $M"
step 6 $R $C ur_modal_01         "plan --modal --rung 1.0 --modal-previous ur_modal_r050" preflight identify run convert report "quicklook --previous ur_modal_r050 --reference $M"
step 7 $R $C ur_modal_leanfwd_01 "plan --modal --rung 0.5 --modal-only shoulder_lift_joint:lean_fwd --repeats 3 --modal-previous ur_modal_r050" preflight identify run convert report "quicklook --reference $M"
step 8 $R $C ur_te_sweeps_01     "plan --te-sweeps" preflight identify run convert report "quicklook --reference $M"

finish $M:calibration ur_standstill_02:calibration ur_probe_all_01:test ur_modal_01:modal_reference \
       ur_modal_leanfwd_01:modal_reference ur_te_sweeps_01:calibration
echo "Then copy the data to peepo: rsync -av --partial data/real_robot/ peepo:Projects/elastic_robot_sim/data/real_robot/"
