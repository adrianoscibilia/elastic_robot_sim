# Operator notes

Covers Step 1 (L0-L2, everything T1.0-T1.10 exercised) and, per RR_04 C-3, the
Step-2 procedure the owner reads before T2.0: network practice, the
commissioning ladder, the E-* protocols and the stop/e-stop rules. This file
does not replace RR_01 (the full spec) or RR_05 (what this pass actually
verified); it is the short version to have open in the lab.

## Build and environment

One command, from the repository root, in every terminal (first time on a
new machine add `--deps`; see `src/ros2/README.md`):

```bash
source workspace_setup.sh          # --build | --clean | --test | --real
```

It builds `ros2_ws/` if needed and leaves the shell with ROS, the overlay and
`ros2_ws/.venv-erd` sourced, `ROS_DOMAIN_ID=87` (`--real`: 0), and
`$ERD_LAB` (lab configs) / `$ERD_DATA_ROOT` (recordings, `data/real_robot/`).
Two environment facts still hold underneath:

1. **pinocchio vs. ROS's sourced environment.** ROS's own
   `ros-jazzy-pinocchio` shadows `.venv-erd`'s `pin==4.1.0` (a numpy-ABI
   mismatch that segfaults). Every `erd_recording` entry point guards against
   it (`env_guard.assert_environment()`) and `plan`/`convert`/`identify` run
   as clean subprocesses. `erd_recording`'s pytest suite therefore runs
   without ROS sourced; `workspace_setup.sh --test` handles both invocations.
2. **Build with the venv's `python -m colcon build`** (the script does), so
   console-script shebangs point at `.venv-erd`. Verify with
   `head -1 $ERD_WS/install/erd_recording/lib/erd_recording/record_iiwa`.

## Laptop (Step 2: the machine at the robots)

The laptop records; peepo analyses (checkpoints and GPU live there). Same repo
layout on both: the laptop clone is `~/projects/elastic_robot_sim`, peepo's is
`~/Projects/elastic_robot_sim`. Below, `peepo` is an SSH host alias for
peepo; replace it with `user@host` if you have none. `~/projects/erd_ws` and
`~/projects/erd_data` are retired: nothing reads them any more.

**1. Once: clone, build, test.**

```bash
git clone https://github.com/adrianoscibilia/elastic_robot_sim.git ~/projects/elastic_robot_sim
cd ~/projects/elastic_robot_sim
git pull                                  # later updates: pull, then the line below again
source workspace_setup.sh --deps --test   # sudo for apt; ends with two "Summary: N tests, 0 errors, 0 failures"
```

Both test summaries must show `0 errors, 0 failures` (the ROS-sourced packages
also report 6 skipped cppcheck tests). Anything else: stop and send the output.

**2. Once: the two training reference contracts and their manifests.** Four
JSON files, ~1.2 MB, no parquet. `convert` refuses a reference whose sha256
differs from `consumer.reference_sha256` in the lab file:

```bash
cd ~/projects/elastic_robot_sim
for f in r6-full-20260929-0049/iiwa_drive/kuka_lbr_iiwa_14_r820_table_round6_drive \
         r6-ur10-20261005-1126/ur10/ur10_table_round6; do
  mkdir -p "data/identification/$(dirname "$f")"
  rsync -av "peepo:Projects/elastic_robot_sim/data/identification/$f.contract.json" \
            "peepo:Projects/elastic_robot_sim/data/identification/$f.manifest.json" \
            "data/identification/$(dirname "$f")/"
done
sha256sum data/identification/*/*/*.contract.json
# expect c64f7aef2e1e0f46...  kuka_lbr_iiwa_14_r820_table_round6_drive.contract.json
#        cc8867e909b0e980...  ur10_table_round6.contract.json
```

The checkpoints stay on peepo.

**3. Every robot terminal:**

```bash
source ~/projects/elastic_robot_sim/workspace_setup.sh --real   # ROS_DOMAIN_ID=0
```

`record_*` refuses a real lab file in a shell without `--real`, and a
`*_sim.yaml`/`*_mock.yaml` file in a shell with it.

**4. After every run: copy the run folder to peepo.** `convert`, `validate`
and `report` run on the laptop right after the run (to catch problems at the
robot) and again on peepo:

```bash
cd ~/projects/elastic_robot_sim
rsync -av --partial data/real_robot/ peepo:Projects/elastic_robot_sim/data/real_robot/
```

On peepo, the offline stages take the copied folder and the run's own frozen
config (one stage per call; `evaluate` only on peepo):

```bash
RUN=data/real_robot/<robot>/<date>/<run_id>
for stage in convert report evaluate; do
  ros2 run erd_recording record_iiwa --config $RUN/config.yaml --run-dir $RUN $stage   # record_ur10 for the UR10
done
```

## Measurement tools (RR_12 A-2)

* `ros2 run erd_recording erd_link_test --config <lab file> 300`: rung 1. No
  motion command; records 300 s and writes `link_test.json` in a run folder.
  iiwa: lost FRI cycles (`fri/cycle`), lost publication cycles
  (`fri/received_cycle`), stamp-step histogram, Sunrise connection quality,
  and `pass_rung1` (<= 5 lost cycles per 300 s, stamp-step p99 <= 2 ms). UR10:
  driver and sidecar rates, sidecar gaps, exact `actual_q` match fraction.
  Both: per-topic recorder losses.
* Every `standstill`/`identify`/`run` writes `monitor.json` (sample age
  p50/p99/max over all monitor evaluations, and every cancel). `report` adds
  the cancel latency of every abort, from the bag: the first sample over the
  threshold the cancel names, to the `cancel` event.
* `ros2 run erd_recording erd_monitor_test --config <sim lab file>`: L1/L2
  only (refuses `hardware: real`). 20 deliberate 0.15 rad / 0.3 s steps with
  the monitor threshold at 5 mrad; reports cancel latency and sample ages.
* Rung 4a on the real robot uses the normal pipeline instead:
  `record_iiwa ... run --ladder 0.1 --abort-tracking-rad <x>`. The override
  is refused above `--ladder 0.1`, lowers only the monitor's threshold (the
  JTC keeps the lab file's path tolerance), acts **inside excitation
  segments only** (approaches and returns keep the lab file's threshold),
  and is written into `manifest.yaml`.
* `<x>` (RR_14 P-3, amends RR_12 S4.6 rung 4a): 0.5 x the largest tracking
  error of the rung-4 `--ladder 0.1` run, and at least 4 encoder counts
  (iiwa 2.4e-7 rad, UR10 1.9e-6 rad). `report` on that run prints it:
  `report: suggested --abort-tracking-rad ...`, also in `report.md`.

## Preflight on real hardware (RR_12 A-3)

Besides RR_01 S5's items, `preflight` on `hardware: real` refuses when:
* the set of active controllers is not exactly the JTC plus the recording
  broadcasters (iiwa: `erd_arm_controller`, `joint_state_broadcaster`,
  `erd_state_broadcaster`; UR10: `scaled_joint_trajectory_controller`,
  `joint_state_broadcaster`, `speed_scaling_state_broadcaster`,
  `io_and_status_controller`, `ur_configuration_controller` (it serves
  the software version), `force_torque_sensor_broadcaster` (the joint-state
  broadcaster is chained to it)). The list goes into `preflight.json`.
  `ur10.launch.py` deactivates the three other controllers
  `ur_robot_driver` 3.9 activates (`gravity_update_controller`,
  `friction_model_controller`, `tcp_pose_broadcaster`) and prints
  `erd_ur10: deactivated ...`;
* UR10: the controller's software version differs from
  `connection.software_version`;
* `safety.vendor_checksum` is empty. It is copied into every run's
  `manifest.yaml` and the dataset contract.

At L1/L2 the same items are listed with `would_pass` but don't gate.

## Running a mock (L1) session

```bash
source workspace_setup.sh
ros2 launch erd_iiwa iiwa.launch.py hardware:=mock lab_config:=$ERD_LAB/iiwa_mock.yaml &
ros2 run erd_recording record_iiwa --config $ERD_LAB/iiwa_mock.yaml all --run-id demo
```

`lab_config:=` on the launch line generates the JTC's per-joint path/goal
tolerances from `limits.abort.tracking_rad` (RR_04 A-5) -- pass the same file
the `record_iiwa`/`record_ur10` command uses, or the tolerances silently stay
at the checked-in defaults instead of the lab's own abort radius.

Output lands in `$ERD_DATA_ROOT/<robot>/<date>/<run_id>/` (`recording.output_root`)
(`config.yaml`, `plan/`, `preflight.json`, `manifest.yaml`, `bag/`,
`bag_standstill/`, `raw/` logs). Kill any previous `ros2 launch`/
`ros2_control_node`/`robot_state_publisher` before relaunching -- a stale
`robot_state_publisher` still latching an old `/robot_description` is a real,
confusing failure mode (the new `ros2_control_node` silently loads the *old*
URDF).

## Pre-session check: the green L1+L2 set (RR_14 P-2)

Before every hardware session, run these four with **the exact code that
goes to the laptop** (one robot stack at a time, domain 87). Each must end
`status: synthetic` in `manifest.yaml` and `validation.json`:

```bash
record_iiwa --config $ERD_LAB/iiwa_mock.yaml all --no-confirm   # iiwa.launch.py hardware:=mock
record_iiwa --config $ERD_LAB/iiwa_sim.yaml  all --no-confirm   # + run_emulator.sh, hardware:=emulator
record_ur10 --config $ERD_LAB/ur10_mock.yaml all --no-confirm   # ur10.launch.py hardware:=mock
record_ur10 --config $ERD_LAB/ur10_sim.yaml  all --no-confirm   # + start_ursim.sh, hardware:=ursim, at home
```

`synthetic` means every check is `true`, or failed on the fixed
simulator-limits list (`validate.SIMULATOR_LIMITS`; `simulator_limit:` on the
check, listed under `simulator_limits`), or is the iiwa's allowed
`not_applicable` freshness:
* mock: `tau != ft`, `sign convention` (no torque channels),
  `identification freshness` (no temperatures; UR10);
* URSim: `tau != ft` (pure feed-forward current);
* emulator: nothing; the iiwa L2 `sign convention` must be true on A2 and A4
  (the sim poses give 52 / 31 N*m of gravity range).

The list is never consulted on `hardware: real`. A failing simulator run
ends `invalid`. Contracts written from mock/emulator/URSim data say
`"source": "synthetic_mock"` with `real.hardware`; `evaluate` refuses a
contract whose `source` and hardware disagree.

## The live safety monitor and stopping a run (RR_04 A-1)

While any goal is active (`run`, `standstill`, `identify`), the orchestrator
checks every `/dynamic_joint_states` tick against: tracking error vs. the
JTC's own reference, torque fraction (iiwa), FRI session state (iiwa), speed
scaling (UR10), robot/safety mode (UR10), sample staleness (>50 ms), and
whether the bag or RTDE sidecar process has exited. The first triggered
condition cancels the goal, waits for a terminal action status *and* 0.5 s of
stationary samples, then marks the run `failed` with the reason. If either
of those two confirmations doesn't complete within `limits.abort.
stop_timeout_s`, the log says **`STOP NOT CONFIRMED -- use the e-stop`** and
the run is left `failed` regardless -- treat that message as a real stop
request to the operator, not a software bug to retry past.

**Ctrl-C** (SIGINT) and SIGTERM route into the same cancel path (a handler
sets a flag the monitor loop checks every tick, since a signal handler
itself must not touch `rclpy` directly). Expect the same cancel/confirm/
`failed` sequence, not a silent process kill.

## Commissioning ladder (RR_01 S7.2), operator present, e-stop in hand

1. **No motion**, 60 s: iiwa in FRI MONITORING (timestamp regularity, lost
   cycles, Sunrise quality events); UR: driver + sidecar only.
2. Standstill poses, 30 s each (`standstill` stage).
3. Approaches home <-> standstill only.
4. Excitation at `--ladder 0.1`, then `--ladder 0.25` (iiwa first, then UR),
   probe off. `--ladder <scale>` re-validates the amplitude-scaled path
   before running it and tags the run `commissioning` -- it is never written
   into a dataset.
5. `identify` (UR sweeps and E-ur-1; iiwa E-iiwa-1). E-ur-4 once, at
   standstill, with the dial gauge.
6. `--ladder 0.5`, then a full-amplitude run (no `--ladder`), probe off.
7. Probe on (if configured), full amplitude.

**Stop, report to the owner, and wait** at any: a tracking abort, a
protective stop, an FRI drop, a speed-scaling dip, or torque above
`abort.torque_fraction`. Only step 6+ runs may enter a dataset.

## E-* protocols, in one line each (RR_01 S2.1; full detail there)

* **E-iiwa-1** (commanded-torque semantics): standstill + one excitation at
  amplitude 0.25; compares `tau_cmd - tau_meas` against SG `q_ddot` and
  Pinocchio RNEA. Feeds the CR-1 verdict.
* **E-iiwa-2** (torque sign/scale): standstill poses, fit `tau_meas = s *
  g(q) + o`; expect `s ~= +1 +/- 0.1`.
* **E-ur-1** (which current is the command, and `K_tau`): `K_tau` from
  `target_moment ~ target_current`; a feedback regression on
  `joint_control_output - target_current` picks `joint_control_output` or
  `actual_current` as the command (RR_01 v3.1's corrected tree, RR_04 A-10).
* **E-ur-2** (optional): the same trajectory at two `servoj_gain` values.
* **E-ur-3** (motor-side identification): speed sweeps for friction, then a
  joint fit of `J_m`/friction on the excitation data. A joint whose `J_m` is
  in the augmented design's null space (SVD check, RR_04 A-10) is fixed at
  the simulation's prior instead of fit.
* **E-ur-4** (encoder side): a dial gauge at a fixed pose, a 5 kg mass hung
  and removed at the flange, comparing the gauge against `FK(actual_q)` and
  `actual_q - target_q`.

## Stop and e-stop rules (RR_01 S9)

* The plan is validated offline and frozen before any motion; `run`/
  `identify`/`standstill` execute only digested segments.
* The discrete collision check protects paths, not people -- **the cell must
  be clear** before any motion, ladder step 1 included.
* E-ur-4 hangs a mass on the flange only at standstill, robot stopped before
  and after, mass removed before resuming.
* Vendor safety configurations (joint limits, collision detection, reduced
  mode) are the last line of defence; their checksums go into the manifest.
  They are not replaced by anything in this stack.
* `confirm_each_trajectory` defaults to true; `--no-confirm` skips the
  per-trajectory prompt and should only be used once a session is already
  trusted (e.g. an unattended production run after commissioning).

## Known gaps after this pass (see RR_05 for the full, numeric list)

* `erd_fri_emulator` (L2 iiwa) and URSim CB3 (L2 UR10) bring-up was verified
  in a prior pass (RR_03); this pass's new safety/monitor/convert code was
  not re-run against either simulator end-to-end -- see RR_05's own
  acceptance table for exactly what was (re-)verified live versus only
  unit-tested this pass.
* No PREEMPT_RT kernel on this laptop; 1 kHz jitter under real load is
  unmeasured.

## Pass-4 recording details

The iiwa driver uses the controller manager's blocking-read timing mode:
FRI packets pace the loop. A second periodic sleep causes a receive backlog.
The recording broadcaster buffers up to 8,192 hardware samples before DDS
publication; queue overflow is an error. The standard joint-state broadcaster
continues publishing `/joint_states` for TF. `fri/cycle` is the remote packet
sequence and `fri/received_cycle` counts local reads, so the converter can
distinguish FRI loss from publication/recording loss. An excitation containing
missing cycles is excluded; its counters remain in `validation.json`.

The FRI receive deadline is 20 ms. After a read error the controller manager
must deactivate the trajectory controller within two control cycles. Stop
confirmation additionally requires 0.5 s of fresh stationary telemetry; a
stale cached velocity cannot confirm a stop.

Until Q-16 supplies training references, the checked-in
`config/reference/synthetic_<robot>.contract.json` files provide the robot's
joint order and differentiation settings. Conversion records
`reference_synthetic: true`, and validation remains `synthetic`, including
when a synthetic reference is explicitly used with a real recording. These
references do not establish identification validity or real-robot sign-off.
Real-data parquet signals retain float64 precision so the recorded positions
and their declared derivatives remain consistent after writing the file.

The monitor's own joint-state/controller-state samples are served by a
dedicated executor thread (`_MonitorFeed`, a `SingleThreadedExecutor`: rclpy's
multi-threaded executor delivered only 2-11 of the 1000 msg/s), not by this
process's main poll loop, so a busy main thread can never leave them stale at
the iiwa's 1 kHz (RR_08 S2c). The 1 kHz logging subscription exists only
while preflight measures the rate (or the UR sidecar alignment runs);
otherwise its queue refills during every blocking pause and draining it
starves the feed. Measured on the emulator (RR_11): trigger -> cancel
3.0-7.7 ms over 20 injections, sample age p99 1.3 ms.

To script the speed-scaling abort on URSim (RR_08 S2d, P4 item 9) instead
of operating the pendant by hand, run `ros2 run erd_ur10
speed_slider_abort --delay-s <seconds into the excitation> --fraction 0.5`
alongside `record_ur10`, only while an excitation/sweep segment is moving.
It calls the driver's `/io_and_status_controller/set_speed_slider` service
(a second RTDE client cannot write the slider: the driver's connection owns
those inputs, S-24), holds the fraction for `--hold-s`, and always resets
the slider to 1.0 at the end, also on Ctrl-C. Expect the live monitor to
abort once `speed_scaling` drops below `safety.SPEED_SCALING_FLOOR`
(0.999).
