# Operator notes

Covers Step 1 (L0-L2, everything T1.0-T1.10 exercised) and, per RR_04 C-3, the
Step-2 procedure the owner reads before T2.0: network practice, the
commissioning ladder, the E-* protocols and the stop/e-stop rules. This file
does not replace RR_01 (the full spec) or RR_05 (what this pass actually
verified); it is the short version to have open in the lab.

## Build and environment

One-time setup: `src/ros2/README.md`. Two environment issues were found and
fixed (not "worked around" -- RR_04 B-1 removed the pass-1 `--prefix`
stopgap):

1. **pinocchio vs. ROS's sourced environment.** `/opt/ros/jazzy/setup.bash`
   puts ROS's own `ros-jazzy-pinocchio` ahead of `.venv-erd`'s `pin==4.1.0` in
   `PYTHONPATH`/`LD_LIBRARY_PATH` -- a numpy-ABI mismatch that segfaults,
   not a clean import error. Every `erd_recording` entry point still guards
   against this (`env_guard.assert_environment()`), and `plan`/`convert`
   still run as **clean subprocesses** (`plan_cli.py`/`convert_cli.py`) so
   pinocchio is never imported in the same process as a sourced ROS
   environment. Consequence for testing: **`erd_recording`'s own pytest
   suite must run without ROS sourced** (it imports `elastic_sim`/pinocchio
   directly in several test files, not just through the subprocess CLIs):
   ```bash
   source ~/projects/erd_ws/.venv-erd/bin/activate   # a shell with NO ROS setup.bash sourced
   cd ~/projects/erd_ws
   python -m colcon test --packages-select erd_recording
   ```
2. **colcon/`ros2 run` and the interpreter.** `pip install
   colcon-common-extensions` into `.venv-erd` (already done by the one-time
   setup) and **build with the venv's own `python -m colcon build`**, not the
   system `colcon`: generated console-script shebangs then point at the
   active interpreter (`erd_recording/setup.cfg`'s `/usr/bin/env python3`
   resolves correctly once the venv is first on `PATH`; other packages get an
   absolute `.venv-erd/bin/python` shebang from setuptools directly). Verify
   with `head -1 install/erd_recording/lib/erd_recording/record_iiwa`.
   `ros2 run erd_recording record_iiwa ...` and `ros2 launch erd_ur10 ...`
   (the sidecar) then work with **no `--prefix` needed**.
3. The ROS/C++ packages' own tests (`erd_iiwa`, `erd_ur10`, `erd_fri_emulator`,
   `erd_msgs`) need `rclpy`/`ament_index_python`, i.e. ROS **sourced**:
   ```bash
   source /opt/ros/jazzy/setup.bash
   source ~/projects/erd_ws/.venv-erd/bin/activate
   cd ~/projects/erd_ws
   python -m colcon build
   python -m colcon test --packages-select erd_iiwa erd_ur10 erd_fri_emulator erd_msgs
   ```
   Building is fine with ROS sourced (nothing at *build* time imports
   pinocchio); only running erd_recording's *test suite* needs the ROS-free
   shell. Two separate test invocations, not a workaround -- this is the same
   process-separation the codebase already uses for `plan`/`convert`.

## Running a mock (L1) session

```bash
source /opt/ros/jazzy/setup.bash
source ~/projects/erd_ws/install/setup.bash
source ~/projects/erd_ws/.venv-erd/bin/activate

ros2 launch erd_iiwa iiwa.launch.py hardware:=mock \
    lab_config:=src/erd/erd_recording/config/lab/iiwa_mock.yaml &
ros2 run erd_recording record_iiwa \
    --config src/erd/erd_recording/config/lab/iiwa_mock.yaml all --run-id demo
```

`lab_config:=` on the launch line generates the JTC's per-joint path/goal
tolerances from `limits.abort.tracking_rad` (RR_04 A-5) -- pass the same file
the `record_iiwa`/`record_ur10` command uses, or the tolerances silently stay
at the checked-in defaults instead of the lab's own abort radius.

Output lands in `<recording.output_root>/<robot>/<date>/<run_id>/`
(`config.yaml`, `plan/`, `preflight.json`, `manifest.yaml`, `bag/`,
`bag_standstill/`, `raw/` logs). Kill any previous `ros2 launch`/
`ros2_control_node`/`robot_state_publisher` before relaunching -- a stale
`robot_state_publisher` still latching an old `/robot_description` is a real,
confusing failure mode (the new `ros2_control_node` silently loads the *old*
URDF).

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
dedicated executor thread (`_MonitorFeed`), not by this process's main
poll loop, so a busy main thread can never leave them stale at the iiwa's
1 kHz (RR_08 S2c). This has not been re-measured live against the
emulator this pass; re-run the P4 item 7 latency/cancel-latency check
before relying on the numbers.

To script the speed-scaling abort on URSim (RR_08 S2d, P4 item 9) instead
of operating the pendant by hand, run `ros2 run erd_ur10
speed_slider_abort --robot-ip <ip> --delay-s <seconds into the excitation>
--fraction 0.5` alongside `record_ur10`; it opens a third, input-only RTDE
connection and expects the live monitor to abort once `speed_scaling` drops
below `safety.SPEED_SCALING_FLOOR` (0.999). Not yet run against a live
URSim this pass.
