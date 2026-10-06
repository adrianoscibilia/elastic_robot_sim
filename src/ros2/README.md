# `src/ros2/` — the RR_01 real-data recording stack

Five packages implementing `REFACTOR_SPECS/ros2_real_data/RR_01_ROS2_RECORDING_STACK_PLAN.md`
Step 1 (iiwa + UR10, L0–L2). Everything needed to build and run lives in the
repository: `workspace_setup.sh` (repo root) generates the colcon workspace
`ros2_ws/` and the recording root `data/real_robot/`, both gitignored.

## Packages

| package | build | contents |
|---|---|---|
| `erd_msgs` | ament_cmake | `RunEvent.msg` |
| `erd_recording` | ament_python | config schema/loader, planning (WorldKinematics, re-limiting, digests), pipeline orchestration (rclpy), converter, identification (E-ur-1/E-ur-3), validation |
| `erd_iiwa` | ament_cmake | `erd_iiwa/FriPositionSystem` hardware plugin (POSITION mode), description overlay, controllers, launch |
| `erd_ur10` | ament_python | launch wrapping `ur_robot_driver`, RTDE sidecar logger |
| `erd_fri_emulator` | ament_cmake | FRI 1.11 robot-side UDP emulator (session state machine, Pinocchio-RNEA plant) for L2 iiwa |

## Setup on a new machine (Ubuntu 24.04)

```bash
git clone https://github.com/adrianoscibilia/elastic_robot_sim.git && cd elastic_robot_sim
source workspace_setup.sh --deps
```

`--deps` installs ROS 2 Jazzy if it is missing plus the stack's apt/rosdep
dependencies (sudo). The script then:

1. links `ros2_ws/src/erd -> ../../src/ros2`;
2. clones `src/ros2/erd.repos` (pinned `iiwa_ros2`: `iiwa_description` and
   KUKA's libFRI, which is compiled from there and never committed) into
   `ros2_ws/src/external/`;
3. creates `ros2_ws/.venv-erd` from the system Python 3.12 with
   `--system-site-packages` (rclpy), installs `requirements-erd.txt` and this
   repo editable (`--no-deps`);
4. runs `python -m colcon build` in `ros2_ws/`;
5. sources ROS, the overlay and the venv, and exports `ROS_DOMAIN_ID=87`
   (`--real`: 0) and `ERD_REPO_ROOT`, `ERD_WS`, `ERD_DATA_ROOT`, `ERD_LAB`,
   `ERD_FRI_SDK_ROOT`, `ERD_CONSUMER_REPO`, `ERD_CONSUMER_PYTHON`.

Every new terminal: `source <repo>/workspace_setup.sh` (no rebuild unless
nothing is built). `--build` rebuilds incrementally, `--clean` from scratch,
`--test` runs both test invocations below. The consumer
(`dynamic_model_nn` with its `.venv`) is expected next to this repo; set
`ERD_CONSUMER_REPO`/`ERD_CONSUMER_PYTHON` before sourcing to point elsewhere.

The lab configs name machine-dependent paths through those variables
(`output_root: ${ERD_DATA_ROOT}`, `consumer.repo: ${ERD_CONSUMER_REPO}`);
`erd_recording.config` expands them and refuses an unset one. The raw text
stays in `LabConfig.raw`, so plan digests are the same on every machine.

## Environment notes (read before `ros2 run`)

1. **pinocchio vs. ROS's `PYTHONPATH`/`LD_LIBRARY_PATH`.** Sourcing
   `/opt/ros/jazzy/setup.bash` breaks `.venv-erd`'s `pinocchio` (a numpy-ABI
   and dynamic-linker conflict with ROS's own `ros-jazzy-pinocchio`). Every
   `erd_recording` entry point calls `env_guard.assert_environment()` first,
   which reorders `sys.path`; the `plan`/`convert`/`identify` stages run as a
   **subprocess** with `env_guard.clean_ros_subprocess_env()`. Consequence:
   `erd_recording`'s own pytest suite must run in a shell with **no ROS
   sourced** -- `workspace_setup.sh --test` does that for you.
2. **Build with the venv's `python -m colcon build`** (the script does), so
   console-script shebangs resolve to `.venv-erd`'s interpreter. Check with
   `head -1 $ERD_WS/install/erd_recording/lib/erd_recording/record_iiwa`.

```bash
source workspace_setup.sh --test        # build + both test invocations
# or by hand, from $ERD_WS:
python -m colcon test --packages-select erd_iiwa erd_ur10 erd_fri_emulator erd_msgs   # ROS sourced
#   ...and, in a fresh shell with only the venv activated (no ROS):
python -m colcon test --packages-select erd_recording

# L1 (mock), from any directory once sourced:
ros2 launch erd_iiwa iiwa.launch.py hardware:=mock lab_config:=$ERD_LAB/iiwa_mock.yaml
ros2 run erd_recording record_iiwa --config $ERD_LAB/iiwa_mock.yaml all
```

## Levels

| level | iiwa | UR10 |
|---|---|---|
| L0 | unit tests, no ROS | unit tests, no ROS |
| L1 | `mock_components/GenericSystem` | `use_mock_hardware` |
| L2 | `erd_fri_emulator`, verified against the real `erd_iiwa/FriPositionSystem` plugin at 1000 Hz (must run in a separate network namespace/container from the client — see RR_03 T1.9); `erd_recording`'s new safety monitor/cancel and standstill-as-plan-segments code (RR_04 A-1/A-2/A-3) is unit-tested but not yet re-run against it this pass (RR_05) | URSim CB3 3.15.8 via `ur_client_library`'s own `start_ursim.sh` (RR_04 B-6), which installs the External Control URCap itself; `headless_mode:=true` lets the driver execute goals without a pendant program (RR_01 v3.1 S3.5) — not re-verified live this pass (RR_05) |
| L3/L4 | real robot | real robot |

**Running L2** (RR_04 B-6 scripts wrap what pass 1 did by hand):

```bash
# iiwa: erd_fri_emulator and the client it talks to (erd_iiwa, on the host)
# cannot both bind the FRI port on one host (the client's SDK code binds a
# UDP wildcard address; two same-host wildcard binds always conflict), so the
# emulator runs in its own Docker network namespace/bridge IP:
bash $ERD_REPO_ROOT/src/ros2/erd_fri_emulator/scripts/run_emulator.sh
# -> prints the container IP; use it as robot_ip:
ros2 launch erd_iiwa iiwa.launch.py hardware:=emulator robot_ip:=<printed IP> \
    lab_config:=$ERD_LAB/iiwa_sim.yaml

# UR10: no port-collision issue (the driver connects outbound, it doesn't
# bind the robot's own ports). This wraps ur_client_library's own
# start_ursim.sh, which downloads and installs the External Control URCap
# itself (S-18) -- nothing to set up by hand in PolyScope/VNC any more.
ros2 run erd_ur10 start_ursim.sh
# -> power on + release brakes once via the dashboard server or the printed
#    noVNC URL, then:
ros2 launch erd_ur10 ur10.launch.py hardware:=ursim headless_mode:=true \
    robot_ip:=<printed IP> reverse_ip:=<printed gateway IP> \
    lab_config:=$ERD_LAB/ur10_sim.yaml
```

`lab_config:=` on both launch lines generates the JTC's per-joint path/goal
tolerances from `limits.abort.tracking_rad` (RR_04 A-5) -- pass the same file
the `record_iiwa`/`record_ur10` command uses.

See `RR_03_IMPLEMENTATION_REPORT.md` and `RR_05_IMPLEMENTATION_REPORT.md` in
`REFACTOR_SPECS/ros2_real_data/` for every acceptance result.
