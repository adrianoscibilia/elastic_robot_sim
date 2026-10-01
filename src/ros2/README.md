# `src/ros2/` — the RR_01 real-data recording stack

Five packages implementing `REFACTOR_SPECS/ros2_real_data/RR_01_ROS2_RECORDING_STACK_PLAN.md`
Step 1 (iiwa + UR10, L0–L2). Build workspace is **outside this repo**, at
`~/projects/erd_ws`, so `build/`/`install/`/`log/` never land in git.

## Packages

| package | build | contents |
|---|---|---|
| `erd_msgs` | ament_cmake | `RunEvent.msg` |
| `erd_recording` | ament_python | config schema/loader, planning (WorldKinematics, re-limiting, digests), pipeline orchestration (rclpy), converter, identification (E-ur-1/E-ur-3), validation |
| `erd_iiwa` | ament_cmake | `erd_iiwa/FriPositionSystem` hardware plugin (POSITION mode), description overlay, controllers, launch |
| `erd_ur10` | ament_python | launch wrapping `ur_robot_driver`, RTDE sidecar logger |
| `erd_fri_emulator` | ament_cmake | FRI 1.11 robot-side UDP emulator (session state machine, Pinocchio-RNEA plant) for L2 iiwa |

## One-time setup

```bash
mkdir -p ~/projects/erd_ws/src
ln -s ~/projects/elastic_robot_sim/src/ros2 ~/projects/erd_ws/src/erd
cd ~/projects/erd_ws
vcs import src < src/erd/erd.repos
touch src/iiwa_ros2/iiwa_bringup/COLCON_IGNORE \
      src/iiwa_ros2/iiwa_controllers/COLCON_IGNORE \
      src/iiwa_ros2/iiwa_description_moveit_config/COLCON_IGNORE

python3 -m venv --system-site-packages .venv-erd
source .venv-erd/bin/activate
pip install -e ~/projects/elastic_robot_sim --no-deps
pip install numpy scipy pandas pyarrow pyyaml "pin==4.1.0" "pin-pink==4.3.0" trimesh pytest
pip install "git+https://github.com/UniversalRobots/RTDE_Python_Client_Library.git"
```

## Building and running (read this before `ros2 run`)

Two environment issues were found; both are now fixed at the source, not
worked around (T1.0/T1.6, RR_04 B-1; details in `RR_03_IMPLEMENTATION_REPORT.md`
and `RR_05_IMPLEMENTATION_REPORT.md`):

1. **pinocchio vs. ROS's `PYTHONPATH`/`LD_LIBRARY_PATH`.** Sourcing
   `/opt/ros/jazzy/setup.bash` breaks `.venv-erd`'s `pinocchio` (a numpy-ABI
   and dynamic-linker conflict with ROS's own `ros-jazzy-pinocchio`). Every
   `erd_recording` entry point calls `env_guard.assert_environment()` first,
   which reorders `sys.path`; the deeper `LD_LIBRARY_PATH` conflict is why
   the `plan`/`convert` stages run as a **subprocess** with
   `env_guard.clean_ros_subprocess_env()` (`plan_cli.py`/`convert_cli.py`)
   rather than in the ROS process. Consequence: `erd_recording`'s own pytest
   suite (which imports `elastic_sim`/pinocchio directly in several test
   files, not just through the subprocess CLIs) must run in a shell with **no
   ROS setup.bash sourced** -- see below.
2. **`colcon`/`ros2 run` and the interpreter.** `pip install
   colcon-common-extensions` into `.venv-erd` (in the one-time setup above)
   and build with **`python -m colcon build`** from the activated venv, not
   the system `colcon` binary: generated console-script shebangs then
   resolve to the active interpreter. Verify with `head -1
   install/erd_recording/lib/erd_recording/record_iiwa`. No `--prefix`
   workaround is needed any more.

```bash
# Build (ROS sourced is fine here -- nothing at build time imports pinocchio):
source /opt/ros/jazzy/setup.bash
source ~/projects/erd_ws/.venv-erd/bin/activate
cd ~/projects/erd_ws && python -m colcon build

# Test: two invocations, split by which environment each package's tests need.
python -m colcon test --packages-select erd_iiwa erd_ur10 erd_fri_emulator erd_msgs   # needs rclpy -> ROS sourced (above)
deactivate && source ~/projects/erd_ws/.venv-erd/bin/activate                          # a shell with NO ROS setup.bash sourced
python -m colcon test --packages-select erd_recording                                 # needs pinocchio, not rclpy

# Run (ROS + venv both sourced, as for build):
source /opt/ros/jazzy/setup.bash
source ~/projects/erd_ws/install/setup.bash
source ~/projects/erd_ws/.venv-erd/bin/activate
ros2 launch erd_iiwa iiwa.launch.py hardware:=mock \
    lab_config:=src/erd/erd_recording/config/lab/iiwa_mock.yaml   # or erd_ur10 ur10.launch.py hardware:=mock
ros2 run erd_recording record_iiwa \
    --config src/erd/erd_recording/config/lab/iiwa_mock.yaml all
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
ros2 run erd_fri_emulator run_emulator.sh
# -> prints the container IP; use it as robot_ip:
ros2 launch erd_iiwa iiwa.launch.py hardware:=emulator robot_ip:=<printed IP> \
    lab_config:=src/erd/erd_recording/config/lab/iiwa_sim.yaml

# UR10: no port-collision issue (the driver connects outbound, it doesn't
# bind the robot's own ports). This wraps ur_client_library's own
# start_ursim.sh, which downloads and installs the External Control URCap
# itself (S-18) -- nothing to set up by hand in PolyScope/VNC any more.
ros2 run erd_ur10 start_ursim.sh
# -> power on + release brakes once via the dashboard server or the printed
#    noVNC URL, then:
ros2 launch erd_ur10 ur10.launch.py hardware:=ursim headless_mode:=true \
    robot_ip:=<printed IP> reverse_ip:=<printed docker0 IP> \
    lab_config:=src/erd/erd_recording/config/lab/ur10_sim.yaml
```

`lab_config:=` on both launch lines generates the JTC's per-joint path/goal
tolerances from `limits.abort.tracking_rad` (RR_04 A-5) -- pass the same file
the `record_iiwa`/`record_ur10` command uses.

See `RR_03_IMPLEMENTATION_REPORT.md` and `RR_05_IMPLEMENTATION_REPORT.md` in
`REFACTOR_SPECS/ros2_real_data/` for every acceptance result.
