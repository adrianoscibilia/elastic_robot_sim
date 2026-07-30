# elastic_robot_sim

Simulation and system-identification framework for an elastic Cartesian robot (3-axis gantry with spring-damper elastic drives).  Supports synthetic dataset generation, multi-backend simulation, and sim-to-real calibration.

---

## Overview

The platform is a 3-axis Cartesian gantry (X, Y, Z) where each axis has:
- A **motor joint** driven by a stiff PD controller (motor stiffness 30 000 N/m, damping 500 N·s/m)
- An **elastic joint** (spring-damper) coupling the motor to the moving link

Each axis is independently parameterised by a stiffness `k` (N/m) and a damping ratio `ζ`, with physical damping computed as `d = 2 ζ √(k m)`.  An optional rigid payload can be attached to the end-effector.

The framework has two main use cases:

1. **Synthetic dataset generation** — sweep robot configurations and trajectory seeds to build training data for learning-based controllers.
2. **Sim-to-real calibration** — record the same trajectory on one or more simulators and the real robot, then optimise the elastic parameters so the simulator matches real behaviour.

---

## Simulation Backends

| Backend | Engine | Entry point | Notes |
|---|---|---|---|
| **Newton** | NVIDIA Warp (GPU) | `scripts/elastic_cart_robot_newton.py` | Primary backend; requires a CUDA-capable GPU |
| **MuJoCo** | DeepMind MuJoCo | `scripts/elastic_cart_robot_mujoco.py` | CPU-based; MJCF built programmatically from params |
| **Isaac Sim** | NVIDIA Isaac Sim | `scripts/elastic_cart_robot_isaacsim.py` | Requires Isaac Sim installation |

All three backends share `scripts/sim_common.py` for settings loading, trajectory generation, noise constants, and CSV column definitions.

---

## Trajectory Modes

Each backend and the real-robot recorder support three reference trajectory types:

| Mode | Name | Description |
|---|---|---|
| `0` | **Hold** | Fixed position for the entire simulation |
| `1` | **Sinusoidal** | Per-axis sine waves with random amplitude, frequency, and phase; Y axis uses cosine to match controller convention |
| `2` | **PTP** | Point-to-point sequence with cubic blending; waypoints sampled uniformly within joint limits |

Trajectories are fully reproducible: all random draws use a seeded `numpy` RNG, and the generated parameters are serialised to `trajectory.json` so the exact same path can be replayed on a second backend or the real robot.

---

## Building the ROS 2 Package

The package uses `ament_cmake` and integrates with `colcon`.  Only `collect_dataset` and `record_real_rollout` (installed without the `.py` suffix) and the `elastic_sim` Python library are installed; all other simulation scripts are left untouched.

```bash
# Source ROS 2 first, then build
source /opt/ros/jazzy/setup.bash
cd <workspace_root>
colcon build --packages-select elastic_robot_sim
source install/setup.bash
```

---

## Entry-Point Scripts

### Standalone simulation (single backend)

```bash
# Newton (GPU) – interactive viewer
python scripts/elastic_cart_robot_newton.py

# Newton – headless CSV export (used by dataset generation)
python scripts/elastic_cart_robot_newton.py --csv --headless

# MuJoCo – interactive viewer
python scripts/elastic_cart_robot_mujoco.py

# MuJoCo – headless CSV export
python scripts/elastic_cart_robot_mujoco.py --csv --headless
```

Simulation parameters (stiffness, damping ratio, trajectory mode, seed, duration) are read from `config/settings.yaml`.

---

### Synthetic dataset generation

```bash
python scripts/run_dataset_generation.py
```

Sweeps `ROBOT_CONFIG_COUNT` random elastic configurations × `TRIALS_PER_ROBOT` trajectories using the Newton backend.  Stiffness is sampled log-uniformly; damping ratio linearly.  Each trial writes a timestamped CSV to `data/`.  Configuration is hardcoded at the top of the script; `config/settings.yaml` is restored to its original values after the run.

---

### Multi-backend recording (calibration data collection)

#### Option A — Simulators only (fixed trajectory)

```bash
# Record Newton + MuJoCo with the same PTP trajectory
python scripts/record_rollouts.py --backends newton mujoco --mode 2 --seed 42

# Re-run an existing trajectory.json on a different backend
python scripts/record_rollouts.py --backends mujoco \
    --traj-config data/rollouts/traj_m2_s42/trajectory.json
```

Saves to `data/rollouts/traj_m<mode>_s<seed>/` with one parquet file per backend.

#### Option B — Continuous real-robot data collection (ROS 2)

Requires a sourced workspace with `joint_trajectory_controller` running (launched by `run_platform_control.launch.py`).

```bash
# Use all settings from config/settings.yaml (output_dir included)
ros2 run elastic_robot_sim collect_dataset

# Override output directory
ros2 run elastic_robot_sim collect_dataset \
    --output-dir data/recordings/session_01

# Override specific settings on the CLI
ros2 run elastic_robot_sim collect_dataset \
    --output-dir data/recordings/session_01 \
    --num-trajectories 20 \
    --modes ptp \
    --sim-time 12.0

# Point at a different settings file
ros2 run elastic_robot_sim collect_dataset \
    --settings /path/to/settings.yaml
```

The output directory is created automatically if it does not exist.

For each trajectory the node:
1. Reads parameters from `config/settings.yaml` (joint limits, duration, modes, seed policy)
2. Picks a random mode (sinusoidal / PTP) and a random seed
3. Saves `trajectory_TS.json` — the seed and waypoints are stored so the exact same trajectory can be replayed in the simulator later
4. Executes on the real robot via the `FollowJointTrajectory` action client and records `/joint_states` + `/ft_sensor_command_broadcaster/wrench`
5. If the robot fails, the trajectory config is still kept (useful for debugging and for replaying in sim)

Output layout:
```
data/recordings/session_01/
  trajectory_20240605_143022.json   ← seed + waypoints, fully reproducible
  real_20240605_143022.parquet      ← ground truth signals
  trajectory_20240605_143058.json
  real_20240605_143058.parquet
  ...
```

#### Real-robot recording (standalone, single trajectory)

```bash
# Execute one pre-generated trajectory and record
ros2 run elastic_robot_sim record_real_rollout \
    --traj-config data/rollouts/traj_m2_s42/trajectory.json \
    --output-dir  data/rollouts/traj_m2_s42/

# Dry-run: plan only, no motion
ros2 run elastic_robot_sim record_real_rollout \
    --traj-config data/rollouts/traj_m2_s42/trajectory.json \
    --output-dir  data/rollouts/traj_m2_s42/ \
    --dry-run
```

Subscribes to `/joint_states` and `/ft_sensor_command_broadcaster/wrench`, sends `FollowJointTrajectory` to `/joint_trajectory_controller/follow_joint_trajectory`, and writes `real.parquet` resampled onto a 100 Hz grid.

---

### Sim-to-real calibration

Defaults are read from `config/calibration.yaml`; all values can be overridden on the CLI.

```bash
# Defaults from config/calibration.yaml, recordings from collect_dataset
python scripts/run_calibration.py --recordings-dir data/recordings/session_01

# Override backend and optimizer
python scripts/run_calibration.py \
    --recordings-dir data/recordings/session_01 \
    --backend mujoco --optimizer bo

# Structured rollout store (from record_rollouts.py)
python scripts/run_calibration.py --rollouts-dir data/rollouts

# Extra overrides
python scripts/run_calibration.py \
    --recordings-dir data/recordings/session_01 \
    --max-evals 300 --output calibrated_newton.yaml --no-payload --verbose
```

The script:
1. Loads all `(real rollout, trajectory config)` pairs from the recordings
2. Splits them into train / validation sets (`train_fraction` from `calibration.yaml`, default 80/20)
3. For each optimizer evaluation: **re-runs the full simulation headlessly** from the saved trajectory seed, then compares against the real recording
4. Validates on the held-out set
5. Saves the best params to `calibrated_<backend>.yaml`
6. Saves `<backend>_calibrated_TS.parquet` alongside the recordings for visual comparison

#### Optimizer backends

| Flag | Algorithm | Package | Notes |
|---|---|---|---|
| `cma` (default) | CMA-ES | `cma` | Fast, robust for 6–7 dims; recommended starting point |
| `bo` | Bayesian optimisation | `scikit-optimize` | Better sample efficiency; slower per iteration |
| `skrl` | PPO contextual bandit | `skrl`, `gymnasium`, `torch` | Experimental; treats each trajectory as a context |

---

## Calibration Parameters

The optimizer tunes these 6 (or 7 with payload) scalar values:

| Parameter | Space | Bounds |
|---|---|---|
| `drive_x.stiffness` | log₁₀ | 1 000 – 50 000 N/m |
| `drive_x.damping_ratio` | linear | 0.05 – 2.0 |
| `drive_y.stiffness` | log₁₀ | 1 000 – 50 000 N/m |
| `drive_y.damping_ratio` | linear | 0.05 – 2.0 |
| `drive_z.stiffness` | log₁₀ | 500 – 30 000 N/m |
| `drive_z.damping_ratio` | linear | 0.05 – 2.0 |
| `payload` *(optional)* | linear | 0.0 – 10.0 kg |

All values are normalised to `[-1, 1]` before being passed to the optimizer.

The fidelity metric is a weighted normalised RMSE over position, velocity, and torque signals (configurable in `config/calibration.yaml`).

### Asset-driven serial-arm simulation and calibration

The Cartesian platform remains supported, but new robots are now defined by
asset files rather than hard-coded joint names.  Put a robot description and
its provenance under `assets/robots/<robot>/`, then create an `asset.yaml`
with its URDF and ordered active joints.  `elastic_sim.assets` discovers the
URDF's non-mimic one-DoF joints, and `elastic_sim.generic_newton_runner`
rebuilds every selected joint as a motor joint followed by a negligible-mass,
passive elastic transmission joint.

The asset layer is intentionally reusable: robot-specific files name assets,
joint order, units and parameter priors; trajectory generation, torque replay,
comparison, optimization and storage do not embed robot names.  See
[`assets/README.md`](assets/README.md) for provenance and the downloaded
benchmark material.

For a serial arm, Pinocchio is the primary non-ROS trajectory dependency: it
loads the URDF and its joint limits, validates the selected one-DoF chain, and
creates seeded, limit-safe minimum-jerk joint-space trajectories.  Install the
Pinocchio Python bindings in the simulator environment, then run:

```bash
python scripts/generate_serial_trajectory.py \
  --urdf assets/robots/baxter/description/baxter_description/urdf/baxter.urdf \
  --joints left_s0,left_s1,left_e0,left_e1,left_w0,left_w1,left_w2 \
  --output data/trajectories/baxter_left.json --seed 42
```

MoveIt is the complementary ROS 2 path for collision-aware planning and real
robot execution; it should consume these asset definitions rather than become
the simulator's required runtime dependency.

`scripts/run_dataset_calibration.py` calibrates a generic Newton asset by
replaying recorded joint torques.  It consumes canonical CSV/Parquet data and
an asset/config-only YAML such as
`config/dataset_calibration.example.yaml`.  Its metric distinguishes observed
output/link state, optional observed motor state, joint torque input, and an
optional end-effector F/T channel—none is silently substituted for another.

---

## Configuration

All configuration lives under `config/`.  There are two files:

### `config/settings.yaml`

Runtime parameters for standalone simulation, dataset generation, and data collection.

```yaml
elastic_drives:
  drive_x:  { stiffness: 5977.08, damping_ratio: 0.49 }
  drive_y:  { stiffness: 7162.08, damping_ratio: 0.68 }
  drive_z:  { stiffness: 3861.11, damping_ratio: 1.20 }
  payload:  0.0

simulation:
  ref_mode:      2       # 0=hold, 1=sinusoidal, 2=PTP
  seed:         -1       # -1 = random
  sim_time:     15.0     # seconds
  time_step:     0.01    # seconds
  cut_off_time:  2.0     # seconds to skip at the start of comparisons

collection:
  num_trajectories: 10          # 0 = run until Ctrl-C
  modes: [sin, ptp]             # trajectory types to randomly sample
  sim_time: 15.0                # seconds per trajectory
  master_seed: null             # null = random; set integer for reproducible sessions
  joint_limits:
    x: [-1.8, 1.8]              # metres — stay within URDF hard limits
    y: [-1.8, 1.8]
    z: [-1.0, 1.0]
  payload: 0.0                  # kg attached during collection
  output_dir: data/recordings/session_latest  # created automatically if missing
  ft_topic: /ft_sensor_command_broadcaster/wrench
```

When installed via `colcon build`, the active copy is at:
```
install/elastic_robot_sim/share/elastic_robot_sim/config/settings.yaml
```
Edit that file to change defaults without rebuilding.  Pass `--settings /path/to/file.yaml` to `collect_dataset` to use a different file entirely.

### `config/calibration.yaml`

Optimizer hyper-parameters and metric weights for `run_calibration.py`.  Loaded automatically as defaults; every field can be overridden on the CLI.

---

## Recorded Signals

Every rollout (sim or real) stores the following 24 time-series columns at 100 Hz:

| Group | Signals |
|---|---|
| Reference | `ref_x/y/z`, `vel_x/y/z` |
| Motor positions | `q_motor_x/y/z` |
| Elastic link positions | `q_link_x/y/z` |
| Motor velocities | `dq_motor_x/y/z` |
| Elastic link velocities | `dq_link_x/y/z` |
| Motor torques | `tau_motor_x/y/z` |
| Elastic torques | `tau_link_x/y/z` |

---

## Project Structure

```
elastic_robot_sim/
├── config/
│   ├── settings.yaml              # Sim + collection parameters
│   ├── calibration.yaml           # Calibration optimizer settings
│   └── elastic_cart_mujoco.xml    # MuJoCo scene template
├── scripts/
│   ├── sim_common.py              # Shared constants, trajectory generators, CSV utils
│   ├── elastic_cart_robot_newton.py   # Newton standalone simulation
│   ├── elastic_cart_robot_mujoco.py   # MuJoCo standalone simulation
│   ├── elastic_cart_robot_isaacsim.py # Isaac Sim standalone simulation
│   ├── run_dataset_generation.py  # Batch synthetic dataset generation (Newton)
│   ├── record_rollouts.py         # Record fixed-trajectory rollouts (sim backends)
│   ├── collect_dataset.py         # ROS 2 node: batch real-robot dataset collection
│   └── run_calibration.py         # Sim-to-real calibration optimizer
├── src/elastic_sim/               # Calibration framework library (installed via colcon)
│   ├── params.py                  # RobotParams dataclass (normalize, bounds, YAML I/O)
│   ├── trajectory.py              # Trajectory classes + factory functions
│   ├── rollout.py                 # RolloutResult + RolloutStore (parquet I/O)
│   ├── compare.py                 # Fidelity metric (normalised RMSE)
│   ├── sim_runner.py              # Newton programmatic API
│   ├── mujoco_runner.py           # MuJoCo programmatic API
│   ├── calibration.py             # SimCalibrationProblem (loss function)
│   ├── ros_recorder.py            # RealRobotRecorder ROS 2 node (shared by both executables)
│   └── optimizers/
│       ├── base.py                # Optimizer ABC
│       ├── cma_backend.py         # CMA-ES
│       ├── bo_backend.py          # Bayesian optimisation
│       └── skrl_backend.py        # skrl PPO/SAC
├── real_robot/
│   └── record_real_rollout.py     # ROS 2 executable: single-trajectory recording
├── CMakeLists.txt                 # ament_cmake build (installs elastic_sim + executables)
├── package.xml                    # ROS 2 package manifest
├── data/                          # Generated CSV datasets
├── data/rollouts/                 # Structured rollout store (record_rollouts.py)
├── data/recordings/               # Flat timestamped recordings (collect_dataset)
├── urdf/
│   └── platform_complete.urdf     # Robot description
└── tests/
    └── test_vertical_deflection_payload_sweep.py
```

---

## Requirements

### Core (all backends)
```
numpy  scipy  pandas  pyarrow  pyyaml  matplotlib
```

### Newton backend
```
# NVIDIA Warp + Newton physics (GPU required)
warp-lang  newton
```

### MuJoCo backend
```
mujoco
```

### Calibration framework
```
# Minimum (CMA-ES only)
cma

# Full optimizer support
scikit-optimize        # Bayesian optimisation
skrl  gymnasium torch  # skrl PPO/SAC
```

### Real robot recording (ROS 2)
```
# ROS 2 Jazzy with:
#   joint_trajectory_controller
#   control_msgs  trajectory_msgs  sensor_msgs  geometry_msgs  builtin_interfaces
#   ament_cmake  ament_cmake_python  ament_index_python
```

---

## Typical Workflows

### A — Pure sim dataset for supervised learning

```bash
# Edit ROBOT_CONFIG_COUNT / TRIALS_PER_ROBOT at the top of the script, then:
python scripts/run_dataset_generation.py
# → data/d<timestamp>_s<seed>_p<payload>_X<k>_Y<k>_Z<k>_ref<mode>.csv
```

### B — Sim-to-real calibration

```bash
# Step 1: build and source the ROS 2 workspace
source /opt/ros/jazzy/setup.bash
colcon build --packages-select elastic_robot_sim
source install/setup.bash

# Step 2: edit collection settings
nano install/elastic_robot_sim/share/elastic_robot_sim/config/settings.yaml

# Step 3: launch the hardware stack (in a separate terminal)
ros2 launch tecnobody_workbench run_platform_control.launch.py

# Step 4: collect real-robot ground truth
ros2 run elastic_robot_sim collect_dataset \
    --output-dir data/recordings/cal_run_01
# → trajectory_TS.json + real_TS.parquet for each executed trajectory

# Step 5a: calibrate Newton against real
python scripts/run_calibration.py \
    --recordings-dir data/recordings/cal_run_01 \
    --backend newton --output calibrated_newton.yaml

# Step 5b: calibrate MuJoCo against the same data
python scripts/run_calibration.py \
    --recordings-dir data/recordings/cal_run_01 \
    --backend mujoco --output calibrated_mujoco.yaml
```

### C — Compare Newton vs MuJoCo (no real robot)

```bash
python scripts/record_rollouts.py \
    --backends newton mujoco \
    --mode 2 --seed 42
# → data/rollouts/traj_m2_s42/newton.parquet
# → data/rollouts/traj_m2_s42/mujoco.parquet
```

Load both parquets and diff them with the `compare()` function in `src/elastic_sim/compare.py`.
