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

#### Option B — Continuous dataset collection (sims + real robot)

```bash
# 10 random trajectories: both sims + real robot, save with datetime stamps
python scripts/collect_dataset.py \
    --backends newton mujoco real \
    --num-trajectories 10 \
    --output-dir data/recordings/session_01

# Sim-only, run until Ctrl-C
python scripts/collect_dataset.py \
    --backends newton mujoco \
    --num-trajectories 0 \
    --output-dir data/recordings/sim_only

# Use previously calibrated params for the sim
python scripts/collect_dataset.py \
    --backends newton real \
    --params calibrated_newton.yaml \
    --num-trajectories 20 \
    --output-dir data/recordings/newton_cal
```

For each trajectory the script:
1. Picks a random mode (sinusoidal / PTP) and random seed
2. Saves `trajectory_TS.json`
3. **Starts the real robot as a background subprocess** (it executes in real-time, ~15 s)
4. **Runs the simulators sequentially** while the robot is moving (they finish in 1–2 s)
5. Waits for the robot subprocess to exit; if it fails, the sim files are kept

Output layout (all files flat in `--output-dir`):
```
data/recordings/session_01/
  trajectory_20240605_143022.json
  newton_20240605_143022.parquet
  mujoco_20240605_143022.parquet
  real_20240605_143022.parquet
  trajectory_20240605_143058.json
  ...
```

#### Real-robot recording (standalone, ROS 2 required)

```bash
# Source ROS 2 workspace first, then:
python real_robot/record_real_rollout.py \
    --traj-config data/rollouts/traj_m2_s42/trajectory.json \
    --output-dir  data/rollouts/traj_m2_s42/

# Dry-run: plan only, no motion
python real_robot/record_real_rollout.py \
    --traj-config data/rollouts/traj_m2_s42/trajectory.json \
    --output-dir  data/rollouts/traj_m2_s42/ \
    --dry-run
```

Subscribes to `/joint_states` and `/ft_sensor/wrench`, sends `FollowJointTrajectory` to `joint_trajectory_controller`, and writes `real.parquet` resampled onto a 100 Hz grid.

---

### Sim-to-real calibration

```bash
# CMA-ES, Newton vs real — from flat recordings directory
python scripts/run_calibration.py \
    --backend newton \
    --recordings-dir data/recordings/session_01

# Bayesian optimisation, MuJoCo vs real — from structured rollout store
python scripts/run_calibration.py \
    --backend mujoco --optimizer bo \
    --rollouts-dir data/rollouts

# Extra options
python scripts/run_calibration.py \
    --backend newton --optimizer cma \
    --recordings-dir data/recordings/session_01 \
    --max-evals 300 \
    --output calibrated_newton.yaml \
    --no-payload \
    --verbose
```

The script:
1. Loads all `(real rollout, trajectory config)` pairs from the recordings
2. Splits them into train / validation sets (80 / 20 by default)
3. Runs the chosen optimizer to minimise the normalised RMSE between sim and real
4. Validates on the held-out set
5. Saves the best params to `calibrated_<backend>.yaml`
6. Saves `<backend>_calibrated_TS.parquet` alongside the source files for visual comparison

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

The fidelity metric is a weighted normalised RMSE over position, velocity, and torque signals (configurable in `configs/calibration.yaml`).

---

## Configuration

### `config/settings.yaml`

Runtime parameters for standalone simulation and dataset generation:

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
```

### `configs/calibration.yaml`

Optimizer hyper-parameters and metric weights for `run_calibration.py`.

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
│   ├── settings.yaml              # Default sim parameters
│   └── elastic_cart_mujoco.xml    # MuJoCo scene template
├── configs/
│   └── calibration.yaml           # Calibration optimizer settings
├── scripts/
│   ├── sim_common.py              # Shared constants, trajectory generators, CSV utils
│   ├── elastic_cart_robot_newton.py   # Newton standalone simulation
│   ├── elastic_cart_robot_mujoco.py   # MuJoCo standalone simulation
│   ├── elastic_cart_robot_isaacsim.py # Isaac Sim standalone simulation
│   ├── run_dataset_generation.py  # Batch synthetic dataset generation (Newton)
│   ├── record_rollouts.py         # Record fixed-trajectory rollouts (sim backends)
│   ├── collect_dataset.py         # Continuous synchronized collection (sim + real)
│   └── run_calibration.py         # Sim-to-real calibration optimizer
├── src/elastic_sim/               # Calibration framework library
│   ├── params.py                  # RobotParams dataclass (normalize, bounds, YAML I/O)
│   ├── trajectory.py              # Trajectory classes + factory functions
│   ├── rollout.py                 # RolloutResult + RolloutStore (parquet I/O)
│   ├── compare.py                 # Fidelity metric (normalised RMSE)
│   ├── sim_runner.py              # Newton programmatic API
│   ├── mujoco_runner.py           # MuJoCo programmatic API
│   ├── calibration.py             # SimCalibrationProblem (loss function)
│   └── optimizers/
│       ├── base.py                # Optimizer ABC
│       ├── cma_backend.py         # CMA-ES
│       ├── bo_backend.py          # Bayesian optimisation
│       └── skrl_backend.py        # skrl PPO/SAC
├── real_robot/
│   └── record_real_rollout.py     # ROS 2 node: trajectory execution + recording
├── data/                          # Generated CSV datasets
├── data/rollouts/                 # Structured rollout store (record_rollouts.py)
├── data/recordings/               # Flat timestamped recordings (collect_dataset.py)
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

### Real robot recording
```
# ROS 2 Jazzy with:
#   joint_trajectory_controller
#   control_msgs  trajectory_msgs  sensor_msgs  geometry_msgs
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
# 1. Collect data (sims run in seconds; robot runs while they do)
python scripts/collect_dataset.py \
    --backends newton mujoco real \
    --num-trajectories 15 \
    --output-dir data/recordings/cal_run_01

# 2. Calibrate Newton against real
python scripts/run_calibration.py \
    --backend newton --optimizer cma \
    --recordings-dir data/recordings/cal_run_01 \
    --output calibrated_newton.yaml

# 3. Calibrate MuJoCo against real
python scripts/run_calibration.py \
    --backend mujoco --optimizer cma \
    --recordings-dir data/recordings/cal_run_01 \
    --output calibrated_mujoco.yaml
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
