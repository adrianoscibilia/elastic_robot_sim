# elastic_robot_sim

`elastic_robot_sim` is a sim-to-real system-identification framework for elastic robots. It generates joint-space excitation trajectories, saves the exact samples, runs those samples in Newton and MuJoCo, executes the same samples through a ROS 2 `joint_trajectory_controller`, records the robot signals and raw rosbag2 data, and calibrates simulator parameters against the real run.

The workflow is asset-based and uses only newly simulated trajectories or recordings collected from real robots. The reference trajectory is the input; joint effort and flange wrench are observations used by the loss when configured.

## Start here

### Install the Python environment

From the repository root:

```bash
uv sync
```

The default environment contains everything needed for Newton and MuJoCo simulation. Add `--group calibration` for optimizers or `--group dev` for pytest.

The examples use POSIX shell continuation (`\`). In PowerShell, run the command on one line or replace each continuation character with a backtick (`` ` ``).

Newton uses a CUDA-capable GPU for the physics backend. MuJoCo is the easiest backend to test on a CPU. ROS 2 is not installed by uv; it is an external system dependency.

### Check a configuration without running a simulator or robot

```bash
uv run python scripts/run_experiment.py \
  --config config/assets/fmrr_tecnobody_sim2real.yaml \
  --dry-run
```

### Run a simulation-only experiment

```bash
uv run python scripts/run_experiment.py \
  --config config/assets/fmrr_tecnobody_sim2real.yaml \
  --sim-only \
  --backends newton mujoco
```

### Record a real experiment

Build and source the ROS 2 package first, start the robot controller and flange sensor, then run:

```bash
ros2 run elastic_robot_sim run_experiment \
  --config config/assets/fmrr_tecnobody_sim2real.yaml
```

The runner performs topic/type/data preflight before enabling motors. It saves the trajectory before motion, starts rosbag2, executes the saved points through `FollowJointTrajectory`, records the configured signals, disables the motors, and writes a manifest.

For a controlled test that does not call the configured motor services:

```bash
ros2 run elastic_robot_sim run_experiment \
  --config config/assets/fmrr_tecnobody_sim2real.yaml \
  --real-only --no-motor-control
```

### Calibrate the model

After at least two complete real trajectories have been recorded:

```bash
uv run python scripts/run_calibration.py \
  --config config/assets/fmrr_tecnobody_sim2real.yaml
```

Calibration rejects incomplete runs and configuration-hash mismatches, splits complete trajectories into deterministic train and validation sets, runs every configured optimizer for every selected backend, and selects the result with the lowest held-out validation loss.

## The two primary commands

### `run_experiment.py`

```text
run_experiment.py --config CONFIG
                      [--sim-only | --real-only]
                      [--dry-run]
                      [--backends newton mujoco ...]
                      [--no-motor-control]
                      [--seed SEED]
                      [--run-id RUN_ID]
```

The default runs both configured simulators and the real-robot path. `--sim-only` skips ROS. `--real-only` skips simulation. `--dry-run` validates the asset and materializes metadata without creating a run or contacting ROS. `--run-id` is useful when a deterministic run directory name is required.

The command always generates a materialized trajectory from YAML first. Simulators and ROS consume those saved arrays; they do not independently regenerate a trajectory from the seed.

### `run_calibration.py`

```text
run_calibration.py --config CONFIG
                       [--methods cma bo skrl all ...]
                       [--backends newton mujoco ...]
                       [--recorded-root PATH]
                       [--max-evals N]
                       [--seed SEED]
                       [--output-root PATH]
```

`--methods all` expands to CMA-ES, Bayesian optimization, and skrl PPO. Dependencies are checked before any evaluations begin. All methods receive the same parameter registry, bounds, initial values, loss weights, and train/validation split.

## Output layout

Artifact kind is always the first directory and robot identity is always second:

```text
data/
├── simulated/<asset>/YYYY-MM-DD/<run_timestamp>/
│   ├── manifest.yaml
│   ├── trajectory.json
│   ├── trajectories/
│   ├── sim_newton.parquet
│   └── sim_mujoco.parquet
└── recorded/<asset>/YYYY-MM-DD/<run_timestamp>/
    ├── manifest.yaml
    ├── trajectory.json
    ├── trajectories/
    ├── real.parquet
    ├── observations.parquet
    └── raw/rosbag2/
```

`real.parquet` is the calibration-ready normalized recording. `observations.parquet` currently preserves the same joined frame and is reserved for broader channels and future losses. Both include source/receipt timing, validity fields, joint-state channels, controller channels, wrench channels, planned reference samples, trajectory IDs, and action result metadata when the real path is used.

Calibration outputs are stored separately:

```text
data/calibrations/<asset>/<calibration_timestamp>/
├── report.json
├── calibrated_newton.yaml
├── calibrated_mujoco.yaml
├── calibrated_newton_<method>.yaml
├── calibrated_mujoco_<method>.yaml
├── history_newton_<method>.json
└── history_mujoco_<method>.json
```

The manifest records the asset and configuration hashes, parameter values, trajectory metadata, ROS topic names and types, joint order, timestamps, completion status, sensor-frame mapping, software information, bag path, units, and validity statistics.

## Configured assets

| Configuration | Asset | Active-joint convention | Intended use |
|---|---|---|---|
| `fmrr_tecnobody_sim2real.yaml` | FMRR Tecnobody | `joint_x`, `joint_y`, `joint_z` | Dedicated Cartesian elastic model and flange-force mapping |
| `ur10_sim2real.yaml` | UR10 | URDF joint names | Generic serial elastic model |
| `tiago_pro_dual_sim2real.yaml` | TIAGo Pro dual arm | Both seven-joint arms | Generic serial elastic model |
| `kuka_lbr_iiwa_7_r800_sim2real.yaml` | KUKA LBR iiwa 7 R800 | Seven arm joints | Generic serial elastic model |
| `kuka_lbr_iiwa_14_r820_sim2real.yaml` | KUKA LBR iiwa 14 R820 | Seven arm joints | Generic serial elastic model |

Each sim2real YAML owns the asset, trajectory, simulation backends, model defaults, calibration registry, loss weights, ROS topics/services, sensor mapping, and three artifact roots. Copy one when creating an experiment; do not hardcode robot joint names in Python.

## How the workflow fits together

```text
YAML config
    │
    ├── asset resolver ── URDF, active joints, limits
    │
    ├── trajectory generator ── exact MaterializedTrajectory
    │                                │
    │                                ├── Newton rollout ── sim_newton.parquet
    │                                ├── MuJoCo rollout ── sim_mujoco.parquet
    │                                └── ROS action + recorder ── real.parquet,
    │                                      observations.parquet, raw/rosbag2
    │
    └── parameter registry + paired artifacts
                   │
                   └── CMA-ES / BO / skrl ── validation-based calibrated YAML
```

The FMRR model preserves its Cartesian transmission semantics. Serial assets use the generic URDF-based motor/transmission/link model. Both expose the same rollout contract and use the same saved trajectory samples.

## Configuration overview

A sim2real file has these sections:

```yaml
asset: ur10
paths:
  simulated_root: data/simulated
  recorded_root: data/recorded
  calibrations_root: data/calibrations

trajectory:
  mode: ptp                 # hold, sin/sinusoidal, or ptp
  num_trajectories: 3
  duration: 10.0
  time_step: 0.01
  seed: 20260903
  ptp:
    waypoints: 5
    limit_margin: 0.12

simulation:
  backends: [newton, mujoco]
  time_step: 0.004

model:
  mode: elastic
  default_stiffness: 1500.0
  default_damping: 40.0
  motor_stiffness: 3000.0
  motor_damping: 100.0
  intermediate_mass: 0.10

calibration:
  train_fraction: 0.67
  methods: [cma, bo, skrl]
  max_evals: 100
  loss_weights:
    position: 1.0
    velocity: 0.3
    effort: 0.1
    force: 0.1
    torque: 0.1
    motor_state: 0.0
  parameters: []       # empty means safe per-joint defaults for serial assets

ros:
  action_server: /joint_trajectory_controller/follow_joint_trajectory
  topics:
    joint_states: /joint_states
    flange_wrench: /ft_sensor_command_broadcaster/wrench
    controller_state: /joint_trajectory_controller/state
  motor_services:
    enable: /ethercat_checker/start_motors
    disable: /ethercat_checker/stop_motors
  extra_topics: []
```

See [docs/CONFIGURATION.md](docs/CONFIGURATION.md) for every supported field and parameter-registry examples.

## ROS 2 recording contract

Before motion, the runner requires these topics with these message types:

| Signal | Default topic | Required type | Required data |
|---|---|---|---|
| Joint state | `/joint_states` | `sensor_msgs/msg/JointState` | active-joint position, velocity, and finite non-empty effort |
| Flange wrench | `/ft_sensor_command_broadcaster/wrench` | `geometry_msgs/msg/WrenchStamped` | force, torque, timestamp, non-empty frame ID |
| Controller state | `/joint_trajectory_controller/state` | `control_msgs/msg/JointTrajectoryControllerState` | desired/actual/error fields when published |

JointState effort is never replaced by zero and is never inferred from controller effort. The flange wrench is retained in its original frame; an optional TF transform and/or axis permutation creates the configured link-side channel. A flange wrench is not silently converted into joint torque.

The raw bag also records the FollowJointTrajectory action topics and every configured `extra_topics` entry. See [docs/ROS2.md](docs/ROS2.md) for setup, preflight behavior, safety, and topic customization.

## Simulation-only tools

The unified command is preferred, but these tools remain useful for asset inspection and synthetic data generation:

```bash
# Validate or simulate one asset with a generated/replayed trajectory
uv run python scripts/run_asset_simulation.py \
  --asset ur10 --backend mujoco --dynamics elastic --dry-run

# Generate a standalone serial-arm trajectory
uv run python scripts/generate_serial_trajectory.py \
  --urdf assets/robots/ur10/description/ur10.urdf \
  --output data/ur10_trajectory.json

# Generate a synthetic CSV batch for controller/data experiments
uv run python scripts/run_asset_dataset_generation.py \
  --asset ur10 --backend mujoco --output-dir data/ur10_synthetic --trials 20
```

These commands are not calibration inputs. Calibration discovers only complete paired experiment directories produced by `run_experiment.py`.

## Repository map

The concise module guide is in [docs/MODULES.md](docs/MODULES.md). At a glance:

```text
config/assets/*_sim2real.yaml   primary experiment configurations
scripts/run_experiment.py       materialize, simulate, execute, record
scripts/run_calibration.py      CMA-ES, BO, skrl, validation, reports
src/elastic_sim/experiment.py   YAML, trajectories, Parquet, manifests, rosbag launch
src/elastic_sim/materialized.py exact trajectory data model and JSON round-trip
src/elastic_sim/ros_experiment.py ROS preflight, action execution, signal capture
src/elastic_sim/sim2real.py     backend-neutral rollouts and calibration loss
src/elastic_sim/parameter_registry.py named physical parameter registry
src/elastic_sim/sim_runner.py   FMRR Newton backend
src/elastic_sim/mujoco_runner.py FMRR MuJoCo backend
src/elastic_sim/generic_*_runner.py generic serial-asset backends
src/elastic_sim/optimizers/     CMA-ES, Bayesian optimization, skrl adapters
tests/                          trajectory, asset, simulator, optimizer, workflow tests
```

## ROS 2 package build

ROS 2 is only needed for the real-robot command. From a sourced ROS 2 Jazzy shell:

```bash
colcon build --packages-select elastic_robot_sim
source install/setup.bash
```

The package installs `run_experiment`, `run_calibration`, the `elastic_sim` Python package, configuration files, and the portable robot descriptions. The Python scripts can also be run directly from the repository with `uv run`.

## Development and tests

```bash
uv sync --all-groups
uv run python -m pytest -q
python -m compileall -q src scripts
```

The test suite does not require ROS 2 hardware. Real execution additionally requires ROS 2 Jazzy, `ros2_control`/`joint_trajectory_controller`, `control_msgs`, `trajectory_msgs`, `sensor_msgs`, `geometry_msgs`, `std_srvs`, `tf2_ros` when TF mapping is enabled, and rosbag2.

For a guide to adding a new robot, see [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).

## Documentation index

- [Workflow guide](docs/WORKFLOW.md) — installation, dry runs, simulation, ROS execution, and calibration.
- [Configuration reference](docs/CONFIGURATION.md) — trajectory, model, optimizer, loss, ROS, and registry settings.
- [Data format](docs/DATA_FORMAT.md) — trajectory JSON, Parquet channels, manifest, and calibration history.
- [ROS 2 guide](docs/ROS2.md) — controller/sensor requirements, preflight, bagging, and safety.
- [Module guide](docs/MODULES.md) — what each Python module and script does.
- [Development guide](docs/DEVELOPMENT.md) — adding assets, running tests, and extending backends.
- [Asset provenance](assets/README.md) — robot descriptions, licenses, and external sources.
- [Refactor specifications](REFACTOR_SPECS/round2/R2_00_OVERVIEW_AND_REVIEW.md) — design history and engineering notes.

## Scope and limitations

- The default calibration is reference-trajectory tracking, not open-loop torque replay.
- Payload remains fixed at `0.0` and excluded from the initial registries.
- Geometry, gravity, solver settings, collision settings, and integration conditions are fixed calibration conditions unless a backend/config explicitly changes them.
- Newton may require a compatible CUDA installation; MuJoCo is the more portable smoke-test backend.
- Serial-asset simulations currently expose joint/link states and efforts. Flange force/torque loss terms contribute only when the selected backend and asset provide matching mapped channels; FMRR has the explicit Cartesian link-force mapping.
- The repository contains legacy standalone and synthetic-dataset utilities for simulation experiments. They are not the source of truth for sim-to-real calibration.
