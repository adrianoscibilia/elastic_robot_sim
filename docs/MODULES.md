# Module and script guide

This page explains where to look when changing or debugging the repository.

## Primary Python modules

| Module | Responsibility |
|---|---|
| `experiment.py` | Loads sim2real YAML, resolves assets, validates limits, generates trajectories, writes Parquet/manifests, and launches rosbag2. |
| `materialized.py` | `MaterializedTrajectory`: exact time/position/velocity/acceleration arrays, interpolation, JSON serialization, and digest. |
| `ros_experiment.py` | Lazy ROS imports, topic/type preflight, JointState/wrench/controller capture, FollowJointTrajectory execution, motor lifecycle, and normalized observations. |
| `sim2real.py` | Backend-neutral simulation dispatch, rollout normalization, reference-tracking loss, experiment discovery, trajectory split, and calibration problem history. |
| `parameter_registry.py` | Named physical parameters, bounds, linear/log scaling, normalized optimizer coordinates, and model overrides. |
| `assets.py` | Asset YAML loading, URDF joint discovery, active-joint validation, resource checking, and the repository asset registry. |
| `serial_trajectory.py` | URDF-limit-based trajectory support for generic serial robots and compatibility trajectory generation. |
| `generic_newton_runner.py` | URDF-backed Newton serial model with kinematic, rigid, and elastic modes; supports transmission/body overrides. |
| `generic_mujoco_runner.py` | URDF-backed MuJoCo serial model and rollout implementation. |
| `sim_runner.py` | Dedicated FMRR Cartesian Newton model and rollout. |
| `mujoco_runner.py` | Dedicated FMRR Cartesian MuJoCo model and rollout. |
| `params.py` | FMRR `RobotParams`, axis stiffness/damping, motor gains, intermediate mass, and legacy YAML/vector helpers. |
| `rollout.py` | Legacy/general `RolloutResult` and `RolloutStore` Parquet API. The unified experiment workflow uses the wider `experiment.py` schema. |
| `trajectory.py` | Original Cartesian trajectory classes and compatibility generators. New sim2real runs use `MaterializedTrajectory`. |
| `compare.py` | Legacy rollout fidelity metrics and per-axis comparisons. |
| `calibration.py` | Legacy Cartesian `SimCalibrationProblem`; retained for compatibility, while the primary command uses `sim2real.py`. |
| `generic_calibration.py` | Legacy torque-replay calibration adapter; not used by the primary sim2real command. |
| `asset_dataset.py` | Legacy KUKA/Baxter dataset adapter; benchmark-data compatibility only, not a calibration input path. |
| `elastic_settings.py` | Configuration helpers for explicit serial motor-to-link elastic transmissions. |
| `ros_recorder.py` | Older ROS recorder retained for compatibility; new recording uses `ros_experiment.py`. |
| `__init__.py` | Package marker and top-level package description. |

## Optimizer modules

| Module | Responsibility |
|---|---|
| `optimizers/base.py` | Common optimizer interface: bounded objective, initial point, evaluation budget, and history. |
| `optimizers/cma_backend.py` | CMA-ES adapter; budget is passed to `minimize()` and not incorrectly used as a constructor option. |
| `optimizers/bo_backend.py` | `scikit-optimize` bounded Bayesian optimization adapter. |
| `optimizers/skrl_backend.py` | skrl PPO contextual-bandit adapter; trajectory summary context is used to propose parameter vectors. |

## Primary scripts

| Script | Use |
|---|---|
| `run_experiment.py` | The one command for materialization, simulation, ROS execution, recording, and manifests. |
| `run_calibration.py` | The one command for discovering paired runs and running all configured calibration methods. |

## Secondary simulation and data scripts

| Script | Use |
|---|---|
| `run_asset_simulation.py` | Inspect or simulate one portable asset, optionally replaying a saved trajectory. |
| `generate_serial_trajectory.py` | Generate a standalone serial-arm trajectory JSON. |
| `run_asset_dataset_generation.py` | Produce synthetic CSV trials for generic asset experiments; not used by calibration discovery. |
| `record_rollouts.py` | Older simulator comparison helper; prefer `run_experiment.py --sim-only`. |
| `elastic_cart_robot_newton.py` | Older standalone FMRR Newton viewer/CSV script. |
| `elastic_cart_robot_mujoco.py` | Older standalone FMRR MuJoCo viewer/CSV script. |
| `elastic_cart_robot_isaacsim.py` | Optional/legacy Isaac Sim integration. It is not one of the unified calibration backends. |
| `run_dataset_generation.py` | Older FMRR synthetic dataset sweep. It is not a sim2real input producer. |
| `compile_urdf_from_description.py` | Convert a ROS/xacro robot description to a flat URDF with usable mesh paths. |
| `view_urdf_mujoco.py` | Small MuJoCo URDF viewer utility. |
| `setup_sim_env.ps1` | Windows helper for preparing/checking the uv environment. |
| `sim_common.py` | Shared helpers for the older standalone Cartesian scripts and synthetic-data utilities. |

## Where to implement common changes

- Add a new trajectory type: `experiment.py`, `MaterializedTrajectory` tests, and `docs/CONFIGURATION.md`.
- Add a new asset: asset YAML/URDF under `assets/robots/`, a `*_sim2real.yaml`, and asset tests.
- Add a tunable physical parameter: `parameter_registry.py`, the relevant backend adapter, and a model application test.
- Add a signal to real capture: `ros_experiment.py`, the normalized schema documentation, and a ROS-independent capture test using message doubles.
- Add a simulator backend: implement the exact-trajectory rollout contract and dispatch it in `sim2real.py`; do not regenerate trajectories in the backend.
- Change calibration metrics: update `ReferenceTrajectoryCalibrationProblem`, configuration docs, and validation tests.
