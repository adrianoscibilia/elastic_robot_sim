# End-to-end workflow

This is the operational guide for a new experiment.

## 1. Install and validate

From the repository root:

```bash
uv sync --all-groups
uv run python scripts/run_experiment.py \
  --config config/assets/fmrr_tecnobody_sim2real.yaml --dry-run
```

The dry run resolves the asset, validates active joints and limits, generates deterministic trajectory metadata, and prints the configuration hash. It does not create an experiment directory, run a simulator, or contact ROS.

If only MuJoCo is available, request it explicitly:

```bash
uv run python scripts/run_experiment.py \
  --config config/assets/ur10_sim2real.yaml \
  --sim-only --backends mujoco
```

## 2. Understand trajectory materialization

The YAML defines a generator, not the final runtime input. The runner creates one `MaterializedTrajectory` per requested trajectory and saves:

- exact time samples;
- position, velocity, and acceleration arrays;
- joint names and order;
- generator type and seed;
- workspace/limit metadata;
- requested/effective speed;
- asset and configuration hashes.

The trajectory JSON is written before simulation or robot motion. Newton, MuJoCo, and the ROS action goal consume these arrays directly. This is what makes a backend comparison and a real replay an apples-to-apples comparison.

Supported generators are `hold`, `sin`/`sinusoidal`, and `ptp`. Joint count is taken from the asset, so the same code works for FMRR's three translational joints and serial arms with six or seven joints.

## 3. Run simulation

```bash
uv run python scripts/run_experiment.py \
  --config config/assets/fmrr_tecnobody_sim2real.yaml \
  --sim-only --backends newton mujoco
```

Each backend is run independently for every saved trajectory. Outputs are validated for finite values and written as a combined `sim_<backend>.parquet` with a `trajectory_id` column.

For asset inspection or a single standalone rollout:

```bash
uv run python scripts/run_asset_simulation.py \
  --asset ur10 --backend mujoco --dynamics elastic \
  --output data/ur10_trial.csv --seed 42
```

The unified experiment command is preferred for calibration because it creates the manifest/hash/storage contract expected by `run_calibration.py`.

## 4. Prepare ROS 2

The Python environment does not provide ROS 2. Install/source ROS 2 Jazzy and build the package:

```bash
source /opt/ros/jazzy/setup.bash
colcon build --packages-select elastic_robot_sim
source install/setup.bash
```

Start the robot controller, joint-state publisher, flange sensor broadcaster, TF tree if used, and any motor lifecycle services. Confirm the configured topics before running the experiment. The runner checks topic names and message types and waits for valid samples before motion.

## 5. Execute and record

```bash
ros2 run elastic_robot_sim run_experiment \
  --config config/assets/fmrr_tecnobody_sim2real.yaml
```

The sequence is:

1. Resolve and validate the asset/configuration.
2. Materialize and save every trajectory.
3. Run selected simulation backends.
4. Validate required ROS topics and first samples.
5. Start rosbag2.
6. Enable motors unless `--no-motor-control` is supplied.
7. Send each exact trajectory through `FollowJointTrajectory`.
8. Capture joint state, controller state, flange wrench, source/receipt timestamps, and configured extras.
9. Disable motors in normal and failure cleanup paths.
10. Stop rosbag2 and write `manifest.yaml`.

If any required preflight check fails, the command aborts before motion. If a runtime JointState effort becomes missing or non-finite, the capture records an error and aborts rather than substituting zero.

## 6. Calibrate

At least two complete trajectories are needed for the default held-out split:

```bash
uv run python scripts/run_calibration.py \
  --config config/assets/fmrr_tecnobody_sim2real.yaml
```

Useful development options:

```bash
uv run python scripts/run_calibration.py \
  --config config/assets/fmrr_tecnobody_sim2real.yaml \
  --methods cma --backends mujoco --max-evals 8
```

The command discovers runs under the configured `output_root`, requires `completion_status: complete`, requires `real.parquet` and trajectory JSON files, and requires an exact configuration hash match. It then:

- splits trajectories, never individual rows;
- creates one backend-neutral calibration problem;
- evaluates each optimizer with the same normalized parameter bounds;
- records every evaluation, component loss, runtime, and failure;
- scores each method on validation trajectories;
- writes per-method and selected backend parameter files.

The default loss weights are position, velocity, effort, force, torque, and optional motor-side state. A component is used only if both simulated and real mapped channels exist and its weight is nonzero.

## Common failure messages

| Message | Meaning | Action |
|---|---|---|
| `no complete, hash-compatible real experiment runs` | No usable paired run was found | Run the experiment with the same config and check `manifest.yaml`. |
| `calibration requires ... validation trajectory` | Too few trajectories for the requested split | Record more trajectories or set a suitable `train_fraction`. |
| `ROS preflight failed` | A required topic is missing or has the wrong type | Check controller/sensor namespaces and `ros.topics`. |
| `JointState effort ... missing` | The robot did not publish usable effort | Fix the JointState source; do not change it to zero. |
| `FollowJointTrajectory action server unavailable` | The controller action is not running at the configured namespace | Check the controller and `ros.action_server`. |
| `rosbag2 exited during startup` | The bag process failed before motion | Inspect `raw/rosbag2.stderr.log`. |
| Newton import/CUDA error | Newton/Warp or a compatible GPU is unavailable | Run MuJoCo first or install the required CUDA stack. |
