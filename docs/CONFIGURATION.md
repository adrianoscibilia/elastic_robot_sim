# Configuration reference

The primary configuration files are `config/assets/*_sim2real.yaml`. They are ordinary YAML files and are loaded directly by both commands. Paths are resolved relative to the configuration file where applicable.

## Top-level fields

| Field | Required | Description |
|---|---:|---|
| `schema_version` | no | Configuration schema marker. |
| `asset` | yes | Asset registry name or an asset YAML path. |
| `output_root` | no | Experiment root; default is `data/experiments`. |
| `trajectory` | no | Generator, count, timing, seed, and workspace. |
| `simulation` | no | Backends and simulator timestep. |
| `model` | no | Backend/model defaults and overrides. |
| `calibration` | no | Registry, loss, split, optimizer, and budget settings. |
| `ros` | no for sim-only | Action, topics, services, sensor mapping, and timeouts. |

## Asset and joint order

`asset` may be a registry name such as `ur10` or a path to an asset YAML. The asset YAML defines `active_joints`; an empty list means every non-mimic one-DoF joint in URDF declaration order. The resolved order is used everywhere. Do not reorder Parquet columns or ROS goal joints manually.

## Trajectory

```yaml
trajectory:
  mode: ptp                 # hold, sin, sinusoidal, or ptp
  num_trajectories: 3
  duration: 8.0
  time_step: 0.01
  seed: 20260903
  max_velocity: null        # optional global cap in asset units/s
  workspace:                # optional; otherwise URDF limits are used
    joint_a: [-1.0, 1.0]
  ptp:
    waypoints: 5
    limit_margin: 0.12
    step_duration: 2.0
  sinusoidal:
    amplitude_fraction: 0.25
    amplitude_min: 0.0
    frequency_min: 0.2
    frequency_max: 1.5
```

`hold` uses the workspace midpoint. `sin` generates per-joint sinusoidal motion with seeded amplitude, frequency, and phase. `ptp` generates seeded minimum-jerk segments between random waypoints inside the configured limits. Workspace keys must exactly match the active joints. Invalid limits or a trajectory outside the safe workspace are rejected.

## Simulation and model

```yaml
simulation:
  backends: [newton, mujoco]
  time_step: 0.004

model:
  mode: elastic          # elastic, rigid, or kinematic for serial assets
  default_stiffness: 1500.0
  default_damping: 40.0
  motor_stiffness: 3000.0
  motor_damping: 100.0
  intermediate_mass: 0.10
  intermediate_size: 0.20
  transmissions: {}
  body_overrides: {}
```

FMRR uses `model.base_params` to load its dedicated Cartesian `RobotParams` configuration. Its transmission values are adapted to the Cartesian model while preserving the common rollout interface. Serial assets use the generic URDF-based builders. Geometry, gravity, solver, collision, and integration settings should be treated as fixed conditions during identification.

## Parameter registry

Set `calibration.parameters` explicitly when the calibration scope must be controlled:

```yaml
calibration:
  parameters:
    - name: transmission.shoulder_pan_joint.stiffness
      lower: 500.0
      upper: 5000.0
      initial: 1500.0
      scale: log
      enabled: true
      backend: all
    - name: transmission.shoulder_pan_joint.damping
      lower: 1.0
      upper: 200.0
      initial: 40.0
      scale: log
    - name: body.upper_arm.mass
      lower: 1.0
      upper: 8.0
      initial: 4.0
      scale: linear
```

Supported naming patterns are:

```text
transmission.<joint>.stiffness
transmission.<joint>.damping
transmission.<joint>.motor_stiffness
transmission.<joint>.motor_damping
transmission.<joint>.intermediate_mass
transmission.<joint>.intermediate_inertia_x|y|z
body.<name>.mass
body.<name>.inertia_x|y|z
```

`scale: log` requires a positive lower bound. Optimizers operate in normalized `[-1, 1]` coordinates; the registry maps those values to physical units before a backend evaluation. An empty explicit registry creates safe per-joint transmission defaults for generic serial assets. Payload is supported by the model layer but is intentionally absent from the initial sim2real registries.

## Calibration and loss

```yaml
calibration:
  train_fraction: 0.67
  split_seed: 0
  methods: [cma, bo, skrl]
  max_evals: 100
  loss_weights:
    position: 1.0
    velocity: 0.3
    effort: 0.1
    force: 0.1
    torque: 0.1
    motor_state: 0.0
  cma:
    sigma0: 0.3
    popsize: null
    tolfun: 1.0e-7
    tolx: 1.0e-7
  bo:
    n_initial_points: 10
    acq_func: EI
    noise: gaussian
  skrl:
    algorithm: ppo
    context_dim: 4
    device: cpu
```

The loss compares normalized RMSE values. Effort uses the real `tau_joint_state_<joint>` channels and simulator motor effort. Force and torque prefer link-side channels and fall back to flange channels only when the mapping declares them available. Missing channels are skipped; if no positive-weight signal is available, the evaluation fails with a large loss and an error in history.

## ROS

```yaml
ros:
  action_server: /joint_trajectory_controller/follow_joint_trajectory
  topics:
    joint_states: /joint_states
    flange_wrench: /ft_sensor_command_broadcaster/wrench
    controller_state: /joint_trajectory_controller/state
  extra_topics:
    - name: /some/topic
      type: package/msg/Type
      required: false
  motor_services:
    enable: /ethercat_checker/start_motors
    disable: /ethercat_checker/stop_motors
  sensor:
    link_side_frame: flange
    flange_to_link_side_axes: [0, 1, 2]
  preflight_timeout: 5.0
  sample_timeout: 5.0
  action_timeout: 30.0
  bag_startup_delay: 0.5
```

The `required` flag on extra topics controls validation policy; all configured extra topics are still included in the raw bag. Required core topic types are fixed by the message contract and are validated before motion. See [ROS2.md](ROS2.md) for details.
