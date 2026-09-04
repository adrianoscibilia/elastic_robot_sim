# Data format and artifact contract

The experiment directory is the handoff boundary between execution and calibration. A run is usable only when its manifest says `completion_status: complete`, its configuration hash matches the calibration config, and its trajectory and `real.parquet` files are present.

## `trajectory.json`

Each materialized trajectory contains:

```json
{
  "schema_version": 1,
  "joint_names": ["joint_x", "joint_y", "joint_z"],
  "time": [0.0, 0.01],
  "position": [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
  "velocity": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
  "acceleration": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
  "metadata": {
    "asset": "fmrr_tecnobody",
    "seed": 20260903,
    "mode": "ptp",
    "workspace": {},
    "config_hash": "..."
  }
}
```

The actual arrays may be longer. `MaterializedTrajectory.load()` validates joint count, monotonic finite time, shapes, and finite numeric values. Its `digest()` is useful for verifying that two consumers received the same trajectory.

## Parquet naming and columns

Columns are wide and joint names are part of the column name. For a joint called `joint_a`:

| Column | Meaning |
|---|---|
| `t` | Source time for simulation or relative source time for real capture |
| `source` | `sim_newton`, `sim_mujoco`, or real-capture source |
| `trajectory_id` | Index within the experiment run |
| `q_ref__joint_a` | Planned position |
| `dq_ref__joint_a` | Planned velocity |
| `ddq_ref__joint_a` | Planned acceleration when available |
| `q__joint_a`, `dq__joint_a` | Simulated or observed link/joint state |
| `q_motor__joint_a`, `dq_motor__joint_a` | Motor-side state when modeled/available |
| `tau_motor__joint_a` | Simulated motor/controller effort |
| `tau_link__joint_a` | Simulated transmission/link effort |
| `tau_joint_state__joint_a` | Real effort, sourced only from `/joint_states.effort` |

Wrench columns use `force_flange__x/y/z`, `torque_flange__x/y/z`, `force_link_side__x/y/z`, and `torque_link_side__x/y/z`. Real recordings additionally preserve `source_ns`, `receipt_ns`, `relative_t`, `frame_id`, `link_side_frame`, and validity columns such as `valid_joint_state`, `valid_wrench`, and `valid_link_side_wrench`.

Controller channels use names such as:

```text
controller_desired_positions__<joint>
controller_actual_velocities__<joint>
controller_error_accelerations__<joint>
controller_desired_efforts__<joint>
```

Only fields actually published by the controller are populated. The action result is repeated per real sample as `action_accepted`, `action_error_code`, and `action_error_string`.

## `real.parquet` versus `observations.parquet`

Both files are written by the experiment runner. `real.parquet` is the stable calibration input. `observations.parquet` is the broader joined observation frame and is intended to retain additional channels when future losses or analysis need them. Raw message provenance belongs in the rosbag; normalized Parquet data preserves source/receipt timestamps and validity masks.

## `manifest.yaml`

The manifest is the run's provenance record. Important fields include:

```text
asset
asset_urdf
config_path
config_hash
trajectory.count / joint_names / seed / digests
backends
parameters
ros.action_server / topics / topic_types
units
software
start_timestamp / end_timestamp
completion_status
raw_bag_path
error                         # only for incomplete runs
```

Calibration ignores incomplete, missing, and hash-mismatched runs rather than mixing incompatible conditions.

## Calibration history

Each `history_<backend>_<method>.json` contains optimizer history plus every model evaluation. An evaluation records normalized `theta`, decoded physical parameter values, aggregate loss, component losses, runtime, and any failure text. `report.json` records train/validation counts, experiment paths, selected method per backend, parameter values, and validation metrics.
