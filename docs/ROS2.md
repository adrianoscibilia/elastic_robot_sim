# ROS 2 execution and recording

The real-robot path assumes ROS 2 Jazzy, a running `joint_trajectory_controller`, a JointState source, a flange-mounted force/torque sensor, and rosbag2. ROS is intentionally imported lazily so simulation-only workflows work without ROS installed.

## Build and launch order

```bash
source /opt/ros/jazzy/setup.bash
colcon build --packages-select elastic_robot_sim
source install/setup.bash
```

Start, in the robot-specific order required by the hardware stack:

1. robot driver and state publisher;
2. `joint_trajectory_controller` with the configured action namespace;
3. flange force/torque broadcaster;
4. TF publishers if `ros.sensor.link_side_frame` is configured;
5. any motor lifecycle services;
6. optional extra topics.

The repository does not prescribe a universal hardware launch file. Topic names, action namespace, and motor services belong in the asset sim2real YAML.

Before motion, run the non-moving preflight:

```bash
ros2 run elastic_robot_sim run_experiment \
  --config config/assets/fmrr_tecnobody_sim2real.yaml \
  --preflight-only
```

For the actual collection, use `--real-only`; this prevents the limited
hardware session from also running the configured simulation backends.

## Required interfaces

The preflight requires these exact types:

| Interface | Default | Type/contract |
|---|---|---|
| Joint state | `/joint_states` | `sensor_msgs/msg/JointState`; every active joint must have position, velocity, and finite effort. |
| Flange wrench | `/ft_sensor_command_broadcaster/wrench` | `geometry_msgs/msg/WrenchStamped`; force, torque, timestamp, and non-empty frame ID. |
| Controller state | `/joint_trajectory_controller/state` | `control_msgs/msg/JointTrajectoryControllerState`; desired/actual/error fields are retained when published. |
| Action | `/joint_trajectory_controller/follow_joint_trajectory` | `control_msgs/action/FollowJointTrajectory`. |

The safety-default raw bag uses `ros2 bag record --all`, so it includes the
action's `send_goal`, `get_result`, `feedback`, `cancel_goal`, and `status`
topics, TF, and any other ROS topics published during the run. Set
`ros.bag.mode: selected` only for a deliberate diagnostic capture; in that
mode the explicit action topics, configured `extra_topics`, and TF topics are
recorded.

## Joint effort rule

The calibration effort channel is named `tau_joint_state_<joint>` and comes exclusively from the matching entry in `sensor_msgs/msg/JointState.effort`. It is not taken from controller desired effort, action feedback, a model torque, or a reconstructed value.

An empty effort array, a missing active-joint effort, or a non-finite effort aborts preflight. The same condition during motion becomes a capture error and aborts the run. There is no silent zero fallback.

## Flange wrench and link-side mapping

Raw flange force and torque are always stored with the incoming frame ID. The optional mapping can be:

```yaml
ros:
  sensor:
    link_side_frame: flange
    flange_to_link_side_axes: [0, 1, 2]
```

`flange_to_link_side_axes` is a permutation of `[0, 1, 2]` and is useful for sensor/controller axis conventions. When `link_side_frame` is configured, the recorder also attempts a TF rotation into that frame. A transform failure marks the transformed channel invalid but does not destroy the raw flange data.

For FMRR, the three Cartesian generalized link forces are explicitly mapped to the XYZ link-side force channels in simulation and real capture. The recorder never interprets a flange wrench as joint torque.

## Controller state

When published, desired, actual, and error points are preserved for positions, velocities, accelerations, and efforts. The recorder joins these events to JointState samples by source timestamp while retaining receipt timestamps. Planned trajectory samples are added from the saved `MaterializedTrajectory`, not regenerated.

## Motor lifecycle and failure cleanup

By default, the runner calls the configured enable Trigger service before the first goal and the disable Trigger service after execution. Disable is attempted again in the `finally` cleanup path. The preflight checks both services before motion. Use `--no-motor-control` only when the surrounding test setup owns the motor lifecycle.

The runner starts rosbag2 before motor enable, checks that the process remains
alive during every action, stops it with SIGINT, checks its exit status, and
stores stdout/stderr under `raw/`. A bag failure aborts the run and prevents a
successful manifest. Inspect `raw/rosbag2.stderr.log` when bag startup fails.

## Useful checks before running

```bash
ros2 action list | Select-String FollowJointTrajectory   # PowerShell
ros2 topic info /joint_states
ros2 topic info /ft_sensor_command_broadcaster/wrench
ros2 topic info /joint_trajectory_controller/state
ros2 topic echo /joint_states --once
ros2 topic echo /ft_sensor_command_broadcaster/wrench --once
```

On Linux, replace `Select-String` with `grep`. Verify that the JointState sample contains the exact configured active joint names and a populated effort array before starting the experiment.
