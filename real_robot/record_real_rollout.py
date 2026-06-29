#!/usr/bin/env python3
"""ROS 2 node: record a single real-robot rollout for sim-to-real calibration.

Usage (ROS 2 Jazzy, sourced workspace):

    ros2 run elastic_robot_sim record_real_rollout.py \\
        --traj-config data/rollouts/<traj_id>/trajectory.json \\
        --output-dir  data/rollouts/<traj_id>

For batch dataset collection use collect_dataset instead:

    ros2 run elastic_robot_sim collect_dataset

The node:
  1. Loads a saved TrajectoryConfig (produced by the calibration framework).
  2. Samples the trajectory onto the configured time grid and sends it to
     joint_trajectory_controller via the FollowJointTrajectory action.
  3. Subscribes to /joint_states and /ft_sensor_command_broadcaster/wrench.
  4. Resamples both streams onto the common time grid.
  5. Writes real.parquet + meta.json under the output directory.

SAFETY: Pass --dry-run to skip hardware motion and only check the trajectory.
A human must be in the loop before executing on hardware.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrap for running from the source tree (no-op when installed)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from elastic_sim.trajectory import TrajectoryConfig, _trajectory_from_config

_HAS_ROS = False
try:
    import rclpy
    from elastic_sim.ros_recorder import RealRobotRecorder, FT_TOPIC_DEFAULT
    _HAS_ROS = True
except ImportError:
    FT_TOPIC_DEFAULT = "/ft_sensor_command_broadcaster/wrench"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record a single real-robot rollout for sim-to-real calibration."
    )
    parser.add_argument(
        "--traj-config", required=True,
        help="Path to trajectory.json (produced by the calibration framework).",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory where real.parquet will be written (legacy mode).",
    )
    parser.add_argument(
        "--output-file", default=None,
        help="Exact path for the output parquet file. Takes precedence over --output-dir.",
    )
    parser.add_argument(
        "--ft-topic", default=FT_TOPIC_DEFAULT,
        help="ROS 2 topic for force/torque data.",
    )
    parser.add_argument(
        "--speed-override", type=float, default=30.0, metavar="PCT",
        help="Trajectory speed as %% of nominal (1–100).",
    )
    parser.add_argument(
        "--record-rate-hz", type=float, default=100.0,
        help="Recording/resample rate in Hz (default: 100.0).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Plan only — do not send motion commands to the robot.",
    )
    args = parser.parse_args()

    if args.output_file is None and args.output_dir is None:
        parser.error("Provide --output-file or --output-dir.")

    config = TrajectoryConfig.load(args.traj_config)
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.dry_run:
        print("Dry run — trajectory loaded, no motion sent.")
        traj = _trajectory_from_config(config)
        dt = 1.0 / args.record_rate_hz
        times, q_refs, dq_refs = traj.sample_grid(dt)
        print(f"  Trajectory: mode={config.mode}, sim_time={config.sim_time:.1f}s, "
              f"{len(times)} steps")
        print(f"  Position range X: [{q_refs[:,0].min():.3f}, {q_refs[:,0].max():.3f}] m")
        print(f"  Position range Y: [{q_refs[:,1].min():.3f}, {q_refs[:,1].max():.3f}] m")
        print(f"  Position range Z: [{q_refs[:,2].min():.3f}, {q_refs[:,2].max():.3f}] m")
        return

    if not _HAS_ROS:
        print("ERROR: ROS 2 packages not found. Source your ROS 2 workspace first.")
        sys.exit(1)

    rclpy.init()
    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.output_file))
    node = RealRobotRecorder(
        config, output_dir,
        ft_topic=args.ft_topic,
        speed_override=args.speed_override,
        record_rate_hz=args.record_rate_hz,
    )
    try:
        ok = node.send_trajectory()
        if ok:
            rclpy.spin_once(node, timeout_sec=1.0)
            node.save_rollout(output_file=args.output_file)
        else:
            print("Trajectory execution failed.")
            sys.exit(1)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
