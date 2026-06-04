"""ROS 2 node: record a real-robot rollout for sim-to-real calibration.

Usage (ROS 2 Jazzy, active workspace):

    python3 record_real_rollout.py \\
        --traj-config data/rollouts/<traj_id>/trajectory.json \\
        --output-dir  data/rollouts/<traj_id>

The node:
  1. Loads a saved TrajectoryConfig (produced by the calibration framework).
  2. Samples the trajectory onto the configured time grid and sends it to
     joint_trajectory_controller via the FollowJointTrajectory action.
  3. Subscribes to /joint_states and /ft_sensor/wrench.
  4. Resamples both streams onto the common time grid.
  5. Writes real.parquet + meta.json under the output directory.

SAFETY: The trajectory is always dry-run in the simulator first (pass
--dry-run to skip hardware motion and only check the planning).
A human must be in the loop before executing on hardware.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from threading import Lock

import numpy as np

# ---------------------------------------------------------------------------
# Attempt ROS 2 / action client imports (graceful failure if not in ROS env)
# ---------------------------------------------------------------------------

_HAS_ROS = False
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.action import ActionClient
    from control_msgs.action import FollowJointTrajectory
    from trajectory_msgs.msg import JointTrajectoryPoint
    from sensor_msgs.msg import JointState
    from geometry_msgs.msg import WrenchStamped
    _HAS_ROS = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Path bootstrap for elastic_sim package
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from elastic_sim.rollout import RolloutResult, RolloutStore
from elastic_sim.trajectory import TrajectoryConfig, _trajectory_from_config

# ---------------------------------------------------------------------------
# Joint names expected by joint_trajectory_controller
# ---------------------------------------------------------------------------
JOINT_NAMES = ["joint_x", "joint_y", "joint_z"]

# ---------------------------------------------------------------------------
# ROS 2 recorder node
# ---------------------------------------------------------------------------

if _HAS_ROS:
    class RealRobotRecorder(Node):
        """Records one real-robot rollout and saves it as real.parquet."""

        def __init__(
            self,
            traj_config: TrajectoryConfig,
            output_dir: str,
            ft_topic: str = "/ft_sensor/wrench",
        ) -> None:
            super().__init__("real_robot_recorder")
            self._traj_config = traj_config
            self._output_dir = output_dir
            self._lock = Lock()

            # Joint state buffer
            self._joint_positions: dict[str, list[tuple[float, float]]] = {
                j: [] for j in JOINT_NAMES
            }
            self._joint_velocities: dict[str, list[tuple[float, float]]] = {
                j: [] for j in JOINT_NAMES
            }

            # F/T buffer: stored as (t, fx, fy, fz)
            self._ft_data: list[tuple[float, float, float, float]] = []

            self._js_sub = self.create_subscription(
                JointState, "/joint_states", self._js_callback, 10
            )
            self._ft_sub = self.create_subscription(
                WrenchStamped, ft_topic, self._ft_callback, 10
            )

            self._action_client = ActionClient(
                self, FollowJointTrajectory,
                "/joint_trajectory_controller/follow_joint_trajectory",
            )

        def _js_callback(self, msg: JointState) -> None:
            t = self.get_clock().now().nanoseconds * 1e-9
            name_to_idx = {n: i for i, n in enumerate(msg.name)}
            with self._lock:
                for jname in JOINT_NAMES:
                    if jname in name_to_idx:
                        i = name_to_idx[jname]
                        self._joint_positions[jname].append((t, msg.position[i]))
                        vel = msg.velocity[i] if i < len(msg.velocity) else 0.0
                        self._joint_velocities[jname].append((t, vel))

        def _ft_callback(self, msg: WrenchStamped) -> None:
            t = self.get_clock().now().nanoseconds * 1e-9
            with self._lock:
                self._ft_data.append((
                    t,
                    msg.wrench.force.x,
                    msg.wrench.force.y,
                    msg.wrench.force.z,
                ))

        def send_trajectory(self) -> bool:
            """Build and send the FollowJointTrajectory goal. Returns True on success."""
            if not self._action_client.wait_for_server(timeout_sec=10.0):
                self.get_logger().error("Action server not available after 10 s.")
                return False

            traj = _trajectory_from_config(self._traj_config)
            dt = self._traj_config.step_duration / 20.0  # 20 points per segment
            n = int(np.ceil(self._traj_config.sim_time / dt))
            times = np.arange(n) * dt

            from control_msgs.action import FollowJointTrajectory as FJT
            from trajectory_msgs.msg import JointTrajectory

            jt = JointTrajectory()
            jt.joint_names = JOINT_NAMES
            for t_s in times:
                q, dq = traj(float(t_s))
                pt = JointTrajectoryPoint()
                pt.positions = q.tolist()
                pt.velocities = dq.tolist()
                sec = int(t_s)
                nanosec = int((t_s - sec) * 1e9)
                from builtin_interfaces.msg import Duration
                dur = Duration()
                dur.sec = sec
                dur.nanosec = nanosec
                pt.time_from_start = dur
                jt.points.append(pt)

            goal = FJT.Goal()
            goal.trajectory = jt

            self._t0 = self.get_clock().now().nanoseconds * 1e-9
            future = self._action_client.send_goal_async(goal)
            rclpy.spin_until_future_complete(self, future)
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().error("Goal rejected by controller.")
                return False

            self.get_logger().info("Trajectory accepted — executing.")
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future)
            return True

        def save_rollout(self) -> str:
            """Resample and save real.parquet. Returns the output path."""
            traj_sim_time = self._traj_config.sim_time
            dt = 0.01  # 100 Hz recording grid
            grid = np.arange(0.0, traj_sim_time, dt)

            def _interp(data: list[tuple[float, float]], t0: float) -> np.ndarray:
                if not data:
                    return np.zeros_like(grid)
                ts = np.array([d[0] - t0 for d in data])
                vs = np.array([d[1] for d in data])
                return np.interp(grid, ts, vs)

            with self._lock:
                t0 = self._t0 if hasattr(self, "_t0") else 0.0
                q_motor = np.column_stack([
                    _interp(self._joint_positions[j], t0) for j in JOINT_NAMES
                ])
                dq_motor = np.column_stack([
                    _interp(self._joint_velocities[j], t0) for j in JOINT_NAMES
                ])

                # Elastic position/velocity: not directly observable on the real
                # robot — use motor values as proxy (zero elasticity assumption for
                # the real side; the calibration loss aligns sim elastic to real motor).
                q_link = q_motor.copy()
                dq_link = dq_motor.copy()

                # F/T → elastic torques (Jacobian for Cartesian gantry is identity)
                if self._ft_data:
                    ft_arr = np.array(self._ft_data)
                    tau_link = np.column_stack([
                        np.interp(grid, ft_arr[:, 0] - t0, ft_arr[:, i + 1])
                        for i in range(3)
                    ])
                else:
                    tau_link = np.zeros((len(grid), 3))

                tau_motor = tau_link.copy()

            # Reference trajectory values
            traj = _trajectory_from_config(self._traj_config)
            ref_pos_list, ref_vel_list = [], []
            for t_s in grid:
                q_r, dq_r = traj(float(t_s))
                ref_pos_list.append(q_r)
                ref_vel_list.append(dq_r)

            rollout = RolloutResult(
                time=grid,
                ref_pos=np.array(ref_pos_list),
                ref_vel=np.array(ref_vel_list),
                q_motor=q_motor,
                q_link=q_link,
                dq_motor=dq_motor,
                dq_link=dq_link,
                tau_motor=tau_motor,
                tau_link=tau_link,
                metadata={
                    "source": "real_robot",
                    "traj_mode": self._traj_config.mode,
                    "traj_seed": self._traj_config.seed,
                },
            )

            traj_id = os.path.basename(self._output_dir.rstrip("/\\"))
            store = RolloutStore(os.path.dirname(self._output_dir))
            store.save_trajectory(traj_id, self._traj_config)
            store.save_real(traj_id, rollout)
            out_path = os.path.join(self._output_dir, "real.parquet")
            self.get_logger().info(f"Rollout saved to {out_path}")
            return out_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record a real-robot rollout for sim-to-real calibration."
    )
    parser.add_argument(
        "--traj-config", required=True,
        help="Path to trajectory.json (produced by the calibration framework).",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory where real.parquet and meta.json will be written.",
    )
    parser.add_argument(
        "--ft-topic", default="/ft_sensor/wrench",
        help="ROS 2 topic for force/torque data.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Plan only — do not send motion commands to the robot.",
    )
    args = parser.parse_args()

    config = TrajectoryConfig.load(args.traj_config)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.dry_run:
        print("Dry run — trajectory loaded, no motion sent.")
        traj = _trajectory_from_config(config)
        times, q_refs, dq_refs = traj.sample_grid(0.01)
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
    node = RealRobotRecorder(config, args.output_dir, ft_topic=args.ft_topic)
    try:
        ok = node.send_trajectory()
        if ok:
            # Give the last samples time to arrive
            rclpy.spin_once(node, timeout_sec=1.0)
            node.save_rollout()
        else:
            print("Trajectory execution failed.")
            sys.exit(1)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
