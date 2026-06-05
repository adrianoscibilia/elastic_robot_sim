"""ROS 2 recording node: shared between record_real_rollout and collect_dataset."""
from __future__ import annotations

import os
from threading import Lock

import numpy as np

_HAS_ROS = False
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.action import ActionClient
    from builtin_interfaces.msg import Duration
    from control_msgs.action import FollowJointTrajectory
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from sensor_msgs.msg import JointState
    from geometry_msgs.msg import WrenchStamped
    _HAS_ROS = True
except ImportError:
    pass

from .rollout import RolloutResult, RolloutStore
from .trajectory import TrajectoryConfig, _trajectory_from_config

JOINT_NAMES = ["joint_x", "joint_y", "joint_z"]
FT_TOPIC_DEFAULT = "/ft_sensor_command_broadcaster/wrench"
ACTION_SERVER = "/joint_trajectory_controller/follow_joint_trajectory"


if _HAS_ROS:
    class RealRobotRecorder(Node):
        """Sends one FollowJointTrajectory goal and records joint-state + F/T data.

        Create one instance per trajectory, or call reset_buffers() between runs.
        """

        def __init__(
            self,
            traj_config: TrajectoryConfig,
            output_dir: str = "",
            ft_topic: str = FT_TOPIC_DEFAULT,
        ) -> None:
            super().__init__("real_robot_recorder")
            self._traj_config = traj_config
            self._output_dir = output_dir
            self._lock = Lock()
            self._reset_buffers()

            self._js_sub = self.create_subscription(
                JointState, "/joint_states", self._js_callback, 10
            )
            self._ft_sub = self.create_subscription(
                WrenchStamped, ft_topic, self._ft_callback, 10
            )
            self._action_client = ActionClient(self, FollowJointTrajectory, ACTION_SERVER)

        def _reset_buffers(self) -> None:
            self._joint_positions: dict[str, list[tuple[float, float]]] = {
                j: [] for j in JOINT_NAMES
            }
            self._joint_velocities: dict[str, list[tuple[float, float]]] = {
                j: [] for j in JOINT_NAMES
            }
            self._ft_data: list[tuple[float, float, float, float]] = []
            self._t0: float = 0.0

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
            dt = self._traj_config.step_duration / 20.0  # 20 points per PTP segment
            n = int(np.ceil(self._traj_config.sim_time / dt))
            times = np.arange(n) * dt

            jt = JointTrajectory()
            jt.joint_names = JOINT_NAMES
            for t_s in times:
                q, dq = traj(float(t_s))
                pt = JointTrajectoryPoint()
                pt.positions = q.tolist()
                pt.velocities = dq.tolist()
                sec = int(t_s)
                nanosec = int((t_s - sec) * 1e9)
                dur = Duration()
                dur.sec = sec
                dur.nanosec = nanosec
                pt.time_from_start = dur
                jt.points.append(pt)

            goal = FollowJointTrajectory.Goal()
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

        def save_rollout(self, output_file: str | None = None) -> str:
            """Resample recorded signals onto a 100 Hz grid and write parquet.

            Pass output_file for a flat-layout path; omit to use the legacy
            RolloutStore layout under self._output_dir.
            """
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
                t0 = self._t0
                q_motor = np.column_stack([
                    _interp(self._joint_positions[j], t0) for j in JOINT_NAMES
                ])
                dq_motor = np.column_stack([
                    _interp(self._joint_velocities[j], t0) for j in JOINT_NAMES
                ])
                # On the real robot the elastic displacement is not directly
                # observable; use motor values as proxy for the real side.
                q_link = q_motor.copy()
                dq_link = dq_motor.copy()
                if self._ft_data:
                    ft_arr = np.array(self._ft_data)
                    tau_link = np.column_stack([
                        np.interp(grid, ft_arr[:, 0] - t0, ft_arr[:, i + 1])
                        for i in range(3)
                    ])
                else:
                    tau_link = np.zeros((len(grid), 3))
                tau_motor = tau_link.copy()

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

            if output_file is not None:
                os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
                rollout.to_dataframe().to_parquet(output_file, index=False)
                out_path = output_file
            else:
                traj_id = os.path.basename(self._output_dir.rstrip("/\\"))
                store = RolloutStore(os.path.dirname(self._output_dir))
                store.save_trajectory(traj_id, self._traj_config)
                store.save_real(traj_id, rollout)
                out_path = os.path.join(self._output_dir, "real.parquet")

            self.get_logger().info(f"Rollout saved to {out_path}")
            return out_path
