"""ROS 2 recording node: shared between record_real_rollout and collect_dataset."""
from __future__ import annotations

import os
import time
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

from .trajectory import TrajectoryConfig, _trajectory_from_config

JOINT_NAMES = ["joint_x", "joint_y", "joint_z"]
FT_TOPIC_DEFAULT = "/ft_sensor_command_broadcaster/wrench"
ACTION_SERVER = "/joint_trajectory_controller/follow_joint_trajectory"
_VEL_LIMIT_MS = 0.5  # m/s hard cap per joint, enforced before sending to controller


def _resample_for_controller(
    waypoint_times: np.ndarray,
    waypoint_positions: np.ndarray,
    dt: float = 0.1,
    speed_override_pct: float = 100.0,
    vel_limit: float = _VEL_LIMIT_MS,
) -> tuple:
    """Mirror of gui_trajectory_manager.resample_with_speed_override().

    Fits a clamped cubic spline through the waypoints, resamples at uniform dt,
    applies speed override scaling, and enforces the per-joint velocity limit by
    automatically reducing the effective speed factor if needed.

    Returns (times, positions, velocities, accelerations, effective_speed_pct).
    """
    from scipy.interpolate import CubicSpline  # deferred — not in system Python

    _t = np.asarray(waypoint_times, dtype=float)
    _P = np.asarray(waypoint_positions, dtype=float)

    # Clamped spline: zero velocity at both endpoints (controller requirement)
    splines = [
        CubicSpline(_t, _P[:, i], bc_type="clamped")
        for i in range(_P.shape[1])
    ]

    # Resample at uniform dt — identical to gui_trajectory_manager
    tk = np.linspace(_t[0], _t[-1], int(_t[-1] / dt))

    positions    = np.stack([s(tk) for s in splines], axis=1)
    velocity     = np.stack([s.derivative(1)(tk) for s in splines], axis=1)
    acceleration = np.stack([s.derivative(2)(tk) for s in splines], axis=1)

    # Determine the maximum speed factor allowed by the velocity limit.
    # The controller receives vel_nominal * factor, so:
    #   vel_nominal * factor ≤ vel_limit  →  factor ≤ vel_limit / max(vel_nominal)
    max_vel_nominal = float(np.abs(velocity).max())
    max_factor_from_limit = (vel_limit / max_vel_nominal) if max_vel_nominal > 0.0 else 1.0

    user_factor = float(np.clip(speed_override_pct / 100.0, 0.01, 1.0))
    factor = min(user_factor, max_factor_from_limit)

    # Apply scaling — same arithmetic as gui_trajectory_manager
    t_scaled   = tk / factor
    vel_scaled = velocity * factor
    acc_scaled = acceleration * (factor ** 2)

    return t_scaled, positions, vel_scaled, acc_scaled, factor * 100.0


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
            speed_override: float = 30.0,
        ) -> None:
            super().__init__("real_robot_recorder")
            # Stamp speed_override onto the config so the JSON carries it and
            # sim backends can reproduce the exact same scaled motion.
            traj_config.speed_override = float(speed_override)
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
            self._joint_efforts: dict[str, list[tuple[float, float]]] = {
                j: [] for j in JOINT_NAMES
            }
            self._ft_data: list[tuple[float, float, float, float]] = []
            self._t0: float = 0.0
            self._warned_missing_effort: bool = False

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
                        if i < len(msg.effort):
                            self._joint_efforts[jname].append((t, msg.effort[i]))
                        elif not self._warned_missing_effort:
                            self.get_logger().warn(
                                "/joint_states has no effort field for "
                                f"'{jname}' — commanded torque will be recorded as 0. "
                                "Check that the hardware's JointStateBroadcaster "
                                "exposes an effort state interface."
                            )
                            self._warned_missing_effort = True

        def _ft_callback(self, msg: WrenchStamped) -> None:
            t = self.get_clock().now().nanoseconds * 1e-9
            with self._lock:
                self._ft_data.append((
                    t,
                    msg.wrench.force.x,
                    msg.wrench.force.y,
                    msg.wrench.force.z,
                ))

        # ------------------------------------------------------------------
        # Trajectory helpers
        # ------------------------------------------------------------------

        def _wait_for_joint_state(self, timeout_sec: float = 5.0) -> np.ndarray | None:
            """Spin until at least one JointState message has been received."""
            deadline = time.monotonic() + timeout_sec
            while time.monotonic() < deadline:
                rclpy.spin_once(self, timeout_sec=0.1)
                with self._lock:
                    if all(self._joint_positions[j] for j in JOINT_NAMES):
                        return np.array([
                            self._joint_positions[j][-1][1] for j in JOINT_NAMES
                        ])
            self.get_logger().error("Timed out waiting for /joint_states.")
            return None

        @staticmethod
        def _build_jt_message(
            times: np.ndarray,
            positions: np.ndarray,
            velocities: np.ndarray,
            accelerations: np.ndarray,
        ) -> JointTrajectory:
            jt = JointTrajectory()
            jt.joint_names = JOINT_NAMES
            for i, t_s in enumerate(times):
                pt = JointTrajectoryPoint()
                pt.positions     = positions[i].tolist()
                pt.velocities    = velocities[i].tolist()
                pt.accelerations = accelerations[i].tolist()
                sec     = int(t_s)
                nanosec = int((t_s - sec) * 1e9)
                dur = Duration()
                dur.sec     = sec
                dur.nanosec = nanosec
                pt.time_from_start = dur
                jt.points.append(pt)
            return jt

        def _send_jt_goal(self, jt: JointTrajectory) -> int:
            """Submit a FollowJointTrajectory goal and wait for completion.

            Returns the error_code from the action result (0 = SUCCESSFUL).
            Returns -1 if the goal is rejected before execution starts.
            """
            goal = FollowJointTrajectory.Goal()
            goal.trajectory = jt
            future = self._action_client.send_goal_async(goal)
            rclpy.spin_until_future_complete(self, future)
            gh = future.result()
            if not gh.accepted:
                self.get_logger().error("Goal rejected by controller.")
                return -1
            rf = gh.get_result_async()
            rclpy.spin_until_future_complete(self, rf)
            return rf.result().result.error_code

        def _go_to_start(self, target: np.ndarray, max_speed: float = 0.15) -> bool:
            """Send a gentle pre-move to the trajectory start position.

            Without this, the controller immediately aborts the main trajectory
            because its first point (traj(0)) is the range-center, which can be
            far from where the robot actually is after a fault or previous abort.

            The pre-move trajectory starts exactly from the current joint position
            (first point at t=0 = current state) so the controller never sees a
            position error at t=0.
            """
            from scipy.interpolate import CubicSpline

            current = self._wait_for_joint_state(timeout_sec=5.0)
            if current is None:
                return False

            delta_max = float(np.abs(target - current).max())
            if delta_max < 0.01:
                self.get_logger().info("Already at trajectory start — skipping pre-move.")
                return True

            # Time for the pre-move: slow and conservative
            duration = max(3.0, delta_max / max_speed)
            self.get_logger().info(
                f"Pre-move to trajectory start: "
                f"Δmax={delta_max:.3f}m  duration={duration:.1f}s"
            )

            # Two-waypoint clamped spline: current (v=0) → target (v=0)
            # Produces a smooth S-curve with zero velocity at both ends.
            raw_times     = np.array([0.0, duration])
            raw_waypoints = np.array([current, target])
            splines = [
                CubicSpline(raw_times, raw_waypoints[:, i], bc_type="clamped")
                for i in range(3)
            ]

            dt = 0.1
            n_pts = max(4, int(duration / dt))
            times = np.linspace(0.0, duration, n_pts)
            pos_s = np.stack([s(times) for s in splines], axis=1)
            vel_s = np.stack([s.derivative(1)(times) for s in splines], axis=1)
            acc_s = np.stack([s.derivative(2)(times) for s in splines], axis=1)

            jt = self._build_jt_message(times, pos_s, vel_s, acc_s)
            ec = self._send_jt_goal(jt)
            if ec != 0:
                self.get_logger().error(f"Pre-move failed (code={ec}).")
                return False
            return True

        # ------------------------------------------------------------------

        def send_trajectory(self) -> bool:
            """Move to trajectory start, then execute the main goal. Returns True on success."""
            if not self._action_client.wait_for_server(timeout_sec=10.0):
                self.get_logger().error("Action server not available after 10 s.")
                return False

            traj = _trajectory_from_config(self._traj_config)

            # Move to trajectory start position before executing the random motion.
            # This prevents immediate tolerance aborts when the robot is away from
            # the trajectory's first point (e.g., after a fault or previous abort).
            if not self._go_to_start(traj(0.0)[0]):
                return False

            # Sample the raw trajectory at 10 Hz as waypoints for the spline.
            dt = 0.1
            n_pts = max(2, int(self._traj_config.sim_time / dt))
            waypoint_times = np.linspace(0.0, self._traj_config.sim_time, n_pts)
            waypoint_positions = np.stack(
                [traj(float(t_s))[0] for t_s in waypoint_times]
            )

            # Resample via clamped spline, apply speed override, enforce velocity cap.
            t_ctrl, pos_ctrl, vel_ctrl, acc_ctrl, effective_pct = _resample_for_controller(
                waypoint_times,
                waypoint_positions,
                dt=dt,
                speed_override_pct=self._traj_config.speed_override,
                vel_limit=_VEL_LIMIT_MS,
            )

            # Persist the effective speed override so that save_rollout() builds the
            # recording grid over the actual execution time (real_duration = sim_time / factor).
            requested_pct = self._traj_config.speed_override
            self._traj_config.speed_override = effective_pct

            if effective_pct < requested_pct - 0.1:
                self.get_logger().warn(
                    f"Speed override reduced {requested_pct:.1f}% → {effective_pct:.1f}% "
                    f"to keep all joints ≤ {_VEL_LIMIT_MS:.2f} m/s."
                )
            self.get_logger().info(
                f"Speed {effective_pct:.1f}%  "
                f"nominal {self._traj_config.sim_time:.1f}s → {t_ctrl[-1]:.1f}s on robot  "
                f"peak vel {float(np.abs(vel_ctrl).max()):.3f} m/s"
            )

            jt = self._build_jt_message(t_ctrl, pos_ctrl, vel_ctrl, acc_ctrl)

            # Set recording start time BEFORE the goal is dispatched so that
            # sensor data timestamped from this point covers the full execution.
            self._t0 = self.get_clock().now().nanoseconds * 1e-9

            goal = FollowJointTrajectory.Goal()
            goal.trajectory = jt
            future = self._action_client.send_goal_async(goal)
            rclpy.spin_until_future_complete(self, future)
            gh = future.result()
            if not gh.accepted:
                self.get_logger().error("Main trajectory goal rejected.")
                return False

            self.get_logger().info("Trajectory accepted — executing.")
            rf = gh.get_result_async()
            rclpy.spin_until_future_complete(self, rf)
            ec = rf.result().result.error_code
            if ec != 0:
                self.get_logger().error(
                    f"Trajectory aborted: code={ec}  "
                    f"{rf.result().result.error_string}"
                )
                return False
            return True

        def save_rollout(self, output_file: str | None = None) -> str:
            """Resample recorded signals onto a 100 Hz grid and write parquet.

            Pass output_file for a flat-layout path; omit to use the legacy
            RolloutStore layout under self._output_dir.
            """
            # Deferred import: pandas (via rollout) may not be in the ROS system Python.
            from .rollout import RolloutResult, RolloutStore  # noqa: PLC0415

            # The robot executed the trajectory over real_duration seconds (not
            # sim_time), so the recording grid must span the actual execution time.
            # ref_pos is evaluated at t_nominal = t_real * factor so that the
            # position profile matches what was commanded to the controller.
            factor = max(0.01, min(self._traj_config.speed_override, 100.0)) / 100.0
            real_duration = self._traj_config.real_duration
            dt = 0.01  # 100 Hz recording grid
            grid = np.arange(0.0, real_duration, dt)

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
                # tau_motor is the commanded joint effort read from /joint_states
                # (msg.effort), NOT the F/T sensor. Do not conflate the two.
                tau_motor = np.column_stack([
                    _interp(self._joint_efforts[j], t0) for j in JOINT_NAMES
                ])

            if not np.any(tau_motor):
                self.get_logger().warn(
                    "Recorded tau_motor (joint effort) is all zero. The "
                    "hardware's JointStateBroadcaster likely does not expose an "
                    "effort state interface for these joints — check "
                    "ros2_control controller config."
                )

            # Reference positions: evaluate the nominal trajectory at the equivalent
            # nominal time (t_real * factor), matching what the controller was sent.
            traj = _trajectory_from_config(self._traj_config)
            ref_pos_list, ref_vel_list = [], []
            for t_real in grid:
                q_r, dq_r = traj(float(t_real * factor))
                # Velocity in real time = nominal velocity * factor
                ref_pos_list.append(q_r)
                ref_vel_list.append(dq_r * factor)

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
