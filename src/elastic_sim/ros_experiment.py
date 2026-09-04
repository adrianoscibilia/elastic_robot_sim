"""ROS 2 execution and complete normalized observation capture.

ROS is imported inside functions so simulation-only and calibration tooling can
run on machines without a ROS installation.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .experiment import start_rosbag
from .materialized import MaterializedTrajectory


def _action_groups(config: Mapping[str, Any], joint_names: Sequence[str]) -> list[tuple[str, tuple[int, ...]]]:
    ros = config.get("ros", {}) or {}
    configured = ros.get("action_servers")
    if configured is None:
        return [(str(ros.get("action_server", "/joint_trajectory_controller/follow_joint_trajectory")), tuple(range(len(joint_names))))]
    by_name = {name: index for index, name in enumerate(joint_names)}
    groups: list[tuple[str, tuple[int, ...]]] = []
    claimed: set[str] = set()
    for item in configured:
        name = str(item.get("name", ""))
        names = tuple(str(value) for value in item.get("joints", ()))
        if not name or not names or any(value not in by_name for value in names):
            raise ValueError("each ros.action_servers entry requires a name and known joints")
        if claimed.intersection(names):
            raise ValueError("ros.action_servers joint groups must not overlap")
        claimed.update(names)
        groups.append((name, tuple(by_name[value] for value in names)))
    if claimed != set(joint_names):
        raise ValueError("ros.action_servers must cover every active joint exactly once")
    return groups


def _ros_imports():
    try:
        import rclpy
        from control_msgs.action import FollowJointTrajectory
        from control_msgs.msg import JointTrajectoryControllerState
        from geometry_msgs.msg import WrenchStamped
        from rclpy.action import ActionClient
        from sensor_msgs.msg import JointState
        from std_srvs.srv import Trigger
        from builtin_interfaces.msg import Duration
    except ImportError as exc:  # pragma: no cover - ROS is external to uv
        raise ImportError("real execution requires ROS 2 Jazzy and control_msgs/sensor_msgs/std_srvs") from exc
    return locals()


def ros_topics(config: Mapping[str, Any]) -> list[str]:
    ros = config.get("ros", {}) or {}
    topics = ros.get("topics", {}) or {}
    result = [
        str(topics.get("joint_states", "/joint_states")),
        str(topics.get("flange_wrench", "/ft_sensor_command_broadcaster/wrench")),
    ]
    controller_states = topics.get("controller_states", (topics.get("controller_state", "/joint_trajectory_controller/state"),))
    result.extend(str(value) for value in controller_states)
    action_names = [str(item["name"]) for item in ros.get("action_servers", ())] or [str(ros.get("action_server", "/joint_trajectory_controller/follow_joint_trajectory"))]
    for action in action_names:
        action = action.rstrip("/")
        result.extend((f"{action}/_action/send_goal", f"{action}/_action/get_result", f"{action}/_action/feedback", f"{action}/_action/cancel_goal", f"{action}/_action/status"))
    for item in ros.get("extra_topics", ()) or ():
        if isinstance(item, Mapping) and item.get("name"):
            result.append(str(item["name"]))
    return list(dict.fromkeys(result))


def _topic_type(node: Any, name: str) -> str | None:
    names_and_types = node.get_topic_names_and_types()
    for topic, types in names_and_types:
        if topic == name:
            return types[0] if types else None
    return None


def preflight_ros(config: Mapping[str, Any], joint_names: Sequence[str]) -> dict[str, Any]:
    """Validate topics, action availability, and joint observability before motors."""
    ros = _ros_imports()
    rclpy, JointState, JointTrajectoryControllerState, WrenchStamped = (
        ros["rclpy"], ros["JointState"], ros["JointTrajectoryControllerState"], ros["WrenchStamped"]
    )
    if not rclpy.ok():
        rclpy.init(args=None)
    node = rclpy.create_node("elastic_sim_ros_preflight")
    ros_config = config.get("ros", {}) or {}
    topics = ros_config.get("topics", {}) or {}
    actions = _action_groups(config, joint_names)
    controller_states = topics.get("controller_states", (topics.get("controller_state", "/joint_trajectory_controller/state"),))
    required = {
        str(topics.get("joint_states", "/joint_states")): "sensor_msgs/msg/JointState",
        str(topics.get("flange_wrench", "/ft_sensor_command_broadcaster/wrench")): "geometry_msgs/msg/WrenchStamped",
    }
    required.update({str(name): "control_msgs/msg/JointTrajectoryControllerState" for name in controller_states})
    try:
        for action_name, _ in actions:
            action_client = ros["ActionClient"](node, ros["FollowJointTrajectory"], action_name)
            if not action_client.wait_for_server(timeout_sec=float(ros_config.get("preflight_timeout", 5.0))):
                raise RuntimeError(f"ROS preflight failed; action server unavailable: {action_name}")
        deadline = time.monotonic() + float(config.get("ros", {}).get("preflight_timeout", 5.0))
        found: dict[str, str] = {}
        while time.monotonic() < deadline:
            for name in required:
                value = _topic_type(node, name)
                if value:
                    found[name] = value
            if len(found) == len(required):
                break
            rclpy.spin_once(node, timeout_sec=0.1)
        missing = [name for name in required if name not in found]
        wrong = {name: found[name] for name in found if found[name] != required[name]}
        if missing or wrong:
            raise RuntimeError(f"ROS preflight failed; missing={missing}, wrong_types={wrong}")
        samples: dict[str, Any] = {}
        subscriptions = []
        subscriptions.append(node.create_subscription(JointState, next(name for name in required if required[name] == "sensor_msgs/msg/JointState"), lambda msg: samples.setdefault("joint_states", msg), 10))
        subscriptions.append(node.create_subscription(WrenchStamped, next(name for name in required if required[name] == "geometry_msgs/msg/WrenchStamped"), lambda msg: samples.setdefault("wrench", msg), 10))
        for name in controller_states:
            subscriptions.append(node.create_subscription(JointTrajectoryControllerState, str(name), lambda msg: samples.setdefault("controller", msg), 10))
        deadline = time.monotonic() + float(config.get("ros", {}).get("sample_timeout", 5.0))
        while time.monotonic() < deadline and len(samples) < 3:
            rclpy.spin_once(node, timeout_sec=0.1)
        joint = samples.get("joint_states")
        if joint is None:
            raise RuntimeError("no JointState sample received during ROS preflight")
        by_name = {name: i for i, name in enumerate(joint.name)}
        if any(name not in by_name for name in joint_names):
            raise RuntimeError("/joint_states does not contain every configured active joint")
        if len(joint.effort) == 0:
            raise RuntimeError("/joint_states has an empty effort field; motion aborted")
        for name in joint_names:
            index = by_name[name]
            if index >= len(joint.position) or index >= len(joint.velocity) or index >= len(joint.effort):
                raise RuntimeError(f"/joint_states is missing position/velocity/effort for {name!r}")
            if not all(np.isfinite(value) for value in (joint.position[index], joint.velocity[index], joint.effort[index])):
                raise RuntimeError(f"/joint_states has non-finite data for {name!r}")
        wrench = samples.get("wrench")
        if wrench is None or not str(getattr(wrench.header, "frame_id", "")):
            raise RuntimeError("flange wrench preflight requires a non-empty frame_id")
        values = (wrench.wrench.force.x, wrench.wrench.force.y, wrench.wrench.force.z, wrench.wrench.torque.x, wrench.wrench.torque.y, wrench.wrench.torque.z)
        if not all(np.isfinite(value) for value in values):
            raise RuntimeError("flange wrench preflight received non-finite force/torque")
        return {"topics": found, "action_servers": [name for name, _ in actions], "joint_names": list(joint_names), "topic_types": required, "sample_validated": True}
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


class _Capture:
    def __init__(self, node: Any, config: Mapping[str, Any], joint_names: Sequence[str]):
        ros = _ros_imports()
        self.node, self.config, self.joint_names = node, config, tuple(joint_names)
        self.receipt0 = node.get_clock().now().nanoseconds
        self.joint: list[dict[str, Any]] = []
        self.wrench: list[dict[str, Any]] = []
        self.controller: list[dict[str, Any]] = []
        self.trajectory_id = 0
        self.error: str | None = None
        sensor = config.get("ros", {}).get("sensor", {}) or {}
        self.link_side_frame = sensor.get("link_side_frame")
        self.axis_mapping = sensor.get("flange_to_link_side_axes")
        self.tf_buffer = None
        if self.link_side_frame:
            try:
                import tf2_ros
                self.tf_buffer = tf2_ros.Buffer()
                tf2_ros.TransformListener(self.tf_buffer, node)
            except ImportError as exc:
                raise ImportError("a configured link_side_frame requires the ROS tf2_ros package") from exc
        topics = config.get("ros", {}).get("topics", {}) or {}
        node.create_subscription(ros["JointState"], str(topics.get("joint_states", "/joint_states")), self.on_joint, 50)
        node.create_subscription(ros["WrenchStamped"], str(topics.get("flange_wrench", "/ft_sensor_command_broadcaster/wrench")), self.on_wrench, 50)
        controller_states = topics.get("controller_states", (topics.get("controller_state", "/joint_trajectory_controller/state"),))
        for name in controller_states:
            node.create_subscription(ros["JointTrajectoryControllerState"], str(name), self.on_controller, 50)

    @staticmethod
    def _stamp(msg: Any) -> int:
        stamp = getattr(getattr(msg, "header", None), "stamp", None)
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec) if stamp is not None else 0

    def _base(self, msg: Any) -> dict[str, Any]:
        receipt = self.node.get_clock().now().nanoseconds
        source = self._stamp(msg) or receipt
        return {"source_ns": source, "receipt_ns": receipt, "t": (source - self.receipt0) * 1.0e-9}

    def on_joint(self, msg: Any) -> None:
        if self.error is not None:
            return
        row = self._base(msg)
        by_name = {name: i for i, name in enumerate(msg.name)}
        if any(name not in by_name for name in self.joint_names):
            self.error = "/joint_states is missing one or more active joints"
            return
        for field, label in (("position", "q"), ("velocity", "dq"), ("effort", "tau_joint_state")):
            values = getattr(msg, field, ())
            if any(index >= len(values) or not np.isfinite(values[index]) for index in (by_name[name] for name in self.joint_names)):
                if field == "effort":
                    # A missing effort is a hard safety/data error, never a zero.
                    self.error = "/joint_states effort is missing or non-finite for an active joint"
                else:
                    self.error = f"/joint_states {field} is missing or non-finite for an active joint"
                return
            for name in self.joint_names:
                row[f"{label}__{name}"] = float(values[by_name[name]])
        row["valid_joint_state"] = True
        row["trajectory_id"] = self.trajectory_id
        self.joint.append(row)

    def on_wrench(self, msg: Any) -> None:
        row = self._base(msg)
        wrench = msg.wrench
        row.update({"frame_id": str(getattr(msg.header, "frame_id", "")),
                    "force_flange__x": float(wrench.force.x), "force_flange__y": float(wrench.force.y), "force_flange__z": float(wrench.force.z),
                    "torque_flange__x": float(wrench.torque.x), "torque_flange__y": float(wrench.torque.y), "torque_flange__z": float(wrench.torque.z),
                    "valid_wrench": all(np.isfinite(value) for value in (wrench.force.x, wrench.force.y, wrench.force.z, wrench.torque.x, wrench.torque.y, wrench.torque.z))})
        if self.axis_mapping is not None:
            mapping = tuple(int(value) for value in self.axis_mapping)
            if len(mapping) != 3 or sorted(mapping) != [0, 1, 2]:
                raise RuntimeError("ros.sensor.flange_to_link_side_axes must be a permutation of [0, 1, 2]")
            force = np.asarray([wrench.force.x, wrench.force.y, wrench.force.z], dtype=float)[list(mapping)]
            torque = np.asarray([wrench.torque.x, wrench.torque.y, wrench.torque.z], dtype=float)[list(mapping)]
            for index, axis in enumerate("xyz"):
                row[f"force_link_side__{axis}"] = force[index]
                row[f"torque_link_side__{axis}"] = torque[index]
            row["link_side_frame"] = str(self.link_side_frame or msg.header.frame_id)
            row["valid_link_side_wrench"] = bool(np.isfinite(force).all() and np.isfinite(torque).all())
        if self.link_side_frame and self.tf_buffer is not None:
            try:
                transform = self.tf_buffer.lookup_transform(self.link_side_frame, msg.header.frame_id, msg.header.stamp)
                q = transform.transform.rotation
                rotation = np.asarray([
                    [1 - 2 * (q.y * q.y + q.z * q.z), 2 * (q.x * q.y - q.z * q.w), 2 * (q.x * q.z + q.y * q.w)],
                    [2 * (q.x * q.y + q.z * q.w), 1 - 2 * (q.x * q.x + q.z * q.z), 2 * (q.y * q.z - q.x * q.w)],
                    [2 * (q.x * q.z - q.y * q.w), 2 * (q.y * q.z + q.x * q.w), 1 - 2 * (q.x * q.x + q.y * q.y)],
                ])
                force = rotation @ np.asarray([wrench.force.x, wrench.force.y, wrench.force.z], dtype=float)
                torque = rotation @ np.asarray([wrench.torque.x, wrench.torque.y, wrench.torque.z], dtype=float)
                for index, axis in enumerate("xyz"):
                    row[f"force_link_side__{axis}"] = force[index]
                    row[f"torque_link_side__{axis}"] = torque[index]
                row["link_side_frame"] = str(self.link_side_frame)
                row["valid_link_side_wrench"] = bool(np.isfinite(force).all() and np.isfinite(torque).all())
            except Exception:
                row["valid_link_side_wrench"] = False
        row["trajectory_id"] = self.trajectory_id
        self.wrench.append(row)

    def on_controller(self, msg: Any) -> None:
        row = self._base(msg)
        message_names = tuple(getattr(msg, "joint_names", ())) or self.joint_names
        for prefix, field in (("desired", "desired"), ("actual", "actual"), ("error", "error")):
            point = getattr(msg, field, None)
            if point is None:
                continue
            for component in ("positions", "velocities", "accelerations", "efforts"):
                values = getattr(point, component, ())
                for index, name in enumerate(message_names):
                    if index < len(values) and np.isfinite(values[index]):
                        row[f"controller_{prefix}_{component}__{name}"] = float(values[index])
        row["valid_controller_state"] = True
        row["trajectory_id"] = self.trajectory_id
        self.controller.append(row)

    def frame(self, trajectory: MaterializedTrajectory, trajectory_id: int) -> pd.DataFrame:
        if not self.joint:
            raise RuntimeError("no valid JointState samples were received")
        base = pd.DataFrame(self.joint).sort_values("source_ns")
        for name, events in (("wrench", self.wrench), ("controller", self.controller)):
            if not events:
                continue
            other = pd.DataFrame(events).sort_values("source_ns")
            base = pd.merge_asof(base, other, on="source_ns", direction="nearest", suffixes=("", f"_{name}"))
        if "trajectory_id" not in base:
            base["trajectory_id"] = int(trajectory_id)
        base["relative_t"] = base["t"] - float(base["t"].iloc[0])
        grid = base["relative_t"].to_numpy(float)
        plan = trajectory.sample()
        for prefix, values in (("q_ref", plan["q"]), ("dq_ref", plan["dq"]), ("ddq_ref", plan["ddq"])):
            if values is None:
                continue
            for index, name in enumerate(self.joint_names):
                base[prefix + "__" + name] = np.interp(grid, trajectory.time, values[:, index])
        return base.reset_index(drop=True)


def _send_trajectory(node: Any, config: Mapping[str, Any], trajectory: MaterializedTrajectory, capture: _Capture, timeout: float) -> dict[str, Any]:
    """Dispatch all configured controller groups before waiting for completion."""
    ros = _ros_imports()
    pending = []
    for action_name, indices in _action_groups(config, trajectory.joint_names):
        client = ros["ActionClient"](node, ros["FollowJointTrajectory"], action_name)
        if not client.wait_for_server(timeout_sec=timeout):
            raise RuntimeError(f"FollowJointTrajectory action server unavailable: {action_name}")
        goal = ros["FollowJointTrajectory"].Goal()
        goal.trajectory.joint_names = [trajectory.joint_names[index] for index in indices]
        for row, t in enumerate(trajectory.time):
            from trajectory_msgs.msg import JointTrajectoryPoint
            point = JointTrajectoryPoint()
            point.positions = trajectory.position[row, list(indices)].tolist()
            point.velocities = trajectory.velocity[row, list(indices)].tolist()
            if trajectory.acceleration is not None:
                point.accelerations = trajectory.acceleration[row, list(indices)].tolist()
            sec = int(t)
            point.time_from_start.sec = sec
            point.time_from_start.nanosec = int(round((float(t) - sec) * 1.0e9))
            goal.trajectory.points.append(point)
        pending.append((action_name, client.send_goal_async(goal)))
    while not all(future.done() for _, future in pending):
        ros["rclpy"].spin_once(node, timeout_sec=0.05)
        if capture.error is not None:
            raise RuntimeError(capture.error)
    handles = [(name, future.result()) for name, future in pending]
    rejected = [name for name, handle in handles if handle is None or not handle.accepted]
    if rejected:
        raise RuntimeError(f"FollowJointTrajectory goal was rejected by {rejected}")
    results = [(name, handle.get_result_async()) for name, handle in handles]
    while not all(future.done() for _, future in results):
        ros["rclpy"].spin_once(node, timeout_sec=0.05)
        if capture.error is not None:
            raise RuntimeError(capture.error)
    reports = []
    for name, future in results:
        result = future.result().result
        if int(result.error_code) != 0:
            raise RuntimeError(f"FollowJointTrajectory failed on {name} with error_code={result.error_code}: {result.error_string}")
        reports.append({"action_server": name, "accepted": True, "error_code": int(result.error_code), "error_string": str(result.error_string)})
    return {"accepted": True, "error_code": 0, "error_string": "", "controllers": reports}


def execute_real_trajectories(config: Mapping[str, Any], trajectories: Sequence[MaterializedTrajectory], output_dir: str | Path, *, no_motor_control: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Preflight, bag, execute saved points, and return normalized observations."""
    if not trajectories:
        raise ValueError("at least one trajectory is required")
    preflight = preflight_ros(config, trajectories[0].joint_names)
    ros = _ros_imports()
    rclpy = ros["rclpy"]
    if not rclpy.ok():
        rclpy.init(args=None)
    node = rclpy.create_node("elastic_sim_experiment")
    capture = _Capture(node, config, trajectories[0].joint_names)
    topics = ros_topics(config)
    lifecycle = config.get("ros", {}).get("motor_services")
    manage_motors = bool(lifecycle) and not no_motor_control
    motor_enabled = False
    bag = None
    try:
        bag = start_rosbag(output_dir, topics)
        time.sleep(float(config.get("ros", {}).get("bag_startup_delay", 0.5)))
        if bag.poll() is not None:
            raise RuntimeError("rosbag2 exited during startup; inspect raw/rosbag2.stderr.log")
        if manage_motors:
            client = node.create_client(ros["Trigger"], str(lifecycle["enable"]))
            if not client.wait_for_service(timeout_sec=5.0):
                raise RuntimeError("motor enable service unavailable")
            future = client.call_async(ros["Trigger"].Request())
            while not future.done():
                rclpy.spin_once(node, timeout_sec=0.05)
            if not future.result().success:
                raise RuntimeError("motor enable service returned failure")
            motor_enabled = True
        results = []
        for index, trajectory in enumerate(trajectories):
            capture.trajectory_id = index
            results.append(_send_trajectory(node, config, trajectory, capture, float(config.get("ros", {}).get("action_timeout", trajectory.duration + 10.0))))
            # Drain callbacks after the action result so the final samples are retained.
            end = time.monotonic() + 0.25
            while time.monotonic() < end:
                rclpy.spin_once(node, timeout_sec=0.05)
            if capture.error is not None:
                raise RuntimeError(capture.error)
        if motor_enabled:
            client = node.create_client(ros["Trigger"], str(lifecycle["disable"]))
            if client.wait_for_service(timeout_sec=5.0):
                future = client.call_async(ros["Trigger"].Request())
                while not future.done():
                    rclpy.spin_once(node, timeout_sec=0.05)
        frame = capture.frame(trajectories[-1], len(trajectories) - 1)
        # Capture currently holds all samples.  Rebuild plan columns per id for
        # multi-trajectory runs; the base rows remain fully timestamped.
        frames = []
        for index, trajectory in enumerate(trajectories):
            subset = frame[frame["trajectory_id"] == index].copy()
            if len(subset):
                grid = subset["relative_t"].to_numpy(float)
                for prefix, values in (("q_ref", trajectory.position), ("dq_ref", trajectory.velocity), ("ddq_ref", trajectory.acceleration)):
                    if values is not None:
                        for col, joint in enumerate(trajectory.joint_names):
                            subset[f"{prefix}__{joint}"] = np.interp(grid, trajectory.time, values[:, col])
                action_result = results[index]
                subset["action_accepted"] = bool(action_result["accepted"])
                subset["action_error_code"] = int(action_result["error_code"])
                subset["action_error_string"] = str(action_result["error_string"])
                frames.append(subset)
        return pd.concat(frames, ignore_index=True), {"preflight": preflight, "action_results": results, "topics": topics}
    finally:
        if motor_enabled:
            try:
                client = node.create_client(ros["Trigger"], str(lifecycle["disable"]))
                if client.wait_for_service(timeout_sec=1.0):
                    future = client.call_async(ros["Trigger"].Request())
                    while not future.done():
                        rclpy.spin_once(node, timeout_sec=0.05)
            except Exception:
                pass
        if bag is not None:
            bag.terminate()
            try:
                bag.wait(timeout=10)
            except Exception:
                bag.kill()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
