"""Optional ROS 2 MoveIt planning-scene validation.

Nothing in this module is imported by the portable simulation path.  A user
must explicitly request validation and provide a ROS environment containing
``rclpy`` and ``moveit_msgs``.
"""

from __future__ import annotations

from typing import Any, Mapping


def validate_with_moveit(config: Mapping[str, Any], trajectory: Any) -> dict[str, Any]:
    settings = (config.get("ros", {}) or {}).get("moveit", {}) or {}
    group = settings.get("group")
    if not isinstance(group, str) or not group:
        raise ValueError("ros.moveit.group is required with --moveit-validate")
    namespace = str(settings.get("namespace", "")).rstrip("/")
    service_name = f"{namespace}/check_state_validity" or "/check_state_validity"
    timeout = float(settings.get("timeout", 5.0))
    try:
        import rclpy
        from moveit_msgs.srv import GetStateValidity
        from sensor_msgs.msg import JointState
    except ImportError as exc:
        raise RuntimeError(
            "--moveit-validate requires an externally installed and sourced ROS 2/MoveIt environment"
        ) from exc
    owns_context = not rclpy.ok()
    if owns_context:
        rclpy.init(args=None)
    node = rclpy.create_node("elastic_sim_moveit_validator")
    client = node.create_client(GetStateValidity, service_name)
    try:
        if not client.wait_for_service(timeout_sec=timeout):
            raise RuntimeError(f"MoveIt state-validity service unavailable: {service_name}")
        checked = 0
        for index, q in enumerate(trajectory.position):
            request = GetStateValidity.Request()
            request.group_name = group
            request.robot_state.joint_state = JointState(
                name=list(trajectory.joint_names), position=[float(value) for value in q]
            )
            future = client.call_async(request)
            rclpy.spin_until_future_complete(node, future, timeout_sec=timeout)
            response = future.result()
            if response is None:
                raise RuntimeError(f"MoveIt validation timed out for trajectory sample {index}")
            checked += 1
            if not response.valid:
                contacts = [f"{item.contact_body_1}/{item.contact_body_2}" for item in response.contacts]
                raise ValueError(f"MoveIt rejected trajectory sample {index}; contacts={contacts}")
        return {"valid": True, "group": group, "service": service_name, "checked_states": checked}
    finally:
        node.destroy_node()
        if owns_context:
            rclpy.shutdown()
