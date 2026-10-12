"""``erd_scene_view``: show a lab file's scene next to the robot in RViz.

Robot-agnostic (any ``erd.lab/1`` file whose robot has a driver description
below), read-only, never commands motion. It reads the lab file WITHOUT the
``hardware: real`` review gate, so a cell can be inspected before the owner
signs it off.

What is drawn, in the bare asset's root frame (where the planner checks
collisions):

* every scene object as the planner sees it, i.e. inflated by
  ``scene.clearance`` (transparent red);
* for ``box``/``keep_out`` objects also the declared, un-inflated box
  (solid orange), so it can be compared with the real cell;
* the object id as a text label.

Where the robot pose comes from (exactly one):

* ``--live``: from the running driver (its ``/joint_states`` and TF). Use the
  same shell/domain as the driver. Lets you watch the real robot against
  the virtual objects while it moves.
* ``--pose a1 a2 ...``: one joint configuration, in DEGREES, in the lab
  file's joint order (the order a pendant shows). Type the pendant's axis
  values to check a pose without connecting to the robot.
* ``--poses``: cycles ``poses.home`` and every ``poses.standstill`` entry.
* ``--plan RUN_DIR``: replays the frozen plan of a run folder (standstill,
  identify and run segments, in that order), sped up by ``--speed``.

The last three publish their own ``/joint_states`` and start their own
``robot_state_publisher``; they refuse to start when something else already
publishes ``/joint_states`` in this ROS domain (e.g. a live driver).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

#: Topics this tool publishes.
MARKER_TOPIC = "/erd/scene_markers"


# ---------------------------------------------------------------------------
# Pure part (no ROS): lab file -> marker specs, plan -> joint samples
# ---------------------------------------------------------------------------


def load_unreviewed(path: str | Path):
    """Robot, scene, poses and joint map of a lab file, without the real-robot
    review gate (display only)."""
    from . import config as cfg

    source = str(Path(path).expanduser().resolve())
    raw = yaml.safe_load(Path(source).read_text(encoding="utf-8")) or {}
    robot = str(cfg._require(raw, "robot", source))
    description = cfg._load_description(raw, source)
    scene = cfg._load_scene(raw, source, robot)
    poses = cfg._load_poses(raw, source)
    return robot, raw, description, scene, poses


def root_frame(robot: str) -> str:
    """The frame scene objects are expressed in after loading: the bare
    asset's own root, i.e. the identity entry of ``frames.KNOWN_FRAMES``."""
    from . import frames

    for name, (xyz, rpy) in frames.KNOWN_FRAMES[robot].items():
        if not any(xyz) and not any(rpy):
            return name
    raise ValueError(f"no identity frame known for {robot!r}")


def rpy_to_quaternion(rpy) -> tuple[float, float, float, float]:
    """(x, y, z, w) of the extrinsic XYZ rotation R = Rz(yaw) Ry(pitch) Rx(roll)."""
    roll, pitch, yaw = (float(v) for v in rpy)
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy)


def scene_marker_specs(scene) -> list[dict[str, Any]]:
    """One dict per marker: ns, id, shape, xyz, quat, scale, rgba, text."""
    from .config import SceneBox, SceneCylinder, SceneMesh, SceneSphere

    clearance = float(scene.clearance)
    specs: list[dict[str, Any]] = []
    red = (0.9, 0.1, 0.1, 0.25)
    orange = (1.0, 0.55, 0.0, 0.8)
    for obj in scene.objects:
        rpy = getattr(obj, "rpy", (0.0, 0.0, 0.0))
        quat = rpy_to_quaternion(rpy)
        base = {"xyz": tuple(float(v) for v in obj.xyz), "quat": quat}
        if isinstance(obj, SceneBox):
            specs.append({**base, "ns": "planner_with_clearance", "shape": "cube",
                          "scale": tuple(obj.size), "rgba": red, "label": obj.id})
            if obj.source_type in ("box", "keep_out"):
                raw = tuple(max(1e-3, s - 2.0 * clearance) for s in obj.size)
                specs.append({**base, "ns": "declared", "shape": "cube", "scale": raw, "rgba": orange,
                              "label": None})
        elif isinstance(obj, SceneCylinder):
            specs.append({**base, "ns": "planner_with_clearance", "shape": "cylinder",
                          "scale": (2 * obj.radius, 2 * obj.radius, obj.height), "rgba": red, "label": obj.id})
        elif isinstance(obj, SceneSphere):
            specs.append({"xyz": base["xyz"], "quat": (0.0, 0.0, 0.0, 1.0), "ns": "planner_with_clearance",
                          "shape": "sphere", "scale": (2 * obj.radius,) * 3, "rgba": red, "label": obj.id})
        elif isinstance(obj, SceneMesh):
            specs.append({**base, "ns": "declared", "shape": "mesh", "file": obj.file,
                          "scale": tuple(obj.scale), "rgba": orange, "label": obj.id})
    return specs


def plan_samples(run_dir: str | Path) -> list[tuple[str, list[str], np.ndarray, np.ndarray]]:
    """``(segment_id, sim_joint_names, time, position)`` for every frozen
    segment of ``RUN_DIR/plan``, in execution order (standstill, identify, run)."""
    plan_dir = Path(run_dir).expanduser() / "plan"
    manifest = json.loads((plan_dir / "plan.manifest.json").read_text(encoding="utf-8"))
    out = []
    for key in ("standstill_segments", "identify_segments", "segments"):
        for entry in manifest.get(key, []):
            data = json.loads((plan_dir / f"{entry['segment_id']}.json").read_text(encoding="utf-8"))
            out.append((entry["segment_id"], list(data["joint_names"]),
                        np.asarray(data["time"], dtype=float), np.asarray(data["position"], dtype=float)))
    if not out:
        raise ValueError(f"{plan_dir}: no segments in plan.manifest.json")
    return out


def driver_description(robot: str, raw: dict) -> str:
    """The driver's own URDF (the joint names `/joint_states` uses)."""
    if robot == "kuka_lbr_iiwa_14_r820":
        from erd_iiwa.description import descriptions

        return descriptions(hardware="mock")[0]
    if robot == "ur10_cb3":
        from erd_ur10.description import export_driver_urdf

        kinematics = (raw.get("connection") or {}).get("kinematics_params_file")
        with tempfile.TemporaryDirectory() as tmp:
            path = export_driver_urdf(str(Path(tmp) / "ur10.urdf"), kinematics_params_file=kinematics)
            return Path(path).read_text(encoding="utf-8")
    raise ValueError(f"erd_scene_view: no driver description exporter for {robot!r} yet "
                     "(add one next to the iiwa/UR10 entries)")


RVIZ_TEMPLATE = """Panels:
  - Class: rviz_common/Displays
    Name: Displays
Visualization Manager:
  Class: ""
  Displays:
    - Class: rviz_default_plugins/Grid
      Name: Grid
      Enabled: true
      Cell Size: 0.1
      Plane Cell Count: 30
    - Class: rviz_default_plugins/RobotModel
      Name: RobotModel
      Enabled: true
      Description Source: Topic
      Description Topic:
        Value: /robot_description
        Depth: 5
        Durability Policy: Transient Local
        History Policy: Keep Last
        Reliability Policy: Reliable
    - Class: rviz_default_plugins/MarkerArray
      Name: Scene
      Enabled: true
      Topic:
        Value: {marker_topic}
        Depth: 5
        Durability Policy: Transient Local
        History Policy: Keep Last
        Reliability Policy: Reliable
  Global Options:
    Fixed Frame: {frame}
    Background Color: 48; 48; 48
  Views:
    Current:
      Class: rviz_default_plugins/Orbit
      Distance: 3.0
      Focal Point:
        X: 0.0
        Y: 0.0
        Z: 0.4
"""


# ---------------------------------------------------------------------------
# ROS part
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    from .env_guard import assert_environment

    assert_environment(require_ros=True, require_pinocchio=False)
    import rclpy
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
    from builtin_interfaces.msg import Duration
    from sensor_msgs.msg import JointState
    from visualization_msgs.msg import Marker, MarkerArray

    parser = argparse.ArgumentParser(prog="erd_scene_view", description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True, help="erd.lab/1 file (reviewed or not)")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true", help="robot pose from the running driver")
    mode.add_argument("--pose", nargs="+", type=float, help="joint values in DEGREES, lab-file joint order")
    mode.add_argument("--poses", action="store_true", help="cycle poses.home and poses.standstill")
    mode.add_argument("--plan", metavar="RUN_DIR", help="replay the frozen plan of a run folder")
    parser.add_argument("--speed", type=float, default=2.0, help="--plan replay speed factor (default 2)")
    parser.add_argument("--hold", type=float, default=4.0, help="--poses seconds per pose (default 4)")
    parser.add_argument("--no-rviz", action="store_true", help="do not start rviz2")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    robot, raw, description, scene, poses = load_unreviewed(args.config)
    frame = root_frame(robot)
    joint_order = tuple(description.joint_map.values())          # sim names, lab-file order
    sim_to_driver = description.sim_to_driver()
    specs = scene_marker_specs(scene)
    print(f"[erd_scene_view] {robot}: {len(scene.objects)} scene object(s), clearance "
          f"{scene.clearance} m, frame {frame}; scene.reviewed={scene.reviewed}")

    rclpy.init()
    node = rclpy.create_node("erd_scene_view")
    latched = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                         reliability=QoSReliabilityPolicy.RELIABLE)
    marker_pub = node.create_publisher(MarkerArray, MARKER_TOPIC, latched)

    array = MarkerArray()
    shapes = {"cube": Marker.CUBE, "cylinder": Marker.CYLINDER, "sphere": Marker.SPHERE,
              "mesh": Marker.MESH_RESOURCE}
    for index, spec in enumerate(specs):
        marker = Marker()
        marker.header.frame_id = frame
        marker.ns, marker.id, marker.type, marker.action = spec["ns"], index, shapes[spec["shape"]], Marker.ADD
        marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = spec["xyz"]
        (marker.pose.orientation.x, marker.pose.orientation.y,
         marker.pose.orientation.z, marker.pose.orientation.w) = spec["quat"]
        marker.scale.x, marker.scale.y, marker.scale.z = (float(v) for v in spec["scale"])
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = spec["rgba"]
        marker.lifetime = Duration()
        if spec["shape"] == "mesh":
            marker.mesh_resource = "file://" + str(Path(spec["file"]).expanduser().resolve())
        array.markers.append(marker)
        if spec["label"]:
            text = Marker()
            text.header.frame_id = frame
            text.ns, text.id, text.type, text.action = "labels", index, Marker.TEXT_VIEW_FACING, Marker.ADD
            text.pose.position.x, text.pose.position.y, text.pose.position.z = spec["xyz"]
            text.pose.position.z = min(spec["xyz"][2], 1.5) + 0.15
            text.pose.orientation.w = 1.0
            text.scale.z = 0.08
            text.color.r = text.color.g = text.color.b = text.color.a = 1.0
            text.text = spec["label"]
            array.markers.append(text)
    marker_pub.publish(array)

    children: list[subprocess.Popen] = []
    tmpdir = tempfile.TemporaryDirectory(prefix="erd_scene_view_")
    try:
        joint_pub = None
        if not args.live:
            # Refuse to fight a live driver over /joint_states and TF.
            deadline = time.time() + 1.5
            while time.time() < deadline:
                rclpy.spin_once(node, timeout_sec=0.1)
            if node.count_publishers("/joint_states") > 0:
                print("[erd_scene_view] refused: something already publishes /joint_states in this ROS "
                      f"domain ({os.environ.get('ROS_DOMAIN_ID', '0')}). Use --live, or run in a shell "
                      "without the driver (e.g. `source workspace_setup.sh` without --real).", file=sys.stderr)
                return 2
            urdf = driver_description(robot, raw)
            params = Path(tmpdir.name) / "rsp.yaml"
            params.write_text(yaml.safe_dump({"robot_state_publisher": {"ros__parameters": {
                "robot_description": urdf}}}), encoding="utf-8")
            children.append(subprocess.Popen(["ros2", "run", "robot_state_publisher", "robot_state_publisher",
                                              "--ros-args", "--params-file", str(params)]))
            joint_pub = node.create_publisher(JointState, "/joint_states", 10)

        if not args.no_rviz:
            rviz = Path(tmpdir.name) / "scene.rviz"
            rviz.write_text(RVIZ_TEMPLATE.format(marker_topic=MARKER_TOPIC, frame=frame), encoding="utf-8")
            children.append(subprocess.Popen(["rviz2", "-d", str(rviz)]))

        def publish(sim_names, q_rad):
            msg = JointState()
            msg.header.stamp = node.get_clock().now().to_msg()
            msg.name = [sim_to_driver[name] for name in sim_names]
            msg.position = [float(v) for v in q_rad]
            joint_pub.publish(msg)

        if args.pose is not None:
            if len(args.pose) != len(joint_order):
                print(f"[erd_scene_view] --pose needs {len(joint_order)} values (degrees) in the order "
                      f"{list(joint_order)}", file=sys.stderr)
                return 2
            q = np.deg2rad(args.pose)
            print(f"[erd_scene_view] pose (deg) {args.pose}")
        frames_list: list = []
        if args.poses:
            frames_list = [("home", poses.home)] + [(f"standstill_{i}", p) for i, p in enumerate(poses.standstill)]
        segments = plan_samples(args.plan) if args.plan else []

        print("[erd_scene_view] running; Ctrl-C to stop.")
        start = last_markers = time.time()
        segment_index, segment_start, announced = 0, time.time(), None
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)
            if time.time() - last_markers > 5.0:            # late RViz windows still get the scene
                marker_pub.publish(array)
                last_markers = time.time()
            if args.pose is not None:
                publish(joint_order, q)
            elif args.poses:
                name, pose = frames_list[int((time.time() - start) / args.hold) % len(frames_list)]
                if name != announced:
                    print(f"[erd_scene_view] {name}: {np.round(np.rad2deg(pose), 1).tolist()} deg")
                    announced = name
                publish(joint_order, pose)
            elif segments:
                seg_id, names, t, pos = segments[segment_index]
                elapsed = (time.time() - segment_start) * args.speed
                if seg_id != announced:
                    print(f"[erd_scene_view] {seg_id}: {t[-1] - t[0]:.1f} s")
                    announced = seg_id
                if elapsed > t[-1] - t[0]:
                    segment_index = (segment_index + 1) % len(segments)
                    segment_start = time.time()
                    continue
                row = int(np.searchsorted(t - t[0], elapsed))
                publish(names, pos[min(row, len(t) - 1)])
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        for child in children:
            child.terminate()
        for child in children:
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
        tmpdir.cleanup()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
