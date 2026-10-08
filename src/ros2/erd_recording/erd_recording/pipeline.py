"""ROS 2 orchestration: preflight, standstill, identify, run, convert,
validate, report (RR_01 S5, T1.6/T1.7; RR_04 A-1/A-2/A-3/A-4/A-11/A-12/B-3).
The only module here that imports ``rclpy``; everything else in this package
is plain Python.
"""

from __future__ import annotations

import json
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

from .env_guard import assert_environment

# This module never imports pinocchio itself: the `plan`/`convert` stages run
# as subprocesses (see `plan_cli.py`/`convert_cli.py` and
# `env_guard.clean_ros_subprocess_env`), specifically to avoid the
# LD_LIBRARY_PATH conflict between ROS Jazzy's sourced environment and this
# venv's cmeel-packaged pinocchio (T1.0/T1.6).
assert_environment(require_ros=True, require_pinocchio=False)

import rclpy  # noqa: E402
from rclpy.action import ActionClient  # noqa: E402
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup  # noqa: E402
from rclpy.executors import SingleThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy  # noqa: E402
from rclpy.time import Time  # noqa: E402

from control_msgs.action import FollowJointTrajectory  # noqa: E402
from control_msgs.msg import DynamicJointState, JointTrajectoryControllerState  # noqa: E402
from controller_manager_msgs.srv import ListControllers, ListHardwareInterfaces, SwitchController  # noqa: E402
from rcl_interfaces.msg import ParameterType  # noqa: E402
from rcl_interfaces.srv import GetParameters  # noqa: E402
from trajectory_msgs.msg import JointTrajectoryPoint  # noqa: E402

from elastic_sim.materialized import MaterializedTrajectory  # noqa: E402

from .bagio import decode_dynamic_joint_state  # noqa: E402
from .config import LabConfig, check_environment_domain, load_lab_config  # noqa: E402
from .planning import PlanBundle, load_plan  # noqa: E402
from .profile import RobotProfile, profile_for  # noqa: E402
from .bag_record import (  # noqa: E402
    PRODUCTION_LOG_LEVEL,
    SIDECAR_STATUS_TOPIC,
    SIDECAR_SUBSCRIBE_TIMEOUT_S,
    SUBSCRIBE_TIMEOUT_S,
    wait_for_bag_start,
)
from .safety import (  # noqa: E402
    EXCITATION_KINDS,
    FRI_COMMANDING_ACTIVE,
    STALE_SAMPLE_BUDGET_S,
    MonitorSample,
    check_abort_override,
    check_active_controllers,
    check_at_home_and_still,
    check_software_version,
    check_start_state,
    evaluate_abort_conditions,
    segment_tracking_rad,
)
from .tolerances import GOAL_TOLERANCE_RAD  # noqa: E402
from .env_guard import clean_ros_subprocess_env  # noqa: E402
from erd_msgs.msg import RunEvent  # noqa: E402


#: Topics recorded on every run, regardless of robot (RR_01 S3.6; RR_04 A-11
#: adds the JTC controller_state and the action topics, computed per profile
#: by :func:`bag_topics_for` since the controller/action names differ).
_BASE_BAG_TOPICS = [
    "/dynamic_joint_states", "/joint_states", "/robot_description", "/tf_static",
    "/rosout", "/diagnostics", "/erd/events",
]

#: RR_15: how long `start_sidecar` waits for the sidecar's first RTDE rows.
SIDECAR_STARTUP_TIMEOUT_S = 10.0

_STATUS_ORDER = ("planned", "recorded", "converted", "valid", "failed", "synthetic")

#: RR_04 A-1: within this long of a triggering sample, a monitor tick must
#: have reached (not necessarily completed) its cancel decision.
_MONITOR_POLL_PERIOD_S = 0.01


def bag_topics_for(profile: RobotProfile, *, sidecar: bool = False) -> list[str]:
    """RR_04 A-11: the base topics plus the JTC ``controller_state``, the
    ``follow_joint_trajectory`` action's hidden topics, and -- UR10 only --
    speed scaling / robot mode / safety mode, and the sidecar status only
    when the stage starts the sidecar (RR_18 R-5)."""
    topics = list(_BASE_BAG_TOPICS)
    topics.append(f"/{profile.controller_name}/controller_state")
    action = f"/{profile.controller_name}/{profile.controller_action}"
    topics += [f"{action}/_action/feedback", f"{action}/_action/status"]
    if profile.has_ur_status_topics:
        topics += [
            "/speed_scaling_state_broadcaster/speed_scaling",
            "/io_and_status_controller/robot_mode",
            "/io_and_status_controller/safety_mode",
        ]
        if sidecar:
            topics.append(SIDECAR_STATUS_TOPIC)
    return topics


def write_qos_overrides(path: Path, *, reliable_deep_topics: list[str], depth: int = 2000) -> Path:
    """RR_04 A-11: a ``ros2 bag record --qos-profile-overrides-path`` file
    (a flat ``{topic: profile}`` mapping, per ``ros2bag.api.
    convert_yaml_to_qos_profile``), reliable/keep_last/``depth`` for the
    listed topics (the 1 kHz ``/dynamic_joint_states`` stream)."""
    data = {topic: {"reliability": "reliable", "history": "keep_last", "depth": depth}
            for topic in reliable_deep_topics}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


@dataclass
class RunPaths:
    root: Path

    @property
    def config_snapshot(self) -> Path:
        return self.root / "config.yaml"

    @property
    def plan_dir(self) -> Path:
        return self.root / "plan"

    @property
    def preflight(self) -> Path:
        return self.root / "preflight.json"

    @property
    def events(self) -> Path:
        return self.root / "events.jsonl"

    @property
    def bag_dir(self) -> Path:
        return self.root / "bag"

    @property
    def standstill_bag_dir(self) -> Path:
        return self.root / "bag_standstill"

    @property
    def raw_log_dir(self) -> Path:
        return self.root / "raw"

    @property
    def rtde_parquet(self) -> Path:
        return self.root / "rtde.parquet"

    @property
    def raw_parquet(self) -> Path:
        return self.root / "raw.parquet"

    @property
    def dataset_dir(self) -> Path:
        return self.root / "dataset"

    @property
    def identification_json(self) -> Path:
        return self.root / "identification.json"

    @property
    def validation(self) -> Path:
        return self.root / "validation.json"

    @property
    def report(self) -> Path:
        return self.root / "report.md"

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.yaml"


def make_run_dir(config: LabConfig, *, run_id: str | None = None) -> RunPaths:
    stamp = time.strftime("%Y%m%d")
    run_id = run_id or time.strftime("run_%H%M%S")
    root = Path(config.recording.output_root).expanduser() / config.robot / stamp / run_id
    root.mkdir(parents=True, exist_ok=True)
    return RunPaths(root)


def _write_manifest_status(paths: RunPaths, status: str, **extra: Any) -> None:
    if status not in _STATUS_ORDER:
        raise ValueError(f"unknown run status {status!r}")
    data = {}
    if paths.manifest.is_file():
        data = yaml.safe_load(paths.manifest.read_text(encoding="utf-8")) or {}
    data["status"] = status
    data.update(extra)
    paths.manifest.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _write_manifest_fields(paths: RunPaths, **fields: Any) -> None:
    """Merge ``fields`` into ``manifest.yaml`` without touching ``status``."""
    data = {}
    if paths.manifest.is_file():
        data = yaml.safe_load(paths.manifest.read_text(encoding="utf-8")) or {}
    data.update(fields)
    paths.manifest.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


class PipelineError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# `plan` (no ROS)
# ---------------------------------------------------------------------------


def _export_driver_urdf(config: LabConfig, paths: RunPaths) -> tuple[Path | None, Path | None]:
    """RR_04 B-2: export the driver's own description *before* the clean
    ``plan`` subprocess, since it needs `xacro`/`ament_index` -- both fine in
    this (ROS-sourced) process, neither touching pinocchio. Returns
    ``(nominal_path, calibrated_path_or_None)``."""
    paths.plan_dir.mkdir(parents=True, exist_ok=True)
    nominal_path = paths.plan_dir / "driver.urdf"
    if config.robot == "kuka_lbr_iiwa_14_r820":
        from erd_iiwa.description import asset_snapshot

        exported = asset_snapshot(str(paths.plan_dir))
        Path(exported.urdf_path).replace(nominal_path)
        return nominal_path, None
    if config.robot == "ur10_cb3":
        from erd_ur10.description import export_driver_urdf

        export_driver_urdf(str(nominal_path))
        calibrated_path = None
        kinematics_params_file = config.connection.kinematics_params_file
        if kinematics_params_file:
            calibrated_path = paths.plan_dir / "driver_calibrated.urdf"
            export_driver_urdf(str(calibrated_path), kinematics_params_file=kinematics_params_file)
        return nominal_path, calibrated_path
    return None, None


def stage_plan(config: LabConfig, config_path: str, paths: RunPaths, *, n_candidates: int | None = None,
              ladder_scale: float | None = None) -> PlanBundle:
    """Run ``plan_cli`` as a subprocess with a cleaned environment (T1.0/T1.6:
    pinocchio and a sourced ROS environment conflict at the dynamic-linker
    level; see :func:`erd_recording.env_guard.clean_ros_subprocess_env`).
    ``ladder_scale`` (RR_06 P3-3b) is forwarded to ``plan_cli --ladder``."""
    paths.config_snapshot.write_text(yaml.safe_dump(config.raw, sort_keys=False), encoding="utf-8")
    driver_urdf_path, driver_urdf_calibrated_path = _export_driver_urdf(config, paths)
    command = [sys.executable, "-m", "erd_recording.plan_cli", "--config", config_path,
              "--output-dir", str(paths.plan_dir)]
    if n_candidates is not None:
        command += ["--n-candidates", str(n_candidates)]
    if driver_urdf_path is not None:
        command += ["--driver-urdf", str(driver_urdf_path)]
    if driver_urdf_calibrated_path is not None:
        command += ["--driver-urdf-calibrated", str(driver_urdf_calibrated_path)]
    if ladder_scale is not None:
        command += ["--ladder", str(ladder_scale)]
    result = subprocess.run(command, env=clean_ros_subprocess_env(), capture_output=True, text=True)
    if result.returncode != 0:
        raise PipelineError(f"plan subprocess failed:\nstdout={result.stdout}\nstderr={result.stderr}")
    print(result.stdout, end="")  # the plan summary and the gravity ranges (RR_12 B-3)
    bundle = load_plan(paths.plan_dir)
    _write_manifest_status(paths, "planned")
    return bundle


def _first_point_by_driver_name(
    trajectory: MaterializedTrajectory, driver_joint_order: tuple[str, ...], sim_to_driver: dict[str, str],
) -> dict[str, float]:
    """The trajectory's first sample, keyed by **driver** joint name (RR_04
    A-2's start-state guard compares against measured state, which is only
    ever available under driver names)."""
    sim_order = trajectory.joint_names
    result = {}
    for sim_name, driver_name in sim_to_driver.items():
        if driver_name not in driver_joint_order or sim_name not in sim_order:
            continue
        result[driver_name] = float(trajectory.position[0, sim_order.index(sim_name)])
    return result


# ---------------------------------------------------------------------------
# ROS node: events, FollowJointTrajectory client, bag/sidecar lifecycle,
# preflight, the live safety monitor (RR_04 A-1/A-2/A-11/A-12/B-3)
# ---------------------------------------------------------------------------


class _MonitorFeed(Node):
    """RR_08 S2c: the monitor's two depth-1 subscriptions (joint state,
    controller state), serviced by a dedicated executor thread so a busy
    main thread can never leave them stale.

    This has to be a **second node**, not a second callback group on
    ``RecordingNode`` spun from the same thread: ``rclpy.spin_once(node,
    ...)`` adds ``node`` to rclpy's global executor on every call
    (``Node.executor`` is a single slot -- assigning it evicts the node
    from whatever executor held it before, see ``rclpy.node.Node.executor``'s
    setter), which would silently tear this down again on the very next
    poll-loop iteration if it shared ``RecordingNode``'s own node object.

    RR_06 P3-2a found that a dedicated *timer* inside the same
    single-threaded poll loop starved this exact data (``spin_once``
    services one ready entity per call, and the timer -- always overdue
    after any blocking pause -- kept winning). A genuinely separate
    executor thread does not share that failure mode: it keeps servicing
    its own (tiny, two-subscription) wait set on its own schedule
    regardless of what the main thread's `spin_once` is doing.

    RR_10, live on the emulator: the executor on that thread is a
    ``SingleThreadedExecutor``, not the ``MultiThreadedExecutor`` RR_08 S2c
    named. rclpy 7.1 (Jazzy)'s multi-threaded executor delivered 2-11
    msg/s of the 1 kHz ``/dynamic_joint_states`` under every QoS tried
    (depth 1/100, reliable/best-effort), the single-threaded one 1000 msg/s
    under all of them; with the multi-threaded one the preflight's FRI
    stability loop saw 22 distinct cycles in 10 s. What S2c needs is the
    separate thread, which this keeps.
    """

    def __init__(self, profile: RobotProfile, on_joint_state: Any, on_controller_state: Any):
        super().__init__("erd_recording_monitor_feed")
        group = MutuallyExclusiveCallbackGroup()
        reliable_latest = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST, depth=1,
        )
        self.create_subscription(
            DynamicJointState, profile.joint_state_topic, on_joint_state, reliable_latest,
            callback_group=group,
        )
        self.create_subscription(
            JointTrajectoryControllerState, f"/{profile.controller_name}/controller_state",
            on_controller_state, reliable_latest, callback_group=group,
        )
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self)
        self._thread = threading.Thread(
            target=self._executor.spin, name="erd_monitor_feed_executor", daemon=True,
        )
        self._thread.start()

    def shutdown(self) -> None:
        self._executor.shutdown(timeout_sec=1.0)
        self._thread.join(timeout=2.0)
        self._executor.remove_node(self)
        self.destroy_node()


class RecordingNode(Node):
    def __init__(self, profile: RobotProfile, config: LabConfig):
        super().__init__("erd_recording_pipeline")
        self.profile = profile
        self.config = config
        self.abort_requested = False  # set by the SIGINT/SIGTERM handler in _run_all
        # RR_12 A-4: the monitor's tracking threshold is the lab file's value;
        # `--abort-tracking-rad` (rung 4a, `--ladder <= 0.1`) lowers it inside
        # excitation segments only (RR_14 P-3): approaches and returns keep
        # the file's value. The JTC keeps the file's path tolerance, which
        # preflight reads back.
        self.monitor_tracking_rad = config.limits.abort.tracking_rad
        self.excitation_tracking_rad: float | None = None
        self._goal_tracking_rad = self.monitor_tracking_rad
        self._goal_kind: str | None = None
        # RR_12 A-2b: every evaluation's sample age and every cancel, written to
        # `monitor.json` per stage.
        self._monitor_ages_s: list[float] = []
        self._monitor_evaluations = 0
        self._monitor_cancels: list[dict[str, Any]] = []
        reliable_deep = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST, depth=2000,
        )
        self._event_pub = self.create_publisher(RunEvent, "/erd/events", 10)

        self._latest_joint_state: DynamicJointState | None = None
        self._latest_joint_state_stamp: Any = None
        self._joint_state_log: list[tuple[float, DynamicJointState]] = []
        self._logging = False
        # Logging-only: feeds `measured_rate_hz` (preflight) and the UR sidecar
        # alignment; the bag already records everything else, so nothing else
        # should read `_joint_state_log`. Subscribed only between
        # start_state_logging()/stop_state_logging() (RR_10, live on the
        # emulator): kept subscribed, this depth-2000 1 kHz queue refilled
        # during every blocking pause (plan load, a standstill hold), and
        # draining it afterwards kept the main thread busy enough to starve
        # `_MonitorFeed` of the GIL -- a 50.1 ms stale-sample abort right
        # after `standstill_hold_0`, with the telemetry itself gap-free.
        self._logging_qos = reliable_deep
        self._logging_joint_state_sub: Any = None

        self._latest_controller_state: JointTrajectoryControllerState | None = None
        # RR_08 S2c: the monitor's own joint-state/controller-state samples
        # -- always kept fresh, independent of `_logging` (also the source
        # for `measured_positions`/`measured_velocities`/the FRI fields,
        # used outside motion too, e.g. preflight's home/still check) --
        # come from `_MonitorFeed`'s dedicated executor thread, not a
        # subscription on this node (see its docstring for why).
        self._monitor_feed = _MonitorFeed(profile, self._on_monitor_joint_state, self._on_controller_state)

        self._speed_scaling: float | None = None
        self._robot_mode_running: bool | None = None
        self._safety_mode_normal: bool | None = None
        if profile.has_ur_status_topics:
            from std_msgs.msg import Float64

            self._speed_scaling_sub = self.create_subscription(
                Float64, "/speed_scaling_state_broadcaster/speed_scaling", self._on_speed_scaling, 10,
            )
            from ur_dashboard_msgs.msg import RobotMode, SafetyMode

            # RR_06 P3-1 item 11, live finding: `io_and_status_controller`
            # publishes `robot_mode`/`safety_mode` TRANSIENT_LOCAL (latched)
            # and only *on change* -- confirmed live: `ros2 topic hz` saw
            # zero messages over 6 s while the robot sat in one mode. The
            # default (bare-int) subscription QoS is VOLATILE, which DDS
            # durability rules never deliver a TRANSIENT_LOCAL publisher's
            # already-sent sample to: a subscription created after the one
            # time the mode last changed (the common case -- this node is
            # created fresh every run, long after the robot was last left
            # RUNNING) saw nothing, for the life of the run. Matching
            # durability is what makes a late subscriber see the latched
            # value immediately.
            latched = QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST, depth=1,
                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            )
            self._robot_mode_sub = self.create_subscription(
                RobotMode, "/io_and_status_controller/robot_mode", self._on_robot_mode, latched,
            )
            self._safety_mode_sub = self.create_subscription(
                SafetyMode, "/io_and_status_controller/safety_mode", self._on_safety_mode, latched,
            )

        action_name = f"/{profile.controller_name}/{profile.controller_action}"
        self._action_client = ActionClient(self, FollowJointTrajectory, action_name)
        self._list_controllers = self.create_client(ListControllers, "/controller_manager/list_controllers")
        self._list_hw_interfaces = self.create_client(
            ListHardwareInterfaces, "/controller_manager/list_hardware_interfaces",
        )
        self._switch_controller = self.create_client(SwitchController, "/controller_manager/switch_controller")
        # RR_06 P3-2c: reads the JTC's own `constraints.*` parameters back, so
        # preflight can refuse a run whose controller didn't actually load the
        # tolerance override (rather than trusting that it did).
        self._get_controller_parameters = self.create_client(
            GetParameters, f"/{profile.controller_name}/get_parameters",
        )

        self._bag_process: subprocess.Popen | None = None
        self._bag_stdout_file = None
        self._bag_stderr_file = None
        self._bag_stderr_path: Path | None = None
        self._sidecar_process: subprocess.Popen | None = None

    def destroy_node(self) -> None:
        self._monitor_feed.shutdown()
        super().destroy_node()

    # -- subscriptions ----------------------------------------------------

    def _on_logging_joint_state(self, msg: DynamicJointState) -> None:
        if self._logging:
            self._joint_state_log.append((time.time(), msg))

    def _on_monitor_joint_state(self, msg: DynamicJointState) -> None:
        self._latest_joint_state = msg
        self._latest_joint_state_stamp = msg.header.stamp

    def _on_controller_state(self, msg: JointTrajectoryControllerState) -> None:
        self._latest_controller_state = msg

    def _on_speed_scaling(self, msg: Any) -> None:
        # RR_06 P3-1 item 2, live finding: `ur_controllers/
        # SpeedScalingStateBroadcaster` publishes a 0-100 percentage
        # (confirmed live: a freshly-activated mock hardware interface,
        # whose state initial_value is 1.0, is reported on this topic as
        # 100.0), not the 0-1 fraction both `stage_preflight`'s
        # `speed_scaling == 1` check and `safety.evaluate_abort_conditions`'
        # `SPEED_SCALING_FLOOR = 0.999` assume. Normalized here, once, at the
        # only place this topic is read, rather than changing either
        # threshold's unit: unnormalized, the live monitor's speed-scaling
        # abort would in practice never fire (RR_04 A-1; RR_06 P3-1 item 11).
        self._speed_scaling = float(msg.data) / 100.0

    def _on_robot_mode(self, msg: Any) -> None:
        self._robot_mode_running = bool(msg.mode == 7)  # RobotMode.RUNNING

    def _on_safety_mode(self, msg: Any) -> None:
        self._safety_mode_normal = bool(msg.mode == 1)  # SafetyMode.NORMAL

    def start_state_logging(self, match_timeout_s: float = 2.0) -> None:
        self._joint_state_log.clear()
        if self._logging_joint_state_sub is None:
            self._logging_joint_state_sub = self.create_subscription(
                DynamicJointState, self.profile.joint_state_topic, self._on_logging_joint_state,
                self._logging_qos,
            )
            # Wait for discovery so the caller's window measures data, not matching.
            deadline = time.time() + match_timeout_s
            while self._logging_joint_state_sub.get_publisher_count() == 0 and time.time() < deadline:
                rclpy.spin_once(self, timeout_sec=0.01)
        self._logging = True

    def stop_state_logging(self) -> list[tuple[float, DynamicJointState]]:
        self._logging = False
        if self._logging_joint_state_sub is not None:
            self.destroy_subscription(self._logging_joint_state_sub)
            self._logging_joint_state_sub = None
        return list(self._joint_state_log)

    def publish_event(self, run_id: str, segment_id: str, kind: str, *, plan_digest: str = "", detail: str = "") -> None:
        msg = RunEvent()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.run_id = run_id
        msg.segment_id = segment_id
        msg.kind = kind
        msg.plan_digest = plan_digest
        msg.detail = detail
        self._event_pub.publish(msg)

    # -- decoded state, by driver joint name (RR_04 A-1/A-2) --------------

    def measured_positions(self) -> dict[str, float]:
        if self._latest_joint_state is None:
            return {}
        decoded = decode_dynamic_joint_state(self._latest_joint_state)
        return {joint: values["position"] for joint, values in decoded.items() if "position" in values}

    def measured_velocities(self) -> dict[str, float]:
        if self._latest_joint_state is None:
            return {}
        decoded = decode_dynamic_joint_state(self._latest_joint_state)
        return {joint: values["velocity"] for joint, values in decoded.items() if "velocity" in values}

    def measured_effort(self) -> dict[str, float] | None:
        if self.profile.torque_abort_state_interface is None or self._latest_joint_state is None:
            return None
        decoded = decode_dynamic_joint_state(self._latest_joint_state)
        interface = self.profile.torque_abort_state_interface
        return {joint: values[interface] for joint, values in decoded.items() if interface in values}

    def fri_session_state(self) -> float | None:
        if not self.profile.has_fri_gpio or self._latest_joint_state is None:
            return None
        decoded = decode_dynamic_joint_state(self._latest_joint_state)
        return decoded.get("fri", {}).get("session_state")

    def fri_field(self, name: str) -> float | None:
        if not self.profile.has_fri_gpio or self._latest_joint_state is None:
            return None
        decoded = decode_dynamic_joint_state(self._latest_joint_state)
        return decoded.get("fri", {}).get(name)

    def latest_controller_reference(self) -> dict[str, float] | None:
        state = self._latest_controller_state
        if state is None or not state.reference.positions:
            return None
        return dict(zip(state.joint_names, state.reference.positions))

    # -- the live abort monitor (RR_04 A-1; RR_06 P3-2a) -------------------

    def _sample_age_s(self) -> float | None:
        """RR_06 P3-2a: age from the message's own **header stamp** against
        the node clock, not the wall-clock time this process happened to
        receive/process it (which a backlog on a shared subscription could
        make look falsely fresh). A zero stamp (mock never fills it in) is
        "missing", not "age zero"."""
        stamp = self._latest_joint_state_stamp
        if stamp is None or (stamp.sec == 0 and stamp.nanosec == 0):
            return None
        age_ns = self.get_clock().now().nanoseconds - Time.from_msg(stamp).nanoseconds
        return age_ns * 1e-9

    def build_monitor_sample(
        self, *, effort_limit: dict[str, float] | None = None, target_speed_fraction: float | None = None,
    ) -> MonitorSample:
        return MonitorSample(
            sample_age_s=self._sample_age_s(),
            measured_position=self.measured_positions(),
            reference_position=self.latest_controller_reference(),
            measured_effort=self.measured_effort(),
            effort_limit=effort_limit,
            fri_session_state=self.fri_session_state(),
            speed_scaling=self._speed_scaling,
            target_speed_fraction=target_speed_fraction,
            robot_mode_running=self._robot_mode_running,
            safety_mode_normal=self._safety_mode_normal,
            bag_alive=self.bag_alive(),
            sidecar_alive=self.sidecar_alive(),
        )

    def wait_for_controller_active(self, timeout_s: float = 10.0) -> dict[str, Any]:
        if not self._list_controllers.wait_for_service(timeout_sec=timeout_s):
            return {"ok": False, "reason": "controller_manager/list_controllers not available"}
        future = self._list_controllers.call_async(ListControllers.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_s)
        if not future.done() or future.result() is None:
            return {"ok": False, "reason": "list_controllers call timed out"}
        for controller in future.result().controller:
            if controller.name == self.profile.controller_name:
                return {"ok": controller.state == "active", "state": controller.state}
        return {"ok": False, "reason": f"{self.profile.controller_name!r} not found"}

    def hardware_interface_names(self, timeout_s: float = 10.0) -> set[str] | None:
        """``{"<joint>/<interface>", ...}`` from
        ``controller_manager/list_hardware_interfaces`` (RR_04 A-7's own test,
        B-3's "the exact interface set")."""
        if not self._list_hw_interfaces.wait_for_service(timeout_sec=timeout_s):
            return None
        future = self._list_hw_interfaces.call_async(ListHardwareInterfaces.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_s)
        if not future.done() or future.result() is None:
            return None
        result = future.result()
        return {hw.name for hw in list(result.command_interfaces) + list(result.state_interfaces)}

    def read_back_tolerances(
        self, driver_joint_order: Sequence[str], tracking_rad: float, *, timeout_s: float = 10.0,
    ) -> dict[str, Any]:
        """RR_06 P3-2c: read the JTC's own ``constraints.<joint>.{trajectory,
        goal}`` parameters back via ``GetParameters`` and compare against
        what the lab config (and :mod:`erd_recording.tolerances`) say they
        should be -- the launch-time override file being written is not
        itself proof the controller actually loaded it."""
        if not self._get_controller_parameters.wait_for_service(timeout_sec=timeout_s):
            return {"ok": False, "reason": f"{self.profile.controller_name}/get_parameters not available"}
        names = []
        for joint in driver_joint_order:
            names += [f"constraints.{joint}.trajectory", f"constraints.{joint}.goal"]
        future = self._get_controller_parameters.call_async(GetParameters.Request(names=names))
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_s)
        if not future.done() or future.result() is None:
            return {"ok": False, "reason": "get_parameters call timed out"}
        values = future.result().values
        problems: list[str] = []
        read_back: dict[str, dict[str, float | None]] = {}
        for index, joint in enumerate(driver_joint_order):
            trajectory_param, goal_param = values[2 * index], values[2 * index + 1]
            trajectory_value = trajectory_param.double_value if trajectory_param.type != ParameterType.PARAMETER_NOT_SET else None
            goal_value = goal_param.double_value if goal_param.type != ParameterType.PARAMETER_NOT_SET else None
            read_back[joint] = {"trajectory": trajectory_value, "goal": goal_value}
            if trajectory_value is None:
                problems.append(f"{joint}: constraints.{joint}.trajectory is not set")
            elif abs(trajectory_value - tracking_rad) > 1e-9:
                problems.append(f"{joint}: constraints.{joint}.trajectory = {trajectory_value} != "
                                f"limits.abort.tracking_rad = {tracking_rad}")
            if goal_value is None:
                problems.append(f"{joint}: constraints.{joint}.goal is not set")
            elif abs(goal_value - GOAL_TOLERANCE_RAD) > 1e-9:
                problems.append(f"{joint}: constraints.{joint}.goal = {goal_value} != {GOAL_TOLERANCE_RAD}")
        return {"ok": not problems, "values": read_back, "problems": problems}

    def competing_action_clients(self, action_name: str, *, timeout_s: float = 5.0) -> dict[str, Any]:
        """RR_06 P3-3c: refuse if any node other than this one, or one of
        ``profile.competing_action_client_allowlist``, already has a client
        on ``action_name``'s hidden services (``_action/send_goal`` et al.)
        -- iterates ``get_node_names_and_namespaces()`` and
        ``get_client_names_and_types_by_node()``, the only introspection
        rclpy exposes for "is anyone else already talking to this action
        server" (RR_04 B-3's second deferred preflight check).

        **Live finding (RR_06 P3-1 item 2):** on the UR10, every
        ``ur10.launch.py`` bring-up (mock included) always has
        ``trajectory_until_node`` (``ur_robot_driver``'s own bridge from
        ``ur_msgs/FollowJointTrajectoryUntil`` onto this exact action) as a
        permanent client -- without the allowlist this refused *every* UR10
        preflight unconditionally, which is a false positive, not a caught
        hazard: that node ships with the driver and is never the operator
        accidentally running a second commander.
        """
        hidden_services = {f"{action_name}/_action/{suffix}" for suffix in
                           ("send_goal", "cancel_goal", "get_result")}
        self_name = self.get_name()
        allowlist = set(self.profile.competing_action_client_allowlist)
        competitors: list[str] = []
        deadline = time.time() + timeout_s
        for name, namespace in self.get_node_names_and_namespaces():
            if time.time() > deadline:
                return {"ok": False, "reason": "competing_action_clients: introspection timed out"}
            if name == self_name and namespace in ("/", ""):
                continue
            if name in allowlist:
                continue
            full_name = f"{namespace.rstrip('/')}/{name}" if namespace not in ("/", "") else f"/{name}"
            try:
                clients = self.get_client_names_and_types_by_node(name, namespace)
            except Exception:  # noqa: BLE001 -- a node that disappears mid-scan is not a competitor
                continue
            for client_name, _types in clients:
                if client_name in hidden_services:
                    competitors.append(full_name)
                    break
        return {"ok": not competitors, "competitors": competitors}

    def activate_controller(self, timeout_s: float = 10.0) -> dict[str, Any]:
        """RR_04 A-4: activate the JTC through ``switch_controller`` (called
        by preflight only once the FRI session is COMMANDING_ACTIVE with
        finite states -- see :func:`stage_preflight`)."""
        if not self._switch_controller.wait_for_service(timeout_sec=timeout_s):
            return {"ok": False, "reason": "controller_manager/switch_controller not available"}
        request = SwitchController.Request()
        request.activate_controllers = [self.profile.controller_name]
        request.strictness = SwitchController.Request.STRICT
        future = self._switch_controller.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_s)
        if not future.done() or future.result() is None:
            return {"ok": False, "reason": "switch_controller call timed out"}
        return {"ok": bool(future.result().ok), "message": future.result().message}

    def measured_rate_hz(self, window_s: float = 2.0) -> dict[str, Any]:
        """RR_04 B-3: rate from message **header stamps** over the window
        (not a wall-clock poll count, which read 1049 Hz on a 1000 Hz stream
        in pass 1), cross-checked against the FRI ``cycle`` counter on the
        iiwa."""
        self.start_state_logging()
        deadline = time.time() + window_s
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
        samples = self.stop_state_logging()
        if len(samples) < 2:
            return {"rate_hz": 0.0, "n_samples": len(samples), "method": "insufficient_samples"}
        stamps = [msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9 for _, msg in samples]
        elapsed = stamps[-1] - stamps[0]
        result: dict[str, Any] = {"n_samples": len(samples)}
        if elapsed > 0:
            result["rate_hz"] = (len(samples) - 1) / elapsed
            result["method"] = "header_stamp"
        else:
            result["rate_hz"] = 0.0
            result["method"] = "header_stamp_zero_elapsed"
        if self.profile.has_fri_gpio and self.config.hardware != "mock":
            decoded_first = decode_dynamic_joint_state(samples[0][1]).get("fri", {})
            decoded_last = decode_dynamic_joint_state(samples[-1][1]).get("fri", {})
            first_cycle, last_cycle = decoded_first.get("cycle"), decoded_last.get("cycle")
            if first_cycle is not None and last_cycle is not None and elapsed > 0:
                result["rate_from_fri_cycle_hz"] = (last_cycle - first_cycle) / elapsed
                result["rate_hz"] = result["rate_from_fri_cycle_hz"]  # the more precise of the two (B-3)
                result["method"] = "fri_cycle"
        return result

    # -- trajectory execution, with the live monitor (RR_04 A-1) ----------

    def build_goal(self, trajectory: MaterializedTrajectory, driver_joint_order: tuple[str, ...],
                   sim_to_driver: dict[str, str]) -> FollowJointTrajectory.Goal:
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(driver_joint_order)
        sim_order = trajectory.joint_names
        driver_to_sim = {driver: sim for sim, driver in sim_to_driver.items()}
        driver_index = []
        for name in driver_joint_order:
            sim_name = driver_to_sim.get(name)
            if sim_name is None or sim_name not in sim_order:
                raise PipelineError(f"driver joint {name!r} has no mapping to a sim-asset joint")
            driver_index.append(sim_order.index(sim_name))
        points = []
        for row in range(len(trajectory.time)):
            point = JointTrajectoryPoint()
            point.positions = [float(trajectory.position[row, driver_index[i]]) for i in range(len(driver_joint_order))]
            point.velocities = [float(trajectory.velocity[row, driver_index[i]]) for i in range(len(driver_joint_order))]
            if trajectory.acceleration is not None:
                point.accelerations = [float(trajectory.acceleration[row, driver_index[i]])
                                       for i in range(len(driver_joint_order))]
            seconds = float(trajectory.time[row] - trajectory.time[0])
            point.time_from_start.sec = int(seconds)
            point.time_from_start.nanosec = int(round((seconds - int(seconds)) * 1e9))
            points.append(point)
        goal.trajectory.points = points
        return goal

    # -- the live abort monitor's poll loop (RR_04 A-1; RR_06 P3-2a) ------

    def tracking_rad_for(self, kind: str | None) -> float:
        """RR_14 P-3: the monitor's tracking threshold for a segment kind."""
        return segment_tracking_rad(kind, file_rad=self.monitor_tracking_rad,
                                    excitation_override=self.excitation_tracking_rad)

    def send_goal_and_wait(
        self, goal: FollowJointTrajectory.Goal, *, timeout_s: float, run_paths: RunPaths,
        effort_limit: dict[str, float] | None = None, target_speed_fraction: float | None = None,
        kind: str | None = None,
    ) -> dict[str, Any]:
        """Send one goal, then monitor every abort condition (RR_04 A-1)
        while waiting for the result, cancelling (with a confirmed stop) on
        the first trigger, a SIGINT/SIGTERM, or the timeout.

        **RR_06 P3-2a, live finding:** an earlier version of this evaluated
        the monitor from a dedicated 500 Hz ``rclpy`` timer instead of inline
        here, to decouple its cadence from whatever else this loop's
        ``spin_once`` happened to drain first. Live testing (RR_06 P3-1 item
        2) found the opposite problem: ``spin_once`` services exactly one
        ready entity per call, and once enough other 100+ Hz entities are
        also active during a real goal (controller_state, action feedback,
        the logging subscription), the timer -- always overdue after any
        blocking pause (the bag's pre-roll sleep, a standstill hold) --
        starved the monitor's own joint-state subscription of ever being
        serviced again, which starved the timer's own inputs in turn: a
        self-inflicted, silent deadlock the mock run exposed only once a
        goal was actually outstanding. Evaluating inline, once per iteration
        of this already-tight poll loop, kept that timer's failure mode from
        recurring; it is driven directly by whichever entity ``spin_once``
        just serviced.

        **RR_08 S2c:** at the iiwa's 1 kHz, measured p99 lag was 9.18 ms at
        125 Hz already (RR_07), close enough to the 10 ms budget that more
        100+ Hz competing entities crowding the same single-threaded
        ``spin_once`` were expected to push it over. The sample this loop
        evaluates (``build_monitor_sample``, reading
        ``_latest_joint_state``/``_latest_controller_state``) no longer
        depends on this loop's own ``spin_once`` ever servicing those two
        subscriptions at all -- `_MonitorFeed` keeps them fresh from its own
        executor thread regardless of what else this loop is doing. The age
        is still computed from the header stamp, not receipt time (both
        still fixed by RR_06).
        """
        if not self._action_client.wait_for_server(timeout_sec=10.0):
            return {"ok": False, "reason": "action server not available"}
        send_future = self._action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=10.0)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            return {"ok": False, "reason": "goal rejected"}
        result_future = goal_handle.get_result_async()
        deadline = time.time() + timeout_s
        torque_fraction = self.config.limits.abort.torque_fraction if effort_limit is not None else None
        self._goal_kind = kind
        self._goal_tracking_rad = self.tracking_rad_for(kind)
        # RR_06 P3-1 item 2, live finding: anything that blocked this
        # process since the last spin (the bag's 1 s pre-roll `time.sleep`,
        # a standstill hold, ...) leaves the monitor's cached sample exactly
        # that old; evaluating immediately would abort on a "stale sample"
        # that has nothing to do with the goal about to be sent. Spin until
        # a fresh one arrives (bounded) before entering the poll loop proper.
        #
        # RR_06 P3-1 item 9, live finding: both this warmup spin and the
        # poll loop below must check `abort_requested` on *every* iteration,
        # checked *before* `result_future.done()` -- a Ctrl-C delivered
        # right as a short segment starts (confirmed live: a 1-2 s
        # excitation segment on this lab's small CI-sized plan) can
        # otherwise land during the warmup spin (which never looked at
        # `abort_requested` at all) or have the goal finish on the very
        # first iteration of the old `while not result_future.done():` loop
        # (whose body, including the abort check, then never runs at all) --
        # either way, the signal was silently dropped and the run reported
        # success.
        warm_deadline = time.time() + 2.0
        while time.time() < warm_deadline:
            if self.abort_requested:
                return self._cancel_and_confirm(goal_handle, result_future, run_paths,
                                                 reason="SIGINT/SIGTERM received")
            rclpy.spin_once(self, timeout_sec=0.05)
            age = self._sample_age_s()
            if age is not None and age < STALE_SAMPLE_BUDGET_S:
                break
        while True:
            if self.abort_requested:
                return self._cancel_and_confirm(goal_handle, result_future, run_paths,
                                                 reason="SIGINT/SIGTERM received")
            if result_future.done():
                break
            rclpy.spin_once(self, timeout_sec=_MONITOR_POLL_PERIOD_S)
            sample = self.build_monitor_sample(effort_limit=effort_limit, target_speed_fraction=target_speed_fraction)
            self._monitor_evaluations += 1
            if sample.sample_age_s is not None:
                self._monitor_ages_s.append(sample.sample_age_s)
            reasons = evaluate_abort_conditions(
                sample, tracking_rad=self._goal_tracking_rad, torque_fraction=torque_fraction,
            )
            if reasons:
                return self._cancel_and_confirm(goal_handle, result_future, run_paths, reason="; ".join(reasons))
            if time.time() > deadline:
                return self._cancel_and_confirm(goal_handle, result_future, run_paths, reason="goal timed out")
        result = result_future.result()
        return {"ok": result.result.error_code == FollowJointTrajectory.Result.SUCCESSFUL,
                "error_code": result.result.error_code, "error_string": result.result.error_string}

    def _cancel_and_confirm(
        self, goal_handle: Any, result_future: Any, run_paths: RunPaths, *, reason: str,
    ) -> dict[str, Any]:
        """RR_04 A-1's ``cancel_and_stop``, corrected per RR_06 P3-2b: cancel
        the goal, then wait (up to ``abort.stop_timeout_s``) for the goal's
        own **result** to reach a terminal status (``CANCELED``, ``ABORTED``
        or ``SUCCEEDED``) -- the cancel response only confirms the request was
        *received*, not that the goal actually ended -- then a separate
        ``abort.stop_timeout_s`` budget for 0.5 s of stationary samples,
        before calling the stop confirmed."""
        from action_msgs.msg import GoalStatus

        self.publish_event(run_paths.root.name, "*", "cancel", detail=reason)
        stop_timeout_s = self.config.limits.abort.stop_timeout_s
        cancel_future = goal_handle.cancel_goal_async()
        rclpy.spin_until_future_complete(self, cancel_future, timeout_sec=stop_timeout_s)

        terminal_statuses = (GoalStatus.STATUS_CANCELED, GoalStatus.STATUS_ABORTED, GoalStatus.STATUS_SUCCEEDED)
        result_deadline = time.time() + stop_timeout_s
        while not result_future.done() and time.time() < result_deadline:
            rclpy.spin_once(self, timeout_sec=min(0.05, max(0.0, result_deadline - time.time())))
        terminal_ok = (result_future.done() and result_future.result() is not None
                      and result_future.result().status in terminal_statuses)

        stationary_ok = self.wait_for_stationary(duration_s=0.5, velocity_tolerance=0.01, timeout_s=stop_timeout_s)
        stop_confirmed = terminal_ok and stationary_ok
        self._monitor_cancels.append({"reason": reason, "stop_confirmed": stop_confirmed,
                                      "segment_kind": self._goal_kind, "tracking_rad": self._goal_tracking_rad,
                                      "terminal_status_seen": bool(terminal_ok), "wall_time": time.time()})
        if not stop_confirmed:
            self.get_logger().error("STOP NOT CONFIRMED -- use the e-stop")
        return {"ok": False, "reason": reason, "stop_confirmed": stop_confirmed}

    def write_monitor_json(self, path: Path, stage: str) -> dict[str, Any]:
        """RR_12 A-2b: sample-age statistics over every monitor evaluation of
        this stage, merged into ``path`` under ``stage``; then reset."""
        ages_ms = np.asarray(self._monitor_ages_s, dtype=float) * 1e3
        entry: dict[str, Any] = {
            "evaluations": self._monitor_evaluations, "aged_samples": int(len(ages_ms)),
            "tracking_rad": self.monitor_tracking_rad,
            "excitation_tracking_rad": self.tracking_rad_for("excitation"),
            "tracking_rad_overridden": self.excitation_tracking_rad is not None,
            "poll_period_s": _MONITOR_POLL_PERIOD_S, "stale_budget_ms": STALE_SAMPLE_BUDGET_S * 1e3,
            "cancels": list(self._monitor_cancels),
        }
        if len(ages_ms):
            entry["sample_age_ms"] = {"p50": float(np.percentile(ages_ms, 50)), "p99": float(np.percentile(ages_ms, 99)),
                                      "max": float(ages_ms.max())}
        data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        data[stage] = entry
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        self._monitor_ages_s, self._monitor_evaluations, self._monitor_cancels = [], 0, []
        return entry

    def controller_states(self, timeout_s: float = 10.0) -> dict[str, str] | None:
        """``{controller: state}`` from ``controller_manager/list_controllers``."""
        if not self._list_controllers.wait_for_service(timeout_sec=timeout_s):
            return None
        future = self._list_controllers.call_async(ListControllers.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_s)
        if not future.done() or future.result() is None:
            return None
        return {controller.name: controller.state for controller in future.result().controller}

    def robot_software_version(self, timeout_s: float = 5.0) -> tuple[int, int, int, int] | None:
        """RR_12 A-3b: (major, minor, bugfix, build) from the UR controller."""
        if self.profile.software_version_service is None:
            return None
        from ur_msgs.srv import GetRobotSoftwareVersion

        client = self.create_client(GetRobotSoftwareVersion, self.profile.software_version_service)
        try:
            if not client.wait_for_service(timeout_sec=timeout_s):
                return None
            future = client.call_async(GetRobotSoftwareVersion.Request())
            rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_s)
            if not future.done() or future.result() is None:
                return None
            result = future.result()
            return (int(result.major), int(result.minor), int(result.bugfix), int(result.build))
        finally:
            self.destroy_client(client)

    def wait_for_stationary(self, *, duration_s: float, velocity_tolerance: float, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        stationary_since_ns = None
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.02)
            velocities = self.measured_velocities()
            age = self._sample_age_s()
            still = (age is not None and 0 <= age < STALE_SAMPLE_BUDGET_S and bool(velocities)
                     and all(np.isfinite(v) and abs(v) < velocity_tolerance for v in velocities.values()))
            if still:
                stamp = self._latest_joint_state_stamp
                stamp_ns = stamp.sec * 10**9 + stamp.nanosec
                if stationary_since_ns is None:
                    stationary_since_ns = stamp_ns
                if (stamp_ns - stationary_since_ns) * 1e-9 >= duration_s:
                    return True
            else:
                stationary_since_ns = None
        return False

    # -- bag lifecycle (RR_04 A-11) ----------------------------------------

    #: RR_16 Q-1(a): ``None`` (production) records through
    #: :mod:`erd_recording.bag_record` with only the ``rosbag2_recorder``
    #: logger at DEBUG, so every bag's stderr log carries rosbag2's per-topic
    #: "Messages lost on transport layer" events. A value is one *global*
    #: level for every logger (``--recorder-log-level``, diagnostic only:
    #: ``debug`` writes ~0.5 MB/s of rcl lines; no other level is accepted,
    #: since below DEBUG the "Subscribed to topic" lines R-5 waits for vanish).
    recorder_log_level: str | None = None

    def sidecar_expected(self) -> bool:
        """Does a stage that records RTDE start the sidecar here? (Never on
        mock: there is no RTDE server, RR_06 P3-1 item 2.)"""
        return self.profile.has_ur_status_topics and self.config.hardware != "mock"

    def start_bag(self, bag_dir: Path, *, sidecar: bool = False, extra_topics: list[str] | None = None) -> None:
        """Start the recorder and block until it has subscribed to every
        listed topic but the sidecar's and is >= 2 s past "Recording...".
        A topic still missing after 15 s stops the bag and raises (RR_18 R-5).
        ``sidecar``: the stage will start the RTDE sidecar, so the bag also
        lists ``/erd/rtde_status`` (confirmed by :meth:`confirm_sidecar_topic`)."""
        topics = bag_topics_for(self.profile, sidecar=sidecar) + list(extra_topics or [])
        bag_dir.parent.mkdir(parents=True, exist_ok=True)
        log_dir = bag_dir.parent / "raw"
        log_dir.mkdir(parents=True, exist_ok=True)
        # RR_13 B-2: every 1 kHz stream gets a deep reliable reader (rosbag2's
        # default depth is 10). This absorbs recorder callback stalls; it cannot
        # recover messages a depth-1 *writer* (JSB /joint_states, JTC
        # controller_state) has already overwritten -- see the pre-roll below.
        qos_path = write_qos_overrides(log_dir / f"{bag_dir.name}.qos.yaml", reliable_deep_topics=[
            "/dynamic_joint_states", "/joint_states", f"/{self.profile.controller_name}/controller_state"])
        self._bag_stdout_file = open(log_dir / f"{bag_dir.name}.rosbag2.stdout.log", "wb")
        self._bag_stderr_file = open(log_dir / f"{bag_dir.name}.rosbag2.stderr.log", "wb")
        # RR_16 Q-1(a): ros2bag's own `record` verb (same parser, same
        # StorageOptions/RecordOptions), with a per-logger level.
        level = [] if self.recorder_log_level is None else ["--erd-global-log-level", self.recorder_log_level]
        self._bag_process = subprocess.Popen(
            [sys.executable, "-m", "erd_recording.bag_record", "-s", "mcap", "-o", str(bag_dir),
             "--include-hidden-topics", *level, "--qos-profile-overrides-path", str(qos_path), "--topics", *topics],
            stdout=self._bag_stdout_file, stderr=self._bag_stderr_file,
        )
        # RR_01 S3.6 pre-roll. RR_13, two findings that a blind 1 s missed:
        # * discovery can take 1.4 s: erd_monitor_test's first 0.32 s of
        #   controller_state never reached the bag;
        # * in about one bag in three the recorder stops receiving for ~0.33 s
        #   at a fixed 1.10-1.44 s after its "Recording..." line (5 of 5
        #   affected bags), and the depth-1 publishers (/joint_states, JTC
        #   controller_state) lose ~175 messages each. l2_iiwa_emustop's first
        #   goals fell inside that window.
        # So: wait for every topic subscribed and >= 2 s past "Recording...".
        # RR_18 R-5: per topic, from rosbag2's own "Subscribed to topic"
        # lines, without the sidecar's topic (it appears only once the
        # sidecar runs: every UR bag used to wait the full 15 s, F-14); a
        # topic still missing fails the stage instead of warning.
        self._bag_stderr_path = log_dir / f"{bag_dir.name}.rosbag2.stderr.log"
        state = wait_for_bag_start(lambda: self._bag_stderr_path.read_text(errors="replace"),
                                   [topic for topic in topics if topic != SIDECAR_STATUS_TOPIC],
                                   timeout_s=SUBSCRIBE_TIMEOUT_S, alive=lambda: self._bag_process.poll() is None)
        if not state["ok"]:
            self.stop_bag()
            raise PipelineError(f"rosbag2 start failed ({state['reason']}): no \"Subscribed to topic\" line for "
                                f"{state['missing']}; bag stopped, no goal sent")

    def confirm_sidecar_topic(self) -> None:
        """RR_18 R-5: after D-6's wait for RTDE rows, the bag must have
        subscribed to ``/erd/rtde_status`` within 5 s, before the first goal."""
        state = wait_for_bag_start(lambda: self._bag_stderr_path.read_text(errors="replace"), [SIDECAR_STATUS_TOPIC],
                                   timeout_s=SIDECAR_SUBSCRIBE_TIMEOUT_S, startup_s=0.0,
                                   alive=lambda: self._bag_process is not None and self._bag_process.poll() is None)
        if not state["ok"]:
            raise PipelineError(f"rosbag2 did not subscribe to {SIDECAR_STATUS_TOPIC} ({state['reason']}); "
                                "no goal sent")
    def bag_alive(self) -> bool:
        return self._bag_process is None or self._bag_process.poll() is None

    def stop_bag(self) -> int | None:
        if self._bag_process is None:
            return None
        self._bag_process.send_signal(signal.SIGINT)
        try:
            self._bag_process.wait(timeout=15.0)
        except subprocess.TimeoutExpired:
            self._bag_process.terminate()
            self._bag_process.wait(timeout=5.0)
        returncode = self._bag_process.returncode
        self._bag_process = None
        for handle in (self._bag_stdout_file, self._bag_stderr_file):
            if handle is not None:
                handle.close()
        self._bag_stdout_file = self._bag_stderr_file = None
        return returncode

    def check_sidecar_actual_q_alignment(
        self, output_dir: Path, *, robot_ip: str, driver_joint_order: Sequence[str],
        window_s: float = 4.0, match_fraction: float = 0.95,
    ) -> dict[str, Any]:
        """RR_06 P3-3c / RR_04 B-3: start a throwaway RTDE sidecar connection
        alongside the driver's own, log both for ``window_s``, and compare
        the last (up to) 125 rows of the sidecar's ``actual_q`` against the
        driver's own ``/dynamic_joint_states`` position at the matching row
        -- both are independent reads of the same CB3 controller state, so a
        real driver/sidecar mismatch (wrong ``robot_ip``, a stale connection)
        shows up as an exact-equality failure, not a tolerance one.

        **RR_06 P3-1 item 11, live finding:** ``window_s`` was ``1.5`` --
        too tight against the real sidecar's own 1 s flush timer plus RTDE
        connection/handshake overhead (confirmed against URSim: the sidecar
        wrote zero rows at 1.5 s, then matched 125/125 exactly at 4.0 s)."""
        if not self.profile.has_ur_status_topics:
            return {"ok": True, "skipped": "not a UR10 profile"}
        import pandas as pd

        self.start_state_logging()
        self.start_sidecar(output_dir, robot_ip=robot_ip)
        deadline = time.time() + window_s
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.02)
            if self._sidecar_process is not None and self._sidecar_process.poll() is not None:
                break
        driver_log = self.stop_state_logging()
        self.stop_sidecar()

        if not output_dir.is_dir() or not any(output_dir.iterdir()):
            return {"ok": False, "reason": "sidecar wrote no rtde parquet rows"}
        sidecar = pd.read_parquet(output_dir)
        if sidecar.empty or not driver_log:
            return {"ok": False, "reason": "no sidecar rows or no driver /dynamic_joint_states samples"}

        n = min(125, len(sidecar), len(driver_log))
        sidecar_tail = sidecar.tail(n).reset_index(drop=True)
        driver_tail = driver_log[-n:]

        matches = 0
        for row_index in range(n):
            decoded = decode_dynamic_joint_state(driver_tail[row_index][1])
            row_ok = True
            for joint_index, joint in enumerate(driver_joint_order):
                driver_value = decoded.get(joint, {}).get("position")
                sidecar_value = sidecar_tail.loc[row_index, f"actual_q{joint_index}"]
                if driver_value is None or abs(float(driver_value) - float(sidecar_value)) > 1e-9:
                    row_ok = False
                    break
            if row_ok:
                matches += 1
        fraction = matches / n if n else 0.0
        return {"ok": fraction >= match_fraction, "fraction_matched": fraction, "n_rows_compared": n}

    # -- RTDE sidecar lifecycle (RR_04 A-12: started/stopped per run, not by
    # the launch file) --------------------------------------------------

    def start_sidecar(self, output_path: Path, *, robot_ip: str) -> None:
        """Launch the RTDE sidecar and block until it reports rows (D-6)."""
        if self.launch_sidecar(output_path, robot_ip=robot_ip):
            self._wait_for_sidecar_rows()

    def launch_sidecar(self, output_path: Path, *, robot_ip: str) -> bool:
        """Start the sidecar process without waiting for rows (RR_19: a stage
        launches it just before the recorder, so its RTDE connect overlaps the
        bag's 2 s start wait; the rows are awaited after ``start_bag``, by
        :meth:`wait_for_sidecar_rows`). False when this stack has no sidecar."""
        # RR_06 P3-1 item 2, live finding: there is no real RTDE server to
        # connect to at hardware=mock (confirmed live: the sidecar process
        # starts, fails to connect, and exits almost immediately, which the
        # live monitor then correctly reads as "the sidecar died" and
        # aborts every single mock run, RR_04 A-1's own abort table doing
        # exactly its job against a condition that was never real).
        # `sidecar_alive()` already treats "never started" (`None`) as
        # alive, so skipping the start here is enough -- no change needed
        # at the monitor end.
        if not self.sidecar_expected():
            return False
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # RR_06 P3-1 item 11, live findings, two of them:
        # 1. `ros2 run <pkg> <exe>` launches the real executable as a
        #    *child* of the `ros2 run` wrapper process (confirmed live: two
        #    distinct PIDs), so `stop_sidecar`'s SIGINT to this `Popen`'s own
        #    pid only ever reached the wrapper -- the actual `rtde_logger`
        #    process, and its live RTDE connection, was never stopped and
        #    leaked past every run.
        # 2. The obvious fix -- invoke `rtde_logger` found on `PATH` directly
        #    -- resolves to `install/erd_ur10/bin/rtde_logger`, which PATH
        #    lists *before* the correct `install/erd_ur10/lib/erd_ur10/`
        #    copy `ros2 run` itself would have used; the `bin/` copy's
        #    shebang is a plain `/usr/bin/python3` (confirmed live:
        #    `ModuleNotFoundError: No module named 'pandas'`), not
        #    `.venv-erd`'s interpreter -- unlike every console script
        #    `erd_recording` installs, which only ever lands in `lib/`, this
        #    package's own `setup.py` additionally produces that broken
        #    `bin/` copy (a packaging inconsistency worth fixing in
        #    `erd_ur10/setup.py` itself, separately from this call site).
        # Resolving the executable through `ament_index_python`'s own
        # package-prefix lookup, the same mechanism `ros2 run` uses
        # internally, sidesteps both: it is always the correct, venv-aware
        # `lib/erd_ur10/rtde_logger`, and `self._sidecar_process` is the
        # real process this function's own `stop_sidecar` SIGINTs.
        from ament_index_python.packages import get_package_prefix

        rtde_logger_path = str(Path(get_package_prefix("erd_ur10")) / "lib" / "erd_ur10" / "rtde_logger")
        self._sidecar_process = subprocess.Popen(
            [rtde_logger_path, "--ros-args",
            "-p", f"robot_ip:={robot_ip}", "-p", "frequency:=125.0",
            "-p", f"output_path:={output_path}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return True

    def wait_for_sidecar_rows(self) -> None:
        """D-6, then RR_18 R-5: the sidecar reports RTDE rows, and the bag has
        subscribed to its status topic, before the first goal."""
        self._wait_for_sidecar_rows()
        self.confirm_sidecar_topic()

    def _wait_for_sidecar_rows(self, *, timeout_s: float = SIDECAR_STARTUP_TIMEOUT_S) -> None:
        """RR_15: the sidecar connects to RTDE ~0.5 s after it starts; a first
        segment that begins sooner (an identify hold at the home pose has a
        ~50 ms approach) loses its first rows (rr15_ur10_sim: hold_0 began
        0.24 s before the first RTDE row, 565 of 588 samples). Block until
        its `/erd/rtde_status` (10 Hz since RR_19) reports rows."""
        from std_msgs.msg import String

        state = {"rows": 0}

        def on_status(msg: String) -> None:
            try:
                state["rows"] = int(json.loads(msg.data).get("rows") or 0)
            except (ValueError, TypeError):
                pass

        subscription = self.create_subscription(String, "/erd/rtde_status", on_status, 10)
        try:
            deadline = time.time() + timeout_s
            while state["rows"] <= 0:
                if time.time() > deadline or not self.sidecar_alive():
                    raise PipelineError(f"RTDE sidecar reported no rows within {timeout_s} s "
                                        f"(alive: {self.sidecar_alive()})")
                rclpy.spin_once(self, timeout_sec=0.1)
        finally:
            self.destroy_subscription(subscription)

    def sidecar_alive(self) -> bool:
        return self._sidecar_process is None or self._sidecar_process.poll() is None

    def stop_sidecar(self) -> int | None:
        if self._sidecar_process is None:
            return None
        self._sidecar_process.send_signal(signal.SIGINT)
        try:
            self._sidecar_process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            self._sidecar_process.terminate()
            self._sidecar_process.wait(timeout=5.0)
        returncode = self._sidecar_process.returncode
        self._sidecar_process = None
        return returncode


# ---------------------------------------------------------------------------
# `preflight` (RR_04 B-3)
# ---------------------------------------------------------------------------


def stage_preflight(node: RecordingNode, config: LabConfig, paths: RunPaths) -> dict[str, Any]:
    checks: dict[str, Any] = {}

    # RR_06 P3-3c / RR_04 B-3: no node other than ours has a client on our
    # action's hidden services (checked early, before any goal is sent).
    action_name = f"/{node.profile.controller_name}/{node.profile.controller_action}"
    checks["no_competing_action_clients"] = node.competing_action_clients(action_name)

    if not node.profile.controller_spawns_active and config.hardware != "mock":
        # RR_04 A-4: outside mock, activate the JTC only once the FRI session
        # has been COMMANDING_ACTIVE with finite states for 100 consecutive
        # cycles -- never before, so it never latches a NaN/stale IPO hold.
        stable_cycles = 0
        previous_cycle = None
        deadline = time.time() + 10.0
        while stable_cycles < 100 and time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.01)
            positions = node.measured_positions()
            all_finite = bool(positions) and all(np.isfinite(v) for v in positions.values())
            if node.fri_session_state() == FRI_COMMANDING_ACTIVE and all_finite:
                cycle = node.fri_field("cycle")
                if cycle != previous_cycle:
                    stable_cycles += 1
                    previous_cycle = cycle
            else:
                stable_cycles = 0
        checks["fri_stable_before_activation"] = {"ok": stable_cycles >= 100, "cycles_seen": stable_cycles}
        activation = node.wait_for_controller_active()
        if not activation["ok"]:
            activation = node.activate_controller()
        checks["controller_activation"] = activation

    controller = node.wait_for_controller_active()
    checks["controller_active"] = controller

    rate = node.measured_rate_hz(window_s=1.0)
    rate_ok = abs(rate["rate_hz"] - config.rate_hz) / config.rate_hz < 0.01 if rate["rate_hz"] > 0 else False
    checks["measured_rate"] = {**rate, "ok": rate_ok}

    interfaces = node.hardware_interface_names()
    checks["hardware_interfaces"] = {"ok": interfaces is not None and len(interfaces) > 0, "names": sorted(interfaces or [])}

    home_by_driver = {driver: config.poses.home[config.joint_order.index(sim)]
                      for sim, driver in config.description.sim_to_driver().items()}
    home_guard = check_at_home_and_still(node.measured_positions(), node.measured_velocities(), home_by_driver)
    checks["home_and_still"] = {"ok": home_guard.ok, "reason": home_guard.reason}

    if node.profile.has_fri_gpio:
        command_mode_ok = node.fri_field("command_mode") == 1  # POSITION (RR_01 S3.3)
        sample_time = node.fri_field("sample_time") or 0.0
        expected_sample_time = config.connection.fri_send_period_ms / 1000.0
        sample_time_ok = abs(sample_time - expected_sample_time) < 1e-4
        checks["fri_mode"] = {"ok": command_mode_ok and sample_time_ok, "command_mode": node.fri_field("command_mode"),
                              "sample_time": sample_time, "expected_sample_time": expected_sample_time}

    if node.profile.has_ur_status_topics:
        # RR_06 P3-1 item 2, live finding: `robot_mode`/`safety_mode` (and
        # the RTDE server `sidecar_actual_q_alignment` needs) are properties
        # of the real CB3 controller/dashboard -- `ur_description`'s mock
        # xacro declares those two GPIO state interfaces with no
        # `initial_value` (ros2_control's default is NaN), and
        # `io_and_status_controller` never publishes `robot_mode`/
        # `safety_mode` at all from a NaN source (confirmed live: the topics
        # exist, with an active publisher, but emit nothing). There is no
        # `use_mock_hardware` passthrough to override this (unlike the iiwa
        # side, where erd_iiwa's own xacro call already controls every mock
        # initial value) -- fully out of this pass's reach without vendoring
        # `ur_description`. Enforced for real; at mock, reported but not
        # gated, since there is nothing real underneath to check.
        mock = config.hardware == "mock"
        if not mock:
            # RR_06 P3-1 item 11, live finding: `/io_and_status_controller/
            # robot_mode` (and `/safety_mode`) publish at a rate low enough
            # that, against real URSim, it had not yet delivered a single
            # message by the time this check ran, right behind
            # `measured_rate_hz`'s own 1 s spin window -- which that window
            # does not itself favour, since it exists to drain
            # `/dynamic_joint_states`'s own high-rate stream, not these.
            # Wait specifically for one, rather than trust whatever spin_once
            # opportunistically picked up as a side effect of an earlier
            # check's own spin loop.
            deadline = time.time() + 2.0
            while node._robot_mode_running is None and time.time() < deadline:
                rclpy.spin_once(node, timeout_sec=0.05)
        speed_scaling_ok = mock or (node._speed_scaling is not None and abs(node._speed_scaling - 1.0) < 1e-6)
        checks["speed_scaling"] = {"ok": speed_scaling_ok, "value": node._speed_scaling,
                                   "skipped": "hardware=mock never publishes a real value" if mock else None}
        checks["robot_mode_running"] = {"ok": mock or bool(node._robot_mode_running),
                                        "skipped": "hardware=mock never publishes a real value" if mock else None}
        checks["sidecar_alive"] = {"ok": node.sidecar_alive()}
        # RR_06 P3-3c / RR_04 B-3: a throwaway sidecar connection, compared
        # against the driver's own state, before the run's real one starts
        # -- needs a real RTDE server (port 30004), which mock has none of.
        if mock:
            checks["sidecar_actual_q_alignment"] = {"ok": True, "skipped": "hardware=mock has no RTDE server"}
        else:
            checks["sidecar_actual_q_alignment"] = node.check_sidecar_actual_q_alignment(
                paths.root / "preflight_rtde.parquet", robot_ip=config.connection.robot_ip,
                driver_joint_order=node.profile.driver_joint_order,
            )

    # RR_06 P3-2c: outside mock, the JTC's path/goal tolerances are now
    # mandatory at launch (both launch files refuse to start without
    # `lab_config:=`), so preflight can always verify the controller actually
    # loaded them.
    if config.hardware != "mock":
        checks["jtc_tolerances"] = node.read_back_tolerances(
            node.profile.driver_joint_order, config.limits.abort.tracking_rad,
        )

    # RR_12 A-3a: exactly the expected controllers active. Gated on real
    # hardware; listed (and judged, ungated) at L1/L2.
    states = node.controller_states()
    active_set = check_active_controllers(states or {}, node.profile.expected_active_controllers)
    if states is None:
        active_set.update(ok=False, reason="list_controllers did not answer")
    active_set["gated"] = config.hardware == "real"
    checks["active_controllers"] = active_set if config.hardware == "real" else {**active_set, "ok": True,
                                                                                    "would_pass": active_set["ok"]}
    # RR_12 A-3b: the UR controller's software version is the configured one.
    if node.profile.software_version_service is not None and config.hardware != "mock":
        version = check_software_version(node.robot_software_version(), config.connection.software_version)
        version["gated"] = config.hardware == "real"
        checks["software_version"] = version if config.hardware == "real" else {**version, "ok": True,
                                                                                 "would_pass": version["ok"]}
    # RR_12 A-3c: the vendor safety checksum is on file (the loader already
    # refuses a null one on real hardware).
    checks["safety_vendor_checksum"] = {"ok": config.hardware != "real" or bool(config.safety.vendor_checksum),
                                        "value": config.safety.vendor_checksum}

    free_bytes = shutil.disk_usage(config.recording.output_root).free
    checks["disk_space"] = {"ok": free_bytes >= 2 * 1024**3, "free_gb": free_bytes / 1024**3}

    ok = all(bool(v.get("ok")) for v in checks.values())
    report = {"checks": checks, "ok": ok, "timestamp": time.time()}
    paths.preflight.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _preflight_is_fresh(paths: RunPaths, max_age_s: float = 3600.0) -> bool:
    if not paths.preflight.is_file():
        return False
    report = json.loads(paths.preflight.read_text(encoding="utf-8"))
    return bool(report.get("ok")) and (time.time() - report.get("timestamp", 0)) < max_age_s


# ---------------------------------------------------------------------------
# `standstill` (RR_04 A-3: frozen, collision-checked plan segments, recorded)
# ---------------------------------------------------------------------------


def stage_standstill(
    node: RecordingNode, config: LabConfig, paths: RunPaths, bundle: PlanBundle,
    sim_to_driver: dict[str, str], driver_joint_order: tuple[str, ...],
) -> dict[str, Any]:
    if not _preflight_is_fresh(paths):
        raise PipelineError("standstill refused: preflight not passed")

    effort_limit = None
    if node.profile.torque_abort_state_interface is not None:
        arrays = config.limits.as_arrays(config.joint_order)
        effort_limit = {sim_to_driver[sim]: float(arrays["effort"][i])
                        for i, sim in enumerate(config.joint_order) if sim in sim_to_driver}

    node.start_bag(paths.standstill_bag_dir)
    try:
        if not node.wait_for_stationary(duration_s=0.5, velocity_tolerance=0.01, timeout_s=5.0):
            raise PipelineError("standstill refused: fresh stationary joint states unavailable")
        for index in range(len(config.poses.standstill)):
            approach = bundle.segment(f"standstill_approach_{index}")
            first_point = _first_point_by_driver_name(approach.trajectory, driver_joint_order, sim_to_driver)
            guard = check_start_state(node.measured_positions(), first_point)
            if not guard.ok:
                _write_manifest_status(paths, "failed", reason=guard.reason)
                return {"ok": False, "reason": guard.reason}

            node.publish_event(paths.root.name, approach.segment_id, f"{approach.kind}_start", plan_digest=approach.digest)
            goal = node.build_goal(approach.trajectory, driver_joint_order, sim_to_driver)
            result = node.send_goal_and_wait(goal, timeout_s=approach.trajectory.duration + 10.0,
                                             run_paths=paths, effort_limit=effort_limit, kind=approach.kind)
            if not result["ok"]:
                _write_manifest_status(paths, "failed", failed_segment=approach.segment_id, reason=result)
                return {"ok": False, "failed_segment": approach.segment_id, "reason": result}
            node.publish_event(paths.root.name, approach.segment_id, f"{approach.kind}_end", plan_digest=approach.digest)

            node.publish_event(paths.root.name, f"standstill_hold_{index}", "standstill_start")
            time.sleep(config.recording.standstill_s)
            node.publish_event(paths.root.name, f"standstill_hold_{index}", "standstill_end")

            ret = bundle.segment(f"standstill_return_{index}")
            node.publish_event(paths.root.name, ret.segment_id, f"{ret.kind}_start", plan_digest=ret.digest)
            goal = node.build_goal(ret.trajectory, driver_joint_order, sim_to_driver)
            result = node.send_goal_and_wait(goal, timeout_s=ret.trajectory.duration + 10.0,
                                             run_paths=paths, effort_limit=effort_limit, kind=ret.kind)
            if not result["ok"]:
                _write_manifest_status(paths, "failed", failed_segment=ret.segment_id, reason=result)
                return {"ok": False, "failed_segment": ret.segment_id, "reason": result}
            node.publish_event(paths.root.name, ret.segment_id, f"{ret.kind}_end", plan_digest=ret.digest)
    finally:
        node.stop_bag()
        node.write_monitor_json(paths.root / "monitor.json", "standstill")
    _write_manifest_status(paths, "planned", standstill_completed=True)
    return {"ok": True}


# ---------------------------------------------------------------------------
# `run` (RR_04 A-1/A-2/A-11/A-12)
# ---------------------------------------------------------------------------


#: RR_06 P3-3b: the ladder's own segment kinds, appended to a plan bundle by
#: `planning.build_ladder_segments` when `plan` was given `--ladder`, never
#: mixed with the full-scale kinds in the same `run`.
_COMMISSIONING_KINDS = frozenset({"approach_commissioning", "excitation_commissioning", "return_commissioning"})
_EXCITATION_KINDS = EXCITATION_KINDS


def _select_segments(bundle: PlanBundle, *, ladder: bool, only: frozenset[str] | None) -> list[PlanSegment]:
    """RR_06 P3-3b: which of ``bundle.segments`` this ``run`` executes
    (``standstill_segments`` are a separate population, always run in full by
    `stage_standstill`). ``ladder`` selects the commissioning-tagged
    population when ``--ladder`` was given to ``plan``, the full-scale one
    otherwise -- the two are never mixed, since one ``plan`` invocation only
    ever builds one ladder scale. ``only`` then restricts that population to
    exactly the named segment ids (RR_01 "Options": ``--only <ids>``),
    verbatim -- it does not re-chain the remaining approach/return quintics
    around whatever gap this might open; RR_04 A-2's start-state guard is
    what catches an ``--only`` subset that isn't actually a safe, physically
    continuous subsequence, by refusing the jump rather than executing it."""
    population = [s for s in bundle.segments if (s.kind in _COMMISSIONING_KINDS) == ladder]
    if only is None:
        return population
    return [s for s in population if s.segment_id in only]


def _confirm_trajectory(segment: PlanSegment, *, no_confirm: bool) -> None:
    """RR_06 P3-3b / RR_01 S9: ``confirm_each_trajectory`` defaults to true
    -- an operator confirms before every excitation goal. ``--no-confirm``
    skips the prompt entirely; without it, running with no TTY attached is
    refused outright (RR_06: "refuses to run unattended ... unless
    --no-confirm"), rather than silently skipped or hung waiting on a prompt
    nobody can answer."""
    if no_confirm:
        return
    if not sys.stdin.isatty():
        raise PipelineError(
            f"segment {segment.segment_id!r}: confirm_each_trajectory is set and no TTY is attached -- "
            "pass --no-confirm to run unattended"
        )
    answer = input(f"About to run {segment.kind} {segment.segment_id!r} -- press Enter to continue, "
                   "or type 'n' to abort: ")
    if answer.strip().lower() in ("n", "no"):
        raise PipelineError(f"segment {segment.segment_id!r}: operator declined at the confirm prompt")


def stage_run(node: RecordingNode, config: LabConfig, paths: RunPaths, bundle: PlanBundle,
              sim_to_driver: dict[str, str], driver_joint_order: tuple[str, ...], *,
              only: frozenset[str] | None = None, ladder_scale: float | None = None,
              no_confirm: bool = False, stage_name: str = "run") -> dict[str, Any]:
    if not _preflight_is_fresh(paths):
        raise PipelineError("run refused: preflight not passed")
    from .planning import config_digest

    if bundle.config_digest != config_digest(config):
        raise PipelineError("run refused: plan digest does not match the frozen config")

    segments_to_run = _select_segments(bundle, ladder=ladder_scale is not None, only=only)

    effort_limit = None
    if node.profile.torque_abort_state_interface is not None:
        arrays = config.limits.as_arrays(config.joint_order)
        effort_limit = {sim_to_driver[sim]: float(arrays["effort"][i])
                        for i, sim in enumerate(config.joint_order) if sim in sim_to_driver}

    completed_segments: list[str] = []
    try:
        # RR_18 R-5: the sidecar's RTDE connect (~0.5 s) and first 1 Hz status
        # overlap the bag's >= 2 s start wait; its rows (D-6) and the bag's
        # /erd/rtde_status subscription are confirmed after it, before any goal.
        sidecar = node.launch_sidecar(paths.rtde_parquet, robot_ip=config.connection.robot_ip)
        node.start_bag(paths.bag_dir, sidecar=sidecar)
        if sidecar:
            node.wait_for_sidecar_rows()
        if not node.wait_for_stationary(duration_s=0.5, velocity_tolerance=0.01, timeout_s=5.0):
            raise PipelineError("run refused: fresh stationary joint states unavailable")
        for segment in segments_to_run:
            first_point = _first_point_by_driver_name(segment.trajectory, driver_joint_order, sim_to_driver)
            guard = check_start_state(node.measured_positions(), first_point)
            if not guard.ok:
                node.publish_event(paths.root.name, segment.segment_id, "fault", detail=guard.reason)
                _write_manifest_status(paths, "failed", failed_segment=segment.segment_id, reason=guard.reason)
                return {"ok": False, "failed_segment": segment.segment_id, "reason": guard.reason}

            if segment.kind in _EXCITATION_KINDS and config.operator.confirm_each_trajectory:
                _confirm_trajectory(segment, no_confirm=no_confirm)

            node.publish_event(paths.root.name, segment.segment_id, f"{segment.kind}_start", plan_digest=segment.digest)
            goal = node.build_goal(segment.trajectory, driver_joint_order, sim_to_driver)
            target_speed_fraction = 1.0 if segment.kind in _EXCITATION_KINDS else None
            result = node.send_goal_and_wait(goal, timeout_s=segment.trajectory.duration + 15.0, run_paths=paths,
                                             effort_limit=effort_limit, target_speed_fraction=target_speed_fraction,
                                             kind=segment.kind)
            if not result["ok"]:
                node.publish_event(paths.root.name, segment.segment_id, "fault", detail=str(result))
                _write_manifest_status(paths, "failed", failed_segment=segment.segment_id, reason=result)
                return {"ok": False, "failed_segment": segment.segment_id, "reason": result}
            node.publish_event(paths.root.name, segment.segment_id, f"{segment.kind}_end", plan_digest=segment.digest)
            completed_segments.append(segment.segment_id)
    finally:
        node.stop_bag()
        node.stop_sidecar()
        node.write_monitor_json(paths.root / "monitor.json", stage_name)
    _write_manifest_status(paths, "recorded", segments=completed_segments)
    return {"ok": True, "segments": completed_segments}


def cancel_and_stop(node: RecordingNode, paths: RunPaths, *, reason: str) -> None:
    """Stop-stage-level cancel (no goal currently active, e.g. a SIGINT
    between stages): stop the bag/sidecar and mark the run failed. See
    :meth:`RecordingNode._cancel_and_confirm` for the mid-goal case, which
    additionally cancels the action and confirms a stationary stop (RR_04
    A-1)."""
    node.publish_event(paths.root.name, "*", "cancel", detail=reason)
    node.stop_bag()
    node.stop_sidecar()
    _write_manifest_status(paths, "failed", reason=reason)


def stage_identify(node, config, config_path, paths, bundle, sim_to_driver, driver_joint_order, *, no_confirm=False,
                   reference_contract=None, reference_sha256=None):
    if not bundle.identify_segments:
        raise PipelineError("identify refused: rebuild the plan to freeze identification segments")
    recording = RunPaths(paths.root / "identification")
    recording.root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(paths.root / "preflight.json", recording.root / "preflight.json")
    identification_bundle = replace(bundle, segments=bundle.identify_segments, standstill_segments=(), identify_segments=())
    result = stage_run(node, config, recording, identification_bundle, sim_to_driver, driver_joint_order,
                       no_confirm=no_confirm, stage_name="identify")
    if not result['ok']:
        _write_manifest_status(paths, "failed", identification=result)
        raise PipelineError(f"identify failed: {result}")
    _prepare_conversion(config, recording.root)
    command = [sys.executable, '-m', 'erd_recording.identify_cli', '--config', config_path, '--run-dir', str(paths.root)]
    if reference_contract:
        command += ['--reference-contract', reference_contract]
    if reference_sha256:
        command += ['--reference-sha256', reference_sha256]
    fit = subprocess.run(command, env=clean_ros_subprocess_env(), capture_output=True, text=True)
    if fit.returncode != 0:
        # RR_12 B-1/B-2: an incomplete or lossy fit-input segment fails the
        # stage; identification.json (if written) names the segments.
        _write_manifest_status(paths, "failed", identification={"ok": False, "stdout": fit.stdout[-2000:],
                                                                "stderr": fit.stderr[-2000:]})
        raise PipelineError(f"identify failed:\nstdout={fit.stdout}\nstderr={fit.stderr}")
    _write_manifest_status(paths, 'planned', identification_completed=True)
    return result


def _prepare_conversion(config, root):
    if config.robot == 'ur10_cb3':
        from .ur_bagio import prepare_ur_conversion
        prepare_ur_conversion(config, root)
    else:
        from .bagio import prepare_iiwa_conversion
        prepare_iiwa_conversion(config, root)


# ---------------------------------------------------------------------------
# `convert` / `validate` / `report`
# ---------------------------------------------------------------------------


def stage_convert(config: LabConfig, config_path: str, paths: RunPaths, *, reference_contract: str | None,
                  reference_sha256: str | None = None) -> dict[str, Any]:
    """Run ``convert_cli`` as a subprocess with a cleaned environment (RR_04
    A-8: like ``plan``, it imports Pinocchio for the baselines/sign-convention
    gravity fit -- T1.0/T1.6's LD_LIBRARY_PATH conflict)."""
    _prepare_conversion(config, paths.root)
    if paths.standstill_bag_dir.is_dir():
        from .bagio import write_recorder_losses

        write_recorder_losses(config, paths.standstill_bag_dir,
                              paths.raw_log_dir / "bag_standstill.rosbag2.stderr.log",
                              paths.root / "recorder_losses_standstill.json")
    command = [sys.executable, "-m", "erd_recording.convert_cli", "--config", config_path,
              "--run-dir", str(paths.root)]
    if reference_contract:
        command += ["--reference-contract", reference_contract]
    if reference_sha256:
        command += ["--reference-sha256", reference_sha256]
    result = subprocess.run(command, env=clean_ros_subprocess_env(), capture_output=True, text=True)
    if result.returncode != 0:
        raise PipelineError(f"convert subprocess failed:\nstdout={result.stdout}\nstderr={result.stderr}")
    converted = json.loads(result.stdout.strip().splitlines()[-1])
    status = converted.get("status", "converted")
    _write_manifest_status(paths, status)
    return converted


def stage_validate(paths: RunPaths) -> dict[str, Any]:
    if paths.validation.is_file():
        return json.loads(paths.validation.read_text(encoding="utf-8"))
    return {"ok": None, "note": "validate runs inside convert_cli (RR_04 A-8); no validation.json was produced"}


def stage_report(paths: RunPaths, bundle: PlanBundle | None, run_result: dict[str, Any] | None,
                 validation: dict[str, Any] | None, *, controller_name: str, robot: str,
                 position_count_rad: float = 0.0) -> dict[str, Any]:
    """``report.md``, plus RR_12 A-2b: the monitor's sample-age statistics per
    stage and the cancel latency (first over-threshold sample -> ``cancel``
    event) of every abort in the run's bags, also merged into
    ``monitor.json`` under ``cancel_latency``."""
    from .reporting import cancel_latencies, judge_monitor_age, latency_summary, tracking_suggestion

    if bundle is None and paths.plan_dir.is_dir():
        bundle = load_plan(paths.plan_dir)
    if validation is None and paths.validation.is_file():
        validation = json.loads(paths.validation.read_text(encoding="utf-8"))
    lines = ["# Run report", "", f"Run folder: `{paths.root.name}`", ""]
    if bundle is not None:
        lines += ["## Plan", "", "| segment | kind | samples | duration (s) |", "|---|---|---|---|"]
        for segment in bundle.segments:
            lines.append(f"| {segment.segment_id} | {segment.kind} | {len(segment.trajectory.time)} | "
                        f"{segment.trajectory.duration:.2f} |")
        lines.append("")
    if run_result is not None:
        lines += ["## Run", "", f"ok: {run_result.get('ok')}", ""]

    latencies: dict[str, Any] = {}
    for name, bag in (("standstill", paths.standstill_bag_dir), ("identify", paths.root / "identification" / "bag"),
                      ("run", paths.bag_dir)):
        if bag.is_dir():
            rows = cancel_latencies(bag, controller_name=controller_name)
            if rows:
                latencies[name] = {"cancels": rows, "summary": latency_summary(rows)}
    monitor = {}
    for path in (paths.root / "monitor.json", paths.root / "identification" / "monitor.json"):
        if path.is_file():
            monitor.update(json.loads(path.read_text(encoding="utf-8")))
    monitor_bounds: dict[str, Any] = {}
    if monitor or latencies:
        lines += ["## Monitor (RR_12 A-2b; rung-3 bound RR_18 R-6)", "",
                  "| stage | evaluations | age p50 (ms) | age p99 (ms) | age max (ms) | cancels | rung-3 bound | pass |",
                  "|---|---|---|---|---|---|---|---|"]
        for stage, entry in monitor.items():
            if not isinstance(entry, dict) or "evaluations" not in entry:
                continue
            age = entry.get("sample_age_ms", {})
            verdict = monitor_bounds[stage] = judge_monitor_age(entry, robot)
            bound = f"p99 <= {verdict['p99_ms']:g} ms" + (
                f", max < {verdict['max_below_ms']:g} ms" if verdict["max_below_ms"] is not None else "")
            lines.append(f"| {stage} | {entry['evaluations']} | {age.get('p50', float('nan')):.3f} | "
                         f"{age.get('p99', float('nan')):.3f} | {age.get('max', float('nan')):.3f} | "
                         f"{len(entry.get('cancels', []))} | {bound} | {'pass' if verdict['pass'] else 'FAIL'} |")
            print(f"report: monitor {stage}: age p99 {age.get('p99', float('nan')):.2f} ms, max "
                  f"{age.get('max', float('nan')):.2f} ms against {bound}: {'pass' if verdict['pass'] else 'FAIL'}")
        lines.append("")
        for stage, entry in latencies.items():
            lines += [f"Cancel latency, {stage}:", "", "| segment | condition | latency (ms) | reason |", "|---|---|---|---|"]
            for row in entry["cancels"]:
                latency = "n/a" if row["latency_ms"] is None else f"{row['latency_ms']:.2f}"
                lines.append(f"| {row['segment']} | {row['condition']} | {latency} | {row['reason'][:80]} |")
            lines.append("")
        if latencies:
            target = paths.root / "monitor.json"
            data = json.loads(target.read_text(encoding="utf-8")) if target.is_file() else {}
            data["cancel_latency"] = latencies
            target.write_text(json.dumps(data, indent=2), encoding="utf-8")
    suggestion = None
    if paths.bag_dir.is_dir():
        suggestion = tracking_suggestion(paths.bag_dir, controller_name=controller_name,
                                         count_rad=position_count_rad)
    if suggestion is not None:
        lines += ["## Rung-4a abort threshold (RR_14 P-3)", "",
                  f"Largest tracking error over the excitation segments: {suggestion['max_abs_error_rad']:.4g} rad "
                  f"(`{suggestion['segment']}`).", "",
                  f"Suggested `--abort-tracking-rad {suggestion['suggested_rad']:.3g}` "
                  f"({suggestion['rule']}{'; floored' if suggestion['floored'] else ''}). "
                  "Meaningful only for a `--ladder 0.1` run.", ""]
        print(f"report: suggested --abort-tracking-rad {suggestion['suggested_rad']:.3g} "
              f"(max |e| {suggestion['max_abs_error_rad']:.4g} rad in {suggestion['segment']})")
    if validation is not None:
        lines += ["## Validation", "", f"status: **{validation.get('status')}**", ""]
        for check in validation.get("checks", []):
            lines.append(f"- **{check.get('check')}**: ok={check.get('ok')}"
                         + (f" ({check.get('status')})" if check.get("status") else "")
                         + (f" -- simulator limit: {check['simulator_limit']}" if check.get("simulator_limit") else ""))
        for warning in validation.get("warnings", []):
            lines.append(f"- warning **{warning['warning']}**: {warning.get('detail', '')}")
        lines.append("")
    evaluation = paths.root / "evaluation" / "report.md"
    if evaluation.is_file():
        lines += ["## Evaluation", "", evaluation.read_text(encoding="utf-8")]
    paths.report.write_text("\n".join(lines), encoding="utf-8")
    return {"cancel_latency": latencies, "monitor": monitor, "monitor_bounds": monitor_bounds,
            "tracking_suggestion": suggestion}


def stage_evaluate(config: LabConfig, paths: RunPaths, *, checkpoints: str | None) -> dict[str, Any]:
    """RR_12 C-4 (offline, not in ``all``): ``evaluate_on.py`` for every
    checkpoint of ``consumer.checkpoints`` (or ``--checkpoints``) on this
    run's dataset, in the consumer environment."""
    summary = checkpoints or config.consumer.checkpoints
    if not summary:
        raise PipelineError("evaluate: no --checkpoints and consumer.checkpoints is null")
    if not config.consumer.python:
        raise PipelineError("evaluate: consumer.python is null")
    command = [sys.executable, "-m", "erd_recording.evaluate_cli", "--run-dir", str(paths.root), "--robot", config.robot,
               "--checkpoints", str(Path(summary).expanduser()), "--consumer-repo", config.consumer.repo,
               "--consumer-python", config.consumer.python]
    result = subprocess.run(command, env=clean_ros_subprocess_env(), capture_output=True, text=True)
    if result.returncode != 0:
        raise PipelineError(f"evaluate failed:\nstdout={result.stdout}\nstderr={result.stderr}")
    return json.loads(result.stdout.strip().splitlines()[-1])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str]) -> Any:
    import argparse

    parser = argparse.ArgumentParser(prog="record_<robot>")
    parser.add_argument("--config", required=True)
    parser.add_argument("stage", choices=["plan", "show", "preflight", "standstill", "identify", "run",
                                          "convert", "validate", "report", "evaluate", "all"])
    parser.add_argument("--ladder", type=float, default=None)
    parser.add_argument("--only", default=None)
    parser.add_argument("--no-confirm", action="store_true")
    # RR_12 C-5: `--reuse-identification` is gone; identification stays in the
    # session that uses it (RR_01 S8.3 freshness rule). RR_14 P-5: so is the
    # never-implemented `--resume`.
    parser.add_argument("--reference-contract", default=None)
    parser.add_argument("--reference-sha256", default=None,
                        help="pin for --reference-contract (the config's pin only covers its own reference)")
    parser.add_argument("--accept-unverified", action="store_true")
    parser.add_argument("--n-candidates", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-dir", default=None,
                        help="offline stages only: an existing run folder, e.g. one copied from another machine")
    parser.add_argument("--abort-tracking-rad", type=float, default=None,
                        help="rung 4a only: lower the monitor's tracking threshold; stage `run`, --ladder <= 0.1, below the lab file's tracking_rad")
    parser.add_argument("--recorder-log-level", default=None, choices=["debug"],
                        help="DIAGNOSTIC ONLY: global rosbag2 log level debug instead of the production "
                             "rosbag2_recorder:=debug (~0.5 MB/s of rcl lines). Other levels are refused: they drop "
                             "the \"Subscribed to topic\" lines the start wait needs (RR_18 R-5)")
    parser.add_argument("--checkpoints", default=None, help="evaluate: chain summary CSV (default consumer.checkpoints)")
    return parser.parse_args(argv)


_ROS_STAGES = frozenset({"preflight", "standstill", "identify", "run"})
#: RR_12 C-3: stages that may run on a copied run folder (`--run-dir`).
_OFFLINE_STAGES = frozenset({"convert", "validate", "report", "evaluate"})


def _run_all(robot: str, argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    config = load_lab_config(args.config)
    if config.robot != robot:
        raise PipelineError(f"{args.config} is a {config.robot!r} config, not {robot!r}")
    stages = ["plan", "preflight", "standstill", "identify", "run", "convert", "validate", "report"] \
        if args.stage == "all" else [args.stage]
    # RR_10 item 6: only stages that join the ROS graph need the shell's
    # domain to match; an offline convert/validate re-run does not. Checked
    # before the run directory exists, so a refusal leaves nothing behind.
    if any(stage in _ROS_STAGES for stage in stages):
        check_environment_domain(config)
    # RR_16 Q-4(i), RR_18 R-4: the digest is only meaningful if the
    # installed workspace was built from it; a missing or stale build stamp,
    # or an installed module that differs from src/ros2, refuses.
    from .code_digest import code_digest, refusal

    digest = code_digest()
    if any(stage in _ROS_STAGES for stage in stages) and refusal(digest):
        raise PipelineError(refusal(digest))
    try:
        check_abort_override(args.abort_tracking_rad, args.ladder, file_rad=config.limits.abort.tracking_rad,
                             stage=args.stage)
    except ValueError as exc:
        raise PipelineError(str(exc)) from None
    profile = profile_for(config.robot)
    if args.run_dir:
        if not set(stages) <= _OFFLINE_STAGES:
            raise PipelineError(f"--run-dir is for the offline stages {sorted(_OFFLINE_STAGES)} only")
        paths = RunPaths(Path(args.run_dir).expanduser().resolve())
        if not paths.plan_dir.is_dir():
            raise PipelineError(f"--run-dir {paths.root}: not a run folder (no plan/)")
    else:
        paths = make_run_dir(config, run_id=args.run_id)
    sim_to_driver = config.description.sim_to_driver()
    if not args.run_dir:
        # RR_12 A-3c/A-4: what was live for this session, in the manifest.
        _write_manifest_fields(paths, hardware=config.hardware, safety_vendor_checksum=config.safety.vendor_checksum,
                               code_digest=digest,
                               recorder_log_level=args.recorder_log_level or PRODUCTION_LOG_LEVEL,
                               **({"abort_tracking_rad_override": args.abort_tracking_rad, "ladder": args.ladder}
                                  if args.abort_tracking_rad is not None else {}))

    bundle = None
    rclpy_started = False
    node: RecordingNode | None = None
    run_result = None
    validation = None

    # RR_19: a signal before the node exists (during `plan`) was dropped, and
    # `all` went on into preflight and motion; it is now remembered and stops
    # the run before the next stage.
    signalled = {"abort": False}

    def _handle_signal(signum, frame) -> None:  # noqa: ANN001
        signalled["abort"] = True
        if node is not None:
            node.abort_requested = True

    previous_sigint = signal.signal(signal.SIGINT, _handle_signal)
    previous_sigterm = signal.signal(signal.SIGTERM, _handle_signal)
    only = frozenset(s.strip() for s in args.only.split(",")) if args.only else None
    try:
        for stage in stages:
            if stage == "plan":
                bundle = stage_plan(config, args.config, paths, n_candidates=args.n_candidates,
                                    ladder_scale=args.ladder)
                continue
            if not rclpy_started and stage in _ROS_STAGES:
                # RR_06 P3-1 item 9, live finding: rclpy.init()'s own
                # default signal handling (SignalHandlerOptions.ALL) installs
                # its own SIGINT/SIGTERM handling that calls rclpy.shutdown()
                # directly -- underneath, not instead of, the Python-level
                # handler just installed above (signal.getsignal() still
                # reports ours, but the process was observed live tearing
                # the ROS context down immediately on SIGINT, skipping
                # _cancel_and_confirm/_write_manifest_status entirely, then
                # crashing in this function's own `finally` with "rcl_shutdown
                # already called"). Disabling it is what makes our own
                # abort_requested-polling handler (RR_04 A-1's intended
                # cancel/confirm/stop-stage path) the only thing that reacts.
                from rclpy.signals import SignalHandlerOptions

                rclpy.init(args=None, signal_handler_options=SignalHandlerOptions.NO)
                rclpy_started = True
                node = RecordingNode(profile, config)
                node.recorder_log_level = args.recorder_log_level
                if args.abort_tracking_rad is not None:
                    node.excitation_tracking_rad = args.abort_tracking_rad
            if node is not None and (node.abort_requested or signalled["abort"]):
                cancel_and_stop(node, paths, reason="SIGINT/SIGTERM received")
                return 130
            if signalled["abort"]:
                _write_manifest_status(paths, "failed", reason=f"SIGINT/SIGTERM received before {stage}")
                return 130
            if stage == "preflight":
                report = stage_preflight(node, config, paths)
                if not report["ok"]:
                    raise PipelineError(f"preflight failed: {report}")
            elif stage == "standstill":
                if bundle is None:
                    bundle = load_plan(paths.plan_dir)
                standstill_result = stage_standstill(node, config, paths, bundle, sim_to_driver,
                                                     profile.driver_joint_order)
                # RR_06 P3-1 item 2, live finding: unlike preflight/run,
                # `all` never checked this stage's own result -- a failed
                # standstill (goal aborted, stop not confirmed, ...) was
                # silently swallowed and the pipeline carried on straight
                # into `run` regardless.
                if not standstill_result["ok"]:
                    raise PipelineError(f"standstill failed: {standstill_result}")
            elif stage == "identify":
                if bundle is None:
                    bundle = load_plan(paths.plan_dir)
                stage_identify(node, config, args.config, paths, bundle, sim_to_driver,
                               profile.driver_joint_order, no_confirm=args.no_confirm,
                               reference_contract=args.reference_contract, reference_sha256=args.reference_sha256)
            elif stage == "run":
                if bundle is None:
                    bundle = load_plan(paths.plan_dir)
                run_result = stage_run(node, config, paths, bundle, sim_to_driver, profile.driver_joint_order,
                                       only=only, ladder_scale=args.ladder, no_confirm=args.no_confirm)
                if not run_result["ok"]:
                    raise PipelineError(f"run failed: {run_result}")
            elif stage == "convert":
                stage_convert(config, args.config, paths, reference_contract=args.reference_contract,
                              reference_sha256=args.reference_sha256)
            elif stage == "validate":
                validation = stage_validate(paths)
            elif stage == "report":
                stage_report(paths, bundle, run_result, validation, controller_name=profile.controller_name,
                             robot=config.robot, position_count_rad=profile.position_count_rad)
            elif stage == "evaluate":
                print(json.dumps(stage_evaluate(config, paths, checkpoints=args.checkpoints)))
    except Exception as exc:
        _write_manifest_status(paths, "failed", error=str(exc))
        raise
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
        if node is not None:
            node.destroy_node()
        if rclpy_started:
            rclpy.shutdown()
    return 0


def main_iiwa(argv: list[str] | None = None) -> int:
    return _run_all("kuka_lbr_iiwa_14_r820", argv)


def main_ur10(argv: list[str] | None = None) -> int:
    return _run_all("ur10_cb3", argv)
