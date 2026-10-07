"""Lab configuration schema and loader (RR_01 S4, T1.2).

One YAML per installation (``erd.lab/1``) carries every lab-specific fact:
network addresses, the description/joint mapping, real safety caps, the
scene, poses, excitation shape and the consumer hookup. Nothing here talks to
ROS; this module is imported by the offline ``plan`` stage and by every ROS
stage that needs the frozen config.

Refusal rules (RR_01 S4):

* ``hardware: real`` is refused while any ``reviewed:`` flag is false or any
  measured value the config declares is ``null``.
* Every scene object/frame name must be explicit (no defaults invented here).
* The probe's top frequency must fit the SG resolvability bound at the real
  rate (S-12): ``<= 0.09 * rate``.
* ``hardware: real`` is refused off ``ros_domain_id: 0``; any non-real
  hardware is refused off domain ``87`` (RR_01 S9 "Separation").
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from . import frames

SCHEMA = "erd.lab/1"

#: Real recording runs on this ROS domain; every mock/emulator/ursim run must
#: use the separated domain below (RR_01 S9).
REAL_ROS_DOMAIN_ID = 0
SIMULATED_ROS_DOMAIN_ID = 87

_HARDWARE_CHOICES = {
    "kuka_lbr_iiwa_14_r820": ("mock", "emulator", "real"),
    "ur10_cb3": ("mock", "ursim", "real"),
}

#: SG resolvability bound (RR_01 S4 "Probe rule"), the same fraction dataset_config
#: (elastic_sim) documents as `0.09 * rate` (the order-3 SG differentiator's floor).
PROBE_SG_BOUND_FRACTION = 0.09


class ConfigError(ValueError):
    """A lab configuration is invalid or (for `real`) not yet reviewed."""


def _require(mapping: Mapping[str, Any], key: str, source: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"{source}: missing required key {key!r}")
    return mapping[key]


#: ``$NAME`` / ``${NAME}`` left after expansion means the variable is unset.
_UNEXPANDED_VARIABLE = re.compile(r"\$(\{[^}]*\}|[A-Za-z_][A-Za-z0-9_]*)")


def _expand_path(value: Any, key: str, source: str) -> str | None:
    """Expand ``${VAR}`` and ``~`` in a machine-dependent path value.

    Lab configs name machine-dependent locations through the variables that
    ``workspace_setup.sh`` exports (``ERD_DATA_ROOT``, ``ERD_CONSUMER_REPO``,
    ``ERD_CONSUMER_PYTHON``), so one checked-in file works on every machine.
    ``LabConfig.raw`` keeps the unexpanded text, so plan/config digests do
    not depend on where the repository was cloned. An unset variable is an
    error, never an empty string.
    """
    if value is None:
        return None
    text = os.path.expanduser(os.path.expandvars(str(value)))
    unexpanded = _UNEXPANDED_VARIABLE.search(text)
    if unexpanded:
        raise ConfigError(
            f"{source}: {key} uses {unexpanded.group(0)}, which is not set in the environment "
            "(source workspace_setup.sh first)"
        )
    return text


def _require_mapping(mapping: Mapping[str, Any], key: str, source: str) -> dict[str, Any]:
    value = _require(mapping, key, source)
    if not isinstance(value, Mapping):
        raise ConfigError(f"{source}.{key} must be a mapping")
    return dict(value)


def _no_nulls(value: Any, path: str) -> list[str]:
    """Return dotted paths of every ``null`` leaf under ``value``."""
    problems: list[str] = []
    if value is None:
        return [path]
    if isinstance(value, Mapping):
        for key, item in value.items():
            problems.extend(_no_nulls(item, f"{path}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            problems.extend(_no_nulls(item, f"{path}[{index}]"))
    return problems


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IiwaConnection:
    robot_ip: str
    client_ip: str
    fri_port: int
    fri_send_period_ms: int
    sunrise_version: str
    ros_domain_id: int


@dataclass(frozen=True)
class Ur10Connection:
    robot_ip: str
    reverse_ip: str
    headless_mode: bool
    kinematics_params_file: str | None
    servoj_gain: float
    servoj_lookahead_time: float
    rtde_frequency: float
    software_version: str
    serial: str
    ros_domain_id: int


Connection = IiwaConnection | Ur10Connection


def _load_connection(robot: str, raw: Mapping[str, Any], source: str) -> Connection:
    block = _require_mapping(raw, "connection", source)
    domain = int(block.get("ros_domain_id", SIMULATED_ROS_DOMAIN_ID))
    if robot == "kuka_lbr_iiwa_14_r820":
        return IiwaConnection(
            robot_ip=str(_require(block, "robot_ip", f"{source}.connection")),
            client_ip=str(_require(block, "client_ip", f"{source}.connection")),
            fri_port=int(block.get("fri_port", 30200)),
            fri_send_period_ms=int(block.get("fri_send_period_ms", 1)),
            sunrise_version=str(_require(block, "sunrise_version", f"{source}.connection")),
            ros_domain_id=domain,
        )
    return Ur10Connection(
        robot_ip=str(_require(block, "robot_ip", f"{source}.connection")),
        reverse_ip=str(_require(block, "reverse_ip", f"{source}.connection")),
        headless_mode=bool(block.get("headless_mode", False)),
        kinematics_params_file=block.get("kinematics_params_file"),
        servoj_gain=float(_require(block, "servoj_gain", f"{source}.connection")),
        servoj_lookahead_time=float(_require(block, "servoj_lookahead_time", f"{source}.connection")),
        rtde_frequency=float(block.get("rtde_frequency", 125.0)),
        software_version=str(_require(block, "software_version", f"{source}.connection")),
        serial=str(_require(block, "serial", f"{source}.connection")),
        ros_domain_id=domain,
    )


# ---------------------------------------------------------------------------
# Description
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolSpec:
    mass: float
    extra_collision: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class DescriptionSpec:
    sim_asset: str
    joint_map: Mapping[str, str]   # driver joint name -> sim asset joint name
    base_frame: str
    tool: ToolSpec

    def sim_to_driver(self) -> dict[str, str]:
        return {sim: driver for driver, sim in self.joint_map.items()}


def _load_description(raw: Mapping[str, Any], source: str) -> DescriptionSpec:
    block = _require_mapping(raw, "description", source)
    joint_map = _require_mapping(block, "joint_map", f"{source}.description")
    if not joint_map:
        raise ConfigError(f"{source}.description.joint_map must not be empty")
    tool_block = block.get("tool", {}) or {}
    tool = ToolSpec(
        mass=float(tool_block.get("mass", 0.0)),
        extra_collision=tuple(tool_block.get("extra_collision", []) or []),
    )
    return DescriptionSpec(
        sim_asset=str(_require(block, "sim_asset", f"{source}.description")),
        joint_map={str(k): str(v) for k, v in joint_map.items()},
        base_frame=str(_require(block, "base_frame", f"{source}.description")),
        tool=tool,
    )


# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AbortLimits:
    tracking_rad: float
    torque_fraction: float
    stop_timeout_s: float


@dataclass(frozen=True)
class ApproachLimits:
    velocity: float
    acceleration: float


@dataclass(frozen=True)
class LimitsSpec:
    reviewed: bool
    position: Mapping[str, tuple[float, float]]
    velocity: Mapping[str, float]
    acceleration: Mapping[str, float]
    jerk: Mapping[str, float]
    effort: Mapping[str, float]
    approach: ApproachLimits
    abort: AbortLimits

    def as_arrays(self, joint_order: Sequence[str]) -> dict[str, np.ndarray]:
        return {
            "position_lower": np.asarray([self.position[j][0] for j in joint_order]),
            "position_upper": np.asarray([self.position[j][1] for j in joint_order]),
            "velocity": np.asarray([self.velocity[j] for j in joint_order]),
            "acceleration": np.asarray([self.acceleration[j] for j in joint_order]),
            "jerk": np.asarray([self.jerk[j] for j in joint_order]),
            "effort": np.asarray([self.effort[j] for j in joint_order]),
        }


#: Placeholder for an unfilled real-hardware value on a non-``real`` config.
#: The generic null scan in :func:`_check_real_preconditions` is what refuses
#: ``hardware: real`` on these (RR_01 S4); a non-real config may legitimately
#: not have measured them yet, so the per-joint parsers below accept the
#: placeholder instead of raising, and anything that actually plans with it
#: fails loudly on the resulting NaN.
_UNFILLED = float("nan")


def _per_joint_pairs(raw: Any, key: str, source: str) -> dict[str, tuple[float, float]]:
    block = raw.get(key)
    if not isinstance(block, Mapping) or not block:
        raise ConfigError(f"{source}.limits.{key} must be a non-empty mapping of joint -> [lo, hi]")
    result = {}
    for joint, pair in block.items():
        if pair is None:
            result[str(joint)] = (_UNFILLED, _UNFILLED)
            continue
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ConfigError(f"{source}.limits.{key}.{joint} must be [lo, hi]")
        lo, hi = float(pair[0]), float(pair[1])
        if lo >= hi:
            raise ConfigError(f"{source}.limits.{key}.{joint} = [{lo}, {hi}] must satisfy lo < hi")
        result[str(joint)] = (lo, hi)
    return result


def _per_joint_scalar(raw: Any, key: str, source: str) -> dict[str, float]:
    block = raw.get(key)
    if not isinstance(block, Mapping) or not block:
        raise ConfigError(f"{source}.limits.{key} must be a non-empty mapping of joint -> value")
    result = {}
    for joint, value in block.items():
        if value is None:
            result[str(joint)] = _UNFILLED
            continue
        value = float(value)
        if value <= 0.0:
            raise ConfigError(f"{source}.limits.{key}.{joint} must be positive")
        result[str(joint)] = value
    return result


def _load_limits(raw: Mapping[str, Any], source: str) -> LimitsSpec:
    block = _require_mapping(raw, "limits", source)
    approach = block.get("approach") or {}
    abort = block.get("abort") or {}
    return LimitsSpec(
        reviewed=bool(block.get("reviewed", False)),
        position=_per_joint_pairs(block, "position", source),
        velocity=_per_joint_scalar(block, "velocity", source),
        acceleration=_per_joint_scalar(block, "acceleration", source),
        jerk=_per_joint_scalar(block, "jerk", source),
        effort=_per_joint_scalar(block, "effort", source),
        approach=ApproachLimits(
            velocity=float(_require(approach, "velocity", f"{source}.limits.approach")),
            acceleration=float(_require(approach, "acceleration", f"{source}.limits.approach")),
        ),
        abort=AbortLimits(
            tracking_rad=float(_require(abort, "tracking_rad", f"{source}.limits.abort")),
            torque_fraction=float(_require(abort, "torque_fraction", f"{source}.limits.abort")),
            stop_timeout_s=float(_require(abort, "stop_timeout_s", f"{source}.limits.abort")),
        ),
    )


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

#: Semantic scene types this loader expands into boxes/cylinders (RR_01 S4).
SEMANTIC_TYPES = ("floor", "ceiling", "wall", "table")
PRIMITIVE_TYPES = ("box", "cylinder", "sphere", "mesh", "keep_out")
_DEFAULT_PLANE_SIZE = 10.0
#: A mesh whose bounding box exceeds this is almost certainly a unit typo
#: (m vs mm), so the loader refuses it (RR_01 S4 "Scene rules").
MESH_BBOX_TYPO_GUARD_M = 3.0


@dataclass(frozen=True)
class SceneBox:
    """An oriented box, in the scene frame, after expansion + clearance."""

    id: str
    size: tuple[float, float, float]        # full extents [m]
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]
    virtual: bool = False                    # keep_out zones are virtual (never physical)
    source_type: str = "box"


@dataclass(frozen=True)
class SceneCylinder:
    id: str
    height: float
    radius: float
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]


@dataclass(frozen=True)
class SceneSphere:
    id: str
    radius: float
    xyz: tuple[float, float, float]


@dataclass(frozen=True)
class SceneMesh:
    id: str
    file: str
    scale: tuple[float, float, float]
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]


SceneObject = SceneBox | SceneCylinder | SceneSphere | SceneMesh


@dataclass(frozen=True)
class SceneSpec:
    reviewed: bool
    frame: str
    clearance: float
    objects: tuple[SceneObject, ...]
    allowed_contacts: tuple[tuple[str, str], ...]


def _norm_axis(normal: Any) -> np.ndarray:
    table = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0), "-y": (0, -1, 0),
             "+z": (0, 0, 1), "-z": (0, 0, -1)}
    if isinstance(normal, str):
        if normal not in table:
            raise ConfigError(f"scene wall normal {normal!r} must be one of {sorted(table)} or a unit vector")
        return np.asarray(table[normal], dtype=float)
    vector = np.asarray(normal, dtype=float)
    if vector.shape != (3,):
        raise ConfigError("scene wall normal must be a 3-vector or an axis string")
    norm = np.linalg.norm(vector)
    if norm <= 0.0:
        raise ConfigError("scene wall normal must be non-zero")
    return vector / norm


def _expand_object(raw: Mapping[str, Any], clearance: float, source: str) -> SceneObject:
    obj_id = _require(raw, "id", source)
    kind = _require(raw, "type", source)
    inflate = float(clearance)

    if kind == "floor":
        z = float(_require(raw, "z", f"{source}[{obj_id}]"))
        size = raw.get("size", (_DEFAULT_PLANE_SIZE, _DEFAULT_PLANE_SIZE))
        thickness = 0.2 + 2.0 * inflate
        # The whole box lies on the far side (below) the declared surface.
        return SceneBox(obj_id, (float(size[0]), float(size[1]), thickness),
                         (0.0, 0.0, z - thickness / 2.0 - inflate), (0.0, 0.0, 0.0),
                         source_type="floor")
    if kind == "ceiling":
        z = float(_require(raw, "z", f"{source}[{obj_id}]"))
        size = raw.get("size", (_DEFAULT_PLANE_SIZE, _DEFAULT_PLANE_SIZE))
        thickness = 0.2 + 2.0 * inflate
        return SceneBox(obj_id, (float(size[0]), float(size[1]), thickness),
                         (0.0, 0.0, z + thickness / 2.0 + inflate), (0.0, 0.0, 0.0),
                         source_type="ceiling")
    if kind == "wall":
        point = np.asarray(_require(raw, "point", f"{source}[{obj_id}]"), dtype=float)
        normal = _norm_axis(_require(raw, "normal", f"{source}[{obj_id}]"))
        size = raw.get("size", (_DEFAULT_PLANE_SIZE, _DEFAULT_PLANE_SIZE))
        thickness = float(raw.get("thickness", 0.2)) + 2.0 * inflate
        centre = point - normal * (thickness / 2.0 + inflate)
        # Build a box whose local z axis is the wall normal, then read its
        # width/height off the plane spanned by the other two axes.
        z_axis = normal
        helper = np.array([1.0, 0.0, 0.0]) if abs(z_axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x_axis = np.cross(helper, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        rotation = np.column_stack([x_axis, y_axis, z_axis])
        rpy = _matrix_to_rpy(rotation)
        return SceneBox(obj_id, (float(size[0]), float(size[1]), thickness),
                         tuple(centre), rpy, source_type="wall")
    if kind == "table":
        top_z = float(_require(raw, "top_z", f"{source}[{obj_id}]"))
        centre_xy = _require(raw, "centre_xy", f"{source}[{obj_id}]")
        size_xy = _require(raw, "size_xy", f"{source}[{obj_id}]")
        thickness = float(raw.get("thickness", 0.05)) + 2.0 * inflate
        return SceneBox(
            obj_id, (float(size_xy[0]), float(size_xy[1]), thickness),
            (float(centre_xy[0]), float(centre_xy[1]), top_z - thickness / 2.0 + inflate),
            (0.0, 0.0, 0.0), source_type="table",
        )
    if kind == "box" or kind == "keep_out":
        size = _require(raw, "size", f"{source}[{obj_id}]")
        xyz = _require(raw, "xyz", f"{source}[{obj_id}]")
        rpy = raw.get("rpy", (0.0, 0.0, 0.0))
        expanded = tuple(float(s) + 2.0 * inflate for s in size)
        return SceneBox(obj_id, expanded, tuple(float(v) for v in xyz), tuple(float(v) for v in rpy),
                         virtual=(kind == "keep_out"), source_type=kind)
    if kind == "cylinder":
        height = float(_require(raw, "height", f"{source}[{obj_id}]")) + 2.0 * inflate
        radius = float(_require(raw, "radius", f"{source}[{obj_id}]")) + inflate
        xyz = _require(raw, "xyz", f"{source}[{obj_id}]")
        rpy = raw.get("rpy", (0.0, 0.0, 0.0))
        return SceneCylinder(obj_id, height, radius, tuple(float(v) for v in xyz), tuple(float(v) for v in rpy))
    if kind == "sphere":
        radius = float(_require(raw, "radius", f"{source}[{obj_id}]")) + inflate
        xyz = _require(raw, "xyz", f"{source}[{obj_id}]")
        return SceneSphere(obj_id, radius, tuple(float(v) for v in xyz))
    if kind == "mesh":
        file = str(_require(raw, "file", f"{source}[{obj_id}]"))
        scale = raw.get("scale", (1.0, 1.0, 1.0))
        xyz = _require(raw, "xyz", f"{source}[{obj_id}]")
        rpy = raw.get("rpy", (0.0, 0.0, 0.0))
        return SceneMesh(obj_id, file, tuple(float(v) for v in scale),
                          tuple(float(v) for v in xyz), tuple(float(v) for v in rpy))
    raise ConfigError(f"{source}[{obj_id}]: unknown scene object type {kind!r}")


def _matrix_to_rpy(rotation: np.ndarray) -> tuple[float, float, float]:
    """Extrinsic XYZ Euler angles from a rotation matrix (URDF/ROS convention)."""
    sy = -rotation[2, 0]
    sy = float(np.clip(sy, -1.0, 1.0))
    pitch = np.arcsin(sy)
    if abs(sy) < 0.999999:
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = np.arctan2(-rotation[1, 2], rotation[1, 1])
        yaw = 0.0
    return float(roll), float(pitch), float(yaw)


def _transform_scene_object(obj: SceneObject, robot: str, frame: str) -> SceneObject:
    """Convert ``obj`` (built in the declared ``scene.frame``) into the bare
    sim asset's own root frame (RR_04 A-6). A rigid transform of the fully
    expanded object is equivalent to transforming its declared inputs first
    (e.g. a wall's ``point``/``normal``) -- so this runs once, generically,
    after ``_expand_object``, regardless of the object's original type."""
    new_xyz = frames.transform_xyz(robot, frame, obj.xyz)
    if isinstance(obj, SceneSphere):
        return replace(obj, xyz=new_xyz)
    new_rpy = frames.transform_rpy(robot, frame, obj.rpy)
    return replace(obj, xyz=new_xyz, rpy=new_rpy)


def _load_scene(raw: Mapping[str, Any], source: str, robot: str) -> SceneSpec:
    block = _require_mapping(raw, "scene", source)
    clearance = float(block.get("clearance", 0.0))
    frame = str(_require(block, "frame", f"{source}.scene"))
    objects_raw = block.get("objects") or []
    if not isinstance(objects_raw, list):
        raise ConfigError(f"{source}.scene.objects must be a list")
    objects: list[SceneObject] = []
    seen_ids: set[str] = set()
    for entry in objects_raw:
        expanded = _expand_object(entry, clearance, f"{source}.scene.objects")
        if expanded.id in seen_ids:
            raise ConfigError(f"{source}.scene.objects has a duplicate id {expanded.id!r}")
        seen_ids.add(expanded.id)
        if isinstance(expanded, SceneMesh):
            _guard_mesh_bbox(expanded, source)
        try:
            expanded = _transform_scene_object(expanded, robot, frame)
        except frames.FrameError as exc:
            raise ConfigError(f"{source}.scene.frame: {exc}") from exc
        objects.append(expanded)
    allowed_raw = block.get("allowed_contacts") or []
    allowed = []
    for pair in allowed_raw:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ConfigError(f"{source}.scene.allowed_contacts entries must be [link, object]")
        allowed.append((str(pair[0]), str(pair[1])))
    return SceneSpec(
        reviewed=bool(block.get("reviewed", False)),
        frame=frame,
        clearance=clearance,
        objects=tuple(objects),
        allowed_contacts=tuple(allowed),
    )


def _guard_mesh_bbox(mesh: SceneMesh, source: str) -> None:
    """Refuse a mesh whose file declares a bounding box over the typo guard.

    Best-effort: if the mesh file is not present at config-load time (a
    relocated lab install), the check is skipped here and left to the
    ``plan`` stage, which resolves and loads the mesh through Coal anyway.
    """
    path = Path(mesh.file).expanduser()
    if not path.is_file():
        return
    try:
        import trimesh
    except ImportError:
        return
    try:
        loaded = trimesh.load(str(path), force="mesh")
        extent = np.asarray(loaded.extents, dtype=float) * np.asarray(mesh.scale, dtype=float)
    except Exception:
        return
    if float(np.max(extent)) > MESH_BBOX_TYPO_GUARD_M:
        raise ConfigError(
            f"{source}.scene.objects[{mesh.id}]: mesh bounding box {extent.tolist()} m exceeds "
            f"the {MESH_BBOX_TYPO_GUARD_M} m typo guard (check `scale` and units)"
        )


# ---------------------------------------------------------------------------
# Poses, excitation, identification, consumer, recording
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PosesSpec:
    home: tuple[float, ...]
    standstill: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class RegimeSampling:
    enabled: bool
    velocity_fraction: tuple[float, float]
    max_acceleration: tuple[float, float]


@dataclass(frozen=True)
class ProbeSpec:
    enabled: bool
    from_training_config: str | None
    acceleration_fraction: float
    budget: str
    #: RR_12 S4.4: the real probe's top frequency cap (iiwa 90 Hz, UR10 11 Hz),
    #: applied on top of the SG bound; ``None`` leaves only the SG bound.
    max_top_hz: float | None = None


@dataclass(frozen=True)
class ExcitationSpec:
    trajectories: int
    seed: int
    base_frequency: float
    n_periods: int
    n_harmonics: int
    velocity_fraction: float
    max_acceleration: float
    centre_jitter: float
    position_window: Mapping[str, tuple[float, float]]
    candidates: int
    regime: RegimeSampling
    probe: ProbeSpec
    settle_s: float
    amplitude_ladder: tuple[float, ...]


@dataclass(frozen=True)
class IdentificationSpec:
    sweep_speeds: tuple[float, ...]
    sweep_range: Mapping[str, tuple[float, float]]
    holdout: int


@dataclass(frozen=True)
class ConsumerSpec:
    repo: str
    python: str | None
    reference_contract: str | None
    checkpoints: str | None
    #: RR_12 C-1: sha256 of ``reference_contract``; ``convert`` refuses a
    #: reference whose bytes differ, and ``hardware: real`` refuses an unpinned one.
    reference_sha256: str | None = None


@dataclass(frozen=True)
class SafetySpec:
    #: RR_12 A-3c: the vendor safety-configuration checksum live on the
    #: controller (Sunrise / PolyScope), copied into every run's manifest.
    #: Required key; ``null`` is refused for ``hardware: real``.
    vendor_checksum: str | None


@dataclass(frozen=True)
class RecordingSpec:
    output_root: str
    standstill_s: float
    pre_roll_s: float
    post_roll_s: float
    max_lost_fraction: float


def _load_poses(raw: Mapping[str, Any], source: str) -> PosesSpec:
    block = _require_mapping(raw, "poses", source)
    home = _require(block, "home", f"{source}.poses")
    standstill = _require(block, "standstill", f"{source}.poses")
    if not isinstance(standstill, list) or len(standstill) < 1:
        raise ConfigError(f"{source}.poses.standstill must be a non-empty list of poses")
    return PosesSpec(
        home=tuple(float(v) for v in home),
        standstill=tuple(tuple(float(v) for v in pose) for pose in standstill),
    )


def _load_excitation(raw: Mapping[str, Any], source: str, rate_hz: float) -> ExcitationSpec:
    block = _require_mapping(raw, "excitation", source)
    regime_raw = block.get("regime") or {}
    probe_raw = block.get("probe") or {}
    position_window_raw = block.get("position_window") or {}
    position_window = {
        str(joint): (float(pair[0]), float(pair[1])) for joint, pair in position_window_raw.items()
    }
    max_top_hz = probe_raw.get("max_top_hz")
    probe = ProbeSpec(
        enabled=bool(probe_raw.get("enabled", False)),
        from_training_config=_expand_path(probe_raw.get("from_training_config"),
                                          "excitation.probe.from_training_config", source),
        acceleration_fraction=float(probe_raw.get("acceleration_fraction", 0.2)),
        budget=str(probe_raw.get("budget", "split")),
        max_top_hz=None if max_top_hz is None else float(max_top_hz),
    )
    if probe.budget not in ("split", "additive"):
        raise ConfigError(f"{source}.excitation.probe.budget must be 'split' or 'additive'")
    if probe.budget == "additive":
        raise ConfigError(
            f"{source}.excitation.probe.budget must not be 'additive' on real hardware "
            "(RR_01 S4: 'never additive on hardware' -- the real caps must bound main + probe together)"
        )
    spec = ExcitationSpec(
        trajectories=int(block.get("trajectories", 10)),
        seed=int(_require(block, "seed", f"{source}.excitation")),
        base_frequency=float(_require(block, "base_frequency", f"{source}.excitation")),
        n_periods=int(block.get("n_periods", 1)),
        n_harmonics=int(block.get("n_harmonics", 5)),
        velocity_fraction=float(_require(block, "velocity_fraction", f"{source}.excitation")),
        max_acceleration=float(_require(block, "max_acceleration", f"{source}.excitation")),
        centre_jitter=float(block.get("centre_jitter", 1.0)),
        position_window=position_window,
        candidates=int(block.get("candidates", 48)),
        regime=RegimeSampling(
            enabled=bool(regime_raw.get("enabled", False)),
            velocity_fraction=tuple(float(v) for v in regime_raw.get("velocity_fraction", (1.0, 1.0))),
            max_acceleration=tuple(float(v) for v in regime_raw.get("max_acceleration", (1.0, 1.0))),
        ),
        probe=probe,
        settle_s=float(block.get("settle_s", 2.0)),
        amplitude_ladder=tuple(float(v) for v in block.get("amplitude_ladder", (0.1, 0.25, 0.5, 1.0))),
    )
    if probe.enabled and probe.from_training_config:
        _check_probe_sg_bound(probe, rate_hz, source)
    return spec


def _check_probe_sg_bound(probe: ProbeSpec, rate_hz: float, source: str) -> None:
    """RR_01 S4 "Probe rule": the real probe top must fit ``0.09 * rate``.

    Resolving the actual comb needs the training config and
    ``elastic_sim.link_modes``; this is a cheap necessary check the loader
    can do on its own -- the full resolution happens in the ``plan`` stage,
    which raises the same way if the resolved comb still violates it.
    """
    bound = PROBE_SG_BOUND_FRACTION * rate_hz
    if probe.max_top_hz is not None:
        bound = min(bound, probe.max_top_hz)
    training_path = Path(probe.from_training_config).expanduser()
    if not training_path.is_file():
        return
    try:
        from elastic_sim.dataset_config import load_config
    except ImportError:
        return
    training = load_config(training_path)
    probe_harmonics = training.excitation.probe_harmonics
    if not probe_harmonics:
        return
    top_hz = float(probe_harmonics[-1]) * float(training.excitation.base_frequency)
    if top_hz > bound:
        raise ConfigError(
            f"{source}.excitation.probe: the training config's probe top {top_hz:g} Hz exceeds "
            f"the real bound {bound:g} Hz (0.09 x {rate_hz:g} Hz, or probe.max_top_hz); cap it or disable the probe"
        )


def _load_identification(raw: Mapping[str, Any], source: str) -> IdentificationSpec:
    block = raw.get("identification") or {}
    sweep_range_raw = block.get("sweep_range") or {}
    return IdentificationSpec(
        sweep_speeds=tuple(float(v) for v in block.get("sweep_speeds", ())),
        sweep_range={str(j): (float(p[0]), float(p[1])) for j, p in sweep_range_raw.items()},
        holdout=int(block.get("holdout", 2)),
    )


def _load_consumer(raw: Mapping[str, Any], source: str) -> ConsumerSpec:
    block = _require_mapping(raw, "consumer", source)
    return ConsumerSpec(
        repo=_expand_path(_require(block, "repo", f"{source}.consumer"), "consumer.repo", source),
        python=_expand_path(block.get("python"), "consumer.python", source),
        reference_contract=_expand_path(block.get("reference_contract"), "consumer.reference_contract", source),
        checkpoints=_expand_path(block.get("checkpoints"), "consumer.checkpoints", source),
        reference_sha256=None if block.get("reference_sha256") is None else str(block["reference_sha256"]).lower(),
    )


def _load_safety(raw: Mapping[str, Any], source: str) -> SafetySpec:
    block = _require_mapping(raw, "safety", source)
    value = _require(block, "vendor_checksum", f"{source}.safety")
    return SafetySpec(vendor_checksum=None if value is None else str(value))


def _load_recording(raw: Mapping[str, Any], source: str) -> RecordingSpec:
    block = _require_mapping(raw, "recording", source)
    return RecordingSpec(
        output_root=_expand_path(_require(block, "output_root", f"{source}.recording"),
                                 "recording.output_root", source),
        standstill_s=float(block.get("standstill_s", 30.0)),
        pre_roll_s=float(block.get("pre_roll_s", 2.0)),
        post_roll_s=float(block.get("post_roll_s", 2.0)),
        max_lost_fraction=float(block.get("max_lost_fraction", 0.001)),
    )


# ---------------------------------------------------------------------------
# LabConfig
# ---------------------------------------------------------------------------

#: Nominal sample rate per robot, used for the probe SG-bound check and the
#: excitation time step (RR_01 S2 table).
NOMINAL_RATE_HZ = {"kuka_lbr_iiwa_14_r820": 1000.0, "ur10_cb3": 125.0}


@dataclass(frozen=True)
class OperatorSpec:
    confirm_each_trajectory: bool


@dataclass(frozen=True)
class LabConfig:
    robot: str
    hardware: str
    operator: OperatorSpec
    connection: Connection
    description: DescriptionSpec
    limits: LimitsSpec
    scene: SceneSpec
    poses: PosesSpec
    excitation: ExcitationSpec
    identification: IdentificationSpec
    consumer: ConsumerSpec
    recording: RecordingSpec
    safety: SafetySpec
    source_path: Path
    raw: Mapping[str, Any]

    @property
    def rate_hz(self) -> float:
        return NOMINAL_RATE_HZ[self.robot]

    @property
    def joint_order(self) -> tuple[str, ...]:
        """Sim-asset joint order (the order every column and array follows)."""
        # joint_map is {driver_name: sim_name}; the sim order is the config's
        # own declaration order, which the loader preserves from the YAML.
        return tuple(self.description.joint_map.values())


def load_lab_config(path: str | Path) -> LabConfig:
    """Load and validate one ``erd.lab/1`` YAML file.

    Raises :class:`ConfigError` for any refusal rule, including every
    ``hardware: real`` precondition. Loading a non-real config never fails on
    unreviewed flags or ``null`` measured values -- L0/L1/L2 runs are exactly
    how those values get filled in before commissioning.
    """
    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise ConfigError(f"lab config not found: {source_path}")
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    source = str(source_path)
    schema = raw.get("schema")
    if schema != SCHEMA:
        raise ConfigError(f"{source}: schema must be {SCHEMA!r}, got {schema!r}")
    robot = str(_require(raw, "robot", source))
    if robot not in _HARDWARE_CHOICES:
        raise ConfigError(f"{source}: robot must be one of {sorted(_HARDWARE_CHOICES)}, got {robot!r}")
    hardware = str(_require(raw, "hardware", source))
    if hardware not in _HARDWARE_CHOICES[robot]:
        raise ConfigError(
            f"{source}: hardware must be one of {_HARDWARE_CHOICES[robot]} for {robot}, got {hardware!r}"
        )
    operator_raw = raw.get("operator") or {}
    operator = OperatorSpec(confirm_each_trajectory=bool(operator_raw.get("confirm_each_trajectory", True)))

    connection = _load_connection(robot, raw, source)
    description = _load_description(raw, source)
    limits = _load_limits(raw, source)
    scene = _load_scene(raw, source, robot)
    poses = _load_poses(raw, source)
    excitation = _load_excitation(raw, source, NOMINAL_RATE_HZ[robot])
    identification = _load_identification(raw, source)
    consumer = _load_consumer(raw, source)
    recording = _load_recording(raw, source)
    safety = _load_safety(raw, source)

    config = LabConfig(
        robot=robot, hardware=hardware, operator=operator, connection=connection,
        description=description, limits=limits, scene=scene, poses=poses,
        excitation=excitation, identification=identification, consumer=consumer,
        recording=recording, safety=safety, source_path=source_path, raw=raw,
    )

    _check_domain_separation(config, source)
    if hardware == "real":
        _check_real_preconditions(config, raw, source)
    return config


def _check_domain_separation(config: LabConfig, source: str) -> None:
    domain = config.connection.ros_domain_id
    if config.hardware == "real" and domain == SIMULATED_ROS_DOMAIN_ID:
        raise ConfigError(
            f"{source}: hardware: real must not run on ROS_DOMAIN_ID {SIMULATED_ROS_DOMAIN_ID} "
            "(reserved for mock/emulator/ursim, RR_01 S9)"
        )
    if config.hardware != "real" and domain != SIMULATED_ROS_DOMAIN_ID:
        raise ConfigError(
            f"{source}: hardware: {config.hardware} must run on ROS_DOMAIN_ID {SIMULATED_ROS_DOMAIN_ID}, "
            f"got {domain} (RR_01 S9 separation)"
        )


def check_environment_domain(config: LabConfig, environ: Mapping[str, str] | None = None) -> None:
    """Refuse a ROS-talking ``record_*`` stage whose shell's ``ROS_DOMAIN_ID``
    differs from ``connection.ros_domain_id`` (RR_10 item 6).

    The config-level rule above only checks the YAML against its own
    ``hardware``; this one catches the shell: a terminal sourced with
    ``workspace_setup.sh --real`` (domain 0) driving ``iiwa_sim.yaml`` would
    otherwise reach a real robot's graph. An unset variable is domain 0, as
    in ROS itself.
    """
    env = os.environ if environ is None else environ
    text = env.get("ROS_DOMAIN_ID", "").strip()
    try:
        actual = int(text) if text else REAL_ROS_DOMAIN_ID
    except ValueError:
        raise ConfigError(f"ROS_DOMAIN_ID={text!r} is not an integer") from None
    expected = config.connection.ros_domain_id
    if actual != expected:
        fix = "source workspace_setup.sh --real" if expected == REAL_ROS_DOMAIN_ID \
            else "source workspace_setup.sh (without --real)"
        raise ConfigError(
            f"{config.source_path}: connection.ros_domain_id is {expected} but this shell has "
            f"ROS_DOMAIN_ID={actual}. Fix: {fix} in a fresh terminal (RR_01 S9 separation)"
        )


def _check_real_preconditions(config: LabConfig, raw: Mapping[str, Any], source: str) -> None:
    problems: list[str] = []
    if not config.limits.reviewed:
        problems.append("limits.reviewed is false")
    if not config.scene.reviewed:
        problems.append("scene.reviewed is false")
    probe = config.excitation.probe
    if probe.enabled and not probe.from_training_config:
        problems.append("excitation.probe.from_training_config is null while excitation.probe.enabled is true")
    null_paths = _no_nulls(raw, "")
    # Ignore keys that are legitimately optional/null (consumer hookup, the
    # UR kinematics file before calibration is asked for by T2.5 not T2.0, and
    # the probe's training config when the probe is off -- checked above).
    ignorable = {".consumer.python", ".consumer.reference_contract", ".consumer.checkpoints",
                 ".consumer.reference_sha256", ".connection.kinematics_params_file",
                 ".excitation.probe.from_training_config", ".excitation.probe.max_top_hz"}
    null_paths = [p for p in null_paths if p not in ignorable]
    if config.consumer.reference_contract and not config.consumer.reference_sha256:
        problems.append("consumer.reference_sha256 must pin consumer.reference_contract (RR_12 C-1)")
    if null_paths:
        problems.append(f"null value(s) at: {', '.join(sorted(null_paths))}")
    if problems:
        raise ConfigError(
            f"{source}: hardware: real refused, not reviewed:\n  - " + "\n  - ".join(problems)
        )


def joint_bounds_arrays(config: LabConfig) -> dict[str, np.ndarray]:
    """``limits`` as arrays in the config's declared (sim-asset) joint order."""
    return config.limits.as_arrays(config.joint_order)
