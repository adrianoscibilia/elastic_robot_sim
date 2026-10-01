"""Offline planning: re-limited excitation, approaches/returns, collision
validation and the frozen plan bundle (RR_01 S3.1 `plan`, S1.5).

Nothing here touches ROS. ``build_plan`` is called by the ``plan`` CLI stage
and by tests directly.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np

from elastic_sim.assets import AssetSpec
from elastic_sim.excitation import FourierExcitationConfig, optimize_excitation
from elastic_sim.kinematics import PortableKinematics, kinematic_groups
from elastic_sim.materialized import MaterializedTrajectory

from .config import LabConfig, SceneBox, SceneCylinder, SceneMesh, SceneSphere


class PlanError(ValueError):
    """A plan cannot be built or a frozen plan failed validation."""


# ---------------------------------------------------------------------------
# WorldKinematics: PortableKinematics + lab scene objects
# ---------------------------------------------------------------------------


class WorldKinematics(PortableKinematics):
    """A repository asset's kinematics, plus the lab's scene as Coal geometry.

    Every scene object from :class:`~erd_recording.config.SceneSpec` is added
    to the asset's collision model on the universe joint (the scene frame is
    the asset's own base frame -- the config loader is responsible for
    converting any other declared frame into it, RR_01 S4 "Frames"). A
    collision pair is then added between every robot-link geometry and every
    scene object, minus ``allowed_contacts``. ``keep_out`` zones (virtual)
    are treated exactly like physical objects for path validation and are
    only distinguished for display.
    """

    def __init__(self, asset: AssetSpec, scene_objects: tuple, allowed_contacts: tuple[tuple[str, str], ...] = ()):
        super().__init__(asset)
        if self._direct or self.model is None:
            raise PlanError("WorldKinematics requires a Cartesian (Pinocchio) asset, not a direct/translation one")
        self._scene_object_ids: list[str] = []
        self._add_scene_objects(scene_objects, allowed_contacts)

    def _add_scene_objects(self, scene_objects: tuple, allowed_contacts: tuple[tuple[str, str], ...]) -> None:
        import coal
        pin = self.pin

        universe_frame = 0  # pin.WORLD/model.frames[0] is the universe frame
        robot_geometry_ids = list(range(self.collision_model.ngeoms))
        placement_identity = pin.SE3.Identity()

        for obj in scene_objects:
            shape, local_placement = _coal_shape(obj, coal)
            if shape is None:
                continue  # mesh without a resolvable file: skipped, plan() will have already refused it
            world_placement = placement_identity * local_placement
            geom_object = pin.GeometryObject(f"scene::{obj.id}", 0, universe_frame, world_placement, shape)
            geom_object.meshColor = np.array([0.8, 0.2, 0.2, 0.35])
            new_id = self.collision_model.addGeometryObject(geom_object)
            self._scene_object_ids.append(obj.id)
            for robot_id in robot_geometry_ids:
                robot_go = self.collision_model.geometryObjects[robot_id]
                link_name = self.model.frames[robot_go.parentFrame].name
                if (link_name, obj.id) in allowed_contacts or (obj.id, link_name) in allowed_contacts:
                    continue
                self.collision_model.addCollisionPair(pin.CollisionPair(robot_id, new_id))
        self.collision_data = pin.GeometryData(self.collision_model)


def _rpy_to_matrix(rpy: tuple[float, float, float]) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return rz @ ry @ rx


def _coal_shape(obj: Any, coal: Any):
    pin_se3 = __import__("pinocchio").SE3
    if isinstance(obj, SceneBox):
        shape = coal.Box(*obj.size)
        placement = pin_se3(_rpy_to_matrix(obj.rpy), np.asarray(obj.xyz))
        return shape, placement
    if isinstance(obj, SceneCylinder):
        shape = coal.Cylinder(obj.radius, obj.height)
        placement = pin_se3(_rpy_to_matrix(obj.rpy), np.asarray(obj.xyz))
        return shape, placement
    if isinstance(obj, SceneSphere):
        shape = coal.Sphere(obj.radius)
        placement = pin_se3(np.eye(3), np.asarray(obj.xyz))
        return shape, placement
    if isinstance(obj, SceneMesh):
        path = Path(obj.file).expanduser()
        if not path.is_file():
            return None, None
        loader = coal.MeshLoader()
        mesh = loader.load(str(path), np.asarray(obj.scale, dtype=float))
        placement = pin_se3(_rpy_to_matrix(obj.rpy), np.asarray(obj.xyz))
        return mesh, placement
    raise PlanError(f"unknown scene object type for Coal conversion: {type(obj)!r}")


# ---------------------------------------------------------------------------
# URDF re-limiting: bake the lab's real caps into a snapshot of the bare asset
# ---------------------------------------------------------------------------


def relimited_asset(asset: AssetSpec, config: LabConfig, output_dir: Path) -> AssetSpec:
    """Write a URDF snapshot with ``<limit velocity>``/``effort`` replaced by
    the lab's real caps, per active joint, and return an :class:`AssetSpec`
    pointing at it (RR_01 S7.1).

    Position limits are left as the bare asset's own (usually wider); the
    real position window is enforced separately via
    :class:`~elastic_sim.excitation.FourierExcitationConfig.position_window`,
    computed from ``config.limits.position`` (RR_01 S4: "windows from
    limits.position").
    """
    arrays = config.limits.as_arrays(config.joint_order)
    limits = dict(zip(config.joint_order, zip(arrays["velocity"], arrays["effort"])))
    tree = ET.parse(asset.urdf_path)
    root = tree.getroot()
    # The snapshot is written into the plan directory, not next to the
    # original URDF, so relative mesh references (the common case in this
    # repository's assets) must become absolute or Pinocchio resolves them
    # against the wrong directory. `package://` references are left as-is;
    # PortableKinematics does not resolve them either (repository assets
    # don't use that scheme).
    original_dir = asset.urdf_path.parent.resolve()
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        if filename and not filename.startswith("package://") and not Path(filename).is_absolute():
            mesh.set("filename", str((original_dir / filename).resolve()))
    touched = set()
    for joint_element in root.findall("joint"):
        name = joint_element.get("name")
        if name not in limits:
            continue
        velocity, effort = limits[name]
        limit_element = joint_element.find("limit")
        if limit_element is None:
            raise PlanError(f"relimited_asset: joint {name!r} has no <limit> element to rewrite")
        limit_element.set("velocity", f"{velocity:.12g}")
        limit_element.set("effort", f"{effort:.12g}")
        touched.add(name)
    # RR_08: attach the measured tool envelope to the planning snapshot,
    # so PortableKinematics includes it in robot/scene collision pairs.
    for index, shape in enumerate(config.description.tool.extra_collision):
        parent = root.find(f"link[@name='{shape['parent']}']")
        if parent is None or shape["shape"] != "cylinder":
            raise PlanError(f"unsupported tool collision or missing parent: {shape}")
        collision = ET.SubElement(parent, "collision", name=f"erd_tool_{index}")
        ET.SubElement(collision, "origin", xyz=" ".join(map(str, shape["xyz"])),
                      rpy=" ".join(map(str, shape.get("rpy", [0, 0, 0]))))
        geometry = ET.SubElement(collision, "geometry")
        ET.SubElement(geometry, "cylinder", radius=str(shape["radius"]), length=str(shape["height"]))
    missing = set(config.joint_order) - touched
    if missing:
        raise PlanError(f"relimited_asset: joints not found in URDF to re-limit: {sorted(missing)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = output_dir / f"{asset.name}.relimited.urdf"
    tree.write(snapshot_path, xml_declaration=True, encoding="utf-8")
    return AssetSpec(
        name=asset.name, urdf_path=snapshot_path, active_joints=asset.active_joints,
        base_position=asset.base_position, base_quaternion=asset.base_quaternion,
        gravity=asset.gravity, self_collisions=asset.self_collisions, metadata=copy.deepcopy(asset.metadata),
    )


def excitation_position_window(config: LabConfig) -> tuple[tuple[str, tuple[float, float]], ...]:
    """``limits.position`` intersected with the optional tighter
    ``excitation.position_window``, as the ``(name, (lo, hi))`` pairs
    :class:`FourierExcitationConfig.position_window` expects."""
    window = dict(config.excitation.position_window)
    result = []
    for joint in config.joint_order:
        lo, hi = config.limits.position[joint]
        if joint in window:
            wlo, whi = window[joint]
            lo, hi = max(lo, wlo), min(hi, whi)
            if lo >= hi:
                raise PlanError(f"excitation.position_window[{joint!r}] does not intersect limits.position")
        result.append((joint, (lo, hi)))
    return tuple(result)


def resolve_probe_harmonics(config: LabConfig, rate_hz: float) -> tuple[int, ...]:
    """Resolve the real probe comb from the training config, capped at the
    SG resolvability bound (RR_01 S4 "Probe rule", S1.4)."""
    if not config.excitation.probe.enabled or not config.excitation.probe.from_training_config:
        return ()
    from elastic_sim.dataset_config import load_config
    from elastic_sim.link_modes import resolve_probe_design

    training = load_config(config.excitation.probe.from_training_config)
    resolved = resolve_probe_design(training)
    harmonics = resolved.excitation.probe_harmonics
    bound_hz = 0.09 * rate_hz
    base = resolved.excitation.base_frequency
    capped = tuple(int(h) for h in harmonics if h * base <= bound_hz)
    return capped


def regime_draw(config: LabConfig, index: int) -> tuple[float, float]:
    """Per-trajectory ``(velocity_fraction, max_acceleration)`` draw inside
    the config's regime bounds (RR_01 S1.4, mirrors
    ``elastic_sim.dataset_config.RegimeSampling``)."""
    regime = config.excitation.regime
    if not regime.enabled:
        return config.excitation.velocity_fraction, config.excitation.max_acceleration
    rng = np.random.default_rng(config.excitation.seed + 1_000_003 * index)
    vf_lo, vf_hi = regime.velocity_fraction
    acc_lo, acc_hi = regime.max_acceleration
    velocity_fraction = config.excitation.velocity_fraction * rng.uniform(vf_lo, vf_hi)
    max_acceleration = config.excitation.max_acceleration * rng.uniform(acc_lo, acc_hi)
    return float(velocity_fraction), float(max_acceleration)


# ---------------------------------------------------------------------------
# Quintic approach/return segments
# ---------------------------------------------------------------------------


def quintic_segment(
    start: np.ndarray, end: np.ndarray, *, velocity_limit: float, acceleration_limit: float, time_step: float,
) -> MaterializedTrajectory:
    """A 5th-order polynomial joint move, zero velocity/acceleration at both
    ends, timed to the slowest joint under ``velocity_limit``/``acceleration_limit``.
    """
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    delta = end - start
    span = float(np.max(np.abs(delta)))
    if span == 0.0:
        duration = time_step * 2
    else:
        # Peak velocity of a zero-boundary quintic is 15/8 * span / T; peak
        # acceleration is 10*sqrt(3)/3 * span / T^2 (Beazley/standard quintic
        # coefficients below). Solve each for the binding T and take the max.
        t_velocity = 15.0 / 8.0 * span / max(velocity_limit, 1e-9)
        t_acceleration = (10.0 * np.sqrt(3.0) / 3.0 * span / max(acceleration_limit, 1e-9)) ** 0.5
        duration = max(t_velocity, t_acceleration, time_step * 2)
    n_samples = max(2, int(np.ceil(duration / time_step)) + 1)
    time = np.linspace(0.0, duration, n_samples)
    tau = time / duration if duration > 0 else time
    # Standard quintic with zero velocity/acceleration at tau=0,1:
    #   s(tau) = 10 tau^3 - 15 tau^4 + 6 tau^5
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    ds = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / duration
    dds = (60 * tau - 180 * tau**2 + 120 * tau**3) / duration**2
    position = start[None, :] + np.outer(s, delta)
    velocity = np.outer(ds, delta)
    acceleration = np.outer(dds, delta)
    return MaterializedTrajectory(
        time=time, position=position, velocity=velocity, acceleration=acceleration,
        joint_names=tuple(f"j{i}" for i in range(len(start))),
        metadata={"generator": "quintic_segment", "duration": float(duration)},
    )


# ---------------------------------------------------------------------------
# Plan bundle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlanSegment:
    segment_id: str
    kind: str            # approach | excitation | return | identify
    trajectory: MaterializedTrajectory
    digest: str


@dataclass(frozen=True)
class PlanBundle:
    config_digest: str
    asset: AssetSpec
    segments: tuple[PlanSegment, ...]
    rejected_for_collision: int
    #: RR_04 A-3: the standstill approach/return quintics, planned and
    #: collision-checked exactly like every excitation segment, kept separate
    #: from ``segments`` so ``run`` and ``standstill`` each only iterate their
    #: own kind. The hold itself is not a trajectory (RR_04 A-3: "a timed wait
    #: with the JTC holding"), so there is no ``standstill_hold_k`` segment --
    #: only ``standstill_approach_k``/``standstill_return_k``.
    standstill_segments: tuple[PlanSegment, ...] = ()

    def to_manifest(self) -> dict[str, Any]:
        return {
            "schema": "erd.plan/1",
            "config_digest": self.config_digest,
            "asset": self.asset.name,
            "joint_names": list(self.asset.joint_names),
            "segments": [
                {"segment_id": s.segment_id, "kind": s.kind, "digest": s.digest,
                 "n_samples": int(len(s.trajectory.time)), "duration": s.trajectory.duration}
                for s in self.segments
            ],
            "standstill_segments": [
                {"segment_id": s.segment_id, "kind": s.kind, "digest": s.digest,
                 "n_samples": int(len(s.trajectory.time)), "duration": s.trajectory.duration}
                for s in self.standstill_segments
            ],
        }

    def segment(self, segment_id: str) -> PlanSegment:
        for segment in self.segments + self.standstill_segments:
            if segment.segment_id == segment_id:
                return segment
        raise KeyError(segment_id)


def config_digest(config: LabConfig) -> str:
    payload = json.dumps(config.raw, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def check_caps_on_analytic(trajectory: MaterializedTrajectory, velocity_limit: np.ndarray,
                            acceleration_limit: np.ndarray, jerk_limit: np.ndarray | None = None,
                            *, oversample: int = 20) -> None:
    """Check velocity/acceleration/jerk caps on the analytic signal at
    ``oversample`` times the trajectory's own grid density (RR_01 T1.5:
    "caps hold on the analytic signal, not just the samples"). No-op if the
    trajectory carries no analytic evaluator (e.g. a quintic segment, whose
    sampled grid already is the exact closed form).

    ``jerk_limit`` (RR_04 B-5) is checked as a finite difference of the
    analytic acceleration at the same oversampled grid -- "a finite
    difference of ddq at 20x oversampling is fine" (RR_01 S5 `plan`); omit it
    to skip the jerk check (e.g. for segments with no declared jerk limit).
    """
    if trajectory.analytic is None:
        return
    step = float(trajectory.time[1] - trajectory.time[0]) / oversample
    fine_time = np.arange(trajectory.time[0], trajectory.time[-1] + 0.5 * step, step)
    values = [trajectory.analytic(t) for t in fine_time]
    velocity = np.asarray([v[1] for v in values])
    acceleration = np.asarray([v[2] for v in values])
    v_peak = np.max(np.abs(velocity), axis=0)
    a_peak = np.max(np.abs(acceleration), axis=0)
    if np.any(v_peak > velocity_limit * (1.0 + 1e-6)):
        raise PlanError(f"velocity cap exceeded on the analytic signal: peak {v_peak} vs limit {velocity_limit}")
    if np.any(a_peak > acceleration_limit * (1.0 + 1e-6)):
        raise PlanError(f"acceleration cap exceeded on the analytic signal: peak {a_peak} vs limit {acceleration_limit}")
    if jerk_limit is not None and len(fine_time) > 1:
        jerk = np.diff(acceleration, axis=0) / step
        j_peak = np.max(np.abs(jerk), axis=0)
        if np.any(j_peak > jerk_limit * (1.0 + 1e-6)):
            raise PlanError(f"jerk cap exceeded on the analytic signal: peak {j_peak} vs limit {jerk_limit}")


def ladder_scale_trajectory(trajectory: MaterializedTrajectory, scale: float) -> MaterializedTrajectory:
    """RR_06 P3-3b: ``q = c + s(q - c)`` around the trajectory's own centre
    ``c`` (its per-joint mean position) -- shrinks (or grows) the excitation
    about where it already spends its time, rather than about some other
    reference the commissioning ladder has no reason to care about.
    Velocity/acceleration scale by ``s`` too (the constant ``c`` drops out
    under differentiation, so this is exact, not an approximation). The
    closed-form analytic evaluator is dropped (the caller re-validates caps
    on the samples, which is enough here: for ``0 < s <= 1`` the velocity/
    acceleration/jerk peaks can only shrink, since they already scale
    linearly by ``s``, so the ladder step is never the one that could
    newly violate a cap the base trajectory already passed on the analytic
    signal)."""
    centre = trajectory.position.mean(axis=0)
    scaled_position = centre + scale * (trajectory.position - centre)
    scaled_velocity = scale * trajectory.velocity
    scaled_acceleration = None if trajectory.acceleration is None else scale * trajectory.acceleration
    return MaterializedTrajectory(
        time=trajectory.time, position=scaled_position, velocity=scaled_velocity,
        acceleration=scaled_acceleration, joint_names=trajectory.joint_names,
        metadata={**trajectory.metadata, "ladder_scale": float(scale)},
    )


def build_ladder_segments(
    config: LabConfig, world: WorldKinematics, excitation_segments: list[PlanSegment], scale: float,
    joint_names: tuple[str, ...], time_step: float, margin: float = 0.0,
) -> tuple[PlanSegment, ...]:
    """RR_06 P3-3b: one ``excitation_<k>_ladder`` (kind ``excitation_
    commissioning``) per base excitation segment, scaled by ``scale`` around
    its own centre, with a matching ``approach``/final ``return`` recomputed
    from scratch (the scaled trajectory starts at a different point than the
    original) and re-validated for collisions exactly like the original
    (RR_06: "re-validated"). The ``_commissioning`` kind suffix is what keeps
    these out of every dataset -- ``convert_cli``'s own segment filter
    already only looks for ``kind == "excitation"``, so a ladder segment is
    excluded by construction, not by an extra check."""
    arrays = config.limits.as_arrays(config.joint_order)
    segments: list[PlanSegment] = []
    previous_end = np.asarray(config.poses.home, dtype=float)
    for base in excitation_segments:
        scaled = ladder_scale_trajectory(base.trajectory, scale)
        check_caps_on_analytic(scaled, arrays["velocity"], arrays["acceleration"], arrays["jerk"])  # no-op: analytic is None
        report = world.validate_path(scaled.position, margin=margin, max_joint_step=0.01)
        if not report.valid:
            raise PlanError(
                f"{base.segment_id}_ladder (scale={scale}): collision at min distance "
                f"{report.minimum_distance:.4g} m between {report.closest_pair}"
            )
        approach = quintic_segment(previous_end, scaled.position[0], velocity_limit=config.limits.approach.velocity,
                                    acceleration_limit=config.limits.approach.acceleration, time_step=time_step)
        approach = _rename(approach, joint_names)
        approach_report = world.validate_path(approach.position, margin=margin, max_joint_step=0.01)
        if not approach_report.valid:
            raise PlanError(f"{base.segment_id}_ladder_approach: collision between {approach_report.closest_pair}")
        segments.append(PlanSegment(f"{base.segment_id}_ladder_approach", "approach_commissioning",
                                    approach, approach.digest()))
        segments.append(PlanSegment(f"{base.segment_id}_ladder", "excitation_commissioning", scaled, scaled.digest()))
        previous_end = scaled.position[-1]

    home = np.asarray(config.poses.home, dtype=float)
    ret = quintic_segment(previous_end, home, velocity_limit=config.limits.approach.velocity,
                          acceleration_limit=config.limits.approach.acceleration, time_step=time_step)
    ret = _rename(ret, joint_names)
    ret_report = world.validate_path(ret.position, margin=margin, max_joint_step=0.01)
    if not ret_report.valid:
        raise PlanError(f"return_0_ladder: collision between {ret_report.closest_pair}")
    segments.append(PlanSegment("return_0_ladder", "return_commissioning", ret, ret.digest()))
    return tuple(segments)


def build_standstill_segments(
    config: LabConfig, world: WorldKinematics, joint_names: tuple[str, ...], time_step: float,
    margin: float = 0.0,
) -> tuple[PlanSegment, ...]:
    """RR_04 A-3: one ``standstill_approach_k``/``standstill_return_k`` pair
    per ``poses.standstill`` entry, home -> pose_k -> home (never pose_k ->
    pose_k+1 directly), collision-checked exactly like an excitation
    approach/return. The hold between them is not a trajectory; ``pipeline``
    runs it as a timed wait with the JTC still holding the approach's last
    point. ``margin`` is the asset's own collision margin (RR_04 B-5)."""
    home = np.asarray(config.poses.home, dtype=float)
    segments: list[PlanSegment] = []
    for index, pose in enumerate(config.poses.standstill):
        target = np.asarray(pose, dtype=float)
        approach = quintic_segment(home, target, velocity_limit=config.limits.approach.velocity,
                                   acceleration_limit=config.limits.approach.acceleration, time_step=time_step)
        approach = _rename(approach, joint_names)
        approach_report = world.validate_path(approach.position, margin=margin, max_joint_step=0.01)
        if not approach_report.valid:
            raise PlanError(f"standstill_approach_{index}: collision between {approach_report.closest_pair}")
        segments.append(PlanSegment(f"standstill_approach_{index}", "standstill_approach", approach, approach.digest()))

        ret = quintic_segment(target, home, velocity_limit=config.limits.approach.velocity,
                              acceleration_limit=config.limits.approach.acceleration, time_step=time_step)
        ret = _rename(ret, joint_names)
        ret_report = world.validate_path(ret.position, margin=margin, max_joint_step=0.01)
        if not ret_report.valid:
            raise PlanError(f"standstill_return_{index}: collision between {ret_report.closest_pair}")
        segments.append(PlanSegment(f"standstill_return_{index}", "standstill_return", ret, ret.digest()))
    return tuple(segments)


def build_plan(
    config: LabConfig, output_dir: Path, *, n_candidates: int | None = None,
    driver_urdf_path: Path | None = None, driver_urdf_calibrated_path: Path | None = None,
    ladder_scale: float | None = None,
) -> PlanBundle:
    """Build, validate and freeze the plan bundle for one lab config.

    Raises :class:`PlanError` naming the joint/sample/object/link on any
    collision, cap or FK failure (RR_01 S5 `plan`).

    ``driver_urdf_path`` is the driver's own description, exported by the ROS
    side before this (subprocess) stage runs, since it needs `xacro`/
    `ament_index` (RR_04 B-2). When given, the FK-identity gate runs against
    the **nominal** driver description and *refuses the plan* on failure.
    ``driver_urdf_calibrated_path`` (UR10 only, when
    `connection.kinematics_params_file` is set) is compared against the
    nominal driver description and its deviation is written in mm, **not**
    gated (RR_01 S5: "the UR10 calibrated deviation is reported in mm and not
    gated").

    ``ladder_scale`` (RR_06 P3-3b), when given, additionally builds one
    ``excitation_<k>_ladder`` (kind ``excitation_commissioning``) per
    excitation segment, scaled around its own centre and re-validated --
    see :func:`build_ladder_segments`. These are appended, not substituted:
    the frozen bundle always carries both the full-scale and the ladder
    segments; ``--only``/``run`` decide which ones actually execute.
    """
    from elastic_sim.assets import AssetRegistry

    registry = AssetRegistry.for_repository()
    bare_asset = registry.load(config.description.sim_asset)
    output_dir.mkdir(parents=True, exist_ok=True)

    if driver_urdf_path is not None:
        nominal_driver = driver_gate_asset(driver_urdf_path, config)
        fk_report = fk_identity_check(bare_asset, nominal_driver, groups=(GATE_GROUP_NAME,))
        if driver_urdf_calibrated_path is not None:
            calibrated_driver = driver_gate_asset(driver_urdf_calibrated_path, config)
            calibrated_kin = PortableKinematics(calibrated_driver)
            nominal_kin = PortableKinematics(nominal_driver)
            neutral_q = nominal_kin.neutral()
            nominal_pose = nominal_kin.forward(neutral_q)[GATE_GROUP_NAME]
            calibrated_pose = calibrated_kin.forward(neutral_q)[GATE_GROUP_NAME]
            deviation_mm = float(np.linalg.norm(nominal_pose[:3] - calibrated_pose[:3])) * 1000.0
            fk_report["calibrated_deviation_mm"] = deviation_mm
            fk_report["calibrated_deviation_gated"] = False
        (output_dir / "fk_identity.json").write_text(json.dumps(fk_report, indent=2), encoding="utf-8")
        if not fk_report["ok"]:
            raise PlanError(f"FK identity gate refused the plan: {fk_report}")

    snapshot_asset = relimited_asset(bare_asset, config, output_dir)
    world = WorldKinematics(snapshot_asset, config.scene.objects, config.scene.allowed_contacts)
    # RR_04 B-5: every validate_path call in this function uses the asset's
    # own collision margin, never 0 -- the bare assets declare it in
    # `metadata["collision"]["margin"]` (assets/robots/*/asset.yaml).
    collision_margin = float(snapshot_asset.metadata.get("collision", {}).get("margin", 0.0))

    arrays = config.limits.as_arrays(config.joint_order)
    position_window = excitation_position_window(config)
    rate_hz = config.rate_hz
    time_step = 1.0 / rate_hz
    probe_harmonics = resolve_probe_harmonics(config, rate_hz)

    segments: list[PlanSegment] = []
    rejected_total = 0
    home = np.asarray(config.poses.home, dtype=float)
    previous_end = home

    for index in range(config.excitation.trajectories):
        velocity_fraction, max_acceleration = regime_draw(config, index)
        exc_config = FourierExcitationConfig(
            n_harmonics=config.excitation.n_harmonics,
            base_frequency=config.excitation.base_frequency,
            n_periods=config.excitation.n_periods,
            time_step=time_step,
            max_acceleration=max_acceleration,
            velocity_fraction=velocity_fraction,
            centre_jitter=config.excitation.centre_jitter,
            probe_harmonics=probe_harmonics,
            probe_acceleration_fraction=config.excitation.probe.acceleration_fraction,
            probe_budget=config.excitation.probe.budget,
            position_window=position_window,
        )
        trajectory = optimize_excitation(
            snapshot_asset, exc_config, seed=config.excitation.seed + index,
            n_candidates=n_candidates or config.excitation.candidates, kinematics=world,
        )
        rejected_total += int(trajectory.metadata.get("collision_rejections", 0))
        check_caps_on_analytic(trajectory, arrays["velocity"], arrays["acceleration"], arrays["jerk"])
        report = world.validate_path(trajectory.position, margin=collision_margin, max_joint_step=0.01)
        if not report.valid:
            raise PlanError(
                f"excitation_{index}: collision at min distance {report.minimum_distance:.4g} m "
                f"between {report.closest_pair}"
            )

        approach = quintic_segment(previous_end, trajectory.position[0], velocity_limit=config.limits.approach.velocity,
                                    acceleration_limit=config.limits.approach.acceleration, time_step=time_step)
        approach = _rename(approach, snapshot_asset.joint_names)
        approach_report = world.validate_path(approach.position, margin=collision_margin, max_joint_step=0.01)
        if not approach_report.valid:
            raise PlanError(f"approach_{index}: collision between {approach_report.closest_pair}")
        segments.append(PlanSegment(f"approach_{index}", "approach", approach, approach.digest()))
        segments.append(PlanSegment(f"excitation_{index}", "excitation", trajectory, trajectory.digest()))
        previous_end = trajectory.position[-1]

    ret = quintic_segment(previous_end, home, velocity_limit=config.limits.approach.velocity,
                          acceleration_limit=config.limits.approach.acceleration, time_step=time_step)
    ret = _rename(ret, snapshot_asset.joint_names)
    ret_report = world.validate_path(ret.position, margin=collision_margin, max_joint_step=0.01)
    if not ret_report.valid:
        raise PlanError(f"return: collision between {ret_report.closest_pair}")
    segments.append(PlanSegment("return_0", "return", ret, ret.digest()))

    if ladder_scale is not None:
        # RR_06 P3-3b: one commissioning-tagged, re-validated copy of every
        # excitation segment at this amplitude scale, appended into the same
        # bundle (never in place of the full-scale segments: `--only` is what
        # picks which ones a given `run` actually executes).
        excitation_segments = [s for s in segments if s.kind == "excitation"]
        segments += list(build_ladder_segments(
            config, world, excitation_segments, ladder_scale, snapshot_asset.joint_names, time_step,
            margin=collision_margin,
        ))

    standstill_segments = build_standstill_segments(config, world, snapshot_asset.joint_names, time_step,
                                                    margin=collision_margin)

    bundle = PlanBundle(config_digest=config_digest(config), asset=snapshot_asset,
                        segments=tuple(segments), rejected_for_collision=rejected_total,
                        standstill_segments=standstill_segments)
    _write_bundle(bundle, output_dir)
    if config.robot == "kuka_lbr_iiwa_14_r820":
        manifest_path = output_dir / "plan.manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["known_driver_description_deviations"] = {
            "source": "ICube iiwa_description", "axis_x_offsets_m": {"A2": 0.000436, "A4": -0.000436},
            "planning_geometry": "LBR-Stack bare asset; no correction applied",
            "tool": "Media Flange I-12 envelope; owner confirms dimensions at T2.0",
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))
    return bundle


def _rename(trajectory: MaterializedTrajectory, joint_names: tuple[str, ...]) -> MaterializedTrajectory:
    return MaterializedTrajectory(
        time=trajectory.time, position=trajectory.position, velocity=trajectory.velocity,
        acceleration=trajectory.acceleration, joint_names=joint_names, metadata=trajectory.metadata,
    )


def _write_bundle(bundle: PlanBundle, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for segment in bundle.segments + bundle.standstill_segments:
        segment.trajectory.save(output_dir / f"{segment.segment_id}.json")
    (output_dir / "plan.manifest.json").write_text(json.dumps(bundle.to_manifest(), indent=2), encoding="utf-8")


# RR_08: tool conventions are reported separately from the joint-axis gate.
DRIVER_TIP_LINK = {"kuka_lbr_iiwa_14_r820": "tool0", "ur10_cb3": "tool0"}
#: The bare simulation assets' single kinematic group name
#: (`assets/robots/*/asset.yaml`).
GATE_GROUP_NAME = "arm"


def driver_gate_asset(driver_urdf_path: Path, config: LabConfig) -> AssetSpec:
    """An :class:`AssetSpec` over the exported driver URDF, with
    ``active_joints`` reordered so position ``i`` is the driver joint mapped
    (via ``description.joint_map``) to ``config.joint_order[i]`` -- the
    positional correspondence :func:`fk_identity_check`'s ``forward()`` calls
    rely on (RR_04 B-2: "joint names mapped through joint_map"; neither
    ``PortableKinematics.forward`` nor ``_model_q`` look at a joint's *name*,
    only its position in ``asset.joint_names``, so the two assets' orders
    must be made to agree explicitly here rather than by naming coincidence).
    """
    sim_to_driver = config.description.sim_to_driver()
    reordered = tuple(sim_to_driver[sim] for sim in config.joint_order)
    tip_link = DRIVER_TIP_LINK.get(config.robot)
    if tip_link is None:
        raise PlanError(f"driver_gate_asset: no DRIVER_TIP_LINK entry for robot {config.robot!r}")
    return AssetSpec(
        name=f"{config.robot}_driver", urdf_path=Path(driver_urdf_path), active_joints=reordered,
        metadata={"kinematic_groups": {GATE_GROUP_NAME: {"joints": list(reordered), "tip_link": tip_link}}},
    )


def _urdf_axis_poses(asset: AssetSpec, q: np.ndarray):
    """Joint axis lines and link transforms in the URDF root (common base).

    Keep the signed direction: reversing a joint axis must fail even though
    the geometric, unoriented line would be the same.
    """
    from scipy.spatial.transform import Rotation

    root = ET.parse(asset.urdf_path).getroot()
    joints = root.findall("joint")
    children = {j.find("child").get("link") for j in joints}
    links = {link.get("name") for link in root.findall("link")}
    poses = {name: np.eye(4) for name in links - children}
    values = dict(zip(asset.joint_names, q))
    axes = {}
    pending = list(joints)
    while pending:
        advanced = False
        for joint in pending[:]:
            parent = joint.find("parent").get("link")
            if parent not in poses:
                continue
            origin = joint.find("origin")
            transform = np.eye(4)
            if origin is not None:
                transform[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
                transform[:3, :3] = _rpy_to_matrix(tuple(map(float, origin.get("rpy", "0 0 0").split())))
            transform = poses[parent] @ transform
            name = joint.get("name")
            if joint.get("type") in ("revolute", "continuous"):
                axis_element = joint.find("axis")
                axis = np.fromstring(axis_element.get("xyz", "1 0 0") if axis_element is not None else "1 0 0", sep=" ")
                axis /= np.linalg.norm(axis)
                axes[name] = (transform[:3, 3].copy(), transform[:3, :3] @ axis, transform.copy())
                transform[:3, :3] = transform[:3, :3] @ Rotation.from_rotvec(axis * values.get(name, 0)).as_matrix()
            poses[joint.find("child").get("link")] = transform
            pending.remove(joint)
            advanced = True
        if not advanced:
            raise PlanError("URDF joint graph is disconnected or cyclic")
    return axes, poses


def fk_identity_check(
    sim_asset: AssetSpec, driver_asset: AssetSpec, *, groups: tuple[str, ...] | None = None,
    n_samples: int = 200, seed: int = 0, position_tolerance: float = 1.5e-3,
    orientation_tolerance: float = 1.0e-3,
) -> dict[str, Any]:
    """RR_08 joint-axis-line identity, with a separate tool-convention check.

    Tool offset is evaluated in the straight pose to isolate the fixed tool
    convention from the known upstream axis-placement deviations. Its length
    is reported, never gated. Random configurations gate every signed joint
    axis and its line in the common base instead of comparing named frames.
    """
    from scipy.spatial.transform import Rotation

    lower, upper = _fk_check_bounds(sim_asset)
    driver_lower, driver_upper = _fk_check_bounds(driver_asset)
    lower, upper = np.maximum(lower, driver_lower), np.minimum(upper, driver_upper)
    if len(sim_asset.joint_names) != len(driver_asset.joint_names) or np.any(lower >= upper):
        raise PlanError("FK identity failed: joint count or common position bounds")
    rng = np.random.default_rng(seed)
    worst_angle = worst_distance = 0.0
    worst_joint = None
    for sample in range(n_samples):
        q = rng.uniform(lower, upper)
        sim_axes, _ = _urdf_axis_poses(sim_asset, q)
        driver_axes, _ = _urdf_axis_poses(driver_asset, q)
        for sim_name, driver_name in zip(sim_asset.joint_names, driver_asset.joint_names):
            a, u, _ = sim_axes[sim_name]
            b, v, _ = driver_axes[driver_name]
            cross = np.cross(u, v)
            angle = float(np.arctan2(np.linalg.norm(cross), np.dot(u, v)))
            distance = float(abs(np.dot(b - a, cross)) / np.linalg.norm(cross)) \
                if np.linalg.norm(cross) > 1e-8 else float(np.linalg.norm(np.cross(b - a, u)))
            worst_angle = max(worst_angle, angle)
            if distance >= worst_distance:
                worst_distance, worst_joint = distance, sim_name
            if angle > orientation_tolerance or distance > position_tolerance:
                raise PlanError(f"FK identity failed at sample {sample}, joint {sim_name}: "
                                f"axis angle {angle} rad, axis-line distance {distance} m; q={q.tolist()}")
    sim_axes, sim_links = _urdf_axis_poses(sim_asset, np.zeros(len(lower)))
    _, driver_links = _urdf_axis_poses(driver_asset, np.zeros(len(lower)))
    last_origin, last_axis, last_frame = sim_axes[sim_asset.joint_names[-1]]
    tool_reports = {}
    for sim_group, driver_group in zip(kinematic_groups(sim_asset, groups), kinematic_groups(driver_asset, groups)):
        a, b = sim_links[sim_group.tip_link], driver_links[driver_group.tip_link]
        delta = b[:3, 3] - a[:3, 3]
        axial = float(np.dot(delta, last_axis))
        lateral = float(np.linalg.norm(delta - axial * last_axis))
        rotation = float(Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude())
        if lateral > 1e-6 or rotation > 1e-6:
            raise PlanError(f"FK identity failed: tool lateral {lateral} m, rotation {rotation} rad")
        tool_reports[sim_group.name] = {"translation_in_last_joint_frame_m": (last_frame[:3, :3].T @ delta).tolist(),
                                        "axial_m": axial, "lateral_m": lateral, "rotation_rad": rotation,
                                        "length_gated": False, "configuration": "straight"}
    return {"ok": True, "n_samples": n_samples, "joint_comparisons": n_samples * len(lower),
            "worst_axis_angle_rad": worst_angle, "worst_axis_line_distance_m": worst_distance,
            "worst_axis_line_joint": worst_joint, "axis_angle_tolerance_rad": orientation_tolerance,
            "axis_line_tolerance_m": position_tolerance, "tools": tool_reports}


def _fk_check_bounds(asset: AssetSpec) -> tuple[np.ndarray, np.ndarray]:
    joints = asset.resolve_active_joints()
    lower = np.asarray([joint.lower if joint.lower is not None else -np.pi for joint in joints])
    upper = np.asarray([joint.upper if joint.upper is not None else np.pi for joint in joints])
    return lower, upper


def load_plan(output_dir: Path) -> PlanBundle:
    manifest = json.loads((output_dir / "plan.manifest.json").read_text(encoding="utf-8"))

    def _load_segments(key: str) -> list[PlanSegment]:
        loaded = []
        for entry in manifest.get(key, []):
            trajectory = MaterializedTrajectory.load(output_dir / f"{entry['segment_id']}.json")
            loaded.append(PlanSegment(entry["segment_id"], entry["kind"], trajectory, entry["digest"]))
        return loaded

    segments = _load_segments("segments")
    standstill_segments = _load_segments("standstill_segments")
    asset_path = output_dir / f"{manifest['asset']}.relimited.urdf"
    asset = AssetSpec(name=manifest["asset"], urdf_path=asset_path, active_joints=tuple(manifest["joint_names"]))
    return PlanBundle(config_digest=manifest["config_digest"], asset=asset, segments=tuple(segments),
                      rejected_for_collision=0, standstill_segments=tuple(standstill_segments))
