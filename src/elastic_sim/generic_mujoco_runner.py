"""Portable URDF asset runners backed by MuJoCo.

The public result contract deliberately mirrors :mod:`generic_newton_runner`:
all modes return canonical ``q_ref``, ``q_link``, ``q_motor`` and torque
arrays in the asset's declared active-joint order.  This lets synthetic data
generation and downstream calibration code switch backend without changing
their CSV schema.

MuJoCo's native URDF importer cannot load the Collada/STL resource mix in all
of the supplied upstream descriptions on Windows.  The loader therefore
converts those local meshes to transient OBJ files before compiling the URDF.
The original URDF/mesh bundle remains the source of truth and is used
unchanged by Newton.  A primitive proxy is used only for an individual mesh
that cannot be converted.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import tempfile
from typing import Any, Mapping
from xml.etree import ElementTree as ET

import numpy as np

from .assets import AssetSpec, discover_urdf_joints
from .generic_newton_runner import _ensure_collision_geometry, _expand_simple_xacro_text
from .materialized import MaterializedTrajectory
from .serial_trajectory import SerialArmTrajectory, SerialTrajectoryConfig, trajectory_evaluator


class GenericMujocoTrajectoryRunner:
    """Run a portable asset in ``kinematic``, ``rigid`` or ``elastic`` mode.

    Rigid mode applies PD torques directly to native URDF joints.  Elastic
    mode uses an ideal motor-side trajectory and applies the declared virtual
    transmission spring/damper torque to each native joint; this is a stable
    MuJoCo counterpart to the experimental Newton elastic path.
    """

    def __init__(
        self, asset: AssetSpec, config: Mapping[str, Any] | None = None, *, mode: str = "kinematic"
    ) -> None:
        if mode not in {"kinematic", "rigid", "elastic"}:
            raise ValueError(f"Unsupported MuJoCo asset mode {mode!r}")
        self.asset = asset
        self.config = dict(config or {})
        self.mode = mode
        self._joint_names = asset.joint_names

    def run(
        self,
        trajectory: SerialTrajectoryConfig | MaterializedTrajectory,
        *,
        time_step: float = 0.004,
        visualize: bool = False,
        realtime_scale: float = 1.0,
    ) -> Mapping[str, Any]:
        if tuple(trajectory.joint_names) != self._joint_names:
            raise ValueError("Trajectory joint_names must match asset active_joints exactly")
        if time_step <= 0.0 or realtime_scale <= 0.0:
            raise ValueError("time_step and realtime_scale must be positive")
        if self.mode == "elastic":
            return _run_explicit_elastic(
                self.asset, self.config, trajectory, time_step=time_step,
                visualize=visualize, realtime_scale=realtime_scale,
            )
        mujoco = _require_mujoco()
        model, uses_mesh_proxies = _build_model(self.asset, mujoco, time_step, body_overrides=self.config.get("body_overrides", {}))
        data = mujoco.MjData(model)
        active = _joint_addresses(model, mujoco, self._joint_names)
        # A whole-body URDF may include joints outside the selected chain.
        # Hold them at their URDF zero pose during dynamic single-arm replay.
        all_names = tuple(
            joint.name for joint in discover_urdf_joints(self.asset.urdf_path)
            if joint.is_one_dof and joint.mimic is None
        )
        all_addresses = _joint_addresses(model, mujoco, all_names, required=False)
        evaluator = trajectory_evaluator(trajectory)
        q0, dq0, _ = evaluator(0.0)
        for index, (qpos, dof) in enumerate(active):
            data.qpos[qpos] = q0[index]
            data.qvel[dof] = dq0[index]
        mujoco.mj_forward(model, data)
        time_grid = trajectory.time.copy() if isinstance(trajectory, MaterializedTrajectory) else np.arange(0.0, trajectory.duration + 0.5 * time_step, time_step)
        q_ref_values: list[Any] = []
        dq_ref_values: list[Any] = []
        q_values: list[Any] = []
        dq_values: list[Any] = []
        q_motor_values: list[Any] = []
        dq_motor_values: list[Any] = []
        tau_values: list[Any] = []
        viewer = _launch_viewer(mujoco, model, data) if visualize else None
        if uses_mesh_proxies:
            print("MuJoCo used primitive proxies only for mesh files that could not be converted to OBJ.")
        try:
            for sample_index, sample_time in enumerate(time_grid):
                if viewer is not None and not viewer.is_running():
                    time_grid = time_grid[:sample_index]
                    break
                target_q, target_dq, _ = evaluator(float(sample_time))
                if self.mode == "kinematic":
                    for index, (qpos, dof) in enumerate(active):
                        data.qpos[qpos] = target_q[index]
                        data.qvel[dof] = target_dq[index]
                    mujoco.mj_forward(model, data)
                measured_q = np.asarray([data.qpos[qpos] for qpos, _ in active], dtype=float)
                measured_dq = np.asarray([data.qvel[dof] for _, dof in active], dtype=float)
                if not (np.isfinite(measured_q).all() and np.isfinite(measured_dq).all()):
                    raise RuntimeError(f"MuJoCo simulation became non-finite at t={sample_time:.6f}s")
                if self.mode == "elastic":
                    motor_q, motor_dq = target_q.copy(), target_dq.copy()
                    tau = _elastic_torque(target_q, target_dq, measured_q, measured_dq, self.config, self.asset)
                elif self.mode == "rigid":
                    motor_q, motor_dq = measured_q.copy(), measured_dq.copy()
                    tau = _rigid_torque(target_q, target_dq, measured_q, measured_dq, self.config, self.asset)
                else:
                    motor_q, motor_dq = target_q.copy(), target_dq.copy()
                    tau = np.zeros_like(target_q)
                q_ref_values.append(target_q)
                dq_ref_values.append(target_dq)
                q_values.append(measured_q)
                dq_values.append(measured_dq)
                q_motor_values.append(motor_q)
                dq_motor_values.append(motor_dq)
                tau_values.append(tau)
                if sample_index + 1 == len(time_grid):
                    break
                if self.mode != "kinematic":
                    data.qfrc_applied[:] = 0.0
                    _apply_hold_torques(data, all_addresses, active, target_q, target_dq, self.config)
                    for index, (_, dof) in enumerate(active):
                        data.qfrc_applied[dof] += tau[index]
                    mujoco.mj_step(model, data)
                if viewer is not None:
                    viewer.sync()
                    _sleep(time_step / realtime_scale)
        finally:
            if viewer is not None:
                viewer.close()
        return {
            "time": time_grid[:len(q_values)],
            "q_ref": np.asarray(q_ref_values), "dq_ref": np.asarray(dq_ref_values),
            "q_link": np.asarray(q_values), "dq_link": np.asarray(dq_values),
            "q_motor": np.asarray(q_motor_values), "dq_motor": np.asarray(dq_motor_values),
            "tau_motor": np.asarray(tau_values), "joint_names": self._joint_names,
        }


class GenericMujocoKinematicTrajectoryRunner(GenericMujocoTrajectoryRunner):
    def __init__(self, asset: AssetSpec, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(asset, config, mode="kinematic")


class GenericMujocoRigidTrajectoryRunner(GenericMujocoTrajectoryRunner):
    def __init__(self, asset: AssetSpec, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(asset, config, mode="rigid")


class GenericMujocoElasticTrajectoryRunner(GenericMujocoTrajectoryRunner):
    def __init__(self, asset: AssetSpec, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(asset, config, mode="elastic")


def _run_explicit_elastic(
    asset: AssetSpec,
    config: Mapping[str, Any],
    trajectory: SerialTrajectoryConfig | MaterializedTrajectory,
    *, time_step: float, visualize: bool, realtime_scale: float,
) -> Mapping[str, Any]:
    """Run an actual motor joint + elastic joint chain in MuJoCo.

    Each active URDF joint is transformed before import into a motor-side joint
    followed by a fictional intermediate link and a passive, zero-reference
    elastic joint.  The output coordinate is their sum, precisely matching
    the public Newton elastic-chain convention.
    """
    mujoco = _require_mujoco()
    parameters = _resolved_transmissions(asset, config)
    model, uses_mesh_proxies = _build_model(
        asset, mujoco, time_step, elastic_transmissions=parameters,
        body_overrides=config.get("body_overrides", {}),
    )
    data = mujoco.MjData(model)
    addresses = _elastic_addresses(model, mujoco, asset.joint_names)
    for name, indices in addresses.items():
        joint_id = indices["elastic_joint_id"]
        model.jnt_stiffness[joint_id] = parameters[name]["stiffness"]
        model.dof_damping[indices["elastic_dof"]] = parameters[name]["damping"]
    _apply_link_mass_overrides(model, mujoco, asset, parameters)
    mujoco.mj_setConst(model, data)
    evaluator = trajectory_evaluator(trajectory)
    q0, dq0, _ = evaluator(0.0)
    for index, name in enumerate(asset.joint_names):
        indices = addresses[name]
        data.qpos[indices["motor_qpos"]] = q0[index]
        data.qvel[indices["motor_dof"]] = dq0[index]
        data.qpos[indices["elastic_qpos"]] = 0.0
        data.qvel[indices["elastic_dof"]] = 0.0
    mujoco.mj_forward(model, data)
    time_grid = trajectory.time.copy() if isinstance(trajectory, MaterializedTrajectory) else np.arange(0.0, trajectory.duration + 0.5 * time_step, time_step)
    q_ref_values: list[Any] = []
    dq_ref_values: list[Any] = []
    q_link_values: list[Any] = []
    dq_link_values: list[Any] = []
    q_motor_values: list[Any] = []
    dq_motor_values: list[Any] = []
    tau_values: list[Any] = []
    viewer = _launch_viewer(mujoco, model, data) if visualize else None
    if uses_mesh_proxies:
        print("MuJoCo used primitive proxies only for mesh files that could not be converted to OBJ.")
    try:
        for sample_index, sample_time in enumerate(time_grid):
            if viewer is not None and not viewer.is_running():
                time_grid = time_grid[:sample_index]
                break
            target_q, target_dq, _ = evaluator(float(sample_time))
            motor_q = np.asarray([data.qpos[addresses[name]["motor_qpos"]] for name in asset.joint_names])
            elastic_q = np.asarray([data.qpos[addresses[name]["elastic_qpos"]] for name in asset.joint_names])
            motor_dq = np.asarray([data.qvel[addresses[name]["motor_dof"]] for name in asset.joint_names])
            elastic_dq = np.asarray([data.qvel[addresses[name]["elastic_dof"]] for name in asset.joint_names])
            link_q, link_dq = motor_q + elastic_q, motor_dq + elastic_dq
            if not all(np.isfinite(value).all() for value in (motor_q, elastic_q, motor_dq, elastic_dq)):
                raise RuntimeError(f"MuJoCo elastic simulation became non-finite at t={sample_time:.6f}s")
            tau = _clip_effort(np.asarray([
                parameters[name]["motor_stiffness"] * (target_q[index] - motor_q[index])
                + parameters[name]["motor_damping"] * (target_dq[index] - motor_dq[index])
                for index, name in enumerate(asset.joint_names)
            ]), asset)
            q_ref_values.append(target_q)
            dq_ref_values.append(target_dq)
            q_link_values.append(link_q)
            dq_link_values.append(link_dq)
            q_motor_values.append(motor_q)
            dq_motor_values.append(motor_dq)
            tau_values.append(tau)
            if sample_index + 1 == len(time_grid):
                break
            data.qfrc_applied[:] = 0.0
            for index, name in enumerate(asset.joint_names):
                data.qfrc_applied[addresses[name]["motor_dof"]] = tau[index]
            mujoco.mj_step(model, data)
            if viewer is not None:
                viewer.sync()
                _sleep(time_step / realtime_scale)
    finally:
        if viewer is not None:
            viewer.close()
    return {
        "time": time_grid[:len(q_link_values)],
        "q_ref": np.asarray(q_ref_values), "dq_ref": np.asarray(dq_ref_values),
        "q_link": np.asarray(q_link_values), "dq_link": np.asarray(dq_link_values),
        "q_motor": np.asarray(q_motor_values), "dq_motor": np.asarray(dq_motor_values),
        "tau_motor": np.asarray(tau_values), "joint_names": asset.joint_names,
    }


def _build_model(
    asset: AssetSpec, mujoco: Any, time_step: float,
    *, elastic_transmissions: Mapping[str, Mapping[str, float | None]] | None = None,
    body_overrides: Mapping[str, Mapping[str, float]] | None = None,
) -> tuple[Any, bool]:
    with _materialized_mujoco_urdf(asset, elastic_transmissions) as (urdf_path, uses_mesh_proxies):
        model = mujoco.MjModel.from_xml_path(str(urdf_path))
    model.opt.timestep = time_step
    model.opt.gravity[:] = asset.gravity
    _apply_body_overrides(model, mujoco, body_overrides or {})
    # The importer may assign no contact masks to visual-only geometries.  The
    # proxy collision geometries and native primitive collisions must all be
    # mutually eligible when an asset asks for self collision.
    if asset.self_collisions:
        model.geom_contype[:] = 1
        model.geom_conaffinity[:] = 1
        # Fixed decorative/support subassemblies in the FMRR description
        # intentionally interpenetrate at their mounting interfaces.  They
        # cannot move relative to the world, so excluding only those bodies
        # avoids a permanent contact singularity while preserving collision
        # between every articulated robot link.
        for geom_index, body_index in enumerate(model.geom_bodyid):
            if model.body_dofnum[body_index] == 0:
                model.geom_contype[geom_index] = 0
                model.geom_conaffinity[geom_index] = 0
    return model, uses_mesh_proxies


def _apply_body_overrides(model: Any, mujoco: Any, overrides: Mapping[str, Mapping[str, float]]) -> None:
    """Apply configured mass/principal-inertia values after MJCF import."""
    for name, values in overrides.items():
        body_id = -1
        for candidate in range(model.nbody):
            label = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, candidate)
            if label and (label == name or label.endswith("/" + str(name))):
                body_id = candidate
                break
        if body_id < 0:
            raise ValueError(f"MuJoCo body override names unknown body {name!r}")
        if "mass" in values:
            old_mass = float(model.body_mass[body_id])
            new_mass = float(values["mass"])
            if new_mass <= 0.0:
                raise ValueError(f"body.{name}.mass must be positive")
            if not all(field in values for field in ("inertia_x", "inertia_y", "inertia_z")) and old_mass > 0.0:
                model.body_inertia[body_id] *= new_mass / old_mass
            model.body_mass[body_id] = new_mass
        for index, field in enumerate(("inertia_x", "inertia_y", "inertia_z")):
            if field in values:
                value = float(values[field])
                if value <= 0.0:
                    raise ValueError(f"body.{name}.{field} must be positive")
                model.body_inertia[body_id, index] = value


@contextmanager
def _materialized_mujoco_urdf(
    asset: AssetSpec, elastic_transmissions: Mapping[str, Mapping[str, float | None]] | None = None,
):
    """Yield a MuJoCo-loadable temporary URDF and converted local mesh files."""
    text = asset.urdf_path.read_text(encoding="utf-8")
    if "${" in text:
        text = _expand_simple_xacro_text(text)
    text = _ensure_collision_geometry(text)
    root = ET.fromstring(text)
    if elastic_transmissions is not None:
        _insert_elastic_transmissions(root, asset, elastic_transmissions)
    if asset.self_collisions:
        _exclude_adjacent_link_contacts(root)
    resources = iter(asset.validate_resources())
    uses_mesh_proxies = False
    with tempfile.TemporaryDirectory(prefix="elastic_mujoco_urdf_") as directory:
        output_dir = Path(directory)
        for mesh_index, mesh in enumerate(root.findall(".//mesh")):
            try:
                source = next(resources)
            except StopIteration as exc:  # pragma: no cover - malformed XML guard
                raise ValueError(f"Asset {asset.name!r} mesh references changed during preparation") from exc
            target = output_dir / f"mesh_{mesh_index:03d}.obj"
            try:
                _convert_mesh_to_obj(source, target)
                mesh.set("filename", target.name)
            except Exception:
                # Use a conservative visible/collidable fallback only when a
                # particular upstream mesh cannot be converted.
                geometry = _parent_geometry(root, mesh)
                geometry.remove(mesh)
                ET.SubElement(geometry, "sphere", {"radius": "0.10"})
                uses_mesh_proxies = True
        try:
            next(resources)
            raise ValueError(f"Asset {asset.name!r} mesh references changed during preparation")
        except StopIteration:
            pass
        urdf_path = output_dir / asset.urdf_path.name
        ET.ElementTree(root).write(urdf_path, encoding="utf-8", xml_declaration=True)
        yield urdf_path, uses_mesh_proxies


def _insert_elastic_transmissions(
    root: ET.Element, asset: AssetSpec, parameters: Mapping[str, Mapping[str, float | None]],
) -> None:
    """Replace each selected URDF joint with motor + intermediate + elastic."""
    for joint in list(root.findall("joint")):
        name = joint.get("name")
        if name not in parameters:
            continue
        original_name = str(name)
        child = joint.find("child")
        parent = joint.find("parent")
        axis = joint.find("axis")
        if child is None or parent is None or not child.get("link"):
            raise ValueError(f"Active joint {original_name!r} has no valid parent/child")
        intermediate_name = f"elastic_link__{original_name}"
        original_child = child.get("link")
        child.set("link", intermediate_name)
        joint.set("name", f"motor__{original_name}")
        link = ET.Element("link", {"name": intermediate_name})
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "mass", {"value": f"{parameters[original_name]['intermediate_mass']:.12g}"})
        inertia = max(float(parameters[original_name]["intermediate_mass"]) * 1.0e-4, 1.0e-8)
        ixx = float(parameters[original_name].get("intermediate_inertia_x") or inertia)
        iyy = float(parameters[original_name].get("intermediate_inertia_y") or inertia)
        izz = float(parameters[original_name].get("intermediate_inertia_z") or inertia)
        ET.SubElement(inertial, "inertia", {
            "ixx": f"{ixx:.12g}", "ixy": "0", "ixz": "0",
            "iyy": f"{iyy:.12g}", "iyz": "0", "izz": f"{izz:.12g}",
        })
        elastic_joint = ET.Element("joint", {
            "name": f"elastic__{original_name}", "type": joint.get("type", "revolute"),
        })
        ET.SubElement(elastic_joint, "parent", {"link": intermediate_name})
        ET.SubElement(elastic_joint, "child", {"link": original_child})
        ET.SubElement(elastic_joint, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        if axis is not None:
            ET.SubElement(elastic_joint, "axis", {"xyz": axis.get("xyz", "1 0 0")})
        root.append(link)
        root.append(elastic_joint)


def _exclude_adjacent_link_contacts(root: ET.Element) -> None:
    """Keep self-collision meaningful by excluding only mechanically adjacent links."""
    mujoco_extension = root.find("mujoco")
    if mujoco_extension is None:
        mujoco_extension = ET.SubElement(root, "mujoco")
    contact = mujoco_extension.find("contact")
    if contact is None:
        contact = ET.SubElement(mujoco_extension, "contact")
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        body1, body2 = parent.get("link"), child.get("link")
        if body1 and body2 and body1 != "world":
            ET.SubElement(contact, "exclude", {"body1": body1, "body2": body2})


def _resolved_transmissions(asset: AssetSpec, config: Mapping[str, Any]) -> dict[str, dict[str, float | None]]:
    configured = config.get("transmissions", {}) or {}
    if not isinstance(configured, Mapping):
        raise ValueError("transmissions must be a mapping keyed by active joint name")
    result: dict[str, dict[str, float | None]] = {}
    for name in asset.joint_names:
        values = configured.get(name, {}) or {}
        if not isinstance(values, Mapping):
            raise ValueError(f"transmissions.{name} must be a mapping")
        result[name] = {
            "stiffness": float(values.get("stiffness", config.get("default_stiffness", 10_000.0))),
            "damping": float(values.get("damping", config.get("default_damping", 100.0))),
            "motor_stiffness": float(values.get("motor_stiffness", config.get("motor_stiffness", 3_000.0))),
            "motor_damping": float(values.get("motor_damping", config.get("motor_damping", 100.0))),
            "intermediate_mass": float(values.get("intermediate_mass", config.get("intermediate_mass", 0.10))),
            "link_mass": values.get("link_mass"),
            "intermediate_inertia_x": values.get("intermediate_inertia_x"),
            "intermediate_inertia_y": values.get("intermediate_inertia_y"),
            "intermediate_inertia_z": values.get("intermediate_inertia_z"),
        }
        if result[name]["link_mass"] is not None:
            result[name]["link_mass"] = float(result[name]["link_mass"])
        for field in ("intermediate_inertia_x", "intermediate_inertia_y", "intermediate_inertia_z"):
            if result[name][field] is not None:
                result[name][field] = float(result[name][field])
    return result


def _elastic_addresses(model: Any, mujoco: Any, names: tuple[str, ...]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for name in names:
        motor_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"motor__{name}")
        elastic_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"elastic__{name}")
        if motor_joint_id < 0 or elastic_joint_id < 0:
            raise ValueError(f"MuJoCo did not import explicit elastic transmission for {name!r}")
        result[name] = {
            "motor_joint_id": motor_joint_id, "elastic_joint_id": elastic_joint_id,
            "motor_qpos": int(model.jnt_qposadr[motor_joint_id]), "elastic_qpos": int(model.jnt_qposadr[elastic_joint_id]),
            "motor_dof": int(model.jnt_dofadr[motor_joint_id]), "elastic_dof": int(model.jnt_dofadr[elastic_joint_id]),
        }
    return result


def _apply_link_mass_overrides(
    model: Any, mujoco: Any, asset: AssetSpec, parameters: Mapping[str, Mapping[str, float | None]],
) -> None:
    for joint in asset.resolve_active_joints():
        override = parameters[joint.name]["link_mass"]
        if override is None:
            continue
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, joint.child)
        if body_id < 0 or model.body_mass[body_id] <= 0.0:
            raise ValueError(f"Cannot override mass for URDF child link {joint.child!r}")
        scale = float(override) / float(model.body_mass[body_id])
        model.body_mass[body_id] = float(override)
        model.body_inertia[body_id] *= scale


def _convert_mesh_to_obj(source: Path, target: Path) -> None:
    """Convert local STL/DAE/etc. into the OBJ dialect MuJoCo loads on Windows."""
    try:
        import trimesh
    except ImportError as exc:  # pragma: no cover - dependency is pinned in pyproject
        raise ImportError("MuJoCo mesh conversion requires trimesh") from exc
    loaded = trimesh.load_mesh(source, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        geometries = tuple(loaded.geometry.values())
        if not geometries:
            raise ValueError(f"Mesh scene is empty: {source}")
        loaded = trimesh.util.concatenate(geometries)
    if not isinstance(loaded, trimesh.Trimesh) or loaded.vertices.size == 0:
        raise ValueError(f"Mesh has no triangular geometry: {source}")
    loaded.export(target)


def _parent_geometry(root: ET.Element, mesh: ET.Element) -> ET.Element:
    for geometry in root.findall(".//geometry"):
        if mesh in list(geometry):
            return geometry
    raise ValueError("URDF mesh has no geometry parent")


def _joint_addresses(model: Any, mujoco: Any, names: tuple[str, ...], *, required: bool = True) -> dict[str, tuple[int, int]] | list[tuple[int, int]]:
    found: dict[str, tuple[int, int]] = {}
    for name in names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            if required:
                raise ValueError(f"Configured URDF joint {name!r} was not imported by MuJoCo")
            continue
        found[name] = (int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id]))
    return [found[name] for name in names] if required else found


def _rigid_torque(q_ref: Any, dq_ref: Any, q: Any, dq: Any, config: Mapping[str, Any], asset: AssetSpec) -> np.ndarray:
    stiffness = float(config.get("motor_stiffness", 3_000.0))
    damping = float(config.get("motor_damping", 100.0))
    return _clip_effort(stiffness * (q_ref - q) + damping * (dq_ref - dq), asset)


def _elastic_torque(q_motor: Any, dq_motor: Any, q_link: Any, dq_link: Any, config: Mapping[str, Any], asset: AssetSpec) -> np.ndarray:
    stiffness = float(config.get("default_stiffness", 10_000.0))
    damping = float(config.get("default_damping", 100.0))
    return _clip_effort(stiffness * (q_motor - q_link) + damping * (dq_motor - dq_link), asset)


def _clip_effort(torque: Any, asset: AssetSpec) -> np.ndarray:
    limits = np.asarray([joint.effort if joint.effort is not None else 10_000.0 for joint in asset.resolve_active_joints()])
    return np.clip(np.asarray(torque, dtype=float), -limits, limits)


def _apply_hold_torques(
    data: Any,
    all_addresses: dict[str, tuple[int, int]] | list[tuple[int, int]],
    active_addresses: list[tuple[int, int]],
    target_q: Any,
    target_dq: Any,
    config: Mapping[str, Any],
) -> None:
    if not isinstance(all_addresses, dict):
        return
    active_dofs = {dof for _, dof in active_addresses}
    stiffness = float(config.get("motor_stiffness", 3_000.0))
    damping = float(config.get("motor_damping", 100.0))
    for qpos, dof in all_addresses.values():
        if dof not in active_dofs:
            data.qfrc_applied[dof] = stiffness * (0.0 - data.qpos[qpos]) + damping * (0.0 - data.qvel[dof])


def _launch_viewer(mujoco: Any, model: Any, data: Any) -> Any:
    try:
        import mujoco.viewer
        return mujoco.viewer.launch_passive(model, data)
    except Exception as exc:
        raise RuntimeError("MuJoCo viewer could not start; retry headless without --visualize") from exc


def _sleep(seconds: float) -> None:
    import time
    time.sleep(seconds)


def _require_mujoco() -> Any:
    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("MuJoCo asset simulation requires mujoco; run `uv sync`") from exc
    return mujoco
