"""Generic Newton construction of elastic transmissions from an :mod:`assets` spec.

The Cartesian Newton script predates this module and remains unchanged.  New
robots should use this builder: it has no knowledge of a robot name, end
effector, or a fixed number of axes.  Every selected 1-DoF URDF joint becomes
two serial degrees of freedom::

    original parent -- motor joint -- massless transmission link
                    -- passive elastic joint -- original child

The observed/link-side coordinate is the elastic joint coordinate; the motor
coordinate is where a position controller or a replayed torque is applied.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, TYPE_CHECKING

from .assets import AssetSpec, load_asset_spec

if TYPE_CHECKING:
    from .generic_calibration import TorqueReplayRollout


@dataclass(frozen=True)
class ElasticTransmissionParams:
    """Physical and controller parameters for one URDF active joint.

    Stiffness/damping are direct physical values (Nm/rad and Nms/rad for a
    revolute joint; N/m and Ns/m for a prismatic joint).  This intentionally
    does not use the Cartesian effective-mass damping-ratio parametrisation.
    """

    stiffness: float
    damping: float
    motor_stiffness: float = 30_000.0
    motor_damping: float = 500.0
    effort_limit: float | None = None

    def __post_init__(self) -> None:
        for field_name in ("stiffness", "damping", "motor_stiffness", "motor_damping"):
            value = getattr(self, field_name)
            if value < 0.0:
                raise ValueError(f"{field_name} must be non-negative")
        if self.effort_limit is not None and self.effort_limit <= 0.0:
            raise ValueError("effort_limit must be positive when supplied")


@dataclass(frozen=True)
class ElasticDofIndices:
    """Newton generalized-coordinate indices for one transmission."""

    motor_q: int
    motor_qd: int
    elastic_q: int
    elastic_qd: int


@dataclass
class ElasticModelBuild:
    """Built Newton model plus stable joint-name-to-DoF mapping."""

    model: Any
    asset: AssetSpec
    active_joint_names: tuple[str, ...]
    dof_index: dict[str, ElasticDofIndices]
    motor_joint_labels: dict[str, str]
    elastic_joint_labels: dict[str, str]

    @property
    def dof_count(self) -> int:
        return len(self.active_joint_names)

    def motor_indices(self) -> tuple[int, ...]:
        return tuple(self.dof_index[name].motor_qd for name in self.active_joint_names)

    def link_indices(self) -> tuple[int, ...]:
        return tuple(self.dof_index[name].elastic_qd for name in self.active_joint_names)


class GenericNewtonElasticBuilder:
    """Build a generic elastic Newton model from an :class:`AssetSpec`."""

    def __init__(
        self,
        asset: AssetSpec,
        transmissions: Mapping[str, ElasticTransmissionParams],
        *,
        gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
        intermediate_mass: float = 1.0e-5,
        intermediate_size: float = 0.01,
    ) -> None:
        self.asset = asset
        self.transmissions = dict(transmissions)
        self.gravity = gravity
        self.intermediate_mass = float(intermediate_mass)
        self.intermediate_size = float(intermediate_size)
        if self.intermediate_mass <= 0.0 or self.intermediate_size <= 0.0:
            raise ValueError("intermediate_mass and intermediate_size must be positive")
        active = asset.resolve_active_joints()
        active_names = tuple(joint.name for joint in active)
        missing = [name for name in active_names if name not in self.transmissions]
        extra = sorted(set(self.transmissions).difference(active_names))
        if missing or extra:
            raise ValueError(
                "Transmission parameters must match active joints exactly; "
                f"missing={missing}, extra={extra}"
            )
        self._active_joint_names = active_names

    @property
    def active_joint_names(self) -> tuple[str, ...]:
        return self._active_joint_names

    def build(self) -> ElasticModelBuild:
        """Load the URDF and return a finalized Newton model.

        Newton/Warp are imported only here so tools that merely discover assets
        or load dataset configurations do not require a GPU installation.
        """
        wp, newton = _require_newton()
        source = newton.ModelBuilder()
        base_xform = wp.transform(wp.vec3(*self.asset.base_position), wp.quat(*self.asset.base_quaternion))
        source.add_urdf(
            str(self.asset.urdf_path),
            xform=base_xform,
            floating=False,
            enable_self_collisions=self.asset.self_collisions,
            ignore_inertial_definitions=False,
            collapse_fixed_joints=False,
            force_position_velocity_actuation=True,
            parse_visuals_as_colliders=False,
        )
        target = newton.ModelBuilder()
        body_map, body_props = _copy_bodies(source, target, wp, newton, self.intermediate_mass, self.intermediate_size)
        _copy_shapes(source, target, newton, body_map)
        _restore_body_properties(target, body_props)

        snapshots = [_joint_snapshot(source, index) for index in range(source.joint_count)]
        # Newton versions differ in whether imported labels retain a model
        # namespace (``robot/joint_1``).  Resolve that implementation detail
        # once while retaining plain URDF names as the public API.
        source_by_urdf_name = {
            urdf_name: _find_snapshot_for_urdf_joint(snapshots, urdf_name)
            for urdf_name in self.active_joint_names
        }
        missing_from_newton = [name for name, snapshot in source_by_urdf_name.items() if snapshot is None]
        if missing_from_newton:
            raise ValueError(
                "Configured active URDF joints were not imported by Newton: "
                + ", ".join(missing_from_newton)
            )

        active_by_source_label = {
            snapshot["label"]: urdf_name
            for urdf_name, snapshot in source_by_urdf_name.items()
            if snapshot is not None
        }
        all_joint_indices: list[int] = []
        motor_indices: dict[str, int] = {}
        elastic_indices: dict[str, int] = {}
        motor_labels: dict[str, str] = {}
        elastic_labels: dict[str, str] = {}
        for snapshot in snapshots:
            source_label = snapshot["label"]
            name = active_by_source_label.get(source_label)
            if name is None:
                all_joint_indices.append(_add_joint_from_snapshot(target, snapshot, newton, wp, body_map))
                continue
            if snapshot["type"] not in (newton.JointType.REVOLUTE, newton.JointType.PRISMATIC):
                raise ValueError(f"Active joint {name!r} is not a supported 1-DoF Newton joint")
            params = self.transmissions[name]
            intermediate = target.add_link(
                xform=source.body_q[snapshot["child"]],
                com=wp.vec3(),
                inertia=_small_inertia(wp, self.intermediate_mass, self.intermediate_size),
                mass=self.intermediate_mass,
                label=f"elastic_transmission/{name}",
                is_kinematic=False,
            )
            motor_label = f"motor/{name}"
            elastic_label = f"elastic/{name}"
            motor_indices[name] = _add_joint_from_snapshot(
                target,
                snapshot,
                newton,
                wp,
                body_map,
                child=intermediate,
                child_xform=wp.transform_identity(),
                label=motor_label,
                target_ke=params.motor_stiffness,
                target_kd=params.motor_damping,
                effort_limit=params.effort_limit,
            )
            elastic_indices[name] = _add_joint_from_snapshot(
                target,
                snapshot,
                newton,
                wp,
                body_map,
                parent=intermediate,
                parent_xform=wp.transform_identity(),
                label=elastic_label,
                target_pos=0.0,
                target_vel=0.0,
                target_ke=params.stiffness,
                target_kd=params.damping,
                limit_lower=-1.0e10,
                limit_upper=1.0e10,
                effort_limit=params.effort_limit,
            )
            motor_labels[name] = motor_label
            elastic_labels[name] = elastic_label
            all_joint_indices.extend((motor_indices[name], elastic_indices[name]))

        target.add_articulation(all_joint_indices, label=self.asset.name)
        dof_index = {
            name: ElasticDofIndices(
                motor_q=int(target.joint_q_start[motor_indices[name]]),
                motor_qd=int(target.joint_qd_start[motor_indices[name]]),
                elastic_q=int(target.joint_q_start[elastic_indices[name]]),
                elastic_qd=int(target.joint_qd_start[elastic_indices[name]]),
            )
            for name in self.active_joint_names
        }
        model = target.finalize()
        model.set_gravity(self.gravity)
        return ElasticModelBuild(model, self.asset, self.active_joint_names, dof_index, motor_labels, elastic_labels)


def build_elastic_model(
    asset: AssetSpec,
    transmissions: Mapping[str, ElasticTransmissionParams],
    **kwargs: Any,
) -> ElasticModelBuild:
    """Convenience façade used by generic trajectory and calibration runners."""
    return GenericNewtonElasticBuilder(asset, transmissions, **kwargs).build()


class GenericNewtonTorqueReplayRunner:
    """Open-loop motor-torque replay adapter for generic dataset calibration.

    It intentionally returns output-side joint motion as ``q_link``/``dq_link``
    even though the transmission is represented by two generalized
    coordinates.  For a serial transmission those are the sums of motor and
    elastic coordinates, which is what an encoder on the original URDF child
    link observes.
    """

    def __init__(self, asset: AssetSpec, config: Mapping[str, Any] | None = None) -> None:
        self.asset = asset
        self.config = dict(config or {})
        self._joint_names = asset.joint_names

    def run_torque_replay(
        self, params: Mapping[str, float], rollout: "TorqueReplayRollout"
    ) -> Mapping[str, Any]:
        if tuple(rollout.joint_names) != self._joint_names:
            raise ValueError(
                "Rollout joint_names must match asset active_joints exactly; "
                f"expected {self._joint_names}, got {tuple(rollout.joint_names)}"
            )
        if rollout.q.shape != rollout.dq.shape or rollout.q.shape != rollout.tau.shape:
            raise ValueError("Torque replay q, dq and tau arrays must have matching shapes")
        if rollout.q.shape[1] != len(self._joint_names):
            raise ValueError("Torque replay dimension does not match the asset")
        transmissions = _transmissions_from_named_params(self._joint_names, params, self.config)
        built = build_elastic_model(
            self.asset,
            transmissions,
            gravity=tuple(self.config.get("gravity", (0.0, 0.0, -9.81))),
            intermediate_mass=float(self.config.get("intermediate_mass", 1.0e-5)),
            intermediate_size=float(self.config.get("intermediate_size", 0.01)),
        )
        _wp, newton = _require_newton()
        model = built.model
        state_in, state_out = model.state(), model.state()
        control = model.control()
        if not hasattr(control, "joint_f"):
            raise RuntimeError("This Newton build does not expose Control.joint_f for torque replay")
        solver = _new_solver(model, newton)
        contacts = model.contacts()
        direct_solvers = tuple(
            cls for cls in (
                getattr(newton.solvers, "SolverMuJoCo", None),
                getattr(newton.solvers, "SolverFeatherstone", None),
            ) if cls is not None
        )
        needs_ik = not isinstance(solver, direct_solvers) if direct_solvers else True

        # When a benchmark exposes a distinct motor encoder (the KUKA raw
        # data does), initialize both sides exactly.  Otherwise use the
        # documented zero-deflection convention: motor equals measured output.
        initial_q = _as_numpy(state_in.joint_q).reshape(-1)
        initial_dq = _as_numpy(state_in.joint_qd).reshape(-1)
        for index, name in enumerate(self._joint_names):
            mapping = built.dof_index[name]
            motor_q0 = rollout.q[0, index] if rollout.motor_q is None else rollout.motor_q[0, index]
            motor_dq0 = rollout.dq[0, index] if rollout.motor_dq is None else rollout.motor_dq[0, index]
            initial_q[mapping.motor_q] = motor_q0
            initial_q[mapping.elastic_q] = rollout.q[0, index] - motor_q0
            initial_dq[mapping.motor_qd] = motor_dq0
            initial_dq[mapping.elastic_qd] = rollout.dq[0, index] - motor_dq0
        state_in.joint_q.assign(initial_q.astype("float32"))
        state_in.joint_qd.assign(initial_dq.astype("float32"))
        newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)

        q_out = []
        dq_out = []
        q_motor = []
        dq_motor = []
        tau_motor = []
        for sample_index in range(len(rollout.time)):
            q, dq = _output_joint_state(state_in, built)
            q_out.append(q)
            dq_out.append(dq)
            motor_q, motor_dq = _motor_joint_state(state_in, built)
            q_motor.append(motor_q)
            dq_motor.append(motor_dq)
            tau = _as_numpy(control.joint_f).reshape(-1)
            tau.fill(0.0)
            for joint_index, name in enumerate(self._joint_names):
                tau[built.dof_index[name].motor_qd] = rollout.tau[sample_index, joint_index]
            control.joint_f.assign(tau.astype("float32"))
            tau_motor.append(rollout.tau[sample_index].copy())
            if sample_index + 1 == len(rollout.time):
                break
            dt = float(rollout.time[sample_index + 1] - rollout.time[sample_index])
            if dt <= 0.0:
                raise ValueError("rollout time must be strictly increasing")
            state_in.clear_forces()
            model.collide(state_in, contacts)
            solver.step(state_in, state_out, control, contacts, dt)
            if needs_ik:
                newton.eval_ik(model, state_out, state_out.joint_q, state_out.joint_qd)
            state_in, state_out = state_out, state_in
        import numpy as np
        return {
            "time": np.asarray(rollout.time, dtype=float),
            "q_link": np.asarray(q_out, dtype=float),
            "dq_link": np.asarray(dq_out, dtype=float),
            "q_motor": np.asarray(q_motor, dtype=float),
            "dq_motor": np.asarray(dq_motor, dtype=float),
            "tau_motor": np.asarray(tau_motor, dtype=float),
            "joint_names": self._joint_names,
        }


def make_torque_replay_runner(*, asset: str | Path | AssetSpec | None = None, config: Mapping[str, Any] | None = None) -> GenericNewtonTorqueReplayRunner:
    """Factory used by ``scripts/run_dataset_calibration.py``.

    ``asset`` may be a loaded :class:`AssetSpec` or a portable asset YAML.
    Per-joint defaults are configured under ``transmissions``; optimization
    values named ``<joint>.stiffness`` and ``<joint>.damping`` override them.
    """
    if asset is None:
        raise ValueError("A generic Newton torque-replay runner requires an asset YAML path")
    spec = asset if isinstance(asset, AssetSpec) else load_asset_spec(asset)
    return GenericNewtonTorqueReplayRunner(spec, config)


def _transmissions_from_named_params(
    joint_names: tuple[str, ...], params: Mapping[str, float], config: Mapping[str, Any]
) -> dict[str, ElasticTransmissionParams]:
    configured = config.get("transmissions", {})
    if configured is None:
        configured = {}
    if not isinstance(configured, Mapping):
        raise ValueError("transmissions must be a mapping keyed by active joint name")
    motor_stiffness = float(config.get("motor_stiffness", 30_000.0))
    motor_damping = float(config.get("motor_damping", 500.0))
    result: dict[str, ElasticTransmissionParams] = {}
    for name in joint_names:
        defaults = configured.get(name, {})
        if not isinstance(defaults, Mapping):
            raise ValueError(f"transmissions.{name} must be a mapping")
        stiffness = params.get(f"{name}.stiffness", defaults.get("stiffness"))
        damping = params.get(f"{name}.damping", defaults.get("damping"))
        if stiffness is None or damping is None:
            raise ValueError(
                f"Missing stiffness/damping for {name!r}; provide optimizer parameters "
                f"or config.transmissions.{name} defaults"
            )
        result[name] = ElasticTransmissionParams(
            stiffness=float(stiffness), damping=float(damping),
            motor_stiffness=float(defaults.get("motor_stiffness", motor_stiffness)),
            motor_damping=float(defaults.get("motor_damping", motor_damping)),
            effort_limit=None if defaults.get("effort_limit") is None else float(defaults["effort_limit"]),
        )
    return result


def _new_solver(model: Any, newton: Any) -> Any:
    errors = []
    for name in ("SolverMuJoCo", "SolverFeatherstone", "SolverSemiImplicit"):
        solver_class = getattr(newton.solvers, name, None)
        if solver_class is None:
            continue
        try:
            return solver_class(model)
        except Exception as exc:  # pragma: no cover - runtime/driver dependent
            errors.append(f"{name}: {exc}")
    raise RuntimeError("No supported Newton solver could be initialized: " + "; ".join(errors))


def _as_numpy(value: Any) -> Any:
    import numpy as np
    return value.numpy() if hasattr(value, "numpy") else np.asarray(value)


def _output_joint_state(state: Any, built: ElasticModelBuild) -> tuple[Any, Any]:
    import numpy as np
    q = _as_numpy(state.joint_q).reshape(-1)
    dq = _as_numpy(state.joint_qd).reshape(-1)
    q_output = np.asarray([q[built.dof_index[name].motor_q] + q[built.dof_index[name].elastic_q] for name in built.active_joint_names])
    dq_output = np.asarray([dq[built.dof_index[name].motor_qd] + dq[built.dof_index[name].elastic_qd] for name in built.active_joint_names])
    return q_output, dq_output


def _motor_joint_state(state: Any, built: ElasticModelBuild) -> tuple[Any, Any]:
    """Read physical motor-side coordinates without exposing engine ordering."""
    import numpy as np
    q = _as_numpy(state.joint_q).reshape(-1)
    dq = _as_numpy(state.joint_qd).reshape(-1)
    return (
        np.asarray([q[built.dof_index[name].motor_q] for name in built.active_joint_names]),
        np.asarray([dq[built.dof_index[name].motor_qd] for name in built.active_joint_names]),
    )


def _require_newton() -> tuple[Any, Any]:
    try:
        import warp as wp
        import newton
    except ImportError as exc:  # pragma: no cover - depends on optional runtime
        raise ImportError(
            "Generic Newton simulation requires NVIDIA Warp and Newton. "
            "Asset discovery and configuration can be used without them."
        ) from exc
    return wp, newton


def _small_inertia(wp: Any, mass: float, side: float) -> Any:
    diagonal = mass * side * side / 6.0
    return wp.mat33(diagonal, 0.0, 0.0, 0.0, diagonal, 0.0, 0.0, 0.0, diagonal)


def _copy_bodies(source: Any, target: Any, wp: Any, newton: Any, mass: float, side: float) -> tuple[dict[int, int], dict[int, tuple[Any, Any, float]]]:
    body_map = {-1: -1}
    body_props: dict[int, tuple[Any, Any, float]] = {}
    flags = getattr(source, "body_flags", [0] * source.body_count)
    body_flags = getattr(newton, "BodyFlags", None)
    kinematic_flag = int(body_flags.KINEMATIC) if body_flags is not None else 0
    for index, label in enumerate(source.body_label):
        is_kinematic = bool(flags[index] & kinematic_flag)
        body_mass = source.body_mass[index]
        body_inertia = source.body_inertia[index]
        if not is_kinematic and body_mass <= 0.0:
            body_mass = mass
            body_inertia = _small_inertia(wp, mass, side)
        target_index = target.add_link(
            xform=source.body_q[index], com=source.body_com[index], inertia=body_inertia,
            mass=body_mass, label=label, is_kinematic=is_kinematic,
        )
        body_map[index] = target_index
        body_props[target_index] = (source.body_com[index], body_inertia, body_mass)
    return body_map, body_props


def _restore_body_properties(target: Any, body_props: Mapping[int, tuple[Any, Any, float]]) -> None:
    # add_shape may derive mass from collision geometry; retain URDF inertials.
    for index, (com, inertia, mass) in body_props.items():
        target.body_com[index] = com
        target.body_inertia[index] = inertia
        target.body_mass[index] = mass


def _copy_shapes(source: Any, target: Any, newton: Any, body_map: Mapping[int, int]) -> None:
    shape_map: dict[int, int] = {}
    flags_enum = getattr(newton, "ShapeFlags", None)
    for index in range(source.shape_count):
        flags = source.shape_flags[index]
        config = target.ShapeConfig(
            ke=source.shape_material_ke[index], kd=source.shape_material_kd[index],
            kf=source.shape_material_kf[index], ka=source.shape_material_ka[index],
            mu=source.shape_material_mu[index], restitution=source.shape_material_restitution[index],
            mu_torsional=source.shape_material_mu_torsional[index], mu_rolling=source.shape_material_mu_rolling[index],
            kh=source.shape_material_kh[index], margin=source.shape_margin[index], gap=source.shape_gap[index],
            is_solid=source.shape_is_solid[index], collision_group=source.shape_collision_group[index],
            has_shape_collision=True if flags_enum is None else bool(flags & int(flags_enum.COLLIDE_SHAPES)),
            has_particle_collision=True if flags_enum is None else bool(flags & int(flags_enum.COLLIDE_PARTICLES)),
            is_visible=True if flags_enum is None else bool(flags & int(flags_enum.VISIBLE)),
        )
        shape_map[index] = target.add_shape(
            body=body_map.get(source.shape_body[index], -1), type=source.shape_type[index],
            xform=source.shape_transform[index], cfg=config, scale=source.shape_scale[index],
            src=source.shape_source[index], label=source.shape_label[index],
        )
    for first, second in source.shape_collision_filter_pairs:
        target.add_shape_collision_filter_pair(shape_map[first], shape_map[second])


def _joint_snapshot(builder: Any, index: int) -> dict[str, Any]:
    q_start, qd_start = builder.joint_q_start[index], builder.joint_qd_start[index]
    q_end = builder.joint_q_start[index + 1] if index + 1 < builder.joint_count else len(builder.joint_q)
    qd_end = builder.joint_qd_start[index + 1] if index + 1 < builder.joint_count else len(builder.joint_qd)
    axes = [
        dict(axis=builder.joint_axis[dof], target_ke=builder.joint_target_ke[dof], target_kd=builder.joint_target_kd[dof],
             limit_ke=builder.joint_limit_ke[dof], limit_kd=builder.joint_limit_kd[dof],
             limit_lower=builder.joint_limit_lower[dof], limit_upper=builder.joint_limit_upper[dof],
             target_pos=builder.joint_target_pos[dof], target_vel=builder.joint_target_vel[dof],
             effort_limit=builder.joint_effort_limit[dof], actuator_mode=builder.joint_target_mode[dof],
             armature=builder.joint_armature[dof], friction=builder.joint_friction[dof])
        for dof in range(qd_start, qd_end)
    ]
    return dict(index=index, label=builder.joint_label[index], type=builder.joint_type[index], parent=builder.joint_parent[index], child=builder.joint_child[index], parent_xform=builder.joint_X_p[index], child_xform=builder.joint_X_c[index], enabled=builder.joint_enabled[index], axes=axes, q=list(builder.joint_q[q_start:q_end]), qd=list(builder.joint_qd[qd_start:qd_end]))


def _find_snapshot_for_urdf_joint(snapshots: list[Mapping[str, Any]], urdf_name: str) -> Mapping[str, Any] | None:
    exact = [snapshot for snapshot in snapshots if str(snapshot["label"]) == urdf_name]
    namespaced = [
        snapshot for snapshot in snapshots
        if str(snapshot["label"]).replace("\\", "/").endswith("/" + urdf_name)
    ]
    matches = exact or namespaced
    if len(matches) > 1:
        raise ValueError(f"Newton imported multiple joints matching URDF joint {urdf_name!r}")
    return matches[0] if matches else None


def _add_joint_from_snapshot(
    target: Any, snapshot: Mapping[str, Any], newton: Any, wp: Any, body_map: Mapping[int, int], *,
    parent: int | None = None, child: int | None = None, parent_xform: Any | None = None, child_xform: Any | None = None,
    label: str | None = None, target_pos: float | None = None, target_vel: float | None = None,
    target_ke: float | None = None, target_kd: float | None = None, limit_lower: float | None = None,
    limit_upper: float | None = None, effort_limit: float | None = None,
) -> int:
    parent = body_map[snapshot["parent"]] if parent is None else parent
    child = body_map[snapshot["child"]] if child is None else child
    parent_xform = snapshot["parent_xform"] if parent_xform is None else parent_xform
    child_xform = snapshot["child_xform"] if child_xform is None else child_xform
    label = snapshot["label"] if label is None else label
    joint_type = snapshot["type"]
    if joint_type == newton.JointType.FIXED:
        return target.add_joint_fixed(parent=parent, child=child, parent_xform=parent_xform, child_xform=child_xform, label=label, enabled=snapshot["enabled"])
    axes = snapshot["axes"]
    if len(axes) != 1:
        raise ValueError(f"Joint {snapshot['label']!r} is not a supported 1-DoF joint")
    axis = axes[0]
    kwargs = dict(parent=parent, child=child, parent_xform=parent_xform, child_xform=child_xform, axis=axis["axis"],
                  target_pos=axis["target_pos"] if target_pos is None else target_pos,
                  target_vel=axis["target_vel"] if target_vel is None else target_vel,
                  target_ke=axis["target_ke"] if target_ke is None else target_ke,
                  target_kd=axis["target_kd"] if target_kd is None else target_kd,
                  limit_lower=axis["limit_lower"] if limit_lower is None else limit_lower,
                  limit_upper=axis["limit_upper"] if limit_upper is None else limit_upper,
                  limit_ke=axis["limit_ke"], limit_kd=axis["limit_kd"], armature=axis["armature"],
                  effort_limit=axis["effort_limit"] if effort_limit is None else effort_limit,
                  friction=axis["friction"], actuator_mode=axis["actuator_mode"], label=label, enabled=snapshot["enabled"])
    if joint_type == newton.JointType.REVOLUTE:
        return target.add_joint_revolute(**kwargs)
    if joint_type == newton.JointType.PRISMATIC:
        return target.add_joint_prismatic(**kwargs)
    raise ValueError(f"Unsupported Newton joint type for {snapshot['label']!r}")
