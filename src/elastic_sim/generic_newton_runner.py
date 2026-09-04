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

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, TYPE_CHECKING
import re
import tempfile
import time as time_module

import numpy as np

from .assets import AssetSpec, load_asset_spec
from .materialized import MaterializedTrajectory
from .serial_trajectory import SerialArmTrajectory, SerialTrajectoryConfig, trajectory_evaluator

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
    intermediate_mass: float | None = None
    link_mass: float | None = None
    intermediate_inertia_x: float | None = None
    intermediate_inertia_y: float | None = None
    intermediate_inertia_z: float | None = None

    def __post_init__(self) -> None:
        for field_name in ("stiffness", "damping", "motor_stiffness", "motor_damping"):
            value = getattr(self, field_name)
            if value < 0.0:
                raise ValueError(f"{field_name} must be non-negative")
        if self.effort_limit is not None and self.effort_limit <= 0.0:
            raise ValueError("effort_limit must be positive when supplied")
        if self.intermediate_mass is not None and self.intermediate_mass <= 0.0:
            raise ValueError("intermediate_mass must be positive when supplied")
        if self.link_mass is not None and self.link_mass <= 0.0:
            raise ValueError("link_mass must be positive when supplied")
        for field_name in ("intermediate_inertia_x", "intermediate_inertia_y", "intermediate_inertia_z"):
            value = getattr(self, field_name)
            if value is not None and value <= 0.0:
                raise ValueError(f"{field_name} must be positive when supplied")


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
        intermediate_size: float = 0.20,
        contact_budget: int = 10_000,
        body_overrides: Mapping[str, Mapping[str, float]] | None = None,
    ) -> None:
        self.asset = asset
        self.transmissions = dict(transmissions)
        self.gravity = gravity
        self.intermediate_mass = float(intermediate_mass)
        self.intermediate_size = float(intermediate_size)
        self.config_body_overrides = dict(body_overrides or {})
        self.config_contact_budget = int(contact_budget)
        if self.intermediate_mass <= 0.0 or self.intermediate_size <= 0.0:
            raise ValueError("intermediate_mass and intermediate_size must be positive")
        if self.config_contact_budget < 1:
            raise ValueError("contact_budget must be positive")
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
        # Newton does not resolve ROS package:// URI references.  Materialise
        # a short-lived URDF with absolute paths only while importing; the
        # asset itself remains portable and contains no machine-specific path.
        with _loader_urdf(self.asset) as urdf_path:
            source.add_urdf(
                str(urdf_path),
                xform=base_xform,
                floating=False,
                enable_self_collisions=self.asset.self_collisions,
                ignore_inertial_definitions=False,
                collapse_fixed_joints=False,
                force_position_velocity_actuation=True,
                parse_visuals_as_colliders=False,
            )
        target = newton.ModelBuilder()
        # Mesh-rich industrial arms can legitimately generate more than
        # Newton's small default contact allocation during self-collision.
        target.num_rigid_contacts_per_world = int(self.config_contact_budget)
        body_map, body_props = _copy_bodies(source, target, wp, newton, self.intermediate_mass, self.intermediate_size, self.config_body_overrides)
        _copy_shapes(source, target, newton, body_map)
        _restore_body_properties(target, body_props)

        snapshots = [_joint_snapshot(source, index) for index in range(source.joint_count)]
        # ``ModelBuilder.body_q`` is only a seed pose.  In particular the
        # Newton URDF importer stores a serial arm's zero-pose offsets in its
        # joint anchors rather than accumulating them into body_q.  The
        # inserted free transmission bodies do need a world pose, so recover
        # the zero-configuration forward kinematics from those anchors.
        source_body_q = _zero_configuration_body_poses(source, snapshots, wp)
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
            _override_child_link_mass(
                target, body_props, body_map[snapshot["child"]], params.link_mass
            )
            intermediate_mass = self.intermediate_mass if params.intermediate_mass is None else params.intermediate_mass
            intermediate = target.add_link(
                # The intermediate body's zero pose is the original joint
                # frame, not the child body's origin.  In a ModelBuilder
                # imported from URDF, ``body_q`` is not forward-kinematics
                # initialized: offsets live in ``joint_X_p``.  Use the
                # parent-side anchor so the replacement motor joint and the
                # elastic joint begin with exactly coincident anchors.
                xform=wp.transform_multiply(source_body_q[snapshot["parent"]], snapshot["parent_xform"])
                if snapshot["parent"] >= 0 else snapshot["parent_xform"],
                com=wp.vec3(),
                inertia=_small_inertia(wp, intermediate_mass, self.intermediate_size, (
                    params.intermediate_inertia_x, params.intermediate_inertia_y, params.intermediate_inertia_z
                )),
                mass=intermediate_mass,
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
                actuator_mode=newton.JointTargetMode.POSITION_VELOCITY,
            )
            elastic_indices[name] = _add_joint_from_snapshot(
                target,
                snapshot,
                newton,
                wp,
                body_map,
                parent=intermediate,
                parent_xform=wp.transform_identity(),
                # Preserve the original child-link joint frame.  Dropping it
                # makes URDF links with an offset child frame start in an
                # inconsistent pose, which quickly destabilises the elastic
                # articulation.
                child_xform=snapshot["child_xform"],
                label=elastic_label,
                target_pos=0.0,
                target_vel=0.0,
                target_ke=params.stiffness,
                target_kd=params.damping,
                limit_lower=-1.0e10,
                limit_upper=1.0e10,
                effort_limit=params.effort_limit,
                actuator_mode=newton.JointTargetMode.POSITION_VELOCITY,
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
    """Compatibility open-loop motor-torque replay adapter.

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
            gravity=tuple(self.config.get("gravity", self.asset.gravity)),
            intermediate_mass=float(self.config.get("intermediate_mass", 0.10)),
            intermediate_size=float(self.config.get("intermediate_size", 0.20)),
            body_overrides=self.config.get("body_overrides", {}),
        )
        _wp, newton = _require_newton()
        model = built.model
        state_in, state_out = model.state(), model.state()
        control = model.control()
        if not hasattr(control, "joint_f"):
            raise RuntimeError("This Newton build does not expose Control.joint_f for torque replay")
        # Prefer the semi-implicit solver for the deliberately light
        # intermediate transmission bodies.
        solver = _new_solver(
            model, newton,
            order=("SolverSemiImplicit", "SolverFeatherstone", "SolverMuJoCo"),
        )
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


class GenericNewtonTrajectoryRunner:
    """Track a saved joint-space trajectory with any elastic asset.

    The trajectory is applied to the motor side of every transmission.  The
    returned signals use the same canonical names as the calibration dataset
    loader, allowing a generated CSV to be replayed immediately as ground
    truth or compared with a real recording.
    """

    def __init__(self, asset: AssetSpec, config: Mapping[str, Any] | None = None) -> None:
        self.asset = asset
        self.config = dict(config or {})
        self._joint_names = asset.joint_names

    def run(
        self, trajectory: SerialTrajectoryConfig | MaterializedTrajectory, *, time_step: float = 0.004,
        visualize: bool = False, realtime_scale: float = 1.0,
    ) -> Mapping[str, Any]:
        if tuple(trajectory.joint_names) != self._joint_names:
            raise ValueError(
                "Trajectory joint_names must match asset active_joints exactly; "
                f"expected {self._joint_names}, got {tuple(trajectory.joint_names)}"
            )
        if time_step <= 0.0 or realtime_scale <= 0.0:
            raise ValueError("time_step and realtime_scale must be positive")
        configured_transmissions = self.config.get("transmissions", {})
        if configured_transmissions is None:
            configured_transmissions = {}
        if not isinstance(configured_transmissions, Mapping):
            raise ValueError("transmissions must be a mapping keyed by active joint name")
        defaults = {
            f"{name}.stiffness": float(
                configured_transmissions.get(name, {}).get("stiffness", self.config.get("default_stiffness", 10_000.0))
            )
            for name in self._joint_names
        } | {
            f"{name}.damping": float(
                configured_transmissions.get(name, {}).get("damping", self.config.get("default_damping", 100.0))
            )
            for name in self._joint_names
        }
        transmissions = _transmissions_from_named_params(self._joint_names, defaults, self.config)
        built = build_elastic_model(
            self.asset,
            transmissions,
            gravity=tuple(self.config.get("gravity", self.asset.gravity)),
            # This invisible body carries motor-side lumped inertia.  Its
            # size affects only Newton's inertia proxy, never the rendering.
            # A near-zero rotational inertia destabilises revolute two-mass
            # chains under gravity.
            intermediate_mass=float(self.config.get("intermediate_mass", 0.10)),
            intermediate_size=float(self.config.get("intermediate_size", 0.20)),
            body_overrides=self.config.get("body_overrides", {}),
        )
        _wp, newton = _require_newton()
        model = built.model
        state_in, state_out = model.state(), model.state()
        control = model.control()
        target_attr = _target_position_attribute(control)
        if not hasattr(control, "joint_target_vel"):
            raise RuntimeError("This Newton build does not expose Control.joint_target_vel")
        # Match the legacy FMRR implementation: Newton's MuJoCo articulation
        # solver is the reliable first choice for these two-mass chains.
        solver = _new_solver(
            model, newton,
            order=("SolverMuJoCo", "SolverFeatherstone", "SolverSemiImplicit"),
        )
        contacts = model.contacts()
        direct_solvers = tuple(
            cls for cls in (
                getattr(newton.solvers, "SolverMuJoCo", None),
                getattr(newton.solvers, "SolverFeatherstone", None),
            ) if cls is not None
        )
        needs_ik = not isinstance(solver, direct_solvers) if direct_solvers else True
        evaluator = trajectory_evaluator(trajectory)
        q0, dq0, _ = evaluator(0.0)
        initial_q = _as_numpy(state_in.joint_q).reshape(-1)
        initial_dq = _as_numpy(state_in.joint_qd).reshape(-1)
        for index, name in enumerate(self._joint_names):
            indices = built.dof_index[name]
            initial_q[indices.motor_q] = q0[index]
            initial_q[indices.elastic_q] = 0.0
            initial_dq[indices.motor_qd] = dq0[index]
            initial_dq[indices.elastic_qd] = 0.0
        state_in.joint_q.assign(initial_q.astype("float32"))
        state_in.joint_qd.assign(initial_dq.astype("float32"))
        newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)

        time = trajectory.time.copy() if isinstance(trajectory, MaterializedTrajectory) else np.arange(0.0, trajectory.duration + 0.5 * time_step, time_step)
        targets = np.zeros(model.joint_dof_count, dtype=np.float32)
        target_velocities = np.zeros(model.joint_dof_count, dtype=np.float32)
        # Initialise the controller to the measured initial motor state before
        # the first integration step.  Leaving importer defaults (normally
        # zero) here creates an artificial impulse whenever a generated
        # trajectory starts away from zero.
        for index, name in enumerate(self._joint_names):
            indices = built.dof_index[name]
            targets[indices.motor_qd] = q0[index]
            target_velocities[indices.motor_qd] = dq0[index]
        getattr(control, target_attr).assign(targets)
        control.joint_target_vel.assign(target_velocities)
        q_ref: list[Any] = []
        dq_ref: list[Any] = []
        q_link: list[Any] = []
        dq_link: list[Any] = []
        q_motor: list[Any] = []
        dq_motor: list[Any] = []
        tau_motor: list[Any] = []
        if visualize:
            from .visualization import NewtonVisualizer
            viewer = NewtonVisualizer(self.asset, trajectory).open(model)
        else:
            viewer = None
        try:
            for sample_index, sample_time in enumerate(time):
                if viewer is not None and hasattr(viewer, "is_running") and not viewer.is_running():
                    time = time[:sample_index]
                    break
                target_q, target_dq, _ = evaluator(float(sample_time))
                link_q, _ = _output_joint_state(state_in, built)
                link_dq = _output_joint_velocity(state_in, built)
                motor_q, motor_dq = _motor_joint_state(state_in, built)
                if not all(np.isfinite(value).all() for value in (link_q, link_dq, motor_q, motor_dq)):
                    raise RuntimeError(f"Newton elastic simulation became non-finite at t={sample_time:.6f}s")
                q_link.append(link_q)
                dq_link.append(link_dq)
                q_motor.append(motor_q)
                dq_motor.append(motor_dq)
                q_ref.append(target_q)
                dq_ref.append(target_dq)
                tau_motor.append(
                    np.asarray([
                        transmissions[name].motor_stiffness * (target_q[index] - motor_q[index])
                        + transmissions[name].motor_damping * (target_dq[index] - motor_dq[index])
                        for index, name in enumerate(self._joint_names)
                    ])
                )
                if sample_index + 1 == len(time):
                    break
                targets.fill(0.0)
                target_velocities.fill(0.0)
                for index, name in enumerate(self._joint_names):
                    indices = built.dof_index[name]
                    targets[indices.motor_qd] = target_q[index]
                    target_velocities[indices.motor_qd] = target_dq[index]
                getattr(control, target_attr).assign(targets)
                control.joint_target_vel.assign(target_velocities)
                state_in.clear_forces()
                model.collide(state_in, contacts)
                step_dt = float(time[sample_index + 1] - sample_time)
                solver.step(state_in, state_out, control, contacts, step_dt)
                if needs_ik:
                    newton.eval_ik(model, state_out, state_out.joint_q, state_out.joint_qd)
                if viewer is not None:
                    viewer.render(float(sample_time), state_out, link_q)
                    time_module.sleep(time_step / realtime_scale)
                state_in, state_out = state_out, state_in
        finally:
            _close_viewer(viewer)
        return {
            "time": time[:len(q_link)],
            "q_ref": np.asarray(q_ref),
            "dq_ref": np.asarray(dq_ref),
            "q_link": np.asarray(q_link),
            "dq_link": np.asarray(dq_link),
            "q_motor": np.asarray(q_motor),
            "dq_motor": np.asarray(dq_motor),
            "tau_motor": np.asarray(tau_motor),
            "joint_names": self._joint_names,
        }


@dataclass(frozen=True)
class RigidModelBuild:
    """A directly imported URDF model with public active-joint indexing."""

    model: Any
    asset: AssetSpec
    active_joint_names: tuple[str, ...]
    q_index: dict[str, int]
    qd_index: dict[str, int]


class GenericNewtonRigidTrajectoryRunner:
    """Stable position-tracking Newton runner for portable URDF assets.

    This is the production path for synthetic trajectories and visualization.
    It intentionally retains the native rigid URDF joint structure.  The
    separate elastic-transmission runner remains available for identification
    experiments, where its stiffness/mass/time-step configuration is explicit.
    """

    def __init__(self, asset: AssetSpec, config: Mapping[str, Any] | None = None) -> None:
        self.asset = asset
        self.config = dict(config or {})
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
        built, newton = _build_rigid_model(
            self.asset, gravity=tuple(self.config.get("gravity", self.asset.gravity))
        )
        model = built.model
        state_in, state_out = model.state(), model.state()
        control = model.control()
        motor_stiffness = float(self.config.get("motor_stiffness", 3_000.0))
        motor_damping = float(self.config.get("motor_damping", 100.0))
        _set_joint_controller_gains(model, built.qd_index, motor_stiffness, motor_damping)
        target_attr = _target_position_attribute(control)
        if not hasattr(control, "joint_target_vel"):
            raise RuntimeError("This Newton build does not expose Control.joint_target_vel")
        solver = _new_solver(model, newton, order=("SolverSemiImplicit", "SolverMuJoCo", "SolverFeatherstone"))
        contacts = model.contacts()
        direct_solvers = tuple(
            cls for cls in (getattr(newton.solvers, "SolverMuJoCo", None), getattr(newton.solvers, "SolverFeatherstone", None))
            if cls is not None
        )
        needs_ik = not isinstance(solver, direct_solvers) if direct_solvers else True
        evaluator = trajectory_evaluator(trajectory)
        q0, dq0, _ = evaluator(0.0)
        initial_q = _as_numpy(state_in.joint_q).reshape(-1)
        initial_dq = _as_numpy(state_in.joint_qd).reshape(-1)
        for index, name in enumerate(self._joint_names):
            initial_q[built.q_index[name]] = q0[index]
            initial_dq[built.qd_index[name]] = dq0[index]
        state_in.joint_q.assign(initial_q.astype("float32"))
        state_in.joint_qd.assign(initial_dq.astype("float32"))
        newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)

        time_grid = trajectory.time.copy() if isinstance(trajectory, MaterializedTrajectory) else np.arange(0.0, trajectory.duration + 0.5 * time_step, time_step)
        targets = np.zeros(model.joint_dof_count, dtype=np.float32)
        target_velocities = np.zeros(model.joint_dof_count, dtype=np.float32)
        q_ref: list[Any] = []
        dq_ref: list[Any] = []
        q: list[Any] = []
        dq: list[Any] = []
        tau: list[Any] = []
        if visualize:
            from .visualization import NewtonVisualizer
            viewer = NewtonVisualizer(self.asset, trajectory).open(model)
        else:
            viewer = None
        try:
            for sample_index, sample_time in enumerate(time_grid):
                if viewer is not None and hasattr(viewer, "is_running") and not viewer.is_running():
                    time_grid = time_grid[:sample_index]
                    break
                target_q, target_dq, _ = evaluator(float(sample_time))
                state_q = _as_numpy(state_in.joint_q).reshape(-1)
                state_dq = _as_numpy(state_in.joint_qd).reshape(-1)
                measured_q = np.asarray([state_q[built.q_index[name]] for name in self._joint_names])
                measured_dq = np.asarray([state_dq[built.qd_index[name]] for name in self._joint_names])
                if not (np.isfinite(measured_q).all() and np.isfinite(measured_dq).all()):
                    raise RuntimeError(f"Newton simulation became non-finite at t={sample_time:.6f}s")
                q_ref.append(target_q)
                dq_ref.append(target_dq)
                q.append(measured_q)
                dq.append(measured_dq)
                tau.append(motor_stiffness * (target_q - measured_q) + motor_damping * (target_dq - measured_dq))
                if sample_index + 1 == len(time_grid):
                    break
                targets.fill(0.0)
                target_velocities.fill(0.0)
                for index, name in enumerate(self._joint_names):
                    targets[built.qd_index[name]] = target_q[index]
                    target_velocities[built.qd_index[name]] = target_dq[index]
                getattr(control, target_attr).assign(targets)
                control.joint_target_vel.assign(target_velocities)
                state_in.clear_forces()
                model.collide(state_in, contacts)
                step_dt = float(time_grid[sample_index + 1] - sample_time)
                solver.step(state_in, state_out, control, contacts, step_dt)
                if needs_ik:
                    newton.eval_ik(model, state_out, state_out.joint_q, state_out.joint_qd)
                if viewer is not None:
                    viewer.render(float(sample_time), state_out, measured_q)
                    time_module.sleep(time_step / realtime_scale)
                state_in, state_out = state_out, state_in
        finally:
            _close_viewer(viewer)
        result_time = time_grid[:len(q)]
        return {
            "time": result_time,
            "q_ref": np.asarray(q_ref), "dq_ref": np.asarray(dq_ref),
            "q_link": np.asarray(q), "dq_link": np.asarray(dq),
            "q_motor": np.asarray(q), "dq_motor": np.asarray(dq),
            "tau_motor": np.asarray(tau), "joint_names": self._joint_names,
        }


class GenericNewtonKinematicTrajectoryRunner:
    """Newton ViewerGL/playback runner with guaranteed finite trajectory data.

    It evaluates forward kinematics at each reference sample without invoking a
    dynamics solver.  Use it to validate descriptions, inspect trajectories,
    and make deterministic reference datasets.  ``GenericNewtonRigidTrajectoryRunner``
    and ``GenericNewtonTrajectoryRunner`` remain opt-in physics modes.
    """

    def __init__(self, asset: AssetSpec, config: Mapping[str, Any] | None = None) -> None:
        self.asset = asset
        self.config = dict(config or {})
        self._joint_names = asset.joint_names

    def run(
        self, trajectory: SerialTrajectoryConfig | MaterializedTrajectory, *, time_step: float = 0.004,
        visualize: bool = False, realtime_scale: float = 1.0,
    ) -> Mapping[str, Any]:
        if tuple(trajectory.joint_names) != self._joint_names:
            raise ValueError("Trajectory joint_names must match asset active_joints exactly")
        if time_step <= 0.0 or realtime_scale <= 0.0:
            raise ValueError("time_step and realtime_scale must be positive")
        built, newton = _build_rigid_model(self.asset, gravity=self.asset.gravity)
        state = built.model.state()
        evaluator = trajectory_evaluator(trajectory)
        time_grid = trajectory.time.copy() if isinstance(trajectory, MaterializedTrajectory) else np.arange(0.0, trajectory.duration + 0.5 * time_step, time_step)
        q_values: list[Any] = []
        dq_values: list[Any] = []
        if visualize:
            from .visualization import NewtonVisualizer
            viewer = NewtonVisualizer(self.asset, trajectory).open(built.model)
        else:
            viewer = None
        try:
            for sample_index, sample_time in enumerate(time_grid):
                if viewer is not None and hasattr(viewer, "is_running") and not viewer.is_running():
                    time_grid = time_grid[:sample_index]
                    break
                q_ref, dq_ref, _ = evaluator(float(sample_time))
                q = _as_numpy(state.joint_q).reshape(-1)
                dq = _as_numpy(state.joint_qd).reshape(-1)
                for index, name in enumerate(self._joint_names):
                    q[built.q_index[name]] = q_ref[index]
                    dq[built.qd_index[name]] = dq_ref[index]
                state.joint_q.assign(q.astype("float32"))
                state.joint_qd.assign(dq.astype("float32"))
                newton.eval_fk(built.model, state.joint_q, state.joint_qd, state)
                q_values.append(q_ref)
                dq_values.append(dq_ref)
                if viewer is not None:
                    viewer.render(float(sample_time), state, q_ref)
                    time_module.sleep(time_step / realtime_scale)
        finally:
            _close_viewer(viewer)
        q_array = np.asarray(q_values)
        dq_array = np.asarray(dq_values)
        return {
            "time": time_grid[:len(q_values)], "q_ref": q_array, "dq_ref": dq_array,
            "q_link": q_array, "dq_link": dq_array, "q_motor": q_array, "dq_motor": dq_array,
            "tau_motor": np.zeros_like(q_array), "joint_names": self._joint_names,
        }


def make_torque_replay_runner(*, asset: str | Path | AssetSpec | None = None, config: Mapping[str, Any] | None = None) -> GenericNewtonTorqueReplayRunner:
    """Compatibility factory for legacy torque-replay callers.

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
            intermediate_mass=None if defaults.get("intermediate_mass") is None else float(defaults["intermediate_mass"]),
            link_mass=None if defaults.get("link_mass") is None else float(defaults["link_mass"]),
            intermediate_inertia_x=None if defaults.get("intermediate_inertia_x") is None else float(defaults["intermediate_inertia_x"]),
            intermediate_inertia_y=None if defaults.get("intermediate_inertia_y") is None else float(defaults["intermediate_inertia_y"]),
            intermediate_inertia_z=None if defaults.get("intermediate_inertia_z") is None else float(defaults["intermediate_inertia_z"]),
        )
    return result


def _new_solver(
    model: Any, newton: Any, *, order: tuple[str, ...] = ("SolverMuJoCo", "SolverFeatherstone", "SolverSemiImplicit")
) -> Any:
    errors = []
    for name in order:
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


def _output_joint_velocity(state: Any, built: ElasticModelBuild) -> Any:
    """Read output/link-side velocities without allocating positions."""
    import numpy as np
    dq = _as_numpy(state.joint_qd).reshape(-1)
    return np.asarray([
        dq[built.dof_index[name].motor_qd] + dq[built.dof_index[name].elastic_qd]
        for name in built.active_joint_names
    ])


def _motor_joint_state(state: Any, built: ElasticModelBuild) -> tuple[Any, Any]:
    """Read physical motor-side coordinates without exposing engine ordering."""
    import numpy as np
    q = _as_numpy(state.joint_q).reshape(-1)
    dq = _as_numpy(state.joint_qd).reshape(-1)
    return (
        np.asarray([q[built.dof_index[name].motor_q] for name in built.active_joint_names]),
        np.asarray([dq[built.dof_index[name].motor_qd] for name in built.active_joint_names]),
    )


def _target_position_attribute(control: Any) -> str:
    for name in ("joint_target_pos", "joint_target_q"):
        if hasattr(control, name):
            return name
    raise RuntimeError("This Newton build has no supported joint target-position attribute")


def _new_viewer(newton: Any) -> Any | None:
    viewer_class = getattr(getattr(newton, "viewer", None), "ViewerGL", None)
    return None if viewer_class is None else viewer_class()


def _close_viewer(viewer: Any | None) -> None:
    if viewer is None:
        return
    try:
        viewer.close()
    except AssertionError:  # viewer can already be closed by its window event
        pass


def _build_rigid_model(asset: AssetSpec, *, gravity: tuple[float, float, float]) -> tuple[RigidModelBuild, Any]:
    """Import a portable URDF directly for robust native-joint tracking."""
    wp, newton = _require_newton()
    builder = newton.ModelBuilder()
    base_xform = wp.transform(wp.vec3(*asset.base_position), wp.quat(*asset.base_quaternion))
    with _loader_urdf(asset) as urdf_path:
        builder.add_urdf(
            str(urdf_path), xform=base_xform, floating=False,
            enable_self_collisions=asset.self_collisions,
            ignore_inertial_definitions=False, collapse_fixed_joints=False,
            force_position_velocity_actuation=True, parse_visuals_as_colliders=False,
        )
    snapshots = [_joint_snapshot(builder, index) for index in range(builder.joint_count)]
    by_name = {
        name: _find_snapshot_for_urdf_joint(snapshots, name)
        for name in asset.joint_names
    }
    missing = [name for name, snapshot in by_name.items() if snapshot is None]
    if missing:
        raise ValueError("Configured active URDF joints were not imported by Newton: " + ", ".join(missing))
    q_index = {name: int(builder.joint_q_start[by_name[name]["index"]]) for name in asset.joint_names}  # type: ignore[index]
    qd_index = {name: int(builder.joint_qd_start[by_name[name]["index"]]) for name in asset.joint_names}  # type: ignore[index]
    model = builder.finalize()
    model.set_gravity(gravity)
    return RigidModelBuild(model, asset, asset.joint_names, q_index, qd_index), newton


def _set_joint_controller_gains(
    model: Any, qd_index: Mapping[str, int], stiffness: float, damping: float
) -> None:
    """Override importer defaults, which are excessively stiff for replay."""
    if stiffness < 0.0 or damping < 0.0:
        raise ValueError("motor controller gains must be non-negative")
    ke = _as_numpy(model.joint_target_ke).reshape(-1).copy()
    kd = _as_numpy(model.joint_target_kd).reshape(-1).copy()
    for index in qd_index.values():
        ke[index] = stiffness
        kd[index] = damping
    model.joint_target_ke.assign(ke.astype("float32"))
    model.joint_target_kd.assign(kd.astype("float32"))


@contextmanager
def _loader_urdf(asset: AssetSpec):
    """Yield a Newton-loadable URDF whose mesh URIs are absolute local paths."""
    resources = iter(asset.validate_resources())
    text = asset.urdf_path.read_text(encoding="utf-8")
    if "${" in text:
        text = _expand_simple_xacro_text(text)
    text = _ensure_collision_geometry(text)

    def replace_mesh(match: re.Match[str]) -> str:
        try:
            resource = next(resources)
        except StopIteration as exc:  # pragma: no cover - catches malformed XML edits
            raise ValueError(f"Asset {asset.name!r} mesh references changed during preparation") from exc
        return match.group(1) + resource.as_posix() + match.group(3)

    text = re.sub(r'(<mesh\b[^>]*\bfilename\s*=\s*")([^"]+)(")', replace_mesh, text)
    try:
        next(resources)
        raise ValueError(f"Asset {asset.name!r} mesh references changed during preparation")
    except StopIteration:
        pass
    with tempfile.TemporaryDirectory(prefix="elastic_asset_urdf_") as directory:
        path = Path(directory) / asset.urdf_path.name
        path.write_text(text, encoding="utf-8")
        yield path


def _expand_simple_xacro_text(xml_text: str) -> str:
    """Expand the arithmetic-only xacro subset used by fmrr_tecnobody."""
    from .assets import _xacro_float
    text = re.sub(r"<\?xacro[^>]*\?>", "", xml_text)
    pattern = re.compile(r'<xacro:property\s+name\s*=\s*"([^"]+)"\s+value\s*=\s*"([^"]+)"\s*/?>')
    properties: dict[str, float] = {}
    for name, value in pattern.findall(text):
        properties[name] = _xacro_float(value, properties)
    text = pattern.sub("", text)
    return re.sub(r"\$\{([^}]+)\}", lambda match: f"{_xacro_float(match.group(1), properties):.12g}", text)


def _ensure_collision_geometry(xml_text: str) -> str:
    """Mirror visuals as collisions only for descriptions that have none.

    The Tecnobody FMRR description supplies visual primitives but no collision
    tags.  Keeping that exception here makes its asset-level
    ``self_collisions: true`` setting effective without changing upstream-like
    URDFs that already provide curated collision geometry.
    """
    from copy import deepcopy
    from xml.etree import ElementTree as ET
    root = ET.fromstring(xml_text)
    if root.findall(".//collision"):
        return xml_text
    for link in root.findall("link"):
        for visual in link.findall("visual"):
            collision = deepcopy(visual)
            collision.tag = "collision"
            link.append(collision)
    return ET.tostring(root, encoding="unicode")


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


def _small_inertia(wp: Any, mass: float, side: float, principal: tuple[float | None, float | None, float | None] | None = None) -> Any:
    if principal is None or any(value is None for value in principal):
        diagonal = mass * side * side / 6.0
        values = (diagonal, diagonal, diagonal)
    else:
        values = tuple(float(value) for value in principal)
    return wp.mat33(values[0], 0.0, 0.0, 0.0, values[1], 0.0, 0.0, 0.0, values[2])


def _copy_bodies(source: Any, target: Any, wp: Any, newton: Any, mass: float, side: float, overrides: Mapping[str, Mapping[str, float]] | None = None) -> tuple[dict[int, int], dict[int, tuple[Any, Any, float]]]:
    body_map = {-1: -1}
    body_props: dict[int, tuple[Any, Any, float]] = {}
    flags = getattr(source, "body_flags", [0] * source.body_count)
    body_flags = getattr(newton, "BodyFlags", None)
    kinematic_flag = int(body_flags.KINEMATIC) if body_flags is not None else 0
    for index, label in enumerate(source.body_label):
        is_kinematic = bool(flags[index] & kinematic_flag)
        body_mass = source.body_mass[index]
        body_inertia = source.body_inertia[index]
        override = _find_body_override(label, overrides or {})
        if override:
            if "mass" in override:
                body_mass = float(override["mass"])
            inertia_names = ("inertia_x", "inertia_y", "inertia_z")
            if all(name in override for name in inertia_names):
                body_inertia = _small_inertia(wp, body_mass, side, tuple(float(override[name]) for name in inertia_names))
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


def _find_body_override(label: Any, overrides: Mapping[str, Mapping[str, float]]) -> Mapping[str, float] | None:
    text = str(label).replace("\\", "/")
    for name, value in overrides.items():
        if text == str(name).replace("\\", "/") or text.endswith("/" + str(name).replace("\\", "/")):
            return value
    return None


def _restore_body_properties(target: Any, body_props: Mapping[int, tuple[Any, Any, float]]) -> None:
    # add_shape may derive mass from collision geometry; retain URDF inertials.
    for index, (com, inertia, mass) in body_props.items():
        target.body_com[index] = com
        target.body_inertia[index] = inertia
        target.body_mass[index] = mass


def _override_child_link_mass(
    target: Any,
    body_props: dict[int, tuple[Any, Any, float]],
    body_index: int,
    mass_override: float | None,
) -> None:
    """Apply an optional URDF-link mass override while preserving inertia shape."""
    if mass_override is None:
        return
    com, inertia, old_mass = body_props[body_index]
    if old_mass <= 0.0:
        raise ValueError("Cannot override a massless URDF child link")
    scale = mass_override / old_mass
    target.body_mass[body_index] = mass_override
    target.body_inertia[body_index] = inertia * scale
    body_props[body_index] = (com, inertia * scale, mass_override)


def _copy_shapes(source: Any, target: Any, newton: Any, body_map: Mapping[int, int]) -> None:
    shape_map: dict[int, int] = {}
    shapes_by_body: dict[int, list[int]] = {}
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
        shapes_by_body.setdefault(body_map.get(source.shape_body[index], -1), []).append(shape_map[index])
    for first, second in source.shape_collision_filter_pairs:
        target.add_shape_collision_filter_pair(shape_map[first], shape_map[second])
    # Self collision should test non-adjacent robot links, not the unavoidable
    # overlaps at every mechanical joint.  URDF importers do not consistently
    # emit those exclusions across Newton versions, so establish them while
    # copying the portable source model.
    for joint_index in range(source.joint_count):
        parent = body_map.get(source.joint_parent[joint_index], -1)
        child = body_map.get(source.joint_child[joint_index], -1)
        for first in shapes_by_body.get(parent, ()):
            for second in shapes_by_body.get(child, ()):
                target.add_shape_collision_filter_pair(first, second)


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


def _zero_configuration_body_poses(builder: Any, snapshots: list[Mapping[str, Any]], wp: Any) -> list[Any]:
    """Return URDF zero-configuration body poses without finalizing a model.

    Newton's builder keeps the zero-pose hierarchy in joint anchor transforms,
    while its ``body_q`` values can remain identity transforms.  That is fine
    for ordinary articulated children, but not for the standalone bodies we
    insert between motor and link joints.
    """
    poses: list[Any | None] = [None] * builder.body_count
    pending = list(snapshots)
    world = wp.transform_identity()
    while pending:
        next_pending: list[Mapping[str, Any]] = []
        made_progress = False
        for snapshot in pending:
            parent = int(snapshot["parent"])
            child = int(snapshot["child"])
            if parent >= 0 and poses[parent] is None:
                next_pending.append(snapshot)
                continue
            parent_pose = world if parent < 0 else poses[parent]
            assert parent_pose is not None
            pose = wp.transform_multiply(parent_pose, snapshot["parent_xform"])
            # At q=0 the child anchor is coincident with the parent anchor.
            pose = wp.transform_multiply(pose, wp.transform_inverse(snapshot["child_xform"]))
            if poses[child] is None:
                poses[child] = pose
            made_progress = True
        if not made_progress:
            labels = ", ".join(str(snapshot["label"]) for snapshot in next_pending)
            raise ValueError(f"Could not resolve zero-configuration body poses for joints: {labels}")
        pending = next_pending
    return [pose if pose is not None else builder.body_q[index] for index, pose in enumerate(poses)]


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
    limit_upper: float | None = None, effort_limit: float | None = None, actuator_mode: Any | None = None,
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
                  friction=axis["friction"], actuator_mode=axis["actuator_mode"] if actuator_mode is None else actuator_mode,
                  label=label, enabled=snapshot["enabled"])
    if joint_type == newton.JointType.REVOLUTE:
        return target.add_joint_revolute(**kwargs)
    if joint_type == newton.JointType.PRISMATIC:
        return target.add_joint_prismatic(**kwargs)
    raise ValueError(f"Unsupported Newton joint type for {snapshot['label']!r}")
