"""Newton physics-based simulation runner with a programmatic API.

Public API
----------
build_model(params, ...)      -> (model, dof_index_map, joint_names)
apply_params_inplace(...)     -> bool   (True if successful)
run_rollout(...)              -> RolloutResult

The module imports warp and newton lazily (inside functions) so that the rest
of the elastic_sim package can be imported on machines without a Warp/Newton
install.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrap: make sim_common importable from scripts/
# ---------------------------------------------------------------------------
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_DIR = os.path.normpath(os.path.join(_SRC_DIR, "..", "..", "scripts"))
_URDF_DIR = os.path.normpath(os.path.join(_SRC_DIR, "..", "..", "urdf"))

if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from sim_common import (  # noqa: E402
    ENCODER_NOISE_POS,
    ENCODER_NOISE_VEL,
    ENCODER_RESOLUTION,
    TORQUE_NOISE_ABS,
    TORQUE_NOISE_REL,
    URDF_MODEL_PATH,
    _expand_simple_xacro_urdf,
    _parse_simple_xacro_properties,
)

from .params import (  # noqa: E402
    EFFECTIVE_AXIS_MASS,
    MOTOR_DAMPING,
    MOTOR_STIFFNESS,
    PAYLOAD_BOX_SIZE,
    RobotParams,
)
from .rollout import RolloutResult
from .trajectory import Trajectory

# ---------------------------------------------------------------------------
# Private helpers  (ported from elastic_cart_robot_newton.py)
# ---------------------------------------------------------------------------


def _get_numpy(array_like: object) -> np.ndarray:
    if hasattr(array_like, "numpy"):
        return array_like.numpy()
    return np.asarray(array_like)


def _make_inertia(mass: float, side: float):
    """Return a diagonal wp.mat33 inertia tensor for a uniform cube."""
    import warp as wp
    d = (mass * side * side / 6.0) if mass > 0.0 else 1e-6
    return wp.mat33(d, 0.0, 0.0, 0.0, d, 0.0, 0.0, 0.0, d)


def _transform(x: float = 0.0, y: float = 0.0, z: float = 0.0):
    import warp as wp
    return wp.transform(wp.vec3(float(x), float(y), float(z)), wp.quat_identity())


def _joint_snapshot(builder, joint_idx: int) -> dict:
    q_start = builder.joint_q_start[joint_idx]
    qd_start = builder.joint_qd_start[joint_idx]
    if joint_idx < builder.joint_count - 1:
        q_dim = builder.joint_q_start[joint_idx + 1] - q_start
        qd_dim = builder.joint_qd_start[joint_idx + 1] - qd_start
    else:
        q_dim = len(builder.joint_q) - q_start
        qd_dim = len(builder.joint_qd) - qd_start

    axes = []
    for k in range(qd_start, qd_start + qd_dim):
        axes.append({
            "axis": builder.joint_axis[k],
            "target_ke": builder.joint_target_ke[k],
            "target_kd": builder.joint_target_kd[k],
            "limit_ke": builder.joint_limit_ke[k],
            "limit_kd": builder.joint_limit_kd[k],
            "limit_lower": builder.joint_limit_lower[k],
            "limit_upper": builder.joint_limit_upper[k],
            "target_pos": builder.joint_target_pos[k],
            "target_vel": builder.joint_target_vel[k],
            "effort_limit": builder.joint_effort_limit[k],
            "actuator_mode": builder.joint_target_mode[k],
            "armature": builder.joint_armature[k],
            "friction": builder.joint_friction[k],
        })

    return {
        "index": joint_idx,
        "label": builder.joint_label[joint_idx],
        "type": builder.joint_type[joint_idx],
        "parent": builder.joint_parent[joint_idx],
        "child": builder.joint_child[joint_idx],
        "parent_xform": builder.joint_X_p[joint_idx],
        "child_xform": builder.joint_X_c[joint_idx],
        "enabled": builder.joint_enabled[joint_idx],
        "axes": axes,
        "q": list(builder.joint_q[q_start: q_start + q_dim]),
        "qd": list(builder.joint_qd[qd_start: qd_start + qd_dim]),
    }


def _copy_shape_cfg(builder, source_builder, shape_idx):
    shape_flags = source_builder.shape_flags[shape_idx]
    shape_flag_enum = getattr(source_builder, "ShapeFlags", None)
    # Prefer module-level enum
    import newton
    shape_flag_enum = getattr(newton, "ShapeFlags", shape_flag_enum)
    has_shape_col = True
    has_particle_col = True
    is_visible = True
    if shape_flag_enum is not None:
        has_shape_col = bool(shape_flags & int(shape_flag_enum.COLLIDE_SHAPES))
        has_particle_col = bool(shape_flags & int(shape_flag_enum.COLLIDE_PARTICLES))
        is_visible = bool(shape_flags & int(shape_flag_enum.VISIBLE))

    return builder.ShapeConfig(
        ke=source_builder.shape_material_ke[shape_idx],
        kd=source_builder.shape_material_kd[shape_idx],
        kf=source_builder.shape_material_kf[shape_idx],
        ka=source_builder.shape_material_ka[shape_idx],
        mu=source_builder.shape_material_mu[shape_idx],
        restitution=source_builder.shape_material_restitution[shape_idx],
        mu_torsional=source_builder.shape_material_mu_torsional[shape_idx],
        mu_rolling=source_builder.shape_material_mu_rolling[shape_idx],
        kh=source_builder.shape_material_kh[shape_idx],
        margin=source_builder.shape_margin[shape_idx],
        gap=source_builder.shape_gap[shape_idx],
        is_solid=source_builder.shape_is_solid[shape_idx],
        collision_group=source_builder.shape_collision_group[shape_idx],
        has_shape_collision=has_shape_col,
        has_particle_collision=has_particle_col,
        is_visible=is_visible,
    )


def _copy_shapes(source_builder, target_builder, body_index_map: dict) -> None:
    shape_index_map = {}
    for shape_idx in range(source_builder.shape_count):
        source_body = source_builder.shape_body[shape_idx]
        target_body = body_index_map.get(source_body, -1)
        new_shape_idx = target_builder.add_shape(
            body=target_body,
            type=source_builder.shape_type[shape_idx],
            xform=source_builder.shape_transform[shape_idx],
            cfg=_copy_shape_cfg(target_builder, source_builder, shape_idx),
            scale=source_builder.shape_scale[shape_idx],
            src=source_builder.shape_source[shape_idx],
            label=source_builder.shape_label[shape_idx],
        )
        shape_index_map[shape_idx] = new_shape_idx

    for shape_a, shape_b in source_builder.shape_collision_filter_pairs:
        if shape_a in shape_index_map and shape_b in shape_index_map:
            target_builder.add_shape_collision_filter_pair(
                shape_index_map[shape_a], shape_index_map[shape_b]
            )


def _add_joint_from_snapshot(
    target_builder,
    joint_data: dict,
    *,
    parent=None,
    child=None,
    label=None,
    joint_type=None,
    axis=None,
    target_ke=None,
    target_kd=None,
    target_pos=None,
    target_vel=None,
    limit_lower=None,
    limit_upper=None,
    actuator_mode=None,
):
    import newton

    parent = joint_data["parent"] if parent is None else parent
    child = joint_data["child"] if child is None else child
    label = joint_data["label"] if label is None else label
    joint_type = joint_data["type"] if joint_type is None else joint_type

    if joint_type == newton.JointType.FIXED:
        return target_builder.add_joint_fixed(
            parent=parent,
            child=child,
            parent_xform=joint_data["parent_xform"],
            child_xform=joint_data["child_xform"],
            label=label,
            enabled=joint_data["enabled"],
        )

    axis_data = joint_data["axes"][0] if joint_data["axes"] else {
        "axis": axis if axis is not None else newton.Axis.X,
        "target_ke": 0.0, "target_kd": 0.0,
        "limit_ke": 10000.0, "limit_kd": 10.0,
        "limit_lower": -1.0e10, "limit_upper": 1.0e10,
        "target_pos": 0.0, "target_vel": 0.0,
        "effort_limit": 1.0e6,
        "actuator_mode": newton.JointTargetMode.POSITION_VELOCITY,
        "armature": 0.0, "friction": 0.0,
    }
    axis = axis_data["axis"] if axis is None else axis
    target_ke = axis_data["target_ke"] if target_ke is None else target_ke
    target_kd = axis_data["target_kd"] if target_kd is None else target_kd
    target_pos = axis_data["target_pos"] if target_pos is None else target_pos
    target_vel = axis_data["target_vel"] if target_vel is None else target_vel
    limit_lower = axis_data["limit_lower"] if limit_lower is None else limit_lower
    limit_upper = axis_data["limit_upper"] if limit_upper is None else limit_upper
    actuator_mode = axis_data["actuator_mode"] if actuator_mode is None else actuator_mode

    common_kwargs = dict(
        parent=parent, child=child,
        parent_xform=joint_data["parent_xform"],
        child_xform=joint_data["child_xform"],
        axis=axis,
        target_pos=target_pos, target_vel=target_vel,
        target_ke=target_ke, target_kd=target_kd,
        limit_lower=limit_lower, limit_upper=limit_upper,
        limit_ke=axis_data["limit_ke"], limit_kd=axis_data["limit_kd"],
        armature=axis_data["armature"],
        effort_limit=axis_data["effort_limit"],
        friction=axis_data["friction"],
        actuator_mode=actuator_mode,
        label=label, enabled=joint_data["enabled"],
    )

    if joint_type == newton.JointType.PRISMATIC:
        return target_builder.add_joint_prismatic(**common_kwargs)
    if joint_type == newton.JointType.REVOLUTE:
        return target_builder.add_joint_revolute(**common_kwargs)
    raise ValueError(f"Unsupported joint type: {joint_type}")


def _motor_joint_kwargs(joint_data: dict) -> dict:
    import newton
    ax = joint_data["axes"][0]
    return dict(
        target_pos=0.0, target_vel=0.0,
        target_ke=MOTOR_STIFFNESS, target_kd=MOTOR_DAMPING,
        limit_lower=ax["limit_lower"], limit_upper=ax["limit_upper"],
        actuator_mode=newton.JointTargetMode.POSITION_VELOCITY,
    )


def _elastic_joint_kwargs(stiffness: float, damping: float) -> dict:
    import newton
    return dict(
        target_pos=0.0, target_vel=0.0,
        target_ke=stiffness, target_kd=damping,
        actuator_mode=newton.JointTargetMode.POSITION_VELOCITY,
    )


def _add_payload_at_ee(
    target_builder,
    body_label_to_index: dict,
    articulation_joint_indices: list,
    payload: float,
) -> None:
    import warp as wp
    import newton

    if payload <= 0.0:
        return

    def by_suffix(mapping, suffix):
        for key, value in mapping.items():
            if key.endswith(f"/{suffix}") or key == suffix:
                return key, value
        raise KeyError(suffix)

    _, ee_body = by_suffix(body_label_to_index, "ee_link")
    payload_body = target_builder.add_link(
        xform=target_builder.body_q[ee_body],
        com=wp.vec3(),
        inertia=_make_inertia(float(payload), PAYLOAD_BOX_SIZE),
        mass=float(payload),
        label="payload",
        is_kinematic=False,
    )
    target_builder.add_shape_box(
        payload_body,
        hx=PAYLOAD_BOX_SIZE / 2.0,
        hy=PAYLOAD_BOX_SIZE / 2.0,
        hz=PAYLOAD_BOX_SIZE / 2.0,
    )
    # Shape addition can inject collider-derived mass — restore authoritative values.
    target_builder.body_com[payload_body] = wp.vec3()
    target_builder.body_inertia[payload_body] = _make_inertia(float(payload), PAYLOAD_BOX_SIZE)
    target_builder.body_mass[payload_body] = float(payload)
    payload_joint = target_builder.add_joint_fixed(
        parent=ee_body, child=payload_body,
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform_identity(),
        label="payload_joint", enabled=True,
    )
    articulation_joint_indices.append(payload_joint)


def _new_solver(model):
    import newton
    for solver_name in ("SolverMuJoCo", "SolverFeatherstone", "SolverSemiImplicit"):
        cls = getattr(newton.solvers, solver_name, None)
        if cls is None:
            continue
        try:
            solver = cls(model)
            return solver
        except Exception:
            continue
    raise RuntimeError("Could not initialise any supported Newton solver.")


def _configure_solver_check(solver) -> bool:
    import newton
    return not isinstance(
        solver, (newton.solvers.SolverMuJoCo, newton.solvers.SolverFeatherstone)
    )


def _make_control(model):
    control = model.control()
    if hasattr(control, "joint_target_pos"):
        return control, "joint_target_pos"
    raise RuntimeError("Newton Control has no supported joint_target_pos attribute.")


def _reference_limits_from_model(model) -> dict[str, tuple[float, float]]:
    qd_starts = _get_numpy(model.joint_qd_start).reshape(-1)
    limit_lower = _get_numpy(model.joint_limit_lower).reshape(-1)
    limit_upper = _get_numpy(model.joint_limit_upper).reshape(-1)

    limits: dict[str, tuple[float, float]] = {}
    for ax_name, joint_suffix in (("x", "joint_x"), ("y", "joint_y"), ("z", "joint_z")):
        joint_idx = None
        for idx, label in enumerate(model.joint_label):
            s = str(label)
            if s.endswith(f"/{joint_suffix}") or s == joint_suffix:
                joint_idx = idx
                break
        if joint_idx is None:
            raise KeyError(f"Motor joint not found for axis '{ax_name}'.")
        qd_idx = int(qd_starts[joint_idx])
        limits[ax_name] = (float(limit_lower[qd_idx]), float(limit_upper[qd_idx]))
    return limits


def _request_effort_state_attributes(model) -> None:
    try:
        model.request_state_attributes("mujoco:qfrc_actuator")
    except Exception:
        pass


def _get_tau_model(state, control) -> np.ndarray | None:
    mujoco_state = getattr(state, "mujoco", None)
    if mujoco_state is not None and hasattr(mujoco_state, "qfrc_actuator"):
        return _get_numpy(mujoco_state.qfrc_actuator).reshape(-1).astype(float, copy=False)
    if control is not None and hasattr(control, "joint_f") and control.joint_f is not None:
        return _get_numpy(control.joint_f).reshape(-1).astype(float, copy=False)
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_JOINT_NAMES = ["joint_x", "joint_y", "joint_z"]


def build_model(
    params: RobotParams,
    *,
    urdf_path: str | None = None,
    show_ground: bool = False,
) -> tuple:
    """Build a Newton model from RobotParams.

    Returns:
        (model, dof_index_map, joint_names)

        dof_index_map: dict mapping joint name → {"motor": dof_idx, "elastic": dof_idx}
        joint_names:   ["joint_x", "joint_y", "joint_z"]
    """
    import warp as wp
    import newton

    if urdf_path is None:
        urdf_path = URDF_MODEL_PATH

    urdf_properties = _parse_simple_xacro_properties(urdf_path)
    expanded_urdf = _expand_simple_xacro_urdf(urdf_path)

    # ---- parse URDF into a source builder --------------------------------
    source_builder = newton.ModelBuilder()
    source_builder.add_urdf(
        expanded_urdf,
        xform=_transform(0.0, 0.0, 0.0),
        floating=False,
        enable_self_collisions=False,
        ignore_inertial_definitions=False,
        collapse_fixed_joints=False,
        force_position_velocity_actuation=True,
        parse_visuals_as_colliders=False,
    )

    # ---- rebuild bodies in target builder --------------------------------
    target_builder = newton.ModelBuilder()
    body_index_map = {-1: -1}
    body_label_to_index: dict = {}
    body_dynamic_props: dict = {}

    body_flags = getattr(source_builder, "body_flags", [0] * source_builder.body_count)
    body_flag_enum = getattr(newton, "BodyFlags", None)
    kinematic_flag = int(body_flag_enum.KINEMATIC) if body_flag_enum is not None else 0

    for body_idx, body_label in enumerate(source_builder.body_label):
        is_kinematic = bool(body_flags[body_idx] & kinematic_flag)
        body_mass = source_builder.body_mass[body_idx]
        body_inertia = source_builder.body_inertia[body_idx]
        if not is_kinematic and body_mass <= 0.0:
            body_mass = 1.0e-4
            body_inertia = _make_inertia(body_mass, 0.02)
        new_body_idx = target_builder.add_link(
            xform=source_builder.body_q[body_idx],
            com=source_builder.body_com[body_idx],
            inertia=body_inertia,
            mass=body_mass,
            label=body_label,
            is_kinematic=is_kinematic,
        )
        body_index_map[body_idx] = new_body_idx
        body_label_to_index[body_label] = new_body_idx
        body_dynamic_props[new_body_idx] = (
            source_builder.body_com[body_idx],
            body_inertia,
            body_mass,
        )

    _copy_shapes(source_builder, target_builder, body_index_map)

    # Restore URDF inertials overwritten by shape-derived mass
    for body_idx, (com, inertia, mass) in body_dynamic_props.items():
        target_builder.body_com[body_idx] = com
        target_builder.body_inertia[body_idx] = inertia
        target_builder.body_mass[body_idx] = mass

    # ---- snapshot all joints from the source builder ---------------------
    joint_snapshots = {
        label: _joint_snapshot(source_builder, idx)
        for idx, label in enumerate(source_builder.joint_label)
    }

    def by_suffix(mapping, suffix):
        for key, value in mapping.items():
            if key.endswith(f"/{suffix}") or key == suffix:
                return key, value
        raise KeyError(suffix)

    jy_label, jy_data = by_suffix(joint_snapshots, "joint_y")
    jy_fixed_label, jy_fixed_data = by_suffix(joint_snapshots, "joint_y_fixed")
    jx_label, jx_data = by_suffix(joint_snapshots, "joint_x")
    jz_label, jz_data = by_suffix(joint_snapshots, "joint_z")
    flange_label, flange_data = by_suffix(joint_snapshots, "flange_joint")

    y_axis = jy_data["axes"][0]["axis"]
    x_axis = jx_data["axes"][0]["axis"]
    z_axis = jz_data["axes"][0]["axis"]

    _, link2_body = by_suffix(body_label_to_index, "link2")
    _, link3_body = by_suffix(body_label_to_index, "link3")

    # ---- add elastic intermediate bodies ---------------------------------
    elastic_x_body = target_builder.add_link(
        xform=source_builder.body_q[jz_data["parent"]],
        mass=1.0e-4, inertia=_make_inertia(1.0e-4, 0.02),
        label="elastic_x_link",
    )
    body_label_to_index["elastic_x_link"] = elastic_x_body

    elastic_z_body = target_builder.add_link(
        xform=source_builder.body_q[jz_data["child"]],
        mass=1.0e-4, inertia=_make_inertia(1.0e-4, 0.02),
        label="elastic_z_link",
    )
    body_label_to_index["elastic_z_link"] = elastic_z_body

    # ---- compute physical damping from params ----------------------------
    kx = params.drive_x.stiffness
    cx = params.drive_x.damping(EFFECTIVE_AXIS_MASS["x"])
    ky = params.drive_y.stiffness
    cy = params.drive_y.damping(EFFECTIVE_AXIS_MASS["y"])
    kz = params.drive_z.stiffness
    cz = params.drive_z.damping(EFFECTIVE_AXIS_MASS["z"])

    # ---- build joints in URDF order, injecting elastic joints -----------
    joints_to_create = []
    for joint_label in source_builder.joint_label:
        if joint_label == jy_fixed_label:
            joints_to_create.append((
                "elastic_joint_y", _add_joint_from_snapshot,
                {"joint_data": jy_fixed_data, "label": "elastic_joint_y",
                 "joint_type": newton.JointType.PRISMATIC, "axis": y_axis,
                 **_elastic_joint_kwargs(ky, cy)},
            ))
            continue

        if joint_label == jx_label:
            joints_to_create.append((
                "joint_x", _add_joint_from_snapshot,
                {"joint_data": jx_data, "label": "joint_x",
                 **_motor_joint_kwargs(jx_data)},
            ))
            joints_to_create.append((
                "elastic_joint_x", target_builder.add_joint_prismatic,
                {"parent": link2_body, "child": elastic_x_body,
                 "parent_xform": wp.transform_identity(),
                 "child_xform": wp.transform_identity(),
                 "axis": x_axis, "label": "elastic_joint_x",
                 **_elastic_joint_kwargs(kx, cx)},
            ))
            continue

        if joint_label == jz_label:
            joints_to_create.append((
                "joint_z", _add_joint_from_snapshot,
                {"joint_data": jz_data, "label": "joint_z",
                 "parent": elastic_x_body, "child": elastic_z_body,
                 **_motor_joint_kwargs(jz_data)},
            ))
            continue

        if joint_label == flange_label:
            joints_to_create.append((
                "elastic_joint_z", target_builder.add_joint_prismatic,
                {"parent": elastic_z_body, "child": link3_body,
                 "parent_xform": wp.transform_identity(),
                 "child_xform": wp.transform_identity(),
                 "axis": z_axis, "label": "elastic_joint_z",
                 **_elastic_joint_kwargs(kz, cz)},
            ))
            joints_to_create.append((
                "flange_joint", _add_joint_from_snapshot,
                {"joint_data": flange_data, "label": "flange_joint"},
            ))
            continue

        if joint_label == jy_label:
            joints_to_create.append((
                "joint_y", _add_joint_from_snapshot,
                {"joint_data": jy_data, "label": "joint_y",
                 **_motor_joint_kwargs(jy_data)},
            ))
            continue

        if joint_label.endswith("/joint_yaw") or joint_label == "joint_yaw":
            joints_to_create.append((
                "joint_yaw_fixed", _add_joint_from_snapshot,
                {"joint_data": joint_snapshots[joint_label],
                 "label": "joint_yaw_fixed",
                 "joint_type": newton.JointType.FIXED},
            ))
            continue

        joints_to_create.append((
            joint_label, _add_joint_from_snapshot,
            {"joint_data": joint_snapshots[joint_label], "label": joint_label},
        ))

    # ---- finalise joints -------------------------------------------------
    joint_index_map: dict = {}
    articulation_joint_indices: list = []
    for jlabel, jfn, kwargs in joints_to_create:
        if "joint_data" in kwargs:
            jdata = kwargs.pop("joint_data")
            if "parent" not in kwargs:
                kwargs["parent"] = body_index_map[jdata["parent"]]
            if "child" not in kwargs:
                kwargs["child"] = body_index_map[jdata["child"]]
            idx = jfn(target_builder, jdata, **kwargs)
        else:
            idx = jfn(**kwargs)
        joint_index_map[jlabel] = idx
        articulation_joint_indices.append(idx)

    _add_payload_at_ee(
        target_builder, body_label_to_index,
        articulation_joint_indices, params.payload,
    )
    target_builder.add_articulation(articulation_joint_indices, label="platform_complete")

    ground_cfg = target_builder.ShapeConfig(is_visible=show_ground)
    ground_height = -0.5 * float(urdf_properties.get("col_height", 0.0))
    target_builder.add_ground_plane(height=ground_height, cfg=ground_cfg, label="ground")

    # ---- build dof_index_map keyed by full joint name --------------------
    axis_meta = {
        "joint_x": {"motor_joint": "joint_x", "elastic_joint": "elastic_joint_x"},
        "joint_y": {"motor_joint": "joint_y", "elastic_joint": "elastic_joint_y"},
        "joint_z": {"motor_joint": "joint_z", "elastic_joint": "elastic_joint_z"},
    }
    dof_index_map: dict = {}
    for jname, labels in axis_meta.items():
        m_idx = joint_index_map[labels["motor_joint"]]
        e_idx = joint_index_map[labels["elastic_joint"]]
        dof_index_map[jname] = {
            "motor": target_builder.joint_qd_start[m_idx],
            "elastic": target_builder.joint_qd_start[e_idx],
        }

    model = target_builder.finalize()
    model.set_gravity((0.0, 0.0, -9.81))
    return model, dof_index_map, list(_JOINT_NAMES)


def apply_params_inplace(
    model,
    params: RobotParams,
    dof_index_map: dict,
) -> bool:
    """Update elastic joint stiffness/damping in the finalized model without rebuilding.

    Writes new target_ke / target_kd values at the elastic DOF indices.
    Returns True if the mutation succeeded, False otherwise.

    NOTE: Under the MuJoCo solver these gains are cached at solver-init time
    and may not take effect until the solver is recreated. For the
    SemiImplicit and Featherstone solvers, changes take effect immediately
    since those solvers read the arrays each step.
    """
    try:
        ke_arr = _get_numpy(model.joint_target_ke).copy()
        kd_arr = _get_numpy(model.joint_target_kd).copy()

        for jname, ax_name in (("joint_x", "x"), ("joint_y", "y"), ("joint_z", "z")):
            elastic_dof = int(dof_index_map[jname]["elastic"])
            ax_params = params.axis(ax_name)
            ke_arr[elastic_dof] = float(ax_params.stiffness)
            kd_arr[elastic_dof] = float(ax_params.damping(EFFECTIVE_AXIS_MASS[ax_name]))

        model.joint_target_ke.assign(ke_arr)
        model.joint_target_kd.assign(kd_arr)
        return True
    except Exception:
        return False


def run_rollout(
    model,
    dof_index_map: dict,
    trajectory: Trajectory,
    *,
    solver=None,
    noise: bool = True,
    cut_off_time: float = 0.0,
    time_step: float = 0.01,
    seed: int | None = None,
) -> RolloutResult:
    """Run one simulation episode and return a RolloutResult.

    A new state is created from the model's rest configuration each call, so
    successive calls with different params (via apply_params_inplace) are
    independent.

    Args:
        model:          Newton Model (from build_model).
        dof_index_map:  Mapping from joint name to DOF indices.
        trajectory:     Reference trajectory callable.
        solver:         Optional pre-built solver (created fresh if None).
        noise:          Whether to add sensor noise to measurements.
        cut_off_time:   Skip recording the initial transient (seconds).
        time_step:      Integration step size (s).
        seed:           Optional numpy seed for noise reproducibility.
    """
    import newton

    if seed is not None:
        np.random.seed(seed)

    _request_effort_state_attributes(model)

    state_in = model.state()
    state_out = model.state()
    control, ctrl_attr = _make_control(model)
    if solver is None:
        solver = _new_solver(model)
    contacts = model.contacts()
    needs_ik = _configure_solver_check(solver)

    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
    control.joint_f.zero_()

    joint_targets = np.zeros(model.joint_dof_count, dtype=np.float32)
    joint_target_vel = np.zeros(model.joint_dof_count, dtype=np.float32)

    sim_time = trajectory.config.effective_sim_time
    num_steps = int(np.ceil(sim_time / time_step))

    # Pre-extract DOF indices
    mx = int(dof_index_map["joint_x"]["motor"])
    my = int(dof_index_map["joint_y"]["motor"])
    mz = int(dof_index_map["joint_z"]["motor"])
    ex = int(dof_index_map["joint_x"]["elastic"])
    ey = int(dof_index_map["joint_y"]["elastic"])
    ez = int(dof_index_map["joint_z"]["elastic"])

    time_list: list = []
    ref_pos_list: list = []
    ref_vel_list: list = []
    q_motor_list: list = []
    q_link_list: list = []
    dq_motor_list: list = []
    dq_link_list: list = []
    tau_motor_list: list = []
    tau_link_list: list = []

    t = 0.0
    for _ in range(num_steps):
        q_raw = _get_numpy(state_in.joint_q).reshape(-1)
        dq_raw = _get_numpy(state_in.joint_qd).reshape(-1)

        if noise:
            q_meas = np.round(q_raw / ENCODER_RESOLUTION) * ENCODER_RESOLUTION
            q_meas += np.random.normal(0.0, ENCODER_NOISE_POS, size=q_raw.shape)
            dq_meas = dq_raw + np.random.normal(0.0, ENCODER_NOISE_VEL, size=dq_raw.shape)
        else:
            q_meas = q_raw
            dq_meas = dq_raw

        q_ref, dq_ref = trajectory(t)

        joint_targets[:] = 0.0
        joint_target_vel[:] = 0.0
        joint_targets[mx] = float(q_ref[0])
        joint_targets[my] = float(q_ref[1])
        joint_targets[mz] = float(q_ref[2])
        joint_target_vel[mx] = float(dq_ref[0])
        joint_target_vel[my] = float(dq_ref[1])
        joint_target_vel[mz] = float(dq_ref[2])
        getattr(control, ctrl_attr).assign(joint_targets)
        control.joint_target_vel.assign(joint_target_vel)

        state_in.clear_forces()
        model.collide(state_in, contacts)
        solver.step(state_in, state_out, control, contacts, time_step)
        if needs_ik:
            newton.eval_ik(model, state_out, state_out.joint_q, state_out.joint_qd)

        tau_model = _get_tau_model(state_out, control)
        if tau_model is None:
            tau_model = np.zeros(model.joint_dof_count, dtype=float)

        if noise:
            tau_meas = tau_model + np.random.normal(
                0.0,
                TORQUE_NOISE_REL * np.abs(tau_model) + TORQUE_NOISE_ABS,
                size=tau_model.shape,
            )
        else:
            tau_meas = tau_model

        t += time_step

        if t >= cut_off_time:
            time_list.append(t)
            ref_pos_list.append(q_ref.copy())
            ref_vel_list.append(dq_ref.copy())
            q_motor_list.append(np.array([q_meas[mx], q_meas[my], q_meas[mz]]))
            q_link_list.append(np.array([q_meas[ex], q_meas[ey], q_meas[ez]]))
            dq_motor_list.append(np.array([dq_meas[mx], dq_meas[my], dq_meas[mz]]))
            dq_link_list.append(np.array([dq_meas[ex], dq_meas[ey], dq_meas[ez]]))
            tau_motor_list.append(np.array([tau_meas[mx], tau_meas[my], tau_meas[mz]]))
            tau_link_list.append(np.array([tau_meas[ex], tau_meas[ey], tau_meas[ez]]))

        state_in, state_out = state_out, state_in

    return RolloutResult(
        time=np.array(time_list),
        ref_pos=np.array(ref_pos_list),
        ref_vel=np.array(ref_vel_list),
        q_motor=np.array(q_motor_list),
        q_link=np.array(q_link_list),
        dq_motor=np.array(dq_motor_list),
        dq_link=np.array(dq_link_list),
        tau_motor=np.array(tau_motor_list),
        tau_link=np.array(tau_link_list),
    )
