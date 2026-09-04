import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import yaml
from elastic_sim.trajectory import (
    load_trajectory_settings,
    validate_trajectory_settings,
    make_sinusoidal_trajectory,
    make_ptp_trajectory,
    make_hold_trajectory,
)

from sim_common import (  # noqa: E402
    CSV_COLUMNS,
    CUT_OFF_TIME,
    DRIVE_X_DAMPING,
    DRIVE_X_DAMPING_RATIO,
    DRIVE_X_STIFFNESS,
    DRIVE_Y_DAMPING,
    DRIVE_Y_DAMPING_RATIO,
    DRIVE_Y_STIFFNESS,
    DRIVE_Z_DAMPING,
    DRIVE_Z_DAMPING_RATIO,
    DRIVE_Z_STIFFNESS,
    EFFECTIVE_AXIS_MASS,
    ELASTIC_FORCE_LIMIT,
    ENCODER_NOISE_POS,
    ENCODER_NOISE_VEL,
    ENCODER_RESOLUTION,
    MASS_LINK_X,
    MASS_LINK_X_MOTOR,
    MASS_LINK_Y,
    MASS_LINK_Y_MOTOR,
    MASS_LINK_Z,
    MASS_LINK_Z_MOTOR,
    MOTOR_DAMPING,
    MOTOR_FORCE_LIMIT,
    MOTOR_STIFFNESS,
    PAYLOAD,
    PAYLOAD_BOX_SIZE,
    REF_MODE,
    SEED,
    SIM_TIME,
    TIME_STEP,
    TORQUE_NOISE_ABS,
    TORQUE_NOISE_REL,
    URDF_MODEL_PATH,
    _expand_simple_xacro_urdf,
    _parse_simple_xacro_properties,
    _safe_eval_expr,
    debug_dynamics,
    debug_kinematics,
    load_settings,
    make_csv_filename,
    set_random_seed,
)

try:
    import warp as wp
except ImportError as exc:
    raise ImportError(
        "This script requires NVIDIA Warp. Install Warp and Newton before running it."
    ) from exc

try:
    import newton
except ImportError as exc:
    raise ImportError(
        "This script requires Newton Physics. Run `uv sync` from the repository root."
    ) from exc


parser = argparse.ArgumentParser(description="Elastic cart model simulation in Newton")
parser.add_argument("--csv", "--save", action="store_true", help="Save simulation data to CSV")
parser.add_argument("--plot", action="store_true", help="Show debug plots")
parser.add_argument(
    "--headless",
    action="store_true",
    help="Run simulation in headless mode (no realtime Newton viewer)",
)
parser.add_argument(
    "--show-ground",
    action="store_true",
    help="Render the ground plane in the viewer instead of hiding it.",
)
# Importing this module is supported by the test and programmatic APIs.  Do
# not consume the host process' argv during import (pytest, notebooks, and
# calibration launchers all have their own command-line arguments).
args = parser.parse_args([] if __name__ != "__main__" else None)

SAVE_CSV = args.csv
SHOW_PLOT = args.plot
HEADLESS = args.headless
SHOW_GROUND = args.show_ground

# Placeholders settable by tests / external callers for per-axis overrides.
# The simulation ignores these when empty; the elastic parameters are read from
# sim_common globals (which are loaded from config/settings.yaml).
ELASTIC_DEFAULTS: dict = {}
ELASTIC_JOINT_OVERRIDES: dict = {}


def _configure_solver_check(solver):
    """Check if solver needs explicit IK evaluation."""
    return not isinstance(solver, (newton.solvers.SolverMuJoCo, newton.solvers.SolverFeatherstone))


def _make_inertia(mass: float, side: float) -> wp.mat33:
    inertia_diag = (mass * side * side / 6.0) if mass > 0.0 else 1e-6
    return wp.mat33(
        inertia_diag,
        0.0,
        0.0,
        0.0,
        inertia_diag,
        0.0,
        0.0,
        0.0,
        inertia_diag,
    )


def _transform(x=0.0, y=0.0, z=0.0):
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
        axes.append(
            {
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
            }
        )

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
        "q": list(builder.joint_q[q_start : q_start + q_dim]),
        "qd": list(builder.joint_qd[qd_start : qd_start + qd_dim]),
    }


def _copy_shape_cfg(builder, source_builder, shape_idx):
    shape_flags = source_builder.shape_flags[shape_idx]
    shape_flag_enum = getattr(newton, "ShapeFlags", None)
    has_shape_collision = True
    has_particle_collision = True
    is_visible = True
    if shape_flag_enum is not None:
        has_shape_collision = bool(shape_flags & int(shape_flag_enum.COLLIDE_SHAPES))
        has_particle_collision = bool(shape_flags & int(shape_flag_enum.COLLIDE_PARTICLES))
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
        has_shape_collision=has_shape_collision,
        has_particle_collision=has_particle_collision,
        is_visible=is_visible,
    )


def _copy_shapes(source_builder, target_builder, body_index_map):
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
            target_builder.add_shape_collision_filter_pair(shape_index_map[shape_a], shape_index_map[shape_b])


def _add_joint_from_snapshot(target_builder, joint_data, *, parent=None, child=None, label=None, joint_type=None, axis=None, target_ke=None, target_kd=None, target_pos=None, target_vel=None, limit_lower=None, limit_upper=None, actuator_mode=None):
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
        "target_ke": 0.0,
        "target_kd": 0.0,
        "limit_ke": 10000.0,
        "limit_kd": 10.0,
        "limit_lower": -1.0e10,
        "limit_upper": 1.0e10,
        "target_pos": 0.0,
        "target_vel": 0.0,
        "effort_limit": 1.0e6,
        "actuator_mode": newton.JointTargetMode.POSITION_VELOCITY,
        "armature": 0.0,
        "friction": 0.0,
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
        parent=parent,
        child=child,
        parent_xform=joint_data["parent_xform"],
        child_xform=joint_data["child_xform"],
        axis=axis,
        target_pos=target_pos,
        target_vel=target_vel,
        target_ke=target_ke,
        target_kd=target_kd,
        limit_lower=limit_lower,
        limit_upper=limit_upper,
        limit_ke=axis_data["limit_ke"],
        limit_kd=axis_data["limit_kd"],
        armature=axis_data["armature"],
        effort_limit=axis_data["effort_limit"],
        friction=axis_data["friction"],
        actuator_mode=actuator_mode,
        label=label,
        enabled=joint_data["enabled"],
    )

    if joint_type == newton.JointType.PRISMATIC:
        return target_builder.add_joint_prismatic(**common_kwargs)
    if joint_type == newton.JointType.REVOLUTE:
        return target_builder.add_joint_revolute(**common_kwargs)

    raise ValueError(f"Unsupported joint type in platform URDF rebuild: {joint_type}")


def _motor_joint_kwargs(joint_data):
    axis_data = joint_data["axes"][0]
    return dict(
        target_pos=0.0,
        target_vel=0.0,
        target_ke=MOTOR_STIFFNESS,
        target_kd=MOTOR_DAMPING,
        limit_lower=axis_data["limit_lower"],
        limit_upper=axis_data["limit_upper"],
        actuator_mode=newton.JointTargetMode.POSITION_VELOCITY,
    )


def _elastic_joint_kwargs(stiffness, damping):
    return dict(
        target_pos=0.0,
        target_vel=0.0,
        target_ke=stiffness,
        target_kd=damping,
        actuator_mode=newton.JointTargetMode.POSITION_VELOCITY,
    )


def _configure_viewer(viewer):
    if viewer is None:
        return
    window = getattr(getattr(viewer, "renderer", None), "window", None)
    if window is not None and hasattr(window, "push_handlers"):
        window.push_handlers(on_deactivate=lambda: None)


def _close_viewer(viewer):
    if viewer is None:
        return
    try:
        viewer.close()
    except AssertionError:
        pass


def _add_payload_at_ee(target_builder, body_label_to_index, articulation_joint_indices):
    if PAYLOAD <= 0.0:
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
        inertia=_make_inertia(float(PAYLOAD), side=PAYLOAD_BOX_SIZE),
        mass=float(PAYLOAD),
        label="payload",
        is_kinematic=False,
    )
    target_builder.add_shape_box(
        payload_body,
        hx=PAYLOAD_BOX_SIZE / 2.0,
        hy=PAYLOAD_BOX_SIZE / 2.0,
        hz=PAYLOAD_BOX_SIZE / 2.0,
    )
    # Adding the payload box shape can inject collider-derived mass. Restore
    # the requested payload inertial properties so PAYLOAD stays authoritative.
    target_builder.body_com[payload_body] = wp.vec3()
    target_builder.body_inertia[payload_body] = _make_inertia(float(PAYLOAD), side=PAYLOAD_BOX_SIZE)
    target_builder.body_mass[payload_body] = float(PAYLOAD)
    payload_joint = target_builder.add_joint_fixed(
        parent=ee_body,
        child=payload_body,
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform_identity(),
        label="payload_joint",
        enabled=True,
    )
    articulation_joint_indices.append(payload_joint)
    print(f"Attached payload of {PAYLOAD:.3f} kg to ee_link.")


def _build_platform_from_urdf():
    urdf_properties = _parse_simple_xacro_properties(URDF_MODEL_PATH)
    expanded_urdf = _expand_simple_xacro_urdf(URDF_MODEL_PATH)

    source_builder = newton.ModelBuilder()
    source_builder.add_urdf(
        expanded_urdf,
        xform=_transform(0.0, 0.0, 0.0),
        floating=False,
        enable_self_collisions=False,
        # Use the URDF inertials as the source of truth for body masses.
        # If these are ignored, Newton derives mass from the visual geometry,
        # which makes the long cylinders unrealistically heavy.
        ignore_inertial_definitions=False,
        collapse_fixed_joints=False,
        force_position_velocity_actuation=True,
        # Keep visuals from contributing collider-derived mass. The URDF
        # inertials should define body mass, especially for static sag tuning.
        parse_visuals_as_colliders=False,
    )

    target_builder = newton.ModelBuilder()
    body_index_map = {-1: -1}
    body_label_to_index = {}
    body_dynamic_props = {}

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

    # Copying shapes can cause Newton to accumulate collider-derived mass onto
    # the target bodies. Restore the URDF inertial properties so the model
    # dynamics are driven by the URDF masses instead of by visual geometry.
    for body_idx, (body_com, body_inertia, body_mass) in body_dynamic_props.items():
        target_builder.body_com[body_idx] = body_com
        target_builder.body_inertia[body_idx] = body_inertia
        target_builder.body_mass[body_idx] = body_mass

    joint_snapshots = {
        label: _joint_snapshot(source_builder, idx) for idx, label in enumerate(source_builder.joint_label)
    }

    def by_suffix(mapping, suffix):
        for key, value in mapping.items():
            if key.endswith(f"/{suffix}") or key == suffix:
                return key, value
        raise KeyError(suffix)

    joint_y_label, joint_y_data = by_suffix(joint_snapshots, "joint_y")
    joint_y_fixed_label, joint_y_fixed_data = by_suffix(joint_snapshots, "joint_y_fixed")
    joint_x_label, joint_x_data = by_suffix(joint_snapshots, "joint_x")
    joint_z_label, joint_z_data = by_suffix(joint_snapshots, "joint_z")
    flange_joint_label, flange_joint_data = by_suffix(joint_snapshots, "flange_joint")

    joint_y_axis = joint_y_data["axes"][0]["axis"]
    joint_x_axis = joint_x_data["axes"][0]["axis"]
    joint_z_axis = joint_z_data["axes"][0]["axis"]

    _, link2_body = by_suffix(body_label_to_index, "link2")
    _, link3_body = by_suffix(body_label_to_index, "link3")

    elastic_x_body = target_builder.add_link(
        xform=source_builder.body_q[joint_z_data["parent"]],
        mass=1.0e-4,
        inertia=_make_inertia(1.0e-4, 0.02),
        label="elastic_x_link",
    )
    body_label_to_index["elastic_x_link"] = elastic_x_body

    elastic_z_body = target_builder.add_link(
        xform=source_builder.body_q[joint_z_data["child"]],
        mass=1.0e-4,
        inertia=_make_inertia(1.0e-4, 0.02),
        label="elastic_z_link",
    )
    body_label_to_index["elastic_z_link"] = elastic_z_body

    joints_to_create = []
    for joint_label in source_builder.joint_label:
        if joint_label == joint_y_fixed_label:
            joints_to_create.append(
                (
                    "elastic_joint_y",
                    _add_joint_from_snapshot,
                    {
                        "joint_data": joint_y_fixed_data,
                        "label": "elastic_joint_y",
                        "joint_type": newton.JointType.PRISMATIC,
                        "axis": joint_y_axis,
                        **_elastic_joint_kwargs(DRIVE_Y_STIFFNESS, DRIVE_Y_DAMPING),
                    },
                )
            )
            continue

        if joint_label == joint_x_label:
            joints_to_create.append(
                (
                    "joint_x",
                    _add_joint_from_snapshot,
                    {
                        "joint_data": joint_x_data,
                        "label": "joint_x",
                        **_motor_joint_kwargs(joint_x_data),
                    },
                )
            )
            joints_to_create.append(
                (
                    "elastic_joint_x",
                    target_builder.add_joint_prismatic,
                    {
                        "parent": link2_body,
                        "child": elastic_x_body,
                        "parent_xform": wp.transform_identity(),
                        "child_xform": wp.transform_identity(),
                        "axis": joint_x_axis,
                        "label": "elastic_joint_x",
                        **_elastic_joint_kwargs(DRIVE_X_STIFFNESS, DRIVE_X_DAMPING),
                    },
                )
            )
            continue

        if joint_label == joint_z_label:
            joints_to_create.append(
                (
                    "joint_z",
                    _add_joint_from_snapshot,
                    {
                        "joint_data": joint_z_data,
                        "label": "joint_z",
                        "parent": elastic_x_body,
                        "child": elastic_z_body,
                        **_motor_joint_kwargs(joint_z_data),
                    },
                )
            )
            continue

        if joint_label == flange_joint_label:
            joints_to_create.append(
                (
                    "elastic_joint_z",
                    target_builder.add_joint_prismatic,
                    {
                        "parent": elastic_z_body,
                        "child": link3_body,
                        "parent_xform": wp.transform_identity(),
                        "child_xform": wp.transform_identity(),
                        "axis": joint_z_axis,
                        "label": "elastic_joint_z",
                        **_elastic_joint_kwargs(DRIVE_Z_STIFFNESS, DRIVE_Z_DAMPING),
                    },
                )
            )
            joints_to_create.append(
                (
                    "flange_joint",
                    _add_joint_from_snapshot,
                    {
                        "joint_data": flange_joint_data,
                        "label": "flange_joint",
                    },
                )
            )
            continue

        if joint_label == joint_y_label:
            joints_to_create.append(
                (
                    "joint_y",
                    _add_joint_from_snapshot,
                    {
                        "joint_data": joint_y_data,
                        "label": "joint_y",
                        **_motor_joint_kwargs(joint_y_data),
                    },
                )
            )
            continue

        if joint_label.endswith("/joint_yaw") or joint_label == "joint_yaw":
            joints_to_create.append(
                (
                    "joint_yaw_fixed",
                    _add_joint_from_snapshot,
                    {
                        "joint_data": joint_snapshots[joint_label],
                        "label": "joint_yaw_fixed",
                        "joint_type": newton.JointType.FIXED,
                    },
                )
            )
            continue

        joints_to_create.append(
            (
                joint_label,
                _add_joint_from_snapshot,
                {
                    "joint_data": joint_snapshots[joint_label],
                    "label": joint_label,
                },
            )
        )

    joint_index_map = {}
    articulation_joint_indices = []
    for joint_label, joint_fn, kwargs in joints_to_create:
        if "joint_data" in kwargs:
            joint_data = kwargs.pop("joint_data")
            if "parent" not in kwargs:
                kwargs["parent"] = body_index_map[joint_data["parent"]]
            if "child" not in kwargs:
                kwargs["child"] = body_index_map[joint_data["child"]]
            joint_idx = joint_fn(target_builder, joint_data, **kwargs)
        else:
            joint_idx = joint_fn(**kwargs)
        joint_index_map[joint_label] = joint_idx
        articulation_joint_indices.append(joint_idx)

    _add_payload_at_ee(target_builder, body_label_to_index, articulation_joint_indices)
    target_builder.add_articulation(articulation_joint_indices, label="fmrr_tecnobody")
    ground_cfg = target_builder.ShapeConfig(is_visible=SHOW_GROUND)
    ground_height = -0.5 * float(urdf_properties.get("col_height", 0.0))
    target_builder.add_ground_plane(height=ground_height, cfg=ground_cfg, label="ground")

    axis_meta = {
        "joint_x": {"motor_joint": "joint_x", "elastic_joint": "elastic_joint_x"},
        "joint_y": {"motor_joint": "joint_y", "elastic_joint": "elastic_joint_y"},
        "joint_z": {"motor_joint": "joint_z", "elastic_joint": "elastic_joint_z"},
    }

    dof_index_map = {}
    for joint_name, labels in axis_meta.items():
        motor_joint_idx = joint_index_map[labels["motor_joint"]]
        elastic_joint_idx = joint_index_map[labels["elastic_joint"]]
        dof_index_map[joint_name] = {
            "motor": target_builder.joint_qd_start[motor_joint_idx],
            "elastic": target_builder.joint_qd_start[elastic_joint_idx],
        }

    model = target_builder.finalize()
    model.set_gravity((0.0, 0.0, -9.81))
    joint_names = ["joint_x", "joint_y", "joint_z"]
    return model, dof_index_map, joint_names


def _new_solver(model):
    solver_attempts = [
        ("SolverMuJoCo", {}),
        ("SolverFeatherstone", {}),
        ("SolverSemiImplicit", {}),
    ]
    errors = []
    for solver_name, kwargs in solver_attempts:
        solver_cls = getattr(newton.solvers, solver_name, None)
        if solver_cls is None:
            continue
        try:
            solver = solver_cls(model, **kwargs)
            print(f"Using Newton solver: {solver_name}")
            return solver
        except Exception as exc:  # pragma: no cover - fallback path depends on local install
            errors.append(f"{solver_name}: {exc}")

    raise RuntimeError(
        "Could not initialize any supported Newton solver. Tried: "
        + "; ".join(errors)
    )


def _new_viewer():
    viewer_cls = getattr(getattr(newton, "viewer", None), "ViewerGL", None)
    if viewer_cls is None:
        return None
    return viewer_cls()


def _make_control(model):
    control = model.control()
    if hasattr(control, "joint_target_pos"):
        return control, "joint_target_pos"
    raise RuntimeError("Newton Control has no supported joint target position attribute.")


def _get_numpy(array_like):
    if hasattr(array_like, "numpy"):
        return array_like.numpy()
    return np.asarray(array_like)


def _reference_limits_from_model(model) -> dict[str, tuple[float, float]]:
    qd_starts = _get_numpy(model.joint_qd_start).reshape(-1)
    limit_lower = _get_numpy(model.joint_limit_lower).reshape(-1)
    limit_upper = _get_numpy(model.joint_limit_upper).reshape(-1)

    limits = {}
    for axis_name, joint_suffix in (("x", "joint_x"), ("y", "joint_y"), ("z", "joint_z")):
        joint_idx = None
        for idx, label in enumerate(model.joint_label):
            label_str = str(label)
            if label_str.endswith(f"/{joint_suffix}") or label_str == joint_suffix:
                joint_idx = idx
                break
        if joint_idx is None:
            raise KeyError(f"Could not find motor joint limits for axis '{axis_name}'.")

        qd_idx = int(qd_starts[joint_idx])
        limits[axis_name] = (float(limit_lower[qd_idx]), float(limit_upper[qd_idx]))

    return limits


def _request_effort_state_attributes(model):
    try:
        model.request_state_attributes("mujoco:qfrc_actuator")
    except Exception:
        pass


def _get_tau_model(state, control):
    mujoco_state = getattr(state, "mujoco", None)
    if mujoco_state is not None and hasattr(mujoco_state, "qfrc_actuator"):
        return _get_numpy(mujoco_state.qfrc_actuator).reshape(-1).astype(float, copy=False)
    if control is not None and hasattr(control, "joint_f") and control.joint_f is not None:
        return _get_numpy(control.joint_f).reshape(-1).astype(float, copy=False)
    return None


def build_model():
    return _build_platform_from_urdf()


def main():
    seed = set_random_seed(SEED)
    model, dof_index_map, _joint_names = build_model()
    reference_limits = _reference_limits_from_model(model)
    _request_effort_state_attributes(model)

    state_in = model.state()
    state_out = model.state()
    control, control_target_attr = _make_control(model)
    solver = _new_solver(model)
    viewer = None if HEADLESS else _new_viewer()
    contacts = model.contacts()
    needs_ik_eval = _configure_solver_check(solver)

    _configure_viewer(viewer)
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)

    if SAVE_CSV:
        data = []
    if SHOW_PLOT:
        time_history = []
        ref_pos_plt = []
        ref_vel_plt = []
        pos_plt = []
        vel_plt = []
        tau_plt = []
        tau_nom_plt = []

    stiffness_vector = np.array(
        [DRIVE_X_STIFFNESS, DRIVE_Y_STIFFNESS, DRIVE_Z_STIFFNESS],
        dtype=float,
    )
    damping_vector = np.array(
        [DRIVE_X_DAMPING, DRIVE_Y_DAMPING, DRIVE_Z_DAMPING],
        dtype=float,
    )

    initial_joint_q = _get_numpy(state_in.joint_q).reshape(-1)
    initial_motor_reference = np.array(
        [
            float(initial_joint_q[dof_index_map["joint_x"]["motor"]]),
            float(initial_joint_q[dof_index_map["joint_y"]["motor"]]),
            float(initial_joint_q[dof_index_map["joint_z"]["motor"]]),
        ],
        dtype=float,
    )

    # Build reference trajectory from settings.yaml (no magic numbers)
    _raw_settings = load_settings()
    tset = load_trajectory_settings(_raw_settings)
    validate_trajectory_settings(tset)
    if REF_MODE == 0:
        _traj_obj = make_hold_trajectory(initial_motor_reference, SIM_TIME, tset)
        print("\nReference mode 0: holding the initial motor position for the full simulation.")
        print(
            "  Hold position:"
            f" x={initial_motor_reference[0]:.3f},"
            f" y={initial_motor_reference[1]:.3f},"
            f" z={initial_motor_reference[2]:.3f}\n"
        )
    elif REF_MODE == 1:
        _traj_obj = make_sinusoidal_trajectory(tset, SIM_TIME, seed, time_step=TIME_STEP)
        print(f"\nSinusoidal trajectory: peak={_traj_obj.config.executed_peak_velocity_ms:.3f} m/s  "
              f"factor={_traj_obj.config.global_speed_factor:.3f}\n")
    elif REF_MODE == 2:
        _traj_obj = make_ptp_trajectory(tset, SIM_TIME, seed, time_step=TIME_STEP)
        print(f"\nPTP trajectory: peak={_traj_obj.config.executed_peak_velocity_ms:.3f} m/s  "
              f"factor={_traj_obj.config.global_speed_factor:.3f}\n")
    else:
        raise ValueError(f"Invalid ref_mode: {REF_MODE}")

    joint_targets = np.zeros(model.joint_dof_count, dtype=np.float32)
    joint_target_velocities = np.zeros(model.joint_dof_count, dtype=np.float32)
    control.joint_f.zero_()

    time = 0.0
    num_steps = int(np.ceil(SIM_TIME / TIME_STEP))

    if viewer is not None:
        viewer.set_model(model)

    for _ in range(num_steps):
        q = _get_numpy(state_in.joint_q).reshape(-1)
        dq = _get_numpy(state_in.joint_qd).reshape(-1)

        q_meas = np.round(q / ENCODER_RESOLUTION) * ENCODER_RESOLUTION
        q_meas += np.random.normal(0.0, ENCODER_NOISE_POS, size=q.shape)
        dq_meas = dq + np.random.normal(0.0, ENCODER_NOISE_VEL, size=dq.shape)
        q_motor = np.array(
            [
                q_meas[dof_index_map["joint_x"]["motor"]],
                q_meas[dof_index_map["joint_y"]["motor"]],
                q_meas[dof_index_map["joint_z"]["motor"]],
            ],
            dtype=float,
        )
        q_link = np.array(
            [
                q_meas[dof_index_map["joint_x"]["elastic"]],
                q_meas[dof_index_map["joint_y"]["elastic"]],
                q_meas[dof_index_map["joint_z"]["elastic"]],
            ],
            dtype=float,
        )
        dq_motor = np.array(
            [
                dq_meas[dof_index_map["joint_x"]["motor"]],
                dq_meas[dof_index_map["joint_y"]["motor"]],
                dq_meas[dof_index_map["joint_z"]["motor"]],
            ],
            dtype=float,
        )
        dq_link = np.array(
            [
                dq_meas[dof_index_map["joint_x"]["elastic"]],
                dq_meas[dof_index_map["joint_y"]["elastic"]],
                dq_meas[dof_index_map["joint_z"]["elastic"]],
            ],
            dtype=float,
        )

        _q_ref, _dq_ref = _traj_obj(time)
        ref_x, ref_y, ref_z = _q_ref
        vel_x, vel_y, vel_z = _dq_ref

        joint_targets[:] = 0.0
        joint_target_velocities[:] = 0.0
        joint_targets[dof_index_map["joint_x"]["motor"]] = ref_x
        joint_targets[dof_index_map["joint_y"]["motor"]] = ref_y
        joint_targets[dof_index_map["joint_z"]["motor"]] = ref_z
        joint_target_velocities[dof_index_map["joint_x"]["motor"]] = vel_x
        joint_target_velocities[dof_index_map["joint_y"]["motor"]] = vel_y
        joint_target_velocities[dof_index_map["joint_z"]["motor"]] = vel_z
        getattr(control, control_target_attr).assign(joint_targets)
        control.joint_target_vel.assign(joint_target_velocities)

        tau_nominal = stiffness_vector * (np.array([ref_x, ref_y, ref_z]) - q_link) + damping_vector * (np.array([vel_x, vel_y, vel_z]) - dq_link)

        state_in.clear_forces()
        model.collide(state_in, contacts)
        solver.step(state_in, state_out, control, contacts, TIME_STEP)
        if needs_ik_eval:
            newton.eval_ik(model, state_out, state_out.joint_q, state_out.joint_qd)

        tau_model = _get_tau_model(state_out, control)
        if tau_model is None:
            tau_model = np.zeros(model.joint_dof_count, dtype=float)
        tau_meas = tau_model + np.random.normal(
            0.0,
            TORQUE_NOISE_REL * np.abs(tau_model) + TORQUE_NOISE_ABS,
            size=tau_model.shape,
        )
        tau_motor = np.array(
            [
                tau_meas[dof_index_map["joint_x"]["motor"]],
                tau_meas[dof_index_map["joint_y"]["motor"]],
                tau_meas[dof_index_map["joint_z"]["motor"]],
            ],
            dtype=float,
        )
        tau_link = np.array(
            [
                tau_meas[dof_index_map["joint_x"]["elastic"]],
                tau_meas[dof_index_map["joint_y"]["elastic"]],
                tau_meas[dof_index_map["joint_z"]["elastic"]],
            ],
            dtype=float,
        )

        if viewer is not None:
            viewer.begin_frame(time)
            viewer.log_state(state_out)
            viewer.end_frame()

        time += TIME_STEP

        if time >= CUT_OFF_TIME and SAVE_CSV:
            data.append(
                [
                    time,
                    ref_x,
                    ref_y,
                    ref_z,
                    vel_x,
                    vel_y,
                    vel_z,
                    q_motor[0],
                    q_link[0],
                    q_motor[1],
                    q_link[1],
                    q_motor[2],
                    q_link[2],
                    dq_motor[0],
                    dq_link[0],
                    dq_motor[1],
                    dq_link[1],
                    dq_motor[2],
                    dq_link[2],
                    tau_motor[0],
                    tau_link[0],
                    tau_motor[1],
                    tau_link[1],
                    tau_motor[2],
                    tau_link[2],
                ]
            )

        if time >= CUT_OFF_TIME and SHOW_PLOT:
            time_history.append(time)
            ref_pos_plt.append([ref_x, ref_y, ref_z])
            ref_vel_plt.append([vel_x, vel_y, vel_z])
            pos_plt.append(q_motor + q_link)
            vel_plt.append(dq_motor + dq_link)
            tau_plt.append(tau_link)
            tau_nom_plt.append(tau_nominal)

        state_in, state_out = state_out, state_in

    if SAVE_CSV:
        df = pd.DataFrame(data, columns=CSV_COLUMNS)
        csv_filename = make_csv_filename(
            seed, REF_MODE, PAYLOAD,
            DRIVE_X_STIFFNESS, DRIVE_X_DAMPING,
            DRIVE_Y_STIFFNESS, DRIVE_Y_DAMPING,
            DRIVE_Z_STIFFNESS, DRIVE_Z_DAMPING,
        )
        df.to_csv(csv_filename, index=False)
        print(f"Elastic dataset saved as {csv_filename}")

    if SHOW_PLOT:
        debug_kinematics(
            np.array(ref_pos_plt),
            np.array(ref_vel_plt),
            np.array(pos_plt),
            np.array(vel_plt),
            time_history,
        )
        debug_dynamics(
            np.array(tau_plt)[:, 0],
            np.array(tau_plt)[:, 1],
            np.array(tau_plt)[:, 2],
            time_history,
            np.array(tau_nom_plt)[:, 0],
            np.array(tau_nom_plt)[:, 1],
            np.array(tau_nom_plt)[:, 2],
        )
        plt.show()

    _close_viewer(viewer)


if __name__ == "__main__":
    main()
