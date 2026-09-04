"""MuJoCo-based simulation runner with the same public API as sim_runner.py.

Public API
----------
build_model(params, ...)          -> (model, data, dof_index_map, act_index_map)
apply_params_inplace(model, data, params) -> bool   (always True for MuJoCo)
run_rollout(...)                  -> RolloutResult

The module imports mujoco lazily so the rest of the elastic_sim package can
be imported on machines without a MuJoCo install.

Notes on MuJoCo in-place mutation
----------------------------------
MuJoCo exposes model parameters as writable NumPy arrays.  Elastic stiffness
and damping, body masses, and inertias can all be updated in-place after
model creation via model.jnt_stiffness, model.dof_damping, model.body_mass,
model.body_inertia.  Calling mujoco.mj_setConst(model, data) after changes
recomputes derived quantities (e.g. composite inertias).

A payload body is always included in the generated MJCF XML (with an
effectively zero mass when payload == 0) so topology never changes and
apply_params_inplace always returns True.
"""

from __future__ import annotations

import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_DIR = os.path.normpath(os.path.join(_SRC_DIR, "..", "..", "scripts"))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from sim_common import (  # noqa: E402
    ENCODER_NOISE_POS,
    ENCODER_NOISE_VEL,
    ENCODER_RESOLUTION,
    MASS_LINK_X,
    MASS_LINK_X_MOTOR,
    MASS_LINK_Y,
    MASS_LINK_Y_MOTOR,
    MASS_LINK_Z,
    MASS_LINK_Z_MOTOR,
    MOTOR_JOINT_LIMIT_XY,
    MOTOR_JOINT_LIMIT_Z,
    TORQUE_NOISE_ABS,
    TORQUE_NOISE_REL,
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

_MOTOR_FORCE_LIMIT = 2000.0
_ELASTIC_FORCE_LIMIT = 2000.0

# ---------------------------------------------------------------------------
# MJCF generation
# ---------------------------------------------------------------------------

def _box_inertia_diag(mass: float, side: float = 0.2) -> float:
    return (mass * side * side / 6.0) if mass > 0.0 else 1e-6


def _build_mjcf_xml(params: RobotParams, time_step: float = 0.01) -> str:
    """Generate an MJCF XML string parametrised by RobotParams.

    The payload body is always included (mass ≈ 0 when payload == 0) so the
    model topology is fixed and apply_params_inplace never needs a rebuild.
    """
    kx = params.drive_x.stiffness
    cx = params.drive_x.damping(EFFECTIVE_AXIS_MASS["x"])
    ky = params.drive_y.stiffness
    cy = params.drive_y.damping(EFFECTIVE_AXIS_MASS["y"])
    kz = params.drive_z.stiffness
    cz = params.drive_z.damping(EFFECTIVE_AXIS_MASS["z"])
    motor_stiffness = float(params.motor_stiffness)
    motor_damping = float(params.motor_damping)

    # Payload: always present; use a tiny stand-in mass when payload == 0
    payload_mass = max(float(params.payload), 1e-6)
    ph = PAYLOAD_BOX_SIZE / 2.0
    pi_d = _box_inertia_diag(payload_mass, PAYLOAD_BOX_SIZE)

    def _diag(m, s=0.2):
        v = _box_inertia_diag(m, s)
        return f"{v:.6g} {v:.6g} {v:.6g}"

    return f"""<mujoco model="elastic_cart">
  <option timestep="{time_step:.6g}" gravity="0 0 -9.81" integrator="implicitfast"/>

  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1" contype="0" conaffinity="0"/>

    <!-- X axis -->
    <body name="link_x_motor">
      <joint name="joint_x_motor" type="slide" axis="1 0 0"
             range="{-MOTOR_JOINT_LIMIT_XY:.4g} {MOTOR_JOINT_LIMIT_XY:.4g}" damping="0"/>
      <inertial mass="{MASS_LINK_X_MOTOR:.4g}" pos="0 0 0"
                diaginertia="{_diag(MASS_LINK_X_MOTOR)}"/>
      <geom type="box" size="0.10 0.10 0.10" rgba="0.3 0.5 0.8 0.9" contype="0" conaffinity="0"/>

      <body name="link_x">
        <joint name="joint_x_elastic" type="slide" axis="1 0 0"
               stiffness="{kx:.6g}" damping="{cx:.6g}"/>
        <inertial mass="{MASS_LINK_X:.4g}" pos="0 0 0"
                  diaginertia="{_diag(MASS_LINK_X)}"/>
        <geom type="box" size="0.09 0.09 0.09" rgba="0.4 0.7 0.9 0.9" contype="0" conaffinity="0"/>

        <!-- Y axis -->
        <body name="link_y_motor">
          <joint name="joint_y_motor" type="slide" axis="0 1 0"
                 range="{-MOTOR_JOINT_LIMIT_XY:.4g} {MOTOR_JOINT_LIMIT_XY:.4g}" damping="0"/>
          <inertial mass="{MASS_LINK_Y_MOTOR:.4g}" pos="0 0 0"
                    diaginertia="{_diag(MASS_LINK_Y_MOTOR)}"/>
          <geom type="box" size="0.08 0.08 0.08" rgba="0.6 0.3 0.8 0.9" contype="0" conaffinity="0"/>

          <body name="link_y">
            <joint name="joint_y_elastic" type="slide" axis="0 1 0"
                   stiffness="{ky:.6g}" damping="{cy:.6g}"/>
            <inertial mass="{MASS_LINK_Y:.4g}" pos="0 0 0"
                      diaginertia="{_diag(MASS_LINK_Y)}"/>
            <geom type="box" size="0.07 0.07 0.07" rgba="0.7 0.5 0.9 0.9" contype="0" conaffinity="0"/>

            <!-- Z axis -->
            <body name="link_z_motor">
              <joint name="joint_z_motor" type="slide" axis="0 0 1"
                     range="{-MOTOR_JOINT_LIMIT_Z:.4g} {MOTOR_JOINT_LIMIT_Z:.4g}" damping="0"/>
              <inertial mass="{MASS_LINK_Z_MOTOR:.4g}" pos="0 0 0"
                        diaginertia="{_diag(MASS_LINK_Z_MOTOR)}"/>
              <geom type="box" size="0.06 0.06 0.06" rgba="0.9 0.5 0.3 0.9" contype="0" conaffinity="0"/>

              <body name="link_z">
                <joint name="joint_z_elastic" type="slide" axis="0 0 1"
                       stiffness="{kz:.6g}" damping="{cz:.6g}"/>
                <inertial mass="{MASS_LINK_Z:.4g}" pos="0 0 0"
                          diaginertia="{_diag(MASS_LINK_Z)}"/>
                <geom type="box" size="0.05 0.05 0.05" rgba="0.9 0.8 0.3 0.9" contype="0" conaffinity="0"/>

                <!-- Payload (always present; mass ≈ 0 when payload == 0) -->
                <body name="payload">
                  <inertial mass="{payload_mass:.6g}" pos="0 0 0"
                            diaginertia="{pi_d:.6g} {pi_d:.6g} {pi_d:.6g}"/>
                  <geom type="box" size="{ph:.4g} {ph:.4g} {ph:.4g}"
                        rgba="1 0 0 0.8" contype="0" conaffinity="0"/>
                </body>

              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <!--
    PD actuators: F = kp*(ctrl - q) - kd*qd
    Velocity feedforward: ctrl = pos_ref + (kd/kp)*vel_ref
  -->
  <actuator>
    <general name="act_x" joint="joint_x_motor"
             gaintype="fixed"  gainprm="{motor_stiffness:.6g}"
             biastype="affine" biasprm="0 -{motor_stiffness:.6g} -{motor_damping:.6g}"
             forcerange="-{_MOTOR_FORCE_LIMIT:.6g} {_MOTOR_FORCE_LIMIT:.6g}"/>
    <general name="act_y" joint="joint_y_motor"
             gaintype="fixed"  gainprm="{motor_stiffness:.6g}"
             biastype="affine" biasprm="0 -{motor_stiffness:.6g} -{motor_damping:.6g}"
             forcerange="-{_MOTOR_FORCE_LIMIT:.6g} {_MOTOR_FORCE_LIMIT:.6g}"/>
    <general name="act_z" joint="joint_z_motor"
             gaintype="fixed"  gainprm="{motor_stiffness:.6g}"
             biastype="affine" biasprm="0 -{motor_stiffness:.6g} -{motor_damping:.6g}"
             forcerange="-{_MOTOR_FORCE_LIMIT:.6g} {_MOTOR_FORCE_LIMIT:.6g}"/>
  </actuator>
</mujoco>"""


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _jnt_id(model, name: str) -> int:
    import mujoco
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise KeyError(f"Joint '{name}' not found in MuJoCo model.")
    return int(jid)


def _body_id(model, name: str) -> int:
    import mujoco
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise KeyError(f"Body '{name}' not found in MuJoCo model.")
    return int(bid)


def _act_id(model, name: str) -> int:
    import mujoco
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if aid < 0:
        raise KeyError(f"Actuator '{name}' not found in MuJoCo model.")
    return int(aid)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_model(
    params: RobotParams,
    *,
    time_step: float = 0.01,
) -> tuple:
    """Build a MuJoCo model from RobotParams.

    Returns:
        (model, data, dof_index_map, act_index_map)

        dof_index_map: {"joint_x": {"motor": int, "elastic": int}, ...}
            values index data.qpos / data.qvel (slide joints: same address)
        act_index_map: {"joint_x": int, "joint_y": int, "joint_z": int}
            values index data.ctrl
    """
    import mujoco

    xml = _build_mjcf_xml(params, time_step)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    axes = ("joint_x", "joint_y", "joint_z")
    mj_names = {
        "joint_x": ("joint_x_motor", "joint_x_elastic", "act_x"),
        "joint_y": ("joint_y_motor", "joint_y_elastic", "act_y"),
        "joint_z": ("joint_z_motor", "joint_z_elastic", "act_z"),
    }

    dof_index_map: dict = {}
    act_index_map: dict = {}
    for jname in axes:
        motor_name, elastic_name, act_name = mj_names[jname]
        motor_jid = _jnt_id(model, motor_name)
        elastic_jid = _jnt_id(model, elastic_name)
        dof_index_map[jname] = {
            "motor":   int(model.jnt_dofadr[motor_jid]),
            "elastic": int(model.jnt_dofadr[elastic_jid]),
        }
        act_index_map[jname] = _act_id(model, act_name)

    return model, data, dof_index_map, act_index_map


def apply_params_inplace(
    model,
    data,
    params: RobotParams,
) -> bool:
    """Update elastic stiffness/damping and payload mass without rebuilding.

    MuJoCo exposes model arrays as writable; changes take effect on the next
    mj_step call after mj_setConst() recomputes derived quantities.

    Returns True (MuJoCo always supports in-place mutation for these params).
    """
    import mujoco

    elastic_joints = {
        "joint_x_elastic": ("x", params.drive_x),
        "joint_y_elastic": ("y", params.drive_y),
        "joint_z_elastic": ("z", params.drive_z),
    }
    for jmj_name, (ax_name, ax_params) in elastic_joints.items():
        try:
            jid = _jnt_id(model, jmj_name)
        except KeyError:
            return False
        model.jnt_stiffness[jid] = float(ax_params.stiffness)
        dofadr = int(model.jnt_dofadr[jid])
        model.dof_damping[dofadr] = float(ax_params.damping(EFFECTIVE_AXIS_MASS[ax_name]))

    # Update payload mass (payload body is always present)
    try:
        bid = _body_id(model, "payload")
        payload_mass = max(float(params.payload), 1e-6)
        model.body_mass[bid] = payload_mass
        pi_d = _box_inertia_diag(payload_mass, PAYLOAD_BOX_SIZE)
        model.body_inertia[bid] = [pi_d, pi_d, pi_d]
    except KeyError:
        pass

    mujoco.mj_setConst(model, data)
    return True


def run_rollout(
    model,
    data,
    dof_index_map: dict,
    act_index_map: dict,
    trajectory: Trajectory,
    *,
    noise: bool = True,
    cut_off_time: float = 0.0,
    seed: int | None = None,
    visualize: bool = False,
    realtime_scale: float = 1.0,
) -> RolloutResult:
    """Run one MuJoCo simulation episode and return a RolloutResult.

    The model is reset to its initial state at the start of each call.
    """
    import mujoco

    if seed is not None:
        np.random.seed(seed)

    # Reset to initial state
    mujoco.mj_resetData(model, data)

    # Pre-extract DOF indices
    mx = int(dof_index_map["joint_x"]["motor"])
    my = int(dof_index_map["joint_y"]["motor"])
    mz = int(dof_index_map["joint_z"]["motor"])
    ex = int(dof_index_map["joint_x"]["elastic"])
    ey = int(dof_index_map["joint_y"]["elastic"])
    ez = int(dof_index_map["joint_z"]["elastic"])
    ax = int(act_index_map["joint_x"])
    ay = int(act_index_map["joint_y"])
    az = int(act_index_map["joint_z"])
    # Velocity-feedforward gain ratio: ctrl = pos_ref + (kd/kp)*vel_ref.
    vff = np.asarray([
        model.actuator_biasprm[ax, 2] / model.actuator_gainprm[ax, 0],
        model.actuator_biasprm[ay, 2] / model.actuator_gainprm[ay, 0],
        model.actuator_biasprm[az, 2] / model.actuator_gainprm[az, 0],
    ], dtype=float)

    time_step = float(model.opt.timestep)
    if hasattr(trajectory, "time") and isinstance(getattr(trajectory, "time"), np.ndarray):
        time_grid = np.asarray(trajectory.time, dtype=float)
    else:
        sim_time_value = getattr(trajectory, "duration", None)
        sim_time = float(sim_time_value if sim_time_value is not None else trajectory.config.effective_sim_time)
        time_grid = np.arange(0.0, sim_time + 0.5 * time_step, time_step)

    time_list: list = []
    ref_pos_list: list = []
    ref_vel_list: list = []
    q_motor_list: list = []
    q_link_list: list = []
    dq_motor_list: list = []
    dq_link_list: list = []
    tau_motor_list: list = []
    tau_link_list: list = []

    if visualize:
        from .assets import AssetRegistry
        from .visualization import MujocoVisualizer
        viewer = MujocoVisualizer(AssetRegistry.for_repository().load("fmrr_tecnobody"), trajectory).open(model, data)
    else:
        viewer = None

    # Keep the same zero-deflection initialization convention as the Newton
    # path.  For saved trajectories this is the first exact sample.
    if hasattr(trajectory, "__call__"):
        q0, dq0, _ = trajectory(0.0)
        data.qpos[mx], data.qpos[my], data.qpos[mz] = q0
        data.qpos[ex], data.qpos[ey], data.qpos[ez] = q0
        data.qvel[mx], data.qvel[my], data.qvel[mz] = dq0
        data.qvel[ex], data.qvel[ey], data.qvel[ez] = dq0
        mujoco.mj_forward(model, data)

    for sample_index, sample_time in enumerate(time_grid):
        if viewer is not None and not viewer.is_running():
            time_grid = time_grid[:sample_index]
            break
        t = float(sample_time)
        q = data.qpos.copy()
        dq = data.qvel.copy()

        if noise:
            q_meas = np.round(q / ENCODER_RESOLUTION) * ENCODER_RESOLUTION
            q_meas += np.random.normal(0.0, ENCODER_NOISE_POS, size=q.shape)
            dq_meas = dq + np.random.normal(0.0, ENCODER_NOISE_VEL, size=dq.shape)
        else:
            q_meas = q
            dq_meas = dq

        sampled = trajectory(t)
        q_ref, dq_ref = sampled[:2]

        # PD control with velocity feedforward
        data.ctrl[ax] = float(q_ref[0]) + vff[0] * float(dq_ref[0])
        data.ctrl[ay] = float(q_ref[1]) + vff[1] * float(dq_ref[1])
        data.ctrl[az] = float(q_ref[2]) + vff[2] * float(dq_ref[2])

        mujoco.mj_step(model, data)

        # Motor torque from actuators; elastic (passive spring) torque
        tau_motor_raw = np.array([
            data.qfrc_actuator[mx],
            data.qfrc_actuator[my],
            data.qfrc_actuator[mz],
        ])
        tau_link_raw = np.array([
            data.qfrc_passive[ex],
            data.qfrc_passive[ey],
            data.qfrc_passive[ez],
        ])

        if noise:
            tau_motor = tau_motor_raw + np.random.normal(
                0.0, TORQUE_NOISE_REL * np.abs(tau_motor_raw) + TORQUE_NOISE_ABS,
                size=tau_motor_raw.shape,
            )
            tau_link = tau_link_raw + np.random.normal(
                0.0, TORQUE_NOISE_REL * np.abs(tau_link_raw) + TORQUE_NOISE_ABS,
                size=tau_link_raw.shape,
            )
        else:
            tau_motor = tau_motor_raw
            tau_link = tau_link_raw

        if t >= cut_off_time:
            time_list.append(t)
            ref_pos_list.append(q_ref.copy())
            ref_vel_list.append(dq_ref.copy())
            q_motor_list.append(np.array([q_meas[mx], q_meas[my], q_meas[mz]]))
            q_link_list.append(np.array([q_meas[ex], q_meas[ey], q_meas[ez]]))
            dq_motor_list.append(np.array([dq_meas[mx], dq_meas[my], dq_meas[mz]]))
            dq_link_list.append(np.array([dq_meas[ex], dq_meas[ey], dq_meas[ez]]))
            tau_motor_list.append(tau_motor)
            tau_link_list.append(tau_link)
        if viewer is not None:
            viewer.render(np.array([q_meas[ex], q_meas[ey], q_meas[ez]]))
            import time as time_module
            time_module.sleep(time_step / realtime_scale)
        if sample_index + 1 >= len(time_grid):
            break

    if viewer is not None:
        viewer.close()

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
