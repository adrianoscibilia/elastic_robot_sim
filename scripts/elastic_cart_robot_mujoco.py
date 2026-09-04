"""Elastic cart robot simulation using MuJoCo.

Model topology (mirrors the Newton/IsaacSim backends):
  world → link_x_motor (slide X, PD actuated)
        → link_x       (slide X, passive elastic spring)
        → link_y_motor (slide Y, PD actuated)
        → link_y       (slide Y, passive elastic spring)
        → link_z_motor (slide Z, PD actuated)
        → link_z       (slide Z, passive elastic spring)
        → [payload]    (fixed, optional)

Motor joints are driven by MuJoCo `general` actuators configured as
position-velocity PD controllers:
    F = kp * (ctrl − q) + kd * (vel_ref − qd)
by setting  ctrl = pos_ref + (kd/kp) * vel_ref.

Elastic joints use MuJoCo's built-in joint stiffness/damping (passive spring).

Run with:
    python elastic_cart_robot_mujoco.py [--csv] [--plot] [--headless]
"""

import argparse
import os
import sys
import time

import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
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
    DRIVE_X_STIFFNESS,
    DRIVE_Y_DAMPING,
    DRIVE_Y_STIFFNESS,
    DRIVE_Z_DAMPING,
    DRIVE_Z_STIFFNESS,
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
    debug_dynamics,
    debug_kinematics,
    load_settings,
    make_csv_filename,
    set_random_seed,
)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Elastic cart model simulation in MuJoCo")
parser.add_argument("--csv", "--save", action="store_true", help="Save simulation data to CSV")
parser.add_argument("--plot", action="store_true", help="Show debug plots")
parser.add_argument(
    "--headless",
    action="store_true",
    help="Run simulation without the interactive MuJoCo viewer",
)
args = parser.parse_args()

SAVE_CSV = args.csv
SHOW_PLOT = args.plot
HEADLESS = args.headless

# ---------------------------------------------------------------------------
# MJCF model builder
# ---------------------------------------------------------------------------

_MJCF_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "elastic_cart_mujoco.xml")


def _box_inertia_diag(mass: float, side: float = 0.2) -> float:
    """Diagonal inertia of a uniform solid cube: I = m * a² / 6."""
    return (mass * side * side / 6.0) if mass > 0.0 else 1e-6


def _load_mjcf_xml() -> str:
    """Load the MJCF template from config/ and substitute runtime parameters.

    All numeric values come from module-level constants (sim_common) so that
    tests can override them before calling build_model().
    """
    with open(_MJCF_TEMPLATE_PATH, "r", encoding="utf-8") as f:
        template = f.read()

    # Optional payload body injected at {payload_body}
    payload_body = ""
    if PAYLOAD > 0.0:
        ph = PAYLOAD_BOX_SIZE / 2.0
        pi_d = _box_inertia_diag(PAYLOAD, PAYLOAD_BOX_SIZE)
        payload_body = (
            f'<body name="payload">'
            f'<inertial mass="{PAYLOAD:.6g}" pos="0 0 0"'
            f' diaginertia="{pi_d:.6g} {pi_d:.6g} {pi_d:.6g}"/>'
            f'<geom type="box" size="{ph:.4g} {ph:.4g} {ph:.4g}"'
            f' rgba="1 0 0 1" contype="0" conaffinity="0"/>'
            f"</body>"
        )

    params = {
        "timestep":        f"{TIME_STEP:.6g}",
        "xy_range":        f"{-MOTOR_JOINT_LIMIT_XY:.4g} {MOTOR_JOINT_LIMIT_XY:.4g}",
        "z_range":         f"{-MOTOR_JOINT_LIMIT_Z:.4g} {MOTOR_JOINT_LIMIT_Z:.4g}",
        # Link masses
        "mass_link_x_motor": f"{MASS_LINK_X_MOTOR:.4g}",
        "mass_link_x":       f"{MASS_LINK_X:.4g}",
        "mass_link_y_motor": f"{MASS_LINK_Y_MOTOR:.4g}",
        "mass_link_y":       f"{MASS_LINK_Y:.4g}",
        "mass_link_z_motor": f"{MASS_LINK_Z_MOTOR:.4g}",
        "mass_link_z":       f"{MASS_LINK_Z:.4g}",
        # Inertia diagonals
        "inertia_x_motor": f"{_box_inertia_diag(MASS_LINK_X_MOTOR):.6g}",
        "inertia_x":       f"{_box_inertia_diag(MASS_LINK_X):.6g}",
        "inertia_y_motor": f"{_box_inertia_diag(MASS_LINK_Y_MOTOR):.6g}",
        "inertia_y":       f"{_box_inertia_diag(MASS_LINK_Y):.6g}",
        "inertia_z_motor": f"{_box_inertia_diag(MASS_LINK_Z_MOTOR):.6g}",
        "inertia_z":       f"{_box_inertia_diag(MASS_LINK_Z):.6g}",
        # Elastic drives
        "drive_x_stiffness": f"{DRIVE_X_STIFFNESS:.6g}",
        "drive_x_damping":   f"{DRIVE_X_DAMPING:.6g}",
        "drive_y_stiffness": f"{DRIVE_Y_STIFFNESS:.6g}",
        "drive_y_damping":   f"{DRIVE_Y_DAMPING:.6g}",
        "drive_z_stiffness": f"{DRIVE_Z_STIFFNESS:.6g}",
        "drive_z_damping":   f"{DRIVE_Z_DAMPING:.6g}",
        # Motor actuators
        "motor_kp":   f"{MOTOR_STIFFNESS:.6g}",
        "motor_kd":   f"{MOTOR_DAMPING:.6g}",
        "motor_flim": f"{MOTOR_FORCE_LIMIT:.6g}",
        # Payload
        "payload_body": payload_body,
    }

    return template.format_map(params)


# ---------------------------------------------------------------------------
# Model + DOF helpers
# ---------------------------------------------------------------------------

def _jnt_dofadr(model: mujoco.MjModel, name: str) -> int:
    jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jnt_id < 0:
        raise KeyError(f"Joint '{name}' not found in MuJoCo model.")
    return int(model.jnt_dofadr[jnt_id])


def _act_id(model: mujoco.MjModel, name: str) -> int:
    act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if act_id < 0:
        raise KeyError(f"Actuator '{name}' not found in MuJoCo model.")
    return act_id


def build_model():
    """Build and return (model, data, dof_index_map, act_index_map).

    dof_index_map mirrors the Newton backend:
        {"x": {"motor": int, "elastic": int}, "y": ..., "z": ...}

    act_index_map:
        {"x": int, "y": int, "z": int}   (indices into data.ctrl)
    """
    xml = _load_mjcf_xml()
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    dof_index_map = {
        axis: {
            "motor":   _jnt_dofadr(model, f"joint_{axis}_motor"),
            "elastic": _jnt_dofadr(model, f"joint_{axis}_elastic"),
        }
        for axis in ("x", "y", "z")
    }

    act_index_map = {
        axis: _act_id(model, f"act_{axis}")
        for axis in ("x", "y", "z")
    }

    return model, data, dof_index_map, act_index_map


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def main():
    seed = set_random_seed(SEED)
    model, data, dof_index_map, act_index_map = build_model()

    # Velocity-feedforward gain ratio: ctrl = pos_ref + (kd/kp)*vel_ref
    vff_ratio = MOTOR_DAMPING / MOTOR_STIFFNESS

    stiffness_vector = np.array([DRIVE_X_STIFFNESS, DRIVE_Y_STIFFNESS, DRIVE_Z_STIFFNESS])
    damping_vector   = np.array([DRIVE_X_DAMPING,   DRIVE_Y_DAMPING,   DRIVE_Z_DAMPING])

    # ------------------------------------------------------------------
    # Build reference trajectory from settings.yaml (no magic numbers)
    # ------------------------------------------------------------------
    _raw_settings = load_settings()
    tset = load_trajectory_settings(_raw_settings)
    validate_trajectory_settings(tset)
    if REF_MODE == 0:
        _traj_obj = make_hold_trajectory(np.zeros(3), SIM_TIME, tset)
        print("\nReference mode 0: holding zero position.\n")
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

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------
    csv_data    = []
    time_history = []
    ref_pos_plt  = []
    ref_vel_plt  = []
    pos_plt      = []
    vel_plt      = []
    tau_plt      = []
    tau_nom_plt  = []

    num_steps = int(np.ceil(SIM_TIME / TIME_STEP))

    # ------------------------------------------------------------------
    # Simulation loop (with optional passive viewer)
    # ------------------------------------------------------------------
    def _run_loop(viewer=None):
        if viewer is not None:
            # Give the viewer thread time to finish initializing the GLFW
            # window before we start querying is_running() or calling sync().
            time.sleep(0.2)

        wall_start = time.perf_counter()
        for step_idx in range(num_steps):
            # Stop early if the viewer window is closed by the user
            if viewer is not None and not viewer.is_running():
                break
            t = float(data.time)

            # ── Read & noise state ───────────────────────────────────
            q   = data.qpos.copy()
            dq  = data.qvel.copy()

            q_meas  = np.round(q / ENCODER_RESOLUTION) * ENCODER_RESOLUTION
            q_meas += np.random.normal(0.0, ENCODER_NOISE_POS, size=q.shape)
            dq_meas = dq + np.random.normal(0.0, ENCODER_NOISE_VEL, size=dq.shape)

            q_motor  = np.array([q_meas [dof_index_map[a]["motor"]]   for a in ("x", "y", "z")])
            q_link   = np.array([q_meas [dof_index_map[a]["elastic"]] for a in ("x", "y", "z")])
            dq_motor = np.array([dq_meas[dof_index_map[a]["motor"]]   for a in ("x", "y", "z")])
            dq_link  = np.array([dq_meas[dof_index_map[a]["elastic"]] for a in ("x", "y", "z")])

            # ── Reference trajectory ────────────────────────────────
            _q_ref, _dq_ref = _traj_obj(t)
            ref_x, ref_y, ref_z = _q_ref
            vel_x, vel_y, vel_z = _dq_ref

            # ── Apply PD + velocity feedforward control ─────────────
            # ctrl = pos_ref + (kd/kp) * vel_ref
            data.ctrl[act_index_map["x"]] = ref_x + vff_ratio * vel_x
            data.ctrl[act_index_map["y"]] = ref_y + vff_ratio * vel_y
            data.ctrl[act_index_map["z"]] = ref_z + vff_ratio * vel_z

            # ── Step ────────────────────────────────────────────────
            mujoco.mj_step(model, data)

            # ── Read torques ────────────────────────────────────────
            # Motor torques come from actuators; elastic torques from passive springs.
            tau_motor_raw = np.array([
                data.qfrc_actuator[dof_index_map[a]["motor"]] for a in ("x", "y", "z")
            ])
            tau_link_raw = np.array([
                data.qfrc_passive[dof_index_map[a]["elastic"]] for a in ("x", "y", "z")
            ])

            tau_meas_motor = tau_motor_raw + np.random.normal(
                0.0,
                TORQUE_NOISE_REL * np.abs(tau_motor_raw) + TORQUE_NOISE_ABS,
                size=tau_motor_raw.shape,
            )
            tau_meas_link = tau_link_raw + np.random.normal(
                0.0,
                TORQUE_NOISE_REL * np.abs(tau_link_raw) + TORQUE_NOISE_ABS,
                size=tau_link_raw.shape,
            )

            # Nominal elastic force estimate
            ref_pos = np.array([ref_x, ref_y, ref_z])
            ref_vel = np.array([vel_x, vel_y, vel_z])
            tau_nominal = stiffness_vector * (ref_pos - q_link) + damping_vector * (ref_vel - dq_link)

            # ── Viewer sync + real-time pacing ──────────────────────
            if viewer is not None:
                viewer.sync()
                # Sleep for the remainder of the current timestep so that
                # the viewer renders at roughly real-time speed.
                elapsed = time.perf_counter() - wall_start
                sim_time_target = (step_idx + 1) * TIME_STEP
                sleep_dt = sim_time_target - elapsed
                if sleep_dt > 0:
                    time.sleep(sleep_dt)

            # ── Record ──────────────────────────────────────────────
            if t >= CUT_OFF_TIME:
                if SAVE_CSV:
                    csv_data.append([
                        t,
                        ref_x, ref_y, ref_z,
                        vel_x, vel_y, vel_z,
                        q_motor[0],      q_link[0],
                        q_motor[1],      q_link[1],
                        q_motor[2],      q_link[2],
                        dq_motor[0],     dq_link[0],
                        dq_motor[1],     dq_link[1],
                        dq_motor[2],     dq_link[2],
                        tau_meas_motor[0], tau_meas_link[0],
                        tau_meas_motor[1], tau_meas_link[1],
                        tau_meas_motor[2], tau_meas_link[2],
                    ])
                if SHOW_PLOT:
                    time_history.append(t)
                    ref_pos_plt.append([ref_x, ref_y, ref_z])
                    ref_vel_plt.append([vel_x, vel_y, vel_z])
                    pos_plt.append(q_motor + q_link)
                    vel_plt.append(dq_motor + dq_link)
                    tau_plt.append(tau_meas_link)
                    tau_nom_plt.append(tau_nominal)

    # Run with or without the interactive viewer
    if HEADLESS:
        _run_loop(viewer=None)
    else:
        try:
            with mujoco.viewer.launch_passive(model, data) as viewer:
                _run_loop(viewer=viewer)
                # Simulation finished — keep the window open so the user can
                # inspect the final state.  Close the window to continue.
                print("Simulation complete. Close the viewer window to exit.")
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(0.1)
        except Exception as exc:
            print(f"MuJoCo viewer unavailable ({type(exc).__name__}: {exc}), running headless.")
            _run_loop(viewer=None)

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------
    if SAVE_CSV:
        df = pd.DataFrame(csv_data, columns=CSV_COLUMNS)
        csv_filename = make_csv_filename(
            seed, REF_MODE, PAYLOAD,
            DRIVE_X_STIFFNESS, DRIVE_X_DAMPING,
            DRIVE_Y_STIFFNESS, DRIVE_Y_DAMPING,
            DRIVE_Z_STIFFNESS, DRIVE_Z_DAMPING,
        )
        df.to_csv(csv_filename, index=False)
        print(f"Elastic dataset saved as {csv_filename}")

    # ------------------------------------------------------------------
    # Debug plots
    # ------------------------------------------------------------------
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
            plot_nominal=False,
        )
        plt.show()


if __name__ == "__main__":
    main()
