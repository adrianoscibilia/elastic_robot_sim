"""
Regression test for static vertical deflection under payload sweep.

This test reads the vertical elastic stiffness from config/settings.yaml and
checks that, in reference mode 0 with zero commanded motion, the measured
static Z deflection follows the expected force balance:

    deflection = (supported_mass + payload) * g / stiffness

The supported mass above elastic_joint_z with zero payload is expected to be
1.0 kg from the URDF inertials.
"""

import os
import sys

import numpy as np
import yaml
from yaml import SafeLoader


REPO_DIR = os.path.join(os.path.dirname(__file__), "..")
SCRIPTS_DIR = os.path.join(REPO_DIR, "scripts")
CONFIG_PATH = os.path.join(REPO_DIR, "config", "settings.yaml")

sys.path.insert(0, SCRIPTS_DIR)

import elastic_cart_robot_newton as robot_sim


GRAVITY = 9.81
SUPPORTED_MASS_NO_PAYLOAD = 1.0
PAYLOAD_SWEEP_KG = [0.0, 0.5, 1.0, 2.0, 3.0]
ABS_TOL_METERS = 2.0e-3
SIM_TIME_SECONDS = 6.0


def _load_settings():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=SafeLoader)


def _get_z_stiffness(settings: dict) -> float:
    """Read Z-axis stiffness from settings.yaml regardless of format."""
    drives = settings["elastic_drives"]
    # New format: elastic_drives.joints.joint_z.stiffness
    joints = drives.get("joints", {})
    if "joint_z" in joints:
        return float(joints["joint_z"]["stiffness"])
    # Current format: elastic_drives.drive_z.stiffness
    if "drive_z" in drives:
        return float(drives["drive_z"]["stiffness"])
    # Fallback: top-level defaults block
    return float(drives.get("defaults", {}).get("stiffness", 5000.0))


def _configure_robot_module(settings, payload):
    robot_sim.SIM_TIME = SIM_TIME_SECONDS
    robot_sim.TIME_STEP = settings["simulation"]["time_step"]
    robot_sim.REF_MODE = 0
    robot_sim.SEED = settings["simulation"]["seed"]
    robot_sim.CUT_OFF_TIME = 0.0
    robot_sim.PAYLOAD = float(payload)
    robot_sim.ELASTIC_DEFAULTS = settings["elastic_drives"].get(
        "defaults", {"stiffness": 5000.0, "damping": 100.0}
    )
    robot_sim.ELASTIC_JOINT_OVERRIDES = settings["elastic_drives"].get("joints", {})
    robot_sim.HEADLESS = True
    robot_sim.SAVE_CSV = False
    robot_sim.SHOW_PLOT = False


def _measure_static_deflection(payload):
    settings = _load_settings()
    _configure_robot_module(settings, payload)

    model, dof_index_map, joint_names = robot_sim.build_model()
    robot_sim._request_effort_state_attributes(model)
    state_in = model.state()
    state_out = model.state()
    control, control_target_attr = robot_sim._make_control(model)
    solver = robot_sim._new_solver(model)
    contacts = model.contacts()
    needs_ik_eval = robot_sim._configure_solver_check(solver)

    robot_sim.newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)

    motor_dof_indices = [dof_index_map[name]["motor"] for name in joint_names]
    elastic_dof_indices = [dof_index_map[name]["elastic"] for name in joint_names]

    initial_joint_q = robot_sim._get_numpy(state_in.joint_q).reshape(-1)
    initial_motor_reference = np.array(
        [float(initial_joint_q[idx]) for idx in motor_dof_indices], dtype=float,
    )

    # Find the Z-axis elastic DOF index (the last joint in the platform, "joint_z")
    z_joint_name = "joint_z"
    if z_joint_name not in dof_index_map:
        raise KeyError(f"This test requires a joint named '{z_joint_name}' in the URDF.")
    z_elastic_idx = dof_index_map[z_joint_name]["elastic"]
    initial_z_elastic = float(initial_joint_q[z_elastic_idx])

    joint_targets = np.zeros(model.joint_dof_count, dtype=np.float32)
    joint_target_velocities = np.zeros(model.joint_dof_count, dtype=np.float32)
    control.joint_f.zero_()

    z_positions = []
    z_velocities = []
    num_steps = int(np.ceil(robot_sim.SIM_TIME / robot_sim.TIME_STEP))

    for _ in range(num_steps):
        q = robot_sim._get_numpy(state_in.joint_q).reshape(-1)
        dq = robot_sim._get_numpy(state_in.joint_qd).reshape(-1)
        z_positions.append(float(q[z_elastic_idx]))
        z_velocities.append(float(dq[z_elastic_idx]))

        joint_targets[:] = 0.0
        joint_target_velocities[:] = 0.0
        for i, idx in enumerate(motor_dof_indices):
            joint_targets[idx] = initial_motor_reference[i]
        getattr(control, control_target_attr).assign(joint_targets)
        control.joint_target_vel.assign(joint_target_velocities)

        state_in.clear_forces()
        model.collide(state_in, contacts)
        solver.step(state_in, state_out, control, contacts, robot_sim.TIME_STEP)
        if needs_ik_eval:
            robot_sim.newton.eval_ik(model, state_out, state_out.joint_q, state_out.joint_qd)
        state_in, state_out = state_out, state_in

    z_positions = np.array(z_positions)
    z_velocities = np.array(z_velocities)
    equilibrium_window = slice(int(0.8 * len(z_positions)), None)
    equilibrium_mean = float(np.mean(z_positions[equilibrium_window]))
    equilibrium_std = float(np.std(z_positions[equilibrium_window]))
    equilibrium_velocity_rms = float(np.sqrt(np.mean(z_velocities[equilibrium_window] ** 2)))
    deflection = float(initial_z_elastic - equilibrium_mean)

    return {
        "payload_kg": float(payload),
        "deflection_m": deflection,
        "equilibrium_std_m": equilibrium_std,
        "equilibrium_velocity_rms_m_s": equilibrium_velocity_rms,
        "all_finite": bool(np.all(np.isfinite(z_positions)) and np.all(np.isfinite(z_velocities))),
        "stiffness_n_m": _get_z_stiffness(settings),
    }


def main():
    settings = _load_settings()
    stiffness = _get_z_stiffness(settings)

    print("Vertical deflection payload sweep")
    print("=" * 72)
    print(f"Using stiffness from settings.yaml: {stiffness:.3f} N/m")
    print(f"Payload sweep (kg): {PAYLOAD_SWEEP_KG}")
    print()
    print(
        f"{'Payload':>8} {'Measured (m)':>14} {'Expected (m)':>14} "
        f"{'Error (mm)':>12} {'Vel RMS':>12} {'Stable':>8}"
    )
    print("-" * 72)

    failures = []
    for payload in PAYLOAD_SWEEP_KG:
        result = _measure_static_deflection(payload)
        expected = (SUPPORTED_MASS_NO_PAYLOAD + payload) * GRAVITY / stiffness
        error = result["deflection_m"] - expected

        print(
            f"{payload:8.3f} {result['deflection_m']:14.6f} {expected:14.6f} "
            f"{error * 1000.0:12.3f} {result['equilibrium_velocity_rms_m_s']:12.3e} "
            f"{str(result['all_finite']):>8}"
        )

        if not result["all_finite"]:
            failures.append(f"payload={payload:.3f} kg produced non-finite state")
        if abs(error) > ABS_TOL_METERS:
            failures.append(
                f"payload={payload:.3f} kg deflection error {error:.6f} m exceeds tolerance {ABS_TOL_METERS:.6f} m"
            )
        if result["equilibrium_velocity_rms_m_s"] > 5.0e-4:
            failures.append(
                f"payload={payload:.3f} kg did not settle; equilibrium velocity RMS {result['equilibrium_velocity_rms_m_s']:.6e} m/s"
            )

    print("-" * 72)
    if failures:
        print("FAIL")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)

    print("PASS")


if __name__ == "__main__":
    main()
