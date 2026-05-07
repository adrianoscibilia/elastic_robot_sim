"""Shared simulation utilities for elastic cart robot simulations.

This module contains simulator-invariant code that is imported by each
individual simulator backend (Newton, MuJoCo, Isaac Sim, etc.).
It provides:
  - Settings loading and all physical / simulation constants
  - Trajectory generators (sinusoidal and PTP)
  - Debug plot helpers
  - Sensor-noise constants
  - XACRO / URDF parsing utilities
  - CSV filename builder and column list
"""

import math
import os
import re
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from yaml import SafeLoader

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
URDF_MODEL_PATH = os.path.join(_SCRIPTS_DIR, "..", "urdf", "platform_complete.urdf")
DATA_DIR = os.path.join(_SCRIPTS_DIR, "..", "data")

# ---------------------------------------------------------------------------
# Settings loader
# ---------------------------------------------------------------------------

def load_settings(config_path: str | None = None) -> dict:
    if config_path is None:
        config_path = os.path.join(_SCRIPTS_DIR, "..", "config", "settings.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=SafeLoader)


_settings = load_settings()

# ---------------------------------------------------------------------------
# Simulation parameters (overridable at the calling module level)
# ---------------------------------------------------------------------------
SIM_TIME: float = _settings["simulation"]["sim_time"]
TIME_STEP: float = _settings["simulation"]["time_step"]
REF_MODE: int = _settings["simulation"]["ref_mode"]
SEED: int = _settings["simulation"]["seed"]
CUT_OFF_TIME: float = _settings["simulation"]["cut_off_time"]

# ---------------------------------------------------------------------------
# Elastic drive parameters (loaded from YAML)
# ---------------------------------------------------------------------------
DRIVE_X_STIFFNESS: float = _settings["elastic_drives"]["drive_x"]["stiffness"]
DRIVE_X_DAMPING_RATIO: float = _settings["elastic_drives"]["drive_x"]["damping_ratio"]
DRIVE_Y_STIFFNESS: float = _settings["elastic_drives"]["drive_y"]["stiffness"]
DRIVE_Y_DAMPING_RATIO: float = _settings["elastic_drives"]["drive_y"]["damping_ratio"]
DRIVE_Z_STIFFNESS: float = _settings["elastic_drives"]["drive_z"]["stiffness"]
DRIVE_Z_DAMPING_RATIO: float = _settings["elastic_drives"]["drive_z"]["damping_ratio"]
PAYLOAD: float = _settings["elastic_drives"]["payload"]

# ---------------------------------------------------------------------------
# Fixed model constants
# ---------------------------------------------------------------------------
PAYLOAD_BOX_SIZE: float = 0.15

MOTOR_STIFFNESS: float = 30000.0
MOTOR_DAMPING: float = 500.0
MOTOR_FORCE_LIMIT: float = 2000.0
ELASTIC_FORCE_LIMIT: float = 2000.0

MASS_LINK_X_MOTOR: float = 2.0
MASS_LINK_X: float = 2.0
MASS_LINK_Y_MOTOR: float = 1.5
MASS_LINK_Y: float = 2.0
MASS_LINK_Z_MOTOR: float = 0.5
MASS_LINK_Z: float = 0.5

# Approximate moving masses seen by each elastic axis, used to convert the
# damping ratio into a physical damping coefficient via d = 2 * zeta * sqrt(k * m).
EFFECTIVE_AXIS_MASS: dict[str, float] = {
    "x": 1.2,
    "y": 1.8,
    "z": 1.0,
}

# Computed physical damping coefficients
DRIVE_X_DAMPING: float = 2.0 * DRIVE_X_DAMPING_RATIO * math.sqrt(DRIVE_X_STIFFNESS * EFFECTIVE_AXIS_MASS["x"])
DRIVE_Y_DAMPING: float = 2.0 * DRIVE_Y_DAMPING_RATIO * math.sqrt(DRIVE_Y_STIFFNESS * EFFECTIVE_AXIS_MASS["y"])
DRIVE_Z_DAMPING: float = 2.0 * DRIVE_Z_DAMPING_RATIO * math.sqrt(DRIVE_Z_STIFFNESS * EFFECTIVE_AXIS_MASS["z"])

# Motor joint travel limits derived from URDF properties:
#   beam_lenght=4, radius=0.1  → ±(beam_lenght/2 - 2*radius) = ±1.8 m
#   z_height=2                 → ±(z_height/2) = ±1.0 m
MOTOR_JOINT_LIMIT_XY: float = 1.8
MOTOR_JOINT_LIMIT_Z: float = 1.0

# ---------------------------------------------------------------------------
# Sensor noise parameters (shared across all simulators)
# ---------------------------------------------------------------------------
ENCODER_RESOLUTION: float = 1.2e-8
ENCODER_NOISE_POS: float = 5e-6
ENCODER_NOISE_VEL: float = 1e-2
TORQUE_NOISE_REL: float = 0.03
TORQUE_NOISE_ABS: float = 0.01

# ---------------------------------------------------------------------------
# CSV output (column names are identical across simulators)
# ---------------------------------------------------------------------------
CSV_COLUMNS = [
    "t",
    "ref_x", "ref_y", "ref_z",
    "vel_x", "vel_y", "vel_z",
    "q_motor_x", "q_link_x",
    "q_motor_y", "q_link_y",
    "q_motor_z", "q_link_z",
    "dq_motor_x", "dq_link_x",
    "dq_motor_y", "dq_link_y",
    "dq_motor_z", "dq_link_z",
    "tau_motor_x", "tau_link_x",
    "tau_motor_y", "tau_link_y",
    "tau_motor_z", "tau_link_z",
]


def make_csv_filename(
    seed: int,
    ref_mode: int,
    payload: float,
    drive_x_stiffness: float,
    drive_x_damping: float,
    drive_y_stiffness: float,
    drive_y_damping: float,
    drive_z_stiffness: float,
    drive_z_damping: float,
) -> str:
    timestamp = datetime.now().strftime("%H%M%S")
    payload_str = f"{payload:.1f}kg"
    drive_x_str = f"X{drive_x_stiffness:.0f}_{drive_x_damping:.0f}"
    drive_y_str = f"Y{drive_y_stiffness:.0f}_{drive_y_damping:.0f}"
    drive_z_str = f"Z{drive_z_stiffness:.0f}_{drive_z_damping:.0f}"
    trajectory_str = f"ref{ref_mode}"
    os.makedirs(DATA_DIR, exist_ok=True)
    return os.path.join(
        DATA_DIR,
        f"d{timestamp}_s{seed}_p{payload_str}_{drive_x_str}_{drive_y_str}_{drive_z_str}_{trajectory_str}.csv",
    )

# ---------------------------------------------------------------------------
# Random seed
# ---------------------------------------------------------------------------

def set_random_seed(seed: int) -> int:
    if seed != -1:
        np.random.seed(seed)
    else:
        seed = int(np.random.randint(0, 10000))
        np.random.seed(seed)
    print(f"Random seed set to: {seed}")
    return seed

# ---------------------------------------------------------------------------
# Debug plots
# ---------------------------------------------------------------------------

def debug_kinematics(ref_pos, ref_vel, pos, vel, time):
    plt.figure(figsize=(14, 12))

    plt.subplot(3, 2, 1)
    plt.plot(time, ref_pos[:, 0], label="X ref", linestyle="dashed")
    plt.plot(time, pos[:, 0], label="X link")
    plt.title("Position - X")
    plt.ylabel("Position (m)")
    plt.legend()
    plt.grid()

    plt.subplot(3, 2, 2)
    plt.plot(time, ref_vel[:, 0], label="VX ref", linestyle="dashed")
    plt.plot(time, vel[:, 0], label="VX link")
    plt.title("Velocity - X")
    plt.ylabel("Velocity (m/s)")
    plt.legend()
    plt.grid()

    plt.subplot(3, 2, 3)
    plt.plot(time, ref_pos[:, 1], label="Y ref", linestyle="dashed")
    plt.plot(time, pos[:, 1], label="Y link")
    plt.title("Position - Y")
    plt.ylabel("Position (m)")
    plt.legend()
    plt.grid()

    plt.subplot(3, 2, 4)
    plt.plot(time, ref_vel[:, 1], label="VY ref", linestyle="dashed")
    plt.plot(time, vel[:, 1], label="VY link")
    plt.title("Velocity - Y")
    plt.ylabel("Velocity (m/s)")
    plt.legend()
    plt.grid()

    plt.subplot(3, 2, 5)
    plt.plot(time, ref_pos[:, 2], label="Z ref", linestyle="dashed")
    plt.plot(time, pos[:, 2], label="Z link")
    plt.title("Position - Z")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.legend()
    plt.grid()

    plt.subplot(3, 2, 6)
    plt.plot(time, ref_vel[:, 2], label="VZ ref", linestyle="dashed")
    plt.plot(time, vel[:, 2], label="VZ link")
    plt.title("Velocity - Z")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.legend()
    plt.grid()

    plt.tight_layout()


def debug_dynamics(tau_x, tau_y, tau_z, time, tau_x_nom, tau_y_nom, tau_z_nom, plot_nominal=False):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(time, tau_x, label="Tau X")
    if plot_nominal:
        plt.plot(time, tau_x_nom, label="Tau X (Nominal)", linestyle="dashed")
    plt.title("Torque - X")
    plt.ylabel("Torque (N*m)")
    plt.legend()
    plt.grid()

    plt.subplot(1, 3, 2)
    plt.plot(time, tau_y, label="Tau Y")
    if plot_nominal:
        plt.plot(time, tau_y_nom, label="Tau Y (Nominal)", linestyle="dashed")
    plt.title("Torque - Y")
    plt.ylabel("Torque (N*m)")
    plt.legend()
    plt.grid()

    plt.subplot(1, 3, 3)
    plt.plot(time, tau_z, label="Tau Z")
    if plot_nominal:
        plt.plot(time, tau_z_nom, label="Tau Z (Nominal)", linestyle="dashed")
    plt.title("Torque - Z")
    plt.ylabel("Torque (N*m)")
    plt.legend()
    plt.grid()

    plt.tight_layout()

# ---------------------------------------------------------------------------
# Trajectory generators
# ---------------------------------------------------------------------------

def generate_random_trajectory_params(min_pos: float = -1.0, max_pos: float = 1.0, seed=None):
    offset = (max_pos + min_pos) / 2
    amplitude_max = (max_pos - min_pos) * 0.4
    amplitude = np.random.uniform(0.2, amplitude_max)
    frequency = np.random.uniform(0.5, 3.0)
    rng = np.random.default_rng(seed)
    phase = rng.choice([np.pi / 2, -np.pi / 2])
    return amplitude, frequency, phase, offset


def generate_random_ptp_sequence(joint_limits, sim_time, step_duration, min_distance=0.2):
    n_segments = int(np.ceil(sim_time / step_duration))
    q_start = np.array([(j[0] + j[1]) / 2 for j in joint_limits], dtype=float)
    points = [q_start]

    for _ in range(n_segments):
        while True:
            q_candidate = np.array(
                [np.random.uniform(j[0], j[1]) for j in joint_limits],
                dtype=float,
            )
            if np.linalg.norm(q_candidate - points[-1]) > min_distance:
                points.append(q_candidate)
                break

    points = np.array(points)

    def trajectory(t):
        segment = int(t // step_duration)
        if segment >= n_segments:
            return points[-1], np.zeros(len(joint_limits))

        tau = (t - segment * step_duration) / step_duration
        tau = np.clip(tau, 0.0, 1.0)

        q0 = points[segment]
        q1 = points[segment + 1]

        s = 3 * tau**2 - 2 * tau**3
        ds = (6 * tau * (1 - tau)) / step_duration

        q = q0 + s * (q1 - q0)
        dq = ds * (q1 - q0)
        return q, dq

    return trajectory, points

# ---------------------------------------------------------------------------
# XACRO / URDF utilities
# ---------------------------------------------------------------------------

def _safe_eval_expr(expr: str, variables: dict[str, float]) -> float:
    scope = {"__builtins__": {}}
    scope.update(variables)
    scope["pi"] = np.pi
    return float(eval(expr, scope, {}))  # noqa: S307 – scope is fully controlled


def _expand_simple_xacro_urdf(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        xml_text = f.read()

    xml_text = re.sub(r"<\?xacro[^>]*\?>", "", xml_text)
    property_pattern = re.compile(
        r"<xacro:property\s+name\s*=\s*\"([^\"]+)\"\s+value\s*=\s*\"([^\"]+)\"\s*/?>"
    )
    properties: dict[str, float] = {}
    for name, value in property_pattern.findall(xml_text):
        expr = value.strip()
        if expr.startswith("${") and expr.endswith("}"):
            expr = expr[2:-1]
        properties[name] = _safe_eval_expr(expr, properties)

    xml_text = property_pattern.sub("", xml_text)

    def replace_expr(match: re.Match) -> str:
        value = _safe_eval_expr(match.group(1), properties)
        return f"{value:.12g}"

    xml_text = re.sub(r"\$\{([^}]+)\}", replace_expr, xml_text)
    return xml_text


def _parse_simple_xacro_properties(path: str) -> dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        xml_text = f.read()

    property_pattern = re.compile(
        r"<xacro:property\s+name\s*=\s*\"([^\"]+)\"\s+value\s*=\s*\"([^\"]+)\"\s*/?>"
    )
    properties: dict[str, float] = {}
    for name, value in property_pattern.findall(xml_text):
        expr = value.strip()
        if expr.startswith("${") and expr.endswith("}"):
            expr = expr[2:-1]
        properties[name] = _safe_eval_expr(expr, properties)
    return properties
