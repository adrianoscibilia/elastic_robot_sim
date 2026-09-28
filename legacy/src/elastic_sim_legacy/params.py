"""RobotParams dataclass: bounds, normalization, and YAML serialization.

Physical parameters live here; nothing Newton/Warp-specific is imported.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import yaml
from yaml import SafeLoader

_DEFAULT_SETTINGS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "config", "settings.yaml"
)

# ---------------------------------------------------------------------------
# Fixed motor joint parameters (not tunable by default)
# ---------------------------------------------------------------------------
MOTOR_STIFFNESS: float = 30000.0
MOTOR_DAMPING: float = 500.0
PAYLOAD_BOX_SIZE: float = 0.15

# Effective moving masses seen by each elastic axis; used to convert the
# damping ratio to a physical damping coefficient: d = 2*zeta*sqrt(k*m).
EFFECTIVE_AXIS_MASS: dict[str, float] = {"x": 1.2, "y": 1.8, "z": 1.0}

# ---------------------------------------------------------------------------
# Parameter bounds (matching run_dataset_generation.py intervals)
# ---------------------------------------------------------------------------
STIFFNESS_BOUNDS: dict[str, tuple[float, float]] = {
    "x": (3000.0, 9000.0),
    "y": (3000.0, 9000.0),
    "z": (2500.0, 7000.0),
}
DAMPING_RATIO_BOUNDS: dict[str, tuple[float, float]] = {
    "x": (0.45, 1.05),
    "y": (0.45, 1.05),
    "z": (0.55, 1.20),
}
PAYLOAD_BOUNDS: tuple[float, float] = (0.0, 6.0)

# Ordering used by to_vector / from_vector / bounds / normalize / denormalize
_AXIS_ORDER = ("x", "y", "z")


@dataclass
class AxisParams:
    stiffness: float
    damping_ratio: float

    def damping(self, effective_mass: float) -> float:
        return 2.0 * self.damping_ratio * math.sqrt(self.stiffness * effective_mass)


@dataclass
class RobotParams:
    """All tunable parameters for one elastic Cartesian robot configuration."""

    drive_x: AxisParams
    drive_y: AxisParams
    drive_z: AxisParams
    payload: float = 0.0
    # These were constants in the original FMRR scripts.  Keeping defaults
    # identical preserves old experiments while allowing the sim-to-real
    # registry to tune controller-side dynamics explicitly.
    motor_stiffness: float = MOTOR_STIFFNESS
    motor_damping: float = MOTOR_DAMPING
    intermediate_mass: float = 1.0e-4

    # ------------------------------------------------------------------
    # YAML I/O
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | None = None) -> RobotParams:
        """Load from settings.yaml (default: repo config/settings.yaml)."""
        if path is None:
            path = _DEFAULT_SETTINGS_PATH
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.load(f, Loader=SafeLoader)
        drives = cfg["elastic_drives"]
        return cls(
            drive_x=AxisParams(
                stiffness=float(drives["drive_x"]["stiffness"]),
                damping_ratio=float(drives["drive_x"]["damping_ratio"]),
            ),
            drive_y=AxisParams(
                stiffness=float(drives["drive_y"]["stiffness"]),
                damping_ratio=float(drives["drive_y"]["damping_ratio"]),
            ),
            drive_z=AxisParams(
                stiffness=float(drives["drive_z"]["stiffness"]),
                damping_ratio=float(drives["drive_z"]["damping_ratio"]),
            ),
            payload=float(drives.get("payload", 0.0)),
            motor_stiffness=float(drives.get("motor_stiffness", MOTOR_STIFFNESS)),
            motor_damping=float(drives.get("motor_damping", MOTOR_DAMPING)),
            intermediate_mass=float(drives.get("intermediate_mass", 1.0e-4)),
        )

    def to_yaml(self, path: str) -> None:
        """Write params back in the settings.yaml drive_x/y/z format."""
        cfg = {
            "elastic_drives": {
                "drive_x": {
                    "stiffness": round(self.drive_x.stiffness, 4),
                    "damping_ratio": round(self.drive_x.damping_ratio, 6),
                },
                "drive_y": {
                    "stiffness": round(self.drive_y.stiffness, 4),
                    "damping_ratio": round(self.drive_y.damping_ratio, 6),
                },
                "drive_z": {
                    "stiffness": round(self.drive_z.stiffness, 4),
                    "damping_ratio": round(self.drive_z.damping_ratio, 6),
                },
                "payload": round(self.payload, 4),
                "motor_stiffness": round(self.motor_stiffness, 4),
                "motor_damping": round(self.motor_damping, 4),
                "intermediate_mass": round(self.intermediate_mass, 8),
            }
        }
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, sort_keys=False)

    # ------------------------------------------------------------------
    # Vector representation  (order: x_k, x_zeta, y_k, y_zeta, z_k, z_zeta[, payload])
    # ------------------------------------------------------------------

    def to_vector(self, include_payload: bool = True) -> np.ndarray:
        axes = [self.drive_x, self.drive_y, self.drive_z]
        v = []
        for ax in axes:
            v.extend([ax.stiffness, ax.damping_ratio])
        if include_payload:
            v.append(self.payload)
        return np.array(v, dtype=float)

    @classmethod
    def from_vector(cls, v: np.ndarray) -> RobotParams:
        return cls(
            drive_x=AxisParams(stiffness=float(v[0]), damping_ratio=float(v[1])),
            drive_y=AxisParams(stiffness=float(v[2]), damping_ratio=float(v[3])),
            drive_z=AxisParams(stiffness=float(v[4]), damping_ratio=float(v[5])),
            payload=float(v[6]) if len(v) > 6 else 0.0,
        )

    # ------------------------------------------------------------------
    # Bounds
    # ------------------------------------------------------------------

    @staticmethod
    def bounds(include_payload: bool = True) -> list[tuple[float, float]]:
        """Bounds in the same order as to_vector().

        Stiffness uses log10-space; damping-ratio and payload use linear space.
        """
        b: list[tuple[float, float]] = []
        for ax in _AXIS_ORDER:
            lo_k, hi_k = STIFFNESS_BOUNDS[ax]
            b.append((math.log10(lo_k), math.log10(hi_k)))
            b.append(DAMPING_RATIO_BOUNDS[ax])
        if include_payload:
            b.append(PAYLOAD_BOUNDS)
        return b

    # ------------------------------------------------------------------
    # Normalization  →  [-1, 1]
    # ------------------------------------------------------------------

    def normalize(self, include_payload: bool = True) -> np.ndarray:
        """Map physical params to [-1, 1] using parameter bounds."""
        v = self.to_vector(include_payload)
        b = self.bounds(include_payload)
        # Stiffness indices (0, 2, 4) are in log-space
        for i in [0, 2, 4]:
            v[i] = math.log10(v[i])
        out = np.empty_like(v)
        for i, (lo, hi) in enumerate(b):
            out[i] = 2.0 * (v[i] - lo) / (hi - lo) - 1.0
        return out

    @classmethod
    def denormalize(cls, x: np.ndarray, include_payload: bool = True) -> RobotParams:
        """Map from [-1, 1] back to physical params."""
        b = cls.bounds(include_payload)
        v = np.empty(len(x))
        for i, (lo, hi) in enumerate(b):
            v[i] = lo + (x[i] + 1.0) * (hi - lo) / 2.0
        # Stiffness back from log-space
        for i in [0, 2, 4]:
            v[i] = 10.0 ** v[i]
        return cls.from_vector(v)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def axis(self, name: str) -> AxisParams:
        """Return the AxisParams for 'x', 'y', or 'z'."""
        return {"x": self.drive_x, "y": self.drive_y, "z": self.drive_z}[name]
