"""Materialized, asset-joint-space trajectories and generic rollout storage.

The older trajectory classes are intentionally kept for backwards
compatibility.  New sim-to-real runs use :class:`MaterializedTrajectory` so
the exact samples sent to a controller are also the samples consumed by every
simulator backend.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _as_2d(value: Any, n: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or array.shape[1] != n:
        raise ValueError(f"{name} must have shape (N, {n}), got {array.shape}")
    return array


@dataclass(frozen=True)
class MaterializedTrajectory:
    """Exact joint-space samples used by simulation and ROS execution."""

    time: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    joint_names: tuple[str, ...]
    acceleration: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        names = tuple(self.joint_names)
        object.__setattr__(self, "joint_names", names)
        t = np.asarray(self.time, dtype=float).reshape(-1)
        if len(names) == 0:
            raise ValueError("joint_names must be non-empty")
        if len(t) < 2 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0.0):
            raise ValueError("trajectory time must be finite and strictly increasing")
        q = _as_2d(self.position, len(names), "position")
        dq = _as_2d(self.velocity, len(names), "velocity")
        if len(q) != len(t) or len(dq) != len(t):
            raise ValueError("time, position, and velocity must have equal lengths")
        ddq = None if self.acceleration is None else _as_2d(self.acceleration, len(names), "acceleration")
        if ddq is not None and len(ddq) != len(t):
            raise ValueError("acceleration must have the same length as time")
        if not (np.isfinite(q).all() and np.isfinite(dq).all() and (ddq is None or np.isfinite(ddq).all())):
            raise ValueError("trajectory samples must be finite")
        object.__setattr__(self, "time", t)
        object.__setattr__(self, "position", q)
        object.__setattr__(self, "velocity", dq)
        object.__setattr__(self, "acceleration", ddq)

    @property
    def n_dof(self) -> int:
        return len(self.joint_names)

    @property
    def duration(self) -> float:
        return float(self.time[-1] - self.time[0])

    def __call__(self, time_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Interpolate ``(position, velocity, acceleration)`` at ``time_s``."""
        t = float(np.clip(time_s, self.time[0], self.time[-1]))
        q = np.asarray([np.interp(t, self.time, self.position[:, i]) for i in range(self.n_dof)])
        dq = np.asarray([np.interp(t, self.time, self.velocity[:, i]) for i in range(self.n_dof)])
        if self.acceleration is None:
            ddq = np.zeros(self.n_dof, dtype=float)
        else:
            ddq = np.asarray([np.interp(t, self.time, self.acceleration[:, i]) for i in range(self.n_dof)])
        return q, dq, ddq

    def sample(self, time_step: float | None = None) -> dict[str, np.ndarray]:
        if time_step is None:
            return {"time": self.time.copy(), "q": self.position.copy(), "dq": self.velocity.copy(),
                    "ddq": None if self.acceleration is None else self.acceleration.copy()}
        if time_step <= 0.0:
            raise ValueError("time_step must be positive")
        grid = np.arange(self.time[0], self.time[-1] + 0.5 * time_step, time_step)
        values = [self(t) for t in grid]
        return {
            "time": grid,
            "q": np.asarray([v[0] for v in values]),
            "dq": np.asarray([v[1] for v in values]),
            "ddq": np.asarray([v[2] for v in values]),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "joint_names": list(self.joint_names),
            "time": self.time.tolist(),
            "position": self.position.tolist(),
            "velocity": self.velocity.tolist(),
            "acceleration": None if self.acceleration is None else self.acceleration.tolist(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MaterializedTrajectory":
        return cls(
            time=np.asarray(value["time"], dtype=float),
            position=np.asarray(value["position"], dtype=float),
            velocity=np.asarray(value["velocity"], dtype=float),
            acceleration=None if value.get("acceleration") is None else np.asarray(value["acceleration"], dtype=float),
            joint_names=tuple(value["joint_names"]),
            metadata=value.get("metadata", {}),
        )

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "MaterializedTrajectory":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def digest(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()
