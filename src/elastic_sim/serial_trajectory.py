"""Asset-driven, serial-arm joint trajectory generation.

Pinocchio is used to parse and validate an asset URDF and to discover its
one-degree-of-freedom joints.  The generated trajectory itself is a smooth
minimum-jerk joint-space path, so replaying a saved JSON needs only NumPy.
This intentionally keeps synthetic-data generation independent of ROS.  A
MoveIt adapter can consume the same saved waypoints when collision-aware
planning is required for a real robot.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .assets import discover_urdf_joints
from .materialized import MaterializedTrajectory


@dataclass(frozen=True)
class SerialTrajectoryConfig:
    """A fully serializable, time-parameterized joint-space trajectory."""

    joint_names: tuple[str, ...]
    waypoints: tuple[tuple[float, ...], ...]
    segment_durations: tuple[float, ...]
    source_urdf: str | None = None
    generator: str = "pinocchio_minimum_jerk"

    def __post_init__(self) -> None:
        n = len(self.joint_names)
        if n == 0:
            raise ValueError("joint_names must be non-empty")
        if len(self.waypoints) < 2:
            raise ValueError("at least two waypoints are required")
        if len(self.segment_durations) != len(self.waypoints) - 1:
            raise ValueError("one duration is required for every waypoint segment")
        if any(len(q) != n for q in self.waypoints):
            raise ValueError("every waypoint must have one value per joint")
        if any(t <= 0.0 for t in self.segment_durations):
            raise ValueError("segment durations must be positive")

    @property
    def duration(self) -> float:
        return float(sum(self.segment_durations))

    def to_dict(self) -> dict:
        return asdict(self) | {"joint_names": list(self.joint_names), "waypoints": [list(q) for q in self.waypoints], "segment_durations": list(self.segment_durations)}

    @classmethod
    def from_dict(cls, value: dict) -> "SerialTrajectoryConfig":
        return cls(
            joint_names=tuple(value["joint_names"]),
            waypoints=tuple(tuple(float(x) for x in q) for q in value["waypoints"]),
            segment_durations=tuple(float(x) for x in value["segment_durations"]),
            source_urdf=value.get("source_urdf"),
            generator=value.get("generator", "pinocchio_minimum_jerk"),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SerialTrajectoryConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


class SerialArmTrajectory:
    """Evaluates a :class:`SerialTrajectoryConfig` with C2 continuity."""

    def __init__(self, config: SerialTrajectoryConfig) -> None:
        self.config = config
        self._points = np.asarray(config.waypoints, dtype=float)
        self._ends = np.cumsum(np.asarray(config.segment_durations, dtype=float))

    def __call__(self, time_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(q, dq, ddq)`` at a clamped wall-clock time."""
        t = float(np.clip(time_s, 0.0, self.config.duration))
        segment = min(int(np.searchsorted(self._ends, t, side="right")), len(self._ends) - 1)
        start = 0.0 if segment == 0 else float(self._ends[segment - 1])
        duration = float(self.config.segment_durations[segment])
        u = np.clip((t - start) / duration, 0.0, 1.0)
        # Quintic minimum-jerk blend and its first two derivatives.
        s = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
        ds = (30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4) / duration
        dds = (60.0 * u - 180.0 * u**2 + 120.0 * u**3) / duration**2
        delta = self._points[segment + 1] - self._points[segment]
        return self._points[segment] + s * delta, ds * delta, dds * delta

    def sample(self, time_step: float) -> dict[str, np.ndarray]:
        if time_step <= 0.0:
            raise ValueError("time_step must be positive")
        time = np.arange(0.0, self.config.duration + 0.5 * time_step, time_step)
        samples = [self(t) for t in time]
        return {
            "time": time,
            "q": np.asarray([item[0] for item in samples]),
            "dq": np.asarray([item[1] for item in samples]),
            "ddq": np.asarray([item[2] for item in samples]),
            "joint_names": np.asarray(self.config.joint_names),
        }


def trajectory_evaluator(trajectory: SerialTrajectoryConfig | MaterializedTrajectory):
    """Return a common ``(q, dq, ddq)`` evaluator for serial simulations."""
    if isinstance(trajectory, MaterializedTrajectory):
        return trajectory
    return SerialArmTrajectory(trajectory)


def _urdf_joint_limits(urdf_path: str | Path, requested_names: Iterable[str] | None = None) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    """Return portable URDF joint limits without requiring Pinocchio.

    Pinocchio remains useful for collision-aware planning, but a seeded
    joint-space excitation only needs the one-DoF limits that are already in
    the asset URDF.  Continuous/unbounded joints receive ``[-pi, pi]``.
    """
    requested_order = tuple(requested_names) if requested_names else None
    requested = set(requested_order) if requested_order else None
    names: list[str] = []
    lower: list[float] = []
    upper: list[float] = []
    for joint in discover_urdf_joints(urdf_path):
        if not joint.is_one_dof or joint.mimic is not None:
            continue
        name = joint.name
        if requested is not None and name not in requested:
            continue
        lo = float("nan") if joint.lower is None else joint.lower
        hi = float("nan") if joint.upper is None else joint.upper
        if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
            lo, hi = -np.pi, np.pi
        names.append(name)
        lower.append(lo)
        upper.append(hi)
    if requested is not None and set(names) != requested:
        missing = sorted(requested - set(names))
        raise ValueError(f"requested joints are not one-DoF URDF joints: {missing}")
    if not names:
        raise ValueError("the asset has no selectable one-DoF joints")
    if requested_order is not None:
        positions = {name: index for index, name in enumerate(names)}
        order = [positions[name] for name in requested_order]
        names = [names[index] for index in order]
        lower = [lower[index] for index in order]
        upper = [upper[index] for index in order]
    return tuple(names), np.asarray(lower), np.asarray(upper)


def generate_serial_arm_trajectory(
    urdf_path: str | Path,
    *,
    joint_names: Iterable[str] | None = None,
    num_waypoints: int = 8,
    max_velocity: float | np.ndarray = 1.0,
    max_acceleration: float | np.ndarray = 2.0,
    limit_margin: float = 0.08,
    seed: int | None = None,
) -> SerialTrajectoryConfig:
    """Create a seeded, joint-limit-safe excitation trajectory from a URDF.

    Segment times satisfy conservative minimum-jerk velocity and acceleration
    bounds.  The caller may supply per-joint scalar arrays for heterogeneous
    arms.  Collision-aware planning is intentionally delegated to the future
    MoveIt adapter; synthetic data needs repeatable dynamic excitation first.
    """
    if num_waypoints < 2:
        raise ValueError("num_waypoints must be at least two")
    if not 0.0 <= limit_margin < 0.5:
        raise ValueError("limit_margin must be in [0, 0.5)")
    names, lower, upper = _urdf_joint_limits(urdf_path, joint_names)
    n = len(names)
    vmax = np.broadcast_to(np.asarray(max_velocity, dtype=float), (n,))
    amax = np.broadcast_to(np.asarray(max_acceleration, dtype=float), (n,))
    if np.any(vmax <= 0.0) or np.any(amax <= 0.0):
        raise ValueError("velocity and acceleration bounds must be positive")
    span = upper - lower
    safe_lower = lower + limit_margin * span
    safe_upper = upper - limit_margin * span
    rng = np.random.default_rng(seed)
    points = rng.uniform(safe_lower, safe_upper, size=(num_waypoints, n))
    # A zero velocity/acceleration quintic has maxima 1.875*|dq|/T and
    # 5.7735*|dq|/T^2.  Use the most restrictive joint for each segment.
    delta = np.abs(np.diff(points, axis=0))
    durations = np.maximum(
        np.max(1.875 * delta / vmax, axis=1),
        np.max(np.sqrt(5.7735 * delta / amax), axis=1),
    )
    durations = np.maximum(durations, 1.0e-3)
    return SerialTrajectoryConfig(
        joint_names=names,
        waypoints=tuple(tuple(float(x) for x in point) for point in points),
        segment_durations=tuple(float(x) for x in durations),
        source_urdf=str(Path(urdf_path).resolve()),
    )
