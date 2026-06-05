"""Trajectory definitions: fully seeded, serializable, and replayable.

A Trajectory is a callable (t: float) -> (q_ref: ndarray, dq_ref: ndarray)
that encapsulates its generator parameters so it can be saved to JSON and
replayed bit-for-bit on both the simulator and the real robot.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np

# Default joint travel limits (match sim_common.py constants)
_DEFAULT_LIMITS: dict[str, tuple[float, float]] = {
    "x": (-1.8, 1.8),
    "y": (-1.8, 1.8),
    "z": (-1.0, 1.0),
}


@dataclass
class TrajectoryConfig:
    """Everything needed to reproduce a trajectory without re-running the sim."""

    mode: int  # 1 = sinusoidal, 2 = PTP, 0 = hold
    sim_time: float
    seed: int
    joint_limits: dict[str, tuple[float, float]]
    step_duration: float = 2.0  # only used by PTP
    params: dict = field(default_factory=dict)  # generated coefficients / points
    speed_override: float = 100.0  # % of nominal speed sent to the robot (1–100)

    @property
    def real_duration(self) -> float:
        """Actual wall-clock duration on the robot: sim_time / (speed_override / 100)."""
        return self.sim_time / max(0.01, self.speed_override / 100.0)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "sim_time": self.sim_time,
            "seed": self.seed,
            "joint_limits": {k: list(v) for k, v in self.joint_limits.items()},
            "step_duration": self.step_duration,
            "params": self.params,
            "speed_override": self.speed_override,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TrajectoryConfig:
        return cls(
            mode=int(d["mode"]),
            sim_time=float(d["sim_time"]),
            seed=int(d["seed"]),
            joint_limits={k: tuple(v) for k, v in d["joint_limits"].items()},
            step_duration=float(d.get("step_duration", 2.0)),
            params=d.get("params", {}),
            speed_override=float(d.get("speed_override", 100.0)),
        )

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> TrajectoryConfig:
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


class Trajectory:
    """Callable trajectory that returns (q_ref, dq_ref) for time t.

    Construct via the factory functions below rather than directly.
    """

    def __init__(
        self,
        fn: Callable[[float], tuple[np.ndarray, np.ndarray]],
        config: TrajectoryConfig,
    ) -> None:
        self._fn = fn
        self.config = config

    def __call__(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        return self._fn(t)

    def sample_grid(
        self, time_step: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample trajectory on a uniform time grid.

        Returns:
            times   shape (N,)
            q_refs  shape (N, 3)
            dq_refs shape (N, 3)
        """
        n = int(np.ceil(self.config.sim_time / time_step))
        times = np.arange(n) * time_step
        q_refs, dq_refs = [], []
        for t in times:
            q, dq = self._fn(t)
            q_refs.append(q)
            dq_refs.append(dq)
        return times, np.array(q_refs), np.array(dq_refs)

    def save(self, path: str) -> None:
        self.config.save(path)

    @classmethod
    def load(cls, path: str) -> Trajectory:
        config = TrajectoryConfig.load(path)
        return _trajectory_from_config(config)


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def make_hold_trajectory(
    hold_pos: np.ndarray,
    sim_time: float,
    joint_limits: dict[str, tuple[float, float]] | None = None,
) -> Trajectory:
    """Hold a fixed position for the entire simulation (REF_MODE 0)."""
    pos = np.asarray(hold_pos, dtype=float)
    config = TrajectoryConfig(
        mode=0,
        sim_time=sim_time,
        seed=0,
        joint_limits=joint_limits or _DEFAULT_LIMITS,
        params={"hold_pos": pos.tolist()},
    )

    def _fn(t: float) -> tuple[np.ndarray, np.ndarray]:
        return pos.copy(), np.zeros(3)

    return Trajectory(_fn, config)


def make_sinusoidal_trajectory(
    joint_limits: dict[str, tuple[float, float]] | None = None,
    sim_time: float = 15.0,
    seed: int = 0,
    ramp_tau: float = 1.0,
) -> Trajectory:
    """Generate a smooth sinusoidal trajectory (REF_MODE 1).

    All random draws use a single seeded Generator so the trajectory is fully
    reproducible from (joint_limits, sim_time, seed) alone.
    """
    lims = joint_limits or _DEFAULT_LIMITS
    rng = np.random.default_rng(seed)

    axes_order = ("x", "y", "z")
    use_cos = (False, True, False)  # y uses cosine to match sim_common convention
    p: dict[str, dict] = {}
    for ax, cos in zip(axes_order, use_cos):
        lo, hi = lims[ax]
        offset = (hi + lo) / 2.0
        amp_max = (hi - lo) * 0.4
        amp_min = (hi - lo) * 0.1
        amp = float(rng.uniform(amp_min, amp_max))
        freq = float(rng.uniform(0.5, 3.0))
        phase = float(rng.choice([np.pi / 2.0, -np.pi / 2.0]))
        p[ax] = {"amp": amp, "freq": freq, "phase": phase, "offset": offset, "cos": cos}

    config = TrajectoryConfig(
        mode=1,
        sim_time=sim_time,
        seed=seed,
        joint_limits=lims,
        params=p,
    )

    def _fn(t: float) -> tuple[np.ndarray, np.ndarray]:
        ramp = 1.0 - np.exp(-t / ramp_tau)
        q = np.empty(3)
        dq = np.empty(3)
        for i, ax in enumerate(axes_order):
            a, f, ph, off = p[ax]["amp"], p[ax]["freq"], p[ax]["phase"], p[ax]["offset"]
            if p[ax]["cos"]:
                q[i] = ramp * a * np.cos(f * t + ph) + off
                dq[i] = ramp * (-a * f * np.sin(f * t + ph))
            else:
                q[i] = ramp * a * np.sin(f * t + ph) + off
                dq[i] = ramp * (a * f * np.cos(f * t + ph))
        return q, dq

    return Trajectory(_fn, config)


def make_ptp_trajectory(
    joint_limits: dict[str, tuple[float, float]] | None = None,
    sim_time: float = 15.0,
    seed: int = 0,
    step_duration: float = 2.0,
    min_distance: float = 0.2,
) -> Trajectory:
    """Generate a point-to-point trajectory with cubic blending (REF_MODE 2)."""
    lims = joint_limits or _DEFAULT_LIMITS
    axes_order = ("x", "y", "z")
    limits_list = [lims[ax] for ax in axes_order]

    rng = np.random.default_rng(seed)
    n_segments = int(np.ceil(sim_time / step_duration))
    q_start = np.array([(lo + hi) / 2.0 for lo, hi in limits_list])
    points = [q_start]
    for _ in range(n_segments):
        for _attempt in range(1000):
            q_cand = np.array(
                [float(rng.uniform(lo, hi)) for lo, hi in limits_list]
            )
            if np.linalg.norm(q_cand - points[-1]) > min_distance:
                points.append(q_cand)
                break
    pts = np.array(points)

    config = TrajectoryConfig(
        mode=2,
        sim_time=sim_time,
        seed=seed,
        joint_limits=lims,
        step_duration=step_duration,
        params={"points": pts.tolist()},
    )

    def _fn(t: float) -> tuple[np.ndarray, np.ndarray]:
        segment = int(t // step_duration)
        if segment >= n_segments:
            return pts[-1].copy(), np.zeros(3)
        tau = np.clip((t - segment * step_duration) / step_duration, 0.0, 1.0)
        q0, q1 = pts[segment], pts[segment + 1]
        s = 3.0 * tau**2 - 2.0 * tau**3
        ds = (6.0 * tau * (1.0 - tau)) / step_duration
        return q0 + s * (q1 - q0), ds * (q1 - q0)

    return Trajectory(_fn, config)


def _trajectory_from_config(config: TrajectoryConfig) -> Trajectory:
    """Reconstruct a Trajectory callable from a saved TrajectoryConfig."""
    if config.mode == 0:
        hold_pos = np.array(config.params["hold_pos"])
        return make_hold_trajectory(hold_pos, config.sim_time, config.joint_limits)

    if config.mode == 1:
        p = config.params
        axes_order = ("x", "y", "z")

        def _sin_fn(t: float) -> tuple[np.ndarray, np.ndarray]:
            ramp = 1.0 - np.exp(-t / 1.0)
            q = np.empty(3)
            dq = np.empty(3)
            for i, ax in enumerate(axes_order):
                a, f, ph, off = p[ax]["amp"], p[ax]["freq"], p[ax]["phase"], p[ax]["offset"]
                if p[ax]["cos"]:
                    q[i] = ramp * a * np.cos(f * t + ph) + off
                    dq[i] = ramp * (-a * f * np.sin(f * t + ph))
                else:
                    q[i] = ramp * a * np.sin(f * t + ph) + off
                    dq[i] = ramp * (a * f * np.cos(f * t + ph))
            return q, dq

        return Trajectory(_sin_fn, config)

    if config.mode == 2:
        pts = np.array(config.params["points"])
        n_segments = len(pts) - 1
        step_duration = config.step_duration

        def _ptp_fn(t: float) -> tuple[np.ndarray, np.ndarray]:
            segment = int(t // step_duration)
            if segment >= n_segments:
                return pts[-1].copy(), np.zeros(3)
            tau = np.clip((t - segment * step_duration) / step_duration, 0.0, 1.0)
            q0, q1 = pts[segment], pts[segment + 1]
            s = 3.0 * tau**2 - 2.0 * tau**3
            ds = (6.0 * tau * (1.0 - tau)) / step_duration
            return q0 + s * (q1 - q0), ds * (q1 - q0)

        return Trajectory(_ptp_fn, config)

    raise ValueError(f"Unknown trajectory mode: {config.mode}")
