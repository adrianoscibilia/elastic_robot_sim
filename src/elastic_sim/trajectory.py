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


# ---------------------------------------------------------------------------
# Settings — typed loader from settings.yaml trajectory: block (TASK 2)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrajectorySettings:
    joint_limits: dict
    peak_cartesian_velocity_ms: float
    axis_velocity_limits_ms: dict
    ramp_tau: float
    amplitude_fraction: float
    amplitude_min_m: float
    freq_min: float
    freq_max: float
    step_duration: float
    min_distance: float
    ros_points_per_segment: int


def load_trajectory_settings(settings: dict) -> TrajectorySettings:
    t = settings["trajectory"]
    s = t.get("sinusoidal", {})
    p = t.get("ptp", {})
    # Backward-compat: if trajectory.joint_limits absent but collection.joint_limits present, use that.
    if "joint_limits" not in t and "joint_limits" in settings.get("collection", {}):
        raw_jl = settings["collection"]["joint_limits"]
    else:
        raw_jl = t["joint_limits"]
    return TrajectorySettings(
        joint_limits={k: tuple(v) for k, v in raw_jl.items()},
        peak_cartesian_velocity_ms=float(t["peak_cartesian_velocity_ms"]),
        axis_velocity_limits_ms={
            k: (None if v is None else float(v))
            for k, v in t.get("axis_velocity_limits_ms", {}).items()
        },
        ramp_tau=float(t.get("ramp_tau", 1.0)),
        amplitude_fraction=float(s.get("amplitude_fraction", 0.4)),
        amplitude_min_m=float(s.get("amplitude_min_m", 0.2)),
        freq_min=float(s.get("freq_min", 0.5)),
        freq_max=float(s.get("freq_max", 3.0)),
        step_duration=float(p.get("step_duration", 2.0)),
        min_distance=float(p.get("min_distance", 0.2)),
        ros_points_per_segment=int(t.get("ros_points_per_segment", 20)),
    )


def validate_trajectory_settings(s: TrajectorySettings) -> None:
    assert s.freq_min < s.freq_max, "freq_min must be < freq_max"
    assert s.peak_cartesian_velocity_ms > 0, "peak_cartesian_velocity_ms must be > 0"
    for ax, lim in s.joint_limits.items():
        lo, hi = lim
        assert lo < hi, f"bad joint_limits for {ax}: lo={lo} >= hi={hi}"
        assert s.amplitude_min_m < (hi - lo) * s.amplitude_fraction, (
            f"amplitude_min_m ({s.amplitude_min_m}) >= max amplitude for axis {ax} "
            f"({(hi - lo) * s.amplitude_fraction:.3f})"
        )


# ---------------------------------------------------------------------------
# TrajectoryConfig (TASK 3)
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryConfig:
    """Everything needed to reproduce a trajectory without re-running the sim."""

    mode: int  # 1 = sinusoidal, 2 = PTP, 0 = hold
    sim_time: float                            # executed duration (post-baking)
    seed: int
    joint_limits: dict
    step_duration: float = 2.0                 # only used by PTP (executed, post-baking)
    params: dict = field(default_factory=dict) # generated coefficients / points (executed)
    speed_override: float = 100.0              # always 100 in new files (baked); kept for back-compat
    vel_limit_ms: float | None = None          # DEPRECATED — kept for backward-compat replay of old JSON

    # --- Provenance fields (new in this refactor) ---
    ramp_tau: float = 1.0
    nominal_sim_time: float = 0.0          # pre-scaling duration; 0.0 → treated as sim_time
    nominal_speed_override: float = 100.0  # user speed % that was requested
    global_speed_factor: float = 1.0       # resolved factor actually applied ∈ (0,1]
    executed_peak_velocity_ms: float = 0.0 # v_nom_cart * global_speed_factor (≤ peak_velocity_limit_ms)
    peak_velocity_limit_ms: float = 0.0    # v_lim_cart used at generation
    ros_points_per_segment: int = 20       # ROS FollowJointTrajectory goal density

    @property
    def effective_sim_time(self) -> float:
        """Actual wall-clock execution duration.

        For new (baked) files speed_override==100 so this equals sim_time directly.
        For legacy files the old formula re-derives the executed duration.
        """
        if self.speed_override == 100.0:
            return self.sim_time
        return self.sim_time * 100.0 / self.speed_override
    speed_override: float = 100.0  # % of nominal speed sent to the robot (1–100)

    @property
    def real_duration(self) -> float:
        """Actual wall-clock duration on the robot: sim_time / (speed_override / 100)."""
        return self.sim_time / max(0.01, self.speed_override / 100.0)

    def to_dict(self) -> dict:
        d = {
            "mode": self.mode,
            "sim_time": self.sim_time,
            "seed": self.seed,
            "joint_limits": {k: list(v) for k, v in self.joint_limits.items()},
            "step_duration": self.step_duration,
            "params": self.params,
            "speed_override": self.speed_override,
            "speed_override": self.speed_override,
            "ramp_tau": self.ramp_tau,
            "nominal_sim_time": self.nominal_sim_time,
            "nominal_speed_override": self.nominal_speed_override,
            "global_speed_factor": self.global_speed_factor,
            "executed_peak_velocity_ms": self.executed_peak_velocity_ms,
            "peak_velocity_limit_ms": self.peak_velocity_limit_ms,
            "ros_points_per_segment": self.ros_points_per_segment,
        }
        if self.vel_limit_ms is not None:
            d["vel_limit_ms"] = self.vel_limit_ms
        return d

    @classmethod
    def from_dict(cls, d: dict) -> TrajectoryConfig:
        sim_time = float(d["sim_time"])
        nominal_sim_time = float(d.get("nominal_sim_time", 0.0))
        if nominal_sim_time == 0.0:
            nominal_sim_time = sim_time
        return cls(
            mode=int(d["mode"]),
            sim_time=sim_time,
            seed=int(d["seed"]),
            joint_limits={k: tuple(v) for k, v in d["joint_limits"].items()},
            step_duration=float(d.get("step_duration", 2.0)),
            params=d.get("params", {}),
            speed_override=float(d.get("speed_override", 100.0)),
            speed_override=float(d.get("speed_override", 100.0)),
            vel_limit_ms=float(d["vel_limit_ms"]) if d.get("vel_limit_ms") is not None else None,
            ramp_tau=float(d.get("ramp_tau", 1.0)),
            nominal_sim_time=nominal_sim_time,
            nominal_speed_override=float(d.get("nominal_speed_override", 100.0)),
            global_speed_factor=float(d.get("global_speed_factor", 1.0)),
            executed_peak_velocity_ms=float(d.get("executed_peak_velocity_ms", 0.0)),
            peak_velocity_limit_ms=float(d.get("peak_velocity_limit_ms", 0.0)),
            ros_points_per_segment=int(d.get("ros_points_per_segment", 20)),
        )

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> TrajectoryConfig:
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


# ---------------------------------------------------------------------------
# Trajectory callable
# ---------------------------------------------------------------------------

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
# Speed-override helpers (TASK 4)
# ---------------------------------------------------------------------------

def _peak_velocities(
    fn: Callable, sim_time: float, time_step: float = 0.01
) -> tuple[float, dict]:
    """Sample fn(t)->(q,dq) and return (v_cart_peak, {axis: v_axis_peak}).

    Peak estimate is grid-sampled at time_step resolution — fine for the
    smooth signals used here.
    """
    n = int(np.ceil(sim_time / time_step))
    dq = np.array([fn(i * time_step)[1] for i in range(n)])  # (n, 3)
    v_axis = np.max(np.abs(dq), axis=0)                       # (3,)
    v_cart = float(np.max(np.linalg.norm(dq, axis=1)))        # scalar
    return v_cart, {a: float(v_axis[i]) for i, a in enumerate(("x", "y", "z"))}


def resolve_global_factor(
    v_cart: float,
    v_axis: dict,
    *,
    speed_override: float,
    v_lim_cart: float,
    v_lim_axis: dict,
) -> float:
    """Compute the single global speed factor ∈ (0, 1].

    Rule: global_factor = min(user_factor, auto_factor).
    The more restrictive of {user request, velocity limits} wins.
    They are never multiplied — see design doc 03_SPEED_OVERRIDE_DESIGN.md.
    """
    user_factor = speed_override / 100.0
    auto = 1.0
    if v_cart > 0:
        auto = min(auto, v_lim_cart / v_cart)
    for a in ("x", "y", "z"):
        lim = v_lim_axis.get(a)
        if lim is not None and v_axis.get(a, 0.0) > 0:
            auto = min(auto, lim / v_axis[a])
    auto = min(1.0, auto)  # never speed up
    return max(1e-6, min(user_factor, auto))


def _bake_factor(config: TrajectoryConfig, f: float) -> TrajectoryConfig:
    """Return a new config whose stored params encode the executed motion.

    Sinusoidal: multiply every frequency by f (position range unchanged —
    slowing down must not move the workspace).
    PTP: divide step_duration by f (waypoints unchanged).
    Duration: executed = nominal_sim_time / f.
    ramp_tau is a wall-clock constant; kept as-is (it already acts in
    executed time).
    """
    import copy
    cfg = copy.deepcopy(config)
    if cfg.mode == 1:  # sinusoidal: scale frequencies
        for ax in cfg.params:
            cfg.params[ax]["freq"] *= f
    elif cfg.mode == 2:  # ptp: stretch step_duration
        cfg.step_duration /= f
    cfg.sim_time = config.nominal_sim_time / f
    cfg.speed_override = 100.0
    cfg.global_speed_factor = f
    return cfg


# ---------------------------------------------------------------------------
# Factory functions (TASK 5)
# ---------------------------------------------------------------------------

def make_hold_trajectory(
    hold_pos: np.ndarray,
    sim_time: float,
    settings: TrajectorySettings | None = None,
) -> Trajectory:
    """Hold a fixed position for the entire simulation (REF_MODE 0)."""
    pos = np.asarray(hold_pos, dtype=float)
    joint_limits = settings.joint_limits if settings is not None else {
        "x": (-1.8, 1.8), "y": (-1.8, 1.8), "z": (-1.0, 1.0)
    }
    config = TrajectoryConfig(
        mode=0,
        sim_time=sim_time,
        seed=0,
        joint_limits=joint_limits,
        params={"hold_pos": pos.tolist()},
        speed_override=100.0,
        nominal_sim_time=sim_time,
        global_speed_factor=1.0,
    )

    def _fn(t: float) -> tuple[np.ndarray, np.ndarray]:
        return pos.copy(), np.zeros(3)

    return Trajectory(_fn, config)


def make_sinusoidal_trajectory(
    settings: TrajectorySettings,
    sim_time: float,
    seed: int,
    *,
    speed_override: float = 100.0,
    time_step: float = 0.01,
) -> Trajectory:
    """Generate a smooth sinusoidal trajectory (REF_MODE 1).

    All random draws use a single seeded Generator so the trajectory is fully
    reproducible from (settings, sim_time, seed) alone. The global speed factor
    is resolved and baked into the stored params so the saved config is self-describing.
    """
    lims = settings.joint_limits
    rng = np.random.default_rng(seed)

    axes_order = ("x", "y", "z")
    use_cos = (False, True, False)  # y uses cosine to match sim_common convention
    p: dict = {}
    for ax, cos in zip(axes_order, use_cos):
        lo, hi = lims[ax]
        offset = (hi + lo) / 2.0
        amp_max = (hi - lo) * 0.4
        amp_min = (hi - lo) * 0.1
        amp = float(rng.uniform(amp_min, amp_max))
        freq = float(rng.uniform(0.5, 3.0))
        phase = float(rng.choice([np.pi / 2.0, -np.pi / 2.0]))
        p[ax] = {"amp": amp, "freq": freq, "phase": phase, "offset": offset, "cos": cos}

    ramp_tau = settings.ramp_tau

    # Build nominal fn to measure peak velocity
    def _nominal_fn(t: float) -> tuple[np.ndarray, np.ndarray]:
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

    # Use the analytical supremum of the Euclidean velocity norm: sqrt(Σ(amp·freq)²).
    # Grid-sampling the quasiperiodic multi-axis norm can underestimate the true max
    # (three incommensurable frequencies may not align within the nominal window).
    # The analytical bound is always achievable, so using it ensures the executed
    # peak never exceeds v_lim_cart regardless of phase alignment.
    v_axis_steady = {ax: abs(p[ax]["amp"] * p[ax]["freq"]) for ax in axes_order}
    v_cart = float(np.sqrt(sum(v**2 for v in v_axis_steady.values())))

    f = resolve_global_factor(
        v_cart, v_axis_steady,
        speed_override=speed_override,
        v_lim_cart=settings.peak_cartesian_velocity_ms,
        v_lim_axis=settings.axis_velocity_limits_ms,
    )

    nominal_config = TrajectoryConfig(
        mode=1,
        sim_time=sim_time,
        seed=seed,
        joint_limits=lims,
        params=p,
        speed_override=speed_override,
        ramp_tau=ramp_tau,
        nominal_sim_time=sim_time,
        nominal_speed_override=speed_override,
        global_speed_factor=f,
        executed_peak_velocity_ms=v_cart * f,
        peak_velocity_limit_ms=settings.peak_cartesian_velocity_ms,
        ros_points_per_segment=settings.ros_points_per_segment,
    )
    baked_config = _bake_factor(nominal_config, f)
    return _trajectory_from_config(baked_config)


def make_ptp_trajectory(
    settings: TrajectorySettings,
    sim_time: float,
    seed: int,
    *,
    speed_override: float = 100.0,
    time_step: float = 0.01,
) -> Trajectory:
    """Generate a point-to-point trajectory with cubic blending (REF_MODE 2)."""
    lims = settings.joint_limits
    axes_order = ("x", "y", "z")
    limits_list = [lims[ax] for ax in axes_order]

    step_duration = settings.step_duration
    min_distance = settings.min_distance

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

    # Analytical peak velocity for PTP cubic blend: max_seg ||dq|| = (3/2)/step_duration * ||q1-q0||.
    # Per-axis: max_seg |dq_i| = (3/2)/step_duration * |q1_i - q0_i|.
    # Using this avoids grid-sampling floating-point drift that can slightly over-scale the factor.
    _scale = 1.5 / step_duration
    _diffs = np.diff(pts, axis=0)  # (n_segments, 3)
    v_axis_ptp = {ax: float(_scale * float(np.max(np.abs(_diffs[:, i]))))
                  for i, ax in enumerate(("x", "y", "z"))}
    v_cart = float(_scale * float(np.max(np.linalg.norm(_diffs, axis=1))))

    f = resolve_global_factor(
        v_cart, v_axis_ptp,
        speed_override=speed_override,
        v_lim_cart=settings.peak_cartesian_velocity_ms,
        v_lim_axis=settings.axis_velocity_limits_ms,
    )

    nominal_config = TrajectoryConfig(
        mode=2,
        sim_time=sim_time,
        seed=seed,
        joint_limits=lims,
        step_duration=step_duration,
        params={"points": pts.tolist()},
        speed_override=speed_override,
        ramp_tau=settings.ramp_tau,
        nominal_sim_time=sim_time,
        nominal_speed_override=speed_override,
        global_speed_factor=f,
        executed_peak_velocity_ms=v_cart * f,
        peak_velocity_limit_ms=settings.peak_cartesian_velocity_ms,
        ros_points_per_segment=settings.ros_points_per_segment,
    )
    baked_config = _bake_factor(nominal_config, f)
    return _trajectory_from_config(baked_config)


# ---------------------------------------------------------------------------
# Config reconstructor (TASK 6)
# ---------------------------------------------------------------------------

def _trajectory_from_config(config: TrajectoryConfig) -> Trajectory:
    """Reconstruct a Trajectory callable from a saved TrajectoryConfig.

    New files (speed_override==100, global_speed_factor set by generation)
    are pre-baked: the stored params already encode the executed motion, so
    no extra scaling is applied here.

    Old files (speed_override != 100 and global_speed_factor == 1.0) hit
    the LEGACY replay path which applies the old time-warp and optional
    velocity clip, preserving exact replay of historical recordings.
    """
    if config.mode == 0:
        hold_pos = np.array(config.params["hold_pos"])
        _fn: Callable = lambda t: (hold_pos.copy(), np.zeros(3))

    elif config.mode == 1:
        p = config.params
        axes_order = ("x", "y", "z")
        ramp_tau = config.ramp_tau  # wall-clock time constant

        def _fn(t: float) -> tuple[np.ndarray, np.ndarray]:  # type: ignore[misc]
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

    elif config.mode == 2:
        pts = np.array(config.params["points"])
        n_segments = len(pts) - 1
        step_duration = config.step_duration

        def _fn(t: float) -> tuple[np.ndarray, np.ndarray]:  # type: ignore[misc]
            segment = int(t // step_duration)
            if segment >= n_segments:
                return pts[-1].copy(), np.zeros(3)
            tau = np.clip((t - segment * step_duration) / step_duration, 0.0, 1.0)
            q0, q1 = pts[segment], pts[segment + 1]
            s = 3.0 * tau**2 - 2.0 * tau**3
            ds = (6.0 * tau * (1.0 - tau)) / step_duration
            return q0 + s * (q1 - q0), ds * (q1 - q0)

    else:
        raise ValueError(f"Unknown trajectory mode: {config.mode}")

    # LEGACY replay path — new files are pre-baked (speed_override==100).
    # Old files written before this refactor stored the nominal params with
    # speed_override != 100 and global_speed_factor == 1.0 (field absent).
    # Replay them exactly as before so historical recordings are not broken.
    if config.speed_override != 100.0 and config.global_speed_factor == 1.0:
        factor = config.speed_override / 100.0
        _base = _fn

        def _fn(t: float, _f: float = factor, _b: Callable = _base) -> tuple:  # type: ignore[misc]
            q, dq = _b(t * _f)
            return q, dq * _f

        if config.vel_limit_ms is not None:
            _vl = float(config.vel_limit_ms)
            _prev = _fn

            def _fn(t: float, _p: Callable = _prev, _v: float = _vl) -> tuple:  # type: ignore[misc]
                q, dq = _p(t)
                return q, np.clip(dq, -_v, _v)

    return Trajectory(_fn, config)
