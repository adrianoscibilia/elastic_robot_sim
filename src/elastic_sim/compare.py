"""Fidelity metric between a simulated and a real-robot rollout.

Usage::

    result = compare(sim_rollout, real_rollout, weights={"position": 1.0, "velocity": 0.3, "force": 0.5})
    print(result["metric"])   # scalar aggregate loss
    print(result["per_axis"]) # dict with per-axis breakdown

The two rollouts are resampled onto a shared time grid before comparison.
"""

from __future__ import annotations

import numpy as np

from .rollout import RolloutResult

# Default signal weights
DEFAULT_WEIGHTS: dict[str, float] = {
    "position": 1.0,
    "velocity": 0.3,
    "force": 0.5,
}

# Axes label order (matches column 0/1/2)
AXES = ("x", "y", "z")


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    diff = a.reshape(-1) - b.reshape(-1)
    return float(np.sqrt(np.mean(diff**2)))


def _norm_rmse(a: np.ndarray, b: np.ndarray, ref_scale: float = 1.0) -> float:
    """RMSE normalised by a reference scale so different signal types are comparable."""
    if ref_scale == 0.0:
        return 0.0
    return _rmse(a, b) / ref_scale


def _signal_scale(arr: np.ndarray) -> float:
    """Robust scale: 95th-percentile absolute value, floor at 1e-6."""
    return max(float(np.percentile(np.abs(arr), 95)), 1e-6)


def compare(
    sim: RolloutResult,
    real: RolloutResult,
    weights: dict[str, float] | None = None,
    *,
    cut_off_time: float = 0.0,
) -> dict:
    """Compute fidelity metrics between sim and real rollouts.

    Steps:
    1. Build a common time grid (intersection of both time ranges).
    2. Resample both rollouts onto that grid.
    3. Optionally skip the initial transient (cut_off_time).
    4. Compute per-signal normalised RMSE, then aggregate with weights.

    Args:
        sim: Simulated rollout.
        real: Real-robot rollout.
        weights: Optional dict with keys "position", "velocity", "force".
                 Defaults to DEFAULT_WEIGHTS.
        cut_off_time: Skip the first this-many seconds of data.

    Returns:
        dict with keys:
            "metric"      – scalar aggregate loss (lower is better)
            "per_signal"  – normalised RMSE for each signal component
            "per_axis"    – average normalised RMSE per axis
    """
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)

    # Build common time grid
    t_start = max(float(sim.time[0]), float(real.time[0]), cut_off_time)
    t_end = min(float(sim.time[-1]), float(real.time[-1]))
    if t_end <= t_start:
        raise ValueError(
            f"No overlapping time after cut-off: sim=[{sim.time[0]:.3f},{sim.time[-1]:.3f}],"
            f" real=[{real.time[0]:.3f},{real.time[-1]:.3f}], cut_off={cut_off_time}"
        )
    # Use the finer of the two grids
    dt = min(
        float(np.median(np.diff(sim.time))),
        float(np.median(np.diff(real.time))),
    )
    grid = np.arange(t_start, t_end, dt)

    s = sim.resample(grid)
    r = real.resample(grid)

    per_signal: dict[str, float] = {}
    per_axis: dict[str, list[float]] = {ax: [] for ax in AXES}

    # Position (use link position as it reflects compliance)
    for i, ax in enumerate(AXES):
        scale = _signal_scale(r.q_link[:, i])
        err = _norm_rmse(s.q_link[:, i], r.q_link[:, i], scale)
        key = f"q_link_{ax}"
        per_signal[key] = err
        per_axis[ax].append(err * w["position"])

    # Velocity
    for i, ax in enumerate(AXES):
        scale = _signal_scale(r.dq_link[:, i])
        err = _norm_rmse(s.dq_link[:, i], r.dq_link[:, i], scale)
        key = f"dq_link_{ax}"
        per_signal[key] = err
        per_axis[ax].append(err * w["velocity"])

    # Force (elastic-link torque)
    for i, ax in enumerate(AXES):
        scale = _signal_scale(r.tau_link[:, i])
        err = _norm_rmse(s.tau_link[:, i], r.tau_link[:, i], scale)
        key = f"tau_link_{ax}"
        per_signal[key] = err
        per_axis[ax].append(err * w["force"])

    w_total = w["position"] + w["velocity"] + w["force"]
    per_axis_scalar = {
        ax: float(np.sum(vals) / w_total) for ax, vals in per_axis.items()
    }
    metric = float(np.mean(list(per_axis_scalar.values())))

    return {
        "metric": metric,
        "per_signal": per_signal,
        "per_axis": per_axis_scalar,
    }
