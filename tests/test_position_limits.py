"""Sampled positions must stay within joint limits (baking must not change workspace)."""

import os
import sys

import numpy as np
import pytest
import yaml

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_REPO, "src"))

SETTINGS = yaml.safe_load(open(os.path.join(_REPO, "config", "settings.yaml"), encoding="utf-8"))

from elastic_sim.trajectory import (
    load_trajectory_settings,
    make_sinusoidal_trajectory,
    make_ptp_trajectory,
)

EPS = 1e-3


@pytest.mark.parametrize("seed", range(10))
def test_sin_positions_in_limits(seed):
    ts = load_trajectory_settings(SETTINGS)
    traj = make_sinusoidal_trajectory(ts, 15.0, seed)
    times = np.arange(0, traj.config.sim_time, 0.01)
    for t in times:
        q, _ = traj(float(t))
        for i, ax in enumerate(("x", "y", "z")):
            lo, hi = ts.joint_limits[ax]
            assert q[i] >= lo - EPS, f"seed={seed} t={t:.2f} {ax} q={q[i]:.4f} < lo={lo}"
            assert q[i] <= hi + EPS, f"seed={seed} t={t:.2f} {ax} q={q[i]:.4f} > hi={hi}"


@pytest.mark.parametrize("seed", range(10))
def test_ptp_positions_in_limits(seed):
    ts = load_trajectory_settings(SETTINGS)
    traj = make_ptp_trajectory(ts, 15.0, seed)
    times = np.arange(0, traj.config.sim_time, 0.01)
    for t in times:
        q, _ = traj(float(t))
        for i, ax in enumerate(("x", "y", "z")):
            lo, hi = ts.joint_limits[ax]
            assert q[i] >= lo - EPS
            assert q[i] <= hi + EPS
