"""Executed peak velocity must not exceed the configured limit (INV-4 part 1)."""

import os

import numpy as np
import pytest
import yaml

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SETTINGS = yaml.safe_load(open(os.path.join(_REPO, "config", "settings.yaml"), encoding="utf-8"))

import sys
sys.path.insert(0, os.path.join(_REPO, "src"))

from elastic_sim.trajectory import (
    load_trajectory_settings,
    make_ptp_trajectory,
    make_sinusoidal_trajectory,
)


def _peak(traj, sim_time, dt=0.01):
    t_arr = np.arange(0, sim_time, dt)
    dq = np.array([traj(float(t))[1] for t in t_arr])
    return float(np.max(np.linalg.norm(dq, axis=1)))


@pytest.mark.parametrize("seed", range(20))
def test_sin_peak_under_limit(seed):
    ts = load_trajectory_settings(SETTINGS)
    traj = make_sinusoidal_trajectory(ts, 15.0, seed)
    assert _peak(traj, traj.config.sim_time) <= ts.peak_cartesian_velocity_ms * (1 + 1e-6)
    assert traj.config.speed_override == 100.0
    assert traj.config.executed_peak_velocity_ms <= ts.peak_cartesian_velocity_ms * (1 + 1e-6)


@pytest.mark.parametrize("seed", range(20))
def test_ptp_peak_under_limit(seed):
    ts = load_trajectory_settings(SETTINGS)
    traj = make_ptp_trajectory(ts, 15.0, seed)
    assert _peak(traj, traj.config.sim_time) <= ts.peak_cartesian_velocity_ms * (1 + 1e-6)
    assert traj.config.speed_override == 100.0
    assert traj.config.executed_peak_velocity_ms <= ts.peak_cartesian_velocity_ms * (1 + 1e-6)
