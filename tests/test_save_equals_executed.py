"""Saved config must reproduce exactly the executed trajectory (INV-4 part 2)."""

import os
import sys
import tempfile

import numpy as np
import yaml

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_REPO, "src"))

SETTINGS = yaml.safe_load(open(os.path.join(_REPO, "config", "settings.yaml"), encoding="utf-8"))

from elastic_sim.trajectory import (
    Trajectory,
    load_trajectory_settings,
    make_sinusoidal_trajectory,
    make_ptp_trajectory,
)


def test_sin_roundtrip_matches():
    ts = load_trajectory_settings(SETTINGS)
    traj = make_sinusoidal_trajectory(ts, 15.0, seed=3)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "traj.json")
        traj.save(p)
        reloaded = Trajectory.load(p)
    assert abs(reloaded.config.sim_time - traj.config.sim_time) < 1e-9
    for t in np.linspace(0, traj.config.sim_time, 50):
        _, dq_a = traj(float(t))
        _, dq_b = reloaded(float(t))
        assert np.allclose(dq_a, dq_b, atol=1e-9)


def test_ptp_roundtrip_matches():
    ts = load_trajectory_settings(SETTINGS)
    traj = make_ptp_trajectory(ts, 15.0, seed=7)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "traj.json")
        traj.save(p)
        reloaded = Trajectory.load(p)
    assert abs(reloaded.config.sim_time - traj.config.sim_time) < 1e-9
    for t in np.linspace(0, traj.config.sim_time, 50):
        _, dq_a = traj(float(t))
        _, dq_b = reloaded(float(t))
        assert np.allclose(dq_a, dq_b, atol=1e-9)
