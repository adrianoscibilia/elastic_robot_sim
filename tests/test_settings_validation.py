"""validate_trajectory_settings must reject bad configs."""

import os
import sys

import pytest
import yaml

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_REPO, "src"))

from elastic_sim.trajectory import TrajectorySettings, validate_trajectory_settings

_GOOD = TrajectorySettings(
    joint_limits={"x": (-1.8, 1.8), "y": (-1.8, 1.8), "z": (-1.0, 1.0)},
    peak_cartesian_velocity_ms=0.5,
    axis_velocity_limits_ms={"x": None, "y": None, "z": None},
    ramp_tau=1.0,
    amplitude_fraction=0.4,
    amplitude_min_m=0.2,
    freq_min=0.5,
    freq_max=3.0,
    step_duration=2.0,
    min_distance=0.2,
    ros_points_per_segment=20,
)


def test_valid_settings_pass():
    validate_trajectory_settings(_GOOD)


def test_freq_min_ge_freq_max_raises():
    bad = TrajectorySettings(**{**_GOOD.__dict__, "freq_min": 3.0, "freq_max": 0.5})
    with pytest.raises(AssertionError):
        validate_trajectory_settings(bad)


def test_zero_peak_velocity_raises():
    bad = TrajectorySettings(**{**_GOOD.__dict__, "peak_cartesian_velocity_ms": 0.0})
    with pytest.raises(AssertionError):
        validate_trajectory_settings(bad)


def test_bad_joint_limits_raises():
    bad = TrajectorySettings(**{**_GOOD.__dict__, "joint_limits": {"x": (1.8, -1.8), "y": (-1.8, 1.8), "z": (-1.0, 1.0)}})
    with pytest.raises(AssertionError):
        validate_trajectory_settings(bad)


def test_amplitude_min_too_large_raises():
    # amplitude_fraction=0.4, z range=2.0 → max_amp_z=0.8; amplitude_min_m=1.0 > 0.8
    bad = TrajectorySettings(**{**_GOOD.__dict__, "amplitude_min_m": 1.0})
    with pytest.raises(AssertionError):
        validate_trajectory_settings(bad)
