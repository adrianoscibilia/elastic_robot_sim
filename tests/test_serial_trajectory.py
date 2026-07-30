import os
import sys

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_REPO, "src"))

from elastic_sim.serial_trajectory import SerialArmTrajectory, SerialTrajectoryConfig


def test_minimum_jerk_hits_waypoints_with_zero_endpoint_velocity():
    cfg = SerialTrajectoryConfig(
        joint_names=("joint_1", "joint_2"),
        waypoints=((0.0, 1.0), (1.0, -1.0), (0.5, 0.0)),
        segment_durations=(2.0, 1.0),
    )
    trajectory = SerialArmTrajectory(cfg)
    q0, dq0, _ = trajectory(0.0)
    q1, dq1, _ = trajectory(2.0)
    q2, dq2, _ = trajectory(3.0)
    assert np.allclose(q0, (0.0, 1.0))
    assert np.allclose(q1, (1.0, -1.0))
    assert np.allclose(q2, (0.5, 0.0))
    assert np.allclose(dq0, 0.0)
    assert np.allclose(dq1, 0.0)
    assert np.allclose(dq2, 0.0)


def test_serialization_roundtrip(tmp_path):
    cfg = SerialTrajectoryConfig(
        joint_names=("joint_1",), waypoints=((0.0,), (1.0,)), segment_durations=(1.5,)
    )
    path = tmp_path / "trajectory.json"
    cfg.save(path)
    assert SerialTrajectoryConfig.load(path) == cfg
