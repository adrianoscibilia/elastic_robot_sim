"""RR_04 A-5: per-joint JTC tolerances generated from the lab config."""

from __future__ import annotations

from pathlib import Path

import yaml

from erd_recording.config import load_lab_config
from erd_recording.tolerances import GOAL_TIME_S, GOAL_TOLERANCE_RAD, write_tolerance_overrides

FIXTURE = Path(__file__).resolve().parents[1] / "config" / "lab" / "iiwa_mock.yaml"


def test_write_tolerance_overrides_uses_abort_tracking_rad(tmp_path):
    config = load_lab_config(FIXTURE)
    driver_joints = tuple(config.description.joint_map.keys())
    output = write_tolerance_overrides(config, "erd_arm_controller", driver_joints, tmp_path / "tol.yaml")

    data = yaml.safe_load(output.read_text(encoding="utf-8"))
    constraints = data["erd_arm_controller"]["ros__parameters"]["constraints"]
    assert constraints["goal_time"] == GOAL_TIME_S
    for joint in driver_joints:
        assert constraints[joint]["trajectory"] == config.limits.abort.tracking_rad
        assert constraints[joint]["goal"] == GOAL_TOLERANCE_RAD
