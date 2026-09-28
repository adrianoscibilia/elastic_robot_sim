"""Kinematics and viewer helpers of the live pipeline.

The experiment/plotting/MoveIt half of the historical
``test_kinematics_and_plotting.py`` is archived in ``legacy/tests`` (R5_10 T-6).
"""
from pathlib import Path

import numpy as np

from elastic_sim.assets import AssetRegistry
from elastic_sim.kinematics import PortableKinematics, kinematic_groups
from elastic_sim.visualization import _frame_segments


ROOT = Path(__file__).resolve().parents[1]


def test_every_asset_declares_valid_kinematic_groups():
    registry = AssetRegistry.for_repository(ROOT)
    for name in ("fmrr_tecnobody", "ur10", "tiago_pro_dual", "kuka_lbr_iiwa_7_r800", "kuka_lbr_iiwa_14_r820"):
        asset = registry.load(name)
        groups = kinematic_groups(asset)
        assert groups
        assert all(set(group.joints).issubset(asset.joint_names) for group in groups)


def test_collision_path_validator_reports_clearance():
    asset = AssetRegistry.for_repository(ROOT).load("kuka_lbr_iiwa_7_r800")
    kin = PortableKinematics(asset)
    q = np.vstack((kin.neutral(), kin.neutral()))
    report = kin.validate_path(q, margin=0.005)
    assert report.valid
    assert report.minimum_distance > 0.005
    assert report.closest_pair is not None


def test_collision_validation_recursively_subdivides_large_steps():
    asset = AssetRegistry.for_repository(ROOT).load("fmrr_tecnobody")
    kin = PortableKinematics(asset)
    report = kin.validate_path(np.asarray(((0.0, 0.0, 0.0), (0.2, 0.0, 0.0))), max_joint_step=0.05)
    assert report.checked_configurations == 5


def test_frame_segments_draw_the_pose_axes():
    starts, ends = _frame_segments(np.asarray((1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0)))
    assert np.allclose(starts, (1.0, 2.0, 3.0))
    assert np.allclose(ends - starts, np.eye(3) * 0.07)
