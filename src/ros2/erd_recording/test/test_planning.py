"""T1.5: WorldKinematics, re-limiting, FK identity, digests, approach validation."""

from __future__ import annotations

import copy
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pytest

from elastic_sim.assets import AssetRegistry
from elastic_sim.materialized import MaterializedTrajectory
from erd_recording.config import load_lab_config
from erd_recording.planning import (
    PlanError,
    WorldKinematics,
    build_plan,
    check_caps_on_analytic,
    fk_identity_check,
    ladder_scale_trajectory,
    load_plan,
    quintic_segment,
    relimited_asset,
)

FIXTURE = Path(__file__).resolve().parents[1] / "config" / "lab" / "iiwa_mock.yaml"


@pytest.fixture(scope="module")
def config():
    return load_lab_config(FIXTURE)


@pytest.fixture(scope="module")
def snapshot(config, tmp_path_factory):
    registry = AssetRegistry.for_repository()
    bare = registry.load(config.description.sim_asset)
    out = tmp_path_factory.mktemp("plan")
    return relimited_asset(bare, config, out)


def test_relimited_asset_bakes_real_velocity_and_effort(config, snapshot):
    tree = ET.parse(snapshot.urdf_path)
    limits = {j.get("name"): j.find("limit") for j in tree.getroot().findall("joint")}
    for joint, (lo, hi) in config.limits.position.items():
        limit = limits[joint]
        assert float(limit.get("velocity")) == pytest.approx(config.limits.velocity[joint])
        assert float(limit.get("effort")) == pytest.approx(config.limits.effort[joint])


def test_world_kinematics_detects_object_on_path(config, snapshot):
    world = WorldKinematics(snapshot, config.scene.objects, config.scene.allowed_contacts)
    home = np.asarray(config.poses.home)
    baseline = world.collision_report([home])
    assert baseline.valid

    # Deliberately place a large sphere at the flange's home-pose position:
    # any type-correct object-on-path check must now see it.
    pose = world.forward(home)
    tip_name = next(iter(pose))
    tip_xyz = pose[tip_name][:3]
    from erd_recording.config import SceneSphere

    blocking = SceneSphere("blocker", radius=0.15, xyz=tuple(tip_xyz))
    blocked_world = WorldKinematics(snapshot, config.scene.objects + (blocking,), config.scene.allowed_contacts)
    blocked = blocked_world.collision_report([home])
    assert not blocked.valid


def test_world_kinematics_respects_allowed_contacts(config, snapshot):
    from erd_recording.config import SceneBox

    home = np.asarray(config.poses.home)
    world = WorldKinematics(snapshot, config.scene.objects, ())
    pose = world.forward(home)
    tip_xyz = pose[next(iter(pose))][:3]
    touching = SceneBox("touching", size=(0.3, 0.3, 0.3), xyz=tuple(tip_xyz), rpy=(0.0, 0.0, 0.0))

    blocked_world = WorldKinematics(snapshot, config.scene.objects + (touching,), ())
    assert not blocked_world.collision_report([home]).valid

    # Naming every robot link that could touch "touching" as allowed clears it.
    link_names = tuple(sorted({snapshot.resolve_active_joints()[0].child}))
    allowed = tuple((frame.name, "touching") for frame in _all_frame_names(blocked_world))
    allowed_world = WorldKinematics(snapshot, config.scene.objects + (touching,), allowed)
    assert allowed_world.collision_report([home]).valid


def _all_frame_names(kinematics):
    return list(kinematics.model.frames)


def test_digest_reproducible_same_seed(config, tmp_path):
    bundle1 = build_plan(config, tmp_path / "run1", n_candidates=6)
    bundle2 = build_plan(config, tmp_path / "run2", n_candidates=6)
    assert [s.digest for s in bundle1.segments] == [s.digest for s in bundle2.segments]


def test_caps_hold_on_analytic_signal(config, tmp_path):
    bundle = build_plan(config, tmp_path / "run", n_candidates=6)
    arrays = config.limits.as_arrays(config.joint_order)
    excitation_segment = bundle.segment("excitation_0")
    # Should not raise: the plan already checked this during build_plan.
    check_caps_on_analytic(excitation_segment.trajectory, arrays["velocity"], arrays["acceleration"])
    # A deliberately tightened cap must now fail.
    with pytest.raises(PlanError, match="cap exceeded"):
        check_caps_on_analytic(excitation_segment.trajectory, arrays["velocity"] * 1e-6, arrays["acceleration"])


def test_standstill_segments_are_planned_collision_checked_and_go_home_between_poses(config, tmp_path):
    # RR_04 A-3: standstill approaches/returns are frozen plan segments (not
    # built ad hoc at run time), one approach+return pair per configured
    # standstill pose, each going home -> pose_k -> home.
    bundle = build_plan(config, tmp_path / "run", n_candidates=6)
    n_poses = len(config.poses.standstill)
    assert len(bundle.standstill_segments) == 2 * n_poses
    home = np.asarray(config.poses.home, dtype=float)
    for index, pose in enumerate(config.poses.standstill):
        approach = bundle.segment(f"standstill_approach_{index}")
        assert approach.kind == "standstill_approach"
        assert np.allclose(approach.trajectory.position[0], home, atol=1e-9)
        assert np.allclose(approach.trajectory.position[-1], np.asarray(pose), atol=1e-6)
        ret = bundle.segment(f"standstill_return_{index}")
        assert ret.kind == "standstill_return"
        assert np.allclose(ret.trajectory.position[0], np.asarray(pose), atol=1e-6)
        assert np.allclose(ret.trajectory.position[-1], home, atol=1e-9)


def test_standstill_segments_round_trip_through_load_plan(config, tmp_path):
    output_dir = tmp_path / "run"
    bundle = build_plan(config, output_dir, n_candidates=6)
    reloaded = load_plan(output_dir)
    assert [s.segment_id for s in reloaded.standstill_segments] == \
        [s.segment_id for s in bundle.standstill_segments]
    assert [s.digest for s in reloaded.standstill_segments] == [s.digest for s in bundle.standstill_segments]


def test_approach_segment_starts_and_ends_at_rest():
    start = np.zeros(3)
    end = np.array([1.0, -0.5, 0.2])
    trajectory = quintic_segment(start, end, velocity_limit=0.5, acceleration_limit=1.0, time_step=0.002)
    assert np.allclose(trajectory.velocity[0], 0.0, atol=1e-9)
    assert np.allclose(trajectory.velocity[-1], 0.0, atol=1e-9)
    assert np.allclose(trajectory.acceleration[0], 0.0, atol=1e-9)
    assert np.allclose(trajectory.acceleration[-1], 0.0, atol=1e-9)
    assert np.allclose(trajectory.position[0], start)
    assert np.allclose(trajectory.position[-1], end)
    assert np.max(np.abs(trajectory.velocity)) <= 0.5 * 1.001
    assert np.max(np.abs(trajectory.acceleration)) <= 1.0 * 1.001


def test_approach_segment_zero_span_is_stationary():
    start = np.array([0.1, 0.2])
    trajectory = quintic_segment(start, start.copy(), velocity_limit=0.5, acceleration_limit=1.0, time_step=0.002)
    assert np.allclose(trajectory.position, start)


def test_ladder_scale_trajectory_scales_about_its_own_centre():
    # RR_06 P3-3b: q = c + s(q - c) around the trajectory's own per-joint
    # mean, velocity/acceleration scaled by s, analytic dropped.
    time = np.linspace(0.0, 1.0, 5)
    position = np.array([[0.0], [1.0], [2.0], [1.0], [0.0]])
    velocity = np.array([[0.0], [1.0], [0.0], [-1.0], [0.0]])
    trajectory = MaterializedTrajectory(time=time, position=position, velocity=velocity, joint_names=("j0",))

    scaled = ladder_scale_trajectory(trajectory, 0.5)

    centre = position.mean(axis=0)
    assert np.allclose(scaled.position, centre + 0.5 * (position - centre))
    assert np.allclose(scaled.velocity, 0.5 * velocity)
    assert scaled.analytic is None
    assert scaled.metadata["ladder_scale"] == pytest.approx(0.5)


def test_build_plan_with_ladder_appends_re_validated_commissioning_segments(config, tmp_path):
    bundle = build_plan(config, tmp_path / "run", n_candidates=6, ladder_scale=0.25)

    full_scale = [s for s in bundle.segments if s.kind == "excitation"]
    commissioning = [s for s in bundle.segments if s.kind == "excitation_commissioning"]
    assert len(full_scale) == len(commissioning) == config.excitation.trajectories

    for base, ladder in zip(full_scale, commissioning):
        assert ladder.segment_id == f"{base.segment_id}_ladder"
        centre = base.trajectory.position.mean(axis=0)
        expected = centre + 0.25 * (base.trajectory.position - centre)
        assert np.allclose(ladder.trajectory.position, expected, atol=1e-9)
        assert np.allclose(ladder.trajectory.velocity, 0.25 * base.trajectory.velocity, atol=1e-9)

    approach_commissioning = [s for s in bundle.segments if s.kind == "approach_commissioning"]
    assert len(approach_commissioning) == config.excitation.trajectories
    return_commissioning = [s for s in bundle.segments if s.kind == "return_commissioning"]
    assert len(return_commissioning) == 1
    home = np.asarray(config.poses.home, dtype=float)
    assert np.allclose(return_commissioning[0].trajectory.position[-1], home, atol=1e-9)

    # Excluded from any dataset by construction: convert_cli's own segment
    # filter only ever looks for kind == "excitation".
    assert all(s.kind != "excitation" for s in commissioning + approach_commissioning + return_commissioning)


def test_fk_identity_passes_for_identical_asset(config, snapshot):
    report = fk_identity_check(snapshot, snapshot, n_samples=20)
    assert report["ok"]
    assert report["worst_axis_line_distance_m"] < 1e-9


@pytest.mark.parametrize("joint_index", [0, -1])
def test_fk_identity_catches_a_joint_sign_flip(config, snapshot, tmp_path, joint_index):
    tree = ET.parse(snapshot.urdf_path)
    root = tree.getroot()
    for joint in root.findall("joint"):
        if joint.get("name") == config.joint_order[joint_index]:
            axis = joint.find("axis")
            values = [float(v) for v in axis.get("xyz").split()]
            axis.set("xyz", " ".join(str(-v) for v in values))
    flipped_path = tmp_path / "flipped.urdf"
    tree.write(flipped_path)
    from elastic_sim.assets import AssetSpec

    flipped = AssetSpec(name=snapshot.name, urdf_path=flipped_path, active_joints=snapshot.active_joints,
                        metadata=copy.deepcopy(snapshot.metadata))
    with pytest.raises(PlanError, match="FK identity failed"):
        fk_identity_check(snapshot, flipped, n_samples=50, seed=1)


def test_media_flange_is_in_the_planning_snapshot(snapshot):
    root = ET.parse(snapshot.urdf_path).getroot()
    collision = root.find("link[@name='iiwa_link_ee']/collision[@name='erd_tool_0']")
    assert collision is not None
    assert float(collision.find('geometry/cylinder').get('length')) == pytest.approx(0.03)
    assert collision.find('origin').get('xyz') == '0 0 0.015'
