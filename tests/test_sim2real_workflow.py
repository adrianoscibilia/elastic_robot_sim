from pathlib import Path

import numpy as np
import pandas as pd

from elastic_sim.experiment import (
    ExperimentStore,
    artifact_root,
    config_digest,
    generate_materialized_trajectory,
    load_experiment_config,
    resolve_asset,
    rollout_to_frame,
)
from elastic_sim.materialized import MaterializedTrajectory
from elastic_sim.parameter_registry import ParameterRegistry, apply_parameter_overrides
from elastic_sim.ros_experiment import _action_groups, ros_topics
from elastic_sim.sim2real import discover_experiment_records


ROOT = Path(__file__).resolve().parents[1]


def test_materialized_round_trip_and_rollout_alignment(tmp_path):
    trajectory = MaterializedTrajectory(
        np.array([0.0, 0.1, 0.2]),
        np.array([[0.0], [1.0], [2.0]]),
        np.ones((3, 1)),
        ("joint_a",),
        np.zeros((3, 1)),
        {"seed": 4},
    )
    path = tmp_path / "trajectory.json"
    trajectory.save(path)
    restored = MaterializedTrajectory.load(path)
    assert restored.digest() == trajectory.digest()
    result = {"time": np.array([0.0, 0.1, 0.2]), "q_link": np.array([[0.0], [1.0], [2.0]]), "dq_link": np.ones((3, 1)), "tau_motor": np.zeros((3, 1))}
    frame = rollout_to_frame(trajectory, result, source="sim")
    assert frame["q_ref__joint_a"].tolist() == [0.0, 1.0, 2.0]
    assert frame["q__joint_a"].tolist() == [0.0, 1.0, 2.0]


def test_all_sim2real_configs_materialize_dynamic_joint_order():
    for path in sorted((ROOT / "config" / "assets").glob("*_sim2real.yaml")):
        config = load_experiment_config(path)
        asset = resolve_asset(config, path)
        trajectory = generate_materialized_trajectory(asset, config, 10)
        assert trajectory.joint_names == asset.joint_names
        assert trajectory.position.shape[1] == len(asset.joint_names)
        assert np.all(np.diff(trajectory.time) > 0)


def test_parameter_registry_supports_named_transmission_and_body_overrides():
    config = {"model": {"default_stiffness": 10.0, "default_damping": 1.0, "transmissions": {}}, "calibration": {}}
    registry = ParameterRegistry.from_config(config, ("joint_a",))
    assert "transmission.joint_a.stiffness" in {item.name for item in registry.specs}
    params = registry.decode(registry.initial)
    changed = apply_parameter_overrides(config, {"transmission.joint_a.stiffness": 50.0, "body.link.mass": 2.0})
    assert changed["model"]["transmissions"]["joint_a"]["stiffness"] == 50.0
    assert changed["model"]["body_overrides"]["link"]["mass"] == 2.0
    assert np.isclose(params["transmission.joint_a.stiffness"], 10.0)


def test_store_discovery_requires_real_pair_and_hash(tmp_path):
    config = {"asset": "test", "trajectory": {"duration": 1.0}}
    trajectory = MaterializedTrajectory(np.array([0.0, 1.0]), np.zeros((2, 1)), np.zeros((2, 1)), ("joint_a",), np.zeros((2, 1)), {"config_hash": config_digest(config)})
    root = tmp_path / "recorded"
    store = ExperimentStore(root, "test", "20260101T000000Z")
    store.save_trajectory(trajectory)
    store.save_frame(pd.DataFrame({"trajectory_id": [0, 0], "t": [0.0, 1.0], "q__joint_a": [0.0, 0.0], "dq__joint_a": [0.0, 0.0]}), "real.parquet")
    store.save_manifest({"asset": "test", "config_hash": config_digest(config), "completion_status": "complete"})
    records = discover_experiment_records(root, "test", config_digest(config))
    assert len(records) == 1
    assert len(records[0].trajectories) == 1


def test_artifact_roots_are_kind_first_and_reject_mixed_paths(tmp_path):
    config = {"paths": {"simulated_root": str(tmp_path / "simulated")}}
    assert artifact_root(config, "simulated", ROOT).name == "simulated"
    import pytest
    with pytest.raises(ValueError, match="must end"):
        artifact_root(config, "recorded", ROOT, tmp_path / "simulated")
    with pytest.raises(ValueError, match="invalid run-id"):
        ExperimentStore(tmp_path / "recorded", "robot", "../escape")


def test_ros_topic_manifest_contains_required_and_extra_topics():
    topics = ros_topics({"ros": {"action_server": "/controller/follow_joint_trajectory", "extra_topics": [{"name": "/imu", "type": "sensor_msgs/msg/Imu"}]}})
    assert "/joint_states" in topics
    assert "/controller/follow_joint_trajectory/_action/feedback" in topics
    assert "/imu" in topics


def test_multi_controller_actions_cover_all_joints_and_are_bagged():
    config = {"ros": {"action_servers": [
        {"name": "/left/follow_joint_trajectory", "joints": ["left_1", "left_2"]},
        {"name": "/right/follow_joint_trajectory", "joints": ["right_1", "right_2"]},
    ]}}
    groups = _action_groups(config, ("left_1", "left_2", "right_1", "right_2"))
    assert groups == [
        ("/left/follow_joint_trajectory", (0, 1)),
        ("/right/follow_joint_trajectory", (2, 3)),
    ]
    topics = ros_topics(config)
    assert "/left/follow_joint_trajectory/_action/feedback" in topics
    assert "/right/follow_joint_trajectory/_action/feedback" in topics
