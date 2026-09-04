from pathlib import Path
import argparse
import importlib.util
import sys
import types

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from elastic_sim.assets import AssetRegistry
from elastic_sim.experiment import _time_parameterize, generate_materialized_trajectory, load_experiment_config, resolve_asset
from elastic_sim.kinematics import PortableKinematics, enrich_rollout_frame, kinematic_groups
from elastic_sim.moveit_validation import validate_with_moveit
from elastic_sim.plotting import plot_rollouts
from elastic_sim.visualization import MujocoVisualizer, NewtonVisualizer, _frame_segments


ROOT = Path(__file__).resolve().parents[1]


def test_every_asset_declares_valid_kinematic_groups():
    registry = AssetRegistry.for_repository(ROOT)
    for name in ("fmrr_tecnobody", "ur10", "tiago_pro_dual", "kuka_lbr_iiwa_7_r800", "kuka_lbr_iiwa_14_r820"):
        asset = registry.load(name)
        groups = kinematic_groups(asset)
        assert groups
        assert all(set(group.joints).issubset(asset.joint_names) for group in groups)


def test_iiwa_fk_and_hold_cartesian_ik_round_trip():
    path = ROOT / "config/assets/kuka_lbr_iiwa_7_r800_sim2real.yaml"
    config = load_experiment_config(path)
    asset = resolve_asset(config, path)
    config["trajectory"] = {
        **config["trajectory"], "space": "cartesian", "mode": "hold",
        "duration": 0.1, "time_step": 0.02, "max_velocity": 1.0,
        "cartesian": {"groups": ["arm"], "max_iterations": 10},
    }
    trajectory = generate_materialized_trajectory(asset, config, 1)
    assert trajectory.metadata["ik"]["solver"] == "pink/proxqp"
    assert np.allclose(trajectory.position, trajectory.position[0])
    assert trajectory.metadata["collision"]["valid"]


def test_iiwa_cartesian_ptp_tracks_an_explicit_pose():
    path = ROOT / "config/assets/kuka_lbr_iiwa_7_r800_sim2real.yaml"
    config = load_experiment_config(path)
    asset = resolve_asset(config, path)
    kin = PortableKinematics(asset)
    initial = kin.neutral()
    initial[:] = asset.metadata["default_configuration"]
    target = kin.forward(initial, [kin.groups[0]])["arm"].copy()
    target[0] += 0.005
    config["trajectory"] = {
        **config["trajectory"], "space": "cartesian", "mode": "ptp",
        "duration": 0.1, "time_step": 0.02, "max_velocity": 1.0,
        "cartesian": {
            "groups": ["arm"], "initial_joint_positions": initial.tolist(),
            "waypoints": {"arm": [target.tolist()]}, "max_iterations": 40,
        },
    }
    trajectory = generate_materialized_trajectory(asset, config, 7)
    assert trajectory.metadata["ik"]["max_position_error"] < 1.0e-3
    assert trajectory.metadata["ik"]["maximum_joint_increment"] < 0.35
    assert trajectory.metadata["collision"]["valid"]


def test_tiago_coordinated_dual_arm_cartesian_hold():
    path = ROOT / "config/assets/tiago_pro_dual_sim2real.yaml"
    config = load_experiment_config(path)
    asset = resolve_asset(config, path)
    config["trajectory"] = {
        **config["trajectory"], "space": "cartesian", "mode": "hold",
        "duration": 0.04, "time_step": 0.02, "max_velocity": 1.0,
        "cartesian": {"groups": ["dual_arm"], "max_iterations": 5},
    }
    trajectory = generate_materialized_trajectory(asset, config, 4)
    assert trajectory.position.shape == (3, 14)
    assert trajectory.metadata["kinematic_groups"] == ["left_arm", "right_arm"]
    assert trajectory.metadata["collision"]["valid"]


def test_time_parameterization_stretches_time_instead_of_inconsistent_derivatives():
    time = np.linspace(0.0, 1.0, 101)
    q = np.column_stack((2.0 * time,))
    scaled_time, dq, ddq, stretch = _time_parameterize(q, time, 0.5, None)
    assert np.isclose(stretch, 4.0)
    assert np.isclose(scaled_time[-1], 4.0)
    assert np.max(np.abs(dq)) <= 0.5 * (1 + 1e-9)
    assert np.max(np.abs(ddq)) < 1e-8


def test_enrichment_and_plots_use_normalized_schema(tmp_path):
    path = ROOT / "config/assets/fmrr_tecnobody_sim2real.yaml"
    config = load_experiment_config(path)
    asset = resolve_asset(config, path)
    config["trajectory"] = {**config["trajectory"], "duration": 0.1, "time_step": 0.05, "mode": "hold"}
    trajectory = generate_materialized_trajectory(asset, config, 2)
    frame = pd.DataFrame({"t": trajectory.time})
    for index, name in enumerate(trajectory.joint_names):
        frame[f"q_ref__{name}"] = trajectory.position[:, index]
        frame[f"q__{name}"] = trajectory.position[:, index]
        frame[f"dq__{name}"] = trajectory.velocity[:, index]
        frame[f"tau_motor__{name}"] = 0.0
    enriched = enrich_rollout_frame(frame, trajectory, asset)
    assert "ee__cartesian__x" in enriched
    assert np.allclose(enriched["ee_position_error__cartesian"], 0.0)
    outputs = plot_rollouts({"test": enriched}, asset, output_dir=tmp_path, show=False)
    assert outputs and all(path.is_file() for path in outputs)


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


def test_moveit_validation_fails_closed_before_importing_ros():
    class StubTrajectory:
        joint_names = ("joint",)
        position = np.zeros((1, 1))

    with np.testing.assert_raises_regex(ValueError, "ros.moveit.group"):
        validate_with_moveit({"ros": {}}, StubTrajectory())


def test_native_viewer_adapter_lifecycles(monkeypatch):
    path = ROOT / "config/assets/fmrr_tecnobody_sim2real.yaml"
    config = load_experiment_config(path)
    asset = resolve_asset(config, path)
    config["trajectory"] = {**config["trajectory"], "duration": 0.02, "time_step": 0.02, "mode": "hold"}
    trajectory = generate_materialized_trajectory(asset, config, 2)

    class Handle:
        device = "cpu"

        def __init__(self, *_args, **_kwargs):
            self.running = True

        def set_model(self, _model):
            pass

        def is_running(self):
            return self.running

        def close(self):
            self.running = False

    monkeypatch.setitem(sys.modules, "newton", types.SimpleNamespace(viewer=types.SimpleNamespace(ViewerGL=Handle)))
    newton_viewer = NewtonVisualizer(asset, trajectory).open(object())
    assert newton_viewer.is_running()
    newton_viewer.close()
    assert not newton_viewer.is_running()

    import mujoco.viewer
    monkeypatch.setattr(mujoco.viewer, "launch_passive", lambda _model, _data: Handle())
    mujoco_viewer = MujocoVisualizer(asset, trajectory).open(object(), object())
    assert mujoco_viewer.is_running()
    mujoco_viewer.close()
    assert not mujoco_viewer.is_running()

    starts, ends = _frame_segments(np.asarray((1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0)))
    assert np.allclose(starts, (1.0, 2.0, 3.0))
    assert np.allclose(ends - starts, np.eye(3) * 0.07)


def test_simulation_only_is_ephemeral_unless_save(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("run_experiment_test", ROOT / "scripts/run_experiment.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    def fake_run(_asset, _config, trajectory, _backend, **_kwargs):
        return {
            "time": trajectory.time, "q_link": trajectory.position,
            "dq_link": trajectory.velocity, "q_motor": trajectory.position,
            "dq_motor": trajectory.velocity, "tau_motor": np.zeros_like(trajectory.position),
        }

    common = dict(
        config=str(ROOT / "config/assets/fmrr_tecnobody_sim2real.yaml"), sim_only=True,
        real_only=False, dry_run=False, backends=["mujoco"], no_motor_control=True,
        seed=3, run_id="ephemeral", simulated_root=str(tmp_path / "simulated"),
        recorded_root=None, headless=True, plot=False, csv=False, visualize=False,
        realtime_scale=1.0, moveit_validate=False,
    )
    monkeypatch.setattr(module, "run_simulation", fake_run)
    monkeypatch.setattr(module, "_args", lambda: argparse.Namespace(**common, save=False))
    assert module.main() == 0
    assert not (tmp_path / "simulated").exists()

    monkeypatch.setattr(module, "_args", lambda: argparse.Namespace(**common, save=True))
    assert module.main() == 0
    manifests = list((tmp_path / "simulated" / "fmrr_tecnobody").rglob("manifest.yaml"))
    assert len(manifests) == 1
