"""Simulation-only excitation and identification-dataset invariants.

These lock in the properties that make a generated dataset usable for dynamic
model identification: the two simulators and Pinocchio agree on the model, the
recorded torque is exactly the applied one, and a least-squares fit recovers
the robot's base parameters from the generated data.
"""

import os
import sys

import numpy as np
import pytest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_REPO, "src"))

from elastic_sim import excitation as exc
from elastic_sim import identification as idn
from elastic_sim.assets import AssetRegistry
from elastic_sim.kinematics import PortableKinematics
from elastic_sim.scene import compose_table_scene, robot_root_link
from elastic_sim.torque_runners import (
    ComputedTorqueController,
    SeaMotorController,
    TransmissionSpec,
    run_mujoco_elastic_torque,
    run_mujoco_torque,
    run_newton_elastic_torque,
)

ASSET = "kuka_lbr_iiwa_14_r820_table"
TABLE_HEIGHT = 0.75


@pytest.fixture(scope="module")
def asset():
    spec = AssetRegistry.for_repository(_REPO).load(ASSET)
    spec.resolve_active_joints()
    return spec


@pytest.fixture(scope="module")
def model(asset):
    return idn.build_model(asset)


@pytest.fixture(scope="module")
def short_trajectory(asset):
    """A brief but genuinely exciting trajectory, to keep the suite quick."""
    config = exc.FourierExcitationConfig(
        n_harmonics=4, base_frequency=0.5, time_step=0.002, max_acceleration=2.0
    )
    return exc.optimize_excitation(asset, config, seed=3, n_candidates=6)


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

def test_table_scene_mounts_robot_at_table_height(asset):
    mujoco = pytest.importorskip("mujoco")
    from elastic_sim.generic_mujoco_runner import _build_model

    built, _ = _build_model(asset, mujoco, 0.002)
    data = mujoco.MjData(built)
    mujoco.mj_forward(built, data)
    body = mujoco.mj_name2id(built, mujoco.mjtObj.mjOBJ_BODY, "iiwa_link_1")
    # iiwa_A1 sits 0.1475 m above the robot's base plate.
    assert data.xpos[body][2] == pytest.approx(TABLE_HEIGHT + 0.1475, abs=1e-6)


def test_table_is_a_collision_obstacle(asset):
    kinematics = PortableKinematics(asset)
    pairs = set()
    for pair in kinematics.collision_model.collisionPairs:
        first = kinematics.collision_model.geometryObjects[pair.first]
        second = kinematics.collision_model.geometryObjects[pair.second]
        pairs.add(frozenset((
            kinematics.model.frames[first.parentFrame].name,
            kinematics.model.frames[second.parentFrame].name,
        )))
    assert any("table" in pair for pair in pairs), "table must be collision-checked against the arm"
    penetrating = np.array([0.0, -2.09, 0.0, 1.219, 0.0, 0.0, 0.0])
    assert not kinematics.collision_report([penetrating], margin=0.01).valid
    home = np.asarray(asset.metadata["default_configuration"], dtype=float)
    assert kinematics.collision_report([home], margin=0.01).valid


def test_compose_table_scene_rejects_bad_geometry(asset):
    source = AssetRegistry.for_repository(_REPO).load("kuka_lbr_iiwa_14_r820")
    with pytest.raises(ValueError):
        compose_table_scene(source, output_path="scene.urdf", table_height=-1.0)
    text = compose_table_scene(source, output_path="scene.urdf", table_height=0.5)
    assert "<box" in text and 'name="table"' in text


def test_robot_root_link_is_unique(asset):
    from xml.etree import ElementTree as ET

    assert robot_root_link(ET.parse(asset.urdf_path).getroot()) == "table"


# ---------------------------------------------------------------------------
# Analytic agreement
# ---------------------------------------------------------------------------

def test_mujoco_inverse_dynamics_matches_pinocchio(asset, model):
    mujoco = pytest.importorskip("mujoco")
    from elastic_sim.generic_mujoco_runner import _build_model, _joint_addresses
    from elastic_sim.torque_runners import neutralize_mujoco_passive

    pin, pin_model, pin_data = model
    built, _ = _build_model(asset, mujoco, 0.002)
    neutralize_mujoco_passive(built)
    data = mujoco.MjData(built)
    active = _joint_addresses(built, mujoco, tuple(asset.joint_names))
    rng = np.random.default_rng(11)
    for _ in range(5):
        q = pin.randomConfiguration(pin_model)
        dq = rng.normal(size=pin_model.nv)
        ddq = rng.normal(size=pin_model.nv)
        for index, (qpos, dof) in enumerate(active):
            data.qpos[qpos] = q[index]
            data.qvel[dof] = dq[index]
            data.qacc[dof] = ddq[index]
        mujoco.mj_inverse(built, data)
        got = np.asarray([data.qfrc_inverse[dof] for _, dof in active])
        assert np.abs(got - pin.rnea(pin_model, pin_data, q, dq, ddq)).max() < 1e-6


def test_regressor_reproduces_inverse_dynamics(asset, model):
    pin, pin_model, pin_data = model
    friction = idn.FrictionModel.from_asset(asset)
    theta = np.concatenate([idn.standard_parameters(pin, pin_model), friction.viscous, friction.coulomb])
    rng = np.random.default_rng(5)
    for _ in range(5):
        q = pin.randomConfiguration(pin_model)
        dq = rng.normal(size=pin_model.nv)
        ddq = rng.normal(size=pin_model.nv)
        regressor = idn.joint_torque_regressor(pin, pin_model, pin_data, q, dq, ddq, include_friction=True)
        expected = idn.inverse_dynamics(pin, pin_model, pin_data, q, dq, ddq, friction=friction)
        assert np.abs(regressor @ theta - expected).max() < 1e-9


def test_friction_comes_from_urdf_dynamics_tags(asset):
    friction = idn.FrictionModel.from_asset(asset)
    assert np.allclose(friction.viscous, 10.0)
    assert np.allclose(friction.coulomb, 0.1)


def test_base_parameter_count(asset, model):
    pin, pin_model, pin_data = model
    assert idn.base_parameter_basis(pin, pin_model, pin_data, n_samples=200, seed=0).shape == (70, 43)
    with_friction = idn.base_parameter_basis(
        pin, pin_model, pin_data, n_samples=200, seed=0, include_friction=True
    )
    assert with_friction.shape == (84, 57)


# ---------------------------------------------------------------------------
# Excitation
# ---------------------------------------------------------------------------

def test_excitation_starts_and_ends_at_rest(short_trajectory):
    for edge in (0, -1):
        assert np.abs(short_trajectory.velocity[edge]).max() < 1e-9
        assert np.abs(short_trajectory.acceleration[edge]).max() < 1e-9


def test_excitation_respects_joint_limits(asset, short_trajectory):
    config = exc.FourierExcitationConfig(n_harmonics=4, base_frequency=0.5, time_step=0.002,
                                         max_acceleration=2.0)
    lower, upper, velocity = exc.joint_bounds(asset, config)
    assert (short_trajectory.position >= lower - 1e-9).all()
    assert (short_trajectory.position <= upper + 1e-9).all()
    assert (np.abs(short_trajectory.velocity) <= velocity + 1e-9).all()
    assert np.abs(short_trajectory.acceleration).max() <= 2.0 + 1e-9


def test_excitation_beats_random_point_to_point(asset, model):
    """The optimized Fourier series must condition the regressor better."""
    from elastic_sim.serial_trajectory import SerialArmTrajectory, generate_serial_arm_trajectory

    pin, pin_model, pin_data = model
    basis = idn.base_parameter_basis(pin, pin_model, pin_data, n_samples=200, seed=1, include_friction=True)

    def condition(q, dq, ddq):
        return idn.regressor_condition(pin, pin_model, pin_data, q, dq, ddq, basis, include_friction=True)

    baseline = []
    for seed in range(3):
        config = generate_serial_arm_trajectory(
            asset.urdf_path, joint_names=asset.joint_names, num_waypoints=8,
            max_velocity=0.8, max_acceleration=1.5, seed=seed,
        )
        samples = SerialArmTrajectory(config).sample(0.02)
        baseline.append(condition(samples["q"], samples["dq"], samples["ddq"]))

    trajectory = exc.optimize_excitation(
        asset, exc.FourierExcitationConfig(n_harmonics=5, base_frequency=0.1, time_step=0.02),
        seed=1, n_candidates=12,
    )
    samples = trajectory.sample()
    optimized = condition(samples["q"], samples["dq"], samples["ddq"])
    assert optimized < np.median(baseline)


def test_excitation_round_trips_through_metadata(asset, short_trajectory):
    rebuilt = exc.trajectory_from_metadata(
        asset, short_trajectory.metadata, time_step=float(np.diff(short_trajectory.time)[0])
    )
    assert np.abs(rebuilt.position - short_trajectory.position).max() < 1e-9


# ---------------------------------------------------------------------------
# Torque-driven rollouts
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rigid_rollout(asset, short_trajectory):
    pytest.importorskip("mujoco")
    friction = idn.FrictionModel.from_asset(asset)
    controller = ComputedTorqueController(asset, short_trajectory, friction=friction, natural_frequency=25.0)
    return run_mujoco_torque(asset, short_trajectory, controller, time_step=5e-4, friction=friction)


def test_rigid_rollout_tracks_the_reference(rigid_rollout):
    error = np.abs(rigid_rollout["q_link"] - rigid_rollout["q_ref"]).max()
    assert error < 1e-3, f"computed torque should track closely, got {error:.2e}"


def test_recorded_torque_is_the_applied_torque(rigid_rollout):
    """The label must be the injected torque, never a controller command.

    ``tau_motor`` is recorded as ``tau_feedforward + tau_feedback`` by
    construction; this guards against a refactor reintroducing a PD command
    or a solver-side re-computation as the label.
    """
    total = rigid_rollout["tau_feedforward"] + rigid_rollout["tau_feedback"]
    assert np.abs(rigid_rollout["tau_motor"] - total).max() < 1e-12


def test_rigid_rollout_is_consistent_with_inverse_dynamics(asset, model, rigid_rollout):
    pin, pin_model, pin_data = model
    friction = idn.FrictionModel.from_asset(asset)
    q, dq, ddq = rigid_rollout["q_link"], rigid_rollout["dq_link"], rigid_rollout["ddq_link"]
    predicted = np.asarray([
        idn.inverse_dynamics(pin, pin_model, pin_data, q[i], dq[i], ddq[i], friction=friction)
        for i in range(0, len(q), 20)
    ])
    residual = rigid_rollout["tau_motor"][::20] - predicted
    assert np.sqrt(np.mean(residual**2)) < 1e-6


def test_feedback_is_a_small_share_of_the_torque(rigid_rollout):
    """A dataset dominated by the controller would not describe the robot."""
    ratio = np.mean(np.abs(rigid_rollout["tau_feedback"])) / np.mean(np.abs(rigid_rollout["tau_feedforward"]))
    assert ratio < 0.05


def test_base_parameters_are_recoverable_from_a_rollout(asset, model, rigid_rollout):
    """The acceptance test: the dataset must identify the robot."""
    pin, pin_model, pin_data = model
    friction = idn.FrictionModel.from_asset(asset)
    basis = idn.base_parameter_basis(pin, pin_model, pin_data, n_samples=300, seed=0, include_friction=True)
    truth = basis.T @ np.concatenate([
        idn.standard_parameters(pin, pin_model), friction.viscous, friction.coulomb
    ])
    stride = slice(None, None, 5)
    estimate, residual = idn.identify_base_parameters(
        pin, pin_model, pin_data,
        rigid_rollout["q_link"][stride], rigid_rollout["dq_link"][stride],
        rigid_rollout["ddq_link"][stride], rigid_rollout["tau_motor"][stride].reshape(-1),
        basis, include_friction=True,
    )
    assert residual < 1e-6
    relative = np.abs(estimate - truth) / np.maximum(np.abs(truth), 1e-9)
    assert relative.max() < 1e-4


@pytest.mark.parametrize("backend", ["mujoco", "newton"])
def test_backends_agree_on_the_same_applied_torque(asset, short_trajectory, backend, rigid_rollout):
    """Newton uses Featherstone, an implementation independent of MuJoCo."""
    if backend == "mujoco":
        pytest.skip("mujoco is the reference rollout")
    pytest.importorskip("newton")
    from elastic_sim.torque_runners import run_newton_torque

    friction = idn.FrictionModel.from_asset(asset)
    controller = ComputedTorqueController(asset, short_trajectory, friction=friction, natural_frequency=25.0)
    other = run_newton_torque(asset, short_trajectory, controller, time_step=5e-4, friction=friction)
    assert other["solver"] == "SolverFeatherstone"
    assert np.abs(other["q_link"] - rigid_rollout["q_link"]).max() < 1e-4


# ---------------------------------------------------------------------------
# Elastic tiers
# ---------------------------------------------------------------------------

def test_transmission_rejects_an_unresolvable_time_step():
    transmission = TransmissionSpec.uniform(7, 1.0e6, damping_ratio=0.1, rotor_inertia=0.1)
    with pytest.raises(ValueError, match="cannot resolve"):
        transmission.require_stable_step(0.002)
    transmission.require_stable_step(transmission.required_time_step())


def test_stiff_transmission_reproduces_the_rigid_case(asset, short_trajectory, rigid_rollout):
    """The near-rigid limit: as stiffness grows the elastic tier must converge."""
    pytest.importorskip("mujoco")
    friction = idn.FrictionModel.from_asset(asset)
    errors = {}
    for stiffness in (1.0e4, 1.0e6):
        transmission = TransmissionSpec.uniform(7, stiffness, damping_ratio=0.1, rotor_inertia=0.1)
        controller = SeaMotorController(
            asset, short_trajectory, transmission, friction=friction, natural_frequency=25.0
        )
        result = run_mujoco_elastic_torque(
            asset, short_trajectory, controller, transmission,
            time_step=min(5e-4, transmission.required_time_step()), friction=friction,
        )
        on_grid = np.column_stack([
            np.interp(rigid_rollout["time"], result["time"], result["q_link"][:, j])
            for j in range(len(asset.joint_names))
        ])
        errors[stiffness] = float(np.sqrt(np.mean((on_grid - rigid_rollout["q_link"]) ** 2)))
    assert errors[1.0e6] < 1e-3
    assert errors[1.0e6] < errors[1.0e4]


@pytest.mark.parametrize("backend", ["mujoco", "newton"])
def test_elastic_link_torque_differs_from_motor_torque(asset, short_trajectory, backend):
    """Otherwise the learning target would just be a copy of the input."""
    pytest.importorskip(backend)
    runner = run_mujoco_elastic_torque if backend == "mujoco" else run_newton_elastic_torque
    friction = idn.FrictionModel.from_asset(asset)
    transmission = TransmissionSpec.uniform(7, 4.0e4, damping_ratio=0.1, rotor_inertia=0.1)
    controller = SeaMotorController(
        asset, short_trajectory, transmission, friction=friction, natural_frequency=25.0
    )
    result = runner(
        asset, short_trajectory, controller, transmission,
        time_step=min(5e-4, transmission.required_time_step()), friction=friction,
    )
    difference = np.sqrt(np.mean((result["tau_link"] - result["tau_motor"]) ** 2))
    assert difference > 0.1
    deflection = np.abs(result["q_link"] - result["q_motor"]).max()
    assert 1e-5 < deflection < 0.1
    tracking = np.sqrt(np.mean((result["q_motor"] - result["q_ref"]) ** 2))
    assert tracking < 1e-2, f"{backend} elastic motor tracking degraded: {tracking:.2e}"
    if backend == "newton":
        # Documented limitation: Newton's own solvers diverge on the elastic
        # chain, so this path runs MuJoCo's engine and is not a second opinion.
        assert result["independent_of_mujoco"] is False


# ---------------------------------------------------------------------------
# Dataset export
# ---------------------------------------------------------------------------

def test_dataset_frame_matches_the_consumer_contract(asset, short_trajectory, rigid_rollout):
    from elastic_sim.dataset import Tier, rollout_frame

    friction = idn.FrictionModel.from_asset(asset)
    frame = rollout_frame(
        asset, short_trajectory, rigid_rollout, bag="bag0", tier=Tier("rigid"),
        backend="mujoco", friction=friction, resample_step=0.002,
    )
    n = len(asset.joint_names)
    for prefix in ("q", "dq", "tau", "ft"):
        for index in range(n):
            assert f"{prefix}{index}" in frame.columns
    assert f"ft{n}" not in frame.columns, "target width must equal the joint count"
    assert "ddq0" not in frame.columns, "the consumer recomputes acceleration"
    assert frame["bag"].nunique() == 1
    steps = np.diff(frame["t"].to_numpy())
    assert (steps > 0).all() and steps.std() < 1e-12, "uniform, increasing time is required"


def test_default_config_loads_and_is_shared_by_both_scripts():
    from elastic_sim.dataset import DEFAULT_CONFIG, load_config

    config = load_config(os.path.join(_REPO, DEFAULT_CONFIG))
    assert config.asset == ASSET
    assert set(config.backends) <= {"mujoco", "newton"}
    assert any(tier.is_rigid for tier in config.tiers)
    assert any(not tier.is_rigid for tier in config.tiers)
    assert config.excitation.time_step == config.sample_time_step
    assert config.output.endswith(".csv")
    assert config.visualize is False


def test_config_rejects_unknown_keys(tmp_path):
    from elastic_sim.dataset import load_config

    path = tmp_path / "bad.yaml"
    path.write_text("asset: x\nnonsense: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys"):
        load_config(path)


def test_config_can_select_rigid_only(tmp_path):
    from elastic_sim.dataset import load_config

    path = tmp_path / "rigid.yaml"
    path.write_text("asset: x\ntiers:\n  rigid: true\n  stiffness: []\n", encoding="utf-8")
    tiers = load_config(path).tiers
    assert len(tiers) == 1 and tiers[0].is_rigid


def test_config_propagates_transmission_settings(tmp_path):
    from elastic_sim.dataset import load_config

    path = tmp_path / "t.yaml"
    path.write_text(
        "asset: x\ntiers:\n  rigid: false\n  stiffness: [1000.0]\n"
        "  transmission_damping_ratio: 0.3\n  rotor_inertia: 0.25\n",
        encoding="utf-8",
    )
    tier = load_config(path).tiers[0]
    assert tier.stiffness == 1000.0
    assert tier.transmission_damping_ratio == 0.3
    assert tier.rotor_inertia == 0.25
    assert np.allclose(tier.transmission(7).rotor_inertia, 0.25)


def test_condition_order_interleaves_tiers():
    """The consumer splits train/test as contiguous halves of the file."""
    from elastic_sim.dataset import DatasetConfig, Tier, iter_conditions

    config = DatasetConfig(
        backends=("mujoco",), tiers=(Tier("rigid"), Tier("soft", stiffness=1.0e4)),
        n_trajectories=2, n_friction_samples=1,
    )
    tiers = [tier.name for _, tier, _, _ in iter_conditions(config)]
    halfway = len(tiers) // 2
    assert set(tiers[:halfway]) == set(tiers[halfway:]), "each half must see every tier"
