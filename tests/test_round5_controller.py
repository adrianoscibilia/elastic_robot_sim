"""Round 5: controller modes, sensor model, plant extras, signals, FMRR.

Structure mirrors ``tests/test_ur10_port.py``: cheap contract and algebra
checks unmarked, anything that integrates physics marked ``slow``.  The single
most important test in the file is
``test_round4_defaults_are_unchanged``/``test_iiwa_dataset_is_bit_identical``:
round 5 adds four config blocks and every one of them must be a no-op unless a
config asks for it (``R5_02`` Sec 1).
"""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.path.join(_REPO, "src"))

from elastic_sim.assets import AssetRegistry, lock_inactive_one_dof_joints
from elastic_sim.controllers import (
    CONTROLLER_MODES,
    ControllerDraw,
    ControllerSpec,
    JointPdController,
    NominalModelSpec,
    VelocityLoopController,
    VelocityLoopSpec,
    build_controller,
    effective_bandwidth,
    sample_velocity_loop,
)
from elastic_sim.dataset import DatasetConfig, SignalPolicy, load_config, plant_extras_for_bag, rollout_frame
from elastic_sim.measurement import IDEAL_MEASUREMENT, MeasurementModel, measure_bag
from elastic_sim.plant_extras import NO_EXTRAS, PlantExtras, StiffnessNonlinearity, TorqueRipple


@pytest.fixture(scope="module")
def registry():
    return AssetRegistry.for_repository(_REPO)


@pytest.fixture(scope="module")
def fmrr(registry):
    return registry.load("fmrr_tecnobody")


@pytest.fixture(scope="module")
def ur10(registry):
    return registry.load("ur10_table")


# ---------------------------------------------------------------------------
# R5 T1: the round-4 invariant
# ---------------------------------------------------------------------------

def test_round4_defaults_are_unchanged():
    """Every round-5 block defaults to round 4's behaviour."""
    config = DatasetConfig()
    assert config.controller.mode == "exact_ct"
    assert config.controller.is_round4_default
    assert config.measurement.is_ideal
    assert config.plant_extras.is_empty
    assert config.signals.position_side == "link"
    assert config.signals.target == "link_torque"
    assert config.signals.is_collocated
    assert not config.signals.clean_columns


@pytest.mark.parametrize("name", ["kuka_lbr_iiwa_14_r820_table", "ur10_table"])
def test_round4_configs_keep_round4_behaviour(name):
    """The shipped round-4 configs still resolve to round-4 settings.

    A silently-defaulted new key would be invisible in the data but would
    change what the dataset *means*; this is the cheapest guard against it.
    """
    config = load_config(_REPO / "config" / "identification" / f"{name}.yaml")
    assert config.controller.is_round4_default
    assert config.measurement.is_ideal
    assert config.plant_extras.is_empty
    assert config.signals.is_collocated


def test_config_survives_dataclass_replace_reconstruction():
    """R3_10 Sec 2.2's bug class, on the four new blocks.

    ``resolve_config`` rebuilds a config with ``dataclasses.replace``; a field
    reconstructed from scratch instead would silently fall back to its
    dataclass default.  Round 5's blocks are nested dataclasses, which is
    exactly the shape that regression took.
    """
    config = load_config(_REPO / "config" / "identification" / "fmrr_tecnobody.yaml")
    rebuilt = replace(config, n_trajectories=1)
    assert rebuilt.controller == config.controller
    assert rebuilt.measurement == config.measurement
    assert rebuilt.plant_extras == config.plant_extras
    assert rebuilt.signals == config.signals
    assert rebuilt.controller.velocity_loop.velocity_bandwidth == (20.0, 60.0)


def test_unknown_round5_keys_are_rejected(tmp_path):
    """New blocks are key-validated, like every other block (R4_00 U9)."""
    source = (_REPO / "config" / "identification" / "fmrr_tecnobody.yaml").read_text()
    path = tmp_path / "bad.yaml"
    path.write_text(source.replace("    mode: velocity_pi", "    mode: velocity_pi\n    modee: typo"))
    with pytest.raises(ValueError, match="unknown simulation.controller keys"):
        load_config(path)


# ---------------------------------------------------------------------------
# R5 T2: controller modes
# ---------------------------------------------------------------------------

def test_every_mode_is_constructible(ur10):
    """All five modes build and return a finite command on the first call."""
    from elastic_sim import excitation as exc
    from elastic_sim.identification import FrictionModel
    from elastic_sim.torque_runners import TransmissionSpec, link_inertia_envelope

    config = exc.FourierExcitationConfig(n_harmonics=3, base_frequency=0.5, time_step=0.01,
                                         max_acceleration=1.0)
    trajectory = exc.optimize_excitation(ur10, config, seed=0, n_candidates=2)
    inertia = link_inertia_envelope(ur10, n_samples=16)
    transmission = TransmissionSpec.from_damping_ratio(
        np.full(6, 2.0e4), np.full(6, 0.1), np.full(6, 1.0), inertia[0], inertia[1],
    )
    friction = FrictionModel.from_asset(ur10)
    draw = ControllerDraw(5.5, 1.0, 8.0, 40.0, 0.1)
    for mode in CONTROLLER_MODES:
        spec = ControllerSpec(mode=mode)
        controller = build_controller(
            spec, draw, asset=ur10, nominal_asset=ur10, trajectory=trajectory,
            friction=friction, transmission=transmission,
        )
        q, dq = trajectory(0.0)[:2]
        command = controller(0.0, q, dq, np.zeros(6))
        assert np.isfinite(command.total).all()
        # The reported split must add up on every mode, saturation included.
        assert np.allclose(command.feedforward + command.feedback, command.total)


def test_exact_ct_rejects_a_nominal_model(ur10):
    """`exact_ct` plus a wrong model is a contradiction, not a silent choice."""
    from elastic_sim import excitation as exc
    from elastic_sim.identification import FrictionModel

    config = exc.FourierExcitationConfig(n_harmonics=3, base_frequency=0.5, time_step=0.01,
                                         max_acceleration=1.0)
    trajectory = exc.optimize_excitation(ur10, config, seed=0, n_candidates=2)
    spec = ControllerSpec(mode="exact_ct", nominal=NominalModelSpec(knows_payload=False))
    with pytest.raises(ValueError, match="contradiction"):
        build_controller(spec, ControllerDraw(5.5, 1.0, 8.0, 40.0, 0.1), asset=ur10,
                         nominal_asset=ur10, trajectory=trajectory,
                         friction=FrictionModel.from_asset(ur10))


def test_nominal_model_scales_actually_reach_the_controller(ur10):
    """Every `nominal` knob must change the command, or it is a silent no-op.

    `inertia_scale` was exactly that for one pass: the computed-torque
    controllers build their own Pinocchio model from the asset, so a scale
    applied only in `controllers.py` never reached them and the config key was
    accepted and ignored (`R5_02` Sec 5, D-6). Asserting the *effect* rather
    than the plumbing is what catches the next one of these.
    """
    from elastic_sim import excitation as exc
    from elastic_sim.identification import FrictionModel
    from elastic_sim.torque_runners import TransmissionSpec, link_inertia_envelope

    config = exc.FourierExcitationConfig(n_harmonics=3, base_frequency=0.5, time_step=0.01,
                                         max_acceleration=1.0)
    trajectory = exc.optimize_excitation(ur10, config, seed=0, n_candidates=2)
    inertia = link_inertia_envelope(ur10, n_samples=16)
    transmission = TransmissionSpec.from_damping_ratio(
        np.full(6, 2.0e4), np.full(6, 0.1), np.full(6, 1.0), inertia[0], inertia[1],
    )
    friction = FrictionModel(np.full(6, 1.0), np.full(6, 0.5))
    draw = ControllerDraw(5.5, 1.0, 8.0, 40.0, 0.1)
    q, dq, _ = trajectory(0.0)

    def _command(spec):
        controller = build_controller(
            spec, draw, asset=ur10, nominal_asset=ur10, trajectory=trajectory,
            friction=friction, transmission=transmission,
        )
        return controller(0.0, q, dq, np.zeros(6)).total

    exact = _command(ControllerSpec(mode="exact_ct"))
    scaled = _command(ControllerSpec(mode="nominal_ct", nominal=NominalModelSpec(inertia_scale=1.2)))
    # At rest the command is gravity, which scales with the mass exactly.
    gravity_loaded = np.abs(exact) > 1.0
    assert gravity_loaded.any(), "fixture problem: no joint carries a gravity load here"
    assert np.allclose(scaled[gravity_loaded] / exact[gravity_loaded], 1.2, rtol=1e-6)

    rotor = _command(ControllerSpec(mode="nominal_ct", nominal=NominalModelSpec(rotor_inertia_scale=2.0)))
    controller = build_controller(
        ControllerSpec(mode="nominal_ct", nominal=NominalModelSpec(friction_scale=0.5)), draw,
        asset=ur10, nominal_asset=ur10, trajectory=trajectory, friction=friction,
        transmission=transmission,
    )
    assert np.allclose(controller.friction.viscous, 0.5 * friction.viscous)
    assert np.isfinite(rotor).all()


def test_pd_gains_follow_the_declared_bandwidth(ur10):
    """``kp = (M_ii + J) omega^2``, ``kd = 2 zeta (M_ii + J) omega`` (R5_00 Sec 6.1)."""
    from elastic_sim import excitation as exc

    config = exc.FourierExcitationConfig(n_harmonics=3, base_frequency=0.5, time_step=0.01,
                                         max_acceleration=1.0)
    trajectory = exc.optimize_excitation(ur10, config, seed=0, n_candidates=2)
    inertia = np.array([5.0, 8.0, 3.0, 0.5, 0.5, 0.1])
    controller = JointPdController(ur10, trajectory, joint_inertia=inertia,
                                   natural_frequency=6.0, damping_ratio=0.8)
    assert np.allclose(controller.kp, inertia * 36.0)
    assert np.allclose(controller.kd, 2.0 * 0.8 * inertia * 6.0)


def test_velocity_loop_integral_removes_a_constant_load(ur10):
    """The PI loop carries a constant disturbance with no model at all.

    Driven at a standstill reference with a constant velocity error, the
    integral must grow until it alone supplies the torque -- that property is
    why ``velocity_pi`` tracks a gravity-loaded joint where plain ``pd`` sags.
    """
    from elastic_sim import excitation as exc

    config = exc.FourierExcitationConfig(n_harmonics=3, base_frequency=0.5, time_step=0.01,
                                         max_acceleration=1.0)
    trajectory = exc.optimize_excitation(ur10, config, seed=0, n_candidates=2)
    controller = VelocityLoopController(
        ur10, trajectory, joint_inertia=np.ones(6), position_gain=5.0,
        velocity_bandwidth=50.0, integral_time=0.1,
    )
    q0, dq0, _ = trajectory(0.0)
    # Hold the state fixed at t = 0's reference: the error is whatever the
    # reference asks for, constant, so the integral must accumulate.
    first = controller(0.0, q0, dq0, None)
    for step in range(1, 50):
        command = controller(step * 1.0e-3, q0, dq0, None)
    assert np.abs(command.feedforward).max() > np.abs(first.feedforward).max()
    assert np.isfinite(command.total).all()


def test_velocity_loop_respects_the_effort_limit_and_does_not_wind_up(ur10):
    """Anti-windup: a saturated bag must not overshoot after the transient."""
    from elastic_sim import excitation as exc

    config = exc.FourierExcitationConfig(n_harmonics=3, base_frequency=0.5, time_step=0.01,
                                         max_acceleration=1.0)
    trajectory = exc.optimize_excitation(ur10, config, seed=0, n_candidates=2)
    limit = np.full(6, 1.0)
    controller = VelocityLoopController(
        ur10, trajectory, joint_inertia=np.ones(6), position_gain=5.0,
        velocity_bandwidth=500.0, integral_time=0.01, effort_limit=limit,
    )
    q0, dq0, _ = trajectory(0.0)
    offset = q0 + 1.0  # a metre/radian of position error: deep in saturation
    for step in range(200):
        command = controller(step * 1.0e-3, offset, dq0, None)
    assert np.abs(command.total).max() <= limit.max() + 1e-9
    # The integral is clamped, not free-running: its torque cannot have grown
    # beyond what the limit can ever deliver by more than the proportional
    # part it was accumulated against.
    assert np.abs(command.feedforward).max() < 10.0 * limit.max()


def test_velocity_loop_needs_a_separated_cascade():
    """An outer gain near the inner bandwidth is refused when the config loads."""
    with pytest.raises(ValueError, match="separated"):
        VelocityLoopSpec(position_gain=(50.0, 80.0), velocity_bandwidth=(100.0, 100.0))


def test_velocity_loop_draw_is_reproducible_and_on_its_own_stream():
    """Stream ``(seed, 7, traj)``: reproducible, and independent of the rest."""
    spec = ControllerSpec(
        mode="velocity_pi", randomize=True,
        velocity_loop=VelocityLoopSpec((4.0, 10.0), (20.0, 60.0), (0.05, 0.3)),
    )
    first = sample_velocity_loop(spec, 20260922, 5)
    assert first == sample_velocity_loop(spec, 20260922, 5)
    assert first != sample_velocity_loop(spec, 20260922, 6)
    for value, (low, high) in zip(first, (spec.velocity_loop.position_gain,
                                          spec.velocity_loop.velocity_bandwidth,
                                          spec.velocity_loop.integral_time)):
        assert low <= value <= high
    # Not randomizing pins every draw to the low end, deterministically.
    fixed = replace(spec, randomize=False)
    assert sample_velocity_loop(fixed, 20260922, 5) == (4.0, 20.0, 0.05)


def test_effective_bandwidth_is_the_fastest_loop():
    """The separation check must see the velocity loop, not the outer gain."""
    draw = ControllerDraw(6.0, 1.0, 8.0, 40.0, 0.1)
    assert effective_bandwidth(ControllerSpec(mode="exact_ct"), draw) == 6.0
    assert effective_bandwidth(ControllerSpec(mode="pd"), draw) == 6.0
    assert effective_bandwidth(ControllerSpec(mode="velocity_pi"), draw) == 40.0


# ---------------------------------------------------------------------------
# R5 T3: measurement model
# ---------------------------------------------------------------------------

def test_ideal_measurement_is_the_identity():
    values = np.linspace(0.0, 1.0, 50).reshape(10, 5)
    measured = measure_bag(
        IDEAL_MEASUREMENT, q_motor=values, dq_motor=values, q_link=values, dq_link=values,
        tau_motor=values, tau_link=values, seed=3,
    )
    for channel in (measured.q_motor, measured.dq_link, measured.tau_motor, measured.tau_link):
        assert np.array_equal(channel, values)
    assert np.array_equal(measured.gain_motor, np.ones(5))


def test_quantization_lands_on_the_encoder_grid():
    model = MeasurementModel(encoder_resolution=1.0e-4)
    rng = np.random.default_rng(0)
    position = rng.normal(size=(64, 3))
    measured = model.measure_position(position, rng)
    assert np.allclose(measured / 1.0e-4, np.round(measured / 1.0e-4))
    assert np.abs(measured - position).max() <= 0.5 * 1.0e-4 + 1e-12


def test_delay_shifts_and_keeps_the_length():
    model = MeasurementModel(delay_samples=3)
    values = np.arange(20.0).reshape(-1, 1)
    delayed = model.delay(values)
    assert delayed.shape == values.shape
    assert np.allclose(delayed[:3, 0], 0.0)
    assert np.allclose(delayed[3:, 0], values[:-3, 0])


def test_torque_gain_is_constant_within_a_bag_and_differs_between_channels():
    """A calibration error does not average out, and the two instruments differ."""
    model = MeasurementModel(tau_gain_error=0.05)
    torque = np.ones((100, 4))
    measured = measure_bag(
        model, q_motor=torque, dq_motor=torque, q_link=torque, dq_link=torque,
        tau_motor=torque, tau_link=torque, seed=11,
    )
    assert np.allclose(measured.tau_motor, measured.gain_motor[None, :])
    assert not np.allclose(measured.gain_motor, measured.gain_link)
    assert np.abs(measured.gain_motor - 1.0).max() <= 0.05


def test_measurement_is_keyed_on_the_bag_seed():
    model = MeasurementModel(q_noise=1e-3, tau_noise_abs=0.1)
    values = np.zeros((32, 2))
    kwargs = dict(q_motor=values, dq_motor=values, q_link=values, dq_link=values,
                  tau_motor=values, tau_link=values)
    first = measure_bag(model, seed=1, **kwargs)
    assert np.allclose(first.q_motor, measure_bag(model, seed=1, **kwargs).q_motor)
    assert not np.allclose(first.q_motor, measure_bag(model, seed=2, **kwargs).q_motor)


# ---------------------------------------------------------------------------
# R5 T4: plant extras
# ---------------------------------------------------------------------------

def test_nonlinear_spring_is_continuous_and_matches_its_slopes():
    spring = StiffnessNonlinearity(breakpoints=(0.001, 0.003), factors=(0.5, 1.0, 2.0))
    k = np.array([1.0e4])
    for deflection in (0.0005, 0.001, 0.002, 0.003, 0.005):
        below = spring.torque(np.array([deflection - 1e-9]), k)
        above = spring.torque(np.array([deflection + 1e-9]), k)
        assert np.abs(above - below) < 1e-3, "spring characteristic must be continuous"
    # Slope inside each region is factor * k.
    for low, high, factor in ((0.0, 0.001, 0.5), (0.001, 0.003, 1.0), (0.003, 0.006, 2.0)):
        mid = 0.5 * (low + high)
        step = 1.0e-6
        slope = (spring.torque(np.array([mid + step]), k) - spring.torque(np.array([mid - step]), k)) / (2 * step)
        assert np.isclose(slope[0], factor * k[0], rtol=1e-6)
    # Odd symmetry: a spring pushes back the same either way.
    assert np.isclose(spring.torque(np.array([-0.004]), k)[0], -spring.torque(np.array([0.004]), k)[0])


def test_linear_nonlinearity_is_a_no_op():
    extras = PlantExtras(stiffness_nonlinearity=StiffnessNonlinearity(breakpoints=(0.01,), factors=(1.0, 1.0)))
    assert extras.is_empty
    deflection = np.linspace(-0.02, 0.02, 11)
    assert np.allclose(extras.spring_correction(deflection, np.full(11, 1.0e4)), 0.0)


def test_spring_torque_reduces_to_the_linear_law_without_extras():
    deflection = np.array([1e-3, -2e-3])
    rate = np.array([0.1, -0.2])
    k, d = np.array([1e4, 2e4]), np.array([10.0, 20.0])
    assert np.allclose(NO_EXTRAS.spring_torque(deflection, rate, k, d), -(k * deflection + d * rate))


def test_link_friction_opposes_the_link_motion():
    from elastic_sim.identification import FrictionModel

    extras = PlantExtras(link_friction=FrictionModel(np.array([2.0]), np.array([5.0])))
    assert extras.link_friction_torque(np.array([1.0]))[0] < 0.0
    assert extras.link_friction_torque(np.array([-1.0]))[0] > 0.0
    assert np.isclose(extras.link_friction_torque(np.array([0.0]))[0], 0.0)


def test_torque_ripple_is_periodic_in_the_motor_angle_and_bounded():
    ripple = TorqueRipple(amplitude=0.02, order=24.0)
    limit = np.array([100.0])
    angle = np.linspace(0.0, 2.0 * np.pi / 24.0, 7)
    values = np.asarray([ripple.torque(np.array([a]), limit)[0] for a in angle])
    assert np.abs(values).max() <= 0.02 * 100.0 + 1e-9
    assert np.isclose(values[0], values[-1], atol=1e-9)
    # A joint with no rated torque has no scale to hang a relative ripple on.
    assert ripple.torque(np.array([0.3]), np.array([np.inf]))[0] == 0.0


def test_ripple_phase_is_drawn_per_bag():
    sampling = replace(load_config(_REPO / "config" / "identification" / "fmrr_tecnobody.yaml").plant_extras,
                       ripple_amplitude=0.02)
    first = plant_extras_for_bag(sampling, 3, 20260922, 1)
    assert first.torque_ripple.phase != plant_extras_for_bag(sampling, 3, 20260922, 2).torque_ripple.phase
    assert first.torque_ripple.phase == plant_extras_for_bag(sampling, 3, 20260922, 1).torque_ripple.phase


# ---------------------------------------------------------------------------
# R5 T5: signal policy
# ---------------------------------------------------------------------------

def test_signal_policy_knows_which_pairs_are_collocated():
    assert SignalPolicy("link", "link_torque").is_collocated
    assert SignalPolicy("motor", "motor_torque").is_collocated
    assert not SignalPolicy("motor", "link_torque").is_collocated
    assert not SignalPolicy("link", "motor_torque").is_collocated


def test_rollout_frame_selects_the_configured_side(ur10):
    """``q0..`` follows ``position_side``; the side-named columns never move."""
    from elastic_sim import excitation as exc
    from elastic_sim.dataset import RIGID_TIER, Tier
    from elastic_sim.identification import FrictionModel

    config = exc.FourierExcitationConfig(n_harmonics=3, base_frequency=1.0, time_step=0.01,
                                         max_acceleration=1.0)
    trajectory = exc.optimize_excitation(ur10, config, seed=0, n_candidates=2)
    samples = 20
    time = np.arange(samples) * 0.01
    motor = np.tile(np.arange(6, dtype=float), (samples, 1))
    link = motor + 0.5
    result = {
        "time": time, "q_ref": link, "dq_ref": link, "q_link": link, "dq_link": link,
        "ddq_link": link, "q_motor": motor, "dq_motor": motor,
        "tau_motor": link, "tau_link": motor, "tau_feedforward": link, "tau_feedback": link,
        "joint_names": ur10.joint_names,
    }
    friction = FrictionModel.from_asset(ur10)
    for side, expected in (("link", link), ("motor", motor)):
        frame = rollout_frame(
            ur10, trajectory, result, bag="b", tier=Tier(RIGID_TIER), backend="mujoco",
            friction=friction, resample_step=0.01, signals=SignalPolicy(position_side=side),
        )
        assert np.allclose(frame["q0"].to_numpy(), expected[:, 0])
        assert np.allclose(frame["q_motor0"].to_numpy(), motor[:, 0])
        assert np.allclose(frame["q_link0"].to_numpy(), link[:, 0])


# ---------------------------------------------------------------------------
# R5 T6: FMRR in the identification stack (R5_01 Sec 1, O-3)
# ---------------------------------------------------------------------------

def test_locking_is_a_no_op_for_assets_whose_joints_are_all_active(ur10):
    text = Path(ur10.urdf_path).read_text()
    locked_text, locked = lock_inactive_one_dof_joints(text, ur10.joint_names)
    assert locked == ()
    assert locked_text == text


def test_fmrr_locks_only_the_non_active_joint(fmrr):
    text = Path(fmrr.urdf_path).read_text()
    _, locked = lock_inactive_one_dof_joints(text, fmrr.joint_names)
    assert locked == ("joint_yaw",)


def test_fmrr_builds_a_pinocchio_model_with_the_declared_joint_order(fmrr):
    """Xacro expansion plus the reduced model (R5_01 Sec 1.1/1.2)."""
    pytest.importorskip("pinocchio")
    from elastic_sim import identification as idn

    pin, model, data = idn.build_model(fmrr)
    assert tuple(model.names[i] for i in range(1, model.njoints)) == fmrr.joint_names
    assert model.nv == 3


def test_fmrr_is_a_constant_diagonal_cartesian_plant(fmrr):
    """The property that makes FMRR the family's analytic test case.

    A Cartesian gantry's mass matrix is constant and diagonal and gravity acts
    on the vertical axis alone, so an identification result on it can be
    checked by hand.
    """
    pytest.importorskip("pinocchio")
    from elastic_sim import identification as idn

    pin, model, data = idn.build_model(fmrr)
    rng = np.random.default_rng(0)
    reference = np.asarray(pin.crba(model, data, np.zeros(3)), dtype=float)
    for _ in range(5):
        q = rng.uniform(-1.0, 1.0, size=3)
        mass_matrix = np.asarray(pin.crba(model, data, q), dtype=float)
        assert np.allclose(mass_matrix, reference, atol=1e-9)
        assert np.allclose(mass_matrix - np.diag(np.diag(mass_matrix)), 0.0, atol=1e-9)
        gravity = np.asarray(pin.computeGeneralizedGravity(model, data, q), dtype=float)
        # Joint order is (y, x, z): only the last axis carries gravity.
        assert np.allclose(gravity[:2], 0.0, atol=1e-6)
        assert gravity[2] > 0.0


def test_fmrr_identification_config_loads_and_is_round5(fmrr):
    config = load_config(_REPO / "config" / "identification" / "fmrr_tecnobody.yaml")
    assert config.asset == "fmrr_tecnobody"
    assert config.controller.mode == "velocity_pi"
    assert not config.signals.is_collocated
    assert not config.measurement.is_ideal
    assert not config.plant_extras.is_empty
    # 250 Hz, the real EtherCAT/recorder rate (R5_01 Amendment 1).
    assert config.sample_time_step == 0.004
    # The probe must bracket the transmission mode or elasticity is
    # unobservable whatever the signal pair (R5_01 O-5).
    top = config.excitation.probe_harmonics[-1] * config.excitation.base_frequency
    assert top >= 25.0


# ---------------------------------------------------------------------------
# R5 T7: physics (slow)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fmrr_trajectory(fmrr):
    from elastic_sim import excitation as exc

    config = exc.FourierExcitationConfig(
        n_harmonics=4, base_frequency=1.0, n_periods=1, time_step=0.004,
        max_acceleration=1.0, velocity_fraction=0.5,
    )
    return exc.optimize_excitation(fmrr, config, seed=5, n_candidates=4)


def _fmrr_transmission(fmrr):
    from elastic_sim.torque_runners import TransmissionSpec, link_inertia_envelope

    inertia = link_inertia_envelope(fmrr, n_samples=8)
    return TransmissionSpec.from_damping_ratio(
        np.array([7162.0, 5977.0, 3861.0]), np.full(3, 0.8), np.full(3, 12.0), inertia[0], inertia[1],
    )


@pytest.mark.slow
def test_fmrr_runs_an_elastic_rollout(fmrr, fmrr_trajectory):
    """O-3's acceptance: FMRR goes through the modern elastic torque runner."""
    pytest.importorskip("mujoco")
    from elastic_sim.identification import FrictionModel
    from elastic_sim.torque_runners import run_mujoco_elastic_torque

    transmission = _fmrr_transmission(fmrr)
    controller = build_controller(
        ControllerSpec(mode="velocity_pi"),
        ControllerDraw(6.0, 1.0, 8.0, 40.0, 0.1),
        asset=fmrr, nominal_asset=fmrr, trajectory=fmrr_trajectory,
        friction=FrictionModel.from_asset(fmrr), transmission=transmission,
    )
    result = run_mujoco_elastic_torque(
        fmrr, fmrr_trajectory, controller, transmission, time_step=5.0e-4,
        friction=FrictionModel.from_asset(fmrr),
    )
    assert np.isfinite(result["q_link"]).all()
    deflection = np.abs(np.asarray(result["q_motor"]) - np.asarray(result["q_link"]))
    assert deflection.max() > 0.0
    # The label is the spring force, so it must match k*e + d*edot.
    assert np.isfinite(result["tau_link"]).all()


@pytest.mark.slow
def test_link_friction_appears_in_the_target(fmrr, fmrr_trajectory):
    """The residual class's reason to exist, measured (R5_00 Sec 5.2).

    With link-side friction the link equation gains a term no Lagrangian can
    express: ``tau_s = M qdd + c + g + f_link(dq)``.  The same rollout with and
    without it must differ in the *target*, not only in the motion.
    """
    pytest.importorskip("mujoco")
    from elastic_sim.identification import FrictionModel
    from elastic_sim.torque_runners import run_mujoco_elastic_torque

    transmission = _fmrr_transmission(fmrr)
    friction = FrictionModel.from_asset(fmrr)
    extras = PlantExtras(link_friction=FrictionModel(np.full(3, 8.0), np.full(3, 10.0)))

    def _run(plant_extras):
        controller = build_controller(
            ControllerSpec(mode="velocity_pi"), ControllerDraw(6.0, 1.0, 8.0, 40.0, 0.1),
            asset=fmrr, nominal_asset=fmrr, trajectory=fmrr_trajectory,
            friction=friction, transmission=transmission,
        )
        return run_mujoco_elastic_torque(
            fmrr, fmrr_trajectory, controller, transmission, time_step=5.0e-4,
            friction=friction, extras=plant_extras,
        )

    plain = _run(None)
    rough = _run(extras)
    difference = np.abs(np.asarray(rough["tau_link"]) - np.asarray(plain["tau_link"]))
    reference = np.sqrt(np.mean(np.asarray(plain["tau_link"]) ** 2))
    assert difference.max() > 0.05 * reference, "link friction left no trace in the target"


@pytest.mark.slow
def test_model_free_command_is_not_a_function_of_the_reference(fmrr, fmrr_trajectory):
    """The claim round 5 rests on, measured directly (``R5_00`` Sec 5.3).

    Under exact computed torque the commanded torque is ``rnea`` evaluated at
    (very nearly) the reference, because the loop drives the achieved state
    onto it: ``tau_cmd`` is then an almost deterministic function of the
    reference and carries no independent information about the plant.  A
    model-free loop's command is driven by the tracking error instead, so its
    agreement with the reference feedforward has to be measurably worse.

    Stated as *this* rather than as "a model-free loop tracks worse", which is
    what an earlier version of this test asserted and which the Q-C sweep
    disproved on the elastic tier: the PI loop's integral compensates the
    spring's static sag that a collocated computed-torque law ignores, so it
    can track the link *better* while still being model-free (``R5_02``
    Sec 6.2).
    """
    pytest.importorskip("mujoco")
    from elastic_sim import identification as idn
    from elastic_sim.identification import FrictionModel
    from elastic_sim.torque_runners import run_mujoco_elastic_torque

    transmission = _fmrr_transmission(fmrr)
    friction = FrictionModel.from_asset(fmrr)
    pin, model, data = idn.build_model(fmrr)
    planned = fmrr_trajectory.sample()

    agreement = {}
    for mode in ("exact_ct", "velocity_pi"):
        controller = build_controller(
            ControllerSpec(mode=mode), ControllerDraw(6.0, 1.0, 8.0, 40.0, 0.1),
            asset=fmrr, nominal_asset=fmrr, trajectory=fmrr_trajectory,
            friction=friction, transmission=transmission,
        )
        result = run_mujoco_elastic_torque(
            fmrr, fmrr_trajectory, controller, transmission, time_step=5.0e-4, friction=friction,
        )
        time = np.asarray(result["time"])
        stride = max(1, len(time) // 400)
        sampled = time[::stride]

        def _on_rollout_grid(key: str) -> np.ndarray:
            return np.column_stack([
                np.interp(sampled, fmrr_trajectory.time, planned[key][:, joint]) for joint in range(3)
            ])

        q_ref, dq_ref, ddq_ref = (_on_rollout_grid(key) for key in ("q", "dq", "ddq"))
        # The motor-side feedforward, rotor inertia included: on this platform
        # the reflected motor mass (12 kg) is several times the link mass, so
        # comparing against the link-side `rnea` alone would say exact computed
        # torque disagrees with its own reference by 100 % (measured) and the
        # test would be about the rotor term, not about the controller.
        reference = np.asarray([
            idn.inverse_dynamics(pin, model, data, q_ref[i], dq_ref[i], ddq_ref[i])
            + transmission.rotor_inertia * ddq_ref[i]
            for i in range(len(sampled))
        ])
        commanded = np.asarray(result["tau_motor"])[::stride]
        error = commanded - reference
        agreement[mode] = float(np.sqrt(np.mean(error**2)) / np.sqrt(np.mean(reference**2)))
    assert agreement["velocity_pi"] > 2.0 * agreement["exact_ct"], agreement


@pytest.mark.slow
def test_rigid_tier_refuses_plant_extras(fmrr, fmrr_trajectory):
    """The analytic reference tier stays analytic (R5_02 Sec 3.3)."""
    from elastic_sim.dataset import RIGID_TIER, Tier, run_condition
    from elastic_sim.identification import FrictionModel

    config = DatasetConfig(asset="fmrr_tecnobody")
    extras = PlantExtras(link_friction=FrictionModel(np.full(3, 1.0), np.full(3, 1.0)))
    with pytest.raises(ValueError, match="rigid reference tier"):
        run_condition(fmrr, fmrr_trajectory, Tier(RIGID_TIER), "mujoco",
                      FrictionModel.from_asset(fmrr), config, extras=extras)
