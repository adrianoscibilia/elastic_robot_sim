"""Round 6 (`R6_00` Sec 11): the model-free PD loop, ground-truth tau_s, noise, bounds.

Cheap contract and algebra checks unmarked; anything that integrates physics
marked ``slow``.  The simulation tests run the real round-6 configs on a short
excitation (2 s, a three-line probe) so a bag costs seconds, not minutes.
"""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.path.join(_REPO, "src"))
sys.path.insert(0, os.path.join(_REPO, "scripts"))

from elastic_sim.assets import AssetRegistry
from elastic_sim.controllers import ControllerDraw, JointPdController, build_controller
from elastic_sim.dataset import load_config, plant_extras_for_bag, resolve_bag, rollout_frame, run_condition
from elastic_sim.dataset_bag import noise_seed
from elastic_sim.dataset_bounds import dof_totals, explicit_term_table, require_explicit_term_bound
from elastic_sim.dataset_config import differentiation_policy, gain_bound, physics_step
from elastic_sim.dataset_worklist import _log_uniform_union, motor_friction_for, sample_control_gains
from elastic_sim.identification import FrictionModel
from elastic_sim.measurement import ChannelNoise, DriveInstrument, LoopInstrument, NoiseModel
from elastic_sim.plant_extras import PlantExtras, StiffnessNonlinearity, TransmissionError

CONFIGS = {
    "iiwa": "config/identification/kuka_lbr_iiwa_14_r820_table_round6_bus.yaml",
    "iiwa_drive": "config/identification/kuka_lbr_iiwa_14_r820_table_round6_drive.yaml",
    "ur10": "config/identification/ur10_table_round6.yaml",
    "fmrr": "config/identification/fmrr_tecnobody_round6.yaml",
}


@pytest.fixture(scope="module")
def registry():
    return AssetRegistry.for_repository(_REPO)


@pytest.fixture(scope="module")
def configs():
    return {name: load_config(_REPO / path, check_bounds=False) for name, path in CONFIGS.items()}


def _yaml(name: str) -> dict:
    return yaml.safe_load((_REPO / CONFIGS[name]).read_text(encoding="utf-8"))


def _write(tmp_path: Path, raw: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Sec 2.2: gain sizing and its bound
# ---------------------------------------------------------------------------

def test_round6_configs_load_with_every_bound(configs):
    for name in CONFIGS:
        config = load_config(_REPO / CONFIGS[name])  # includes the Sec 3 table
        assert config.is_round6 and config.controller.mode == "pd"
        assert gain_bound(config)["ok"]
        assert config.control_gains.bands[0][0] == pytest.approx(0.5 * config.omega_max())


def test_omega_max_is_the_spec_number(configs):
    # Bus: 0.5 / (2 * 1.3 * T_c * 2): ~96 / 24 rad/s at 1 kHz / 250 Hz.
    assert configs["iiwa"].omega_max() == pytest.approx(96.15, abs=0.01)
    # UR10 in its drive (R6_02 Sec 7): 0.5 / (2 * 1.3 * 5e-4 * 1) at 2 kHz, d = 0.
    assert configs["ur10"].controller.location == "drive"
    assert configs["ur10"].omega_max() == pytest.approx(384.62, abs=0.01)
    # FMRR and the iiwa drive variant at 4 kHz (R6_04 A-1).
    for name in ("fmrr", "iiwa_drive"):
        assert configs[name].controller.location == "drive"
        assert configs[name].omega_max() == pytest.approx(769.23, abs=0.01)


def test_gain_bound_refuses_omega_over_the_limit(tmp_path):
    raw = _yaml("ur10")
    raw["simulation"]["control_gains"]["natural_frequency_fraction"] = [0.5, 1.2]
    with pytest.raises(ValueError, match=r"2 zeta omega T_c \(1 \+ d\)"):
        load_config(_write(tmp_path, raw), check_bounds=False)


def test_motor_inertia_sizing_gives_the_hand_computed_gains(configs, registry):
    """FMRR joint_z: kp = J_nom omega^2, kd = 2 zeta J_nom omega (Sec 2.2's sanity check)."""
    config = configs["fmrr"]
    asset = registry.load("fmrr_tecnobody")
    trajectory = _short(config, asset, "e00")
    draw = ControllerDraw(20.0, 0.9, 1.0, 1.0, 1.0)
    # The bus code path (FMRR production is `drive` since R6_04 A-1).
    bus = replace(config.controller, location="bus", gain_sizing="motor_inertia")
    controller = build_controller(bus, draw, asset=asset, nominal_asset=asset, trajectory=trajectory,
                                  friction=FrictionModel.from_asset(asset),
                                  motor_inertia=config.nominal_rotor_inertia(3))
    assert isinstance(controller, JointPdController)
    assert controller.kp[2] == pytest.approx(19.236 * 20.0**2)          # 7694 N/m
    assert controller.kd[2] == pytest.approx(2 * 0.9 * 19.236 * 20.0)
    # R6_00's order-of-magnitude check: the bus band [0.5, 0.9] x 24.04 rad/s on joint_z is 0.3-0.9e4 N/m.
    low, high = 0.5 * 24.04, 0.9 * 24.04
    assert 2.5e3 < 19.236 * low**2 < 19.236 * high**2 < 1.0e4
    assert controller.compensates_friction is False


def test_gainshift_bands_draw_outside_training_with_the_same_zeta(configs):
    gains = configs["ur10"].control_gains
    shifted = gains.with_fraction_bands(((0.3, 0.5), (0.9, 1.0)))
    omega_max = configs["ur10"].omega_max()
    for seed in range(200):
        omega, zeta = sample_control_gains(shifted, 0.0, 0.0, 7, seed)
        _, zeta_production = sample_control_gains(gains, 0.0, 0.0, 7, seed)
        fraction = omega / omega_max
        assert (0.3 - 1e-9 <= fraction <= 0.5) or (0.9 <= fraction <= 1.0 + 1e-9)
        assert zeta == zeta_production
    assert _log_uniform_union(0.0, ((1.0, 2.0), (4.0, 8.0))) == pytest.approx(1.0)
    assert _log_uniform_union(1.0, ((1.0, 2.0), (4.0, 8.0))) == pytest.approx(8.0)
    assert _log_uniform_union(0.5, ((1.0, 2.0), (4.0, 8.0))) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Sec 3 and Sec 4: the load-time refusals
# ---------------------------------------------------------------------------

def test_explicit_bound_refuses_the_round5_rigid_iiwa_friction(registry):
    """`R5_11` Sec 3.2's divergence: b + c/eps = 110 on A7's 3.1e-4 kg m^2, h = 5e-4 -> ~180."""
    config = load_config(_REPO / "config/identification/kuka_lbr_iiwa_14_r820_table_round5.yaml")
    asset = registry.load(config.asset)
    rows = [r for r in explicit_term_table(config, asset) if r.tier == "rigid" and r.joint == "iiwa_A7"]
    assert rows[0].term == "motor friction" and rows[0].value > 150.0
    with pytest.raises(ValueError, match=r"(?s)rigid iiwa_A7 \(joint DOF\).*motor friction"):
        require_explicit_term_bound(config, asset=asset)


def test_round6_rigid_tier_armature_brings_the_same_friction_under_the_bound(registry, configs):
    """The same round-5 friction on the round-6 rigid tier (nominal rotor inertia as armature)."""
    config = replace(configs["iiwa"], motor_friction=None, rigid_time_step=5.0e-4)
    asset = registry.load(config.asset)
    rows = [r for r in explicit_term_table(config, asset) if r.tier == "rigid" and r.term == "motor friction"]
    a7 = next(r for r in rows if r.joint == "iiwa_A7")
    assert a7.value == pytest.approx(110.0 * 5.0e-4 / 0.15, rel=0.05)   # ~0.37, R6_00 Sec 2.3


def test_schema3_rejects_each_item_of_the_list(tmp_path):
    cases = {
        "probe_resolvable": ("excitation", "probe_harmonics", [80, 1000]),
        "'pd'": ("simulation.controller", "mode", "velocity_pi"),
        "2 zeta omega": ("simulation.control_gains", "natural_frequency_fraction", [0.9, 1.5]),
    }
    for message, (block, key, value) in cases.items():
        raw = _yaml("iiwa")
        if key == "probe_harmonics":
            raw["excitation"].pop("probe_design")
        if key == "mode":
            raw["simulation"]["controller"].pop("integral")      # the pd-only remedy would refuse first
        target = raw
        for part in block.split("."):
            target = target[part]
        target[key] = value
        with pytest.raises(ValueError, match=message.replace("(", r"\(")):
            load_config(_write(tmp_path, raw), check_bounds=False)
    raw = _yaml("iiwa")
    raw["simulation"]["max_time_step"] = 5.0e-4
    with pytest.raises(ValueError, match=r"(?s)explicit-integration bound violated.*motor friction"):
        load_config(_write(tmp_path, raw))
    raw = _yaml("iiwa")
    raw["simulation"]["control_decimation"] = 4
    with pytest.raises(ValueError, match="derives the control decimation"):
        load_config(_write(tmp_path, raw), check_bounds=False)


def test_implicit_link_friction_is_exempt_but_tabulated(registry, configs):
    config = configs["ur10"]
    rows = explicit_term_table(config, registry.load(config.asset))
    link = [r for r in rows if r.term == "link friction"]
    assert link and all(not r.enforced for r in link)
    assert max(r.value for r in link) > 100.0           # why it is implicit: wrist_3's link
    assert max(dof_totals(rows).values()) <= 1.0


def test_physics_step_divides_the_sample_period():
    assert physics_step(6.1e-5, 1e-3) == pytest.approx(1e-3 / 17)
    assert physics_step(5e-4, 4e-3) == pytest.approx(5e-4)
    assert physics_step(9e-5, 1e-3) == pytest.approx(1e-3 / 12)


# ---------------------------------------------------------------------------
# Sec 5: plant contributions
# ---------------------------------------------------------------------------

def test_torque_knees_reproduce_the_catalog_points_ur10_size_32(configs, registry):
    """HD size 32 [S-4]: K1 = 5.4e4, K2 = 7.8e4, T1 = 29, T2 = 108 Nm."""
    k2 = 7.8e4
    curve = StiffnessNonlinearity.from_torque_knees([[0.69, 1.0, 1.44]], [[29.0, 108.0]], [k2])
    theta1, theta2 = curve.breakpoints[0]
    assert theta1 == pytest.approx(29.0 / (0.69 * k2))
    assert theta2 == pytest.approx(theta1 + (108.0 - 29.0) / k2)
    assert curve.torque(np.array([theta1]), [k2])[0] == pytest.approx(29.0)
    assert curve.torque(np.array([theta2]), [k2])[0] == pytest.approx(108.0)
    assert curve.torque(np.array([-theta2]), [k2])[0] == pytest.approx(-108.0)
    # The same numbers through the config's per-joint table, per robot.
    config = configs["ur10"]
    asset = registry.load(config.asset)
    stiffness = np.array([k2, k2, 3.4e4, 1.1e4, 1.1e4, 1.1e4])
    extras = plant_extras_for_bag(config.plant_extras, 6, config.seed, 1, robot="e00", stiffness=stiffness,
                                  effort=np.array([j.effort for j in asset.resolve_active_joints()]))
    assert extras.stiffness_nonlinearity.breakpoints[0] == pytest.approx((theta1, theta2))
    assert extras.stiffness_nonlinearity.torque(np.array([0, 0, 0, 12.0 / 1.1e4 + 3.9 / 1.1e4 * (1 / 0.74 - 1),
                                                          0, 0.0]), stiffness)[3] == pytest.approx(12.0)


def test_iiwa_knees_are_fractions_of_the_effort_limit(configs, registry):
    config = configs["iiwa"]
    asset = registry.load(config.asset)
    effort = np.array([j.effort for j in asset.resolve_active_joints()])
    stiffness = np.full(7, 1.0e4)
    extras = plant_extras_for_bag(config.plant_extras, 7, config.seed, 3, robot="e00", stiffness=stiffness,
                                  effort=effort)
    curve = extras.stiffness_nonlinearity
    for j in range(7):
        theta1, theta2 = curve.breakpoints[j]
        assert curve.torque(np.eye(7)[j] * theta1, stiffness)[j] == pytest.approx(0.2 * effort[j])
        assert curve.torque(np.eye(7)[j] * theta2, stiffness)[j] == pytest.approx(0.75 * effort[j])


def test_transmission_error_period_amplitude_and_per_bag_phase(configs):
    config = configs["iiwa"]
    first = plant_extras_for_bag(config.plant_extras, 7, config.seed, 11, robot="e00", stiffness=np.ones(7) * 1e4,
                                 effort=np.full(7, 200.0))
    second = plant_extras_for_bag(config.plant_extras, 7, config.seed, 12, robot="e00", stiffness=np.ones(7) * 1e4,
                                  effort=np.full(7, 200.0))
    error = first.transmission_error
    theta = np.linspace(-1.0, 1.0, 20001)[:, None] * np.ones((1, 7))
    values = error.error(theta)
    assert np.max(np.abs(values)) == pytest.approx(2.0e-4, rel=1e-3)
    period = 2.0 * np.pi / 320.0
    assert np.allclose(error.error(theta + period), values, atol=1e-15)
    assert not np.allclose(error.phase, second.transmission_error.phase)
    # Rigid tier: no spring, no transmission error.
    assert plant_extras_for_bag(config.plant_extras, 7, config.seed, 11, robot="rigid").transmission_error is None


def test_spring_forces_are_the_gradient_of_the_spring_energy():
    """With d = 0 the forces on (theta, e) are -dU/d(theta, e), U = int K_nl(e - err(theta))."""
    curve = StiffnessNonlinearity.from_torque_knees([[0.8, 1.0, 1.2]], [[40.0, 150.0]], [1.0e4])
    extras = PlantExtras(stiffness_nonlinearity=curve,
                         transmission_error=TransmissionError(amplitude=(2e-4,), order=(320.0,), phase=(0.3,)))
    k = np.array([1.0e4])

    def energy(theta: float, e: float) -> float:
        grid = np.linspace(0.0, e - extras.transmission_error.error(np.array([theta]))[0], 4001)
        return float(np.trapezoid(curve.torque(grid[:, None], k)[:, 0], grid))

    theta, e, h = 0.37, 0.004, 1e-7
    on_motor, on_elastic = extras.spring_forces(np.array([e]), np.zeros(1), np.array([theta]), np.zeros(1), k,
                                                np.zeros(1))
    # The simulator adds -k e on the elastic coordinate itself.
    assert on_elastic[0] - k[0] * e == pytest.approx(-(energy(theta, e + h) - energy(theta, e - h)) / (2 * h), rel=1e-4)
    assert on_motor[0] == pytest.approx(-(energy(theta + h, e) - energy(theta - h, e)) / (2 * h), rel=1e-3)


def test_spring_forces_without_transmission_error_are_round5s_correction():
    curve = StiffnessNonlinearity(breakpoints=(1e-3, 2e-3), factors=(0.6, 1.0, 1.4))
    extras = PlantExtras(stiffness_nonlinearity=curve)
    e = np.array([0.0005, -0.0031, 0.0017])
    k = np.array([1e4, 2e4, 3e4])
    on_motor, on_elastic = extras.spring_forces(e, np.ones(3), np.ones(3), np.ones(3), k, np.ones(3))
    assert np.array_equal(on_elastic, extras.spring_correction(e, k)) and not on_motor.any()


def test_motor_friction_prior_is_the_rule_and_per_robot(configs, registry):
    config = configs["fmrr"]
    asset = registry.load(config.asset)
    nominal = config.motor_friction.nominal(asset)
    assert nominal.viscous == pytest.approx(0.05 * np.array([261.9, 261.9, 986.1]) / 1.0)
    assert nominal.coulomb == pytest.approx(0.02 * np.array([261.9, 261.9, 986.1]))
    assert nominal.epsilon == 1.0e-2
    rigid = next(t for t in config.tiers if t.is_rigid)
    rigid_friction = motor_friction_for(config, asset, rigid)
    assert np.array_equal(rigid_friction.viscous, nominal.viscous)
    assert np.array_equal(rigid_friction.coulomb, nominal.coulomb)
    e00, e01 = (next(t for t in config.tiers if t.name == name) for name in ("e00", "e01"))
    a, b = motor_friction_for(config, asset, e00), motor_friction_for(config, asset, e01)
    assert not np.allclose(a.viscous, b.viscous)
    ratio = np.concatenate([a.viscous / nominal.viscous, a.coulomb / nominal.coulomb])
    assert np.all((ratio >= 0.5) & (ratio <= 2.0))
    assert np.array_equal(motor_friction_for(config, asset, e00).viscous, a.viscous)


# ---------------------------------------------------------------------------
# Sec 6: the instrument
# ---------------------------------------------------------------------------

def test_channel_noise_definition_matches_the_block():
    rng = np.random.default_rng(0)
    clean = np.column_stack([np.sin(np.linspace(0, 40, 20000)) * 10.0, np.cos(np.linspace(0, 30, 20000)) * 2.0])
    noise = NoiseModel(ft=ChannelNoise(p=0.02, gain=0.03, offset=0.005), delay_samples=1)
    measured = noise.measure("ft", clean, seed=5, effort=np.array([100.0, 50.0]))
    rms = np.sqrt(np.mean((clean - clean.mean(axis=0)) ** 2, axis=0))
    assert measured.sigma == pytest.approx(0.02 * rms)
    assert np.all(np.abs(measured.gain - 1.0) <= 0.03)
    assert np.all(np.abs(measured.offset) <= 0.005 * np.array([100.0, 50.0]))
    residual = measured.values[1:] - (measured.gain * clean[:-1] + measured.offset)
    assert np.std(residual, axis=0) == pytest.approx(measured.gain * measured.sigma, rel=0.03)
    assert np.array_equal(measured.values[0], measured.values[1])            # delay holds row 0
    off = NoiseModel(ft=ChannelNoise(p=0.02), enabled=False).measure("ft", clean, seed=5, effort=np.ones(2))
    assert np.array_equal(off.values, clean)
    del rng


def test_loop_instrument_delays_and_differentiates_causally():
    from scipy.signal import savgol_coeffs

    noise = NoiseModel(q=ChannelNoise(quantization=(1e-6,)), dq=ChannelNoise(), dq_source="position_derivative",
                       delay_samples=1)
    instrument = LoopInstrument(noise, n_dof=1, seed=0, sigma_q=np.zeros(1), sigma_dq=np.zeros(1),
                                sample_time=1e-3, sg_window=5, sg_poly=3)
    t = np.arange(50) * 1e-3
    q = np.sin(3.0 * t)[:, None]
    outputs = [instrument.sample(q[i], np.zeros(1)) for i in range(len(t))]
    recorded = np.asarray([o[0] for o in outputs])
    assert np.allclose(recorded[1:], np.round(q[:-1] / 1e-6) * 1e-6)
    coefficients = savgol_coeffs(5, 3, deriv=1, delta=1e-3, pos=4, use="dot")
    assert outputs[20][1][0] == pytest.approx(coefficients @ recorded[16:21, 0])
    assert outputs[20][1][0] == pytest.approx(3.0 * np.cos(3.0 * t[19]), rel=1e-3)


# ---------------------------------------------------------------------------
# Simulation (slow)
# ---------------------------------------------------------------------------

def _short_config(config):
    """The same config on a 2 s excitation with a three-line probe (4, 8, 16 Hz)."""
    excitation = replace(config.excitation, base_frequency=0.5, probe_harmonics=(8, 16, 32))
    return replace(config, excitation=excitation, candidates=4)


def _short(config, asset, tier_name: str):
    config = _short_config(config)
    tier = next(t for t in config.tiers if t.name == tier_name)
    return resolve_bag(config, asset, tier, 0, None).trajectory


def _run(config, asset, tier_name: str):
    config = _short_config(config)
    tier = next(t for t in config.tiers if t.name == tier_name)
    resolved = resolve_bag(config, asset, tier, 0, None)
    friction = motor_friction_for(config, asset, tier)
    bag = f"t0_{tier.name}_f0_mujoco"
    seed = noise_seed(config, bag)
    result = run_condition(asset, resolved.trajectory, tier, "mujoco", friction, config, payload=None,
                           draw=resolved.draw, extras=resolved.extras, seed=seed)
    frame = rollout_frame(asset, resolved.trajectory, result, bag=bag, tier=tier, backend="mujoco",
                          friction=friction, resample_step=config.sample_time_step, signals=config.signals,
                          measurement_seed=seed, noise=config.noise, differentiation=differentiation_policy(config))
    return config, result, frame


@pytest.mark.slow
@pytest.mark.parametrize("platform", ["iiwa", "ur10", "fmrr"])
def test_ft_clean_is_the_spring_torque_at_the_physics_step(platform, configs, registry):
    config = configs[platform]
    asset = registry.load(config.asset)
    config, result, frame = _run(config, asset, "e00")
    n = len(asset.joint_names)
    decimation = result["control_decimation"]
    assert decimation * result["time_step"] == pytest.approx(config.control_period)
    spring = np.asarray(result["tau_link"])[::decimation * config.drive_ticks]
    ft_clean = frame[[f"ft_clean{i}" for i in range(n)]].to_numpy()
    # Equal up to the round-off of resampling at k T_c vs k N h, which differ in the last ulp.
    assert np.allclose(ft_clean, spring, rtol=1e-12, atol=1e-12 * float(np.max(np.abs(spring))))
    assert np.array_equal(ft_clean, frame[[f"tau_link_clean{i}" for i in range(n)]].to_numpy())


@pytest.mark.slow
@pytest.mark.parametrize("platform", ["iiwa", "fmrr"])
def test_rigid_tier_target_satisfies_the_motor_identity(platform, configs, registry):
    """Sec 2.3: tau_s = rnea_link + f_link = tau_applied - J ddq - f_motor, to solver tolerance."""
    config = configs[platform]
    asset = registry.load(config.asset)
    _, result, _ = _run(config, asset, "rigid")
    identity = (np.asarray(result["tau_applied"]) - np.asarray(result["armature"]) * np.asarray(result["ddq_link"])
                - np.asarray(result["friction_motor"]))
    tau_s = np.asarray(result["tau_link"])
    assert np.sqrt(np.mean((tau_s - identity) ** 2)) < 1e-6 * max(1.0, float(np.sqrt(np.mean(tau_s**2))))


@pytest.mark.slow
def test_controller_is_held_per_sample_on_the_delayed_measured_state(configs, registry):
    fmrr = configs["fmrr"]                                         # its bus code path: a bus loop with a drive velocity
    config = replace(fmrr, controller=replace(fmrr.controller, location="bus", gain_sizing="motor_inertia"))
    config = replace(config, control_gains=config.control_gains.resolved(config.omega_max()))
    asset = registry.load(config.asset)
    config, result, frame = _run(config, asset, "e00")
    n = len(asset.joint_names)
    decimation = result["control_decimation"]
    tau = np.asarray(result["tau_motor"])
    blocks = tau[: (len(tau) - 1) // decimation * decimation].reshape(-1, decimation, n)
    assert np.all(blocks == blocks[:, :1, :])                     # zero-order hold
    # The command at sample k is the PD law on the *recorded* (measured, delayed) q, dq.
    q = frame[[f"q{i}" for i in range(n)]].to_numpy(dtype=float)
    dq = frame[[f"dq{i}" for i in range(n)]].to_numpy(dtype=float)
    q_true = frame[[f"q_motor_clean{i}" for i in range(n)]].to_numpy(dtype=float)
    kp, kd = np.asarray(result["kp"]), np.asarray(result["kd"])
    q_ref = np.asarray(result["q_ref"])[::decimation]
    dq_ref = np.asarray(result["dq_ref"])[::decimation]
    expected = kp * (q_ref - np.asarray(result["q_motor_measured"])) + kd * (dq_ref - np.asarray(result["dq_motor_measured"]))
    assert np.allclose(tau[::decimation], expected, rtol=1e-6, atol=1e-6 * float(np.max(np.abs(tau))))
    assert np.allclose(np.asarray(result["q_motor_measured"]), q.astype(float), atol=1e-6)
    assert np.max(np.abs(q[1:] - q_true[:-1])) < np.max(np.abs(q[1:] - q_true[1:]))  # one sample old
    assert dq.shape == q.shape


@pytest.mark.slow
def test_rigid_iiwa_pd_at_round5_gains_is_stable_once_the_tier_has_its_rotor(configs, registry):
    """`R5_11` Sec 3.2's bag: omega 30.76, zeta 0.768 on (M_jj + J), b = 10, c = 0.1, eps = 1e-3, h = 5e-4."""
    from elastic_sim.controllers import nominal_joint_inertia
    from elastic_sim.torque_runners import RolloutDiverged, run_mujoco_torque

    config = configs["iiwa"]
    asset = registry.load(config.asset)
    trajectory = _short(config, asset, "rigid")
    friction = FrictionModel.from_asset(asset)                   # the round-5 blanket law
    armature = config.nominal_rotor_inertia(7)

    def controller(rotor):
        inertia = nominal_joint_inertia(asset, trajectory) + rotor
        return JointPdController(asset, trajectory, joint_inertia=inertia, natural_frequency=30.76,
                                 damping_ratio=0.768)

    with pytest.raises(RolloutDiverged):
        run_mujoco_torque(asset, trajectory, controller(np.zeros(7)), time_step=5e-4, friction=friction)
    result = run_mujoco_torque(asset, trajectory, controller(armature), time_step=5e-4, friction=friction,
                               armature=armature)
    assert np.sqrt(np.mean((result["q_link"] - result["q_ref"]) ** 2)) < 0.05


@pytest.mark.slow
def test_noise_statistics_and_iiwa_velocity_derivative(configs, registry):
    from elastic_sim.round6_checks import check_noise, check_position_derivative

    config = configs["iiwa"]
    asset = registry.load(config.asset)
    config, result, frame = _run(config, asset, "e00")
    manifest = {"noise": config.noise.describe(), "joint_names": list(asset.joint_names), "n_dof": 7,
                "differentiation": differentiation_policy(config), "sample_time_step": config.sample_time_step}
    noise = check_noise(frame, manifest)
    assert noise["ok"], noise
    derivative = check_position_derivative(frame, manifest)
    assert derivative["ok"], derivative


@pytest.mark.slow
def test_gainshift_file_is_the_test_split_outside_the_training_band(configs, registry, tmp_path):
    import build_round6
    from diagnose_controller_modes import _shrink

    from elastic_sim.round6_checks import gate_gainshift

    config = _shrink(_short_config(configs["fmrr"]), trajectories=1, robots=3, backends=("mujoco",))
    asset = registry.load(config.asset)
    cache: dict = {}
    production = build_round6.build_variant(config, asset, "production", out_dir=tmp_path, stem="p", jobs=1,
                                            verbose=False, save=False, trajectory_cache=cache)
    shifted = build_round6.build_variant(config, asset, "gainshift", out_dir=tmp_path, stem="p", jobs=1,
                                         verbose=False, save=False, trajectory_cache=cache,
                                         production_manifest=production["manifest"])
    gate = gate_gainshift(production["manifest"], shifted["manifest"])
    assert gate["ok"], gate
    assert gate["gainshift_robots"] == production["manifest"]["split"]["test"]
    fractions = [r["omega_fraction"] for r in shifted["manifest"]["records"]]
    assert all(f < 0.5 or f > 0.9 for f in fractions)


# ---------------------------------------------------------------------------
# R6_02 pass 2
# ---------------------------------------------------------------------------

def test_iiwa_asset_has_the_kuka_per_axis_limits(registry):
    """P2-4 [S-13]: effort 320/320/176/176/110/40/40 Nm; speeds 85/85/100/75/130/135/135 deg/s."""
    for name in ("kuka_lbr_iiwa_14_r820", "kuka_lbr_iiwa_14_r820_table"):
        joints = registry.load(name).resolve_active_joints()
        assert [j.effort for j in joints] == [320.0, 320.0, 176.0, 176.0, 110.0, 40.0, 40.0]
        assert np.allclose(np.degrees([j.velocity for j in joints]), [85, 85, 100, 75, 130, 135, 135])


def test_auto_time_steps_are_the_largest_that_keep_the_bound(registry):
    """P2-4: ``auto`` steps come from the Sec 3 table, not by hand."""
    from elastic_sim.dataset_bounds import bound_limited_steps

    config = load_config(_REPO / CONFIGS["iiwa"])
    assert set(config.auto_time_steps) == {"max_time_step", "rigid_time_step"}
    asset = registry.load(config.asset)
    steps = bound_limited_steps(config, asset)
    assert config.max_time_step == steps["elastic"]["time_step"] == pytest.approx(2.5e-4)
    assert config.rigid_time_step == steps["rigid"]["time_step"] == pytest.approx(5.0e-4)
    for tier, row in steps.items():
        assert row["value"] <= 1.0
        # The next coarser step that divides the period breaks the bound (or the ceiling).
        count = round(config.control_period / row["time_step"])
        coarser = config.control_period / max(count - 1, 1)
        assert coarser > min(row["bound_step"], row["ceiling"])
    totals = dof_totals(explicit_term_table(config, asset))
    assert max(totals.values()) <= 1.0


def test_encoder_noise_is_one_quantum_not_a_percentage(tmp_path):
    """P2-3: q is quantization plus sigma = sigma_quanta steps; a percentage of RMS is refused."""
    channel = ChannelNoise(quantization=(2.0e-6, 1.0e-6), sigma_quanta=1.0)
    assert np.allclose(channel.sigma_for(np.random.default_rng(0).normal(size=(100, 2))), [2.0e-6, 1.0e-6])
    for name in CONFIGS:
        q = load_config(_REPO / CONFIGS[name], check_bounds=False).noise.q
        assert q.p == 0.0 and q.sigma_quanta == 1.0 and all(v > 0 for v in q.quantization)
    raw = _yaml("iiwa")
    raw["simulation"]["noise"]["q"]["p"] = 1.0e-4
    with pytest.raises(ValueError, match="sigma_quanta"):
        load_config(_write(tmp_path, raw), check_bounds=False)


def test_link_mode_table_and_observable_joints(registry, configs):
    """P2-1: f_link = sqrt(K / M_link) / 2 pi over the prior; observable = band inside 0.09 x rate."""
    from elastic_sim.link_modes import link_mode_table

    expected = {"iiwa": ["iiwa_A2", "iiwa_A4"], "iiwa_drive": ["iiwa_A2", "iiwa_A4"],
                "ur10": ["shoulder_lift_joint", "elbow_joint"],
                "fmrr": ["joint_y", "joint_x", "joint_z"]}
    for name, config in configs.items():
        modes = link_mode_table(config, registry.load(config.asset))
        for mode in modes:
            assert mode.f_low == pytest.approx(np.sqrt(mode.k_min / mode.m_link_max) / (2 * np.pi))
            assert mode.resolvable_hz == pytest.approx(0.09 / config.sample_time_step)
        assert [m.joint for m in modes if m.observable] == expected[name]
        assert all(m.mostly_observable for m in modes if m.observable)
    # Excluded by the strict band only through rare folded postures: >= 95 % of their prior resolves.
    mostly = {"iiwa": ["iiwa_A1", "iiwa_A2", "iiwa_A3", "iiwa_A4"],
              "ur10": ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint"]}
    for name, joints in mostly.items():
        modes = link_mode_table(configs[name], registry.load(configs[name].asset))
        assert [m.joint for m in modes if m.mostly_observable] == joints
    # The wrist that cannot be seen at this rate: iiwa A7's band starts above 20 Hz and ends near 1 kHz.
    a7 = link_mode_table(configs["iiwa"], registry.load(configs["iiwa"].asset))[-1]
    assert a7.f_high > 10 * a7.resolvable_hz and not a7.observable


def test_probe_design_covers_the_link_modes_inside_the_resolvable_band(registry, configs, tmp_path):
    from elastic_sim.link_modes import link_mode_table

    for name, config in configs.items():
        excitation = config.excitation
        frequencies = np.asarray(excitation.probe_harmonics) * excitation.base_frequency
        modes = link_mode_table(config, registry.load(config.asset))
        resolvable = 0.09 / config.sample_time_step
        assert frequencies[-1] <= resolvable + 1e-9
        assert frequencies[0] <= min(config.probe_design.lowest_hz, *(0.5 * m.f_low for m in modes)) + 0.1
        assert differentiation_policy(config)["probe_resolvable"]
        for mode in modes:
            top = min(1.5 * mode.f_high, resolvable)
            assert frequencies.min() <= 0.5 * mode.f_low + 0.1 and frequencies.max() >= top - 1e-9
    raw = _yaml("ur10")
    raw["excitation"]["probe_harmonics"] = [30, 40]
    with pytest.raises(ValueError, match="exclusive"):
        load_config(_write(tmp_path, raw), check_bounds=False)


def test_drive_location_bounds_and_sizing(registry, configs, tmp_path):
    """Sec 7: the UR10 PD in its 2 kHz drive, d = 0, gains on J + m_nom, per-axis omega_max."""
    from elastic_sim.dataset_bag import drive_gains

    config = configs["ur10"]
    assert (config.control_period, config.control_delay, config.drive_ticks) == (5.0e-4, 0, 4)
    asset = registry.load(config.asset)
    trajectory = _short(config, asset, "e00")
    sizing = drive_gains(config, asset, trajectory)
    motor = config.nominal_rotor_inertia(6)
    assert np.allclose(sizing["frequency_scale"], motor / (motor + sizing["load_inertia"]))
    # m_nom is the nominal M_jj averaged along the reference (R6_02 Sec 7 amended).
    from elastic_sim.controllers import nominal_joint_inertia
    assert np.allclose(sizing["load_inertia"], nominal_joint_inertia(asset, trajectory))
    omega = 0.9 * config.omega_max()
    draw = ControllerDraw(omega, 1.3, 1.0, 1.0, 1.0)
    controller = build_controller(config.controller, draw, asset=asset, nominal_asset=asset, trajectory=trajectory,
                                  friction=FrictionModel.from_asset(asset), motor_inertia=motor,
                                  load_inertia=sizing["load_inertia"], frequency_scale=sizing["frequency_scale"])
    inertia = motor + sizing["load_inertia"]
    assert np.allclose(controller.kp, inertia * (omega * sizing["frequency_scale"]) ** 2)
    # The rotor's own loop: kd T / J is the bus bound's value, whatever the load.
    assert np.allclose(controller.kd * config.control_period / motor, 2 * 1.3 * omega * config.control_period)
    assert np.all(controller.kd * config.control_period / motor <= 0.5 + 1e-9)
    # A drive without a velocity sensor, or sized on the rotor alone, is refused.
    raw = _yaml("ur10")
    raw["simulation"]["controller"]["gain_sizing"] = "motor_inertia"
    with pytest.raises(ValueError, match="motor_plus_load"):
        load_config(_write(tmp_path, raw), check_bounds=False)
    raw = _yaml("ur10")
    raw["simulation"]["controller"]["drive"]["rate"] = 1250.0
    with pytest.raises(ValueError, match="integer multiple"):
        load_config(_write(tmp_path, raw), check_bounds=False)


def test_drive_instrument_differences_its_encoder_and_records_every_nth_reading():
    """R6_04 A-4: the loop reads a low-passed encoder difference; the percentage noise is on the record only."""
    noise = NoiseModel(q=ChannelNoise(quantization=(1e-3,)), dq=ChannelNoise(p=0.5), delay_samples=1)
    recorder = LoopInstrument(noise, n_dof=1, seed=0, sigma_q=np.zeros(1), sigma_dq=np.array([10.0]),
                              sample_time=0.002)
    period, cutoff = 5.0e-4, 200.0
    drive = DriveInstrument(recorder, ticks_per_sample=4, drive_period=period, cutoff_hz=cutoff)
    alpha = 1.0 - np.exp(-2.0 * np.pi * cutoff * period)
    speed = 2.0                                                    # rad/s, a ramp
    reads = [drive.sample(np.array([speed * period * k]), np.array([speed])) for k in range(400)]
    q_reads = np.array([r[0][0] for r in reads])
    assert q_reads == pytest.approx(np.round(speed * period * np.arange(400) / 1e-3) * 1e-3)
    # The in-loop velocity is the filter of the quantized difference: no 10 rad/s noise in it.
    velocity = speed
    for k in range(1, 400):
        velocity += alpha * ((q_reads[k] - q_reads[k - 1]) / period - velocity)
    assert reads[-1][1][0] == pytest.approx(velocity)
    assert abs(reads[-1][1][0] - speed) < 0.5
    history = drive.history()
    # Ticks 0, 4, 8, ... on the bus, one sample late; the recorded dq carries the record-only noise.
    assert history["q_motor_measured"][1:, 0] == pytest.approx(q_reads[0:396:4])
    assert np.std(history["dq_motor_measured"][:, 0] - speed) > 3.0
    # No velocity interface (the iiwa): the recorder derives dq itself, the drive still has its own.
    derivative = NoiseModel(q=ChannelNoise(), dq=ChannelNoise(), dq_source="position_derivative")
    iiwa = DriveInstrument(LoopInstrument(derivative, n_dof=1, seed=0, sigma_q=np.zeros(1), sigma_dq=np.zeros(1),
                                          sample_time=0.001), 4, drive_period=2.5e-4, cutoff_hz=400.0)
    for k in range(8):
        iiwa.sample(np.array([0.001 * k]), np.array([4.0]))
    assert "dq_motor_measured" not in iiwa.history()


def test_sag_gate_report_mode():
    from elastic_sim.round6_checks import gate_sag

    manifest = {"records": [{"bag": "a", "sag_max_fraction": 0.3, "sag_worst_joint": "joint_y"}]}
    assert not gate_sag(manifest, 0.05)["ok"]
    reported = gate_sag(manifest, 0.05, mode="report", reason="R6_02 Sec 7")
    assert reported["ok"] and not reported["within_limit"] and reported["reason"]


@pytest.mark.slow
@pytest.mark.parametrize("platform", ["ur10", "fmrr", "iiwa_drive"])
def test_drive_loop_runs_at_the_drive_rate_on_delayed_setpoints(platform, configs, registry):
    """R6_02 Sec 7 amended: PD at the drive rate on the drive's own readings, setpoints one bus sample late,
    interpolated over the bus period (CSP)."""
    config = replace(configs[platform], noise=replace(configs[platform].noise, enabled=False))
    asset = registry.load(config.asset)
    config, result, frame = _run(config, asset, "e00")
    n = len(asset.joint_names)
    decimation = result["control_decimation"]
    per_sample = decimation * config.drive_ticks
    tau = np.asarray(result["tau_motor"])
    blocks = tau[: (len(tau) - 1) // decimation * decimation].reshape(-1, decimation, n)
    assert np.all(blocks == blocks[:, :1, :])                     # held one drive period
    changes = np.any(np.diff(tau[: 40 * per_sample], axis=0) != 0.0, axis=1).sum()
    assert changes > 20 * config.drive_ticks                       # several commands per sample period
    # At sample instant k the drive used its own (here noise-free, so recorded) reading and, at the
    # start of its interpolation from setpoint k - 2 to the one-sample-late setpoint k - 1, setpoint k - 2.
    kp, kd = np.asarray(result["kp"]), np.asarray(result["kd"])
    q_ref = np.asarray(result["q_ref"])[::per_sample]
    dq_ref = np.asarray(result["dq_ref"])[::per_sample]
    q_read = np.asarray(result["q_motor_measured"])
    if "dq_motor_measured" in result:
        dq_read = np.asarray(result["dq_motor_measured"])
        k = np.arange(2, len(q_read))
        expected = kp * (q_ref[k - 2] - q_read[k]) + kd * (dq_ref[k - 2] - dq_read[k])
        assert np.allclose(tau[::per_sample][k], expected, rtol=1e-6, atol=1e-6 * float(np.max(np.abs(tau))))
    # The recorded tau is the demand at the sample instant.
    recorded = frame[[f"tau{i}" for i in range(n)]].to_numpy(dtype=float)
    assert np.allclose(recorded, tau[::per_sample][: len(recorded)], rtol=1e-12, atol=1e-9)
    # The rotor's own loop meets the bus bound (R6_04 A-2).
    assert np.max(kd * config.control_period / config.nominal_rotor_inertia(n)) <= 0.5 + 1e-9


@pytest.mark.slow
def test_bus_code_path_still_runs_for_a_drive_platform(configs, registry):
    """R6_04 A-1: `bus` stays a code path on UR10/FMRR -- one smoke test, no tuning, no gates."""
    config = configs["fmrr"]
    bus = replace(config, controller=replace(config.controller, location="bus", gain_sizing="motor_inertia"))
    bus = replace(bus, control_gains=bus.control_gains.resolved(bus.omega_max()))
    asset = registry.load(bus.asset)
    _, result, frame = _run(bus, asset, "e00")
    assert np.isfinite(frame[[f"ft{i}" for i in range(3)]].to_numpy()).all()
    assert result["control_decimation"] * result["time_step"] == pytest.approx(bus.sample_time_step)


def test_drive_setpoints_are_interpolated_one_sample_late(configs, registry):
    config = configs["ur10"]
    asset = registry.load(config.asset)
    trajectory = _short(config, asset, "e00")
    controller = JointPdController(asset, trajectory, joint_inertia=np.ones(6), natural_frequency=10.0,
                                   setpoint_period=0.002, setpoint_delay=1)
    period = 0.002
    for t in (0.0101, 0.0137, 0.5):
        k = int(np.floor(t / period + 1e-9))
        fraction = t / period - k
        q_a = trajectory((k - 2) * period)[0]
        q_b = trajectory((k - 1) * period)[0]
        assert np.allclose(controller.setpoint(t)[0], q_a + fraction * (q_b - q_a))
    # Continuous across a bus instant: no staircase for kp to turn into a spike.
    before, after = controller.setpoint(0.02 - 1e-9)[0], controller.setpoint(0.02)[0]
    assert np.max(np.abs(after - before)) < 1e-6
