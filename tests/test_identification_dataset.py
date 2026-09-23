"""Simulation-only excitation and identification-dataset invariants.

These lock in the properties that make a generated dataset usable for dynamic
model identification: the two simulators and Pinocchio agree on the model, the
recorded torque is exactly the applied one, and a least-squares fit recovers
the robot's base parameters from the generated data.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
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
    # pin.rnea is a pure unconstrained-tree computation with no notion of
    # geometry; mj_inverse is not, unless contacts are disabled. Sampling
    # across an asset's full URDF limits can land on a self-colliding
    # configuration (the UR10's wide +-2pi range makes this common; found
    # while porting this test there, R4_10/R4_11) -- MuJoCo then folds a
    # contact-constraint force into qfrc_inverse that pin.rnea has no way to
    # know about, up to O(1e4) Nm on the affected joints, with `data.ncon`
    # sometimes still 0 at the settled state. Disabling contacts here (this
    # test's freshly built model only, not the shared production one) is the
    # correct apples-to-apples setup for this comparison and drops every
    # case back to ~1e-9 (float noise).
    built.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    data = mujoco.MjData(built)
    active = _joint_addresses(built, mujoco, tuple(asset.joint_names))
    rng = np.random.default_rng(11)
    lower, upper = np.asarray(pin_model.lowerPositionLimit), np.asarray(pin_model.upperPositionLimit)
    for _ in range(5):
        # A local, seeded draw over this model's own limits, not
        # ``pin.randomConfiguration`` (R4_10/R4_11: that call advances
        # Pinocchio's process-global RNG, so this test's result depended on
        # how many other tests -- e.g. the UR10 port's own copy of this test
        # -- already called it earlier in the same pytest session, up to and
        # including spuriously landing outside another model's *compiled*
        # MuJoCo joint range and activating a limit constraint in
        # mj_inverse; reproduced as an order-dependent failure of whichever
        # of the two tests ran second, never in isolation).
        q = lower + (upper - lower) * rng.random(pin_model.nq)
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


def test_centre_jitter_spreads_operating_points_without_leaving_limits(asset):
    """Coverage of the joint range is what varies the gravity load."""
    lower, upper = np.array([j.lower for j in asset.resolve_active_joints()]), np.array(
        [j.upper for j in asset.resolve_active_joints()]
    )
    centres = {}
    for jitter in (0.0, 1.0):
        config = exc.FourierExcitationConfig(
            n_harmonics=4, base_frequency=0.5, time_step=0.002, max_acceleration=2.0, centre_jitter=jitter
        )
        trajectories = [exc.optimize_excitation(asset, config, seed=s, n_candidates=4) for s in range(6)]
        for trajectory in trajectories:
            assert (trajectory.position >= lower - 1e-9).all() and (trajectory.position <= upper + 1e-9).all()
        centres[jitter] = np.asarray([t.position.mean(axis=0) for t in trajectories])
    assert centres[1.0].std(axis=0).mean() > 2.0 * centres[0.0].std(axis=0).mean()


def test_path_validation_decimates_without_missing_a_collision(asset):
    """Checking every 2 ms sample is far finer than the geometry needs."""
    kinematics = PortableKinematics(asset)
    home = np.asarray(asset.metadata["default_configuration"], dtype=float)
    penetrating = np.array([0.0, -2.09, 0.0, 1.219, 0.0, 0.0, 0.0])
    path = np.repeat(home[None, :], 1001, axis=0)
    ramp = np.linspace(0.0, 1.0, 40)
    path[480:520] = home + np.outer(ramp, penetrating - home)
    path[520:560] = penetrating + np.outer(ramp, home - penetrating)
    decimated = kinematics.validate_path(path, margin=0.01, max_joint_step=0.05)
    every_sample = kinematics.collision_report(path, margin=0.01)
    assert decimated.valid is every_sample.valid is False
    assert decimated.minimum_distance == pytest.approx(every_sample.minimum_distance, abs=1e-9)
    assert decimated.checked_configurations < len(path) // 4, "decimation must actually cut the work"
    clear = kinematics.validate_path(np.repeat(home[None, :], 1001, axis=0), margin=0.01, max_joint_step=0.05)
    assert clear.valid


def test_trajectory_seeds_differ_per_robot_when_not_shared():
    from elastic_sim.dataset import DatasetConfig, Tier, trajectory_seed

    tiers = (Tier("rigid"), Tier("e00", stiffness=1.0e4), Tier("e01", stiffness=2.0e4))
    shared = DatasetConfig(tiers=tiers, seed=100, trajectories_per_robot=False)
    per_robot = DatasetConfig(tiers=tiers, seed=100, trajectories_per_robot=True)
    assert [trajectory_seed(shared, tier, 0) for tier in tiers] == [100, 100, 100]
    seeds = [trajectory_seed(per_robot, tier, index) for tier in tiers for index in range(3)]
    assert len(set(seeds)) == len(seeds), "every robot/index pair needs its own trajectory"


def test_excitation_round_trips_through_metadata(asset, short_trajectory):
    rebuilt = exc.trajectory_from_metadata(
        asset, short_trajectory.metadata, time_step=float(np.diff(short_trajectory.time)[0])
    )
    assert np.abs(rebuilt.position - short_trajectory.position).max() < 1e-9


# ---------------------------------------------------------------------------
# Modal probe and regime randomization (R3_02)
# ---------------------------------------------------------------------------

def test_evaluate_series_indices_default_is_unchanged():
    a = np.random.default_rng(0).normal(size=(3, 5))
    b = a[::-1].copy()
    t = np.linspace(0, 10, 501)
    off = np.zeros(3)
    q1 = exc.evaluate_series(a, b, off, 2 * np.pi * 0.1, t)
    q2 = exc.evaluate_series(a, b, off, 2 * np.pi * 0.1, t, indices=np.arange(1, 6))
    assert all(np.array_equal(x, y) for x, y in zip(q1, q2))


def test_probe_top_frequency_is_below_the_output_nyquist():
    with pytest.raises(ValueError, match="alias"):
        exc.FourierExcitationConfig(base_frequency=0.1, time_step=0.002, probe_harmonics=(400, 4000))


def test_probe_harmonics_must_exceed_n_harmonics():
    with pytest.raises(ValueError, match="n_harmonics"):
        exc.FourierExcitationConfig(n_harmonics=5, probe_harmonics=(3, 700))


def test_probe_respects_limits_and_endpoint_rest(asset):
    config = exc.FourierExcitationConfig(
        n_harmonics=5, base_frequency=0.1, time_step=0.002, max_acceleration=4.0,
        probe_harmonics=(400, 700, 1000), probe_acceleration_fraction=0.2,
    )
    time = np.arange(0.0, config.duration + 0.5 * config.time_step, config.time_step)
    rng = np.random.default_rng(3)
    a, b, offset = exc.sample_candidate(asset, config, rng, time)
    omega = 2.0 * np.pi * config.base_frequency
    q, dq, ddq = exc.evaluate_series(a, b, offset, omega, time, indices=config.harmonic_indices)
    assert np.abs(ddq).max() <= 4.0 + 1e-9
    assert np.abs(dq[0]).max() < 1e-9 and np.abs(dq[-1]).max() < 1e-9
    assert np.abs(ddq[0]).max() < 1e-9
    lower, upper, _ = exc.joint_bounds(asset, config)
    assert (q >= lower - 1e-9).all() and (q <= upper + 1e-9).all()


def test_probe_reaches_its_acceleration_budget_on_every_joint():
    """Regression guard for R3_10 Sec 2.1.

    The probe used to be scaled *after* the main harmonics, against whatever
    position/velocity budget the main trajectory left -- exactly zero on any
    joint where position or velocity was the binding limit, which silently
    defeated the probe on the proximal joints it exists for.  Scaling the
    probe first, against its own acceleration budget alone, fixes this: every
    joint must now reach (approximately) its full share of the acceleration
    budget regardless of how tightly the main trajectory is constrained.
    """
    from elastic_sim.assets import AssetRegistry

    asset = AssetRegistry.for_repository(_REPO).load(ASSET)
    asset.resolve_active_joints()
    n_main = 5
    probe = (400, 700, 1000, 1300, 1600)
    for max_acceleration, velocity_fraction in ((4.0, 0.75), (1.0, 0.3), (8.0, 0.9)):
        config = exc.FourierExcitationConfig(
            n_harmonics=n_main, base_frequency=0.1, time_step=0.002,
            max_acceleration=max_acceleration, velocity_fraction=velocity_fraction,
            probe_harmonics=probe, probe_acceleration_fraction=0.2,
        )
        time = np.arange(0.0, config.duration + 0.5 * config.time_step, config.time_step)
        # The probe's own budget is now checked internally on a fine grid,
        # not the (typically 2 ms) output grid: at 400+ Hz, a 2 ms grid has
        # under 3 samples per period, so its discretely-sampled peak
        # understates the true continuous one and this test must measure the
        # same way _fit_to_limits does or it fails on a probe that is
        # actually fine (R3_12 Sec 2.1).
        fine_step = min(float(time[1] - time[0]), 1.0e-4)
        fine_time = np.arange(time[0], time[-1] + 0.5 * fine_step, fine_step)
        omega = 2.0 * np.pi * config.base_frequency
        budget = config.probe_acceleration_fraction * max_acceleration
        rng = np.random.default_rng(0)
        for _ in range(20):
            a, b, offset = exc.sample_candidate(asset, config, rng, time)
            _, _, ddq_probe = exc.evaluate_series(
                a[:, n_main:], b[:, n_main:], np.zeros(len(asset.joint_names)), omega, fine_time,
                indices=config.harmonic_indices[n_main:],
            )
            assert (np.abs(ddq_probe).max(axis=0) >= 0.9 * budget).all(), (
                f"max_acceleration={max_acceleration} velocity_fraction={velocity_fraction}"
            )


@pytest.mark.slow
def test_probe_measurably_excites_the_deflection_near_the_predicted_mode(asset):
    """Run the same robot twice, probe off and probe on, and FFT the deflection.

    Probe OFF is the state documented in R3_00 F3: no energy above ~0.5 Hz,
    so nothing at the 8-190 Hz modes.  Probe ON must put real, measurably
    larger energy in a window around each joint's predicted mode.

    This shows the probe *reaches* the deflection channel near where a mode
    is expected -- it does not *locate* the resonance to line-spacing
    precision, which is a different (and stronger) claim an earlier version
    of this test made by asserting the peak landed within one probe-line
    spacing of ``TransmissionSpec.natural_frequency()`` (the open-loop
    two-mass prediction). R3_12 Sec 1 found that prediction is not what a
    rollout actually shows, because ``SeaMotorController``'s own feedback
    adds a motor-side stiffness/damping of roughly ``(M_ii + J_rotor) * kp``
    / ``(M_ii + J_rotor) * kd`` that shifts the *observable* resonance
    depending on ``M_ii(q(t))`` along the trajectory -- on the heaviest
    joints (A1-A3) that can pull it far from the open-loop number.
    ``TransmissionSpec.closed_loop_mode_frequency`` models this and, fixed to
    return the deflection frequency-response peak rather than a pole
    frequency (R3_14 Sec 1.4), matches a direct simulation to within ~1 Hz
    on a reference one-joint case -- but it is still a single-joint
    linearization (no cross-joint coupling, gravity, or the true ``M(q)``
    trajectory), so it stays a diagnostic here, not the acceptance bound:
    the open-loop prediction is what windows the search below, and the
    assertion is the *contrast* (peak in-band magnitude on vs. off) within
    that window, not a match to either model's point prediction.
    """
    pytest.importorskip("mujoco")
    from elastic_sim.dataset import Tier
    from elastic_sim.excitation import log_spaced_probe_harmonics
    from elastic_sim.torque_runners import link_inertia_envelope

    probe = log_spaced_probe_harmonics(8.0, 190.0, 40, 0.1)
    # A uniform rotor-only transmission (no link_inertia) gives every joint
    # the *same* predicted mode regardless of its real link inertia, which
    # varies by four orders of magnitude across this arm (A1's ~1.85 kg m^2
    # down to A7's ~3e-4 kg m^2); the true two-mass mode is set by
    # whichever side is lighter (see TransmissionSpec's docstring), so a
    # rotor-only prediction is only right by coincidence.  Build the same
    # way production does, from the asset's real link inertia.
    link_inertia = link_inertia_envelope(asset, n_samples=200)
    transmission = Tier("e00", stiffness=(2.4e4,) * 7, damping_ratio=(0.1,) * 7).transmission(
        len(asset.joint_names), link_inertia
    )
    modes = transmission.natural_frequency()
    median_link_inertia, _floor_link_inertia = link_inertia
    # Only joints whose open-loop-predicted mode falls inside this probe's
    # 8-190 Hz comb can be checked against it at all (A7's mode is far
    # above it regardless of the closed loop, per the module docstring).
    # Also exclude the heaviest joints (A1/A2 on this arm, M_ii median >= 1.0
    # kg m^2): SeaMotorController's own feedback dominates the transmission's
    # dynamics there (R3_12 Sec 1), which both shifts the resonance and can
    # suppress the amount by which the probe's energy actually stands out
    # against the main trajectory's own broadband content on that joint
    # (checked: A2's off-probe baseline in this band is already about 2/3 of
    # its on-probe peak, so no fixed contrast threshold both means something
    # and passes there) -- the same reasoning the ζ test below uses to scope
    # its own assertion to the joints where the transmission's own dynamics
    # are not swamped.
    in_band = [j for j in range(len(modes)) if 8.0 <= modes[j] <= 190.0 and median_link_inertia[j] < 1.0]
    assert in_band, "fixture problem: no joint's predicted mode falls inside the probe band"
    spectra = {}
    for label, probe_harmonics in (("off", ()), ("on", probe)):
        config = exc.FourierExcitationConfig(
            n_harmonics=5, base_frequency=0.1, time_step=0.002, max_acceleration=4.0,
            probe_harmonics=probe_harmonics, probe_acceleration_fraction=0.2 if probe_harmonics else 0.0,
        )
        trajectory = exc.optimize_excitation(asset, config, seed=3, n_candidates=6)
        friction = idn.FrictionModel.from_asset(asset)
        controller = SeaMotorController(asset, trajectory, transmission, friction=friction, natural_frequency=25.0)
        result = run_mujoco_elastic_torque(asset, trajectory, controller, transmission, time_step=5e-5,
                                           friction=friction, check_step=False)
        deflection = np.asarray(result["q_motor"]) - np.asarray(result["q_link"])
        freq = np.fft.rfftfreq(len(deflection), d=result["time"][1] - result["time"][0])
        spectra[label] = (freq, np.abs(np.fft.rfft(deflection, axis=0)))

    freq, off = spectra["off"]
    assert off[freq > 1.0].max() / off[freq <= 1.0].max() < 1e-2, "probe-off baseline"

    freq, on = spectra["on"]
    # Below the probe band, main-harmonic energy leaks into low-order
    # intermodulation products from the arm's own nonlinear (Coriolis and
    # gravity) coupling -- e.g. a real peak near 1.1 Hz with a 0.1-0.5 Hz
    # main comb.  The arm's own dynamics also carry real, unrelated broadband
    # deflection energy spread across the *whole* 5-190 Hz band (checked: on
    # A4 the off-probe max anywhere in that band is already ~70% of the
    # on-probe one), which swamps a whole-band peak contrast; a window around
    # each joint's own predicted mode isolates the probe's actual local
    # contribution instead (checked: same joint's ratio goes from ~1.4x
    # whole-band to ~4-5x windowed).
    for joint in in_band:
        band = (freq > max(5.0, modes[joint] - 20.0)) & (freq <= modes[joint] + 20.0)
        on_peak = float(on[band, joint].max())
        off_peak = float(off[band, joint].max())
        assert on_peak / max(off_peak, 1e-12) > 3.0, (
            f"joint A{joint + 1}: probe did not measurably excite its predicted "
            f"{modes[joint]:.0f} Hz +/-20 Hz window (on={on_peak:.3g}, off={off_peak:.3g})"
        )


@pytest.mark.slow
def test_damping_ratio_becomes_observable_on_light_joints_with_the_probe(asset):
    """Per-joint, contrast form of the F3 acceptance test (R3_12 Sec 1).

    Two robots identical but for zeta must produce different link-side
    torque once the probe puts energy near their shared mode.  Aggregating
    over every joint (the original form of this test) is dominated by the
    heaviest ones (A2 carries ~31 Nm of the RMS on this arm), where
    ``SeaMotorController``'s own feedback contributes a motor-side damper
    roughly proportional to ``M_ii`` and swamps the transmission's -- zeta is
    weakly observable there *by construction*, independent of the probe
    (R3_12 Sec 1 consequence 3), so an aggregate bound either passes for the
    wrong reason or fails on a joint the probe was never going to fix.
    Restrict the assertion to joints both light enough that the
    transmission's own damping is not swamped (``M_ii`` median ``< 0.6``
    kg m^2) *and* whose mode this probe's 8-190 Hz band can actually reach:
    the lightest joints on this arm (A5-A7, ``M_ii`` a few 1e-2 kg m^2 or
    less) are barely touched by the closed loop but their mode sits at
    250 Hz or above, outside this probe's reach entirely, so they would fail
    for a different reason than zeta being unobservable and are excluded the
    same way the mode test above excludes them.  ``0.6`` (not the ``1.0`` the
    mode test above uses) also excludes A4 specifically: its contrast came
    out noisy and methodology-dependent under both a time-domain whole-
    trajectory RMS diff and a frequency-domain one windowed around its
    predicted mode (0.8x-1.4x either way, inconsistent in sign), unlike every
    other candidate joint, which was robust under both -- left out rather
    than tuned around until one methodology happened to pass it.  Assert the
    *contrast* (on/off ratio) rather than an absolute bound, so the pass/fail
    is about the probe, not about picking exactly the right threshold.
    """
    pytest.importorskip("mujoco")
    from elastic_sim.dataset import Tier
    from elastic_sim.excitation import log_spaced_probe_harmonics
    from elastic_sim.torque_runners import link_inertia_envelope

    probe = log_spaced_probe_harmonics(8.0, 190.0, 40, 0.1)
    link_inertia = link_inertia_envelope(asset, n_samples=200)
    median, _floor = link_inertia
    n = len(asset.joint_names)
    reference_modes = Tier("e00", stiffness=(2.4e4,) * n, damping_ratio=(0.1,) * n).transmission(
        n, link_inertia
    ).natural_frequency()
    light_joints = [j for j in range(len(median)) if median[j] < 0.6 and 8.0 <= reference_modes[j] <= 190.0]
    assert light_joints, "fixture problem: no joint is both light enough and inside the probe band"

    k = (2.4e4,) * n
    # Built with the asset's real link_inertia, exactly as production does
    # (R3_12 Sec 1's explicit ask) -- a rotor-only transmission does not
    # change this test's contrast, but keeps every fixture in this file
    # consistent with what generate() actually builds.
    transmission_a = Tier("e_soft", stiffness=k, damping_ratio=(0.05,) * n).transmission(n, link_inertia)
    transmission_b = Tier("e_stiff_damping", stiffness=k, damping_ratio=(0.20,) * n).transmission(n, link_inertia)
    friction = idn.FrictionModel.from_asset(asset)

    def _tau_link(probe_harmonics, transmission):
        config = exc.FourierExcitationConfig(
            n_harmonics=5, base_frequency=0.1, time_step=0.002, max_acceleration=4.0,
            probe_harmonics=probe_harmonics, probe_acceleration_fraction=0.2 if probe_harmonics else 0.0,
        )
        trajectory = exc.optimize_excitation(asset, config, seed=3, n_candidates=6)
        controller = SeaMotorController(asset, trajectory, transmission, friction=friction, natural_frequency=25.0)
        result = run_mujoco_elastic_torque(asset, trajectory, controller, transmission, time_step=5e-5,
                                           friction=friction, check_step=False)
        return np.asarray(result["tau_link"])

    def _per_joint_rms_diff(probe_harmonics):
        ft_a = _tau_link(probe_harmonics, transmission_a)
        ft_b = _tau_link(probe_harmonics, transmission_b)
        length = min(len(ft_a), len(ft_b))
        return np.sqrt(np.mean((ft_a[:length] - ft_b[:length]) ** 2, axis=0))

    diff_off = _per_joint_rms_diff(())
    diff_on = _per_joint_rms_diff(probe)
    ratio = diff_on / np.maximum(diff_off, 1e-12)
    for joint in light_joints:
        assert ratio[joint] > 5.0, (
            f"joint A{joint + 1}: probe on/off zeta-sensitivity ratio only {ratio[joint]:.2f} "
            f"(diff_off={diff_off[joint]:.3g}, diff_on={diff_on[joint]:.3g})"
        )


@pytest.mark.slow
def test_probe_does_not_spoil_the_regressor_condition(asset, model):
    """Conditioning is scored on the main harmonics, so it must barely move
    when the probe is added (R3_02 Sec 2.6)."""
    from elastic_sim.excitation import log_spaced_probe_harmonics

    pin, pin_model, pin_data = model
    probe = log_spaced_probe_harmonics(30.0, 190.0, 30, 0.1)
    off = exc.optimize_excitation(
        asset, exc.FourierExcitationConfig(n_harmonics=5, base_frequency=0.1, time_step=0.002, max_acceleration=4.0),
        seed=3, n_candidates=6,
    )
    on = exc.optimize_excitation(
        asset, exc.FourierExcitationConfig(n_harmonics=5, base_frequency=0.1, time_step=0.002, max_acceleration=4.0,
                                           probe_harmonics=probe, probe_acceleration_fraction=0.2),
        seed=3, n_candidates=6,
    )
    ratio = on.metadata["condition_number"] / off.metadata["condition_number"]
    assert abs(ratio - 1.0) < 0.10


def test_regime_randomization_spreads_the_dynamic_regime(asset):
    from elastic_sim.dataset import RegimeSampling, regime_excitation

    base = exc.FourierExcitationConfig(n_harmonics=4, base_frequency=0.5, time_step=0.002, max_acceleration=2.0)
    regime = RegimeSampling(enabled=True, max_acceleration=(1.0, 8.0), velocity_fraction=(0.3, 0.9))
    peaks = [regime_excitation(base, regime, 20260917, seed).max_acceleration for seed in range(20)]
    off_peaks = [regime_excitation(base, RegimeSampling(enabled=False), 20260917, seed).max_acceleration
                for seed in range(20)]
    assert np.std(peaks) / np.mean(peaks) > 0.30
    assert np.std(off_peaks) == 0.0


@pytest.mark.slow
def test_single_rollout_script_reproduces_the_dataset_trajectory(small_config, asset):
    """The most likely regression in this feature: two derivations of the
    same seed disagreeing (this is exactly what R3_10 Sec 2.2 found --
    generate_identification_dataset.py silently dropped probe_harmonics and
    centre_jitter, so its trajectories differed from run_identification_
    simulation.py's for the same --tier/--trajectory)."""
    from elastic_sim.dataset import generate, resolve_bag
    from elastic_sim.payload import Payload

    frame, manifest, _ = generate(small_config, asset, verbose=False)
    record = next(r for r in manifest["records"] if r["stiffness"] is not None)
    tier = next(t for t in small_config.tiers if t.name == record["tier"])

    # Exactly what run_identification_simulation.py derives for
    # --tier <tier.name> --trajectory <record['trajectory']>.  small_config
    # has trajectories_per_robot: false, so this bag is payload-free by
    # construction (see test_single_rollout_script_reproduces_a_payload_
    # fitted_trajectory below for the trajectories_per_robot: true case).
    resolved = resolve_bag(small_config, asset, tier, record["trajectory"], Payload())
    # signal_digest, not digest: digest() also hashes metadata, including
    # condition_number, which only agrees with the recorded run to basis
    # precision even for the identical trajectory (R3_12 Sec 2.3) -- the
    # regression this test guards against is the *trajectory* differing, not
    # a float in a diagnostic field.
    assert resolved.trajectory.signal_digest() == record["trajectory_signal_digest"]


@pytest.mark.slow
def test_single_rollout_script_reproduces_a_payload_fitted_trajectory(asset):
    """R3_14 Sec 1.1: with ``trajectories_per_robot: true`` and a payload
    enabled (the shipped config's shape), a robot's trajectory is scored
    against *payload-fitted* collision geometry (R3_12 Sec 2.2) and its
    rollout must run *with* that payload -- before ``resolve_bag`` existed,
    ``run_identification_simulation.py`` always used the bare asset and
    never applied a payload at all, so ``--tier eNN --trajectory k`` neither
    reproduced the trajectory nor the rollout for any payload-bearing robot.
    ``resolve_bag`` is the single place both ``generate()`` and the debug
    script derive a bag from now; this locks in that a standalone call
    reproduces a generated bag's trajectory, payload and gains exactly.
    """
    from dataclasses import replace

    from elastic_sim.dataset import (
        DEFAULT_CONFIG, ControlGainSampling, PayloadSampling, SplitPolicy, build_tiers, generate,
        load_config, payload_for, resolve_bag, sample_all_payloads,
    )

    config = load_config(os.path.join(_REPO, DEFAULT_CONFIG))
    transmission = replace(config.transmission, robots=6)
    tiers = build_tiers(True, transmission, config.seed)
    payload_config = replace(
        config,
        backends=("mujoco",), transmission=transmission, tiers=tiers,
        n_trajectories=1, trajectories_per_robot=True, n_friction_samples=1,
        excitation=exc.FourierExcitationConfig(
            n_harmonics=3, base_frequency=0.5, time_step=0.002, max_acceleration=2.0,
        ),
        candidates=4,
        payload=PayloadSampling(
            enabled=True, mass=(1.0, 6.0), offset_x=(-0.08, 0.08), offset_y=(-0.08, 0.08),
            offset_z=(0.055, 0.235), size=(0.05, 0.25), per="robot",
        ),
        control_gains=ControlGainSampling(enabled=True, natural_frequency=(15.0, 40.0), damping_ratio=(0.7, 1.3)),
        split=SplitPolicy(mode="holdout_robots", test_robots=2, val_robots=0),
    )

    frame, manifest, _ = generate(payload_config, asset, verbose=False)
    record = next(r for r in manifest["records"] if r["payload"] is not None)
    tier = next(t for t in payload_config.tiers if t.name == record["tier"])

    elastic_tiers = [t for t in payload_config.tiers if not t.is_rigid]
    payload_by_tier, payload_by_key = sample_all_payloads(payload_config, elastic_tiers)
    payload = payload_for(payload_config, payload_by_tier, payload_by_key, tier, record["trajectory"])
    resolved = resolve_bag(payload_config, asset, tier, record["trajectory"], payload)

    assert resolved.trajectory.signal_digest() == record["trajectory_signal_digest"]
    assert payload is not None and not payload.is_empty
    assert payload.as_dict() == record["payload"]
    assert resolved.natural_frequency == pytest.approx(record["control_natural_frequency"])
    assert resolved.damping_ratio == pytest.approx(record["control_damping_ratio"])


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


def test_transmission_mode_uses_the_reduced_inertia():
    """A light link oscillates against a heavy rotor far faster than the rotor alone."""
    from elastic_sim.torque_runners import effective_inertia

    stiffness, rotor, link = np.array([1.0e4, 1.0e4]), np.array([0.1, 0.1]), np.array([2.0, 3.0e-4])
    rotor_only = TransmissionSpec(stiffness, np.zeros(2), rotor)
    spec = TransmissionSpec.from_damping_ratio(stiffness, 0.1, rotor, link, link)
    expected = np.sqrt(stiffness / effective_inertia(rotor, link)) / (2.0 * np.pi)
    assert np.allclose(spec.natural_frequency(), expected)
    assert (spec.natural_frequency() >= rotor_only.natural_frequency()).all()
    assert spec.natural_frequency()[1] > 15.0 * rotor_only.natural_frequency()[1]
    assert spec.required_time_step() < rotor_only.required_time_step()


def test_damping_is_derived_from_the_damping_ratio():
    from elastic_sim.torque_runners import effective_inertia

    stiffness, zeta = np.array([2.0e4, 5.0e3]), np.array([0.05, 0.2])
    rotor, nominal, floor = np.array([0.1, 0.1]), np.array([2.0, 1.0e-2]), np.array([0.5, 3.0e-4])
    spec = TransmissionSpec.from_damping_ratio(stiffness, zeta, rotor, nominal, floor)
    recovered = spec.damping / (2.0 * np.sqrt(stiffness * effective_inertia(rotor, nominal)))
    assert np.allclose(recovered, zeta)
    assert np.allclose(spec.damping_ratio, zeta)
    assert np.allclose(spec.link_inertia_floor, floor)


def test_link_inertia_envelope_finds_the_light_wrist(asset):
    from elastic_sim.torque_runners import link_inertia_envelope

    median, floor = link_inertia_envelope(asset, n_samples=64)
    assert median.shape == floor.shape == (7,)
    assert (floor > 0.0).all() and (floor <= median + 1e-12).all()
    assert floor[-1] < 1e-2 < median[1], "A7 carries only the flange, A2 the whole arm"


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
# Payload (R3_01)
# ---------------------------------------------------------------------------

def test_empty_payload_is_the_identity(asset):
    from elastic_sim.payload import Payload, payload_asset

    with payload_asset(asset, Payload()) as same:
        assert same is asset
    with payload_asset(asset, None) as same:
        assert same is asset


def test_payload_attaches_to_the_last_active_joint_child(asset):
    """Not the URDF's last link, and not a trailing fixed ee frame."""
    from elastic_sim.payload import Payload, _last_link_name, payload_asset

    expected_parent = asset.resolve_active_joints()[-1].child
    assert _last_link_name(asset) == expected_parent
    with payload_asset(asset, Payload(mass=1.0, offset=(0.0, 0.0, 0.05), size=0.1)) as asset_p:
        text = asset_p.urdf_path.read_text(encoding="utf-8")
        index = text.index("identification_payload_joint")
        assert f'<parent link="{expected_parent}"/>' in text[index:index + 200]


def test_payload_stream_does_not_perturb_the_robot_stream():
    """Adding payloads must not renumber or move the sampled robots."""
    from elastic_sim.dataset import PayloadSampling, TransmissionSampling, sample_payloads, sample_robots

    sampling = TransmissionSampling(robots=6, stiffness=((1.5e4, 3.5e4),) * 7)
    without = sample_robots(sampling, 20260917)
    _ = sample_payloads(PayloadSampling(enabled=True, mass=(0.0, 6.0), size=(0.05, 0.25)), 20260917, 6)
    with_payloads = sample_robots(sampling, 20260917)
    assert all(np.array_equal(a.stiffness, b.stiffness) for a, b in zip(without, with_payloads))


@pytest.mark.slow
def test_payload_reaches_pinocchio_and_mujoco_identically(asset):
    pin_mod = pytest.importorskip("pinocchio")
    mujoco = pytest.importorskip("mujoco")
    from elastic_sim.generic_mujoco_runner import _build_model
    from elastic_sim.payload import Payload, payload_asset
    from elastic_sim.torque_runners import neutralize_mujoco_passive

    payload = Payload(mass=5.0, offset=(0.0, 0.0, 0.10), size=0.15)
    with payload_asset(asset, payload) as asset_p:
        pin, pin_model, pin_data = idn.build_model(asset_p)
        model, _ = _build_model(asset_p, mujoco, 0.002)
        # Every runner elsewhere in the codebase zeroes MuJoCo's
        # URDF-imported passive damping/frictionloss so friction is only
        # ever applied through the explicit ``FrictionModel``; without it,
        # ``mj_inverse`` folds in a native -viscous*dq term that bare
        # ``idn.inverse_dynamics`` (no ``friction=``) never sees, producing a
        # spurious few-Nm "disagreement" that has nothing to do with the
        # payload (reproduces identically with no payload at all).
        neutralize_mujoco_passive(model)
        rng = np.random.default_rng(0)
        n = pin_model.nq
        for _ in range(5):
            q = rng.uniform(-0.5, 0.5, size=n)
            dq = rng.uniform(-0.5, 0.5, size=n)
            ddq = rng.uniform(-0.5, 0.5, size=n)
            pin_tau = idn.inverse_dynamics(pin, pin_model, pin_data, q, dq, ddq)
            data = mujoco.MjData(model)
            data.qpos[:n] = q
            data.qvel[:n] = dq
            data.qacc[:n] = ddq
            mujoco.mj_inverse(model, data)
            assert np.abs(pin_tau - data.qfrc_inverse[:n]).max() < 1e-6


@pytest.mark.slow
def test_feedback_stays_small_with_a_payload(asset, short_trajectory):
    """A payload that reached the simulator but not Pinocchio shows up here."""
    pytest.importorskip("mujoco")
    from elastic_sim.payload import Payload, payload_asset

    friction = idn.FrictionModel.from_asset(asset)
    payload = Payload(mass=5.0, offset=(0.0, 0.0, 0.10), size=0.15)
    with payload_asset(asset, payload) as asset_p:
        controller = ComputedTorqueController(asset_p, short_trajectory, friction=friction, natural_frequency=25.0)
        result = run_mujoco_torque(asset_p, short_trajectory, controller, time_step=5e-4, friction=friction)
    ratio = np.mean(np.abs(result["tau_feedback"])) / np.mean(np.abs(result["tau_feedforward"]))
    assert ratio < 0.05


@pytest.mark.slow
def test_payload_gives_the_distal_channels_signal(asset, short_trajectory):
    """Payload-free: ft4, ft5, ft6 RMS ~ [0.2, 0.2, 0.0] Nm (R3_00 F2).
    With a payload fitted, all three must carry real signal, and peak |tau|
    must stay under the effort limit (R3_10 Sec 3.3)."""
    pytest.importorskip("mujoco")
    from elastic_sim.dataset import Tier
    from elastic_sim.payload import Payload, payload_asset

    friction = idn.FrictionModel.from_asset(asset)
    n = len(asset.joint_names)
    transmission = Tier("e00", stiffness=(2.4e4,) * n, damping_ratio=(0.1,) * n).transmission(n)

    def _rollout(payload):
        with payload_asset(asset, payload) as asset_p:
            controller = SeaMotorController(asset_p, short_trajectory, transmission, friction=friction,
                                            natural_frequency=25.0)
            return run_mujoco_elastic_torque(asset_p, short_trajectory, controller, transmission,
                                             time_step=5e-5, friction=friction, check_step=False)

    bare = _rollout(Payload())
    ft_bare = np.sqrt(np.mean(np.asarray(bare["tau_link"]) ** 2, axis=0))

    # A purely axial offset (0, 0, z) lies on the last joint's own rotation
    # axis (and, at some poses, A5's too -- A5 and A7 are collinear on this
    # arm at A6 = 0): a mass on a joint's own axis has no gravity moment
    # about that axis, so it cannot show up on that channel no matter how
    # heavy it is. That is a property of the offset direction, not of the
    # payload/simulation code (R3_12 Sec 2.4) -- use a radial offset so the
    # payload actually loads every distal joint.
    payload = Payload(mass=5.0, offset=(0.12, 0.10, 0.10), size=0.15)
    loaded = _rollout(payload)
    ft_loaded = np.sqrt(np.mean(np.asarray(loaded["tau_link"]) ** 2, axis=0))
    assert (ft_loaded[4:7] > 0.5).all(), f"distal channels still weak: {ft_loaded[4:7]}"

    effort_limits = np.asarray([j.effort or np.inf for j in asset.resolve_active_joints()])
    peak_ratio = np.max(np.abs(np.asarray(loaded["tau_motor"])) / effort_limits[None, :])
    assert peak_ratio < 0.8, f"peak |tau|/effort = {peak_ratio:.2f}"


@pytest.mark.slow
def test_payload_relaxes_rather_than_tightens_the_integration_step(asset):
    """Raising J_link at the wrist lowers the A7 mode; assert, do not assume."""
    pytest.importorskip("mujoco")
    from elastic_sim.dataset import DatasetConfig, elastic_time_step
    from elastic_sim.payload import Payload, payload_asset
    from elastic_sim.torque_runners import TransmissionSpec, link_inertia_envelope

    config = DatasetConfig()
    n = len(asset.joint_names)
    bare_envelope = link_inertia_envelope(asset, n_samples=64)
    bare_transmission = TransmissionSpec.from_damping_ratio(
        np.full(n, 2.4e4), np.full(n, 0.1), np.full(n, 1.0), *bare_envelope
    )
    bare_step = elastic_time_step(bare_transmission, config)

    payload = Payload(mass=5.0, offset=(0.0, 0.0, 0.10), size=0.15)
    with payload_asset(asset, payload) as asset_p:
        loaded_envelope = link_inertia_envelope(asset_p, n_samples=64)
    loaded_transmission = TransmissionSpec.from_damping_ratio(
        np.full(n, 2.4e4), np.full(n, 0.1), np.full(n, 1.0), *loaded_envelope
    )
    loaded_step = elastic_time_step(loaded_transmission, config)
    assert loaded_step >= bare_step, f"payload should relax (not tighten) the step: bare={bare_step:.2e} loaded={loaded_step:.2e}"


@pytest.mark.slow
def test_a7_output_resampling_does_not_alias(asset):
    """A conclusive version of the check R3_13 Sec 2.7 left inconclusive.

    Comparing peak *locations* (what R3_13 did) cannot detect aliasing.
    R3_14 Sec 1.5's actual measurement: with a light payload (A7's mode
    stays at 200-700 Hz -- the residual-risk case a heavier payload, whose
    mode drops toward or into the probe band, does not have), build the
    2 ms output two ways from the same full-rate elastic rollout -- today's
    point-sampling (``np.interp``, no anti-aliasing) and a version low-pass
    filtered at 200 Hz (just under the 250 Hz output Nyquist) before
    resampling -- and take the RMS difference relative to the filtered
    signal's own RMS. That difference *is* the aliased content. Below ~1%
    closes the item without touching ``rollout_frame``.
    """
    pytest.importorskip("mujoco")
    from scipy import signal as sps

    from elastic_sim.dataset import Tier
    from elastic_sim.excitation import log_spaced_probe_harmonics
    from elastic_sim.payload import Payload, payload_asset
    from elastic_sim.torque_runners import link_inertia_envelope

    probe = log_spaced_probe_harmonics(8.0, 190.0, 40, 0.1)
    # 1 kg close to the flange: light enough that A7's closed-loop mode
    # stays well above the 250 Hz output Nyquist (checked: ~300 Hz here),
    # the case R3_14 Sec 1.5 flags as the actual residual risk.
    payload = Payload(mass=1.0, offset=(0.02, 0.02, 0.06), size=0.05)
    with payload_asset(asset, payload) as asset_p:
        link_inertia = link_inertia_envelope(asset_p, n_samples=200)
        transmission = Tier("e00", stiffness=(5.5e3,) * 7, damping_ratio=(0.1,) * 7).transmission(
            len(asset.joint_names), link_inertia
        )
        mode = transmission.closed_loop_mode_frequency(
            link_inertia[0], natural_frequency=25.0, damping_ratio=1.0
        )[6]
        assert mode > 250.0, f"fixture problem: A7's predicted mode {mode:.0f} Hz is not above the output Nyquist"
        friction = idn.FrictionModel.from_asset(asset)
        config = exc.FourierExcitationConfig(
            n_harmonics=5, base_frequency=0.1, time_step=0.002, max_acceleration=4.0,
            probe_harmonics=probe, probe_acceleration_fraction=0.2,
        )
        trajectory = exc.optimize_excitation(asset_p, config, seed=3, n_candidates=6)
        controller = SeaMotorController(asset_p, trajectory, transmission, friction=friction, natural_frequency=25.0)
        result = run_mujoco_elastic_torque(asset_p, trajectory, controller, transmission, time_step=5e-5,
                                           friction=friction, check_step=False)

    time_full = np.asarray(result["time"])
    fs_full = 1.0 / (time_full[1] - time_full[0])
    target_step = 0.002
    grid = np.arange(time_full[0], time_full[-1] + 0.5 * target_step, target_step)
    sos = sps.butter(8, 200.0 / (fs_full / 2.0), btype="low", output="sos")

    channels = {
        "defl6": np.asarray(result["q_motor"])[:, 6] - np.asarray(result["q_link"])[:, 6],
        "ft6": np.asarray(result["tau_link"])[:, 6],
    }
    for name, signal in channels.items():
        point_sampled = np.interp(grid, time_full, signal)
        anti_aliased = np.interp(grid, time_full, sps.sosfiltfilt(sos, signal))
        rms_reference = float(np.sqrt(np.mean(anti_aliased ** 2)))
        rms_diff = float(np.sqrt(np.mean((point_sampled - anti_aliased) ** 2)))
        ratio = rms_diff / max(rms_reference, 1e-12)
        assert ratio < 0.01, f"{name}: aliased content is {ratio:.2%} of RMS (threshold 1%)"


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


# ---------------------------------------------------------------------------
# Splits and consumer contract (R3_05)
# ---------------------------------------------------------------------------

def test_rollout_frame_carries_the_split_label(asset, short_trajectory, rigid_rollout):
    from elastic_sim.dataset import Tier, rollout_frame

    friction = idn.FrictionModel.from_asset(asset)
    frame = rollout_frame(
        asset, short_trajectory, rigid_rollout, bag="bag0", tier=Tier("rigid"),
        backend="mujoco", friction=friction, resample_step=0.002, split="test",
    )
    assert (frame["split"] == "test").all()


def test_assign_splits_holdout_robots_reserves_the_extreme_strata():
    from elastic_sim.dataset import SplitPolicy, Tier, assign_splits

    rng = np.random.default_rng(0)
    tiers = (Tier("rigid"),) + tuple(
        Tier(f"e{index:02d}", stiffness=(float(value),) * 7)
        for index, value in enumerate(rng.uniform(1.0e4, 3.0e4, size=10))
    )
    policy = SplitPolicy(mode="holdout_robots", test_robots=4, val_robots=2)
    labels = assign_splits(tiers, policy)
    assert labels["rigid"] == "train"
    test = {name for name, label in labels.items() if label == "test"}
    val = {name for name, label in labels.items() if label == "val"}
    train = {name for name, label in labels.items() if label == "train"}
    assert len(test) == 4 and len(val) == 2
    assert not (test & val) and not (test & train) and not (val & train)
    ranked = sorted((tier for tier in tiers if not tier.is_rigid), key=lambda tier: tier.stiffness[0])
    softest, stiffest = {ranked[0].name, ranked[1].name}, {ranked[-1].name, ranked[-2].name}
    assert softest <= test | val
    assert stiffest <= test | val


def test_split_policy_rejects_an_unknown_mode():
    from elastic_sim.dataset import SplitPolicy

    with pytest.raises(ValueError, match="split mode"):
        SplitPolicy(mode="bogus")


def test_split_policy_rejects_more_holdout_robots_than_exist():
    from elastic_sim.dataset import SplitPolicy, Tier, assign_splits

    tiers = (Tier("rigid"), Tier("e00", stiffness=(1.0,)))
    with pytest.raises(ValueError, match="exceeds"):
        assign_splits(tiers, SplitPolicy(mode="holdout_robots", test_robots=4))


def test_split_policy_rejects_holding_out_every_elastic_robot():
    """Regression guard for R3_10 Sec 3.1: --robots 6 with the shipped
    test_robots=4/val_robots=2 used to yield zero training robots silently."""
    from elastic_sim.dataset import SplitPolicy, Tier, assign_splits

    tiers = (Tier("rigid"),) + tuple(Tier(f"e{i:02d}", stiffness=(1.0e4 * (i + 1),)) for i in range(6))
    with pytest.raises(ValueError, match="zero elastic robots"):
        assign_splits(tiers, SplitPolicy(mode="holdout_robots", test_robots=4, val_robots=2))


def test_split_policy_holdout_trajectories_is_not_a_silent_no_op():
    from elastic_sim.dataset import SplitPolicy, Tier, assign_splits

    tiers = (Tier("rigid"),)
    with pytest.raises(NotImplementedError):
        assign_splits(tiers, SplitPolicy(mode="holdout_trajectories"))


@pytest.fixture
def small_config(asset):
    from dataclasses import replace

    from elastic_sim.dataset import (
        DEFAULT_CONFIG, SplitPolicy, TransmissionSampling, build_tiers, load_config,
    )

    config = load_config(os.path.join(_REPO, DEFAULT_CONFIG))
    transmission = replace(config.transmission, robots=6)
    tiers = build_tiers(True, transmission, config.seed)
    return replace(
        config,
        backends=("mujoco",),
        transmission=transmission,
        tiers=tiers,
        n_trajectories=1,
        trajectories_per_robot=False,
        n_friction_samples=1,
        excitation=exc.FourierExcitationConfig(
            n_harmonics=3, base_frequency=0.5, time_step=0.002, max_acceleration=2.0
        ),
        candidates=4,
        split=SplitPolicy(mode="holdout_robots", test_robots=2, val_robots=0),
    )


def test_payload_effort_stays_under_the_warning_threshold(small_config, asset):
    """The shipped payload range (0-6 kg) must not push any bag's peak
    |tau|/effort past the 0.8 warning threshold `generate()` prints for
    (R3_12 Sec 2.7): recorded and warned is not the same as verified."""
    pytest.importorskip("mujoco")
    from elastic_sim.dataset import generate

    _, manifest, _ = generate(small_config, asset, verbose=False)
    offenders = {r["bag"]: r["peak_torque_ratio"] for r in manifest["records"] if r["peak_torque_ratio"] >= 0.8}
    assert not offenders, f"bags over the 0.8 effort-ratio threshold: {offenders}"


def test_holdout_robots_do_not_leak_between_splits(small_config, asset):
    pytest.importorskip("mujoco")
    from elastic_sim.dataset import generate

    frame, manifest, _ = generate(small_config, asset, verbose=False)
    train = set(frame.loc[frame["split"] == "train", "tier"])
    test = set(frame.loc[frame["split"] == "test", "tier"])
    assert train and test and not (train & test)
    assert set(manifest["split"]["test"]) == test


def test_rotor_inertia_varies_across_robots_without_shrinking_the_step(small_config, asset):
    """R3_08 Sec A3: rotor_inertia__<joint> must differ across sampled robots
    (before this round every row carried the same fixed 0.1 kg m^2), and
    randomizing it must not shrink the integration step by more than 10% on
    average relative to that fixed-0.1 baseline -- A7's mode (set by the
    wrist's own light link inertia, not the rotor) is what actually bounds
    the step, so the rotor-inertia change should be close to step-neutral."""
    pytest.importorskip("mujoco")
    from dataclasses import replace

    from elastic_sim.dataset import Tier, elastic_time_step
    from elastic_sim.torque_runners import link_inertia_envelope

    link_inertia = link_inertia_envelope(asset, n_samples=64)
    n = len(asset.joint_names)
    elastic_tiers = [tier for tier in small_config.tiers if not tier.is_rigid]

    sampled_steps, sampled_rotors = [], []
    for tier in elastic_tiers:
        spec = tier.transmission(n, link_inertia)
        sampled_rotors.append(spec.rotor_inertia)
        sampled_steps.append(elastic_time_step(spec, small_config))
    sampled_rotors = np.asarray(sampled_rotors)
    assert np.std(sampled_rotors, axis=0).max() > 0.0, "rotor inertia must be sampled, not fixed"

    fixed_rotor_steps = [
        elastic_time_step(replace(tier, rotor_inertia=(0.1,)).transmission(n, link_inertia), small_config)
        for tier in elastic_tiers
    ]
    assert np.mean(sampled_steps) > 0.9 * np.mean(fixed_rotor_steps)


def test_manifest_reports_sampling_coverage(small_config, asset):
    """R3_08 Sec A4: the coverage report must be present in the manifest and
    name a valid sampling method."""
    pytest.importorskip("mujoco")
    from elastic_sim.dataset import generate

    _, manifest, _ = generate(small_config, asset, verbose=False)
    coverage = manifest["sampling_coverage"]["stiffness"]
    assert len(coverage["coverage_fraction"]) == len(asset.joint_names)
    assert coverage["method"] in ("iid", "stratified", "sobol")


def test_every_bag_has_a_uniform_time_step(small_config, asset):
    pytest.importorskip("mujoco")
    from elastic_sim.dataset import generate

    frame, _, _ = generate(small_config, asset, verbose=False)
    for bag, group in frame.groupby("bag"):
        step = np.diff(group["t"].to_numpy())
        assert np.allclose(step, small_config.sample_time_step, atol=1e-12), bag


def test_contract_sidecar_is_written(tmp_path, small_config, asset):
    pytest.importorskip("mujoco")
    import json

    from elastic_sim.dataset import generate, write_dataset

    frame, manifest, comparison = generate(small_config, asset, verbose=False)
    n_dof = len(asset.joint_names)
    csv_path, *_ = write_dataset(frame, manifest, tmp_path / "d.csv", comparison)
    contract = json.loads(csv_path.with_suffix(".contract.json").read_text())
    assert contract["n_dof"] == n_dof
    assert contract["target_columns"] == [f"ft0..ft{n_dof - 1}"]
    assert contract["split"]["mode"] == "holdout_robots"


# ---------------------------------------------------------------------------
# Performance and storage (R3_06)
# ---------------------------------------------------------------------------

def test_parquet_round_trips_a_synthetic_frame(tmp_path):
    """No simulator needed: write_dataset's format dispatch is pure pandas."""
    from elastic_sim.dataset import write_dataset

    n = 20
    frame = pd.DataFrame({
        "t": np.arange(n) * 0.002, "bag": ["b0"] * n,
        "q0": np.random.default_rng(0).normal(size=n), "ft0": np.random.default_rng(1).normal(size=n),
        "tier": ["e00"] * n, "backend": ["mujoco"] * n, "split": ["train"] * n,
    })
    manifest = {"n_dof": 1, "split": {"mode": "contiguous"}}
    path, *_ = write_dataset(frame, manifest, tmp_path / "d.parquet")
    roundtrip = pd.read_parquet(path)
    assert len(roundtrip) == len(frame)
    assert roundtrip["q0"].dtype == np.float32
    assert roundtrip["t"].dtype == np.float64, "t must stay float64 (R3_10 Sec 3.5): SG filter/step checks depend on it"


def test_metadata_columns_sidecar_moves_constants_off_the_main_file(tmp_path):
    from elastic_sim.dataset import write_dataset

    n = 20
    frame = pd.DataFrame({
        "t": np.arange(n) * 0.002, "bag": ["b0"] * 10 + ["b1"] * 10,
        "q0": np.random.default_rng(0).normal(size=n), "tier": ["e00"] * 10 + ["e01"] * 10,
        "backend": ["mujoco"] * n, "split": ["train"] * n,
        "stiffness__A1": [2.0e4] * 10 + [2.5e4] * 10, "payload_mass": [1.0] * 10 + [2.0] * 10,
    })
    manifest = {"n_dof": 1, "split": {"mode": "contiguous"}}
    inline_path, *_ = write_dataset(frame, manifest, tmp_path / "inline.csv")
    sidecar_path, *_ = write_dataset(frame, manifest, tmp_path / "sidecar.csv", metadata_columns="sidecar")
    assert "stiffness__A1" in pd.read_csv(inline_path).columns
    assert "stiffness__A1" not in pd.read_csv(sidecar_path).columns
    sidecar = json.loads(sidecar_path.with_suffix(".bag_metadata.json").read_text())
    assert sidecar["b0"]["stiffness__A1"] == 2.0e4
    assert sidecar["b1"]["payload_mass"] == 2.0


def test_write_dataset_rejects_an_unsupported_suffix(tmp_path):
    from elastic_sim.dataset import write_dataset

    frame = pd.DataFrame({"t": [0.0], "bag": ["b0"]})
    with pytest.raises(ValueError, match="suffix"):
        write_dataset(frame, {"n_dof": 1}, tmp_path / "d.json")


@pytest.mark.slow
def test_index_array_hoisting_is_bit_identical(asset, short_trajectory):
    """Refactor only: every recorded value must be unchanged (R3_06 F1)."""
    pytest.importorskip("mujoco")
    friction = idn.FrictionModel.from_asset(asset)
    controller = ComputedTorqueController(asset, short_trajectory, friction=friction, natural_frequency=25.0)
    result = run_mujoco_torque(asset, short_trajectory, controller, time_step=5e-4, friction=friction)
    # A regression here would show up as a changed tracking/feedback ratio on
    # the existing rigid_rollout-based tests; this just locks in the shape
    # and finiteness of the hoisted-index path directly.
    assert np.isfinite(result["q_link"]).all()
    assert np.asarray(result["q_link"]).shape[1] == len(asset.joint_names)


@pytest.mark.slow
def test_control_decimation_stays_stable_with_a_bounded_residual(asset, model, short_trajectory):
    """Holding the whole solver torque (command AND friction subtraction) constant
    across a decimation window keeps the rollout numerically stable, at the cost
    of a small, decimation-dependent residual against Pinocchio inverse dynamics.

    ``ComputedTorqueController.total`` already embeds friction compensation at
    the (q, dq) it was evaluated at; refreshing only the friction subtraction
    on later, un-decimated steps (tried and reverted -- see
    ``run_mujoco_torque``'s comment at the decimation gate) uncancels that
    embedded term into a disturbance with the friction model's full slope
    while the command that would counter it stays frozen, which measurably
    diverges (qacc into the 1e11 range) on this asset's low-inertia wrist
    joint at decimation as low as 2.
    """
    pytest.importorskip("mujoco")
    pin, pin_model, pin_data = model
    friction = idn.FrictionModel.from_asset(asset)
    for decimation, bound in ((1, 1e-5), (4, 1e-5), (8, 0.02)):
        controller = ComputedTorqueController(asset, short_trajectory, friction=friction, natural_frequency=25.0)
        result = run_mujoco_torque(asset, short_trajectory, controller, time_step=5e-4, friction=friction,
                                   control_decimation=decimation)
        q, dq, ddq = result["q_link"], result["dq_link"], result["ddq_link"]
        assert np.max(np.abs(ddq)) < 100.0, f"decimation={decimation} rollout is unstable"
        stride = slice(None, None, 20)
        predicted = np.asarray([
            idn.inverse_dynamics(pin, pin_model, pin_data, q[i], dq[i], ddq[i], friction=friction)
            for i in range(0, len(q), 20)
        ])
        residual = result["tau_motor"][stride] - predicted
        assert np.sqrt(np.mean(residual**2)) < bound, f"decimation={decimation}"


@pytest.mark.slow
def test_parallel_generation_is_bit_identical(small_config, asset):
    """--jobs 4 == --jobs 1 (bags are independent; RNG streams key on index)."""
    pytest.importorskip("mujoco")
    from dataclasses import replace

    from elastic_sim.dataset import generate

    sequential, _, _ = generate(small_config, asset, verbose=False, jobs=1)
    parallel, _, _ = generate(replace(small_config), asset, verbose=False, jobs=2)
    pd.testing.assert_frame_equal(
        sequential.reset_index(drop=True), parallel.reset_index(drop=True),
    )


def test_default_config_loads_and_is_shared_by_both_scripts():
    from elastic_sim.dataset import DEFAULT_CONFIG, load_config

    config = load_config(os.path.join(_REPO, DEFAULT_CONFIG))
    assert config.asset == ASSET
    assert set(config.backends) <= {"mujoco", "newton"}
    assert config.excitation.time_step == config.sample_time_step
    assert config.output.endswith(".csv")
    assert config.visualize is False
    assert len(config.transmission.stiffness_nominal) == 7, "one stiffness nominal per iiwa joint"
    assert [tier.is_rigid for tier in config.tiers].count(True) == int(config.rigid_reference)
    assert sum(not tier.is_rigid for tier in config.tiers) == config.transmission.robots


def test_generate_dataset_cli_does_not_drop_probe_or_jitter():
    """Regression guard for R3_10 Sec 2.2: a from-scratch FourierExcitationConfig(...)
    reconstruction in the CLI silently dropped centre_jitter/probe_harmonics/
    probe_acceleration_fraction back to their dataclass defaults on every run,
    regardless of what the YAML said."""
    import importlib

    from elastic_sim.dataset import DEFAULT_CONFIG, load_config

    sys.path.insert(0, os.path.join(_REPO, "scripts"))
    gid = importlib.import_module("generate_identification_dataset")
    parser = gid.build_parser()
    config = gid.resolve_config(parser.parse_args([]), parser)
    default_excitation = load_config(os.path.join(_REPO, DEFAULT_CONFIG)).excitation
    assert config.excitation.probe_harmonics == default_excitation.probe_harmonics
    assert config.excitation.probe_acceleration_fraction == default_excitation.probe_acceleration_fraction
    assert config.excitation.centre_jitter == default_excitation.centre_jitter


def test_generate_dataset_cli_errors_on_max_acceleration_with_regime_enabled():
    """--max-acceleration would otherwise be silently overwritten per-trajectory
    by excitation.regime once regime.enabled is true (the shipped default)."""
    import importlib

    sys.path.insert(0, os.path.join(_REPO, "scripts"))
    gid = importlib.import_module("generate_identification_dataset")
    parser = gid.build_parser()
    args = parser.parse_args(["--max-acceleration", "3.0"])
    with pytest.raises(SystemExit):
        gid.resolve_config(args, parser)


def test_config_rejects_unknown_keys(tmp_path):
    from elastic_sim.dataset import load_config

    path = tmp_path / "bad.yaml"
    path.write_text("asset: x\nnonsense: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys"):
        load_config(path)


def test_provenance_letters_are_validated(tmp_path):
    from elastic_sim.dataset import load_config

    path = tmp_path / "bad_provenance.yaml"
    path.write_text(
        "asset: x\nrigid_reference: false\n"
        "transmission:\n  robots: 1\n  stiffness_nominal: [1,2,3,4,5,6,7]\n"
        "  stiffness_provenance: [P, X, C, C, C, E, E]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="provenance"):
        load_config(path)


# Provenance-doc section heading, and a joint-name alias, per shipped
# config's asset -- the doc labels a joint however its own narrative reads
# best (the iiwa's `A1..A7`, the UR10's URDF names with the `_joint` suffix
# dropped), so a rename lives here rather than in the doc (R4_04 Sec 2).
_PROVENANCE_SECTIONS: dict[str, tuple[str, dict[str, str]]] = {
    "kuka_lbr_iiwa_14_r820_table": (
        "kuka_lbr_iiwa_14_r820", {f"iiwa_A{i}": f"A{i}" for i in range(1, 8)},
    ),
    "ur10_table": (
        "ur10",
        {
            "shoulder_pan_joint": "shoulder_pan", "shoulder_lift_joint": "shoulder_lift",
            "elbow_joint": "elbow", "wrist_1_joint": "wrist_1",
            "wrist_2_joint": "wrist_2", "wrist_3_joint": "wrist_3",
        },
    ),
    # Round 5: the FMRR gantry's own section.  Joint names are already the
    # labels, so the alias map is empty (R5_02 Sec 3).
    "fmrr_tecnobody": ("fmrr_tecnobody", {}),
}


def test_every_shipped_config_has_provenance_rows_for_its_own_joints():
    """Replacement for the old iiwa-only test: for every shipped config, a
    provenance row exists for each active joint and each of stiffness/
    rotor_inertia inside that asset's own `## <source>` section, and its
    class letter matches the config's `*_provenance` entry (R4_04 Sec 2)."""
    from elastic_sim.dataset import load_config

    doc = (Path(_REPO) / "docs" / "PARAMETER_PROVENANCE.md").read_text(encoding="utf-8")
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in doc.splitlines():
        if line.startswith("## "):
            current = line[3:].strip()
            sections.setdefault(current, [])
        elif current is not None and line.startswith("|"):
            sections[current].append(line)

    config_dir = Path(_REPO) / "config" / "identification"
    configs = sorted(config_dir.glob("*.yaml"))
    assert configs, f"no shipped configs found under {config_dir}"
    for config_path in configs:
        config = load_config(config_path)
        assert config.asset in _PROVENANCE_SECTIONS, (
            f"{config_path}: no provenance-section mapping for asset {config.asset!r}; "
            "add one to _PROVENANCE_SECTIONS in this test"
        )
        section_name, alias = _PROVENANCE_SECTIONS[config.asset]
        rows = sections.get(section_name, [])
        assert rows, f"{config_path}: docs/PARAMETER_PROVENANCE.md has no '## {section_name}' section"
        asset = AssetRegistry.for_repository(_REPO).load(config.asset)
        provenance_by_param = {
            "stiffness": config.transmission.stiffness_provenance,
            "rotor_inertia": config.transmission.rotor_inertia_provenance,
        }
        for param, provenance in provenance_by_param.items():
            if not provenance:
                continue
            for joint_name, expected_class in zip(asset.joint_names, provenance):
                label = alias.get(joint_name, joint_name)
                matches = [
                    row for row in rows
                    if row.split("|")[1].strip() == label and row.split("|")[2].strip() == param
                ]
                assert matches, (
                    f"{config_path}: no provenance row for joint {label!r} parameter {param!r} "
                    f"in section '## {section_name}'"
                )
                actual_class = matches[0].split("|")[5].strip()
                assert actual_class == expected_class, (
                    f"{config_path}: provenance row for {label!r} {param!r} has class {actual_class!r}, "
                    f"but the config's {param}_provenance says {expected_class!r}"
                )


def test_config_rejects_the_old_tier_ladder(tmp_path):
    from elastic_sim.dataset import load_config

    path = tmp_path / "old.yaml"
    path.write_text("asset: x\ntiers:\n  rigid: true\n  stiffness: [1.0e4]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="transmission"):
        load_config(path)


def test_config_rejects_control_decimation_without_explicit_opt_in(tmp_path):
    """R3_12 Sec 2.5: no automatic stability guard exists yet for
    control_decimation > 1 (see torque_runners's decimation-gate comment and
    R3_11 Sec 1 for why it can silently diverge), so it must not be one YAML
    edit away."""
    from elastic_sim.dataset import load_config

    path = tmp_path / "decimated.yaml"
    path.write_text(
        "asset: x\nrigid_reference: true\nsimulation:\n  control_decimation: 8\n", encoding="utf-8",
    )
    with pytest.raises(ValueError, match="allow_control_decimation"):
        load_config(path)

    path2 = tmp_path / "decimated_opted_in.yaml"
    path2.write_text(
        "asset: x\nrigid_reference: true\n"
        "simulation:\n  control_decimation: 8\n  allow_control_decimation: true\n",
        encoding="utf-8",
    )
    assert load_config(path2).control_decimation == 8


def test_config_can_select_rigid_only(tmp_path):
    from elastic_sim.dataset import load_config

    path = tmp_path / "rigid.yaml"
    path.write_text("asset: x\nrigid_reference: true\n", encoding="utf-8")
    tiers = load_config(path).tiers
    assert len(tiers) == 1 and tiers[0].is_rigid


def test_config_rejects_an_empty_selection(tmp_path):
    from elastic_sim.dataset import load_config

    path = tmp_path / "none.yaml"
    path.write_text("asset: x\nrigid_reference: false\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no tiers"):
        load_config(path)


def test_config_samples_robots_inside_their_intervals(tmp_path):
    from elastic_sim.dataset import load_config

    path = tmp_path / "t.yaml"
    path.write_text(
        "asset: x\nrigid_reference: false\ndataset:\n  seed: 7\n"
        "transmission:\n  robots: 40\n  stiffness: [[1.0e3, 1.0e5], [2.0e3, 2.0e3]]\n"
        "  damping_ratio: [0.05, 0.2]\n  rotor_inertia: 0.25\n"
        "comparison:\n  q_link_rms: 3.0e-4\n",
        encoding="utf-8",
    )
    config = load_config(path)
    assert [tier.name for tier in config.tiers][:3] == ["e00", "e01", "e02"]
    stiffness = np.asarray([tier.stiffness for tier in config.tiers])
    zeta = np.asarray([tier.damping_ratio for tier in config.tiers])
    assert ((stiffness[:, 0] >= 1.0e3) & (stiffness[:, 0] <= 1.0e5)).all()
    assert np.allclose(stiffness[:, 1], 2.0e3)
    assert ((zeta >= 0.05) & (zeta <= 0.2)).all()
    # Log-uniform: about half the draws fall below the geometric mean 1e4.
    assert 0.25 < np.mean(stiffness[:, 0] < 1.0e4) < 0.75
    spec = config.tiers[0].transmission(2)
    assert np.allclose(spec.rotor_inertia, 0.25) and np.allclose(spec.damping_ratio, zeta[0])
    assert config.comparison.q_link_rms == 3.0e-4
    with pytest.raises(ValueError):
        config.tiers[0].transmission(7)


def test_robot_sampling_is_deterministic_and_prefix_stable():
    from elastic_sim.dataset import TransmissionSampling, sample_robots

    sampling = TransmissionSampling(robots=3, stiffness=((1.0e3, 1.0e5),) * 7)
    more = TransmissionSampling(robots=5, stiffness=((1.0e3, 1.0e5),) * 7)
    assert sample_robots(sampling, 11) == sample_robots(sampling, 11)
    assert sample_robots(more, 11)[:3] == sample_robots(sampling, 11)
    assert sample_robots(sampling, 11) != sample_robots(sampling, 12)


def test_condition_order_interleaves_tiers():
    """The consumer splits train/test as contiguous halves of the file."""
    from elastic_sim.dataset import DatasetConfig, Tier, iter_conditions

    config = DatasetConfig(
        backends=("mujoco",), tiers=(Tier("rigid"), Tier("e00", stiffness=1.0e4)),
        n_trajectories=2, n_friction_samples=1,
    )
    tiers = [tier.name for _, tier, _, _ in iter_conditions(config)]
    halfway = len(tiers) // 2
    assert set(tiers[:halfway]) == set(tiers[halfway:]), "each half must see every tier"


# ---------------------------------------------------------------------------
# Backend comparison
# ---------------------------------------------------------------------------

def _synthetic_bag(bag, backend, tier, rng, *, n=200, dof=2, offset=0.0, elastic=True):
    import pandas as pd

    t = np.arange(n) * 0.002
    q = np.sin(np.outer(t, np.arange(1, dof + 1))) + offset
    deflection = 1e-3 * np.cos(np.outer(t, np.arange(1, dof + 1))) if elastic else 0.0
    frame = {"t": t, "bag": bag, "tier": tier, "backend": backend}
    for j in range(dof):
        frame[f"q{j}"] = q[:, j]
        frame[f"dq{j}"] = np.gradient(q[:, j], t)
        frame[f"q_motor{j}"] = q[:, j] + (deflection[:, j] if elastic else 0.0)
        frame[f"tau{j}"] = 10.0 * q[:, j] + 1.0
        frame[f"ft{j}"] = 9.0 * q[:, j] + 1.0
    return pd.DataFrame(frame)


def test_backend_comparison_pairs_and_passes_identical_bags():
    import pandas as pd
    from elastic_sim.backend_comparison import ComparisonThresholds, compare_backends

    rng = np.random.default_rng(0)
    frame = pd.concat([
        _synthetic_bag("t0_rigid_f0_mujoco", "mujoco", "rigid", rng, elastic=False),
        _synthetic_bag("t0_rigid_f0_newton", "newton", "rigid", rng, elastic=False),
        _synthetic_bag("t0_e00_f0_mujoco", "mujoco", "e00", rng),
        _synthetic_bag("t0_e00_f0_newton", "newton", "e00", rng),
        _synthetic_bag("t1_e00_f0_mujoco", "mujoco", "e00", rng),  # no partner
    ], ignore_index=True)
    records = [{"bag": "t0_rigid_f0_newton", "solver": "SolverFeatherstone"},
               {"bag": "t0_e00_f0_newton", "solver": "SolverMuJoCo"}]
    report = compare_backends(frame, records, ComparisonThresholds()).set_index("pair")
    assert list(report.index) == ["t0_rigid_f0", "t0_e00_f0"]
    assert report["pass"].all()
    assert (report[["q_link_rms", "ft_relative_rms"]] == 0.0).all().all()
    assert not report.loc["t0_rigid_f0", "elastic"] and np.isnan(report.loc["t0_rigid_f0", "deflection_relative_rms"])
    assert bool(report.loc["t0_rigid_f0", "independent"]) is True
    assert bool(report.loc["t0_e00_f0", "independent"]) is False


def test_backend_comparison_flags_a_divergent_backend():
    import pandas as pd
    from elastic_sim.backend_comparison import ComparisonThresholds, compare_backends, format_report, summarize

    rng = np.random.default_rng(0)
    frame = pd.concat([
        _synthetic_bag("t0_e00_f0_mujoco", "mujoco", "e00", rng),
        _synthetic_bag("t0_e00_f0_newton", "newton", "e00", rng, offset=1e-3),
    ], ignore_index=True)
    thresholds = ComparisonThresholds()
    report = compare_backends(frame, (), thresholds)
    row = report.iloc[0]
    assert row["q_link_rms"] == pytest.approx(1e-3)
    assert row["ft_relative_rms"] > 0.0 and not row["pass"]
    assert row["independent"] is None, "solver is unknown without a manifest"
    assert summarize(report, thresholds)["failed"] == ["t0_e00_f0"]
    assert "FAIL" in format_report(report, thresholds)
