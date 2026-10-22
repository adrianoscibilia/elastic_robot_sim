"""T1.7: E-ur-1/E-ur-3 fitting code recovers known fixture parameters."""

from __future__ import annotations

import numpy as np
import pytest

from elastic_sim import identification as idn
from elastic_sim.assets import AssetRegistry

from erd_recording.identification import (
    JOINT_SIZE_MAX_RESIDUAL_INCREASE,
    K_TAU_R2_THRESHOLD,
    IdentificationRefused,
    choose_ur_tau_source,
    controller_rotor_inertia,
    fit_linear_per_joint,
    fit_sweep_friction,
    identify_motor_side,
)


@pytest.fixture(scope="module")
def ur10_asset():
    return AssetRegistry.for_repository().load("ur10_table")


@pytest.fixture(scope="module")
def synthetic_motion(ur10_asset):
    from elastic_sim.excitation import FourierExcitationConfig, optimize_excitation

    pin, model, data = idn.build_model(ur10_asset)
    # A real E-ur-3 fit runs on the actual (well-conditioned) excitation
    # data, not on IID-random joint states: IID sampling leaves some
    # rotor-inertia columns (e.g. the base joint's) collinear with the rigid
    # regressor's own base-parameter combinations for this asset, an
    # identifiability artefact of the sampling, not of the fitting code.
    # Reusing the same excitation generator the real experiment protocol
    # calls for is the faithful fixture.
    exc_config = FourierExcitationConfig(
        n_harmonics=5, base_frequency=0.1, n_periods=1, time_step=0.01,
        max_acceleration=2.0, velocity_fraction=0.8, centre_jitter=1.0,
    )
    # A single periodic trajectory under-excites the base joint's rotor
    # inertia (its mass-matrix column barely varies over one Fourier
    # series' state distribution, so J_m aliases with the rigid regressor):
    # exactly why E-ur-3 fits on several bags at different regimes/centres,
    # not one (RR_01 S2.1 step 2 "on the excitation data"). Concatenate a
    # handful of differently-seeded trajectories, as the real protocol would.
    positions, velocities, accelerations = [], [], []
    for seed in range(6):
        trajectory = optimize_excitation(ur10_asset, exc_config, seed=seed, n_candidates=16)
        positions.append(trajectory.position)
        velocities.append(trajectory.velocity)
        accelerations.append(trajectory.acceleration)
    q = np.concatenate(positions, axis=0)
    dq = np.concatenate(velocities, axis=0)
    ddq = np.concatenate(accelerations, axis=0)
    return pin, model, data, q, dq, ddq


def test_fit_linear_per_joint_recovers_slope_and_intercept():
    rng = np.random.default_rng(1)
    n_dof = 4
    candidate = rng.uniform(-2, 2, size=(300, n_dof))
    true_slope = np.array([1.5, -0.8, 2.0, 0.3])
    true_intercept = np.array([0.1, -0.2, 0.0, 0.05])
    target = candidate * true_slope[None, :] + true_intercept[None, :]
    fit = fit_linear_per_joint(target, candidate)
    assert np.allclose(fit.slope, true_slope, atol=1e-9)
    assert np.allclose(fit.intercept, true_intercept, atol=1e-9)
    assert np.all(fit.r_squared > 0.999999)


def _currents(rng, n, n_dof, *, feedback):
    """Model-side signals plus a measured current that tracks the command
    (the CB3's current loop, 0.008-0.027 A rms on the 2026-10-21 runs)."""
    target_moment = rng.uniform(-10, 10, size=(n, n_dof))
    tracking_error = rng.uniform(-0.01, 0.01, size=(n, n_dof))
    tracking_error_rate = rng.uniform(-0.1, 0.1, size=(n, n_dof))
    k_tau = np.array([0.15, 0.15, 0.10][:n_dof])
    target_current = target_moment / k_tau[None, :]
    joint_control_output = target_current + feedback * tracking_error
    actual_current = joint_control_output + rng.normal(scale=0.01, size=(n, n_dof))
    return dict(target_moment=target_moment, joint_control_output=joint_control_output,
                target_current=target_current, actual_current=actual_current,
                tracking_error=tracking_error, tracking_error_rate=tracking_error_rate), k_tau


def test_choose_ur_tau_source_picks_joint_control_output_with_feedback():
    # K_tau from target_moment ~ target_current (exact, URSim-like); the
    # command differs from the feed-forward far more than the measured
    # current differs from the command (UR_11 rule): joint_control_output.
    signals, k_tau = _currents(np.random.default_rng(2), 500, 3, feedback=200.0)
    result = choose_ur_tau_source(**signals)
    assert result["tau_source"] == "joint_control_output"
    assert result["diagnostics"]["k_tau_source"] == "target_moment_vs_target_current"
    assert np.allclose(result["k_tau"], k_tau, rtol=1e-9)
    assert min(result["diagnostics"]["command_deviation"]["ratio"]) > 3.0


def test_choose_ur_tau_source_picks_actual_current_when_pure_feedforward():
    # joint_control_output == target_current (URSim: identically): the
    # command is pure feed-forward, read off actual_current.
    rng = np.random.default_rng(3)
    signals, _ = _currents(rng, 500, 3, feedback=0.0)
    signals["actual_current"] = signals["target_current"] + rng.normal(scale=0.01, size=(500, 3))
    assert choose_ur_tau_source(**signals)["tau_source"] == "actual_current"
    exact = dict(signals, actual_current=signals["target_current"].copy())  # URSim: all three equal
    assert choose_ur_tau_source(**exact)["tau_source"] == "actual_current"


def test_choose_ur_tau_source_refuses_an_undecided_command():
    # UR_11: mixed over joints (or in between) is refused, never flipped.
    rng = np.random.default_rng(4)
    signals, _ = _currents(rng, 500, 3, feedback=200.0)
    signals["joint_control_output"][:, 2] = signals["target_current"][:, 2]  # joint 2 pure feed-forward
    signals["actual_current"][:, 2] = signals["target_current"][:, 2] + rng.normal(scale=0.01, size=500)
    with pytest.raises(IdentificationRefused, match="tau_source undecided"):
        choose_ur_tau_source(**signals)


def _cb3_like(rng, *, gravity_joints, n_dof=3, n=600, holds=5):
    """target_current = target_moment / K + friction/inertia feed-forward
    (the real CB3, R^2 of the plain fit ~0.01); standstill holds where only
    ``gravity_joints`` carry a moment."""
    k_tau = np.array([14.18, 14.18, 11.56][:n_dof])
    qd = rng.uniform(-0.5, 0.5, size=(n, n_dof))
    qdd = rng.uniform(-2, 2, size=(n, n_dof))
    moment = 0.3 * rng.normal(size=(n, n_dof)) + 0.02 * qdd
    target_current = moment / k_tau + 0.05 * qdd + 1.5 * qd + 1.0 * np.tanh(qd / 0.01)
    hold_moment = np.zeros((holds * 50, n_dof))
    for j in gravity_joints:
        hold_moment[:, j] = np.repeat(np.linspace(-80, 40, holds), 50)
    standstill = dict(standstill_target_moment=hold_moment, standstill_target_current=hold_moment / k_tau)
    feedforward = dict(target_current=target_current, target_moment=moment, target_qd=qd, target_qdd=qdd)
    jco = target_current + 0.3 * rng.normal(size=(n, n_dof))
    signals = dict(target_moment=moment, joint_control_output=jco, target_current=target_current,
                   actual_current=jco + rng.normal(scale=0.01, size=(n, n_dof)),
                   tracking_error=rng.normal(size=(n, n_dof)), tracking_error_rate=rng.normal(size=(n, n_dof)))
    return signals, standstill, feedforward, k_tau


def test_controller_standstill_ratio_and_joint_size_identity():
    # UR_11 steps 2-3: joints 1-2 carry gravity (standstill ratio), joint 0
    # (pan, same drive size as joint 1) takes joint 1's K by the identity.
    signals, standstill, feedforward, k_tau = _cb3_like(np.random.default_rng(5), gravity_joints=(1, 2))
    result = choose_ur_tau_source(**signals, **standstill, feedforward=feedforward,
                                  joint_sizes=("size4", "size4", "size3"), joint_size_source="test table")
    diagnostics = result["diagnostics"]
    assert diagnostics["k_tau_source"] == "controller"
    assert diagnostics["k_tau_source_per_joint"] == ["joint_size_identity:size4", "controller_standstill_ratio",
                                                     "controller_standstill_ratio"]
    assert np.allclose(result["k_tau"], k_tau, rtol=1e-9)
    assert diagnostics["joint_size_source"] == "test table"
    assert result["tau_source"] == "joint_control_output"
    assert diagnostics["fit_target_current"]["r_squared"][1] < K_TAU_R2_THRESHOLD  # step 1 cannot work here


def test_unidentified_k_tau_fails_instead_of_passing():
    # UR_10 S3: K_tau ~0 on joints without gravity passed as `ok`. Now a
    # joint with neither a standstill ratio nor a same-size donor refuses.
    signals, standstill, feedforward, _ = _cb3_like(np.random.default_rng(6), gravity_joints=(1,))
    with pytest.raises(IdentificationRefused, match=r"K_tau unidentified for joints \[2\]") as refused:
        choose_ur_tau_source(**signals, **standstill, feedforward=feedforward,
                             joint_sizes=("size4", "size4", "size3"))
    assert refused.value.diagnostics["k_tau_source_per_joint"][2] is None
    with pytest.raises(IdentificationRefused, match="K_tau unidentified"):  # no size table at all
        choose_ur_tau_source(**signals, **standstill, feedforward=feedforward)


def test_joint_size_identity_rejected_when_the_feedforward_contradicts_it():
    # The pan's own controller feed-forward says K = 5, not the donor's 14.18:
    # the residual with 14.18 rises far above 10 %, so K stays unidentified.
    rng = np.random.default_rng(7)
    signals, standstill, feedforward, k_tau = _cb3_like(rng, gravity_joints=(1, 2))
    feedforward = dict(feedforward)
    moment = 30.0 * rng.normal(size=len(feedforward["target_moment"]))
    feedforward["target_moment"] = feedforward["target_moment"].copy()
    feedforward["target_current"] = feedforward["target_current"].copy()
    feedforward["target_moment"][:, 0] = moment
    feedforward["target_current"][:, 0] = moment / 5.0 + 1.5 * feedforward["target_qd"][:, 0]
    with pytest.raises(IdentificationRefused, match=r"joints \[0\]") as refused:
        choose_ur_tau_source(**signals, **standstill, feedforward=feedforward, joint_sizes=("size4", "size4", "size3"))
    identity = refused.value.diagnostics["joint_size_identity"][0]
    assert not identity["accepted"] and identity["residual_increase"] > JOINT_SIZE_MAX_RESIDUAL_INCREASE
    assert identity["k_tau_free"] == pytest.approx(5.0, rel=1e-6)


def test_constant_target_moment_is_not_a_step_1_fit():
    # A constant target_moment fits with R^2 1 by the ss_tot = 0 convention:
    # that must not become K_tau = slope 0 (the wrist-3 trap of UR_10 S3).
    signals, _ = _currents(np.random.default_rng(8), 300, 3, feedback=200.0)
    signals["target_moment"][:, 2] = 0.0
    signals["target_current"][:, 2] = 0.0
    with pytest.raises(ValueError, match="standstill_target_moment"):
        choose_ur_tau_source(**signals)


def test_choose_ur_tau_source_without_fallback_data_raises():
    rng = np.random.default_rng(5)
    n_dof = 2
    n = 100
    target_moment = rng.uniform(-10, 10, size=(n, n_dof))
    target_current = rng.uniform(-10, 10, size=(n, n_dof))
    tracking_error = rng.uniform(-0.01, 0.01, size=(n, n_dof))
    tracking_error_rate = rng.uniform(-0.1, 0.1, size=(n, n_dof))
    with pytest.raises(ValueError):
        choose_ur_tau_source(
            target_moment=target_moment, joint_control_output=target_current,
            target_current=target_current, actual_current=target_current,
            tracking_error=tracking_error, tracking_error_rate=tracking_error_rate,
        )


def test_controller_rotor_inertia_recovers_the_controllers_j_m(ur10_asset, synthetic_motion):
    # UR_11: target_moment = RNEA(target) + J_m * target_qdd on the real CB3.
    pin, model, data, q, dq, ddq = synthetic_motion
    j_m = np.array([7.14, 7.14, 1.92, 0.61, 0.61, 0.61])
    rnea = np.asarray([pin.rnea(model, data, a, b, c).copy() for a, b, c in zip(q, dq, ddq)])
    result = controller_rotor_inertia(ur10_asset, q, dq, ddq, rnea + j_m[None, :] * ddq)
    assert np.allclose(result["j_m"], j_m, rtol=1e-6)
    assert all(result["identified"])
    flat = controller_rotor_inertia(ur10_asset, q, dq, ddq, rnea + 1e-3 * np.random.default_rng(0).normal(size=q.shape))
    assert not any(flat["identified"])  # no J_m * qdd in the moment: nothing to take


def test_fit_sweep_friction_is_viscous_plus_coulomb_without_stribeck():
    # UR_11: six plateaus above any Stribeck velocity; the former grid search
    # traded a "Stribeck" term against the viscous slope (pan: 70.9 / 1.05).
    speeds = np.array([0.024, 0.05, 0.099, 0.172, 0.245, 0.343])
    torques = 39.0 * speeds + 14.5 * np.tanh(speeds / 0.01)  # friction_torque's own law
    fit = fit_sweep_friction(speeds, torques)
    assert fit["viscous"] == pytest.approx(39.0, rel=1e-3)
    assert fit["coulomb"] == pytest.approx(14.5, rel=1e-3)
    assert fit["stribeck"] == 0.0 and fit["stribeck_identifiable"] is False
    assert fit["stribeck_velocity"] == pytest.approx(0.024)
    assert fit["residual_rms"] < 1e-9


def test_identify_motor_side_recovers_j_m_and_friction(ur10_asset, synthetic_motion):
    pin, model, data, q, dq, ddq = synthetic_motion
    n_dof = q.shape[1]
    true_j_m = np.array([4.0, 4.0, 1.7, 0.5, 0.5, 0.5])
    true_viscous = np.array([0.5, 0.5, 0.3, 0.1, 0.1, 0.1])
    true_coulomb = np.array([2.0, 2.0, 1.0, 0.4, 0.4, 0.4])
    epsilon = 1.0e-2
    k_tau = np.array([0.15, 0.15, 0.15, 0.08, 0.08, 0.08])

    rigid = np.asarray([idn.inverse_dynamics(pin, model, data, q[i], dq[i], ddq[i]) for i in range(len(q))])
    friction = true_viscous[None, :] * dq + true_coulomb[None, :] * np.tanh(dq / epsilon)
    tau_m = rigid + true_j_m[None, :] * ddq + friction
    i_act = tau_m / k_tau[None, :]

    holdout_mask = np.zeros(len(q), dtype=bool)
    holdout_mask[:60] = True  # "first identify trajectory" held out
    holdout_mask[-60:] = True  # "last identify trajectory" held out

    fit = identify_motor_side(ur10_asset, q, dq, ddq, i_act, k_tau, epsilon=epsilon, holdout_mask=holdout_mask)

    # Joint 0 (shoulder_pan, the vertical-axis base joint) is a documented
    # exception: its rotor-inertia column is exactly linearly dependent on a
    # rigid-body base parameter for this asset (verified separately by SVD;
    # see identify_motor_side's docstring), so no excitation data identifies
    # it -- only every other joint is held to the 1% recovery bar.
    identifiable = slice(1, None)
    assert np.allclose(fit.j_m[identifiable], true_j_m[identifiable], rtol=0.01)
    assert np.allclose(fit.viscous, true_viscous, rtol=0.01)
    assert np.allclose(fit.coulomb, true_coulomb, rtol=0.01)
    assert fit.residual_rms < 1e-6
    assert fit.holdout_rms is not None and fit.holdout_rms < 1e-3
    # Without a nominal prior, RR_04 A-10's SVD detection never fires: every
    # joint is reported "identified" (the pre-v3.1 behaviour), even joint 0.
    assert fit.j_m_source == ("identified",) * n_dof


def test_identify_motor_side_fixes_unidentifiable_joints_at_the_prior(ur10_asset, synthetic_motion):
    # RR_04 A-10: with a nominal prior supplied, the SVD check (exactly the
    # spec's rule: singular value < 1e-8 * max, null loading > 0.5) must fix
    # every joint whose J_m column it flags, and refit the rest.
    #
    # On this asset and this trajectory family, an SVD of the augmented
    # design (found while writing this test, not asserted a priori) shows
    # *two* directions below the 1e-8 threshold, not only the documented
    # base joint (index 0, shoulder_pan): one loads onto joint 0 at
    # essentially machine precision (ratio ~3e-15, an exact kinematic
    # degeneracy independent of the excitation), the other onto joint 1
    # (shoulder_lift) at ratio ~1e-11, stable from 6 to 16 concatenated
    # trajectories -- so it is this asset/regressor's own structure, not a
    # data-poverty artefact this fixture could out-excite. Both are exactly
    # the case A-10 asks the detector to catch; the unconstrained fit above
    # only recovers joint 1 "by accident" because the fixture is noiseless.
    pin, model, data, q, dq, ddq = synthetic_motion
    true_j_m = np.array([4.0, 4.0, 1.7, 0.5, 0.5, 0.5])
    true_viscous = np.array([0.5, 0.5, 0.3, 0.1, 0.1, 0.1])
    true_coulomb = np.array([2.0, 2.0, 1.0, 0.4, 0.4, 0.4])
    epsilon = 1.0e-2
    k_tau = np.array([0.15, 0.15, 0.15, 0.08, 0.08, 0.08])

    rigid = np.asarray([idn.inverse_dynamics(pin, model, data, q[i], dq[i], ddq[i]) for i in range(len(q))])
    friction = true_viscous[None, :] * dq + true_coulomb[None, :] * np.tanh(dq / epsilon)
    tau_m = rigid + true_j_m[None, :] * ddq + friction
    i_act = tau_m / k_tau[None, :]

    nominal_prior = np.array([4.0, 4.0, 1.7, 0.5, 0.5, 0.5])  # the simulation's own prior, correct here
    fit = identify_motor_side(ur10_asset, q, dq, ddq, i_act, k_tau, epsilon=epsilon,
                              rotor_inertia_nominal=nominal_prior)

    flagged = {i for i, source in enumerate(fit.j_m_source) if source == "prior"}
    assert flagged == {0, 1}
    for j in flagged:
        assert fit.j_m[j] == pytest.approx(nominal_prior[j])
        assert fit.j_m_sensitivity is not None and np.isfinite(fit.j_m_sensitivity[j])
    identifiable = [j for j in range(len(true_j_m)) if j not in flagged]
    assert np.allclose(fit.j_m[identifiable], true_j_m[identifiable], rtol=0.01)
    assert np.allclose(fit.viscous, true_viscous, rtol=0.01)
    assert np.allclose(fit.coulomb, true_coulomb, rtol=0.01)


def test_motor_fit_with_independent_sweep_friction(ur10_asset, synthetic_motion):
    from erd_recording.identification import friction_torque
    pin, model, data, q, dq, ddq = synthetic_motion
    prior = np.array([4.,4.,1.7,.5,.5,.5])
    parameters = {key: np.full(6,value) for key,value in
                  [('viscous',.4),('coulomb',1.),('stribeck',.2),('stribeck_velocity',.1)]}
    rigid = np.asarray([idn.inverse_dynamics(pin,model,data,qq,vv,aa) for qq,vv,aa in zip(q,dq,ddq)])
    torque = rigid + prior * ddq + friction_torque(dq,parameters)
    heldout = np.arange(len(q)) % 5 == 0
    fit = identify_motor_side(ur10_asset,q,dq,ddq,torque,np.ones(6),holdout_mask=heldout,
                              rotor_inertia_nominal=prior,friction_parameters=parameters)
    assert fit.holdout_rms < 1e-6
    assert np.allclose(fit.j_m,prior,rtol=.01)
    assert np.array_equal(fit.stribeck,parameters['stribeck'])


def test_identify_motor_side_keeps_fixed_j_m(ur10_asset, synthetic_motion):
    # UR_11: j_m_fixed holds the given joints (source "fixed"), fits the rest,
    # and the holdout reports its own target RMS.
    from erd_recording.identification import friction_torque
    pin, model, data, q, dq, ddq = synthetic_motion
    true_j_m = np.array([7.14, 7.14, 1.92, 0.61, 0.61, 0.61])
    parameters = {key: np.full(6, value) for key, value in
                  [('viscous', .4), ('coulomb', 1.), ('stribeck', 0.), ('stribeck_velocity', .03)]}
    rigid = np.asarray([idn.inverse_dynamics(pin, model, data, a, b, c) for a, b, c in zip(q, dq, ddq)])
    torque = rigid + true_j_m * ddq + friction_torque(dq, parameters)
    heldout = np.arange(len(q)) % 5 == 0
    fixed = np.array([7.14, 7.14, np.nan, 0.61, np.nan, 0.61])
    fit = identify_motor_side(ur10_asset, q, dq, ddq, torque, np.ones(6), holdout_mask=heldout,
                              rotor_inertia_nominal=np.array([4., 4., 1.7, .5, .5, .5]),
                              friction_parameters=parameters, j_m_fixed=fixed)
    assert fit.j_m_source == ("fixed", "fixed", "identified", "fixed", "identified", "fixed")
    assert np.array_equal(fit.j_m[[0, 1, 3, 5]], fixed[[0, 1, 3, 5]])
    assert np.allclose(fit.j_m[[2, 4]], true_j_m[[2, 4]], rtol=0.01)
    assert fit.holdout_rms < 1e-6
    target = torque - friction_torque(dq, parameters)  # the fit's own target: K i - friction
    assert fit.holdout_target_rms == pytest.approx(float(np.sqrt(np.mean(target[heldout] ** 2))))


def test_stribeck_is_kept_only_where_the_f_test_identifies_it():
    # UR_11: inside the measured speeds and F > F_0.99(2, n - 4).
    speeds = np.array([.02, .05, .1, .2, .35, .5])
    from erd_recording.identification import friction_torque
    clean = fit_sweep_friction(speeds, friction_torque(speeds, dict(viscous=.4, coulomb=1., stribeck=.3,
                                                                    stribeck_velocity=.1)))
    assert clean["stribeck_identifiable"] and clean["stribeck"] == pytest.approx(.3, rel=.05)
    # A_02 wrist 2 shape: a small curvature the 0.95 test accepted at F 19.2.
    noisy = 3.3 * speeds + 3.1 * np.tanh(speeds / .01) + np.array([0.03, -0.02, 0.01, -0.03, 0.02, 0.0])
    fit = fit_sweep_friction(speeds, noisy)
    assert not fit["stribeck_identifiable"] and fit["stribeck"] == 0.0
    assert fit["coulomb"] == pytest.approx(3.1, rel=.05)
