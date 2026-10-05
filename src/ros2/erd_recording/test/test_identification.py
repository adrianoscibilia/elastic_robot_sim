"""T1.7: E-ur-1/E-ur-3 fitting code recovers known fixture parameters."""

from __future__ import annotations

import numpy as np
import pytest

from elastic_sim import identification as idn
from elastic_sim.assets import AssetRegistry

from erd_recording.identification import (
    choose_ur_tau_source,
    fit_linear_per_joint,
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


def test_choose_ur_tau_source_picks_joint_control_output_with_feedback():
    # RR_04 A-10 tree: K_tau always from target_moment ~ target_current (the
    # two model-side, noiseless signals); the source choice then turns on
    # whether joint_control_output differs from target_current in a way the
    # tracking error explains.
    rng = np.random.default_rng(2)
    n_dof = 3
    n = 500
    k_tau = np.array([0.15, 0.15, 0.10])
    target_moment = rng.uniform(-10, 10, size=(n, n_dof))
    tracking_error = rng.uniform(-0.01, 0.01, size=(n, n_dof))
    tracking_error_rate = rng.uniform(-0.1, 0.1, size=(n, n_dof))
    feedback_gain = 200.0
    target_current = target_moment / k_tau[None, :]  # exact, model-side: fits K_tau to machine precision
    # joint_control_output carries an extra feedback term on top of
    # target_current -- the signature E-ur-1's step-2 regression looks for.
    joint_control_output = target_current + feedback_gain * tracking_error
    actual_current = target_current + rng.normal(scale=0.01, size=(n, n_dof))

    result = choose_ur_tau_source(
        target_moment=target_moment, joint_control_output=joint_control_output,
        target_current=target_current, actual_current=actual_current,
        tracking_error=tracking_error, tracking_error_rate=tracking_error_rate,
    )
    assert result["tau_source"] == "joint_control_output"
    assert result["diagnostics"]["k_tau_source"] == "target_moment_vs_target_current"
    assert np.allclose(result["k_tau"], k_tau, rtol=1e-9)


def test_choose_ur_tau_source_picks_actual_current_when_pure_feedforward():
    # joint_control_output == target_current within noise (no feedback
    # signature): the current loop tracks the command, so the real command
    # is read off actual_current (RR_04 A-10 step 2, second branch).
    rng = np.random.default_rng(3)
    n_dof = 3
    n = 500
    k_tau = np.array([0.15, 0.15, 0.10])
    target_moment = rng.uniform(-10, 10, size=(n, n_dof))
    tracking_error = rng.uniform(-0.01, 0.01, size=(n, n_dof))
    tracking_error_rate = rng.uniform(-0.1, 0.1, size=(n, n_dof))
    target_current = target_moment / k_tau[None, :]
    joint_control_output = target_current + rng.normal(scale=1e-6, size=(n, n_dof))  # noise only
    actual_current = target_current + rng.normal(scale=0.01, size=(n, n_dof))

    result = choose_ur_tau_source(
        target_moment=target_moment, joint_control_output=joint_control_output,
        target_current=target_current, actual_current=actual_current,
        tracking_error=tracking_error, tracking_error_rate=tracking_error_rate,
    )
    assert result["tau_source"] == "actual_current"


def test_choose_ur_tau_source_falls_back_to_standstill_gravity_fit():
    # When target_moment ~ target_current doesn't fit (R^2 below threshold),
    # K_tau instead comes from the standstill gravity fit (E-ur-1 step 3).
    rng = np.random.default_rng(4)
    n_dof = 2
    n = 300
    target_moment = rng.uniform(-10, 10, size=(n, n_dof))
    target_current = rng.uniform(-10, 10, size=(n, n_dof))  # uncorrelated: R^2 ~ 0
    tracking_error = rng.uniform(-0.01, 0.01, size=(n, n_dof))
    tracking_error_rate = rng.uniform(-0.1, 0.1, size=(n, n_dof))
    joint_control_output = target_current.copy()
    actual_current = target_current.copy()

    true_k_tau = np.array([0.2, 0.12])
    standstill_gravity = rng.uniform(-5, 5, size=(50, n_dof))
    standstill_actual_current = standstill_gravity / true_k_tau[None, :]

    result = choose_ur_tau_source(
        target_moment=target_moment, joint_control_output=joint_control_output,
        target_current=target_current, actual_current=actual_current,
        tracking_error=tracking_error, tracking_error_rate=tracking_error_rate,
        standstill_actual_current=standstill_actual_current, standstill_gravity=standstill_gravity,
    )
    assert result["diagnostics"]["k_tau_source"] == "standstill_gravity_fit"
    assert np.allclose(result["k_tau"], true_k_tau, rtol=1e-9)


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
