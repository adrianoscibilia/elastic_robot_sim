"""E-ur-1 (which current is the command, and K_tau) and E-ur-3 (motor-side
identification for the tau_s proxy) fitting code (RR_01 S2.1).

Pure numpy/Pinocchio; no ROS dependency, so these are unit-testable on
synthetic fixtures with known ground truth (T1.7).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LinearFit:
    slope: np.ndarray       # per joint
    intercept: np.ndarray   # per joint
    r_squared: np.ndarray   # per joint


def fit_linear_per_joint(target: np.ndarray, candidate: np.ndarray) -> LinearFit:
    """Per-joint ``target[:, j] = slope[j] * candidate[:, j] + intercept[j]``."""
    target = np.atleast_2d(target)
    candidate = np.atleast_2d(candidate)
    n_dof = target.shape[1]
    slope = np.zeros(n_dof)
    intercept = np.zeros(n_dof)
    r2 = np.zeros(n_dof)
    for j in range(n_dof):
        design = np.column_stack([candidate[:, j], np.ones(len(candidate))])
        (a, b), *_ = np.linalg.lstsq(design, target[:, j], rcond=None)
        slope[j], intercept[j] = a, b
        residual = target[:, j] - (a * candidate[:, j] + b)
        total = target[:, j] - target[:, j].mean()
        ss_res = float(np.sum(residual**2))
        ss_tot = float(np.sum(total**2))
        r2[j] = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    return LinearFit(slope, intercept, r2)


def regression_significance(residual: np.ndarray, *regressors: np.ndarray) -> np.ndarray:
    """Per-joint R^2 of ``residual`` on ``regressors`` (E-ur-1's feedback test).

    A high R^2 means ``residual`` (e.g. ``joint_control_output -
    target_moment/K_tau``) is well explained by the tracking error and its
    derivative -- i.e. it contains feedback, so it is the real command.
    """
    residual = np.atleast_2d(residual)
    n_dof = residual.shape[1]
    stacked = np.column_stack([*(np.atleast_2d(r) for r in regressors), np.ones(len(residual))])
    r2 = np.zeros(n_dof)
    for j in range(n_dof):
        solution, *_ = np.linalg.lstsq(stacked, residual[:, j], rcond=None)
        fitted = stacked @ solution
        ss_res = float(np.sum((residual[:, j] - fitted) ** 2))
        total = residual[:, j] - residual[:, j].mean()
        ss_tot = float(np.sum(total**2))
        r2[j] = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    return r2


#: E-ur-1's R^2 threshold for "the controller's own K_tau" (RR_01 S2.1).
K_TAU_R2_THRESHOLD = 0.999
#: E-ur-1's feedback-significance threshold: above this, the candidate current
#: carries the position feedback loop and is therefore the real command.
FEEDBACK_R2_THRESHOLD = 0.2


def choose_ur_tau_source(
    *,
    target_moment: np.ndarray,
    joint_control_output: np.ndarray,
    target_current: np.ndarray,
    actual_current: np.ndarray,
    tracking_error: np.ndarray,
    tracking_error_rate: np.ndarray,
    standstill_actual_current: np.ndarray | None = None,
    standstill_gravity: np.ndarray | None = None,
) -> dict[str, Any]:
    """E-ur-1's decision tree (RR_01 v3.1 S2.1 / RR_04 A-10): which current is
    ``tau``, and K_tau.

    v3.1 rewrite of the original two-candidate tree, which asked the *same*
    candidate current to both fit ``target_moment`` with R^2 > 0.999 *and*
    carry significant feedback -- self-contradictory, since a large feedback
    share is exactly what lowers that R^2 (RR_04 A-10 finding). The corrected
    tree fits K_tau once, from the two model-side signals
    (``target_moment ~ target_current``, expected exactly linear since
    neither is measured through the servo loop), then asks a *separate*
    question -- does ``joint_control_output`` differ from ``target_current``
    in a way the tracking error explains -- to choose the command source:

    1. ``K_tau`` from ``target_moment ~ target_current``. If that R^2 is
       below :data:`K_TAU_R2_THRESHOLD` on any joint, ``K_tau`` instead comes
       from the standstill gravity fit (``standstill_gravity ~
       standstill_actual_current``), which the caller must then supply.
    2. ``r = joint_control_output - target_current``, regressed on the
       tracking error and its rate.
       * Significant (> :data:`FEEDBACK_R2_THRESHOLD`): ``joint_control_output``
         carries the position feedback loop, so it is the real command:
         ``tau_source = joint_control_output``.
       * Not significant, i.e. ``joint_control_output`` is ``target_current``
         within noise (pure feedforward): the current loop tracks the
         command, so ``tau_source = actual_current``.

    Returns a dict with ``tau_source`` (``joint_control_output`` |
    ``actual_current``), ``k_tau`` (per joint) and the diagnostics behind the
    choice, including ``k_tau_source`` (``target_moment_vs_target_current`` |
    ``standstill_gravity_fit``).
    """
    fit_tc = fit_linear_per_joint(target_moment, target_current)
    diagnostics: dict[str, Any] = {
        "fit_target_current": {"r_squared": fit_tc.r_squared.tolist(), "k_tau": fit_tc.slope.tolist()},
    }

    if np.all(fit_tc.r_squared > K_TAU_R2_THRESHOLD):
        k_tau = fit_tc.slope
        k_tau_source = "target_moment_vs_target_current"
    else:
        if standstill_actual_current is None or standstill_gravity is None:
            raise ValueError(
                "choose_ur_tau_source: target_moment ~ target_current R^2 fell below "
                f"{K_TAU_R2_THRESHOLD} on at least one joint (R^2={fit_tc.r_squared.tolist()}) and no "
                "standstill_actual_current/standstill_gravity was given for the E-ur-1 step-3 fallback"
            )
        gravity_fit = fit_linear_per_joint(standstill_gravity, standstill_actual_current)
        diagnostics["fit_standstill_gravity"] = {
            "r_squared": gravity_fit.r_squared.tolist(), "k_tau": gravity_fit.slope.tolist(),
        }
        k_tau = gravity_fit.slope
        k_tau_source = "standstill_gravity_fit"
    diagnostics["k_tau_source"] = k_tau_source

    residual = joint_control_output - target_current
    feedback_r2 = regression_significance(residual, tracking_error, tracking_error_rate)
    diagnostics["feedback_r2"] = feedback_r2.tolist()

    if np.all(feedback_r2 > FEEDBACK_R2_THRESHOLD):
        tau_source = "joint_control_output"
    else:
        tau_source = "actual_current"

    if standstill_actual_current is not None and standstill_gravity is not None:
        # E-ur-1 step 4: this sanity check is written whichever branch wins.
        sanity = fit_linear_per_joint(standstill_gravity, k_tau[None, :] * standstill_actual_current)
        diagnostics["standstill_sanity_check"] = {
            "slope": sanity.slope.tolist(),  # expected ~= 1 +/- 0.1 per joint
            "r_squared": sanity.r_squared.tolist(),
        }

    return {"tau_source": tau_source, "k_tau": k_tau, "diagnostics": diagnostics}


#: RR_04 A-10: a singular value this much smaller than the largest marks a
#: null direction of the augmented (base params + J_m + friction) design.
J_M_SVD_SINGULAR_VALUE_RTOL = 1.0e-8
#: A null-space vector loading this much onto one J_m column means that
#: joint's rotor inertia is (numerically) unidentifiable from this data.
J_M_SVD_NULL_LOADING_THRESHOLD = 0.5
#: Default multiplicative half-range for the unidentified-J_m sensitivity
#: figure (RR_01 S2.1 step 2: "the prior's factor range").
J_M_PRIOR_FACTOR_RANGE = 0.3


@dataclass(frozen=True)
class MotorSideFit:
    j_m: np.ndarray           # reflected rotor inertia per joint [kg m^2]
    viscous: np.ndarray       # per joint [Nm s/rad]
    coulomb: np.ndarray       # per joint [Nm]
    base_parameters: np.ndarray
    residual_rms: float
    holdout_rms: float | None
    j_m_source: tuple[str, ...] = ()          # "identified" | "prior", per joint
    j_m_sensitivity: np.ndarray | None = None  # RMS(dJ * ddq) per joint; NaN where identified
    stribeck: np.ndarray | None = None
    stribeck_velocity: np.ndarray | None = None


def identify_motor_side(
    asset: Any, q: np.ndarray, dq: np.ndarray, ddq: np.ndarray, i_act: np.ndarray, k_tau: np.ndarray,
    *, epsilon: float = 1.0e-2, holdout_mask: np.ndarray | None = None,
    rotor_inertia_nominal: np.ndarray | None = None,
    rotor_inertia_prior_factor_range: float = J_M_PRIOR_FACTOR_RANGE,
    friction_parameters: dict[str, np.ndarray] | None = None,
) -> MotorSideFit:
    """E-ur-3: jointly fit ``K_tau * i_act = Y(q,dq,ddq) @ theta_base + J_m * ddq + f(dq)``.

    ``holdout_mask`` (boolean, one value per sample) marks rows excluded from
    the fit; the holdout RMS is reported on exactly those rows (RR_01 S2.1
    step 4: "the first and last identify trajectories are not used in the
    fit").

    **J_m identifiability (RR_01 v3.1 S2.1 / RR_04 A-10).** On some serial
    arms a base joint's own diagonal mass-matrix element does not depend on
    configuration (typically the first, vertical-axis joint), so its
    rotor-inertia column is *exactly* linearly dependent on one rigid-body
    base-parameter combination -- no amount of excitation data separates
    them. Detected here by SVD of the augmented design restricted to the fit
    rows: a singular value below ``J_M_SVD_SINGULAR_VALUE_RTOL`` times the
    largest, whose right-singular (null) vector loads more than
    ``J_M_SVD_NULL_LOADING_THRESHOLD`` onto one ``J_m`` column, marks that
    joint unidentifiable. If ``rotor_inertia_nominal`` is given, such joints'
    ``J_m`` is fixed at that prior and the remaining parameters (base +
    every other joint's ``J_m``/friction) are refit on the residual; if not
    given, the raw least-squares solution on that column is returned
    unchanged (numerically stable, just not physically meaningful alone) --
    the pre-v3.1 behaviour, still exercised by
    ``test_identify_motor_side_recovers_j_m_and_friction``. Either way
    ``j_m_source`` records which branch each joint took, and, for a fixed
    joint, ``j_m_sensitivity`` reports ``RMS(dJ * ddq)`` over
    ``dJ = rotor_inertia_nominal[j] * rotor_inertia_prior_factor_range`` --
    how much the target would move if the prior were off by that factor.
    """
    from elastic_sim import identification as idn

    pin, model, data = idn.build_model(asset)
    basis = idn.base_parameter_basis(pin, model, data, n_samples=200, seed=0)
    n_dof = q.shape[1]
    n_base = basis.shape[1]
    target = (k_tau[None, :] * i_act).reshape(-1)
    if friction_parameters is not None:
        friction = friction_torque(dq, friction_parameters, epsilon)
        target = target - friction.reshape(-1)

    def _design(qq, dqq, ddqq) -> np.ndarray:
        rows = []
        for i in range(len(qq)):
            base = idn.joint_torque_regressor(pin, model, data, qq[i], dqq[i], ddqq[i], include_friction=False)
            projected = base @ basis
            friction_columns = np.hstack([np.diag(dqq[i]), np.diag(np.tanh(dqq[i] / epsilon))])
            if friction_parameters is not None:
                friction_columns = np.empty((n_dof, 0))
            rows.append(np.hstack([projected, np.diag(ddqq[i]), friction_columns]))
        return np.vstack(rows)

    if holdout_mask is None:
        holdout_mask = np.zeros(len(q), dtype=bool)
    fit_mask = ~holdout_mask
    design_fit = _design(q[fit_mask], dq[fit_mask], ddq[fit_mask])
    ddq_fit = ddq[fit_mask]
    target_fit_2d = target.reshape(len(q), n_dof)[fit_mask]
    target_fit = target_fit_2d.reshape(-1)

    j_m_source = ["identified"] * n_dof
    j_m_sensitivity = np.full(n_dof, np.nan)
    unidentifiable: set[int] = set()
    if rotor_inertia_nominal is not None:
        singular_values, right_vectors = np.linalg.svd(design_fit, full_matrices=False)[1:]
        max_sv = float(singular_values[0]) if len(singular_values) else 0.0
        for row, sv in enumerate(singular_values):
            if max_sv > 0.0 and sv < J_M_SVD_SINGULAR_VALUE_RTOL * max_sv:
                j_m_loading = np.abs(right_vectors[row, n_base:n_base + n_dof])
                if j_m_loading.max() > J_M_SVD_NULL_LOADING_THRESHOLD:
                    unidentifiable.add(int(np.argmax(j_m_loading)))

    if unidentifiable:
        fixed_j_m = np.zeros(n_dof)
        for j in unidentifiable:
            fixed_j_m[j] = rotor_inertia_nominal[j]
            j_m_source[j] = "prior"
            delta_j = rotor_inertia_nominal[j] * rotor_inertia_prior_factor_range
            j_m_sensitivity[j] = float(np.sqrt(np.mean((delta_j * ddq[:, j]) ** 2)))
        adjustment = np.zeros_like(target_fit_2d)
        for j in unidentifiable:
            adjustment[:, j] = fixed_j_m[j] * ddq_fit[:, j]
        target_fit_adjusted = (target_fit_2d - adjustment).reshape(-1)
        keep_cols = [c for c in range(design_fit.shape[1])
                     if not (n_base <= c < n_base + n_dof and (c - n_base) in unidentifiable)]
        solution_reduced, *_ = np.linalg.lstsq(design_fit[:, keep_cols], target_fit_adjusted, rcond=None)
        solution = np.zeros(design_fit.shape[1])
        for index, column in enumerate(keep_cols):
            solution[column] = solution_reduced[index]
        for j in unidentifiable:
            solution[n_base + j] = fixed_j_m[j]
    else:
        solution, *_ = np.linalg.lstsq(design_fit, target_fit, rcond=None)

    theta_base = solution[:n_base]
    j_m = solution[n_base:n_base + n_dof]
    viscous = solution[n_base + n_dof:n_base + 2 * n_dof]
    coulomb = solution[n_base + 2 * n_dof:n_base + 3 * n_dof]
    if friction_parameters is not None:
        viscous = friction_parameters['viscous']
        coulomb = friction_parameters['coulomb']

    fitted_fit = design_fit @ solution
    residual_rms = float(np.sqrt(np.mean((fitted_fit - target_fit) ** 2)))

    holdout_rms = None
    if np.any(holdout_mask):
        design_holdout = _design(q[holdout_mask], dq[holdout_mask], ddq[holdout_mask])
        target_holdout = target.reshape(len(q), n_dof)[holdout_mask].reshape(-1)
        fitted_holdout = design_holdout @ solution
        holdout_rms = float(np.sqrt(np.mean((fitted_holdout - target_holdout) ** 2)))

    return MotorSideFit(j_m=j_m, viscous=viscous, coulomb=coulomb, base_parameters=theta_base,
                        residual_rms=residual_rms, holdout_rms=holdout_rms,
                        j_m_source=tuple(j_m_source), j_m_sensitivity=j_m_sensitivity,
                        stribeck=None if friction_parameters is None else friction_parameters['stribeck'],
                        stribeck_velocity=None if friction_parameters is None else friction_parameters['stribeck_velocity'])


def friction_torque(dq, parameters, epsilon=0.01):
    """FrictionModel's viscous/Coulomb law plus the measured Stribeck term."""
    velocity = np.asarray(dq)
    smooth_sign = np.tanh(velocity / epsilon)
    result = parameters['viscous'] * velocity + parameters['coulomb'] * smooth_sign
    if parameters.get('stribeck') is not None:
        result += parameters['stribeck'] * np.exp(-(velocity / parameters['stribeck_velocity'])**2) * smooth_sign
    return result


def fit_sweep_friction(speeds, torques, *, epsilon=0.01):
    """Fit six bidirectional sweep pairs; grid-search the Stribeck velocity."""
    from scipy.optimize import nnls
    best = None
    for scale in np.geomspace(max(min(speeds) / 2, 1e-4), max(speeds) * 2, 40):
        sign = np.tanh(np.asarray(speeds) / epsilon)
        design = np.column_stack([speeds, sign, np.exp(-(np.asarray(speeds) / scale)**2) * sign])
        coefficients, residual = nnls(design, torques)
        if best is None or residual < best[0]:
            best = (residual, coefficients, scale)
    residual, c, scale = best
    return {'viscous': float(c[0]), 'coulomb': float(c[1]), 'stribeck': float(c[2]),
            'stribeck_velocity': float(scale), 'residual_rms': float(residual / np.sqrt(len(speeds)))}
