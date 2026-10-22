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
#: E-ur-1's feedback-significance threshold (RR_04 A-10). UR_11: reported
#: only; it no longer decides ``tau_source`` (wrist 3 sat on it, 0.19-0.30,
#: and the choice flipped between runs of the same plan).
FEEDBACK_R2_THRESHOLD = 0.2
#: UR_11 E-ur-1 step 2: the controller's standstill ratio identifies a
#: joint's K_tau only where target_moment spans at least this much across
#: the holds (pan, wrist 2 and wrist 3 carry no gravity load: ~0 N*m).
CONTROLLER_RATIO_MIN_RANGE_NM = 1.0
#: UR_11 E-ur-1 step 3: a K_tau taken from an identified joint of the same
#: drive size may raise that joint's controller feed-forward residual RMS
#: by at most this fraction (real CB3: <= 4.8 %), else the data contradict it.
JOINT_SIZE_MAX_RESIDUAL_INCREASE = 0.10
#: UR_11 tau_source: ``joint_control_output`` is the command when it differs
#: from ``target_current`` by more than this many times its difference from
#: ``actual_current`` on every joint (real CB3: 6.7-40; URSim: 0/0) ...
COMMAND_DEVIATION_RATIO = 3.0
#: ... and the pure-feedforward case (``actual_current``) needs at most this
#: on every joint. Anything between, or mixed over joints, is refused.
FEEDFORWARD_DEVIATION_RATIO = 1.0


class IdentificationRefused(ValueError):
    """E-ur-1 cannot decide K_tau or ``tau_source`` from this data; the
    identification fails (``diagnostics`` says why) instead of passing with
    a placeholder (UR_10 S3: K_tau 0 on four joints passed as ``ok``)."""

    def __init__(self, message: str, diagnostics: dict[str, Any]):
        super().__init__(message)
        self.diagnostics = diagnostics


def controller_feedforward_fit(
    target_current: np.ndarray, target_moment: np.ndarray, target_qd: np.ndarray, target_qdd: np.ndarray,
    *, k_tau: np.ndarray | None = None, epsilon: float = 1.0e-2,
) -> dict[str, list]:
    """UR_11: the CB3 controller's own feed-forward, per joint,
    ``target_current = target_moment / K + a * target_qdd + b * target_qd
    + c * tanh(target_qd / epsilon) + d`` (R^2 >= 0.995 on every joint of the
    2026-10-21 runs). Free fit (``k_tau_free``), the residual RMS with
    ``k_tau`` fixed, and ``moment_r2_gain``: how much of ``target_current``
    the moment term explains beyond the inertia/friction terms (0 where
    ``target_moment`` carries no information, so ``K`` is not testable)."""
    n_dof = target_current.shape[1]
    out: dict[str, list] = {"k_tau_free": [], "r_squared_free": [], "residual_rms_free": [],
                            "residual_rms_fixed": [], "moment_r2_gain": []}
    for j in range(n_dof):
        y = target_current[:, j]
        others = np.column_stack([target_qdd[:, j], target_qd[:, j], np.tanh(target_qd[:, j] / epsilon),
                                  np.ones(len(y))])
        full = np.column_stack([target_moment[:, j], others])
        coefficients, *_ = np.linalg.lstsq(full, y, rcond=None)
        residual = y - full @ coefficients
        variance = float(np.var(y))
        without, *_ = np.linalg.lstsq(others, y, rcond=None)
        r2_free = 1.0 - float(np.var(residual)) / variance if variance > 0 else 0.0
        r2_without = 1.0 - float(np.var(y - others @ without)) / variance if variance > 0 else 0.0
        out["k_tau_free"].append(float(1.0 / coefficients[0]) if coefficients[0] != 0 else None)
        out["r_squared_free"].append(r2_free)
        out["residual_rms_free"].append(float(np.sqrt(np.mean(residual**2))))
        out["moment_r2_gain"].append(r2_free - r2_without)
        if k_tau is None or not np.isfinite(k_tau[j]) or k_tau[j] == 0:
            out["residual_rms_fixed"].append(None)
            continue
        shifted = y - target_moment[:, j] / k_tau[j]
        fixed, *_ = np.linalg.lstsq(others, shifted, rcond=None)
        out["residual_rms_fixed"].append(float(np.sqrt(np.mean((shifted - others @ fixed) ** 2))))
    return out


def controller_k_tau(
    *, standstill_target_moment: np.ndarray, standstill_target_current: np.ndarray,
    feedforward: dict[str, np.ndarray] | None = None, joint_sizes: tuple[str, ...] | None = None,
    joint_size_source: str | None = None,
) -> tuple[np.ndarray, list[str | None], dict[str, Any]]:
    """UR_11 E-ur-1 steps 2-3: K_tau per joint from the controller's own model.

    2. **Standstill ratio.** At standstill the CB3 controller's
       ``target_current`` is ``target_moment / K`` exactly (its friction and
       inertia terms vanish): ``target_moment ~ target_current`` over the
       hold samples, identified where the moment spans at least
       :data:`CONTROLLER_RATIO_MIN_RANGE_NM` and R^2 >= :data:`K_TAU_R2_THRESHOLD`.
    3. **Joint-size identity.** A joint not identified in step 2 takes the K
       of an identified joint of the same drive size (``joint_sizes``, cited
       in ``joint_size_source``) if every identified joint of that size
       agrees within 1 % and the joint's own controller feed-forward
       (``feedforward``: ``target_current``/``target_moment``/``target_qd``/
       ``target_qdd`` over moving segments) does not contradict it
       (:data:`JOINT_SIZE_MAX_RESIDUAL_INCREASE`).

    Returns ``(k_tau, source_per_joint, diagnostics)``; a joint with neither
    source has ``None`` as its source and ``nan`` as K (the caller refuses).
    """
    n_dof = standstill_target_moment.shape[1]
    span = standstill_target_moment.max(axis=0) - standstill_target_moment.min(axis=0)
    ratio = fit_linear_per_joint(standstill_target_moment, standstill_target_current)
    k_tau = np.full(n_dof, np.nan)
    sources: list[str | None] = [None] * n_dof
    standstill = []
    for j in range(n_dof):
        identified = bool(span[j] >= CONTROLLER_RATIO_MIN_RANGE_NM and ratio.r_squared[j] >= K_TAU_R2_THRESHOLD)
        if identified:
            k_tau[j] = ratio.slope[j]
            sources[j] = "controller_standstill_ratio"
        standstill.append({"moment_range_nm": float(span[j]), "k_tau": float(ratio.slope[j]),
                           "intercept_nm": float(ratio.intercept[j]), "r_squared": float(ratio.r_squared[j]),
                           "identified": identified})
    diagnostics: dict[str, Any] = {"controller_standstill_ratio": standstill}

    identity = []
    if joint_sizes is not None:
        for j in range(n_dof):
            if sources[j] is not None:
                continue
            donors = [d for d in range(n_dof) if sources[d] == "controller_standstill_ratio"
                      and joint_sizes[d] == joint_sizes[j]]
            entry = {"joint": j, "size": joint_sizes[j], "donors": donors, "accepted": False}
            identity.append(entry)
            if not donors:
                entry["reason"] = f"no identified joint of {joint_sizes[j]}"
                continue
            values = k_tau[donors]
            if values.max() - values.min() > 0.01 * abs(values.mean()):
                entry["reason"] = f"identified joints of {joint_sizes[j]} disagree: {values.tolist()}"
                continue
            entry["k_tau"] = float(values.mean())
            if feedforward is None:
                entry["reason"] = "no controller feed-forward data to check the identity against"
                continue
            entry["accepted"] = True
            k_tau[j] = values.mean()
            sources[j] = f"joint_size_identity:{joint_sizes[j]}"
    if feedforward is not None:
        check = controller_feedforward_fit(feedforward["target_current"], feedforward["target_moment"],
                                           feedforward["target_qd"], feedforward["target_qdd"], k_tau=k_tau)
        diagnostics["controller_feedforward"] = check
        for entry in identity:
            if not entry["accepted"]:
                continue
            j = entry["joint"]
            free, fixed = check["residual_rms_free"][j], check["residual_rms_fixed"][j]
            # floored: a noise-free feed-forward fits to ~1e-16 on both sides
            floor = 1e-9 * float(np.sqrt(np.mean(feedforward["target_current"][:, j] ** 2)))
            increase = (fixed - free) / max(free, floor, np.finfo(float).tiny)
            entry.update(residual_increase=increase, moment_r2_gain=check["moment_r2_gain"][j],
                         k_tau_free=check["k_tau_free"][j])
            if increase > JOINT_SIZE_MAX_RESIDUAL_INCREASE:
                entry.update(accepted=False, reason=f"feed-forward residual +{increase:.1%} with K {k_tau[j]:.4g}"
                             f" > {JOINT_SIZE_MAX_RESIDUAL_INCREASE:.0%}: the data contradict the identity")
                k_tau[j] = np.nan
                sources[j] = None
    diagnostics["joint_size_identity"] = identity
    diagnostics["joint_size_source"] = joint_size_source
    return k_tau, sources, diagnostics


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
    standstill_target_moment: np.ndarray | None = None,
    standstill_target_current: np.ndarray | None = None,
    feedforward: dict[str, np.ndarray] | None = None,
    joint_sizes: tuple[str, ...] | None = None,
    joint_size_source: str | None = None,
) -> dict[str, Any]:
    """E-ur-1's decision tree (RR_01 v3.1 S2.1 / RR_04 A-10, revised UR_11):
    which current is ``tau``, and K_tau.

    1. ``K_tau`` from ``target_moment ~ target_current`` when R^2 >
       :data:`K_TAU_R2_THRESHOLD` on every joint whose ``target_moment``
       varies (URSim: exact; its currents are pure feed-forward).
    2.-3. Otherwise (real CB3: ``target_current`` also carries the
       controller's friction and rotor-inertia feed-forward, so the plain
       fit has R^2 ~ 0.01-0.2) from :func:`controller_k_tau`: the standstill
       ratio, then the joint-size identity. A joint left without either
       raises :class:`IdentificationRefused`. The former step 3, the
       standstill gravity fit of ``actual_current``, is reported only
       (``fit_standstill_gravity``): at standstill static friction and the
       position loop's integral hold part of the load, so ``actual_current``
       scatters around ``target_current`` (``target_moment / actual_current``
       from 2 to 56 on the shoulder over the 2026-10-21 holds) and its slope
       is biased (shoulder 17.6 vs the controller's 14.18 N*m/A).

    ``tau_source`` (UR_11): ``joint_control_output`` is the real command
    when it differs from the feed-forward ``target_current`` far more than
    from the measured ``actual_current`` (the current loop tracks it):
    ratio ``rms(jco - target_current) / rms(jco - actual_current)`` above
    :data:`COMMAND_DEVIATION_RATIO` on every joint. Where
    ``joint_control_output`` is ``target_current`` (ratio <=
    :data:`FEEDFORWARD_DEVIATION_RATIO` everywhere; URSim: identically 0)
    the command is pure feed-forward and ``tau_source = actual_current``,
    as before. A mixed or in-between result is refused, never flipped. The
    feedback regression (``feedback_r2``) stays as a diagnostic.

    Returns a dict with ``tau_source``, ``k_tau`` (per joint) and the
    diagnostics, including ``k_tau_source`` (``target_moment_vs_target_current``
    | ``controller`` when steps 2-3 ran) and ``k_tau_source_per_joint``.
    """
    fit_tc = fit_linear_per_joint(target_moment, target_current)
    diagnostics: dict[str, Any] = {
        "fit_target_current": {"r_squared": fit_tc.r_squared.tolist(), "k_tau": fit_tc.slope.tolist()},
    }
    n_dof = target_moment.shape[1]
    if standstill_actual_current is not None and standstill_gravity is not None:
        gravity_fit = fit_linear_per_joint(standstill_gravity, standstill_actual_current)
        diagnostics["fit_standstill_gravity"] = {
            "r_squared": gravity_fit.r_squared.tolist(), "k_tau": gravity_fit.slope.tolist(),
            "note": "diagnostic only (UR_11): static friction biases actual_current at standstill",
        }

    # A constant target_moment fits with R^2 1 by convention (ss_tot = 0): no K.
    varies = np.ptp(target_moment, axis=0) > 0
    if np.all(fit_tc.r_squared > K_TAU_R2_THRESHOLD) and np.all(varies):
        k_tau = fit_tc.slope
        k_tau_source = "target_moment_vs_target_current"
        per_joint: list[str | None] = [k_tau_source] * n_dof
    else:
        if standstill_target_moment is None or standstill_target_current is None:
            raise ValueError(
                "choose_ur_tau_source: target_moment ~ target_current R^2 fell below "
                f"{K_TAU_R2_THRESHOLD} on at least one joint (R^2={fit_tc.r_squared.tolist()}) and no "
                "standstill_target_moment/standstill_target_current was given for E-ur-1 steps 2-3"
            )
        k_tau, per_joint, controller = controller_k_tau(
            standstill_target_moment=standstill_target_moment, standstill_target_current=standstill_target_current,
            feedforward=feedforward, joint_sizes=joint_sizes, joint_size_source=joint_size_source)
        diagnostics.update(controller)
        k_tau_source = "controller"
    diagnostics["k_tau_source"] = k_tau_source
    diagnostics["k_tau_source_per_joint"] = per_joint

    residual = joint_control_output - target_current
    feedback_r2 = regression_significance(residual, tracking_error, tracking_error_rate)
    diagnostics["feedback_r2"] = feedback_r2.tolist()
    deviation = np.sqrt(np.mean(residual**2, axis=0))
    tracking = np.sqrt(np.mean((joint_control_output - actual_current) ** 2, axis=0))
    ratio = np.where(tracking > 0, deviation / np.where(tracking > 0, tracking, 1.0),
                     np.where(deviation > 0, np.inf, 0.0))
    diagnostics["command_deviation"] = {
        "rms_jco_minus_target_current": deviation.tolist(), "rms_jco_minus_actual_current": tracking.tolist(),
        "ratio": ratio.tolist(), "command_threshold": COMMAND_DEVIATION_RATIO,
        "feedforward_threshold": FEEDFORWARD_DEVIATION_RATIO,
    }

    unidentified = [j for j, source in enumerate(per_joint) if source is None]
    if unidentified:
        raise IdentificationRefused(f"K_tau unidentified for joints {unidentified}: no controller standstill ratio "
                                    "and no accepted joint-size identity", diagnostics)
    if np.all(ratio > COMMAND_DEVIATION_RATIO):
        tau_source = "joint_control_output"
    elif np.all(ratio <= FEEDFORWARD_DEVIATION_RATIO):
        tau_source = "actual_current"
    else:
        raise IdentificationRefused(f"tau_source undecided: rms(jco - target_current) / rms(jco - actual_current) = "
                                    f"{np.round(ratio, 3).tolist()} is neither > {COMMAND_DEVIATION_RATIO} nor <= "
                                    f"{FEEDFORWARD_DEVIATION_RATIO} on every joint", diagnostics)

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
    holdout_target_rms: float | None = None   # UR_11: RMS of the target on the holdout rows


#: UR_11: the controller's rotor inertia is taken only where
#: ``target_moment - RNEA(target)`` is explained this well by ``target_qdd``.
CONTROLLER_J_M_MIN_R_SQUARED = 0.9


def controller_rotor_inertia(asset: Any, target_q: np.ndarray, target_qd: np.ndarray, target_qdd: np.ndarray,
                             target_moment: np.ndarray) -> dict[str, list]:
    """UR_11: the CB3 controller's reflected rotor inertia per joint. On the
    real UR10 ``target_moment`` is the nominal rigid-body RNEA of the target
    motion (the bare asset's own inertials: slope 1.0016 at standstill) plus
    ``J_m * target_qdd`` (R^2 0.94-1.0000, the same J_m per drive size, A = B
    within 0.6 %). Fit ``target_moment - RNEA = J_m * target_qdd + c * target_qd
    + d`` per joint; ``identified`` where R^2 >= :data:`CONTROLLER_J_M_MIN_R_SQUARED`
    and J_m > 0."""
    from elastic_sim import identification as idn

    pin, model, data = idn.build_model(asset)
    rnea = np.asarray([pin.rnea(model, data, a, b, c).copy() for a, b, c in zip(target_q, target_qd, target_qdd)])
    excess = target_moment - rnea
    out: dict[str, list] = {"j_m": [], "r_squared": [], "identified": []}
    for j in range(target_q.shape[1]):
        design = np.column_stack([target_qdd[:, j], target_qd[:, j], np.ones(len(excess))])
        coefficients, *_ = np.linalg.lstsq(design, excess[:, j], rcond=None)
        variance = float(np.var(excess[:, j]))
        r2 = 1.0 - float(np.var(excess[:, j] - design @ coefficients)) / variance if variance > 0 else 0.0
        out["j_m"].append(float(coefficients[0]))
        out["r_squared"].append(r2)
        out["identified"].append(bool(r2 >= CONTROLLER_J_M_MIN_R_SQUARED and coefficients[0] > 0))
    return out


def identify_motor_side(
    asset: Any, q: np.ndarray, dq: np.ndarray, ddq: np.ndarray, i_act: np.ndarray, k_tau: np.ndarray,
    *, epsilon: float = 1.0e-2, holdout_mask: np.ndarray | None = None,
    rotor_inertia_nominal: np.ndarray | None = None,
    rotor_inertia_prior_factor_range: float = J_M_PRIOR_FACTOR_RANGE,
    friction_parameters: dict[str, np.ndarray] | None = None,
    j_m_fixed: np.ndarray | None = None,
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

    UR_11: ``j_m_fixed`` (finite entries) fixes those joints' J_m before the
    fit, e.g. at the controller's own value (:func:`controller_rotor_inertia`);
    their ``j_m_source`` is ``"fixed"`` and the SVD prior applies only to
    the remaining joints.
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
    fixed: set[int] = set()
    if j_m_fixed is not None:
        fixed = {j for j in range(n_dof) if np.isfinite(j_m_fixed[j])}
    if rotor_inertia_nominal is not None:
        singular_values, right_vectors = np.linalg.svd(design_fit, full_matrices=False)[1:]
        max_sv = float(singular_values[0]) if len(singular_values) else 0.0
        for row, sv in enumerate(singular_values):
            if max_sv > 0.0 and sv < J_M_SVD_SINGULAR_VALUE_RTOL * max_sv:
                j_m_loading = np.abs(right_vectors[row, n_base:n_base + n_dof])
                if j_m_loading.max() > J_M_SVD_NULL_LOADING_THRESHOLD:
                    unidentifiable.add(int(np.argmax(j_m_loading)))
    unidentifiable -= fixed

    if unidentifiable or fixed:
        fixed_j_m = np.zeros(n_dof)
        for j in fixed:
            fixed_j_m[j] = j_m_fixed[j]
            j_m_source[j] = "fixed"
        for j in unidentifiable:
            fixed_j_m[j] = rotor_inertia_nominal[j]
            j_m_source[j] = "prior"
            delta_j = rotor_inertia_nominal[j] * rotor_inertia_prior_factor_range
            j_m_sensitivity[j] = float(np.sqrt(np.mean((delta_j * ddq[:, j]) ** 2)))
        held = unidentifiable | fixed
        adjustment = np.zeros_like(target_fit_2d)
        for j in held:
            adjustment[:, j] = fixed_j_m[j] * ddq_fit[:, j]
        target_fit_adjusted = (target_fit_2d - adjustment).reshape(-1)
        keep_cols = [c for c in range(design_fit.shape[1])
                     if not (n_base <= c < n_base + n_dof and (c - n_base) in held)]
        solution_reduced, *_ = np.linalg.lstsq(design_fit[:, keep_cols], target_fit_adjusted, rcond=None)
        solution = np.zeros(design_fit.shape[1])
        for index, column in enumerate(keep_cols):
            solution[column] = solution_reduced[index]
        for j in held:
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
    holdout_target_rms = None
    if np.any(holdout_mask):
        design_holdout = _design(q[holdout_mask], dq[holdout_mask], ddq[holdout_mask])
        target_holdout = target.reshape(len(q), n_dof)[holdout_mask].reshape(-1)
        fitted_holdout = design_holdout @ solution
        holdout_rms = float(np.sqrt(np.mean((fitted_holdout - target_holdout) ** 2)))
        holdout_target_rms = float(np.sqrt(np.mean(target_holdout**2)))

    return MotorSideFit(j_m=j_m, viscous=viscous, coulomb=coulomb, base_parameters=theta_base,
                        residual_rms=residual_rms, holdout_rms=holdout_rms,
                        j_m_source=tuple(j_m_source), j_m_sensitivity=j_m_sensitivity,
                        stribeck=None if friction_parameters is None else friction_parameters['stribeck'],
                        stribeck_velocity=None if friction_parameters is None else friction_parameters['stribeck_velocity'],
                        holdout_target_rms=holdout_target_rms)


def friction_torque(dq, parameters, epsilon=0.01):
    """FrictionModel's viscous/Coulomb law plus the measured Stribeck term."""
    velocity = np.asarray(dq)
    smooth_sign = np.tanh(velocity / epsilon)
    result = parameters['viscous'] * velocity + parameters['coulomb'] * smooth_sign
    if parameters.get('stribeck') is not None:
        result += parameters['stribeck'] * np.exp(-(velocity / parameters['stribeck_velocity'])**2) * smooth_sign
    return result


#: UR_11: the Stribeck term's two extra parameters must pass an F-test at
#: this level against viscous + Coulomb alone. Six sweep pairs leave two
#: residual degrees of freedom: at 0.95 (F > 19) the 2026-10-21 elbow passed
#: on A_02 (F 26.7) and failed on B_02 (F 0.2), and A_02's wrist 2 passed
#: (19.2) only by trading its Coulomb term to 0 -- not identified. At 0.99
#: (F > 99) the RSS must drop ~100x.
STRIBECK_F_TEST_LEVEL = 0.99


def fit_sweep_friction(speeds, torques, *, epsilon=0.01):
    """Fit six bidirectional sweep pairs: non-negative viscous + Coulomb, and
    a Stribeck term only where the data identify it.

    UR_11: the former fit grid-searched the Stribeck velocity from half the
    slowest to twice the fastest plateau and kept whatever lowered the
    residual -- three non-negative amplitudes on six points. On the
    2026-10-21 runs it returned amplitudes up to 92 N*m with "Stribeck
    velocities" up to 1.4 rad/s, traded against the viscous slope (pan
    70.9 / 1.05), which ``friction_torque`` then put into the target near
    every velocity reversal. Now the Stribeck velocity is searched only
    inside the measured speeds (below the slowest plateau it is invisible)
    and the term is kept only if an F-test at :data:`STRIBECK_F_TEST_LEVEL`
    accepts its two parameters; otherwise ``stribeck`` is 0 and
    ``stribeck_velocity`` the slowest speed (a finite placeholder that
    multiplies 0)."""
    from scipy.optimize import nnls
    from scipy.stats import f as f_distribution

    speeds = np.abs(np.asarray(speeds, dtype=float))
    torques = np.asarray(torques, dtype=float)
    sign = np.tanh(speeds / epsilon)
    base, base_norm = nnls(np.column_stack([speeds, sign]), torques)
    rss_base = float(base_norm ** 2)
    best = None
    dof = len(speeds) - 4
    if dof > 0 and speeds.max() > speeds.min():
        for scale in np.geomspace(speeds.min(), speeds.max(), 40):
            design = np.column_stack([speeds, sign, np.exp(-(speeds / scale) ** 2) * sign])
            coefficients, norm = nnls(design, torques)
            if best is None or norm < best[0]:
                best = (float(norm), coefficients, float(scale))
    f_value = None
    if best is not None and best[1][2] > 0:
        rss = best[0] ** 2
        f_value = (rss_base - rss) / 2 / (rss / dof) if rss > 0 else (np.inf if rss_base > 0 else 0.0)
    accepted = f_value is not None and f_value > f_distribution.ppf(STRIBECK_F_TEST_LEVEL, 2, dof)
    if accepted:
        norm, c, scale = best
        return {'viscous': float(c[0]), 'coulomb': float(c[1]), 'stribeck': float(c[2]),
                'stribeck_velocity': scale, 'stribeck_identifiable': True, 'stribeck_f': float(f_value),
                'residual_rms': float(norm / np.sqrt(len(speeds)))}
    return {'viscous': float(base[0]), 'coulomb': float(base[1]), 'stribeck': 0.0,
            'stribeck_velocity': float(speeds.min()), 'stribeck_identifiable': False,
            'stribeck_f': None if f_value is None else float(f_value),
            'residual_rms': float(base_norm / np.sqrt(len(speeds)))}
