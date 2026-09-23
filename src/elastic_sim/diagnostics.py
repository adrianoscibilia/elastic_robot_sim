"""Q-C diagnostics: what a dataset's inputs actually contain (round 5).

``R5_00`` Q-C asks for five measurements, before any training, on a handful of
rollouts per controller mode.  This module computes them from a written dataset
(CSV or Parquet) plus its manifest, so the same code answers the question for a
simulated dataset and for a recorded one:

1. **Collinearity of ``tau_cmd`` with the state.**  ``R^2`` of a least-squares
   fit of the commanded torque on the rigid regressor ``Phi(q, dq, ddq)``, and
   the condition number of ``[Phi | tau_cmd]`` against that of ``Phi`` alone.
   Under exact computed torque ``tau_cmd`` is very nearly a deterministic
   function of the reference state, so it adds a column the regressor almost
   already spans: the model's tau-path is then barely identifiable and "copy
   tau" fits as well as any physics (``R5_00`` Sec 5.3).
2. **Regressor conditioning on the achieved motion**, not the designed
   reference -- the two part company as soon as the controller stops cancelling
   the plant.
3. **Decomposition of ``ft - tau_rigid,nominal``** into the terms that can
   explain it: payload, the consumer's own Savitzky-Golay differentiation loss,
   link-side friction, and noise.  Energy per term, so "the residual class has
   something to learn" becomes a number instead of an argument.
4. **Spectral content** of the achieved acceleration, ``tau_cmd`` and the
   deflection inside the probe band.
5. **Deflection and damping-ratio observability** per bag: deflection RMS
   against the noise floor, which is what decided that the legacy FMRR dataset
   could not see elasticity at all (``R5_01`` Sec 3, item 3).

Every number is computed per bag and aggregated per ``(tier, controller mode)``
group, because the comparison the round is about is *between* modes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

#: The consumer's own differentiation, reproduced exactly: ``dynamic_model_nn``
#: takes ``ddq`` from a Savitzky-Golay filter of order 3 over an 11-sample
#: window, which at a 2 ms grid is a 22 ms window and so a low-pass well below
#: the modal probe's top line (``R5_00`` Sec 5.2, Q-E).
SG_WINDOW = 11
SG_ORDER = 3

#: A sample counts as near-singular when its ``sigma_min(J)`` falls below this
#: fraction of the bag's own median (``R5_07`` T-15).  Relative, because the
#: full 6xn frame Jacobian's smallest singular value mixes force and moment
#: units and only compares meaningfully against the same asset's own
#: distribution.
SINGULARITY_FRACTION = 0.25


def savitzky_golay_acceleration(
    velocity: np.ndarray, time_step: float, *, window: int = SG_WINDOW, order: int = SG_ORDER,
) -> np.ndarray:
    """``ddq`` as the consumer computes it: SG derivative of the velocity."""
    from scipy.signal import savgol_filter

    velocity = np.asarray(velocity, dtype=float)
    window = min(int(window), len(velocity) - (1 - len(velocity) % 2))
    if window <= order:
        return np.gradient(velocity, time_step, axis=0)
    return savgol_filter(velocity, window, order, deriv=1, delta=time_step, axis=0)


def _column_block(frame: pd.DataFrame, prefix: str, n_dof: int) -> np.ndarray | None:
    names = [f"{prefix}{index}" for index in range(n_dof)]
    if not all(name in frame.columns for name in names):
        return None
    return frame[names].to_numpy(dtype=float)


def fit_r2(design: np.ndarray, target: np.ndarray) -> float:
    """``R^2`` of the least-squares fit ``design @ x ~ target``.

    Computed on centred columns with an explicit intercept, so a constant
    offset (a gravity bias, a torque-sensor offset) does not inflate it.
    """
    design = np.asarray(design, dtype=float)
    target = np.asarray(target, dtype=float).reshape(-1)
    augmented = np.hstack([design, np.ones((len(design), 1))])
    solution, *_ = np.linalg.lstsq(augmented, target, rcond=None)
    residual = augmented @ solution - target
    variance = float(np.var(target))
    if variance <= 0.0:
        return float("nan")
    return float(1.0 - np.var(residual) / variance)


def partial_information(design: np.ndarray, target: np.ndarray, noise: np.ndarray | None) -> float:
    """How much of ``target`` survives projecting ``design`` out, over the noise.

    ``rho = ||(I - P) tau|| / ||(I - P) tau_noise||`` with ``P`` the orthogonal
    projector onto ``design``'s column space (`R5_03` Sec 2.1, statistic 2).
    This is the statistic that says what the torque channel carries *that the
    state channels do not*, which is the part an elastic model's ``tau_cmd``
    path can actually use -- and unlike an R^2 it is not dominated by the shared
    component.  Scaled by the noise's own unexplained part, so a value near 1
    means "nothing above the noise" whatever the units.

    ``None`` noise (a dataset without clean columns) gives NaN rather than an
    unscaled number that would not be comparable between datasets.
    """
    if noise is None:
        return float("nan")
    design = np.asarray(design, dtype=float)
    augmented = np.hstack([design, np.ones((len(design), 1))])

    def _residual(values: np.ndarray) -> float:
        values = np.asarray(values, dtype=float).reshape(-1)
        solution, *_ = np.linalg.lstsq(augmented, values, rcond=None)
        return float(np.linalg.norm(augmented @ solution - values))

    reference = _residual(noise)
    if reference <= 0.0:
        return float("nan")
    return float(_residual(target) / reference)


def _condition(matrix: np.ndarray) -> float:
    singular = np.linalg.svd(np.asarray(matrix, dtype=float), compute_uv=False)
    smallest = float(singular[-1])
    if not np.isfinite(smallest) or smallest <= 0.0:
        return float("inf")
    return float(singular[0] / smallest)


@dataclass(frozen=True)
class BagDiagnostics:
    """Q-C's five measurements for one bag."""

    bag: str
    tier: str
    backend: str
    samples: int
    # Q-C.1
    tau_on_regressor_r2: float
    tau_on_reference_r2: float
    tau_unexplained_fraction: float
    tau_partial_information: float
    tau_vs_reference_feedforward: float
    regressor_condition: float
    augmented_condition: float
    condition_inflation: float
    # Q-C.3
    target_rms: float
    rigid_nominal_residual_rms: float
    payload_mass: float
    differentiation_share: float
    link_friction_share: float
    noise_share: float
    # Q-C.4
    probe_band_acceleration: float
    probe_band_tau: float
    probe_band_deflection: float
    # T-2's linear baselines: what a least-squares fit already achieves, so a
    # trained network's number means something.
    baseline_rms_state: float
    baseline_rms_state_tau: float
    baseline_rms_tau: float
    baseline_rms_mean: float
    # Q-C.5
    deflection_rms: float
    deflection_over_noise: float
    tracking_rms: float
    # T-15: how much of the wrench the mapping could still carry.  NaN on a
    # dataset whose target is not an end-effector wrench, since there is no
    # mapping to lose information in.
    sigma_min_j_median: float
    sigma_min_j_fraction_below: float
    #: The Savitzky-Golay window this row was computed with (`R5_06` T-11).
    #: ``differentiation_share`` is not a property of the data alone -- it is
    #: what *this* consumer setting loses -- so the setting travels with it.
    sg_window: int

    def as_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


def bag_diagnostics(
    frame: pd.DataFrame,
    *,
    asset: Any,
    n_dof: int,
    time_step: float,
    probe_band_hz: tuple[float, float] = (3.0, 200.0),
    pin_model: tuple[Any, Any, Any] | None = None,
    sg_window: int = SG_WINDOW,
) -> BagDiagnostics:
    """Compute Q-C's measurements for one bag's rows.

    ``frame`` must be one bag (one ``bag`` value), already on the uniform
    output grid.  ``asset`` is the *bare* asset: the nominal model whose
    prediction the decomposition subtracts, i.e. the one a real identification
    would start from, payload excluded on purpose -- the payload's contribution
    is one of the terms being measured.
    """
    from . import identification as idn

    pin, model, data = idn.build_model(asset) if pin_model is None else pin_model

    q_motor = _column_block(frame, "q_motor", n_dof)
    q_link = _column_block(frame, "q_link", n_dof)
    dq_link = _column_block(frame, "dq_link", n_dof)
    dq_motor = _column_block(frame, "dq_motor", n_dof)
    tau_cmd = _column_block(frame, "tau", n_dof)
    target = _column_block(frame, "ft", n_dof)
    if q_link is None or tau_cmd is None or target is None:
        raise ValueError("frame is missing the q_link*/tau*/ft* columns a diagnostic needs")

    # The state the *target* lives on: the link side, whatever the dataset's
    # own input columns carry.  Decomposing the link-side torque against a
    # motor-side state would attribute the deflection to the model error.
    ddq_link = savitzky_golay_acceleration(dq_link, time_step, window=sg_window)
    clean_dq = _column_block(frame, "dq_link_clean", n_dof)
    clean_target = _column_block(frame, "tau_link_clean", n_dof)

    # -- T-15: posture, for a target that goes through J^T ------------------
    # Near a singularity the degenerate wrench directions map to near-zero
    # joint torque: the target stops carrying part of what the cell measured.
    # That is information loss, not a numerical failure, so it is scored per
    # bag rather than guarded against.  The threshold is a fraction of the
    # bag's own median, which makes it a statement about *this* excitation
    # rather than a unit-mixed absolute anyone would have to calibrate.
    sigma_min_median = float("nan")
    sigma_min_fraction = float("nan")
    if "sigma_min_j" in frame.columns:
        sigma = frame["sigma_min_j"].to_numpy(dtype=float)
        sigma_min_median = float(np.median(sigma))
        sigma_min_fraction = float(np.mean(sigma < SINGULARITY_FRACTION * sigma_min_median))

    # -- Q-C.1/2: collinearity and conditioning on the achieved motion -------
    regressor = idn.stack_regressor(pin, model, data, q_link, dq_link, ddq_link)
    basis = idn.base_parameter_basis(pin, model, data, n_samples=200, seed=0)
    projected = regressor @ basis
    tau_column = tau_cmd.reshape(-1, 1)
    r2 = fit_r2(projected, tau_column.reshape(-1))
    condition = _condition(projected)
    augmented = _condition(np.hstack([projected, tau_column]))

    # The *reference* regressor, which is the one `R5_00` Sec 5.3 is actually
    # about: "tau_cmd = n_hat of the reference, q ~ q_r".  Fitting tau_cmd on
    # the achieved state measures a physical identity that holds under every
    # controller (the motor equation, tau_cmd = ft + J thetadd + f), so it
    # cannot separate them; fitting it on the *designed* trajectory measures
    # how much of the command the reference alone predicts, which is exactly
    # the "the net can learn to copy the controller" failure mode.  NaN when
    # the dataset carries no reference columns (`signals.clean_columns`).
    reference_q = _column_block(frame, "q_ref_clean", n_dof)
    reference_dq = _column_block(frame, "dq_ref_clean", n_dof)
    reference_r2 = float("nan")
    if reference_q is not None and reference_dq is not None:
        reference_ddq = savitzky_golay_acceleration(reference_dq, time_step, window=sg_window)
        reference_regressor = idn.stack_regressor(
            pin, model, data, reference_q, reference_dq, reference_ddq,
        )
        reference_projected = reference_regressor @ basis
        reference_r2 = fit_r2(reference_projected, tau_column.reshape(-1))

    # The measure that actually separates the modes, where both R^2 above do
    # not (R5_02 Sec 6.3): the *unfitted* relative disagreement between the
    # commanded torque and the feedforward a model-based controller would have
    # computed from the reference alone.  The two R^2 statistics let a
    # least-squares fit absorb the difference into the regressor's 43-57
    # dimensions; this does not, so it reads "how far is tau_cmd from being a
    # function of the reference" directly.  The rotor inertia is part of the
    # motor-side feedforward and is read from the bag's own metadata columns;
    # without them (a rigid bag) it is zero, which is correct there.
    feedforward_gap = float("nan")
    if reference_q is not None and reference_dq is not None:
        rotor_columns = sorted(c for c in frame.columns if c.startswith("rotor_inertia__"))
        rotor = np.zeros(n_dof)
        if rotor_columns:
            values = frame[rotor_columns].to_numpy(dtype=float)[0]
            rotor = np.where(np.isfinite(values), values, 0.0)
        expected = np.asarray([
            idn.inverse_dynamics(pin, model, data, reference_q[i], reference_dq[i], reference_ddq[i])
            + rotor * reference_ddq[i]
            for i in range(len(reference_q))
        ])
        scale = float(np.sqrt(np.mean(expected**2)))
        if scale > 0.0:
            feedforward_gap = float(np.sqrt(np.mean((tau_cmd - expected) ** 2)) / scale)

    # `1 - R^2`, which is what `R5_03` Sec 2.1 statistic 1 asks to report: the
    # unexplained *fraction* separates modes on a log axis where the R^2 itself
    # does not (0.973 vs 0.959 is a 1.5x difference in 2.7 % vs 4.1 %).
    unexplained = float("nan") if np.isnan(reference_r2) else float(max(1.0 - reference_r2, 0.0))
    clean_tau = _column_block(frame, "tau_motor_clean", n_dof)
    tau_noise = None if clean_tau is None else (tau_cmd - clean_tau).reshape(-1, 1)
    partial = partial_information(projected, tau_column, tau_noise)

    # -- Q-C.3: what the nominal rigid model does not explain -----------------
    nominal = np.asarray([
        idn.inverse_dynamics(pin, model, data, q_link[i], dq_link[i], ddq_link[i])
        for i in range(len(q_link))
    ])
    residual = target - nominal
    target_rms = float(np.sqrt(np.mean(target**2)))
    residual_rms = float(np.sqrt(np.mean(residual**2)))

    # Differentiation loss: what the SG filter removed from the acceleration,
    # mapped through the nominal mass matrix.  This is the artefact R5_00
    # Sec 5.2 identifies and it depends on time, not on (q, dq), so no
    # residual r(q, dq) can represent it -- it is a floor on the comparison,
    # not something to learn.
    reference_dq = dq_link if clean_dq is None else clean_dq
    exact_ddq = np.gradient(reference_dq, time_step, axis=0)
    differentiation = np.asarray([
        np.asarray(pin.crba(model, data, q_link[i]), dtype=float) @ (exact_ddq[i] - ddq_link[i])
        for i in range(len(q_link))
    ])

    # Noise: what the sensor model added to the target, when the clean copy is
    # in the file.  Without it, noise cannot be separated from physics and the
    # share is reported as NaN rather than folded into the friction term.
    noise = None if clean_target is None else target - clean_target

    link_friction = None
    viscous_columns = [c for c in frame.columns if c.startswith("link_viscous__")]
    coulomb_columns = [c for c in frame.columns if c.startswith("link_coulomb__")]
    if viscous_columns and coulomb_columns:
        viscous = frame[sorted(viscous_columns)].to_numpy(dtype=float)[0]
        coulomb = frame[sorted(coulomb_columns)].to_numpy(dtype=float)[0]
        link_friction = viscous * dq_link + coulomb * np.tanh(dq_link / idn.COULOMB_EPSILON)

    def _share(term: np.ndarray | None) -> float:
        """Energy of one term relative to the unexplained residual's."""
        if term is None or residual_rms <= 0.0:
            return float("nan")
        return float(np.sqrt(np.mean(np.asarray(term, dtype=float) ** 2)) / residual_rms)

    # The payload term of the decomposition, reported as the *mass* the nominal
    # model does not carry rather than as a torque share: turning it into one
    # needs the payload's offset in the flange frame and a second Pinocchio
    # model built with it welded on, which `payload_asset` does and this
    # diagnostic deliberately does not rebuild.  Read it against
    # `rigid_nominal_residual_rms` alongside the other shares.
    payload_mass = float("nan")
    if "payload_mass" in frame.columns:
        mass = float(frame["payload_mass"].to_numpy(dtype=float)[0])
        payload_mass = 0.0 if np.isnan(mass) else mass

    # -- T-2: linear baselines ------------------------------------------------
    # `R5_03` T-2: "in round 4's data a 1-parameter linear fit beat the trained
    # residual class; without that baseline a comparison of two networks says
    # nothing about either".  Four fits of the *target*, in RMS of the same
    # units the target is in, so they sit directly beside a network's test RMSE:
    #   state      least squares on the rigid base regressor alone
    #   state+tau  the regressor plus the commanded torque -- what an elastic
    #              class has access to, linearly
    #   tau        the commanded torque alone, the one-column fit
    #   mean       predicting the target's own mean, the floor any model must beat
    target_column = target.reshape(-1)
    ones = np.ones((len(projected), 1))
    tau_only = np.hstack([tau_column, ones])

    def _fit_rms(design: np.ndarray) -> float:
        solution, *_ = np.linalg.lstsq(design, target_column, rcond=None)
        return float(np.sqrt(np.mean((design @ solution - target_column) ** 2)))

    baseline_state = _fit_rms(np.hstack([projected, ones]))
    baseline_state_tau = _fit_rms(np.hstack([projected, tau_column, ones]))
    baseline_tau = _fit_rms(tau_only)
    baseline_mean = float(np.sqrt(np.mean((target_column - target_column.mean()) ** 2)))

    # -- Q-C.4: probe-band content -------------------------------------------
    def _band_energy(values: np.ndarray | None) -> float:
        if values is None:
            return float("nan")
        spectrum = np.abs(np.fft.rfft(values - values.mean(axis=0), axis=0))
        frequency = np.fft.rfftfreq(len(values), d=time_step)
        band = (frequency >= probe_band_hz[0]) & (frequency <= probe_band_hz[1])
        if not band.any():
            return float("nan")
        total = float(np.sum(spectrum**2))
        return float("nan") if total <= 0.0 else float(np.sum(spectrum[band] ** 2) / total)

    deflection = _column_block(frame, "defl", n_dof)
    if deflection is None and q_motor is not None:
        deflection = q_motor - q_link

    # -- Q-C.5: is the deflection above the noise floor? ---------------------
    deflection_rms = float("nan") if deflection is None else float(np.sqrt(np.mean(deflection**2)))
    noise_rms = float("nan") if noise is None else float(np.sqrt(np.mean(noise**2)))
    deflection_over_noise = float("nan")
    if deflection is not None and np.isfinite(noise_rms) and noise_rms > 0.0:
        stiffness_columns = sorted(c for c in frame.columns if c.startswith("stiffness__"))
        if stiffness_columns:
            stiffness = frame[stiffness_columns].to_numpy(dtype=float)[0]
            # The elastic signature in torque units: what the spring's own
            # deflection contributes to the target, against the noise the
            # sensor model injected into the same channel.  Below one, the
            # dataset cannot see elasticity however it is modelled.
            with np.errstate(invalid="ignore"):
                signature = float(np.sqrt(np.mean((stiffness * deflection) ** 2)))
            deflection_over_noise = signature / noise_rms

    reference = _column_block(frame, "q_ref_clean", n_dof)
    tracking = float("nan")
    if reference is not None:
        tracking = float(np.sqrt(np.mean((q_link - reference) ** 2)))

    return BagDiagnostics(
        bag=str(frame["bag"].iloc[0]),
        tier=str(frame["tier"].iloc[0]) if "tier" in frame.columns else "",
        backend=str(frame["backend"].iloc[0]) if "backend" in frame.columns else "",
        samples=int(len(frame)),
        tau_on_regressor_r2=r2,
        tau_on_reference_r2=reference_r2,
        tau_unexplained_fraction=unexplained,
        tau_partial_information=partial,
        tau_vs_reference_feedforward=feedforward_gap,
        regressor_condition=condition,
        augmented_condition=augmented,
        condition_inflation=float(augmented / condition) if np.isfinite(condition) and condition > 0 else float("nan"),
        target_rms=target_rms,
        rigid_nominal_residual_rms=residual_rms,
        payload_mass=payload_mass,
        differentiation_share=_share(differentiation),
        link_friction_share=_share(link_friction),
        noise_share=_share(noise),
        probe_band_acceleration=_band_energy(ddq_link),
        probe_band_tau=_band_energy(tau_cmd),
        probe_band_deflection=_band_energy(deflection),
        baseline_rms_state=baseline_state,
        baseline_rms_state_tau=baseline_state_tau,
        baseline_rms_tau=baseline_tau,
        baseline_rms_mean=baseline_mean,
        deflection_rms=deflection_rms,
        deflection_over_noise=deflection_over_noise,
        tracking_rms=tracking,
        sg_window=int(sg_window),
        sigma_min_j_median=sigma_min_median,
        sigma_min_j_fraction_below=sigma_min_fraction,
    )


def dataset_diagnostics(
    frame: pd.DataFrame,
    manifest: Mapping[str, Any],
    asset: Any,
    *,
    bags: Sequence[str] | None = None,
    probe_band_hz: tuple[float, float] | None = None,
    sg_window: int | None = None,
) -> pd.DataFrame:
    """Run :func:`bag_diagnostics` over a dataset, one row per bag.

    The Pinocchio model is built once and reused: it is the same nominal model
    for every bag, and building it per bag dominated the runtime.

    The differentiation window comes from the dataset's own
    ``differentiation.sg_window`` (`R5_06` T-11): quoting a
    ``differentiation_share`` computed with the historical fixed 11 samples
    against a dataset whose contract recommends another window mixes a
    property of the data with a consumer setting, which is exactly the caveat
    `R5_05` Sec 6.5 attached to its own numbers.  A caller may still override
    it -- "what would today's consumer see?" is a legitimate question -- and
    the window used is written into every row.
    """
    from . import identification as idn

    n_dof = int(manifest.get("n_dof") or len(asset.joint_names))
    time_step = float(manifest.get("sample_time_step", 0.002))
    if probe_band_hz is None:
        top = max(
            [float(record.get("exc_probe_top_hz") or 0.0) for record in manifest.get("records", [])] or [0.0]
        )
        probe_band_hz = (3.0, top if top > 0.0 else 0.5 / time_step)
    if sg_window is None:
        declared = (manifest.get("differentiation") or {}).get("sg_window")
        sg_window = int(declared) if declared else SG_WINDOW
    pin_model = idn.build_model(asset)
    selected = frame if bags is None else frame[frame["bag"].isin(list(bags))]
    rows = [
        bag_diagnostics(
            group.reset_index(drop=True), asset=asset, n_dof=n_dof, time_step=time_step,
            probe_band_hz=probe_band_hz, pin_model=pin_model, sg_window=int(sg_window),
        ).as_dict()
        for _, group in selected.groupby("bag", sort=False)
    ]
    diagnostics = pd.DataFrame(rows)
    controller = (manifest.get("controller") or {}).get("mode", "unknown")
    diagnostics.insert(1, "controller_mode", controller)
    return diagnostics


def summarize_diagnostics(diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Median of every measurement per ``(controller_mode, tier kind)``.

    Tiers are grouped into ``rigid`` and ``elastic`` rather than kept per
    robot: the question is what a *controller* does to the data, and twenty
    sampled robots under one controller are twenty samples of one answer.
    """
    grouped = diagnostics.copy()
    grouped["tier_kind"] = np.where(grouped["tier"].eq("rigid"), "rigid", "elastic")
    numeric = grouped.select_dtypes(include="number").columns
    keys = ["controller_mode", "tier_kind"]
    # The plant_extras-off control row is a separate condition, not a sample of
    # the same one (R5_03 Sec 2.2), so it never gets averaged into its own
    # comparison.
    if "plant_extras" in grouped.columns:
        keys.append("plant_extras")
    return grouped.groupby(keys, sort=True)[list(numeric)].median().reset_index()


def load_dataset(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read a written dataset and its manifest sidecar."""
    path = Path(path).expanduser().resolve()
    frame = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    manifest_path = path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    return frame, manifest
