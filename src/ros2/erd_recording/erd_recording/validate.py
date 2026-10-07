"""Dataset validation checks (RR_01 S8.3). Errors block `valid`; warnings don't.

Each ``check_*`` function is independently unit-testable on plain arrays/
frames. :func:`validate_dataset` assembles whichever of them the caller has
inputs for -- some (session health, identification freshness, reference
tracking) need raw run telemetry that only exists once ``run`` has recorded
something, and are reported ``"not_available"`` rather than skipped silently
when that telemetry isn't supplied.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


def _block(frame: pd.DataFrame, prefix: str, n: int) -> np.ndarray:
    return frame[[f"{prefix}{i}" for i in range(n)]].to_numpy(dtype=float)


def check_uniform_grid(frame: pd.DataFrame, *, nominal_dt: float, rtol: float = 0.01) -> dict[str, Any]:
    """Robot-clock ``t``: ``Delta t`` within ``rtol`` of nominal, no gaps inside any bag."""
    worst_ratio = 0.0
    offending: list[str] = []
    for bag, group in frame.groupby("bag", sort=False):
        t = group["t"].to_numpy(dtype=float)
        dt = np.diff(t)
        if len(dt) == 0:
            continue
        ratio = float(np.max(np.abs(dt - nominal_dt)) / nominal_dt)
        worst_ratio = max(worst_ratio, ratio)
        if ratio > rtol:
            offending.append(str(bag))
    return {"check": "uniform grid", "ok": not offending, "nominal_dt": nominal_dt,
            "worst_relative_deviation": worst_ratio, "offending_bags": offending}


def check_no_nan_and_ranges(
    frame: pd.DataFrame, *, n_dof: int, position_lower: np.ndarray, position_upper: np.ndarray,
) -> dict[str, Any]:
    """No NaN anywhere; ``q`` inside ``limits.position``."""
    columns = [c for c in frame.columns if c not in ("bag", "split")]
    nan_columns = [c for c in columns if frame[c].isna().any()]
    q = _block(frame, "q", n_dof)
    below = q < (position_lower[None, :] - 1e-9)
    above = q > (position_upper[None, :] + 1e-9)
    out_of_range = int(np.sum(below | above))
    return {"check": "units and ranges", "ok": not nan_columns and out_of_range == 0,
            "nan_columns": nan_columns, "position_out_of_range_samples": out_of_range}


#: quantization-multiple threshold and correlation ceiling for `tau != ft` (RR_01 S8.3).
TAU_FT_QUANTIZATION_MULTIPLE = 10.0
TAU_FT_CORRELATION_ERROR = 0.9999
TAU_FT_CORRELATION_WARN = 0.99


def check_tau_ne_ft(
    frame: pd.DataFrame, *, n_dof: int, quantization: float, tau_source: str, ft_source: str,
) -> dict[str, Any]:
    """``tau`` and ``ft`` must be declared from different sources and not be
    numerically identical (RR_01 S8.3, "Never again" RR_00 S2, the I-9 defect)."""
    tau = _block(frame, "tau", n_dof)
    ft = _block(frame, "ft", n_dof)
    diff = np.abs(tau - ft)
    max_abs_diff = float(np.max(diff))
    correlation = np.array([
        float(np.corrcoef(tau[:, j], ft[:, j])[0, 1]) if np.std(tau[:, j]) > 0 and np.std(ft[:, j]) > 0 else 0.0
        for j in range(n_dof)
    ])
    same_source = tau_source == ft_source
    numerically_identical = max_abs_diff <= TAU_FT_QUANTIZATION_MULTIPLE * quantization or bool(
        np.any(correlation >= TAU_FT_CORRELATION_ERROR)
    )
    ok = not same_source and not numerically_identical
    warn = ok and bool(np.any(correlation >= TAU_FT_CORRELATION_WARN))
    return {"check": "tau != ft", "ok": ok, "warning": warn, "same_declared_source": same_source,
            "max_abs_difference": max_abs_diff, "correlation_per_joint": correlation.tolist(),
            "tau_source": tau_source, "ft_source": ft_source}


def check_saturation(
    frame: pd.DataFrame, *, n_dof: int, effort_limit: np.ndarray, fraction: float = 0.95,
) -> dict[str, Any]:
    tau = _block(frame, "tau", n_dof)
    fraction_of_limit = np.abs(tau) / effort_limit[None, :]
    worst = float(np.max(fraction_of_limit))
    ok = worst < fraction
    return {"check": "saturation", "ok": ok, "worst_fraction_of_limit": worst, "limit": fraction}


def check_digests(bag_digests: Mapping[str, str], plan_digests: Mapping[str, str]) -> dict[str, Any]:
    """Every recorded ``bag`` maps to a frozen plan segment digest."""
    missing = [bag for bag, digest in bag_digests.items() if plan_digests.get(bag) != digest]
    return {"check": "digests", "ok": not missing, "mismatched_or_missing_bags": missing}


def check_sign_convention(
    measured: np.ndarray, gravity_torque: np.ndarray, *, slope_tolerance: float = 0.1,
    testability: Sequence[Mapping[str, Any]] | None = None, joint_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Standstill ``measured`` (iiwa ``ft``, UR ``K_tau i_act``) vs Pinocchio
    gravity torque: slope ``+1 +/- tol`` per joint (RR_01 S8.3).

    RR_12 B-3: only joints whose gravity range across the poses supports a
    slope are judged (``testability``, from ``identification.json``); the rest
    are listed as ``not testable at these poses``. With no testable joint
    the check cannot pass."""
    n_dof = measured.shape[1]
    names = list(joint_names) if joint_names is not None else [str(j) for j in range(n_dof)]
    joints = []
    for j in range(n_dof):
        design = np.column_stack([gravity_torque[:, j], np.ones(len(gravity_torque))])
        (a, _b), *_ = np.linalg.lstsq(design, measured[:, j], rcond=None)
        entry = {"joint": names[j], "slope": float(a)}
        if testability is not None:
            entry["testable"] = bool(testability[j]["testable"])
            entry["reason"] = testability[j].get("reason")
        else:
            entry["testable"] = True
        entry["ok"] = bool(abs(a - 1.0) <= slope_tolerance) if entry["testable"] else None
        joints.append(entry)
    judged = [e for e in joints if e["testable"]]
    ok = bool(judged) and all(e["ok"] for e in judged)
    return {"check": "sign convention", "ok": ok, "joints": joints, "tolerance": slope_tolerance,
            "testable_joints": [e["joint"] for e in judged],
            "note": None if judged else "no joint is testable at these standstill poses (RR_12 B-3)"}


def quintic_hermite(time: np.ndarray, position: np.ndarray, velocity: np.ndarray, acceleration: np.ndarray,
                    query: np.ndarray) -> np.ndarray:
    """The JTC's ``splines`` interpolation: a quintic per interval matching
    position, velocity and acceleration at both ends. ``query`` is clipped
    to the trajectory's span (the JTC holds the last point afterwards)."""
    query = np.clip(np.asarray(query, dtype=float), time[0], time[-1])
    index = np.clip(np.searchsorted(time, query, side="right") - 1, 0, len(time) - 2)
    t0, t1 = time[index], time[index + 1]
    h = (t1 - t0)[:, None]
    s = ((query - t0) / (t1 - t0))[:, None]
    p0, p1 = position[index], position[index + 1]
    v0, v1 = velocity[index] * h, velocity[index + 1] * h
    a0, a1 = acceleration[index] * h**2, acceleration[index + 1] * h**2
    s2, s3, s4, s5 = s**2, s**3, s**4, s**5
    h00 = 1 - 10 * s3 + 15 * s4 - 6 * s5
    h10 = s - 6 * s3 + 8 * s4 - 3 * s5
    h20 = 0.5 * s2 - 1.5 * s3 + 1.5 * s4 - 0.5 * s5
    h01 = 10 * s3 - 15 * s4 + 6 * s5
    h11 = -4 * s3 + 7 * s4 - 3 * s5
    h21 = 0.5 * s3 - s4 + 0.5 * s5
    return h00 * p0 + h10 * v0 + h20 * a0 + h01 * p1 + h11 * v1 + h21 * a1


def fit_start_time(stamps_s: np.ndarray, values: np.ndarray, plan_time: np.ndarray, plan_position: np.ndarray,
                   plan_velocity: np.ndarray, plan_acceleration: np.ndarray, *, search_s: float = 0.5,
                   timing_slack_s: float = 0.0) -> dict[str, Any]:
    """Executed reference vs plan.

    Fits the one start time ``t0`` that maps the samples onto the plan's
    spline (``values ~ spline(stamps - t0)``) by bounded least squares.

    RR_13: recorded stamps are not the instant the controller sampled its
    trajectory (the JTC samples at its own clock read; ``controller_state``'s
    stamp comes microseconds later, and the FRI command is one cycle behind).
    A stamp error ``d`` shows up as ``velocity * d``: about 1e-6 rad at
    0.6 rad/s and 2 us, three orders above a 1e-9 rad tolerance. So each
    sample's residual is split into a timing part (along the plan velocity,
    ``|d| <= timing_slack_s``) and the remaining **path** error, which is what
    ``max_abs_error`` reports. ``max_abs_error_at_stamps`` keeps the raw value.
    """
    from scipy.optimize import minimize_scalar

    rel = plan_time - plan_time[0]

    def residual(t0: float) -> tuple[np.ndarray, np.ndarray]:
        mask = (stamps_s >= t0) & (stamps_s <= t0 + rel[-1])
        if mask.sum() < 10:
            return np.full((1, values.shape[1]), np.inf), mask
        return values[mask] - quintic_hermite(rel, plan_position, plan_velocity, plan_acceleration,
                                              stamps_s[mask] - t0), mask

    def cost(t0: float) -> float:
        r, _ = residual(t0)
        return float(np.mean(r**2)) if np.isfinite(r).all() else 1e9

    grid = np.linspace(stamps_s[0] - 0.05, stamps_s[0] + search_s, 551)
    coarse = grid[int(np.argmin([cost(g) for g in grid]))]
    step = grid[1] - grid[0]
    best = minimize_scalar(cost, bounds=(coarse - step, coarse + step), method="bounded",
                           options={"xatol": 1e-9})
    r, mask = residual(best.x)
    velocity = quintic_hermite(rel, plan_velocity, plan_acceleration, np.zeros_like(plan_acceleration),
                               stamps_s[mask] - best.x)
    speed2 = np.sum(velocity**2, axis=1)
    # residual ~ -velocity * d  =>  least-squares d per sample, bounded by the slack.
    delay = np.where(speed2 > 1e-12, -np.sum(r * velocity, axis=1) / np.where(speed2 > 1e-12, speed2, 1.0), 0.0)
    delay = np.clip(delay, -timing_slack_s, timing_slack_s)
    path = r + velocity * delay[:, None]
    implied = np.abs(delay) * 1e6
    return {"start_s": float(best.x), "max_abs_error": float(np.max(np.abs(path))),
            "max_abs_error_at_stamps": float(np.max(np.abs(r))), "timing_slack_s": timing_slack_s,
            "implied_time_error_us": {"p50": float(np.percentile(implied, 50)), "p99": float(np.percentile(implied, 99)),
                                      "max": float(implied.max())},
            "rms_per_joint": np.sqrt(np.mean(path**2, axis=0)).tolist(), "samples": int(len(r))}


def check_reference_tracking(
    segments: Mapping[str, Mapping[str, Any]], *, jtc_tolerance: float = 1e-9,
    commanded_tolerance: float | None = 1e-6, commanded_report_only: bool = False,
) -> dict[str, Any]:
    """RR_01 S8.3 reference tracking, per dataset segment: the executed JTC
    reference against the frozen plan (``jtc``), the drive-side command
    against the plan (``commanded``: iiwa FRI ``commanded_position``, gated
    at 1e-6 rad; UR ``target_q``, reported only), and RMS(q - reference) per
    joint. Each entry carries ``fit_start_time`` results; the tolerances
    apply to the path error (see there for the stamp-timing split)."""
    problems = []
    for sid, entry in segments.items():
        jtc = entry.get("jtc")
        if jtc is not None and jtc["max_abs_error"] > jtc_tolerance:
            problems.append(f"{sid}: JTC reference - plan = {jtc['max_abs_error']:.3g} rad > {jtc_tolerance:g}")
        commanded = entry.get("commanded")
        if (commanded is not None and not commanded_report_only and commanded_tolerance is not None
                and commanded["max_abs_error"] > commanded_tolerance):
            problems.append(f"{sid}: commanded - plan = {commanded['max_abs_error']:.3g} rad > {commanded_tolerance:g}")
    return {"check": "reference tracking", "ok": bool(segments) and not problems, "segments": dict(segments),
            "jtc_tolerance": jtc_tolerance, "commanded_tolerance": commanded_tolerance,
            "commanded_report_only": commanded_report_only, "problems": problems}


def check_session_health_iiwa(windows: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    """RR_01 S8.3: FRI COMMANDING_ACTIVE (4) and command mode POSITION (1)
    on every sample of every dataset segment."""
    segments, problems = {}, []
    for sid, frame in windows.items():
        state = frame["fri_session_state"].to_numpy(dtype=float)
        mode = frame["fri_command_mode"].to_numpy(dtype=float)
        bad_state, bad_mode = int(np.sum(state != 4)), int(np.sum(mode != 1))
        segments[sid] = {"samples": int(len(frame)), "not_commanding_active": bad_state, "not_position_mode": bad_mode}
        if bad_state or bad_mode:
            problems.append(sid)
    return {"check": "session health", "ok": bool(windows) and not problems, "segments": segments,
            "rule": "fri session_state == COMMANDING_ACTIVE and command_mode == POSITION on every sample"}


def check_session_health_ur(windows: Mapping[str, pd.DataFrame], *, floor: float = 0.999) -> dict[str, Any]:
    """RR_01 S8.3: speed scaling x target speed fraction = 1, robot mode
    RUNNING (7), safety status NORMAL (1, so no reduced mode or protective
    stop), from the RTDE stream, on every sample of every dataset segment."""
    segments, problems = {}, []
    for sid, frame in windows.items():
        scaling = frame["speed_scaling"].to_numpy(dtype=float) * frame["target_speed_fraction"].to_numpy(dtype=float)
        entry = {"samples": int(len(frame)), "min_speed_scaling": float(np.min(scaling)),
                 "not_running": int(np.sum(frame["robot_mode"].to_numpy() != 7)),
                 "safety_not_normal": int(np.sum(frame["safety_status"].to_numpy() != 1))}
        segments[sid] = entry
        if entry["min_speed_scaling"] < floor or entry["not_running"] or entry["safety_not_normal"]:
            problems.append(sid)
    return {"check": "session health", "ok": bool(windows) and not problems, "segments": segments,
            "rule": f"speed_scaling * target_speed_fraction >= {floor}, robot_mode RUNNING, safety_status NORMAL"}


#: RR_01 S7.1: percentiles sent to the simulation side.
ENVELOPE_PERCENTILES = (50.0, 90.0, 99.0, 99.9, 100.0)


def check_envelopes(frame: pd.DataFrame, *, n_dof: int, effort_limit: np.ndarray, sg_window: int, sg_poly: int,
                    time_step: float) -> dict[str, Any]:
    """RR_01 S7.1: per joint, percentiles of |dq|, |ddq| (SG of dq with the
    contract's window, per bag) and |tau_cmd| / effort over the dataset."""
    from scipy.signal import savgol_filter

    dq = _block(frame, "dq", n_dof)
    ddq = np.empty_like(dq)
    for _, group in frame.groupby("bag", sort=False):
        index = group.index.to_numpy()
        ddq[index] = savgol_filter(dq[index], int(sg_window), int(sg_poly), deriv=1, delta=time_step, axis=0,
                                   mode="interp")
    tau = np.abs(_block(frame, "tau", n_dof)) / effort_limit[None, :]
    table = {}
    for name, values in (("abs_dq_rad_s", np.abs(dq)), ("abs_ddq_rad_s2", np.abs(ddq)), ("abs_tau_over_effort", tau)):
        table[name] = {f"p{q:g}": np.percentile(values, q, axis=0).tolist() for q in ENVELOPE_PERCENTILES}
    finite = all(np.isfinite(v).all() for v in (dq, ddq, tau))
    return {"check": "envelopes", "ok": finite, "percentiles": table, "samples": int(len(frame)),
            "statistic": "per-joint percentiles over every dataset row (RR_01 S7.1)"}


def noise_table(
    standstill: Sequence[pd.DataFrame], production: pd.DataFrame, *, n_dof: int,
    channels: Sequence[str] = ("q", "dq", "tau", "ft"), ft_offset: Sequence[float] | None = None,
) -> dict[str, Any]:
    """RR_12 S5.1 per joint and channel: standstill sigma (linear-detrended
    per hold, averaged over holds), the observed quantisation step (smallest
    non-zero |delta|), sigma relative to the production excitation (sigma /
    RMS(x - mean x) over the dataset's bags) and, iiwa, the ft offset
    against the gravity model."""
    table: dict[str, Any] = {}
    for prefix in channels:
        sigmas, steps = [], []
        for frame in standstill:
            values = _block(frame, prefix, n_dof)
            t = np.arange(len(values), dtype=float)
            design = np.column_stack([t, np.ones_like(t)])
            coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
            sigmas.append(np.std(values - design @ coefficients, axis=0))
            delta = np.abs(np.diff(values, axis=0))
            steps.append(np.asarray([np.min(d[d > 0]) if np.any(d > 0) else np.nan for d in delta.T]))
        sigma = np.mean(sigmas, axis=0)
        excitation = _block(production, prefix, n_dof)
        rms = np.sqrt(np.mean((excitation - excitation.mean(axis=0)) ** 2, axis=0))
        table[prefix] = {"sigma": sigma.tolist(), "quantisation_step": np.nanmin(steps, axis=0).tolist(),
                         "sigma_over_production_rms": (sigma / np.where(rms > 0, rms, np.nan)).tolist()}
    if ft_offset is not None:
        table["ft"]["offset_vs_gravity_nm"] = list(map(float, ft_offset))
    return table


def check_noise(standstill: Sequence[pd.DataFrame], production: pd.DataFrame, *, n_dof: int,
                ft_offset: Sequence[float] | None = None, source: str = "") -> dict[str, Any]:
    """RR_01 S8.3 noise (Q-6), in RR_12 S5.1's format."""
    if not standstill:
        return {"check": "noise", "ok": None, "status": "not_available", "note": "no standstill holds in the run"}
    table = noise_table(standstill, production, n_dof=n_dof, ft_offset=ft_offset)
    finite = all(np.isfinite(np.asarray(v["sigma"], dtype=float)).all() for v in table.values())
    return {"check": "noise", "ok": finite, "holds": len(standstill), "channels": table, "source": source,
            "statistic": "sigma: linear-detrended per hold, mean over holds; relative: / RMS(x - mean x) over the dataset"}


def check_identification_freshness_ur(
    identification: Mapping[str, Any], *, run_config_digest: str, run_temperature_c: np.ndarray,
    tolerance_c: float = 5.0,
) -> dict[str, Any]:
    """RR_01 S8.3 (UR): the identification belongs to this session (same run
    folder and frozen config, RR_12 S2: no reuse across sessions) and every
    joint's temperature during the dataset is within +-5 C of the
    temperature during the identification sweeps."""
    sweeps = identification.get("friction_sweeps") or []
    n_dof = len(run_temperature_c)
    reference = np.full(n_dof, np.nan)
    for j in range(n_dof):
        values = [0.5 * (s["temperature_before_c"] + s["temperature_after_c"]) for s in sweeps
                  if s["segment"].startswith(f"identify_sweep_{j}_")]
        if values:
            reference[j] = float(np.mean(values))
    same_session = identification.get("config_digest") == run_config_digest
    delta = np.asarray(run_temperature_c, dtype=float) - reference
    ok = bool(same_session and np.isfinite(delta).all() and np.all(np.abs(delta) <= tolerance_c))
    return {"check": "identification freshness", "ok": ok, "same_session": same_session,
            "identification_temperature_c": reference.tolist(), "run_temperature_c": list(map(float, run_temperature_c)),
            "delta_c": delta.tolist(), "tolerance_c": tolerance_c}


#: RR_12 C-2c: the only (check, robot) pairs allowed to be `not_applicable`
#: on real hardware, each with its reason.
NOT_APPLICABLE_ALLOWED = {
    ("identification freshness", "kuka_lbr_iiwa_14_r820"):
        "the iiwa target is the raw joint-torque sensor: no identified model enters the dataset",
}


def not_applicable(check: str, robot: str) -> dict[str, Any]:
    reason = NOT_APPLICABLE_ALLOWED[(check, robot)]
    return {"check": check, "ok": None, "status": "not_applicable", "reason": reason}


_NOT_AVAILABLE = {"ok": None, "status": "not_available", "note": "not available: needs raw run telemetry not supplied"}


#: RR_14 P-2b: the checks a simulator cannot pass by construction, per
#: ``hardware`` value. A listed failure stays ``ok: false`` with
#: ``simulator_limit: <reason>`` and doesn't fail the stage. Never consulted
#: when ``hardware: real``; ``emulator`` has no entry (its plant produces
#: independent torques, so every check must pass on it).
SIMULATOR_LIMITS = {
    ("mock", "tau != ft"):
        "mock hardware has no torque channels: tau and ft are not independent measurements",
    ("mock", "sign convention"):
        "mock hardware has no torque channels: there is no measured gravity load",
    # RR_15 addition (UR10 mock had never reached convert before): the check
    # compares joint temperatures, which mock hardware doesn't have.
    ("mock", "identification freshness"):
        "mock hardware has no joint temperatures (and no identification fit)",
    ("ursim", "tau != ft"):
        "URSim is pure feed-forward: the reported current is the model's own torque, so the current proxy "
        "tracks tau (wrist 2 correlation 0.99998, rr13_ur10_all_b)",
}


def simulator_limit(check: str, hardware: str) -> str | None:
    """The listed reason a simulator fails ``check``; ``None`` on real hardware."""
    if hardware == "real":
        return None
    return SIMULATOR_LIMITS.get((hardware, check))


def finalize_checks(checks: Sequence[Mapping[str, Any]], *, hardware: str, robot: str) -> dict[str, Any]:
    """RR_12 C-2c: on ``hardware: real`` a ``null`` check is an error; only a
    pair in :data:`NOT_APPLICABLE_ALLOWED` may stay ``not_applicable``.
    Off real hardware nulls stay visible but don't fail, and a failure on
    :data:`SIMULATOR_LIMITS` is marked, not counted (RR_14 P-2b)."""
    final = []
    for check in checks:
        entry = dict(check)
        if entry.get("ok") is None:
            allowed = (entry.get("status") == "not_applicable"
                       and (entry.get("check"), robot) in NOT_APPLICABLE_ALLOWED)
            if hardware == "real" and not allowed:
                entry["ok"] = False
                entry["error"] = "null check on real hardware (RR_12 C-2c)"
        elif entry.get("ok") is False:
            reason = simulator_limit(entry.get("check"), hardware)
            if reason is not None:
                entry["simulator_limit"] = reason
        final.append(entry)
    errors = [c["check"] for c in final if c.get("ok") is False and "simulator_limit" not in c]
    return {"ok": not errors, "errors": errors, "checks": final,
            "simulator_limits": [c["check"] for c in final if "simulator_limit" in c],
            "nulls": [c["check"] for c in final if c.get("ok") is None and not (
                c.get("status") == "not_applicable" and (c.get("check"), robot) in NOT_APPLICABLE_ALLOWED)],
            "not_applicable": [c["check"] for c in final if c.get("status") == "not_applicable"]}


def validate_dataset(
    frame: pd.DataFrame, *, n_dof: int, nominal_dt: float, position_lower: np.ndarray, position_upper: np.ndarray,
    quantization: float, tau_source: str, ft_source: str, effort_limit: np.ndarray,
    bag_digests: Mapping[str, str] | None = None, plan_digests: Mapping[str, str] | None = None,
    session_health: Mapping[str, Any] | None = None, identification_freshness: Mapping[str, Any] | None = None,
    reference_tracking: Mapping[str, Any] | None = None, envelopes: Mapping[str, Any] | None = None,
    noise: Mapping[str, Any] | None = None, sign_convention: Mapping[str, Any] | None = None,
    hardware: str = "mock", robot: str = "",
) -> dict[str, Any]:
    """Assemble every S8.3 check into one ``validation.json``-shaped report."""
    checks = [
        check_uniform_grid(frame, nominal_dt=nominal_dt),
        check_no_nan_and_ranges(frame, n_dof=n_dof, position_lower=position_lower, position_upper=position_upper),
        check_tau_ne_ft(frame, n_dof=n_dof, quantization=quantization, tau_source=tau_source, ft_source=ft_source),
        check_saturation(frame, n_dof=n_dof, effort_limit=effort_limit),
    ]
    if bag_digests is not None and plan_digests is not None:
        checks.append(check_digests(bag_digests, plan_digests))
    else:
        checks.append({"check": "digests", **_NOT_AVAILABLE})
    for name, value in (("sign convention", sign_convention), ("session health", session_health),
                        ("identification freshness", identification_freshness),
                        ("reference tracking", reference_tracking), ("envelopes", envelopes), ("noise", noise)):
        checks.append({"check": name, **(dict(value) if value is not None else _NOT_AVAILABLE)})
    return finalize_checks(checks, hardware=hardware, robot=robot)
