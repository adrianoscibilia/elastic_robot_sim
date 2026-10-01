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
) -> dict[str, Any]:
    """Standstill ``measured`` vs Pinocchio gravity torque: slope ``+1 +/- tol``
    per joint (RR_01 S8.3; E-iiwa-2 / the UR standstill sanity check)."""
    n_dof = measured.shape[1]
    slope = np.zeros(n_dof)
    for j in range(n_dof):
        design = np.column_stack([gravity_torque[:, j], np.ones(len(gravity_torque))])
        (a, _b), *_ = np.linalg.lstsq(design, measured[:, j], rcond=None)
        slope[j] = a
    ok = bool(np.all(np.abs(slope - 1.0) <= slope_tolerance))
    return {"check": "sign convention", "ok": ok, "slope_per_joint": slope.tolist(), "tolerance": slope_tolerance}


def check_reference_tracking(
    reference: np.ndarray, plan: np.ndarray, *, tolerance: float, report_only: bool = False,
) -> dict[str, Any]:
    """Executed reference vs the frozen plan, at the sample times."""
    error = np.max(np.abs(reference - plan))
    rms = float(np.sqrt(np.mean((reference - plan) ** 2, axis=0)).max())
    ok = report_only or error <= tolerance
    return {"check": "reference tracking", "ok": ok, "max_abs_error": float(error), "worst_joint_rms": rms,
            "tolerance": tolerance, "report_only": report_only}


def check_noise(standstill_frames: Sequence[pd.DataFrame], *, n_dof: int, channel_prefixes: Sequence[str]) -> dict[str, Any]:
    """Standstill sigma per channel and joint (RR_01 S8.3, Q-6)."""
    sigmas: dict[str, list[float]] = {}
    for prefix in channel_prefixes:
        values = []
        for frame in standstill_frames:
            block = _block(frame, prefix, n_dof)
            values.append(np.std(block, axis=0))
        sigmas[prefix] = np.mean(np.vstack(values), axis=0).tolist() if values else [float("nan")] * n_dof
    return {"check": "noise", "ok": True, "sigma_per_channel": sigmas}


_NOT_AVAILABLE = {"ok": None, "note": "not available: needs raw run telemetry not supplied"}


def validate_dataset(
    frame: pd.DataFrame, *, n_dof: int, nominal_dt: float, position_lower: np.ndarray, position_upper: np.ndarray,
    quantization: float, tau_source: str, ft_source: str, effort_limit: np.ndarray,
    bag_digests: Mapping[str, str] | None = None, plan_digests: Mapping[str, str] | None = None,
    session_health: Mapping[str, Any] | None = None, identification_freshness: Mapping[str, Any] | None = None,
    reference_tracking: Mapping[str, Any] | None = None, envelopes: Mapping[str, Any] | None = None,
    noise: Mapping[str, Any] | None = None,
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
    checks.append({"check": "session health", **(session_health or _NOT_AVAILABLE)})
    checks.append({"check": "identification freshness", **(identification_freshness or _NOT_AVAILABLE)})
    checks.append({"check": "reference tracking", **(reference_tracking or _NOT_AVAILABLE)})
    checks.append({"check": "envelopes", **(envelopes or _NOT_AVAILABLE)})
    checks.append({"check": "noise", **(noise or _NOT_AVAILABLE)})
    errors = [c for c in checks if c.get("ok") is False]
    return {"ok": not errors, "checks": checks}
