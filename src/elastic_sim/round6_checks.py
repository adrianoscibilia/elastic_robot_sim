"""Round-6 controller-influence gates and per-file hard checks (`R6_00` Secs 7, 9).

Everything here reads a *written* dataset -- its frame and manifest -- so the
build script, the launcher's doctor and the tests run the same code on the
same bytes.  Each check returns a record with ``ok`` and the numbers behind
it; :func:`check_dataset` gathers them and a production build that fails any
is refused.

Gates (Sec 7):

1. **feedforward share = 0**, structural: the manifest says the controller has
   no feedforward, and no bag recorded a non-zero one;
2. **sag**: achieved-path RMS deviation <= ``sag_max_fraction`` of each
   joint's excitation span, on every bag and joint (Sec 2.2);
3. **Q-C v2** (``1 - R^2`` of ``tau_cmd`` on the reference regressor, the
   statistic `R5_08` Sec 9 tabulates) at or above round 5's ``pd`` level on
   the same platform, per tier kind -- recorded, not failed, where round 5 has
   no level;
4. **gain shift**: the gain-shift file holds the test split's robots and
   trajectories (same digests) and every bag's omega outside the training
   band.  The model-side acceptance (RMSE ratio <= 1.3) is the consumer's.

Hard checks (Sec 9): one backend; unstable bags <= 2 % and listed;
``ft_clean == tau_link_clean`` exactly; the recorded noise matches the
``noise`` block within sampling error.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


def _block(frame: pd.DataFrame, prefix: str, n: int) -> np.ndarray:
    return frame[[f"{prefix}{i}" for i in range(n)]].to_numpy(dtype=float)


def _joint_value(frame: pd.DataFrame, prefix: str, names: Sequence[str]) -> np.ndarray:
    return np.asarray([float(frame[f"{prefix}{name}"].iloc[0]) for name in names])


# ---------------------------------------------------------------------------
# Sec 9 hard checks
# ---------------------------------------------------------------------------

def check_one_backend(frame: pd.DataFrame) -> dict[str, Any]:
    backends = sorted(set(frame["backend"])) if "backend" in frame.columns else []
    return {"check": "one backend", "ok": len(backends) == 1, "backends": backends}


def check_unstable(manifest: Mapping[str, Any], max_fraction: float) -> dict[str, Any]:
    unstable = list(manifest.get("unstable_bags") or [])
    total = len(unstable) + int(manifest.get("n_bags", 0))
    fraction = len(unstable) / total if total else 0.0
    return {"check": "unstable bags", "ok": fraction <= max_fraction, "n_unstable": len(unstable),
            "n_bags": total, "fraction": fraction, "limit": max_fraction,
            "listed": [u.get("bag") for u in unstable]}


def check_clean_target(frame: pd.DataFrame, n: int) -> dict[str, Any]:
    ft_clean = _block(frame, "ft_clean", n)
    tau_link = _block(frame, "tau_link_clean", n)
    identical = bool(np.array_equal(ft_clean, tau_link))
    return {"check": "ft_clean == tau_link_clean", "ok": identical,
            "max_abs_difference": float(np.max(np.abs(ft_clean - tau_link)))}


def _delayed(values: np.ndarray, delay: int) -> np.ndarray:
    if delay == 0:
        return values
    out = np.empty_like(values)
    out[:delay] = values[0]
    out[delay:] = values[:-delay]
    return out


def _quantization(noise: Mapping[str, Any], channel: str, n: int) -> np.ndarray:
    values = np.atleast_1d(np.asarray((noise.get(channel) or {}).get("quantization", [0.0]), dtype=float))
    return np.broadcast_to(values, (n,)).copy()


def noise_statistics(frame: pd.DataFrame, manifest: Mapping[str, Any]) -> pd.DataFrame:
    """Per bag, channel and joint: the realized noise against the declared one.

    ``realized`` is the std of ``recorded - model(clean)`` after the delay
    (and, for ``ft``, the fitted gain and offset); ``expected`` is the
    declared ``sigma`` (times the gain for ``ft``) combined with the
    quantization's uniform ``step / sqrt(12)``.  ``sigma_over_basis`` is the
    recorded ``sigma`` over the RMS it was scaled from; ``declared_sigma`` is
    what the block says it must be, ``p * RMS (+) sigma (+) sigma_quanta * step``.
    """
    noise = manifest.get("noise") or {}
    names = list(manifest["joint_names"])
    n = len(names)
    delay = int(noise.get("delay_samples", 0)) if noise.get("enabled", True) else 0
    rows: list[dict[str, Any]] = []
    for bag, group in frame.groupby("bag", sort=False):
        group = group.reset_index(drop=True)
        channels = [("q", "q", "q_motor_clean", "q_ref_clean")]
        if (noise.get("dq") or {}).get("source", "sensor") == "sensor":
            # Under a drive the recorded dq is the drive's estimate through the
            # instrument (R6_04 A-4): measure the noise against that estimate.
            dq_clean = "dq_drive_clean" if f"dq_drive_clean0" in group.columns else "dq_motor_clean"
            channels.append(("dq", "dq", dq_clean, "dq_ref_clean"))
        measured_target = "noise_gain_ft__" + names[0] in group.columns
        for channel, recorded_prefix, clean_prefix, basis_prefix in channels:
            recorded = _block(group, recorded_prefix, n)[delay:]
            clean = _delayed(_block(group, clean_prefix, n), delay)[delay:]
            sigma = _joint_value(group, f"noise_sigma_{channel}__", names)
            basis = _block(group, basis_prefix, n)
            rms = np.sqrt(np.mean((basis - basis.mean(axis=0)) ** 2, axis=0))
            step = _quantization(noise, channel, n)
            realized = np.std(recorded - clean, axis=0)
            expected = np.sqrt(sigma**2 + step**2 / 12.0)
            p = float((noise.get(channel) or {}).get("p", 0.0))
            declared = _declared_sigma(noise.get(channel) or {}, rms, step)
            for j, name in enumerate(names):
                rows.append({"bag": bag, "channel": channel, "joint": name, "samples": len(recorded),
                             "realized": realized[j], "expected": expected[j], "sigma": sigma[j],
                             "declared_sigma": declared[j],
                             "sigma_over_basis": sigma[j] / rms[j] if rms[j] > 0 else np.nan, "p": p})
        if measured_target and noise.get("enabled", True):
            recorded = _block(group, "ft", n)[delay:]
            clean = _delayed(_block(group, "ft_clean", n), delay)[delay:]
            sigma = _joint_value(group, "noise_sigma_ft__", names)
            gain = _joint_value(group, "noise_gain_ft__", names)
            offset = _joint_value(group, "noise_offset_ft__", names)
            step = _quantization(noise, "ft", n)
            full = _block(group, "ft_clean", n)
            rms = np.sqrt(np.mean((full - full.mean(axis=0)) ** 2, axis=0))
            p = float((noise.get("ft") or {}).get("p", 0.0))
            declared = _declared_sigma(noise.get("ft") or {}, rms, step)
            for j, name in enumerate(names):
                design = np.column_stack([clean[:, j], np.ones(len(clean))])
                (fitted_gain, fitted_offset), *_ = np.linalg.lstsq(design, recorded[:, j], rcond=None)
                residual = recorded[:, j] - (gain[j] * clean[:, j] + offset[j])
                rows.append({"bag": bag, "channel": "ft", "joint": name, "samples": len(recorded),
                             "realized": float(np.std(residual)),
                             "expected": float(np.sqrt((gain[j] * sigma[j]) ** 2 + step[j] ** 2 / 12.0)),
                             "sigma": sigma[j], "declared_sigma": declared[j],
                             "sigma_over_basis": sigma[j] / rms[j] if rms[j] > 0 else np.nan, "p": p,
                             "gain": gain[j], "fitted_gain": float(fitted_gain),
                             "offset": offset[j], "fitted_offset": float(fitted_offset)})
    return pd.DataFrame(rows)


def _declared_sigma(channel: Mapping[str, Any], rms: np.ndarray, step: np.ndarray) -> np.ndarray:
    return np.sqrt((float(channel.get("p", 0.0)) * rms) ** 2 + float(channel.get("sigma", 0.0)) ** 2
                   + (float(channel.get("sigma_quanta", 0.0)) * step) ** 2)


def check_noise(frame: pd.DataFrame, manifest: Mapping[str, Any], *, tolerance: float | None = None) -> dict[str, Any]:
    """Sec 9: the recorded noise statistics match the noise block within sampling error.

    The tolerance on ``realized / expected`` is five standard errors of a
    sample std, ``5 / sqrt(2 N)``, plus 2 % for the uniform-quantization
    approximation; each bag's ``sigma`` must equal the declared one to rounding.
    """
    noise = manifest.get("noise") or {}
    if not noise.get("enabled", True):
        return {"check": "noise statistics", "ok": True, "note": "noise block disabled (ablation)"}
    stats = noise_statistics(frame, manifest)
    active = stats[stats["expected"] > 0.0]
    ratio = active["realized"] / active["expected"]
    allowed = 5.0 / np.sqrt(2.0 * active["samples"]) + 0.02 if tolerance is None else tolerance
    ratio_ok = bool(np.all(np.abs(ratio - 1.0) <= allowed))
    p_ok = bool(np.allclose(stats["sigma"], stats["declared_sigma"], rtol=1e-6, atol=0.0))
    per_channel = {
        channel: {"median_ratio": float(np.median(ratio[active["channel"] == channel])),
                  "worst_ratio": float(ratio[active["channel"] == channel].iloc[
                      int(np.argmax(np.abs(ratio[active["channel"] == channel].to_numpy() - 1.0)))])}
        for channel in sorted(set(active["channel"]))
    }
    return {"check": "noise statistics", "ok": ratio_ok and p_ok, "sigma_equals_declared": p_ok,
            "realized_over_expected": per_channel,
            "tolerance": "5 / sqrt(2 N) + 0.02 on realized / expected"}


def check_position_derivative(frame: pd.DataFrame, manifest: Mapping[str, Any]) -> dict[str, Any] | None:
    """iiwa: ``dq`` is the consumer's SG derivative of the recorded ``q`` (Sec 1)."""
    noise = manifest.get("noise") or {}
    if (noise.get("dq") or {}).get("source") != "position_derivative":
        return None
    from scipy.signal import savgol_filter

    n = int(manifest["n_dof"])
    policy = manifest.get("differentiation") or {}
    step = float(manifest["sample_time_step"])
    worst = 0.0
    for _, group in frame.groupby("bag", sort=False):
        q = _block(group, "q", n)
        expected = savgol_filter(q, int(policy["sg_window"]), int(policy["sg_poly"]), deriv=1, delta=step,
                                 axis=0, mode="interp")
        dq = _block(group, "dq", n)
        worst = max(worst, float(np.max(np.abs(dq - expected.astype(dq.dtype)) / (np.abs(expected) + 1e-6))))
    return {"check": "dq is the SG derivative of the recorded q", "ok": worst < 1e-5,
            "max_relative_difference": worst}


# ---------------------------------------------------------------------------
# Sec 7 gates
# ---------------------------------------------------------------------------

def gate_feedforward(manifest: Mapping[str, Any]) -> dict[str, Any]:
    controller = manifest.get("controller") or {}
    largest = max([float(r.get("feedforward_max_abs") or 0.0) for r in manifest.get("records", [])] or [0.0])
    ok = controller.get("feedforward") == "none" and largest == 0.0
    return {"gate": "1 feedforward share = 0", "ok": ok, "controller_feedforward": controller.get("feedforward"),
            "max_abs_feedforward": largest}


def gate_sag(manifest: Mapping[str, Any], limit: float, *, mode: str = "gate", reason: str = "") -> dict[str, Any]:
    """Sec 2.2's sag limit; ``mode: report`` records it without refusing (`R6_02` Sec 7)."""
    records = [r for r in manifest.get("records", []) if r.get("sag_max_fraction") is not None]
    if not records:
        return {"gate": "2 sag", "ok": False, "note": "no sag recorded (clean_columns off?)"}
    worst = max(records, key=lambda r: r["sag_max_fraction"])
    failing = [r["bag"] for r in records if r["sag_max_fraction"] > limit]
    return {"gate": "2 sag", "ok": not failing or mode == "report", "mode": mode,
            "within_limit": not failing, **({"reason": reason} if mode == "report" else {}),
            "limit": limit, "worst_bag": worst["bag"],
            "worst_joint": worst["sag_worst_joint"], "worst_fraction": worst["sag_max_fraction"],
            "median_of_bag_maxima": float(np.median([r["sag_max_fraction"] for r in records])),
            "worst_motor_fraction": max((max(r["sag_motor_fraction"]) for r in records
                                         if r.get("sag_motor_fraction")), default=None),
            "worst_per_joint": np.nanmax(np.asarray([r["sag_fraction"] for r in records if r.get("sag_fraction")],
                                                    dtype=float), axis=0).tolist()
            if any(r.get("sag_fraction") for r in records) else None,
            "failing_bags": failing}


#: `R6_04` A-6: a Q-C level is a verdict only over at least this many bags of
#: a tier kind; fewer (a preflight's single rigid bag) is reported, not gated.
QC_MIN_BAGS = 3


def gate_qc(diagnostics: pd.DataFrame, gates: Mapping[str, Any], *, min_bags: int = QC_MIN_BAGS) -> dict[str, Any]:
    kinds = np.where(diagnostics["tier"].eq("rigid"), "rigid", "elastic")
    levels = {kind: float(np.nanmedian(diagnostics.loc[kinds == kind, "tau_unexplained_fraction"]))
              for kind in ("elastic", "rigid") if np.any(kinds == kind)}
    counts = {kind: int(np.sum(kinds == kind)) for kind in levels}
    references = {"elastic": gates.get("qc_v2_reference_elastic"), "rigid": gates.get("qc_v2_reference_rigid")}
    verdicts = {}
    for kind, level in levels.items():
        reference = references.get(kind)
        gated = reference is not None and counts[kind] >= min_bags
        verdicts[kind] = {"level": level, "reference": reference, "bags": counts[kind],
                          "ok": bool(level >= reference) if gated else None,
                          **({} if gated or reference is None else
                             {"note": f"reported, not gated: {counts[kind]} bag(s) < {min_bags} (R6_04 A-6)"})}
    applicable = [v["ok"] for v in verdicts.values() if v["ok"] is not None]
    # R6_02 Sec 5: the statistic is on tau_cmd, as round 5 measured it (R5_08 Sec 9).
    return {"gate": "3 Q-C v2 >= round-5 reference level (pd on bus, velocity_pi in a drive)", "signal": "tau_cmd", "ok": all(applicable) if applicable else True,
            "applicable": bool(applicable), "statistic": "median tau_unexplained_fraction (1 - R^2)",
            "per_tier": verdicts, "reference_source": gates.get("qc_v2_reference_source")}


def gate_gainshift(production: Mapping[str, Any], gainshift: Mapping[str, Any]) -> dict[str, Any]:
    """The gain-shift file is the test split's bags, same trajectories, omega outside training."""
    test_robots = set(production.get("split", {}).get("test", []))
    shifted_tiers = {r["tier"] for r in gainshift.get("records", [])} | {
        u["tier"] for u in gainshift.get("unstable_bags", [])}
    production_digests = {r["bag"]: r["trajectory_digest"] for r in production.get("records", [])}
    same = [production_digests.get(r["bag"]) == r["trajectory_digest"] for r in gainshift.get("records", [])]
    train = [tuple(b) for b in production.get("gain_sizing", {}).get("natural_frequency_bands", [])]
    omega = [r["control_natural_frequency"] for r in gainshift.get("records", [])]
    inside = [w for w in omega if any(lo <= w <= hi for lo, hi in train)]
    ok = shifted_tiers == test_robots and all(same) and bool(same) and not inside
    return {"gate": "4 gain-shift file", "ok": ok, "test_robots": sorted(test_robots),
            "gainshift_robots": sorted(shifted_tiers), "same_trajectory_digests": bool(same) and all(same),
            "omega_inside_training_band": len(inside), "training_bands": train,
            "omega_fraction_range": [min((r["omega_fraction"] for r in gainshift.get("records", [])), default=None),
                                     max((r["omega_fraction"] for r in gainshift.get("records", [])), default=None)]}


def check_dataset(frame: pd.DataFrame, manifest: Mapping[str, Any], *, diagnostics: pd.DataFrame | None = None,
                  gates: Mapping[str, Any] | None = None, production: bool = True) -> dict[str, Any]:
    """Every Sec 9 hard check and the Sec 7 gates that one file carries."""
    gates = dict(manifest.get("gates") or {}) if gates is None else dict(gates)
    n = int(manifest["n_dof"])
    checks = [
        check_one_backend(frame),
        check_unstable(manifest, float(gates.get("unstable_max_fraction", 0.02))),
        check_clean_target(frame, n),
        check_noise(frame, manifest),
    ]
    derivative = check_position_derivative(frame, manifest)
    if derivative is not None:
        checks.append(derivative)
    gate_rows = [gate_feedforward(manifest), gate_sag(manifest, float(gates.get("sag_max_fraction", 0.05)),
                                                          mode=str(gates.get("sag_mode", "gate")),
                                                          reason=str(gates.get("sag_reason", "")))]
    if diagnostics is not None:
        gate_rows.append(gate_qc(diagnostics, gates))
    ok = all(c["ok"] for c in checks) and (not production or all(g["ok"] for g in gate_rows))
    return {"ok": bool(ok), "hard_checks": checks, "gates": gate_rows}
