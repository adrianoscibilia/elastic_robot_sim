"""The physical metric suite (S2R_02 A-2, S2R_04 A-11, P3-4).

One function, :func:`metrics`, computes the suite for a set of replay
results; the problem, the report and ``replay.py`` all call it, so every
number has one definition:

(a) **torque** (FR_05-comparable): RMSE in N m over all joints and rows per
    bag, median over bags (the contract baselines' statistic); per-joint
    RMSE pooled over the split's rows, target RMS and NRMSE = RMSE / target
    RMS; the run contracts' baselines alongside. Excitation-type bags only.
(b) **hold gravity slopes** per run: ``ft = a * (gain * tau_s) + b`` over the
    run's hold samples (``tau_s`` at a hold is the plant's gravity at the
    deflected pose), on the joints whose ``gain * tau_s`` spans >= 2 N m
    over those samples (S2R_04 N-9; A2, A4 expected). The joint set is
    returned with the numbers. Target 1.00. Uses every hold in the results,
    whatever its split: a slope is a property of the run.
(c) **friction signature**: residual mean per joint in velocity bins; a
    flat residual means friction is captured.
(d) **residual spectrum**: band mean-square power <= 0.5 Hz, 0.5-2 Hz,
    > 2 Hz per joint (DC in the first band; the bands sum to the MS).
(e) R2 only (tracking error q - cmd): not in R0.
(f) **plausibility** (:func:`plausibility`): per parameter of the stage's
    set its prior distance in sigma, its data share d_post (A-11 as amended
    by S2R_06 F-2, ``d_post < 0.9`` = prior-determined; the diagonal d as a
    diagnostic) and whether it is calibrated (S2R_06 A-12: a parameter the
    data cannot see is frozen and reported "not calibrated"), the
    physical-consistency verdict, mass and COM moves in physical units.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .result import ReplayResult

#: Velocity-bin edges for (c), rad/s (symmetric; |dq| < 0.005 is "at rest").
VELOCITY_BIN_EDGES = (-np.inf, -0.5, -0.2, -0.1, -0.05, -0.02, -0.005, 0.005, 0.02, 0.05, 0.1, 0.2, 0.5, np.inf)
MIN_BIN_ROWS = 20
#: (d) bands, Hz.
SPECTRUM_BANDS_HZ = ((0.0, 0.5), (0.5, 2.0), (2.0, np.inf))
#: (b) S2R_04 N-9: a joint enters the slope table if its gravity spans this much.
HOLD_GRAVITY_RANGE_MIN = 2.0
#: (f) S2R_06 F-2: below this d_post a parameter is prior-determined.
PRIOR_DETERMINED_BELOW = 0.9

_EXCITATION_KINDS = ("excitation", "identify_excitation")
_BASELINE_KEYS = ("target_rms", "baseline_rms_mean", "baseline_rms_rigid", "baseline_rms_state",
                  "baseline_rms_tau", "baseline_rms_state_tau")


def metrics(results: Sequence[ReplayResult], split: str | None = None, *,
            contracts: Mapping[str, Mapping[str, Any]] | None = None,
            plausibility: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The A-2 suite on ``results`` restricted to ``split`` (None = all).
    ``contracts``: run id -> dataset contract (for (a)'s baselines);
    ``plausibility``: :func:`plausibility`'s block (f), passed through."""
    chosen = [r for r in results if split is None or r.split == split]
    dynamic = [r for r in chosen if r.kind in _EXCITATION_KINDS]
    out: dict[str, Any] = {"split": split or "all", "bags": [f"{r.run_id}:{r.bag}" for r in chosen],
                           "mode": chosen[0].mode if chosen else None}
    if dynamic:
        out["torque"] = torque_errors(dynamic)
        out["friction_signature"] = friction_signature(dynamic)
        out["spectrum"] = residual_spectrum(dynamic)
        if contracts:
            runs = sorted({r.run_id for r in dynamic})
            out["baselines"] = {run: _contract_baselines(contracts[run]) for run in runs if run in contracts}
    holds = [r for r in results if r.kind == "hold"]
    if holds:
        out["hold_slopes"] = hold_slopes(holds)
    holds_in_split = [r for r in chosen if r.kind == "hold"]
    if holds_in_split:
        residual = np.vstack([r.residual for r in holds_in_split])
        out["hold_residual"] = {"rms_per_joint": np.sqrt(np.mean(residual ** 2, axis=0)).tolist(),
                                "mean_per_joint": residual.mean(axis=0).tolist()}
    if plausibility is not None:
        out["plausibility"] = dict(plausibility)
    return out


def torque_errors(results: Sequence[ReplayResult]) -> dict[str, Any]:
    """(a)."""
    per_bag = {f"{r.run_id}:{r.bag}": float(np.sqrt(np.mean(r.residual ** 2))) for r in results}
    residual = np.vstack([r.residual for r in results])
    real = np.vstack([r.ft for r in results])
    rmse = np.sqrt(np.mean(residual ** 2, axis=0))
    target = np.sqrt(np.mean(real ** 2, axis=0))
    return {"rmse_median_over_bags": float(np.median(list(per_bag.values()))),
            "rmse_pooled": float(np.sqrt(np.mean(residual ** 2))),
            "per_joint_rmse": rmse.tolist(), "per_joint_target_rms": target.tolist(),
            "per_joint_nrmse": (rmse / np.where(target > 0, target, np.nan)).tolist(),
            "per_bag_rmse": per_bag,
            "statistic": "RMS over joints and kept rows per bag, median over bags (contract baselines' statistic); "
                         "per joint pooled over the split's kept rows"}


def hold_slopes(holds: Sequence[ReplayResult]) -> dict[str, Any]:
    """(b), per run."""
    out: dict[str, Any] = {}
    for run in sorted({r.run_id for r in holds}):
        rows = [r for r in holds if r.run_id == run]
        real = np.vstack([r.ft for r in rows])
        offset = np.vstack([r.ft_sim - _gain_tau(r) for r in rows])
        model = np.vstack([_gain_tau(r) for r in rows])
        span = model.max(axis=0) - model.min(axis=0)
        joints = [int(j) for j in np.flatnonzero(span >= HOLD_GRAVITY_RANGE_MIN)]
        slope, intercept, r2 = {}, {}, {}
        for j in joints:
            design = np.column_stack([model[:, j], np.ones(len(model))])
            (a, b), *_ = np.linalg.lstsq(design, real[:, j], rcond=None)
            fit = design @ np.array([a, b])
            ss = float(np.sum((real[:, j] - real[:, j].mean()) ** 2))
            slope[j], intercept[j] = float(a), float(b)
            r2[j] = 1.0 - float(np.sum((real[:, j] - fit) ** 2)) / ss if ss > 0 else 1.0
        out[run] = {"joints": joints, "slope": slope, "intercept": intercept, "r2": r2,
                    "gravity_span": span.tolist(), "mean_offset": offset.mean(axis=0).tolist(),
                    "n_holds": len(rows), "rows": int(len(real))}
    out["statistic"] = (f"ft = a * (gain * tau_s) + b per joint over the run's hold samples, joints whose gain * tau_s "
                        f"spans >= {HOLD_GRAVITY_RANGE_MIN:g} N m (S2R_04 N-9)")
    return out


def _gain_tau(result: ReplayResult) -> np.ndarray:
    """gain * tau_s: what the observation model adds to the offset."""
    if result.offset is None:
        raise ValueError(f"{result.run_id}:{result.bag}: hold slopes need the result's observation offset")
    return result.ft_sim - result.offset


def friction_signature(results: Sequence[ReplayResult]) -> dict[str, Any]:
    """(c): residual mean per joint in velocity bins of the replay's dq."""
    dq = np.vstack([r.dq for r in results])
    residual = np.vstack([r.residual for r in results])
    edges = np.asarray(VELOCITY_BIN_EDGES)
    n = dq.shape[1]
    bins = []
    worst = np.zeros(n)
    odd = np.full(n, np.nan)
    for j in range(n):
        index = np.digitize(dq[:, j], edges[1:-1])
        means, counts = [], []
        for b in range(len(edges) - 1):
            mask = index == b
            counts.append(int(mask.sum()))
            means.append(float(residual[mask, j].mean()) if mask.sum() >= MIN_BIN_ROWS else None)
        bins.append({"mean": means, "count": counts})
        valid = [abs(m) for m in means if m is not None]
        worst[j] = max(valid) if valid else np.nan
        positive = residual[dq[:, j] > 0.005, j]
        negative = residual[dq[:, j] < -0.005, j]
        if len(positive) >= MIN_BIN_ROWS and len(negative) >= MIN_BIN_ROWS:
            odd[j] = 0.5 * (positive.mean() - negative.mean())
    return {"edges_rad_s": [float(e) for e in edges], "per_joint": bins, "max_abs_bin_mean": worst.tolist(),
            "odd_component": odd.tolist(),
            "statistic": f"mean residual per dq bin (bins with >= {MIN_BIN_ROWS} rows); odd component = "
                         "0.5 (mean | dq > 0.005 - mean | dq < -0.005), a Coulomb-like leftover"}


def residual_spectrum(results: Sequence[ReplayResult]) -> dict[str, Any]:
    """(d): mean-square power per band and joint, row-weighted over bags.
    Bags with non-uniform kept rows (a quasi-static filter) are skipped."""
    n = results[0].n_dof
    power = np.zeros((len(SPECTRUM_BANDS_HZ), n))
    total_rows, used = 0, 0
    for r in results:
        if r.n_rows < 8:
            continue
        steps = np.diff(r.t)
        step = float(np.median(steps))
        if not np.allclose(steps, step, rtol=1e-6, atol=1e-9):
            continue
        residual = r.residual
        spectrum = np.fft.rfft(residual, axis=0)
        freq = np.fft.rfftfreq(len(residual), step)
        weights = np.full(len(freq), 2.0)
        weights[0] = 1.0
        if len(residual) % 2 == 0:
            weights[-1] = 1.0
        ms = weights[:, None] * np.abs(spectrum) ** 2 / len(residual) ** 2  # sums to mean(r^2)
        for b, (lo, hi) in enumerate(SPECTRUM_BANDS_HZ):
            mask = (freq >= lo) & (freq < hi) if lo > 0 else freq < hi
            power[b] += ms[mask].sum(axis=0) * r.n_rows
        total_rows += r.n_rows
        used += 1
    if total_rows:
        power /= total_rows
    total = power.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        fraction = np.where(total > 0, power / total, np.nan)
    return {"bands_hz": [[lo, None if not np.isfinite(hi) else hi] for lo, hi in SPECTRUM_BANDS_HZ],
            "power_nm2": power.tolist(), "fraction": fraction.tolist(), "bags": used,
            "statistic": "periodogram of the residual per bag, mean-square per band (DC in the first), "
                         "row-weighted over bags"}


def plausibility(registry: Any, theta: Mapping[str, float], plant: Any, *,
                 data_share: Mapping[str, float] | None = None,
                 data_share_posterior: Mapping[str, float] | None = None,
                 not_calibrated: Mapping[str, str] | None = None,
                 partners: Mapping[str, Sequence[Mapping[str, Any]]] | None = None) -> dict[str, Any]:
    """(f): per parameter of ``registry``'s free set: prior distance (sigma),
    d_post (F-2) and d, and ``calibrated`` (False with the reason for the
    ``not_calibrated`` ones, A-12'/A-13, and their ``partners``, S2R_08);
    the physical-consistency verdict;
    inertial moves in physical units against the asset."""
    rows = []
    not_calibrated = dict(not_calibrated or {})
    for row in registry.rows(theta):
        name = row["name"]
        d = None if data_share is None else data_share.get(name)
        d_post = None if data_share_posterior is None else data_share_posterior.get(name)
        rows.append({"name": name, "value": row["value"], "mean": row["mean"],
                     "prior_distance_sigma": row["prior_distance_sigma"], "data_share": d,
                     "data_share_posterior": d_post,
                     "prior_determined": None if d_post is None else bool(d_post < PRIOR_DETERMINED_BELOW),
                     "calibrated": name not in not_calibrated, "not_calibrated_reason": not_calibrated.get(name),
                     "partners": (partners or {}).get(name)})
    moves = {}
    for link, inertial in plant.inertials.items():
        nominal = plant.base.inertials[link]
        delta_com = inertial.com - nominal.com
        moves[link] = {"mass_kg": inertial.mass, "delta_mass_kg": inertial.mass - nominal.mass,
                       "delta_mass_pct": 100.0 * (inertial.mass / nominal.mass - 1.0),
                       "delta_com_m": delta_com.tolist(), "delta_com_norm_m": float(np.linalg.norm(delta_com)),
                       "inertia_scale": float(theta.get(f"inertial.{link}.inertia_scale", 1.0))}
    problems = plant.check_physical()
    return {"parameters": rows, "physical": {"consistent": not problems, "problems": problems},
            "inertial_moves_vs_asset": moves,
            "n_prior_determined": sum(1 for r in rows if r["prior_determined"] and r["calibrated"]),
            "n_not_calibrated": sum(1 for r in rows if not r["calibrated"]),
            "rule": f"prior-determined: d_post < {PRIOR_DETERMINED_BELOW} (S2R_06 F-2); not calibrated: d_post below "
                    "the threshold on the full free set (S2R_08 A-12', fitted value kept; or frozen by S2R_06 A-12 "
                    "in freeze_refit mode), or excluded by A-13 (another session's offset)"}


def _contract_baselines(contract: Mapping[str, Any]) -> dict[str, Any]:
    splits = (contract.get("baselines") or {}).get("splits") or {}
    block = splits.get("all") or {}
    return {key: block.get(key) for key in _BASELINE_KEYS} | {"n_bags": block.get("n_bags"),
                                                                "statistic": "contract, median over the run's bags"}


def headline(m: Mapping[str, Any]) -> dict[str, Any]:
    """A compact row for history records and comparison tables."""
    torque = m.get("torque") or {}
    out = {"rmse": torque.get("rmse_median_over_bags"), "per_joint_rmse": torque.get("per_joint_rmse")}
    slopes = m.get("hold_slopes") or {}
    for run, block in slopes.items():
        if isinstance(block, Mapping) and "slope" in block:
            out[f"slopes_{run}"] = {int(j): v for j, v in block["slope"].items()}
    return out


__all__ = ["HOLD_GRAVITY_RANGE_MIN", "PRIOR_DETERMINED_BELOW", "SPECTRUM_BANDS_HZ", "VELOCITY_BIN_EDGES",
           "friction_signature", "headline", "hold_slopes", "metrics", "plausibility", "residual_spectrum",
           "torque_errors"]
