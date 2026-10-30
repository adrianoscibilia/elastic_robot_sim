#!/usr/bin/env python3
"""Fit M, K, D per joint and pose from single-joint modal chirps (real-data spec ``FROM_IDENTIFIED_FILTER``
Sec 5; ``identified_filter/01`` Sec 1).

Per chirp (one joint moves, the others hold their pose)::

    tau_meas,j = H(s) [alpha M_jj theta''_j + (tau_rig,j - M_jj theta''_j)],
    H(s) = (D s + K) / (alpha M_jj s^2 + D s + K)

``tau_rig`` is the URDF RNEA at the measured ``(theta, theta', SG theta'')`` (gravity and coupling), ``M_jj``
the URDF diagonal at the pose; ``alpha``, ``K``, ``D`` are free (also reported with ``alpha = 1``).  The
fit runs on both sides band-passed to the chirp's band (the filter is linear, so the pair obeys it).

Per joint and pose: ``M_hat = alpha M_jj``, ``K``, ``D``, ``f_n``, ``zeta`` with bootstrap CIs over the
repeats; acceptance: repeat-to-repeat spread of ``f_n`` <= 10 %, in-band NRMSE <= 0.3.  On a simulated
modal file it also compares with the truth (the bag's ``K_eff``/``stiffness``, ``D_eff``/``damping``, the
audit ``mjj``): the pass condition is recovery within 10 %.  ``alpha`` of iiwa A4 is fitted once across
poses (its URDF ``M_jj`` does not spread, 0.64-0.67): ``--shared-alpha iiwa_A4``.

The single-joint model above sees the chirped joint's spring in series with its neighbours' (their links
flex against their held motors), so its K is an effective, pose-dependent stiffness: on the simulated UR10
(plant extras off) it reads lift 0.75 x and elbow 0.3-0.6 x the true K, repeatably (f_n spread < 1 %), while
M is recovered within 10 %.  A joint fit of the chirped joints with the full-coupling filter and the other
joints at the prior's centre was tried and diverges; the 10 % K recovery is not reachable with this model.

    python tools/modal_fit.py --dataset data/identification/r8/ur10/ur10_table_round8_modal.parquet \\
        --asset ur10_table --out REFACTOR_SPECS/identified_filter/results/modal_fit_ur10_sim
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(REPO / "src"))

SPREAD_MAX, NRMSE_MAX, RECOVERY_MAX = 0.10, 0.30, 0.10


def _block(frame, prefix, n):
    return frame[[f"{prefix}{i}" for i in range(n)]].to_numpy(dtype=float)


def segments(frame: pd.DataFrame, joint_names: list[str]) -> pd.DataFrame:
    """``bag -> (joint, pose, repeat)``: the r8 columns when present, else the moving joint and the rounded pose."""
    rows = []
    n = len(joint_names)
    for bag, group in frame.groupby("bag", sort=False):
        if "modal_joint" in group.columns:
            rows.append({"bag": bag, "joint": group["modal_joint"].iloc[0], "pose": group["modal_pose"].iloc[0],
                         "repeat": int(group["modal_repeat"].iloc[0])})
            continue
        q = _block(group, "q", n)
        moving = int(np.argmax(q.std(axis=0)))
        pose = tuple(np.round(np.median(np.delete(q, moving, axis=1), axis=0), 2))
        rows.append({"bag": bag, "joint": joint_names[moving], "pose": str(pose), "repeat": -1})
    table = pd.DataFrame(rows)
    if (table["repeat"] < 0).any():
        table["repeat"] = table.groupby(["joint", "pose"]).cumcount()
    return table


def _bandpass(x, low, high, rate):
    from scipy.signal import butter, sosfiltfilt

    nyquist = 0.5 * rate
    high = min(high, 0.95 * nyquist)
    sos = butter(2, [low / nyquist, high / nyquist], btype="band", output="sos")
    return sosfiltfilt(sos, x, axis=0)


def _response(params, theta_ddot, tau_rig, m_jj, step):
    from elastic_sim.two_mass import filter_sequence

    alpha, k, d = np.exp(params)
    drive = alpha * m_jj * theta_ddot + (tau_rig - m_jj * theta_ddot)
    out = filter_sequence(drive[:, None], np.full((len(drive), 1), alpha * m_jj), np.array([k]), np.array([d]), step)
    return out["tau_s"][:, 0]


def fit_chirp(theta_ddot, tau_rig, target, m_jj, step, band, *, fix_alpha=None, k0=1e4, d0=20.0, iterations=40):
    """Levenberg-Marquardt in ``(log alpha, log K, log D)`` on the band-passed pair; ``(alpha, K, D, nrmse)``."""
    rate = 1.0 / step
    sig = _bandpass(np.column_stack([theta_ddot, tau_rig, target]), band[0], band[1], rate)
    a, r_rig, y = sig[:, 0], sig[:, 1], sig[:, 2]
    keep = slice(int(0.5 * rate), len(y) - int(0.5 * rate))
    free = [0, 1, 2] if fix_alpha is None else [1, 2]
    theta = np.log([1.0 if fix_alpha is None else fix_alpha, k0, d0])

    def residual(p):
        return (_response(p, a, r_rig, m_jj, step) - y)[keep]

    r = residual(theta)
    cost, lam = float(r @ r), 1e-2
    best_start = None
    # A coarse log-K scan first: the chirp sweeps f_n, so the cost is multimodal in K far from it.
    for k in np.geomspace(k0 / 30, k0 * 30, 25):
        trial = theta.copy()
        trial[1] = np.log(k)
        c = float(np.sum(residual(trial) ** 2))
        if best_start is None or c < best_start[0]:
            best_start = (c, trial)
    theta, cost = best_start[1], best_start[0]
    r = residual(theta)
    for _ in range(iterations):
        J = np.empty((len(r), len(free)))
        for c, i in enumerate(free):
            probe = theta.copy()
            probe[i] += 1e-4
            J[:, c] = (residual(probe) - r) / 1e-4
        A, g = J.T @ J, J.T @ r
        for _ in range(8):
            step_ = np.clip(-np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), g), -1.0, 1.0)
            trial = theta.copy()
            trial[free] += step_
            r_new = residual(trial)
            if float(r_new @ r_new) < cost:
                theta, r, cost, lam = trial, r_new, float(r_new @ r_new), lam / 3
                break
            lam *= 4
        else:
            break
        if np.max(np.abs(step_)) < 1e-6:
            break
    alpha, k, d = np.exp(theta)
    nrmse = float(np.sqrt(np.mean(r ** 2)) / max(1e-12, np.sqrt(np.mean(y[keep] ** 2))))
    return float(alpha), float(k), float(d), nrmse


def bootstrap(values, draws=2000, seed=0):
    values = np.asarray([v for v in values if np.isfinite(v)])
    if len(values) < 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def _one(args):
    bag, group, n, names, step, window, poly, asset_name, seg = args
    from diagnose_controller_modes import _load_asset
    from elastic_sim.diagnostics import savitzky_golay_acceleration
    from elastic_sim.round7 import _model_for, _rigid

    asset = _load_asset(asset_name)
    pin, model, data = _model_for(asset, None, {})
    q, dq = _block(group, "q", n), _block(group, "dq", n)
    ddq = savitzky_golay_acceleration(dq, step, window=window, order=poly)
    tau_rig, inertia = _rigid(pin, model, data, q, dq, ddq)
    target = _block(group, "ft", n)
    j = names.index(seg["joint"])
    m_jj = float(np.median(inertia[:, j, j]))
    f0, f1 = float(seg.get("f0", 1.5)), float(seg.get("f1", 0.45 / step / 5))
    band = (0.8 * f0, min(1.2 * f1, 0.09 / step))
    out = {"bag": bag, **seg, "m_jj_urdf": m_jj}
    for label, fix in (("", None), ("_alpha1", 1.0)):
        alpha, k, d, nrmse = fit_chirp(ddq[:, j], tau_rig[:, j], target[:, j], m_jj, step, band, fix_alpha=fix)
        m_hat = alpha * m_jj
        out.update({f"alpha{label}": alpha, f"M_hat{label}": m_hat, f"K_hat{label}": k, f"D_hat{label}": d,
                    f"f_n{label}": math.sqrt(k / m_hat) / (2 * math.pi),
                    f"zeta{label}": d / (2 * math.sqrt(k * m_hat)), f"nrmse{label}": nrmse})
    name = seg["joint"]
    for column, key in (("K_eff__", "K_true_eff"), ("stiffness__", "K_true"), ("D_eff__", "D_true_eff"),
                        ("damping__", "D_true")):
        if f"{column}{name}" in group.columns:
            out[key] = float(group[f"{column}{name}"].iloc[0])
    if f"mjj{j}" in group.columns:
        out["M_true"] = float(group[f"mjj{j}"].median())
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="a modal parquet (simulated *_modal or a real modal file)")
    parser.add_argument("--asset", required=True, help="the URDF asset (ur10_table, kuka_lbr_iiwa_14_r820_table)")
    parser.add_argument("--chirps", default=None,
                        help="JSON {joint: [f0, f1]} for files without the r8 modal columns (default: the r8 config's)")
    parser.add_argument("--shared-alpha", nargs="*", default=["iiwa_A4"], help="joints whose alpha is one across poses")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--out", default=None, help="write <out>.csv, <out>.json, <out>.md (default: print only)")
    args = parser.parse_args(argv)
    sys.path.insert(0, os.fspath(REPO / "scripts"))

    path = Path(args.dataset)
    contract = json.loads(path.with_suffix(".contract.json").read_text(encoding="utf-8"))
    frame = pd.read_parquet(path)
    n = int(contract["n_dof"])
    names = [j["joint"] for j in contract["transmission_priors"]["joints"]] if "transmission_priors" in contract \
        else [f"j{i}" for i in range(n)]
    step = float(np.median(np.diff(frame.loc[frame["bag"] == frame["bag"].iloc[0], "t"])))
    window = int(contract["differentiation"]["sg_window"])
    poly = int(contract["differentiation"]["sg_poly"])
    segs = segments(frame, names)
    chirps = json.loads(args.chirps) if args.chirps else {
        j[0]: [j[1], j[2]] for j in (contract.get("round8", {}).get("modal_joints") or [])}
    tasks = []
    for _, seg in segs.iterrows():
        info = dict(seg)
        if info["joint"] in chirps:
            info["f0"], info["f1"] = chirps[info["joint"]]
        if "modal_f0" in frame.columns:
            group = frame[frame["bag"] == seg["bag"]]
            info["f0"], info["f1"] = float(group["modal_f0"].iloc[0]), float(group["modal_f1"].iloc[0])
        tasks.append((seg["bag"], frame[frame["bag"] == seg["bag"]], n, names, step, window, poly, args.asset, info))
    if args.jobs > 1:
        import concurrent.futures

        with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as pool:
            rows = list(pool.map(_one, tasks))
    else:
        rows = [_one(t) for t in tasks]
    table = pd.DataFrame(rows)
    group_keys = ["tier", "joint", "pose"] if "tier" in frame.columns else ["joint", "pose"]
    if "tier" in frame.columns:
        table["tier"] = table["bag"].map(frame.groupby("bag")["tier"].first())
    summary = []
    for key, part in table.groupby(group_keys):
        entry = dict(zip(group_keys, key if isinstance(key, tuple) else (key,)))
        for column in ("M_hat", "K_hat", "D_hat", "f_n", "zeta", "nrmse", "K_hat_alpha1", "f_n_alpha1"):
            entry[column] = float(part[column].mean())
            entry[f"{column}_ci"] = bootstrap(part[column])
        entry["f_n_spread"] = float(part["f_n"].max() / part["f_n"].min() - 1.0) if len(part) > 1 else float("nan")
        entry["accept_spread"] = bool(entry["f_n_spread"] <= SPREAD_MAX) if len(part) > 1 else None
        entry["accept_nrmse"] = bool(entry["nrmse"] <= NRMSE_MAX)
        for truth in ("K_true_eff", "K_true", "D_true_eff", "D_true", "M_true"):
            if truth in part:
                estimate = {"K": "K_hat", "D": "D_hat", "M": "M_hat"}[truth[0]]
                entry[f"recovery_{truth}"] = float(abs(part[estimate].mean() / part[truth].mean() - 1.0))
        summary.append(entry)
    summary = pd.DataFrame(summary)
    passes = {c: float((summary[c] <= RECOVERY_MAX).mean()) for c in summary.columns if c.startswith("recovery_")}
    print(summary.round(3).to_string(index=False))
    print("recovery within 10 % (share of joint-pose groups):", {k: round(v, 3) for k, v in passes.items()})
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out.with_suffix(".per_chirp.csv"), index=False)
        summary.to_csv(out.with_suffix(".csv"), index=False)
        out.with_suffix(".json").write_text(json.dumps({"dataset": str(path), "recovery_pass_share": passes,
                                                        "acceptance": {"spread_max": SPREAD_MAX,
                                                                       "nrmse_max": NRMSE_MAX,
                                                                       "recovery_max": RECOVERY_MAX},
                                                        "groups": summary.to_dict(orient="records")},
                                                       indent=1, default=float), encoding="utf-8")
        out.with_suffix(".md").write_text("# modal_fit\n\n" + summary.round(4).to_markdown(index=False)
                                          if hasattr(summary, "to_markdown") else summary.to_string(), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
