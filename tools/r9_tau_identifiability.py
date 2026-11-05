#!/usr/bin/env python3
"""identified_filter/12: can tau_cmd + motor-side signals identify each robot's transmission stiffness? (no network)

Per robot and main joint, from the recorded motor-side q, dq (ddq the contract's SG derivative) and the commanded
motor torque tau only; the link side (q_link, tau_s, delta) and the manifest records only score:

* **M1** the motor-side accelerance ``omega |H|``, ``H = Theta_dot / T_cmd`` (H1, Welch 50 %), on the chirp bags
  whose excited joint it is, the repeats of one pose pooled: antiresonance ``f_ar`` (the deepest local minimum on
  lines with coherence >= 0.8 inside the chirp's band), resonance ``f_r`` (the largest line above it);
  ``K = (2 pi f_ar)^2 M_L``, ``B = M_L / ((f_r / f_ar)^2 - 1)``, ``zeta`` from the half-power width at ``f_r``
  (converted to the spring's definition, ``x f_ar / f_r``).  The accelerance, not ``|H|``: a mass's ``|H|`` falls as
  ``1 / f`` and its minimum would be the band's top edge.
* **M2** grey-box least squares on the motor equation, ``tau_cmd = B ddq + b dq + c tanh(dq / eps) + tau_s_hat``,
  ``tau_s_hat`` the r9 tangent filter (catalogue curve, fixed-point selector; ``twomass_filter``) driven by the
  rigid URDF ``tau_r`` with ``M_f = alpha M_jj`` and ``D = 2 zeta sqrt(K_0 Mbar_f)``; ``(B, b, c)`` by linear
  least squares inside each residual (variable projection), ``(K_0, zeta, alpha)`` by ``least_squares`` from 5
  log-uniform starts in the prior, ``zeta`` in [0.01, 1].  The recording lag of q, dq behind tau (1-2 samples) is
  chosen per fit by a rigid motor-side pre-fit.  ``M2ub`` adds the link-side tau_s as a second target (reference).
* **M3** M2 on windows of the production test bags (1, 2, 5 s and the whole bag; regimes full, real_like).
* **pooled** the population median of the 180 robots (what a pooled model knows).

Sec 4: the identified (M2, chirp; robot median) ``K_0, zeta, alpha`` in the oracle's filter (full coupling, fixed
point, rigid URDF drive) on the same robots' production test bags and chirps, scored beside 11's cached models.

    uv run python tools/r9_tau_identifiability.py fit        # estimates.csv, spectra.pkl
    uv run python tools/r9_tau_identifiability.py filter     # Sec 4 predictions + metrics
    uv run python tools/r9_tau_identifiability.py figures
    uv run python tools/r9_tau_identifiability.py report     # results/r9/12_tau_identifiability.md
    uv run python tools/r9_tau_identifiability.py all
    uv run python tools/r9_tau_identifiability.py inspect --platform ur10 --bag t0_e16_f0_mujoco   # nothing saved

Output under ``REFACTOR_SPECS/identified_filter/results/r9/tau_ident/``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import pickle
import sys
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
SIM = Path(__file__).resolve().parents[1]
NN = (SIM / ".." / "dynamic_model_nn").resolve()
sys.path[:0] = [str(SIM / "src"), str(SIM / "scripts"), str(SIM / "tools")]

import numpy as np
import pandas as pd
from scipy import optimize, signal

import r9_timeseries as ts

OUT = ts.RESULTS / "tau_ident"
REPORT = ts.RESULTS / "12_tau_identifiability.md"
MAIN, GATED = ts.MAIN, ts.GATED
GATED_JOINTS = {("iiwa_drive", "iiwa_A2"), ("ur10", "shoulder_lift_joint"), ("ur10", "elbow_joint")}
REGIMES = ("full", "real_like")
WINDOWS = {"1s": (1.0, 4), "2s": (2.0, 4), "5s": (5.0, 2), "bag": (None, 1)}   # length [s], windows per bag
STARTS = 5
ZETA_BOUNDS = (0.01, 1.0)
TRAINING_ZETA = (0.04, 0.12)
ALPHA_RANGE = (0.7, 1.4)
K_WIDENING = 2.0             # the fit's K_0 bound: the prior widened by this factor each way
COULOMB_EPS = 0.01           # the motor friction's tanh velocity (manifest motor_friction.epsilon)
MAX_LAG = 3
SETTLE_S = 0.1               # each fit drops this much of its window: the filter's quasi-static start
COHERENCE_MIN = 0.8
WELCH_S = 4.0                # M1 segment length (0.25 Hz lines)
AR_DEPTH_MIN = 1.5           # M1: an antiresonance is a notch at least this deep against both sides' maxima
WITHIN = 0.15
CONTROL_NEAR = 1.5           # a controller frequency within this factor of f_ar or f_r is flagged


# ---------------------------------------------------------------------------------------------------- data -----

def manifest(platform: str, file: str) -> dict:
    return json.loads(ts.file_path(platform, file).with_suffix(".manifest.json").read_text(encoding="utf-8"))


def chirp_robots(platform: str) -> list[str]:
    return list(manifest(platform, "modal")["only_tiers"])


def priors(platform: str) -> dict:
    return ts.contract(platform)["transmission_priors"]


def load_frame(platform: str, file: str, tiers: list[str], regimes: tuple[str, ...] | None = None) -> pd.DataFrame:
    import pyarrow.parquet as pq

    path = ts.file_path(platform, file)
    names = ts.joint_names(platform)
    n = len(names)
    schema = set(pq.read_schema(path).names)
    blocks = ("q", "dq", "tau", "ft", "ft_clean", "q_motor_clean", "q_link_clean")
    per_joint = ("stiffness__", "damping__", "rotor_inertia__", "viscous__", "coulomb__", "link_viscous__",
                 "link_coulomb__", "te_amplitude__", "te_order__", "te_phase__", "te_amplitude_n__", "te_order_n__",
                 "te_phase_n__")
    columns = ["t", "bag", "tier", "split", "probe_regime", "modal_joint", "modal_pose", "modal_repeat", "modal_f0",
               "modal_f1"] + [f"{b}{j}" for b in blocks for j in range(n)] \
        + [f"{p}{x}" for p in per_joint for x in names]
    filters = [("tier", "in", tiers)] + ([("probe_regime", "in", list(regimes))] if regimes else []) \
        + ([("split", "==", "test")] if file == "production" else [])
    return pq.read_table(path, columns=[c for c in columns if c in schema], filters=filters).to_pandas()


def bag_items(platform: str, file: str, frame: pd.DataFrame) -> list[dict]:
    names = ts.joint_names(platform)
    n = len(names)
    records = {r["bag"]: r for r in manifest(platform, file)["records"]}
    out = []
    for bag, g in frame.groupby("bag", sort=False):
        item = {"name": str(bag), "tier": str(g["tier"].iloc[0]), "regime": str(g["probe_regime"].iloc[0]),
                "t": g["t"].to_numpy(float)}
        for b in ("q", "dq", "tau", "ft", "ft_clean", "q_motor_clean", "q_link_clean"):
            item[b] = g[[f"{b}{j}" for j in range(n)]].to_numpy(float)
        for p in ("stiffness", "damping", "rotor_inertia", "viscous", "coulomb", "link_viscous", "link_coulomb",
                  "te_amplitude", "te_order", "te_phase", "te_amplitude_n", "te_order_n", "te_phase_n"):
            item[p] = np.asarray([float(g[f"{p}__{x}"].iloc[0]) if f"{p}__{x}" in g else np.nan for x in names])
        if file == "modal":
            item.update(joint=str(g["modal_joint"].iloc[0]), pose=str(g["modal_pose"].iloc[0]),
                        repeat=int(g["modal_repeat"].iloc[0]), f0=float(g["modal_f0"].iloc[0]),
                        f1=float(g["modal_f1"].iloc[0]))
        record = records.get(str(bag), {})
        item["control_hz"] = np.asarray(record.get("joint_frequency", [np.nan] * n), dtype=float) / (2 * math.pi)
        out.append(item)
    return out


def population(platform: str) -> dict:
    """The pooled reference: medians over the 180 robots of the production manifest."""
    robots = manifest(platform, "production")["robots"]
    stiffness = np.asarray([r["stiffness"] for r in robots], dtype=float)
    damping = np.asarray([r["damping"] for r in robots], dtype=float)
    rotor = np.asarray([r["rotor_inertia"] for r in robots], dtype=float)
    inertia = np.asarray([j["link_inertia_median_range"][0] for j in priors(platform)["joints"]], dtype=float)
    return {"K": np.median(stiffness, axis=0), "B": np.median(rotor, axis=0),
            "zeta": np.median(damping / (2.0 * np.sqrt(stiffness * inertia)), axis=0)}


def prepare(platform: str, item: dict) -> dict:
    """Consumer signals (SG ddq), the rigid URDF's tau_r and M(theta), the true spring torque."""
    from elastic_sim.diagnostics import savitzky_golay_acceleration
    from elastic_sim.round7 import _model_for, _rigid

    contract = ts.contract(platform)
    step = float(contract["transmission_priors"]["sample_time_step"])
    window, poly = int(contract["differentiation"]["sg_window"]), int(contract["differentiation"]["sg_poly"])
    ts.curve(platform)
    item["step"] = step
    item["ddq"] = savitzky_golay_acceleration(item["dq"], step, window=window, order=poly)
    model = _model_for(ts._CURVES[platform + ":asset"], None, {})
    item["tau_r"], item["inertia"] = _rigid(*model, item["q"], item["dq"], item["ddq"])
    _, item["S"] = ts.true_spring(item, platform)
    return item


# ------------------------------------------------------------------------------------------------ filters -----

def tangent_fixed_point(tau_r: np.ndarray, m: np.ndarray, K0: float, D: float, factors: np.ndarray,
                        knees: np.ndarray, blend: float, step: float, start: np.ndarray) -> np.ndarray:
    """One joint's r9 tangent filter (diagonal), region selected by its own S (the ladder's fixed point)."""
    import torch

    sys.path.insert(0, str(NN))
    from twomass_filter import catalogue_deflection, knee_factor, two_mass_filter

    tau = torch.from_numpy(np.ascontiguousarray(tau_r, dtype=np.float64)).view(1, -1, 1)
    mm = torch.from_numpy(np.ascontiguousarray(m, dtype=np.float64)).view(1, -1, 1)
    f, k = torch.from_numpy(factors[None].astype(np.float64)), torch.from_numpy(knees[None].astype(np.float64))
    K = torch.tensor([K0], dtype=torch.float64)
    Dt = torch.tensor([[D]], dtype=torch.float64)
    delta0 = catalogue_deflection(tau[:, 0], K, f, k)
    selector = torch.from_numpy(np.ascontiguousarray(start, dtype=np.float64)).view(1, -1, 1)
    tolerance = 1e-6 * max(1.0, float(np.abs(tau_r).max()))
    spring = None
    for _ in range(10):
        tau_s, _, velocity = two_mass_filter(tau, mm, K0 * knee_factor(selector, f, k, blend), Dt, step,
                                             coupling="diagonal", spring="tangent", initial_deflection=delta0)
        new = tau_s - Dt * velocity
        moved = float("inf") if spring is None else float((new - spring).abs().max())
        spring, selector = new, new
        if moved < tolerance:
            break
    return tau_s.view(-1).numpy()


def delayed(x: np.ndarray, lag: int) -> np.ndarray:
    return x if lag <= 0 else np.concatenate([np.repeat(x[:1], lag), x[:-lag]])


def fit_motor_equation(item: dict, platform: str, j: int, sl: slice, seed: int, link_target: bool = False) -> dict:
    """M2 / M3 on samples ``sl`` of joint ``j``: ``(K_0, zeta, alpha)`` by least squares, ``(B, b, c)`` projected."""
    import r8_physics_ladder as ladder

    step = item["step"]
    factors, knees, blend = ts.curve(platform)
    cutoff = ts._CURVES[platform + ":cutoff"]
    tau_r, m = item["tau_r"][sl, j], item["inertia"][sl, j, j]
    dq, ddq = item["dq"][sl, j], item["ddq"][sl, j]
    phi = np.column_stack([ddq, dq, np.tanh(dq / COULOMB_EPS)])
    settle = max(int(round(SETTLE_S / step)), MAX_LAG)
    keep = np.arange(len(tau_r)) >= settle
    # The recording lag: q, dq are 1-2 samples behind tau; pick it by a rigid motor-side pre-fit.
    rigid = np.column_stack([phi, tau_r])[keep]
    lags = []
    for lag in range(MAX_LAG + 1):
        target = delayed(item["tau"][sl, j], lag)[keep]
        coef, *_ = np.linalg.lstsq(rigid, target, rcond=None)
        lags.append(float(np.sum((target - rigid @ coef) ** 2)))
    lag = int(np.argmin(lags))
    commands = [delayed(item["tau"][sl, j], k) for k in range(MAX_LAG + 1)]
    tau_cmd = commands[lag]
    ft = item["ft"][sl, j]
    start = ladder.selection(tau_r[:, None], "lowpass", cutoff, step)[:, 0]
    mean_m = float(m.mean())

    def predict(x):
        K0, zeta, alpha = np.exp(x)
        D = 2.0 * zeta * math.sqrt(K0 * alpha * mean_m)
        return tangent_fixed_point(tau_r, alpha * m, K0, D, factors[j], knees[j], blend, step, start)

    def project(tau_s, command=None):
        y = ((tau_cmd if command is None else command) - tau_s)[keep]
        coef, *_ = np.linalg.lstsq(phi[keep], y, rcond=None)
        return coef, y - phi[keep] @ coef

    def residual(x):
        tau_s = predict(x)
        if not np.all(np.isfinite(tau_s)):
            return np.full(int(keep.sum()) * (2 if link_target else 1), 1e3)
        r = project(tau_s)[1]
        return np.concatenate([r, (ft - tau_s)[keep]]) if link_target else r

    joint = priors(platform)["joints"][j]
    k_lo, k_hi = joint["stiffness_range"]
    z_lo, z_hi = joint["damping_ratio_range"]
    lower = np.log([k_lo / K_WIDENING, ZETA_BOUNDS[0], ALPHA_RANGE[0]])
    upper = np.log([k_hi * K_WIDENING, ZETA_BOUNDS[1], ALPHA_RANGE[1]])
    rng = np.random.default_rng(seed)
    runs = []
    for _ in range(STARTS):
        x0 = np.log([math.exp(rng.uniform(math.log(k_lo), math.log(k_hi))),
                     math.exp(rng.uniform(math.log(z_lo), math.log(z_hi))),
                     math.exp(rng.uniform(*np.log(ALPHA_RANGE)))])
        res = optimize.least_squares(residual, x0, bounds=(lower, upper), diff_step=1e-3, x_scale=1.0,
                                     max_nfev=80, xtol=1e-6, ftol=1e-8)
        runs.append(res)
    best = min(runs, key=lambda r: r.cost)
    # The rigid pre-fit's lag is biased by the unmodelled mode: re-pick it on the fitted tau_s, refit warm if it moved.
    tau_s = predict(best.x)
    costs = [float(np.sum(project(tau_s, c)[1] ** 2)) for c in commands]
    if int(np.argmin(costs)) != lag:
        lag = int(np.argmin(costs))
        tau_cmd = commands[lag]
        best = optimize.least_squares(residual, best.x, bounds=(lower, upper), diff_step=1e-3, x_scale=1.0,
                                      max_nfev=80, xtol=1e-6, ftol=1e-8)
        runs.append(best)
        tau_s = predict(best.x)
    coef, r = project(tau_s)
    near = [r_ for r_ in runs if r_.cost <= best.cost * 1.01]
    k_starts = np.exp([r_.x[0] for r_ in near])
    K0, zeta, alpha = np.exp(best.x)
    return {"K_hat": K0, "zeta_hat": zeta, "alpha_hat": alpha, "B_hat": coef[0], "bv_hat": coef[1],
            "c_hat": coef[2], "lag": lag, "rmse": float(np.sqrt(np.mean(r ** 2))),
            "rel_rmse": float(np.sqrt(np.mean(r ** 2)) / np.std(tau_cmd[keep])),
            "starts_near": len(near), "starts_K_spread": float(k_starts.max() / k_starts.min()),
            "at_K_bound": bool(np.any(np.isclose(best.x[0], [lower[0], upper[0]], atol=1e-3))),
            "at_zeta_bound": bool(np.any(np.isclose(best.x[1], [lower[1], upper[1]], atol=1e-3))),
            "at_alpha_bound": bool(np.any(np.isclose(best.x[2], [lower[2], upper[2]], atol=1e-3)))}


def truth_of(item: dict, platform: str, j: int, sl: slice) -> dict:
    from elastic_sim.two_mass import knee_factor

    factors, knees, blend = ts.curve(platform)
    load = float(np.mean(np.abs(item["S"][sl, j])))
    phi = float(knee_factor(np.asarray([[load]]), factors[j:j + 1], knees[j:j + 1], blend)[0, 0])
    K0, D, B = item["stiffness"][j], item["damping"][j], item["rotor_inertia"][j]
    M = float(item["inertia"][sl, j, j].mean())
    K_eff = K0 * phi
    return {"K0": K0, "K_eff": K_eff, "phi": phi, "load": load, "D": D, "zeta_spring": D / (2 * math.sqrt(K0 * M)),
            "B": B, "bv": item["viscous"][j], "c": item["coulomb"][j], "M_L": M,
            "f_ar_true": math.sqrt(K_eff / M) / (2 * math.pi),
            "f_r_true": math.sqrt(K_eff * (B + M) / (B * M)) / (2 * math.pi),
            "control_hz": float(item["control_hz"][j])}


# ---------------------------------------------------------------------------------------------------- fit -----

def _fit_chirp(args) -> tuple[list[dict], dict]:
    platform, item, seed = args
    item = prepare(platform, item)
    names = ts.joint_names(platform)
    j = names.index(item["joint"])
    rate = 1.0 / item["step"]
    nper = int(round(WELCH_S * rate))
    u, y = item["tau"][:, j], item["dq"][:, j]
    f, suu = signal.welch(u, fs=rate, nperseg=nper)
    _, syy = signal.welch(y, fs=rate, nperseg=nper)
    _, suy = signal.csd(u, y, fs=rate, nperseg=nper)
    whole = slice(0, len(u))
    truth = truth_of(item, platform, j, whole)
    amplitude = float(np.max(np.abs(item["q"][:, j] - np.median(item["q"][:, j]))))
    base = {"platform": platform, "file": "modal", "regime": "modal", "robot": item["tier"], "bag": item["name"],
            "joint": item["joint"], "pose": item["pose"], "repeat": item["repeat"], "f0": item["f0"],
            "f1": item["f1"], "amplitude_rad": amplitude, "window": "chirp", "start_s": 0.0, **truth}
    rows = [{**base, "method": "M2", **fit_motor_equation(item, platform, j, whole, seed)},
            {**base, "method": "M2ub", **fit_motor_equation(item, platform, j, whole, seed, link_target=True)}]
    spectra = {"f": f, "suu": suu, "syy": syy, "suy": suy, **base}
    return rows, spectra


def _fit_production(args) -> list[dict]:
    platform, item, seed, joints = args
    item = prepare(platform, item)
    names = ts.joint_names(platform)
    steps = len(item["t"])
    rows = []
    for name in joints:
        j = names.index(name)
        for label, (length, count) in WINDOWS.items():
            width = steps if length is None else int(round(length / item["step"]))
            for start in np.linspace(0, steps - width, count).astype(int):
                sl = slice(int(start), int(start) + width)
                rows.append({"platform": platform, "file": "production", "regime": item["regime"],
                             "robot": item["tier"], "bag": item["name"], "joint": name, "pose": "", "window": label,
                             "start_s": float(start * item["step"]), "method": f"M3_{label}",
                             **truth_of(item, platform, j, sl),
                             **fit_motor_equation(item, platform, j, sl, seed + int(start))})
    return rows


def cmd_fit(args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, spectra = [], []
    for platform in args.platform:
        robots = chirp_robots(platform)
        names = ts.joint_names(platform)
        chirps = [i for i in bag_items(platform, "modal", load_frame(platform, "modal", robots)) if i["joint"] in names]
        production = bag_items(platform, "production", load_frame(platform, "production", robots, REGIMES))
        work_c = [(platform, item, 1000 + k) for k, item in enumerate(chirps)]
        work_p = [(platform, item, 5000 + 100 * k, (name,)) for k, item in enumerate(production)
                  for name in MAIN[platform]]
        print(f"== {platform}: {len(work_c)} chirp bags, {len(work_p)} production bags", flush=True)
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            for k, (r, s) in enumerate(pool.map(_fit_chirp, work_c, chunksize=1)):
                rows += r
                spectra.append(s)
                if k % 20 == 0:
                    print(f"  chirp {platform} {k + 1}/{len(work_c)}", flush=True)
            for k, r in enumerate(pool.map(_fit_production, work_p, chunksize=1)):
                rows += r
                print(f"  production {platform} {k + 1}/{len(work_p)}", flush=True)
    frame = pd.DataFrame(rows)
    path = OUT / "fits.csv"
    if path.exists() and set(args.platform) != {"iiwa_drive", "ur10"}:
        old = pd.read_csv(path)
        frame = pd.concat([old[~old.platform.isin(args.platform)], frame], ignore_index=True)
    frame.to_csv(path, index=False)
    spath = OUT / "spectra.pkl"
    if spath.exists() and set(args.platform) != {"iiwa_drive", "ur10"}:
        spectra = [s for s in pickle.loads(spath.read_bytes()) if s["platform"] not in args.platform] + spectra
    spath.write_bytes(pickle.dumps(spectra, protocol=pickle.HIGHEST_PROTOCOL))
    m1 = m1_table(spectra)
    m1.to_csv(OUT / "m1.csv", index=False)
    print(f"wrote {path} ({len(frame)} rows), {OUT / 'm1.csv'} ({len(m1)} rows)")


# ----------------------------------------------------------------------------------------------------- M1 -----

def m1_features(f: np.ndarray, H: np.ndarray, coh: np.ndarray, f0: float, f1: float) -> dict:
    """Antiresonance and resonance of the accelerance on the coherent lines inside the chirp band."""
    A = np.abs(H) * 2 * np.pi * f
    band = (f >= 1.05 * f0) & (f <= 0.95 * f1)
    valid = band & (coh >= COHERENCE_MIN)
    out = {"lines_band": int(band.sum()), "lines_valid": int(valid.sum()),
           "coh_band_mean": float(np.mean(coh[band])) if band.any() else np.nan,
           "f_ar": np.nan, "f_r": np.nan, "zeta_hp": np.nan, "ar_depth": np.nan, "f_r_edge": False}
    idx = np.flatnonzero(valid)
    if len(idx) < 5:
        return out
    # Local minima among valid lines whose two neighbours are valid lines too; the deepest one against the
    # largest value on either side.
    best = None
    for k in idx:
        if not (valid[k - 1] and valid[k + 1]) or not (A[k] < A[k - 1] and A[k] <= A[k + 1]):
            continue
        left, right = A[idx[idx < k]], A[idx[idx > k]]
        if not len(left) or not len(right):
            continue
        depth = min(left.max(), right.max()) / A[k]
        if best is None or depth > best[1]:
            best = (k, depth)
    if best is None or best[1] < AR_DEPTH_MIN:
        return out
    k, depth = best
    la = np.log(A[k - 1:k + 2])                 # parabolic refinement on log A
    denom = la[0] - 2 * la[1] + la[2]
    shift = 0.5 * (la[0] - la[2]) / denom if denom > 0 else 0.0
    df = f[1] - f[0]
    out["f_ar"], out["ar_depth"] = float(f[k] + shift * df), float(depth)
    above = idx[idx > k]
    p = above[np.argmax(A[above])]
    out["f_r_edge"] = bool(p == idx[-1] or not valid[min(p + 1, len(f) - 1)])
    if out["f_r_edge"]:
        return out
    lp = np.log(A[p - 1:p + 2])
    denom = lp[0] - 2 * lp[1] + lp[2]
    shift = 0.5 * (lp[0] - lp[2]) / denom if denom < 0 else 0.0
    out["f_r"] = float(f[p] + shift * df)
    half = A[p] / math.sqrt(2.0)
    lo = p
    while lo > k and A[lo] > half:
        lo -= 1
    hi = p
    while hi < idx[-1] and A[hi] > half:
        hi += 1
    if A[lo] <= half and A[hi] <= half and valid[lo:hi + 1].all():
        f1_ = np.interp(half, [A[lo], A[lo + 1]], [f[lo], f[lo + 1]])
        f2_ = np.interp(half, [A[hi], A[hi - 1]], [f[hi], f[hi - 1]])
        out["zeta_hp"] = float((f2_ - f1_) / (2 * out["f_r"]))
    return out


def m1_table(spectra: list[dict]) -> pd.DataFrame:
    """Per robot, joint and pose (the repeats' spectra summed): M1's features and estimates."""
    groups: dict = {}
    for s in spectra:
        groups.setdefault((s["platform"], s["robot"], s["joint"], s["pose"]), []).append(s)
    rows = []
    for (platform, robot, joint, pose), items in groups.items():
        suu, syy = sum(s["suu"] for s in items), sum(s["syy"] for s in items)
        suy = sum(s["suy"] for s in items)
        f = items[0]["f"]
        H, coh = suy / suu, np.abs(suy) ** 2 / (suu * syy)
        feat = m1_features(f, H, coh, items[0]["f0"], items[0]["f1"])
        truth = {k: float(np.mean([s[k] for s in items])) for k in
                 ("K0", "K_eff", "phi", "load", "D", "zeta_spring", "B", "M_L", "f_ar_true", "f_r_true", "control_hz")}
        M = truth["M_L"]
        K = (2 * math.pi * feat["f_ar"]) ** 2 * M
        ratio = feat["f_r"] / feat["f_ar"]
        B = M / (ratio ** 2 - 1) if np.isfinite(ratio) and ratio > 1 else np.nan
        near = [abs(math.log(truth["control_hz"] / x)) < math.log(CONTROL_NEAR)
                for x in (feat["f_ar"], feat["f_r"]) if np.isfinite(x)]
        rows.append({"platform": platform, "robot": robot, "joint": joint, "pose": pose, "method": "M1",
                     "repeats": len(items), "f0": items[0]["f0"], "f1": items[0]["f1"], **feat, **truth,
                     "K_hat": K, "B_hat": B, "zeta_hat": feat["zeta_hp"] / ratio if np.isfinite(ratio) else np.nan,
                     "control_near": bool(any(near)),
                     "f_ar_true_in_band": bool(1.05 * items[0]["f0"] <= truth["f_ar_true"] <= 0.95 * items[0]["f1"]),
                     "f_r_true_in_band": bool(1.05 * items[0]["f0"] <= truth["f_r_true"] <= 0.95 * items[0]["f1"])})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------- tables -----

def modal_truth() -> pd.DataFrame:
    """11 A4.4's FRF of the truth (tau_m -> tau_s): per robot and joint, the median effective modal f_n, zeta."""
    frf = pd.read_csv(ts.OUT / "frf.csv")
    frf = frf[frf.model == "rigid"].copy()
    frf["robot"] = frf["bag"].str.split("_").str[1]
    return frf.groupby(["platform", "robot", "joint"]).agg(f_modal=("f_peak_true", "median"),
                                                           zeta_modal=("zeta_true", "median")).reset_index()


def robot_level(estimates: pd.DataFrame) -> pd.DataFrame:
    """One estimate per (method, regime, robot, joint): the median over its bags / poses / windows' log values.

    M3 windows are kept per window (each a separate estimate, as an encoder would make one per window)."""
    cols = ["K_hat", "zeta_hat", "B_hat", "alpha_hat", "K0", "K_eff", "zeta_spring", "B"]
    est = estimates.copy()
    est["unit"] = np.where(est.method.str.startswith("M3"), est["bag"] + "@" + est["start_s"].astype(str), "")
    keys = ["platform", "method", "regime", "robot", "joint", "unit"]

    def logmed(x):
        x = x.to_numpy(float)
        x = x[np.isfinite(x) & (x > 0)]
        return float(np.exp(np.median(np.log(x)))) if len(x) else np.nan

    agg = {c: logmed for c in cols if c in est}
    agg["bag"] = "count"
    return est.groupby(keys).agg(agg).rename(columns={"bag": "n"}).reset_index()


def score(group: pd.DataFrame, hat: str, truth: str) -> dict:
    a, b = group[hat].to_numpy(float), group[truth].to_numpy(float)
    ok = np.isfinite(a) & (a > 0) & np.isfinite(b) & (b > 0)
    err = np.abs(a[ok] / b[ok] - 1)
    corr = float(np.corrcoef(np.log(a[ok]), np.log(b[ok]))[0, 1]) if ok.sum() > 2 and np.std(a[ok]) > 0 else np.nan
    return {"n": int(len(a)), "found": int(ok.sum()),
            "med_err": float(np.median(err)) if ok.any() else np.nan,
            "within": float(np.sum(err <= WITHIN) / len(a)) if len(a) else np.nan,   # misses count as outside
            "corr_log": corr}


def accuracy_table(level: pd.DataFrame) -> pd.DataFrame:
    rows = []
    modal = modal_truth()
    level = level.merge(modal, on=["platform", "robot", "joint"], how="left")
    for (platform, method, regime, joint), g in level.groupby(["platform", "method", "regime", "joint"]):
        row = {"platform": platform, "joint": joint, "method": method, "regime": regime}
        for hat, truth, tag in (("K_hat", "K0", "K0"), ("K_hat", "K_eff", "Keff"), ("zeta_hat", "zeta_spring", "zs"),
                                ("zeta_hat", "zeta_modal", "zm"), ("B_hat", "B", "B")):
            for k, v in score(g, hat, truth).items():
                if k in ("n", "found") and tag != "Keff":
                    continue
                row[f"{tag}_{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def pooled_rows(level: pd.DataFrame) -> pd.DataFrame:
    """The pooled reference on the chirp robots: the population medians against each robot's truth."""
    rows = []
    base = level[level.method == "M2"]
    for platform, g in base.groupby("platform"):
        pop = population(platform)
        names = ts.joint_names(platform)
        g = g.copy()
        j = g["joint"].map(names.index).to_numpy()
        g["K_hat"], g["zeta_hat"], g["B_hat"] = pop["K"][j], pop["zeta"][j], pop["B"][j]
        g["method"], g["alpha_hat"] = "pooled", 1.0
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


# ------------------------------------------------------------------------------------------------- Sec 4 -----

def identified_parameters(platform: str, level: pd.DataFrame, method: str = "M2", regime: str = "modal") -> dict:
    """``robot -> (K_0, zeta, alpha)`` arrays over all joints: the identified main joints, the pooled elsewhere."""
    names = ts.joint_names(platform)
    pop = population(platform)
    out = {}
    part = level[(level.platform == platform) & (level.method == method) & (level.regime == regime)]
    for robot, g in part.groupby("robot"):
        K, zeta, alpha = pop["K"].copy(), pop["zeta"].copy(), np.ones(len(names))
        source = ["pooled"] * len(names)
        for _, row in g.iterrows():
            j = names.index(row["joint"])
            K[j], zeta[j], alpha[j], source[j] = row["K_hat"], row["zeta_hat"], row["alpha_hat"], method
        out[robot] = (K, zeta, alpha, source)
    return out


def _filter_bag(args) -> tuple[str, dict]:
    platform, item, variants = args
    import r8_physics_ladder as ladder
    from elastic_sim.identification import FrictionModel

    item = prepare(platform, item)
    step = item["step"]
    curve = ts.curve(platform)
    cutoff = ts._CURVES[platform + ":cutoff"]
    out = {}
    for label, (K, zeta, alpha, friction) in variants.items():
        tau_r = item["tau_r"] + (FrictionModel(item["link_viscous"], item["link_coulomb"]).torque(item["dq"])
                                 if friction else 0.0)
        root = np.sqrt(alpha)
        inertia = root[None, :, None] * item["inertia"] * root[None, None, :]
        D = 2.0 * zeta * np.sqrt(K * np.diagonal(inertia, axis1=1, axis2=2).mean(axis=0))
        start = ladder.selection(tau_r, "lowpass", cutoff, step)
        result, _ = ladder.fixed_point_pass(tau_r, inertia, K, D, curve, step, start=start)
        out[label] = {"pred": result["tau_s"].astype(np.float32), "delta": result["delta"].astype(np.float32)}
    return item["name"], out


def cmd_filter(args) -> None:
    estimates = pd.read_csv(OUT / "fits.csv")
    level = robot_level(estimates)
    m3 = level[(level.method == "M3_bag")].copy()
    cache = OUT / "cache"
    rows, frf = [], []
    for platform in args.platform:
        robots = chirp_robots(platform)
        names = ts.joint_names(platform)
        ident = identified_parameters(platform, level)
        m3_params = {r: identified_parameters(platform, m3, "M3_bag", r) for r in REGIMES}
        contract = ts.contract(platform)["transmission_priors"]
        rate, burn_in = 1.0 / float(contract["sample_time_step"]), float(contract["burn_in_s"])
        for file in ("production", "modal"):
            path = cache / f"{platform}-{file}.pkl"
            frame = load_frame(platform, file, robots, REGIMES if file == "production" else None)
            items = bag_items(platform, file, frame)
            if file == "modal":
                items = [i for i in items if i["joint"] in MAIN[platform]]
            work = []
            for item in items:
                K, zeta, alpha, _ = ident[item["tier"]]
                variants = {"ident": (K, zeta, alpha, False),
                            "ident_zclip": (K, np.clip(zeta, *TRAINING_ZETA), alpha, False),
                            "ident_linkfric": (K, zeta, alpha, True)}
                if file == "production":
                    other = "real_like" if item["regime"] == "full" else "full"
                    if item["tier"] in m3_params[other]:
                        K3, z3, a3, _ = m3_params[other][item["tier"]]
                        variants["ident_M3x"] = (K3, z3, a3, False)
                work.append((platform, item, variants))
            if path.exists() and not args.refresh:
                preds = pickle.loads(path.read_bytes())
            else:
                with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
                    preds = dict(pool.map(_filter_bag, work, chunksize=1))
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(pickle.dumps(preds, protocol=pickle.HIGHEST_PROTOCOL))
            print(f"  filter {platform} {file}: {len(preds)} bags", flush=True)
            truth = ts.load(platform, file, "truth")
            rigid = ts.load(platform, file, "rigid")
            cached = {m: ts.load(platform, file, m) for m in ("rigid", "lpf_rigid", "J2", "J1", "oracle_filter")}
            for bag, variants in preds.items():
                item = truth[bag]
                models = {**variants, **{m: c[bag] for m, c in cached.items() if c is not None and bag in c}}
                for name in MAIN[platform]:
                    j = names.index(name)
                    if file == "modal" and item["excited"] != name:
                        continue
                    for model, pred in models.items():
                        row = {"platform": platform, "file": file, "model": model, "bag": bag, "joint": name,
                               "regime": "modal" if file == "modal" else item["regime"], "robot": item["tier"],
                               **ts.bag_metrics(item, pred, rigid[bag], j, rate, burn_in)}
                        rows.append(row)
                        if file == "modal":
                            frf.append({"platform": platform, "model": model, "bag": bag, "joint": name,
                                        "robot": item["tier"], **ts.frf_metrics(item, pred, j, rate, burn_in)})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "sec4_bag_metrics.csv", index=False)
    pd.DataFrame(frf).to_csv(OUT / "sec4_frf.csv", index=False)
    print(f"wrote {OUT / 'sec4_bag_metrics.csv'} ({len(metrics)} rows)")


# ------------------------------------------------------------------------------------------------ figures -----

def two_mass_accelerance(f: np.ndarray, B: float, M: float, K: float, D: float, b: float) -> np.ndarray:
    """``omega |Theta_dot / T|`` of the free two-mass (motor B with viscous b, spring K + D s, link M)."""
    s = 2j * np.pi * f
    link = M * s ** 2 + D * s + K
    H = s * link / ((B * s ** 2 + b * s) * link + M * s ** 2 * (D * s + K))
    return np.abs(H) * 2 * np.pi * f


def cmd_figures(args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = OUT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    spectra = pickle.loads((OUT / "spectra.pkl").read_bytes())
    m1 = pd.read_csv(OUT / "m1.csv")
    m2 = robot_level(pd.read_csv(OUT / "fits.csv"))
    m2 = m2[m2.method == "M2"]
    for platform, joint in sorted(GATED_JOINTS):
        robots = sorted({s["robot"] for s in spectra if s["platform"] == platform and s["joint"] == joint})[:3]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=False)
        for ax, robot in zip(axes, robots):
            items = [s for s in spectra if (s["platform"], s["robot"], s["joint"]) == (platform, robot, joint)
                     and s["pose"] == sorted({x["pose"] for x in spectra if x["joint"] == joint})[0]]
            f = items[0]["f"]
            suu, syy, suy = (sum(s[k] for s in items) for k in ("suu", "syy", "suy"))
            coh = np.abs(suy) ** 2 / (suu * syy)
            A = np.abs(suy / suu) * 2 * np.pi * f
            band = (f >= items[0]["f0"]) & (f <= 1.6 * items[0]["f1"])
            ok = band & (coh >= COHERENCE_MIN)
            ax.semilogy(f[band], A[band], color="0.7", lw=1, label="|A| all lines")
            ax.semilogy(f[ok], A[ok], ".", color="C0", ms=4, label=f"γ² ≥ {COHERENCE_MIN}")
            s0 = items[0]
            ax.semilogy(f[band], two_mass_accelerance(f[band], s0["B"], s0["M_L"], s0["K_eff"], s0["D"], s0["bv"]),
                        color="C3", lw=1.2, label="truth two-mass (K_eff)")
            row = m1[(m1.platform == platform) & (m1.robot == robot) & (m1.joint == joint) & (m1.pose == s0["pose"])]
            for col, color, label in (("f_ar", "C2", "f_ar M1"), ("f_r", "C1", "f_r M1")):
                if len(row) and np.isfinite(row[col].iloc[0]):
                    ax.axvline(row[col].iloc[0], color=color, ls="--", lw=1, label=label)
            ax.axvline(s0["f_ar_true"], color="C2", ls=":", lw=1, label="f_ar true")
            ax.axvline(s0["f_r_true"], color="C1", ls=":", lw=1, label="f_r true")
            ax.axvline(items[0]["f1"], color="k", lw=0.6, label="chirp f1")
            if np.isfinite(s0["control_hz"]) and s0["control_hz"] < f[band].max():
                ax.axvline(s0["control_hz"], color="C4", lw=0.8, label="control f")
            ax.set_title(f"{robot} {joint} ({s0['pose']}): K0 {s0['K0']:.0f}, K_eff {s0['K_eff']:.0f}", fontsize=9)
            ax.set_xlabel("Hz")
        axes[0].set_ylabel("ω |Θ̇ / T_cmd|  [1/(kg m²)]")
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figures / f"accelerance_{platform}_{joint}.png", dpi=120)
        plt.close(fig)
    level = robot_level(pd.read_csv(OUT / "fits.csv"))
    m1r = m1.copy()
    m1r["regime"] = "modal"
    m1r = robot_level(m1r.assign(bag=m1r["pose"], start_s=0.0))
    level = pd.concat([level, m1r, pooled_rows(level)], ignore_index=True)
    methods = [("M1", "modal"), ("M2", "modal"), ("M3_bag", "full"), ("M3_2s", "full"), ("pooled", "modal")]
    joints = [(p, j) for p in ("iiwa_drive", "ur10") for j in MAIN[p]]
    fig, axes = plt.subplots(len(methods), len(joints), figsize=(3.0 * len(joints), 2.8 * len(methods)))
    for r, (method, regime) in enumerate(methods):
        for c, (platform, joint) in enumerate(joints):
            ax = axes[r, c]
            g = level[(level.method == method) & (level.regime == regime) & (level.platform == platform)
                      & (level.joint == joint)]
            if len(g):
                ax.loglog(g["K_eff"], g["K_hat"], ".", ms=5)
                lim = [min(g["K_eff"].min(), np.nanmin(g["K_hat"].where(g["K_hat"] > 0))) * 0.8,
                       max(g["K_eff"].max(), np.nanmax(g["K_hat"])) * 1.25]
                ax.loglog(lim, lim, "k-", lw=0.6)
                ax.fill_between(lim, [x * (1 - WITHIN) for x in lim], [x * (1 + WITHIN) for x in lim],
                                color="0.85", zorder=0)
            ax.set_title(f"{method} ({regime}) {joint}", fontsize=8)
            ax.tick_params(labelsize=7)
            if r == len(methods) - 1:
                ax.set_xlabel("K_eff true", fontsize=8)
            if c == 0:
                ax.set_ylabel("K̂", fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "k_hat_vs_k.png", dpi=110)
    plt.close(fig)
    print(f"wrote {figures}")


# ------------------------------------------------------------------------------------------------- report -----

def fmt(x, digits=2) -> str:
    if isinstance(x, (bool, np.bool_)):
        return str(bool(x))
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        return "–" if not np.isfinite(x) else f"{x:.{digits}f}"
    return str(x)


def md_table(frame: pd.DataFrame, digits: int = 2) -> list[str]:
    head = "| " + " | ".join(frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    return [head, rule] + ["| " + " | ".join(fmt(v, digits) for v in row) + " |" for row in frame.itertuples(index=False)]


def sec4_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(OUT / "sec4_bag_metrics.csv")
    table = ts.pooled_table(metrics)
    lpf = table[table.model == "lpf_rigid"].set_index(["platform", "file", "joint", "regime"])["rmse"]
    table["vs_lpf"] = table["rmse"] / lpf.reindex(pd.MultiIndex.from_frame(
        table[["platform", "file", "joint", "regime"]])).to_numpy()
    frf = pd.read_csv(OUT / "sec4_frf.csv").groupby(["platform", "joint", "model"]).agg(
        bags=("bag", "count"), f_peak_ratio=("f_peak_ratio", "median"), zeta_ratio=("zeta_ratio", "median"),
        peak_gain_ratio=("peak_gain_ratio", "median")).reset_index()
    return table, frf


def cmd_report(args) -> None:
    fits = pd.read_csv(OUT / "fits.csv")
    m1 = m1_table(pickle.loads((OUT / "spectra.pkl").read_bytes()))     # recomputed: M1's rules are cheap to revisit
    m1.to_csv(OUT / "m1.csv", index=False)
    level = robot_level(fits)
    m1r = robot_level(m1.assign(regime="modal", bag=m1["pose"], start_s=0.0))
    level_all = pd.concat([m1r, level, pooled_rows(level)], ignore_index=True)
    acc = accuracy_table(level_all)
    acc.to_csv(OUT / "accuracy.csv", index=False)
    table, frf = sec4_tables()
    table.to_csv(OUT / "sec4_pooled.csv", index=False)
    frf.to_csv(OUT / "sec4_frf_pooled.csv", index=False)

    lines = ["# 12 — τ_cmd identifiability of the transmission stiffness (simulation, no training)", "",
             "Generated by `tools/r9_tau_identifiability.py report` from `results/r9/tau_ident/*.csv`.", ""]
    lines += ["## Data", ""]
    chirp = fits[fits.method == "M2"]
    design = chirp.groupby(["platform", "joint", "pose"]).agg(
        robots=("robot", "nunique"), bags=("bag", "count"), f0=("f0", "first"), f1=("f1", "first"),
        amplitude_mrad=("amplitude_rad", lambda x: 1e3 * float(np.median(x))),
        f_ar_true=("f_ar_true", "median"), f_r_true=("f_r_true", "median"), load_Nm=("load", "median"),
        phi=("phi", "median"), control_hz=("control_hz", "median")).reset_index()
    lines += md_table(design) + [""]
    lines += ["## Table 1 — identification accuracy (per robot; M3 per window)", ""]
    cols = ["platform", "joint", "method", "regime", "Keff_n", "Keff_found", "K0_med_err", "K0_within",
            "K0_corr_log", "Keff_med_err", "Keff_within", "Keff_corr_log", "zs_med_err", "zs_corr_log",
            "zm_med_err", "zm_corr_log", "B_med_err", "B_within", "B_corr_log"]
    order = {m: i for i, m in enumerate(["M1", "M2", "M2ub", "M3_bag", "M3_5s", "M3_2s", "M3_1s", "pooled"])}
    acc = acc.assign(_o=acc.method.map(order)).sort_values(["platform", "joint", "_o", "regime"]).drop(columns="_o")
    lines += md_table(acc[[c for c in cols if c in acc]]) + [""]
    lines += ["## M1 diagnostics (per pose, repeats pooled)", ""]
    diag = m1.groupby(["platform", "joint"]).agg(
        poses=("pose", "count"), lines_band=("lines_band", "median"), lines_valid=("lines_valid", "median"),
        coh_band=("coh_band_mean", "median"), f_ar_found=("f_ar", lambda x: float(np.isfinite(x).mean())),
        f_r_found=("f_r", lambda x: float(np.isfinite(x).mean())),
        f_ar_true_in_band=("f_ar_true_in_band", "mean"), f_r_true_in_band=("f_r_true_in_band", "mean"),
        control_near=("control_near", "mean")).reset_index()
    lines += md_table(diag) + [""]
    lines += ["## M2 diagnostics (chirp bags)", ""]
    d2 = fits[fits.method.isin(["M2", "M2ub", "M3_bag", "M3_1s"])].groupby(["platform", "joint", "method"]).agg(
        fits=("bag", "count"), rel_rmse=("rel_rmse", "median"), lag=("lag", "median"),
        K_spread=("starts_K_spread", "median"), at_K_bound=("at_K_bound", "mean"),
        at_zeta_bound=("at_zeta_bound", "mean"), alpha=("alpha_hat", "median"),
        zeta_hat=("zeta_hat", "median")).reset_index()
    lines += md_table(d2, 3) + [""]
    lines += ["## Table 2 — Sec 4: identified filter on the chirp robots' production test bags", ""]
    keep = ["J1", "J2", "ident", "ident_M3x", "ident_linkfric", "ident_zclip", "lpf_rigid", "oracle_filter", "rigid"]
    t = table[table.model.isin(keep)].copy()
    t["_o"] = t.model.map({m: i for i, m in enumerate(keep)})
    short = ["model", "bags", "rmse", "vs_rigid", "vs_lpf", "rmse_low", "rmse_mode", "rmse_high", "coherence",
             "gain", "phase_deg", "delta_corr", "delta_rms_ratio"]
    for file in ("production", "modal"):
        for (platform, joint, regime), g in t[t.file == file].sort_values("_o").groupby(
                ["platform", "joint", "regime"], sort=False):
            tag = " **(gated)**" if (platform, joint, regime) in GATED else ""
            lines += [f"### {platform} {joint} {regime}{tag}", ""] + md_table(g[short], 3) + [""]
    lines += ["### Modal FRF (τ_m → τ_s), median ratios to the truth", ""]
    lines += md_table(frf[frf.model.isin(keep)], 3) + [""]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {REPORT}")


# ------------------------------------------------------------------------------------------------ inspect -----

def cmd_inspect(args) -> None:
    """One bag: M1 / M2 estimates against the truth, and its accelerance on screen.  Writes nothing."""
    platform = args.platform[0]
    file = "modal" if "_" in args.bag and args.file == "modal" else args.file
    import pyarrow.parquet as pq

    frame = pq.read_table(ts.file_path(platform, file), filters=[("bag", "==", args.bag)]).to_pandas()
    item = bag_items(platform, file, frame)[0]
    if file == "modal":
        rows, s = _fit_chirp((platform, item, 0))
        H, coh = s["suy"] / s["suu"], np.abs(s["suy"]) ** 2 / (s["suu"] * s["syy"])
        print(pd.Series(m1_features(s["f"], H, coh, s["f0"], s["f1"])).to_string())
    else:
        rows = _fit_production((platform, item, 0, MAIN[platform]))
    show = ["method", "joint", "window", "start_s", "K_hat", "K0", "K_eff", "zeta_hat", "zeta_spring", "alpha_hat",
            "B_hat", "B", "lag", "rel_rmse", "starts_K_spread"]
    print(pd.DataFrame(rows)[show].to_string())
    if file == "modal" and not args.no_plot:
        import matplotlib.pyplot as plt

        f, A = s["f"], np.abs(H) * 2 * np.pi * s["f"]
        band = f <= 1.6 * s["f1"]
        plt.semilogy(f[band], A[band], label="measured")
        plt.semilogy(f[band], two_mass_accelerance(f[band], s["B"], s["M_L"], s["K_eff"], s["D"], s["bv"]),
                     label="truth two-mass")
        plt.legend()
        plt.show()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("fit", "filter", "figures", "report", "all", "inspect"))
    parser.add_argument("--platform", nargs="+", default=["iiwa_drive", "ur10"])
    parser.add_argument("--workers", type=int, default=30)
    parser.add_argument("--refresh", action="store_true", help="filter: recompute the cached predictions")
    parser.add_argument("--bag", help="inspect: the bag")
    parser.add_argument("--file", default="modal", choices=("modal", "production"), help="inspect: its file")
    parser.add_argument("--no-plot", action="store_true", help="inspect: print only")
    args = parser.parse_args(argv)
    if args.command == "inspect":
        cmd_inspect(args)
        return 0
    for name in (("fit", "filter", "figures", "report") if args.command == "all" else (args.command,)):
        {"fit": cmd_fit, "filter": cmd_filter, "figures": cmd_figures, "report": cmd_report}[name](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
