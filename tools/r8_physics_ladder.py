#!/usr/bin/env python3
"""identified_filter/04 A1: the physics-oracle ladder.  Can the two-mass equation, given the truth, beat rigid?

The campaign's held-out robots are re-simulated with the plant's features switched on one by one:

    L0  linear spring, no transmission error (TE), no ripple, no noise
    L1  + sensor noise and the consumer's SG-differentiated acceleration
    L2  + nonlinear spring
    L3  + TE
    L3p L2 + TE from the ``te_randomised`` tier (``identified_filter/05`` D1; ``config/campaigns/te_randomised.yaml``)
    L4  + torque ripple: the r8 plant, the configs unchanged

The r8 configs set no ``torque_ripple``, so L4 is L3 by construction; it is simulated anyway and the report
says whether the two agree.  Same robots, seeds and trajectories on every rung (checked on ``q_ref_clean``), and
L4 is checked against the production file.  At each rung, against the clean ``tau_s`` on the rows past the
contract's burn-in, overall and in ``[0.5, 2] f_n``:

* ``rigid``: ``tau_r`` = URDF RNEA + the bag's true link friction, no filter;
* ``nominal``: the filter with the nominal K, the prior-centre zeta and that same ``tau_r``.  It differs from
  the contract baseline ``rigid_elastic_nominal``, which has no friction, so only (K, D) separate it from the oracle;
* ``oracle_sampled`` / ``oracle_effective``: the filter with the true (K, D), sampled and ``K_eff``, ``D_eff``;
* ``oracle_ltv`` (05 D2, on the rungs with a nonlinear spring): ``K_k = K_0 phi_j(|tau_r,k|)``, ``K_0`` the sampled K,
  ``phi_j`` the catalogue curve in torque, with ``D_eff``;
* ``oracle_ltv_tangent``: the same ``K_k``, read as the curve's slope (``two_mass``, ``spring="tangent"``).  Not in 05:
  ``K_k delta_k`` steps at each knee crossing, this form does not.  Reported beside the spec's candidate, never for it;
* ``oracle_ltv_tangent[two_pass]``: the region re-selected by the first pass's spring torque ``S`` instead of
  ``|tau_r|`` and the bag filtered again; each pass is the same linear recursion, so the scan still applies;
* ``oracle_ltv_tangent[<curve>]``, ``[two_pass,<curve>]`` (``identified_filter/06`` Sec 1.4): both forms with the knee
  curve deliberately mis-specified (knee torques x 0.8 and x 1.25; the UR10 shoulders given the size-25 curve), on
  the consumer input of L2 and L3p.  Does the model depend on knowing the gear size exactly?
* ``oracle_ltv_tangent[lowpass]``, ``[lowpass_signed]`` (``identified_filter/07`` S1 (iii)): the region selected by
  ``|tau_r|`` low-passed below the modes (causal, critically damped, 2nd order, cutoff 0.5 x the platform's lowest
  main-joint f_n of the band table), or by ``tau_r`` low-passed and then rectified; one scan.  ``[fixed_point]`` is
  warm-started from ``[lowpass]``'s selection (07 S1.2), and ``[lowpass,<curve>]`` repeats Sec 1.4 with that selector
  (``[fixed_point,<curve>]`` with ``--fixed-point-sensitivity``, the selector r9 trains with);
* ``best_linear``: the per-bag LS (K, D) of that filter, fitted on the scored rows, so the best any constant linear
  pair can do on that bag.  In-sample on purpose: if it fails too, no linear pair passes.

Inputs, one set per variant: ``consumer`` (recorded ``q``, ``dq``, SG ``ddq``; L0 has none), ``sg_clean`` (clean
states, SG ``ddq``) and ``clean`` (clean states, ``ddq`` the central difference of the clean ``dq_motor``; the
simulator logs no acceleration, and this is its noise-free stand-in, 0.98 of the true gain at 5 % of the rate).

Gate G-phys, per main joint: oracle <= 0.9 x rigid in band, and <= nominal overall.

05 D3 decides where the gate applies: a joint-regime is gated when its bags' in-band elastic content (rigid's error
on the clean input) is at least 3 x the L0 oracle's in-band error.  The content is read on L2, the full spring without
TE, since TE in the numerator would let a larger TE gate more joints; the L3p reading is reported beside it.

06 Sec 1.2 (a), the tangent form's drift: K_k is selected by |tau_r,k|, not by the spring torque S, so S can leave the
catalogue curve f(delta).  On the L2 bags, every training window (burn-in + 1 s, half a window apart) is filtered from
rest with delta_0 = f^-1(tau_r,0), and max over the window of |S - f(delta)| / RMS(tau_s) is reported; 06 accepts 5 %.
07 S1 adds the low-passed selectors and the fixed point warm-started from ``lowpass``.

Privileged simulator truth throughout; nothing here touches a real file, a model or a training.

    uv run python tools/r8_physics_ladder.py
    uv run python tools/r8_physics_ladder.py --platform ur10 --robots 2 --no-report   # look, write nothing
    uv run python tools/r8_physics_ladder.py --rungs L0 L2 L3p --knee-blend 0.25 --prefix 05_blend25
    uv run python tools/r8_physics_ladder.py --rungs L0 L2 L3p --no-fit --prefix 06   # 06 Sec 1.2 (a) and 1.4
    uv run python tools/r8_physics_ladder.py --rungs L0 L2 L3p --no-fit --prefix 07   # 07 S1
    uv run python tools/r8_physics_ladder.py --rungs L0 L2 L3p --no-fit --fixed-point-sensitivity --prefix 07_fp
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import sys
from dataclasses import replace
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
_REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [os.fspath(_REPO / "src"), os.fspath(_REPO / "scripts")]

import numpy as np
import pandas as pd
import yaml
from scipy.signal import lfilter

from diagnose_controller_modes import _load_asset
from elastic_sim.dataset import assign_splits, generate, load_config, torque_knees
from elastic_sim.dataset_bounds import require_explicit_term_bound
from elastic_sim.diagnostics import savitzky_golay_acceleration
from elastic_sim.identification import FrictionModel
from elastic_sim.round7 import _block, _model_for, _rigid
from elastic_sim.round8 import (FIT_BAND, SG_BAND_FRACTION, _bandpass, effective_transmission,
                                fit_two_mass_multistart, regimes_of, resolve_round8)
from elastic_sim.two_mass import (KNEE_BLEND, catalogue_deflection, catalogue_torque, damping_from_ratio,
                                  filter_sequence, knee_factor, mean_diagonal)

RESULTS = _REPO / "REFACTOR_SPECS" / "identified_filter" / "results"
CAMPAIGN = _REPO / "config" / "campaigns" / "r8_campaign.yaml"
TE_TIER = _REPO / "config" / "campaigns" / "te_randomised.yaml"
RUNGS = ("L0", "L1", "L2", "L3", "L3p", "L4")
LTV_RUNGS = ("L2", "L3", "L3p", "L4")   # the rungs whose plant has the spring oracle_ltv's curve describes
REGIMES = {0: "full", 1: "real_like"}   # a test robot's bag index -> probe regime (round8.regime_for)
MAIN_JOINTS = {"iiwa_drive": ("iiwa_A1", "iiwa_A2", "iiwa_A4"),
               "ur10": ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint")}
ORACLES = ("oracle_effective", "oracle_sampled", "oracle_ltv", "oracle_ltv_tangent")
GATE_BAND_RATIO = 0.9
CONTENT_RATIO, CONTENT_RUNG = 3.0, "L2"   # 05 D3
REQUIRED_LTV = {"ur10": ("shoulder_lift_joint",)}   # 05 Sec 2.3
BOOTSTRAP = 2000
SENSITIVITY_RUNGS = ("L2", "L3p")   # 06 Sec 1.4
DRIFT_RUNG, DRIFT_LIMIT = "L2", 0.05   # 06 Sec 1.2 (a)
#: 06 Sec 1.4: the UR10 shoulders (size 32) given the elbow's size-25 curve.
SIZE25_SOURCE = {"ur10": ("elbow_joint", ("shoulder_pan_joint", "shoulder_lift_joint"))}
#: 07 S1 (iii): the low-passed selector's cutoff, x the platform's lowest main-joint f_n of the band table.
SELECTOR_CUTOFF_RATIO = 0.5
SELECTORS = ("abs", "lowpass", "lowpass_signed")


def rung_config(config, rung: str):
    """The config of a rung: L4 is the plant unchanged, every lower rung switches features off."""
    if rung == "L4":
        return config
    extras = replace(config.plant_extras, ripple_amplitude=0.0)
    if rung == "L3p":
        return replace(config, plant_extras=te_randomised(extras))
    if rung in ("L0", "L1", "L2"):
        extras = replace(extras, transmission_error_amplitude=())
    if rung in ("L0", "L1"):
        extras = replace(extras, stiffness_factors=(), stiffness_breakpoints=(), stiffness_breakpoints_torque=(),
                         stiffness_breakpoints_effort_fraction=())
    built = replace(config, plant_extras=extras)
    if rung == "L0":
        built = replace(built, noise=replace(config.noise, enabled=False),
                        signals=replace(config.signals, target_source="clean"))
    return built


def te_randomised(extras):
    """``extras`` with the 05 D1 tier on, at the config's own amplitude (the bound) and order (2n)."""
    tier = yaml.safe_load(TE_TIER.read_text(encoding="utf-8"))["transmission_error"]
    return replace(extras, transmission_error_tier=tier["tier"],
                   transmission_error_amplitude_factor=tuple(tier["amplitude_factor"]),
                   transmission_error_amplitude_n_factor=tuple(tier["amplitude_n_factor"]),
                   transmission_error_justification=tier["justification"])


def spring_curve(config, asset, blend: float = KNEE_BLEND) -> tuple[np.ndarray, np.ndarray, float]:
    """``(factors, knees, blend)`` of 05 D2's ``phi_j``: the catalogue curve the simulator's spring uses."""
    effort = np.asarray([np.inf if j.effort is None else j.effort for j in asset.resolve_active_joints()])
    return (np.asarray(config.plant_extras.stiffness_factors, dtype=float), torque_knees(config.plant_extras, effort),
            blend)


def spring_curves(config, asset, blend: float, platform: str, names: list[str]) -> dict[str, tuple]:
    """``label -> (factors, knees, blend)``: the catalogue curve (``""``) and 06 Sec 1.4's mis-specified ones."""
    factors, knees, blend = spring_curve(config, asset, blend)
    n = len(names)
    factors = np.broadcast_to(np.atleast_2d(factors), (n, np.atleast_2d(factors).shape[-1])).copy()
    knees = np.broadcast_to(np.atleast_2d(knees), (n, np.atleast_2d(knees).shape[-1])).copy()
    curves = {"": (factors, knees, blend), "knees_x0.8": (factors, 0.8 * knees, blend),
              "knees_x1.25": (factors, 1.25 * knees, blend)}
    if platform in SIZE25_SOURCE:
        source, targets = SIZE25_SOURCE[platform]
        swapped_f, swapped_k = factors.copy(), knees.copy()
        for target in targets:
            swapped_f[names.index(target)] = factors[names.index(source)]
            swapped_k[names.index(target)] = knees[names.index(source)]
        curves["shoulders_size25"] = (swapped_f, swapped_k, blend)
        # Which half of the size swap matters: its factors on the size-32 knees, its knees on the size-32 factors.
        curves["shoulders_size25_factors"] = (swapped_f, knees, blend)
        curves["shoulders_size25_knees"] = (factors, swapped_k, blend)
    return curves


def window_starts(steps: int, length: int) -> list[int]:
    """Training windows (``twomass_filter.window_spec``): ``length`` samples, half a window apart, the last flush."""
    if steps <= length:
        return [0]
    starts = list(range(0, steps - length + 1, max(1, length // 2)))
    return starts if starts[-1] == steps - length else starts + [steps - length]


class BlendedCurve:
    """The curve the tangent filter encodes when its region follows its own spring torque: ``df/ddelta = K_0 phi(|f|)``,
    ``phi`` blended (:func:`two_mass.knee_factor`).  Tabulated in torque, odd in ``delta``."""

    def __init__(self, stiffness: np.ndarray, factors: np.ndarray, knees: np.ndarray, blend: float, top: float):
        self.torque = np.linspace(0.0, top, 20001)
        n = len(stiffness)
        slope = stiffness * knee_factor(np.repeat(self.torque[:, None], n, axis=1), factors, knees, blend)
        inverse = 1.0 / slope
        self.deflection = np.concatenate([np.zeros((1, n)), np.cumsum(
            0.5 * (inverse[1:] + inverse[:-1]) * np.diff(self.torque)[:, None], axis=0)])
        self.top_slope = slope[-1]

    def __call__(self, deflection: np.ndarray) -> np.ndarray:
        magnitude = np.abs(deflection)
        out = np.empty_like(magnitude)
        for j in range(magnitude.shape[-1]):
            out[..., j] = np.interp(magnitude[..., j], self.deflection[:, j], self.torque)
            beyond = magnitude[..., j] > self.deflection[-1, j]
            out[..., j][beyond] = self.torque[-1] + self.top_slope[j] * (magnitude[..., j][beyond] - self.deflection[-1, j])
        return np.sign(deflection) * out

    def inverse(self, torque: np.ndarray) -> np.ndarray:
        magnitude = np.abs(torque)
        return np.sign(torque) * np.array([np.interp(magnitude[j], self.torque, self.deflection[:, j])
                                           for j in range(len(magnitude))])


def low_passed(x: np.ndarray, cutoff_hz: float, step: float) -> np.ndarray:
    """Causal, critically damped 2nd-order low-pass along time: two identical first-order stages (a double pole at
    ``cutoff_hz``), each the exact discretisation of ``1 / (1 + s / w)``, started at rest on ``x[0]``."""
    gain = 1.0 - math.exp(-2.0 * math.pi * cutoff_hz * step)
    out = np.asarray(x, dtype=float)
    for _ in range(2):
        out = lfilter([gain], [1.0, gain - 1.0], out, axis=0, zi=(1.0 - gain) * out[:1])[0]
    return out


def selection(tau_r: np.ndarray, name: str, cutoff_hz: float, step: float) -> np.ndarray:
    """The torque whose magnitude selects the knee region: ``abs`` ``tau_r`` itself (05 D2), ``lowpass`` ``|tau_r|``
    low-passed (07 S1 (iii)), ``lowpass_signed`` ``tau_r`` low-passed (rectified by :func:`knee_factor`)."""
    if name == "abs":
        return tau_r
    return low_passed(np.abs(tau_r) if name == "lowpass" else tau_r, cutoff_hz, step)


def tangent_pass(tau_r, inertia, stiffness, damping, curve, step, selector, initial_deflection=None) -> dict:
    """One scan of the tangent filter, its region selected by ``|selector|``; adds ``spring`` = ``S``."""
    factors, knees, blend = curve
    varying = stiffness * knee_factor(selector, factors, knees, blend)
    result = filter_sequence(tau_r, inertia, varying, damping, step, coupling="full", spring="tangent",
                             initial_deflection=initial_deflection)
    result["spring"] = result["tau_s"] - damping * result["delta_dot"]
    return result


#: The fixed point of the spring-torque selection: at most this many passes, until ``S`` moves less than the tolerance.
FIXED_POINT_PASSES, FIXED_POINT_TOLERANCE = 10, 1e-6


def fixed_point_pass(tau_r, inertia, stiffness, damping, curve, step, initial_deflection=None,
                     start=None) -> tuple[dict, int]:
    """The tangent filter with its region selected by its own ``S``: passes repeated until ``S`` stops moving.

    The first pass is selected by ``start`` (default ``tau_r``; 07 S1.2 warm-starts from the low-passed selector)."""
    result = tangent_pass(tau_r, inertia, stiffness, damping, curve, step, tau_r if start is None else start,
                          initial_deflection)
    tolerance = FIXED_POINT_TOLERANCE * max(1.0, float(np.abs(tau_r).max()))
    for passes in range(2, FIXED_POINT_PASSES + 1):
        new = tangent_pass(tau_r, inertia, stiffness, damping, curve, step, result["spring"], initial_deflection)
        moved = float(np.abs(new["spring"] - result["spring"]).max())
        result = new
        if moved < tolerance:
            return result, passes
    return result, FIXED_POINT_PASSES


def spring_drift(tau_r: np.ndarray, inertia: np.ndarray, stiffness: np.ndarray, damping: np.ndarray,
                 curve: tuple, step: float, window: int, burn_in: int = 0,
                 cutoff_hz: float | None = None) -> dict[str, np.ndarray]:
    """06 Sec 1.2 (a), each ``[windows, n]``: max over each window of ``|S - f(delta)|`` over the window's RMS ``tau_s``.

    ``catalogue``: ``f`` the plant's unblended curve; ``blended``: ``f`` the curve the filter encodes
    (:class:`BlendedCurve`), so only the selector's share of the drift is left.  ``two_pass_*``: the same with the
    region re-selected by the first pass's ``S`` and the window filtered again, still a linear recursion per pass;
    ``fixed_point_*``: passes repeated to convergence (:func:`fixed_point_pass`), warm-started from ``lowpass`` when
    ``cutoff_hz`` is given; ``lowpass_*``, ``lowpass_signed_*``: one pass with 07 S1 (iii)'s selectors, each
    low-pass started on the window's first sample.  ``*_scored``: only the window's rows past ``burn_in`` samples,
    the rows a training scores; ``passes``: what the fixed point needed.
    """
    factors, knees, blend = curve
    blended = BlendedCurve(stiffness, factors, knees, blend, top=3.0 * float(np.max(np.abs(tau_r))) + 3.0 * float(knees.max()))
    out: dict[str, list] = {"passes": []}
    for start in window_starts(len(tau_r), window):
        part = slice(start, start + window)
        initial = catalogue_deflection(tau_r[start], stiffness, factors, knees)
        first = tangent_pass(tau_r[part], inertia[part], stiffness, damping, curve, step, tau_r[part], initial)
        second = tangent_pass(tau_r[part], inertia[part], stiffness, damping, curve, step, first["spring"], initial)
        results = {"": first, "two_pass_": second}
        warm = None
        if cutoff_hz is not None:
            for name in SELECTORS[1:]:
                chosen = selection(tau_r[part], name, cutoff_hz, step)
                results[name + "_"] = tangent_pass(tau_r[part], inertia[part], stiffness, damping, curve, step, chosen,
                                                   initial)
                warm = chosen if name == "lowpass" else warm
        fixed, passes = fixed_point_pass(tau_r[part], inertia[part], stiffness, damping, curve, step, initial, warm)
        results["fixed_point_"] = fixed
        out["passes"].append(np.full(tau_r.shape[1], float(passes)))
        for prefix, result in results.items():
            rms = np.sqrt(np.mean(result["tau_s"] ** 2, axis=0))
            shifted = result["delta"] - initial + blended.inverse(tau_r[start])
            errors = {"catalogue": np.abs(result["spring"] - catalogue_torque(result["delta"], stiffness, factors, knees)),
                      "blended": np.abs(result["spring"] - blended(shifted))}
            for name, error in errors.items():
                out.setdefault(prefix + name, []).append(error.max(axis=0) / rms)
                out.setdefault(prefix + name + "_scored", []).append(error[burn_in:].max(axis=0) / rms)
    return {key: np.asarray(value) for key, value in out.items()}


def te_draws(frame: pd.DataFrame) -> pd.DataFrame:
    """The amplitudes and phases L3p actually used, per bag and joint, with the robot's K."""
    rows = []
    for bag, group in frame.groupby("bag", sort=False):
        first = group.iloc[0]
        for column in (c for c in group.columns if c.startswith("te_amplitude__")):
            joint = column.split("__", 1)[1]
            rows.append({"bag": bag, "tier": first["tier"], "joint": joint, "K": first[f"stiffness__{joint}"],
                         "A_2n": first[column], "order_2n": first[f"te_order__{joint}"],
                         "phase_2n": first[f"te_phase__{joint}"], "A_n": first[f"te_amplitude_n__{joint}"],
                         "order_n": first[f"te_order_n__{joint}"], "phase_n": first[f"te_phase_n__{joint}"]})
    return pd.DataFrame(rows)


def inputs_of(group: pd.DataFrame, n: int, step: float, window: int, poly: int, rung: str) -> dict:
    """``variant -> (q, dq, ddq)``.  L0 is noise-free, so its consumer signals would only repeat ``sg_clean``."""
    q_clean, dq_clean = _block(group, "q_motor_clean", n), _block(group, "dq_motor_clean", n)
    sg = lambda dq: savitzky_golay_acceleration(dq, step, window=window, order=poly)  # noqa: E731
    out = {"clean": (q_clean, dq_clean, np.gradient(dq_clean, step, axis=0)),
           "sg_clean": (q_clean, dq_clean, sg(dq_clean))}
    if rung != "L0":
        q, dq = _block(group, "q", n), _block(group, "dq", n)
        out["consumer"] = (q, dq, sg(dq))
    return out


def delayed(x: np.ndarray, samples: int) -> np.ndarray:
    """``x`` as a recorder ``samples`` old reads it; the first rows hold the initial value."""
    return x if samples <= 0 else np.concatenate([np.repeat(x[:1], samples, axis=0), x[:-samples]])


def recorded_lag(group: pd.DataFrame, candidates: range = range(6)) -> int:
    """How many samples ``q0`` is behind ``q_motor_clean0``: the delay the bag's recorded signals carry."""
    q, clean = group["q0"].to_numpy(), group["q_motor_clean0"].to_numpy()
    return min(candidates, key=lambda lag: float(np.mean((q - delayed(clean, lag)) ** 2)))


def score_bag(group: pd.DataFrame, manifest: dict, contract: dict, asset, rung: str, regime: str,
              fit: bool = True, curves: dict | None = None,
              cutoff_hz: float | None = None, fixed_point_sensitivity: bool = False) -> tuple[list[dict], list[dict]]:
    """One row per variant, candidate and joint: overall and in-band MSE on the kept rows, with the pair used.

    Second, on the drift rung's consumer input, one row per training window and joint of :func:`spring_drift`.

    The recorded ``q``, ``dq`` and ``ft`` are all ``delay_samples`` old and ``ft_clean`` is not, so the consumer
    variant is scored on ``ft_clean`` delayed alike: otherwise a sample of lag (16 % of the signal at 13 Hz and
    500 Hz) would be charged to the oracle.
    """
    n, names = int(manifest["n_dof"]), list(manifest["joint_names"])
    step = float(manifest["sample_time_step"])
    differentiation, priors = contract["differentiation"], contract["transmission_priors"]
    window, poly = int(differentiation["sg_window"]), int(differentiation["sg_poly"])
    rate, f_sg = 1.0 / step, SG_BAND_FRACTION / step
    clean_target = _block(group, "ft_clean", n)
    lag = recorded_lag(group)
    time = group["t"].to_numpy(dtype=float)
    kept = (time - time[0]) >= float(priors["burn_in_s"]) - 1e-9
    friction = FrictionModel(*(np.asarray([float(group[f"{law}__{x}"].iloc[0]) for x in names])
                               for law in ("link_viscous", "link_coulomb")))
    stiffness = np.asarray([float(group[f"stiffness__{x}"].iloc[0]) for x in names])
    damping = np.asarray([float(group[f"damping__{x}"].iloc[0]) for x in names])
    nominal_k = np.asarray([j["stiffness_nominal"] for j in priors["joints"]], dtype=float)
    centre_zeta = np.asarray([0.5 * sum(j["damping_ratio_range"]) for j in priors["joints"]], dtype=float)
    k_eff, d_eff = effective_transmission(group, n, stiffness, damping)
    tier = str(group["tier"].iloc[0])
    robot = next(r for r in manifest["robots"] if r["name"] == tier)
    f_n = np.sqrt(stiffness / np.asarray(robot["plant_link_inertia_median"], float)) / (2.0 * math.pi)
    free = f_n <= f_sg
    model = _model_for(asset, None, {})
    rows, drifts = [], []
    curve = None if curves is None else curves[""]
    for variant, (q, dq, ddq) in inputs_of(group, n, step, window, poly, rung).items():
        target = delayed(clean_target, lag) if variant == "consumer" else clean_target
        tau_rnea, inertia = _rigid(*model, q, dq, ddq)
        tau_r = tau_rnea + friction.torque(dq)
        pairs = {"nominal": (nominal_k, damping_from_ratio(nominal_k, centre_zeta, mean_diagonal(inertia))),
                 "oracle_sampled": (stiffness, damping), "oracle_effective": (k_eff, d_eff)}
        if fit and variant != "sg_clean":
            low = FIT_BAND[0] * float(np.min(f_n[free]))
            band = _bandpass(np.column_stack([tau_r, target]), low, 0.95 * f_sg, rate)
            pairs["best_linear"] = fit_two_mass_multistart(band[:, :n], inertia, band[:, n:], kept, step,
                                                           stiffness, damping, free)
        predictions = {"rigid": (tau_r, None)}
        for label, pair in pairs.items():
            predictions[label] = (filter_sequence(tau_r, inertia, *pair, step, coupling="full")["tau_s"], pair)
        if curve is not None and rung in LTV_RUNGS:
            varying = stiffness * knee_factor(tau_r, *curve)
            predictions["oracle_ltv"] = (filter_sequence(tau_r, inertia, varying, d_eff, step, coupling="full")["tau_s"],
                                         (stiffness, d_eff))
            predictions["oracle_ltv_tangent"] = (filter_sequence(tau_r, inertia, varying, d_eff, step, coupling="full",
                                                                 spring="tangent")["tau_s"], (stiffness, d_eff))
            if variant == "consumer" and rung in SENSITIVITY_RUNGS:
                first = tangent_pass(tau_r, inertia, stiffness, d_eff, curve, step, tau_r)
                predictions["oracle_ltv_tangent[two_pass]"] = (tangent_pass(
                    tau_r, inertia, stiffness, d_eff, curve, step, first["spring"])["tau_s"], (stiffness, d_eff))
                chosen = {name: selection(tau_r, name, cutoff_hz, step) for name in SELECTORS[1:]} \
                    if cutoff_hz is not None else {}
                for name, selector in chosen.items():
                    predictions[f"oracle_ltv_tangent[{name}]"] = (tangent_pass(
                        tau_r, inertia, stiffness, d_eff, curve, step, selector)["tau_s"], (stiffness, d_eff))
                predictions["oracle_ltv_tangent[fixed_point]"] = (fixed_point_pass(
                    tau_r, inertia, stiffness, d_eff, curve, step, start=chosen.get("lowpass"))[0]["tau_s"],
                    (stiffness, d_eff))
                for label, other in curves.items():
                    if label:
                        wrong = tangent_pass(tau_r, inertia, stiffness, d_eff, other, step, tau_r)
                        predictions[f"oracle_ltv_tangent[{label}]"] = (wrong["tau_s"], (stiffness, d_eff))
                        if "lowpass" in chosen and fixed_point_sensitivity:
                            # 07 S1: the selector r9 trains with, on the wrong curve (warm-started on it too).
                            predictions[f"oracle_ltv_tangent[fixed_point,{label}]"] = (fixed_point_pass(
                                tau_r, inertia, stiffness, d_eff, other, step, start=chosen["lowpass"])[0]["tau_s"],
                                (stiffness, d_eff))
                        elif "lowpass" in chosen:
                            predictions[f"oracle_ltv_tangent[lowpass,{label}]"] = (tangent_pass(
                                tau_r, inertia, stiffness, d_eff, other, step, chosen["lowpass"])["tau_s"],
                                (stiffness, d_eff))
            if variant == "consumer" and rung == DRIFT_RUNG:
                window = int(round((float(priors["burn_in_s"]) + 1.0) / step))
                measured = spring_drift(tau_r, inertia, stiffness, d_eff, curve, step, window,
                                        int(round(float(priors["burn_in_s"]) / step)), cutoff_hz)
                for w in range(len(measured["catalogue"])):
                    drifts += [{"bag": group["bag"].iloc[0], "tier": tier, "regime": regime, "window": w, "joint": name,
                                **{f"drift_{key}": float(value[w, j]) for key, value in measured.items()}}
                               for j, name in enumerate(names)]
        for label, (prediction, pair) in predictions.items():
            error = prediction - target
            for j, name in enumerate(names):
                band_error = _bandpass(error[:, [j]], 0.5 * f_n[j], 2.0 * f_n[j], rate) if free[j] else None
                rows.append({"bag": group["bag"].iloc[0], "tier": tier, "regime": regime,
                             "rung": rung, "variant": variant, "candidate": label, "joint": name, "lag": lag,
                             "f_n": float(f_n[j]), "K": float(stiffness[j]), "D": float(damping[j]),
                             "K_used": float(pair[0][j]) if pair else float("nan"),
                             "D_used": float(pair[1][j]) if pair else float("nan"),
                             "mse": float(np.mean(error[kept, j] ** 2)),
                             "mse_band": float(np.mean(band_error[kept, 0] ** 2)) if band_error is not None
                             else float("nan")})
    return rows, drifts


def _score_star(arguments) -> tuple[list[dict], list[dict]]:
    return score_bag(*arguments)


def selector_cutoff(config, platform: str) -> float:
    """07 S1 (iii): ``SELECTOR_CUTOFF_RATIO`` x the lowest ``f_n_min`` of the platform's main joints in the band table."""
    return SELECTOR_CUTOFF_RATIO * min(float(row["f_n_min_hz"]) for row in config.round8.band
                                       if row["joint"] in MAIN_JOINTS[platform])


def run_platform(platform: str, spec: dict, robots: int, jobs: int, fit: bool, rungs=RUNGS,
                 blend: float = KNEE_BLEND, fixed_point_sensitivity: bool = False) -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
    """Simulate every rung of the platform's first ``robots`` test robots; scores, integrity checks, TE draws, drift."""
    asset_config = load_config(_REPO / spec["config"])
    asset = _load_asset(asset_config.asset)
    config = resolve_round8(asset_config, asset)
    labels = assign_splits(config.tiers, config.split)
    tiers = [t.name for t in config.tiers if not t.is_rigid and labels[t.name] == "test"][:robots]
    selection = replace(config, only_bags=tuple((t, i) for t in tiers for i in REGIMES), rigid_reference=False)
    stem = _REPO / settings_data_root() / platform / spec["stem"]
    contract = json.loads(stem.with_suffix(".contract.json").read_text(encoding="utf-8"))
    cache: dict = {}
    tables, references, checks = [], None, {"robots": tiers, "bags": len(tiers) * len(REGIMES)}
    frames, draws, drifts = {}, pd.DataFrame(), []
    curves = spring_curves(config, asset, blend, platform, [j["joint"] for j in contract["transmission_priors"]["joints"]])
    cutoff = selector_cutoff(config, platform)
    checks["selector_cutoff_hz"] = cutoff
    if "L3p" in rungs:
        require_explicit_term_bound(rung_config(config, "L3p"), source="L3p", asset=asset)
        checks["explicit_bound_L3p"] = True
    for rung in rungs:
        print(f"[{platform}] {rung}: simulating {checks['bags']} bags", flush=True)
        frame, manifest, _ = generate(rung_config(selection, rung), asset, verbose=False, jobs=jobs,
                                      trajectory_cache=cache)
        reference = frame.groupby("bag", sort=False)[[c for c in frame.columns if c.startswith("q_ref_clean")]].sum()
        references = reference if references is None else references
        checks[f"same_reference_{rung}"] = bool(np.allclose(reference.to_numpy(), references.to_numpy(),
                                                            rtol=0.0, atol=1e-9))
        groups = [group for _, group in frame.groupby("bag", sort=False)]
        regimes = regimes_of(selection, frame)
        with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as pool:
            scored = list(pool.map(_score_star, [(g, manifest, contract, asset, rung, regimes[g["bag"].iloc[0]], fit,
                                                  curves, cutoff, fixed_point_sensitivity) for g in groups]))
            tables += [row for rows, _ in scored for row in rows]
            drifts += [row for _, rows in scored for row in rows]
        if rung in ("L3", "L4"):
            frames[rung] = frame
        if rung == "L3p":
            draws = te_draws(frame)
    if len(frames) == 2:
        columns = [c for c in frames["L4"].columns if c.startswith(("ft_clean", "dq_motor_clean"))]
        checks["L4_equals_L3"] = bool(np.array_equal(frames["L3"][columns].to_numpy(), frames["L4"][columns].to_numpy()))
    if "L4" in frames:
        checks["L4_equals_production"] = _matches_production(frames["L4"], stem.with_suffix(".parquet"))
    return pd.DataFrame(tables), checks, draws, pd.DataFrame(drifts)


def _matches_production(frame: pd.DataFrame, path: Path) -> bool:
    """The rebuilt L4 bags are the campaign's test bags: same target and reference, row for row, at the file's precision."""
    columns = [c for c in frame.columns if c.startswith(("ft_clean", "q_ref_clean"))]
    stored = pd.read_parquet(path, columns=["bag", *columns], filters=[("bag", "in", list(frame["bag"].unique()))])
    if len(stored) != len(frame):
        return False
    order = list(frame["bag"].unique())
    ours = frame.sort_values("bag", key=lambda s: s.map(order.index), kind="stable")[columns].to_numpy()
    theirs = stored.sort_values("bag", key=lambda s: s.map(order.index), kind="stable")[columns].to_numpy()
    return bool(np.array_equal(ours.astype(np.float32), theirs.astype(np.float32)))   # the file stores float32


def settings_data_root() -> str:
    return yaml.safe_load(CAMPAIGN.read_text(encoding="utf-8"))["data_root"]


def pooled(table: pd.DataFrame) -> pd.DataFrame:
    """RMSE pooled over the bags (root of the mean MSE), per regime, rung, variant, candidate and joint."""
    keys = ["regime", "rung", "variant", "candidate", "joint"]
    out = table.groupby(keys, sort=False).agg(bags=("bag", "nunique"), mse=("mse", "mean"),
                                              mse_band=("mse_band", "mean"), f_n=("f_n", "median")).reset_index()
    out["rmse"], out["rmse_band"] = np.sqrt(out["mse"]), np.sqrt(out["mse_band"])
    return out.drop(columns=["mse", "mse_band"])


def _bootstrap_ratio(a: np.ndarray, b: np.ndarray, seed: int = 0) -> tuple[float, float]:
    """95 % interval, bags resampled, of ``sqrt(mean a / mean b)``."""
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(a), size=(BOOTSTRAP, len(a)))
    ratios = np.sqrt(a[draws].mean(axis=1) / b[draws].mean(axis=1))
    return float(np.quantile(ratios, 0.025)), float(np.quantile(ratios, 0.975))


def gate(table: pd.DataFrame, platform: str, variant: str = "consumer") -> pd.DataFrame:
    """G-phys per regime, rung, main joint and oracle.  L0 has no consumer signals: it is read on ``clean``."""
    rows = []
    main = MAIN_JOINTS[platform]
    for (regime, rung), part in table[table["joint"].isin(main)].groupby(["regime", "rung"], sort=False):
        chosen = part[part["variant"] == ("clean" if rung == "L0" else variant)]
        for joint in main:
            at = chosen[chosen["joint"] == joint].set_index(["candidate", "bag"])
            if not len(at):
                continue
            rigid, nominal = at.loc["rigid"], at.loc["nominal"]
            present = list(dict.fromkeys(at.index.get_level_values("candidate")))
            for oracle in ORACLES + tuple(c for c in present if c.startswith("oracle_ltv_tangent[")):
                if oracle not in present:
                    continue
                row = at.loc[oracle].reindex(rigid.index)
                band = math.sqrt(row["mse_band"].mean() / rigid["mse_band"].mean())
                overall = math.sqrt(row["mse"].mean() / nominal["mse"].reindex(rigid.index).mean())
                low, high = _bootstrap_ratio(row["mse_band"].to_numpy(), rigid["mse_band"].to_numpy())
                rows.append({"regime": regime, "rung": rung, "joint": joint, "oracle": oracle,
                             "band_vs_rigid": band, "band_vs_rigid_low": low, "band_vs_rigid_high": high,
                             "overall_vs_nominal": overall,
                             "passes": bool(band <= GATE_BAND_RATIO and overall <= 1.0)})
    return pd.DataFrame(rows)


def first_failures(verdict: pd.DataFrame) -> pd.DataFrame:
    """Per regime, joint and oracle: the first rung failing G-phys, and by how much."""
    rows = []
    for (regime, joint, oracle), part in verdict.groupby(["regime", "joint", "oracle"], sort=False):
        failed = part[~part["passes"]].sort_values("rung")
        first = failed.iloc[0] if len(failed) else None
        rows.append({"regime": regime, "joint": joint, "oracle": oracle,
                     "first_fail": first["rung"] if first is not None else "none",
                     "band_vs_rigid": first["band_vs_rigid"] if first is not None else float("nan"),
                     "overall_vs_nominal": first["overall_vs_nominal"] if first is not None else float("nan")})
    return pd.DataFrame(rows)


def content_gating(table: pd.DataFrame, platform: str) -> pd.DataFrame:
    """05 D3, per regime and main joint: the in-band elastic content against the L0 oracle's in-band error."""
    clean = table[(table["variant"] == "clean") & table["joint"].isin(MAIN_JOINTS[platform])]
    rows = []
    for (regime, joint), part in clean.groupby(["regime", "joint"], sort=False):
        band = lambda rung, candidate: math.sqrt(  # noqa: E731
            part[(part["rung"] == rung) & (part["candidate"] == candidate)]["mse_band"].mean())
        floor = band("L0", "oracle_effective")
        row = {"regime": regime, "joint": joint, "oracle_L0": floor}
        for rung in ("L0", CONTENT_RUNG, "L3p"):
            row[f"content_{rung}"] = band(rung, "rigid")
            row[f"ratio_{rung}"] = row[f"content_{rung}"] / floor
        row["gated"] = bool(row[f"ratio_{CONTENT_RUNG}"] >= CONTENT_RATIO)
        row["gated_on_L3p_bags"] = bool(row["ratio_L3p"] >= CONTENT_RATIO)
        rows.append(row)
    return pd.DataFrame(rows)


def step1_gate(verdict: pd.DataFrame, gating: pd.DataFrame, platform: str) -> pd.DataFrame:
    """05 Sec 2.3, per main joint on ``full`` at L3p: G-phys with either oracle; a required joint needs oracle-LTV.

    ``passes`` is the spec's candidates only.  ``passes_tangent`` is the same rule with the tangent form as the LTV.
    """
    rows = []
    at = verdict[(verdict["regime"] == "full") & (verdict["rung"] == "L3p")].set_index(["joint", "oracle"])
    for _, joint in gating[gating["regime"] == "full"].iterrows():
        ltv, effective = at.loc[(joint["joint"], "oracle_ltv")], at.loc[(joint["joint"], "oracle_effective")]
        tangent = at.loc[(joint["joint"], "oracle_ltv_tangent")]
        required = joint["joint"] in REQUIRED_LTV.get(platform, ())
        passes = bool(ltv["passes"]) if required else bool(ltv["passes"] or effective["passes"])
        passes_tangent = bool(tangent["passes"]) if required else bool(tangent["passes"] or effective["passes"])
        rows.append({"joint": joint["joint"], "gated": joint["gated"], "gated_on_L3p_bags": joint["gated_on_L3p_bags"],
                     "ltv_required": required, "ltv_band": ltv["band_vs_rigid"], "ltv_overall": ltv["overall_vs_nominal"],
                     "ltv_low": ltv["band_vs_rigid_low"], "ltv_high": ltv["band_vs_rigid_high"],
                     "effective_band": effective["band_vs_rigid"], "effective_overall": effective["overall_vs_nominal"],
                     "tangent_band": tangent["band_vs_rigid"], "tangent_overall": tangent["overall_vs_nominal"],
                     "tangent_low": tangent["band_vs_rigid_low"], "tangent_high": tangent["band_vs_rigid_high"],
                     "passes": passes, "passes_tangent": passes_tangent})
    return pd.DataFrame(rows)


def sensitivity_gate(verdict: pd.DataFrame, gating: pd.DataFrame) -> pd.DataFrame:
    """06 Sec 1.4, per gated joint-regime, rung and curve: G-phys of the tangent oracle with that curve.

    The tangent oracle itself must pass: oracle-effective reads no curve, so letting it stand in would make the
    check vacuous.
    """
    rows = []
    gated = gating[gating["gated"]]
    for _, joint in gated.iterrows():
        part = verdict[(verdict["regime"] == joint["regime"]) & (verdict["joint"] == joint["joint"])
                       & verdict["rung"].isin(SENSITIVITY_RUNGS) & verdict["oracle"].str.startswith("oracle_ltv_tangent")]
        for _, row in part.iterrows():
            label = row["oracle"][len("oracle_ltv_tangent"):].strip("[]")
            form = next((f for f in ("two_pass", "fixed_point", "lowpass_signed", "lowpass") if label.startswith(f)),
                        "one_pass")
            label = label.removeprefix(form).strip(",") or "catalogue"
            rows.append({"regime": row["regime"], "rung": row["rung"], "joint": row["joint"], "form": form, "curve": label,
                         "band_vs_rigid": row["band_vs_rigid"], "band_high": row["band_vs_rigid_high"],
                         "overall_vs_nominal": row["overall_vs_nominal"], "passes": row["passes"]})
    return pd.DataFrame(rows)


def drift_summary(drift: pd.DataFrame, platform: str) -> pd.DataFrame:
    """06 Sec 1.2 (a), per regime, main joint and measure over the L2 windows: median, 95th percentile, worst drift.

    ``accepted``: every window within the limit.
    """
    if drift.empty:
        return pd.DataFrame()
    part = drift[drift["joint"].isin(MAIN_JOINTS[platform])]
    rows = []
    for (regime, joint), at in part.groupby(["regime", "joint"], sort=False):
        for column in (c for c in at.columns if c.startswith("drift_") and c != "drift_passes"):
            values = at[column].to_numpy()
            rows.append({"regime": regime, "joint": joint, "measure": column[len("drift_"):], "windows": len(values),
                         "median": float(np.median(values)), "p95": float(np.quantile(values, 0.95)),
                         "worst": float(values.max()), "accepted": bool(values.max() <= DRIFT_LIMIT),
                         "fixed_point_passes_max": float(at["drift_passes"].max())})
    return pd.DataFrame(rows)


def _markdown_table(frame: pd.DataFrame, digits: int = 3) -> list[str]:
    def cell(value):
        return f"{value:.{digits}g}" if isinstance(value, float) else str(value)

    return (["| " + " | ".join(frame.columns) + " |", "|" + "---|" * len(frame.columns)]
            + ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False)])


def ladder_view(pool: pd.DataFrame, platform: str, regime: str) -> pd.DataFrame:
    """One row per rung and main joint: every candidate's in-band and overall RMSE on the gate's input."""
    part = pool[(pool["regime"] == regime) & pool["joint"].isin(MAIN_JOINTS[platform])]
    part = part[((part["rung"] == "L0") & (part["variant"] == "clean"))
                | ((part["rung"] != "L0") & (part["variant"] == "consumer"))]
    wide = part.pivot_table(index=["rung", "joint"], columns="candidate", values=["rmse_band", "rmse"], sort=False)
    out = pd.DataFrame(index=wide.index)
    for candidate in ("rigid", "nominal", "oracle_effective", "oracle_ltv", "oracle_ltv_tangent", "oracle_sampled",
                      "best_linear"):
        if ("rmse_band", candidate) in wide.columns:
            out[f"{candidate} band"] = wide[("rmse_band", candidate)]
            out[f"{candidate} all"] = wide[("rmse", candidate)]
    return out.reset_index()


def clean_view(pool: pd.DataFrame, platform: str, regime: str) -> pd.DataFrame:
    """L3, L3p, L4: the effective oracle against rigid on each input, which separates TE from differentiation."""
    part = pool[(pool["regime"] == regime) & pool["joint"].isin(MAIN_JOINTS[platform])
                & pool["rung"].isin(("L3", "L3p", "L4"))]
    wide = part[part["candidate"].isin(("rigid", "oracle_effective"))].pivot_table(
        index=["rung", "joint"], columns=["variant", "candidate"], values="rmse_band", sort=False)
    out = pd.DataFrame(index=wide.index)
    for variant in ("consumer", "sg_clean", "clean"):
        out[f"{variant}: oracle / rigid"] = wide[(variant, "oracle_effective")] / wide[(variant, "rigid")]
        out[f"{variant}: oracle band"] = wide[(variant, "oracle_effective")]
    return out.reset_index()


def report(platform: str, table: pd.DataFrame, checks: dict, drift: pd.DataFrame | None = None) \
        -> tuple[list[str], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pool = pooled(table)
    verdict = gate(table, platform)
    lines = [f"## {platform}", "", f"Integrity: `{json.dumps({k: v for k, v in checks.items() if k != 'robots'})}`.", ""]
    for regime in REGIMES.values():
        if regime not in set(table["regime"]):
            continue
        lines += [f"### {regime}: RMSE [N·m], pooled over the bags", ""]
        lines += _markdown_table(ladder_view(pool, platform, regime)) + [""]
        if {"L3", "L3p", "L4"} & set(table["rung"]):
            lines += [f"### {regime}: L3, L3p, L4 by input (in band)", ""] + _markdown_table(clean_view(pool, platform, regime)) + [""]
    for label, frame in (("G-phys per rung", verdict.drop(columns=["band_vs_rigid_low", "band_vs_rigid_high"])),
                         ("First failing rung", first_failures(verdict))):
        lines += [f"### {label}", ""] + _markdown_table(frame) + [""]
    gating, step, sensitivity = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if {"L0", CONTENT_RUNG, "L3p"} <= set(table["rung"]):
        gating = content_gating(table, platform)
        step = step1_gate(verdict, gating, platform)
        sensitivity = sensitivity_gate(verdict, gating)
        lines += ["### D3 gating", ""] + _markdown_table(gating) + [""]
        lines += ["### 05 Sec 2.3 gate on `full` at L3p", ""] + _markdown_table(step) + [""]
        lines += ["### 06 Sec 1.4: mis-specified knee curve, gated joints", ""] + _markdown_table(sensitivity) + [""]
    if drift is not None and not drift.empty:
        lines += [f"### 06 Sec 1.2 (a): tangent drift on {DRIFT_RUNG} windows", ""] \
            + _markdown_table(drift_summary(drift, platform)) + [""]
    return lines, verdict, gating, step, sensitivity


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--platform", nargs="+", default=None, help="default: every platform of the campaign")
    parser.add_argument("--robots", type=int, default=10, help="test robots per platform (04 A1: 10)")
    parser.add_argument("--rungs", nargs="+", default=list(RUNGS), choices=RUNGS)
    parser.add_argument("--no-fit", action="store_true", help="skip the per-bag LS fit (much faster)")
    parser.add_argument("--knee-blend", type=float, default=KNEE_BLEND, help="oracle_ltv's blend half-width, x the knee")
    parser.add_argument("--prefix", default="07", help="results file prefix")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--no-report", action="store_true", help="print only; write nothing")
    parser.add_argument("--fixed-point-sensitivity", action="store_true",
                        help="06 Sec 1.4's wrong curves under the fixed-point selector (07 S1), not the low-passed one")
    args = parser.parse_args(argv)

    settings = yaml.safe_load(CAMPAIGN.read_text(encoding="utf-8"))
    out = ["# Physics-oracle ladder (identified_filter/04 A1, 05 Sec 2)", ""]
    tables, verdicts, gatings, steps, draws, sensitivities, drifts, checks = [], [], [], [], [], [], [], {}
    for platform in args.platform or settings["platforms"]:
        table, checks[platform], drawn, drift = run_platform(platform, settings["platforms"][platform], args.robots,
                                                             args.jobs, not args.no_fit, args.rungs, args.knee_blend,
                                                             args.fixed_point_sensitivity)
        section, verdict, gating, step, sensitivity = report(platform, table, checks[platform], drift)
        print("\n".join(section), flush=True)
        out += section
        tables.append(table.assign(platform=platform))
        verdicts.append(verdict.assign(platform=platform))
        gatings.append(gating.assign(platform=platform))
        steps.append(step.assign(platform=platform))
        draws.append(drawn.assign(platform=platform))
        sensitivities.append(sensitivity.assign(platform=platform))
        drifts.append(drift.assign(platform=platform))
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / f"{args.prefix}_ladder.md").write_text("\n".join(out), encoding="utf-8")
        pd.concat(tables).to_csv(RESULTS / f"{args.prefix}_ladder.csv", index=False)
        pd.concat(verdicts).to_csv(RESULTS / f"{args.prefix}_ladder_gate.csv", index=False)
        pd.concat(gatings).to_csv(RESULTS / f"{args.prefix}_d3_gating.csv", index=False)
        pd.concat(steps).to_csv(RESULTS / f"{args.prefix}_step1_gate.csv", index=False)
        pd.concat(draws).to_csv(RESULTS / f"{args.prefix}_te_draws.csv", index=False)
        pd.concat(sensitivities).to_csv(RESULTS / f"{args.prefix}_sensitivity.csv", index=False)
        pd.concat(drifts).to_csv(RESULTS / f"{args.prefix}_drift.csv", index=False)
        (RESULTS / f"{args.prefix}_ladder_checks.json").write_text(json.dumps(checks, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
