"""FR_04 contract blocks of a round-7 (torque-free comparison) file: S-2, S-3, S-4.

* ``audit_columns`` (S-2): the simulator's noise-free link-side truth already
  in every round-6/7 file (``q_link*``, ``dq_link*``, ``defl*``), declared for
  checking a model's latent state, never as an input or target.
* ``transmission_priors`` (S-3): per joint, the stiffness prior of the
  manifest's ``transmission_sampling``, and the damping-ratio and mode
  frequency ranges over the dataset's robots, in the terms of the FR_04 Sec 0
  filter (see :func:`transmission_priors`); plus the burn-in they give, or the
  config's ``transmission_priors.burn_in_override_s`` (FR_04 D-2).
* ``baselines.two_mass`` (S-4): ``rigid_elastic_nominal`` and
  ``rigid_elastic_oracle`` per split, on the rows the consumer scores
  (each bag's first ``burn_in_s`` excluded), with the filter of
  :mod:`elastic_sim.two_mass` (its default trapezoidal integrator, the
  consumer's); and a one-bag reference fixture (both couplings, both
  integrators) for the consumer's cross-test.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .two_mass import burn_in_seconds, damping_from_ratio, filter_sequence, mean_diagonal

#: The audit families (S-2); the consumer's ``dataset.py`` already loads q_link/dq_link when present.
AUDIT_FAMILIES = {
    "q_link": "link-side position, simulator truth, noise-free, on the recording grid [rad | m]",
    "dq_link": "link-side velocity, simulator truth, noise-free [rad/s | m/s]",
    "defl": "true transmission deflection q_motor_clean - q_link_clean = theta - q_link [rad | m]",
}


#: ``(coupling, integrator, name suffix)`` of every baseline filter; the unsuffixed one is the default.
VARIANTS_OF_FILTER = (("diagonal", "trapezoidal", ""), ("full", "trapezoidal", "_full"),
                      ("diagonal", "backward_euler", "_be"), ("full", "backward_euler", "_full_be"))


def audit_columns(n_dof: int) -> dict[str, Any]:
    """The ``audit_columns`` contract block (FR_04 S-2)."""
    return {
        "columns": [f"{family}0..{family}{n_dof - 1}" for family in AUDIT_FAMILIES],
        "semantics": dict(AUDIT_FAMILIES),
        "noise_free": True,
        "use": ("audit only (FR_04 S-2): simulator truth to check a model's latent deflection against; "
                "never a model input or target. The *_clean columns are debug columns under the same rule"),
    }


def _units(asset: Any) -> list[dict[str, str]]:
    out = []
    for joint in asset.resolve_active_joints():
        prismatic = str(joint.joint_type).lower() == "prismatic"
        out.append({"stiffness": "N/m" if prismatic else "N·m/rad", "damping": "N·s/m" if prismatic else "N·m·s/rad",
                    "inertia": "kg" if prismatic else "kg·m^2", "frequency": "Hz", "damping_ratio": "1"})
    return out


def transmission_priors(manifest: Mapping[str, Any], asset: Any, *,
                        burn_in_override_s: float | None = None) -> dict[str, Any]:
    """The ``transmission_priors`` block (FR_04 S-3), from the manifest only.

    Stiffness: the declared prior, ``stiffness_nominal`` x ``stiffness_factor``.

    Damping ratio and mode frequency are given in the terms of the filter the
    consumer integrates (FR_04 Sec 0): the link-side oscillator driven by the
    measured motor position, ``zeta = D / (2 sqrt(K M_jj))`` and
    ``f = sqrt(K / M_jj) / 2 pi`` with ``M_jj`` the median link inertia of the
    plant as built (``robots[].plant_link_inertia_median``), ranges over the
    dataset's robots.  The manifest's own ``damping_ratio`` (the sampling
    prior) and ``mode_frequency_hz`` refer to the rotor-link mode about
    ``J_eff = J M / (J + M)`` and the inertia floor; they are kept beside
    (``*_declared``, ``*_manifest``) but do not describe the filter: on the
    FMRR y axis the declared ``zeta`` is 0.4-1.2 while the filter's is
    0.05-0.16 (FR_04 implementation report).

    ``burn_in_s`` is the Sec 3.1 rule's (``burn_in_rule_s``, always written)
    unless ``burn_in_override_s`` is given (the config's
    ``transmission_priors.burn_in_override_s``, FR_04 D-2): 0 where every bag
    starts at rest and in static equilibrium, so the filter's quasi-static
    initial condition is exact at the bag start and no head needs masking.
    """
    names = list(manifest["joint_names"])
    n = len(names)
    sampling = manifest["transmission_sampling"]
    robots = list(manifest.get("robots") or [])
    if not robots:
        raise ValueError("the manifest has no elastic robots to take the priors over")
    missing = [r["name"] for r in robots if "plant_link_inertia_median" not in r]
    if missing:
        raise ValueError(f"robots {missing} carry no plant_link_inertia_median (a payload.mode: fixed/none build "
                         "writes it); the filter-side priors need the plant's link inertia")
    stiffness = np.asarray([r["stiffness"] for r in robots], dtype=float)
    damping = np.asarray([r["damping"] for r in robots], dtype=float)
    inertia = np.asarray([r["plant_link_inertia_median"] for r in robots], dtype=float)
    zeta = damping / (2.0 * np.sqrt(stiffness * inertia))
    frequency = np.sqrt(stiffness / inertia) / (2.0 * math.pi)
    nominal = np.asarray(sampling.get("stiffness_nominal") or [], dtype=float)
    factor = sampling.get("stiffness_factor") or [1.0, 1.0]
    if nominal.size != n:
        raise ValueError("transmission_sampling.stiffness_nominal is needed for the stiffness prior")
    zeta_min, f_min = float(zeta.min()), float(frequency.min())
    burn_in_rule = burn_in_seconds(zeta_min, f_min)
    if burn_in_override_s is not None and not burn_in_override_s >= 0.0:
        raise ValueError(f"burn_in_override_s must be >= 0 s, got {burn_in_override_s}")
    burn_in = burn_in_rule if burn_in_override_s is None else float(burn_in_override_s)
    units = _units(asset)
    joints = []
    for j, name in enumerate(names):
        joints.append({
            "joint": name,
            "stiffness_nominal": float(nominal[j]),
            "stiffness_range": [float(nominal[j] * factor[0]), float(nominal[j] * factor[1])],
            "damping_ratio_range": [float(zeta[:, j].min()), float(zeta[:, j].max())],
            "mode_frequency_hz_range": [float(frequency[:, j].min()), float(frequency[:, j].max())],
            "link_inertia_median_range": [float(inertia[:, j].min()), float(inertia[:, j].max())],
            "damping_ratio_declared": [float(v) for v in sampling["damping_ratio"]],
            "mode_frequency_hz_manifest_range": [float(min(r["mode_frequency_hz"][j] for r in robots)),
                                                 float(max(r["mode_frequency_hz"][j] for r in robots))],
            "units": units[j],
        })
    return {
        "source": "manifest transmission_sampling (stiffness) and robots[] (damping, plant link inertia); "
                  "no new physics (FR_04 S-3)",
        "n_robots": len(robots),
        "robots": [r["name"] for r in robots],
        "definitions": {
            "stiffness_range": "stiffness_nominal x transmission_sampling.stiffness_factor",
            "damping_ratio_range": "D / (2 sqrt(K M_jj)) over the dataset's robots, M_jj = the plant's median link "
                                   "inertia (the filter's link-side mode, FR_04 Sec 0)",
            "mode_frequency_hz_range": "sqrt(K / M_jj) / (2 pi) over the dataset's robots (the same mode)",
            "damping_ratio_declared": "transmission_sampling.damping_ratio: the sampling prior, about the rotor-link "
                                      "mode, J_eff = J M / (J + M) with the sizing payload's envelope",
            "mode_frequency_hz_manifest_range": "robots[].mode_frequency_hz: the rotor-link mode at the inertia "
                                                "floor (it sizes the integration step)",
        },
        "joints": joints,
        "zeta_min": zeta_min,
        "frequency_min_hz": f_min,
        "burn_in_s": burn_in,
        "burn_in_rule": "max(3 / (zeta_min 2 pi f_min), 2 / f_min), rounded up to 0.05 s (FR_04 Sec 3.1)",
        "burn_in_rule_s": burn_in_rule,
        "burn_in_override_s": None if burn_in_override_s is None else float(burn_in_override_s),
        "burn_in_source": ("rule" if burn_in_override_s is None else
                           "config transmission_priors.burn_in_override_s (FR_04 D-2: every bag starts at rest "
                           "and in static equilibrium, so the quasi-static initial condition is exact)"),
        "sample_time_step": float(manifest["sample_time_step"]),
    }


def _block(frame: pd.DataFrame, prefix: str, n: int) -> np.ndarray:
    return frame[[f"{prefix}{i}" for i in range(n)]].to_numpy(dtype=float)


def _model_for(asset: Any, payload: Mapping[str, Any] | None, cache: dict) -> tuple:
    from . import identification as idn
    from .payload import Payload, payload_asset

    body = Payload() if not payload else Payload(mass=payload["mass"], offset=tuple(payload["offset"]),
                                                 size=payload["size"])
    key = () if body.is_empty else (body.mass, body.offset, body.size)
    if key not in cache:
        with payload_asset(asset, body) as asset_p:
            cache[key] = idn.build_model(asset_p)
    return cache[key]


def _rigid(pin: Any, model: Any, data: Any, q: np.ndarray, dq: np.ndarray, ddq: np.ndarray) -> tuple:
    tau = np.empty_like(q)
    inertia = np.empty((len(q), q.shape[1], q.shape[1]))
    for i in range(len(q)):
        tau[i] = pin.rnea(model, data, q[i], dq[i], ddq[i])
        matrix = np.asarray(pin.crba(model, data, q[i]))
        inertia[i] = np.triu(matrix) + np.triu(matrix, 1).T
    return tau, inertia


def bag_baselines(frame: pd.DataFrame, manifest: Mapping[str, Any], contract: Mapping[str, Any], asset: Any,
                  priors: Mapping[str, Any], *, coupling: str = "diagonal", keep: Sequence[str] | None = None,
                  fixture_bag: str | None = None) -> tuple[pd.DataFrame, dict | None]:
    """Per bag: the rigid, nominal and oracle two-mass predictions' squared errors on the kept rows.

    Inputs as the consumer sees them: recorded ``q``/``dq``, ``ddq`` the
    Savitzky-Golay derivative of ``dq`` with the contract's window (the
    consumer's ``dataset.py``), the target ``ft``.  ``nominal``: the nominal
    URDF's rigid torque and inertia, nominal K, the prior-centre damping ratio
    and ``M_bar`` the bag's mean diagonal of that inertia.  ``oracle``: the
    plant's model (with whatever payload is attached), the robot's true K and
    D.  A row is kept when ``t - t_bag_start >= priors.burn_in_s``.
    Returns the per-bag table (sums of squares per joint and kept-row counts)
    and, for ``fixture_bag``, the cross-test fixture.
    """
    from .diagnostics import savitzky_golay_acceleration

    n = int(manifest["n_dof"])
    names = list(manifest["joint_names"])
    step = float(manifest["sample_time_step"])
    window = int((contract.get("differentiation") or {}).get("sg_window") or 11)
    poly = int((contract.get("differentiation") or {}).get("sg_poly") or 3)
    nominal_k = np.asarray([j["stiffness_nominal"] for j in priors["joints"]], dtype=float)
    centre_zeta = np.asarray([0.5 * sum(j["damping_ratio_range"]) for j in priors["joints"]], dtype=float)
    head = float(priors["burn_in_s"])
    payloads = {r["bag"]: r.get("payload") for r in manifest.get("records", [])}
    cache: dict = {}
    nominal_model = _model_for(asset, None, cache)
    rows, fixture = [], None
    for bag, group in frame.groupby("bag", sort=False):
        if keep is not None and bag not in keep:
            continue
        tier = str(group["tier"].iloc[0])
        q, dq = _block(group, "q", n), _block(group, "dq", n)
        ddq = savitzky_golay_acceleration(dq, step, window=window, order=poly)
        target = _block(group, "ft", n)
        time = group["t"].to_numpy(dtype=float)
        kept = (time - time[0]) >= head - 1e-9
        tau_nom, inertia_nom = _rigid(*nominal_model, q, dq, ddq)
        predictions = {"rigid": tau_nom}
        if tier != "rigid":
            damping_nom = damping_from_ratio(nominal_k, centre_zeta, mean_diagonal(inertia_nom))
            stiffness = np.asarray([float(group[f"stiffness__{name}"].iloc[0]) for name in names])
            damping = np.asarray([float(group[f"damping__{name}"].iloc[0]) for name in names])
            plant = _model_for(asset, payloads.get(bag), cache)
            tau_plant, inertia_plant = (tau_nom, inertia_nom) if plant is nominal_model else _rigid(*plant, q, dq, ddq)
            # The default (``coupling``, trapezoidal) under the spec's names; every other coupling and
            # integrator beside it, suffixed, so the run can be read against whichever the models use.
            for name_coupling, name_integrator, suffix in VARIANTS_OF_FILTER:
                options = {"coupling": coupling if not suffix else name_coupling, "integrator": name_integrator}
                predictions["rigid_elastic_nominal" + suffix] = filter_sequence(
                    tau_nom, inertia_nom, nominal_k, damping_nom, step, **options)["tau_s"]
                predictions["rigid_elastic_oracle" + suffix] = filter_sequence(
                    tau_plant, inertia_plant, stiffness, damping, step, **options)["tau_s"]
            if fixture_bag is not None and bag == fixture_bag:
                fixture = {"bag": bag, "tau_r": tau_plant, "inertia": inertia_plant, "stiffness": stiffness,
                           "damping": damping, "time_step": np.float64(step),
                           }
                for name in ("diagonal", "full"):
                    for scheme in ("trapezoidal", "backward_euler"):
                        out = filter_sequence(tau_plant, inertia_plant, stiffness, damping, step,
                                              coupling=name, integrator=scheme)
                        fixture[f"tau_s_{name}_{scheme}"] = out["tau_s"]
                        fixture[f"delta_{name}_{scheme}"] = out["delta"]
                fixture["kept"] = kept
                fixture["target"] = target
        else:
            # The rigid reference has no transmission: its "elastic" baselines are the rigid one.
            for _, _, suffix in VARIANTS_OF_FILTER:
                predictions["rigid_elastic_nominal" + suffix] = predictions["rigid_elastic_oracle" + suffix] = tau_nom
        row = {"bag": bag, "split": str(group["split"].iloc[0]), "tier": tier, "kept": int(kept.sum()),
               "rows": int(len(group))}
        for name, prediction in predictions.items():
            error = prediction[kept] - target[kept]
            row[f"{name}_sse"] = (error ** 2).sum(axis=0).tolist()
            row[f"{name}_bag_rms"] = float(np.sqrt(np.mean(error ** 2))) if kept.any() else float("nan")
        rows.append(row)
    return pd.DataFrame(rows), fixture


def baseline_block(table: pd.DataFrame, priors: Mapping[str, Any], *, coupling: str = "diagonal") -> dict[str, Any]:
    """``contract.baselines.two_mass`` from :func:`bag_baselines`' table, per split and ``all``."""
    names = ["rigid"] + [f"rigid_elastic_{kind}{suffix}" for _, _, suffix in VARIANTS_OF_FILTER
                         for kind in ("nominal", "oracle")]
    splits = {}
    for split in ["all"] + sorted(set(table["split"])):
        subset = table if split == "all" else table[table["split"] == split]
        kept = int(subset["kept"].sum())
        entry: dict[str, Any] = {"n_bags": int(len(subset)), "n_rows": kept, "n_rows_unmasked": int(subset["rows"].sum())}
        for name in names:
            sse = np.sum(np.asarray(subset[f"{name}_sse"].tolist(), dtype=float), axis=0)
            entry[name] = {
                "rms": float(np.sqrt(sse.sum() / (kept * len(sse)))) if kept else None,
                "rms_joints": [float(v) for v in np.sqrt(sse / kept)] if kept else None,
                "median_bag_rms": (float(np.nanmedian(subset[f"{name}_bag_rms"]))
                                   if subset[f"{name}_bag_rms"].notna().any() else None),
            }
        splits[split] = entry
    return {
        "exclude_head_s": float(priors["burn_in_s"]),
        "statistic": ("rms: pooled RMS over every joint and kept row of the split (evaluate_on's rmse); "
                      "rms_joints: pooled per joint; median_bag_rms: median over bags of the per-bag RMS (the "
                      "statistic of the other baselines). A row is kept when t - t_bag_start >= exclude_head_s"),
        "target": "ft (the measured target the models are scored on)",
        "inputs": "recorded q, dq; ddq the Savitzky-Golay derivative of dq with the contract's window (the consumer's)",
        "filter": f"elastic_sim.two_mass.filter_sequence, coupling {coupling}, trapezoidal (Newmark average "
                  "acceleration) at the sample step, quasi-static initial condition at the bag start",
        "rigid": "nominal URDF rnea at the recorded motor states (the existing baseline_rms_rigid, on the kept rows)",
        "rigid_elastic_nominal": "nominal URDF rigid torque and inertia, K = stiffness_nominal, zeta = the centre of "
                                 "transmission_priors damping_ratio_range, D = 2 zeta sqrt(K M_bar), M_bar the bag's "
                                 "mean diagonal inertia",
        "rigid_elastic_oracle": "the plant's own model (with any attached payload), each robot's true K and D",
        "suffixes": {suffix: f"coupling {c}, integrator {i}" for c, i, suffix in VARIANTS_OF_FILTER if suffix},
        "splits": splits,
        "relabelling": "a relabel_val.py copy keeps this block verbatim: its train/val entries then follow the "
                       "source labelling; test and all stay exact",
    }


def augment_contract(output: Path, frame: pd.DataFrame, manifest: Mapping[str, Any], asset: Any, *,
                     priors: Mapping[str, Any] | None = None, fixture: bool = True,
                     burn_in_override_s: float | None = None) -> dict[str, Any]:
    """Write S-2, S-3 and S-4 into ``output``'s contract; return the priors used.

    ``priors`` given (the production file's) keeps a gain-shift twin on the
    same burn-in and prior; otherwise they are computed from ``manifest``
    (with ``burn_in_override_s``, the config's, in place of the rule's).
    With ``fixture``, the first elastic test bag's filter inputs and outputs go
    to ``<stem>.two_mass_reference.npz`` for the consumer's cross-test.
    """
    contract_path = output.with_suffix(".contract.json")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    priors = (dict(priors) if priors is not None
              else transmission_priors(manifest, asset, burn_in_override_s=burn_in_override_s))
    first = None
    if fixture:
        bags = frame.groupby("bag", sort=False)[["split", "tier"]].first()
        test = bags[(bags["split"] == "test") & (bags["tier"] != "rigid")]
        first = str(test.index[0]) if len(test) else None
    table, reference = bag_baselines(frame, manifest, contract, asset, priors, fixture_bag=first)
    contract["audit_columns"] = audit_columns(int(manifest["n_dof"]))
    contract["transmission_priors"] = priors
    contract.setdefault("baselines", {})["two_mass"] = baseline_block(table, priors)
    contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    if reference is not None:
        np.savez(output.with_name(output.stem + ".two_mass_reference.npz"), **reference)
    return priors
