#!/usr/bin/env python3
"""identified_filter/11 Part A: time-series and spectral diagnostics of the r9 models (existing checkpoints).

Every model is run on the motor-side inputs the networks get (recorded q, dq; ddq the consumer's SG derivative), on
the production test bags, the modal (chirp) bags whose excited joint is a main joint, and the gain-shift test bags:

* baselines: the rigid URDF (Pinocchio RNEA, the harness's rigid), Zollo ID-rigid and bounded ID-elastic
  (``reports/zollo/r9``'s identification), Zollo ``id_inverse`` (privileged: the measured link-side state);
* physics references (true link friction): the nominal linear filter (nominal K, centre zeta: the ladder's L0
  ``nominal``) and the oracle tangent filter (true K_0, D_eff, the catalogue curve, fixed point: the ladder's L2
  ``oracle_ltv_tangent[fixed_point]``);
* ``lpf_rigid``: one causal second-order low-pass per joint on the rigid prediction, fitted on 20 train robots;
* the networks' deployed seeds (``r8_summary.deployed_checkpoints``), J1 with its privileged truth.

    uv run python tools/r9_timeseries.py predict [--platform ur10] [--jobs J16 J17 J14x]   # caches predictions
    uv run python tools/r9_timeseries.py metrics      # A4, A5 tables (CSV)
    uv run python tools/r9_timeseries.py figures      # A3
    uv run python tools/r9_timeseries.py report       # results/r9/11_timeseries.md
    uv run python tools/r9_timeseries.py all

Output under ``REFACTOR_SPECS/identified_filter/results/r9/timeseries/`` (``cache/`` is the per-bag predictions).
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
from scipy import signal

CONFIG = SIM / "config" / "campaigns" / "r9_complete.yaml"
RESULTS = SIM / "REFACTOR_SPECS" / "identified_filter" / "results" / "r9"
OUT = RESULTS / "timeseries"
CACHE = OUT / "cache"
FILES = ("production", "modal", "gainshift")
MAIN = {"iiwa_drive": ("iiwa_A1", "iiwa_A2", "iiwa_A4"),
        "ur10": ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint")}
GATED = {("iiwa_drive", "iiwa_A2", "full"), ("ur10", "shoulder_lift_joint", "full"),
         ("ur10", "shoulder_lift_joint", "real_like"), ("ur10", "elbow_joint", "full")}
NN_JOBS = ("J11", "J4", "J12", "J13", "J2", "J14", "J5", "J15", "J3", "J1")
RESIDUALS = ("J11", "J4", "J12", "J13", "J16", "J17")
TWO_MASS = ("J1", "J2", "J3", "J5", "J14", "J15", "J14x", "J15x")
BASELINES = ("rigid", "zollo_id_rigid", "zollo_id_elastic_bounded", "zollo_id_inverse", "nominal_filter",
             "oracle_filter", "lpf_rigid")
PRIVILEGED = ("J1", "oracle_filter", "nominal_filter", "zollo_id_inverse")
LPF_TRAIN_ROBOTS = 20
BANDS = ("low", "mode", "high")       # < 0.5 f_n, [0.5, 2] f_n, > 2 f_n


# ---------------------------------------------------------------------------------------------------- data -----

def settings() -> dict:
    import r8_campaign

    return r8_campaign.load_settings(CONFIG)


def file_path(platform: str, file: str) -> Path:
    import r8_campaign

    return r8_campaign.platform_paths(settings(), platform)[file]


def contract(platform: str) -> dict:
    return json.loads(file_path(platform, "production").with_suffix(".contract.json").read_text(encoding="utf-8"))


def joint_names(platform: str) -> list[str]:
    return [j["joint"] for j in contract(platform)["transmission_priors"]["joints"]]


def load_bags(platform: str, file: str, split: str = "test", bags: list[str] | None = None) -> dict[str, dict]:
    """``bag -> arrays`` of one file: recorded inputs and target, the truth the diagnostics need, per-bag metadata.

    The modal file keeps the bags whose excited joint (largest motor-velocity RMS) is a main joint.
    """
    import pyarrow.parquet as pq

    path = file_path(platform, file)
    names = joint_names(platform)
    n = len(names)
    schema = set(pq.read_schema(path).names)
    blocks = ("q", "dq", "ft", "ft_clean", "tau", "q_motor_clean", "q_link_clean", "dq_motor_clean", "dq_link_clean",
              "tau_link_clean", "defl", "q_link", "dq_link")
    per_joint = ("stiffness__", "damping__", "f_n_filter__", "zeta_filter__", "link_coulomb__", "link_viscous__",
                 "te_amplitude__", "te_order__", "te_phase__", "te_amplitude_n__", "te_order_n__", "te_phase_n__",
                 "observable__")
    columns = ["t", "bag", "split", "tier", "probe_regime"] + [f"{b}{j}" for b in blocks for j in range(n)] \
        + [f"{p}{x}" for p in per_joint for x in names]
    columns = [c for c in columns if c in schema]
    filters = [("bag", "in", bags)] if bags is not None else [("split", "==", split)]
    frame = pq.read_table(path, columns=columns, filters=filters).to_pandas()
    out = {}
    for bag, group in frame.groupby("bag", sort=False):
        if str(group["tier"].iloc[0]) == "rigid":
            continue
        item = {"name": str(bag), "tier": str(group["tier"].iloc[0]), "split": str(group["split"].iloc[0]),
                "regime": str(group["probe_regime"].iloc[0]), "t": group["t"].to_numpy(float), "group": group}
        for b in blocks:
            if f"{b}0" in group:
                item[b] = group[[f"{b}{j}" for j in range(n)]].to_numpy(float)
        for p in per_joint:
            item[p.rstrip("_")] = np.asarray([float(group[f"{p}{x}"].iloc[0]) if f"{p}{x}" in group else np.nan
                                              for x in names])
        if file == "modal":
            excited = int(np.argmax(np.sqrt(np.mean(item["dq"] ** 2, axis=0))))
            item["excited"] = names[excited]
            if names[excited] not in MAIN[platform]:
                continue
        out[str(bag)] = item
    return out


def kept(item: dict, burn_in: float) -> np.ndarray:
    return (item["t"] - item["t"][0]) >= burn_in - 1e-9


def true_spring(item: dict, platform: str) -> tuple[np.ndarray, np.ndarray]:
    """``(delta, S)``: delta = theta + e(theta) - q from the clean columns, S = f(delta) on the catalogue curve."""
    from elastic_sim.two_mass import catalogue_torque

    theta, q = item["q_motor_clean"], item["q_link_clean"]
    error = np.zeros_like(theta)
    for suffix in ("", "_n"):     # e(theta) = A sin(order theta + phase) + A_n sin(order_n theta + phase_n)
        amplitude = item.get(f"te_amplitude{suffix}")
        if amplitude is None or not np.isfinite(amplitude).any():
            continue
        order, phase = item[f"te_order{suffix}"], item[f"te_phase{suffix}"]
        error += np.nan_to_num(amplitude) * np.sin(np.nan_to_num(order) * theta + np.nan_to_num(phase))
    delta = theta + error - q
    factors, knees, _ = curve(platform)
    return delta, catalogue_torque(delta, item["stiffness"], factors, knees)


_CURVES: dict = {}


def curve(platform: str) -> tuple:
    if platform not in _CURVES:
        import r8_physics_ladder as ladder
        from diagnose_controller_modes import _load_asset
        from elastic_sim.dataset import load_config

        config = load_config(SIM / settings()["platforms"][platform]["config"])
        asset = _load_asset(config.asset)
        _CURVES[platform] = ladder.spring_curves(config, asset, ladder.KNEE_BLEND, platform, joint_names(platform))[""]
        _CURVES[platform + ":cutoff"] = ladder.selector_cutoff(config, platform)
        _CURVES[platform + ":asset"] = asset
    return _CURVES[platform]


# ------------------------------------------------------------------------------------------------ predict -----

def cache_path(platform: str, file: str, model: str) -> Path:
    return CACHE / f"{platform}-{file}" / f"{model}.pkl"


def save(platform: str, file: str, model: str, data: dict) -> None:
    path = cache_path(platform, file, model)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load(platform: str, file: str, model: str) -> dict | None:
    path = cache_path(platform, file, model)
    if not path.exists():
        return None
    with path.open("rb") as handle:
        data = pickle.load(handle)
    if model == "truth" and file == "modal":
        # The modal file carries no f_n_filter; f_n is one per robot, so take each robot's from its production bags.
        per_robot = {item["tier"]: item["f_n"] for item in load(platform, "production", "truth").values()}
        for item in data.values():
            item["f_n"] = per_robot[item["tier"]]
    return data


def _baselines_bag(args) -> tuple[str, dict]:
    """CPU baselines of one bag: rigid, nominal and oracle filters, Zollo (``models`` pickled in)."""
    platform, item, step, window, poly, zollo = args
    import r8_physics_ladder as ladder
    from elastic_sim.diagnostics import savitzky_golay_acceleration
    from elastic_sim.identification import FrictionModel
    from elastic_sim.round7 import _model_for, _rigid
    from elastic_sim.round8 import effective_transmission
    from elastic_sim.two_mass import damping_from_ratio, filter_sequence, mean_diagonal

    names = joint_names(platform)
    n = len(names)
    priors = contract(platform)["transmission_priors"]
    q, dq = item["q"], item["dq"]
    ddq = savitzky_golay_acceleration(dq, step, window=window, order=poly)
    curve(platform)
    model = _model_for(_CURVES[platform + ":asset"], None, {})
    tau_rnea, inertia = _rigid(*model, q, dq, ddq)
    out = {"rigid": tau_rnea}
    friction = FrictionModel(item["link_viscous"], item["link_coulomb"])
    tau_r = tau_rnea + friction.torque(dq)
    nominal_k = np.asarray([j["stiffness_nominal"] for j in priors["joints"]], dtype=float)
    centre_zeta = np.asarray([0.5 * sum(j["damping_ratio_range"]) for j in priors["joints"]], dtype=float)
    out["nominal_filter"] = filter_sequence(tau_r, inertia, nominal_k, damping_from_ratio(
        nominal_k, centre_zeta, mean_diagonal(inertia)), step, coupling="full")["tau_s"]
    stiffness, damping = item["stiffness"], item["damping"]
    _, d_eff = effective_transmission(item["group"], n, stiffness, damping)
    start = ladder.selection(tau_r, "lowpass", _CURVES[platform + ":cutoff"], step)
    oracle, _ = ladder.fixed_point_pass(tau_r, inertia, stiffness, d_eff, curve(platform), step, start=start)
    out["oracle_filter"] = oracle["tau_s"]
    out["oracle_filter:delta"] = oracle["delta"]
    out["oracle_filter:S"] = oracle["spring"]
    if zollo:
        from elastic_sim.zollo.data import Bag

        if platform not in _ZOLLO:      # built once per worker: the models hold the pinocchio module
            _ZOLLO[platform] = zollo_models(platform)
        zollo = _ZOLLO[platform]

        delay = zollo["delay"]
        delayed = (lambda x: x if delay <= 0 else np.concatenate([np.repeat(x[:1], delay, axis=0), x[:-delay]]))
        bag = Bag(item["name"], item["tier"], item["split"], item["t"], q, dq, item["ft"],
                  q=delayed(item["q_link"]), dq=delayed(item["dq_link"]))
        link, bounded = zollo["link"], zollo["bounded"]
        out["zollo_id_rigid"] = link.torque(q, dq, ddq)
        probes = [q[i] for i in np.linspace(0, len(q) - 1, 20).astype(int)]
        run = bounded.with_damping_floor(zollo["zeta_min"], probes).run(q, dq, step)
        out["zollo_id_elastic_bounded"] = run.tau
        out["zollo_id_inverse"] = link.torque(bag.q, bag.dq, savitzky_golay_acceleration(bag.dq, step, window=window,
                                                                                          order=poly))
    return item["name"], out


_ZOLLO: dict = {}


def zollo_models(platform: str) -> dict | None:
    """Zollo's link and bounded elastic models from ``reports/zollo/r9/<platform>`` (as ``zollo_evaluate`` builds them)."""
    path = SIM / "reports" / "zollo" / "r9" / platform / "identification_Apc_pooled.json"
    if not path.exists():
        return None
    from elastic_sim import identification as idn
    from elastic_sim.assets import AssetRegistry
    from elastic_sim.zollo.config import load_platform
    from elastic_sim.zollo.data import sidecar
    from elastic_sim.zollo.identify import load_identification
    from elastic_sim.zollo.predictors import ElasticModel, ElasticSettings, LinkModel, TransmissionBounds

    cfg = load_platform(str(SIM / "config" / "zollo" / "r9" / f"{platform}.yaml"), SIM)
    result = load_identification(path)
    rigid_model = idn.build_model(AssetRegistry.for_repository(SIM).load(result.meta["asset"]))
    train_contract = sidecar(cfg.dataset, "contract")
    ev = cfg.evaluation
    elastic = ElasticSettings.from_mapping(ev.get("elastic"))
    bounds = TransmissionBounds.from_two_mass(train_contract, ev.get("lnn_repo") and cfg.resolve(ev["lnn_repo"]))
    return {"link": LinkModel.from_identification(result, rigid_model),
            "bounded": ElasticModel.bounded(result, rigid_model, bounds, elastic),
            "zeta_min": float(train_contract["transmission_priors"]["zeta_min"]),
            "delay": int((train_contract.get("noise") or {}).get("delay_samples") or 0)}


def predict_baselines(platform: str, file: str, bags: dict, jobs: int) -> None:
    done = [m for m in ("rigid", "nominal_filter", "oracle_filter", "zollo_id_rigid") if load(platform, file, m)]
    if len(done) == 4:
        return
    priors = contract(platform)
    step = float(priors["transmission_priors"]["sample_time_step"])
    window, poly = int(priors["differentiation"]["sg_window"]), int(priors["differentiation"]["sg_poly"])
    curve(platform)
    zollo = (SIM / "reports" / "zollo" / "r9" / platform / "identification_Apc_pooled.json").exists()
    results: dict[str, dict] = {}
    work = [(platform, {k: v for k, v in item.items()}, step, window, poly, zollo) for item in bags.values()]
    with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as pool:
        for name, out in pool.map(_baselines_bag, work, chunksize=1):
            results[name] = out
            print(f"  baselines {platform} {file} {len(results)}/{len(work)}", flush=True)
    for model in {key.split(":")[0] for out in results.values() for key in out}:
        save(platform, file, model, {bag: {k.split(":")[1] if ":" in k else "pred": v.astype(np.float32)
                                           for k, v in out.items() if k.split(":")[0] == model}
                                     for bag, out in results.items()})


def fit_lpf_rigid(platform: str, jobs: int) -> dict:
    """Per joint ``(f_c, zeta)`` of the causal second-order low-pass on the rigid prediction, fitted on train robots."""
    path = OUT / f"lpf_rigid_{platform}.json"
    if path.exists():
        return json.loads(path.read_text())
    import pyarrow.parquet as pq

    frame = pq.read_table(file_path(platform, "production"), columns=["bag", "split", "tier"],
                          filters=[("split", "==", "train")]).to_pandas().groupby("bag", sort=False).first()
    robots = sorted({t for t in frame["tier"] if t != "rigid"})[:LPF_TRAIN_ROBOTS]
    bags = load_bags(platform, "production", bags=[b for b, r in frame["tier"].items() if r in robots])
    priors = contract(platform)
    step = float(priors["transmission_priors"]["sample_time_step"])
    window, poly = int(priors["differentiation"]["sg_window"]), int(priors["differentiation"]["sg_poly"])
    burn_in = float(priors["transmission_priors"]["burn_in_s"])
    curve(platform)
    work = [(platform, item, step, window, poly, None) for item in bags.values()]
    with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as pool:
        rigid = dict(pool.map(_baselines_bag, work, chunksize=1))
    names = joint_names(platform)
    out = {}
    for j, name in enumerate(names):
        best = None
        for f_c in np.geomspace(0.5, min(150.0, 0.4 / step), 48):
            for zeta in (0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5):
                total, count = 0.0, 0
                for bag, item in bags.items():
                    smooth = lowpass2(rigid[bag]["rigid"][:, j], f_c, zeta, step)
                    keep = kept(item, burn_in)
                    total += float(np.sum((smooth[keep] - item["ft"][keep, j]) ** 2))
                    count += int(keep.sum())
                rmse = (total / count) ** 0.5
                if best is None or rmse < best[0]:
                    best = (rmse, float(f_c), float(zeta))
        raw = (sum(float(np.sum((rigid[b]["rigid"][kept(i, burn_in), j] - i["ft"][kept(i, burn_in), j]) ** 2))
                   for b, i in bags.items()) / sum(int(kept(i, burn_in).sum()) for i in bags.values())) ** 0.5
        out[name] = {"f_c_hz": best[1], "zeta": best[2], "train_rmse": best[0], "train_rmse_rigid": raw}
        print(f"  lpf_rigid {platform} {name}: {out[name]}", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"robots": robots, "joints": out}, indent=1))
    return json.loads(path.read_text())


def lowpass2(x: np.ndarray, f_c: float, zeta: float, step: float) -> np.ndarray:
    """``w^2 / (s^2 + 2 zeta w s + w^2)``, bilinear with prewarping at ``f_c``, causal, started at rest on ``x[0]``."""
    w = 2.0 * math.pi * f_c
    b, a = signal.bilinear([w * w], [1.0, 2.0 * zeta * w, w * w], fs=1.0 / step)
    zi = signal.lfilter_zi(b, a) * x[0]
    return signal.lfilter(b, a, x, zi=zi)[0]


def predict_lpf(platform: str, file: str, jobs: int) -> None:
    rigid = load(platform, file, "rigid")
    fit = fit_lpf_rigid(platform, jobs)["joints"]
    step = float(contract(platform)["transmission_priors"]["sample_time_step"])
    names = joint_names(platform)
    save(platform, file, "lpf_rigid", {bag: {"pred": np.stack([lowpass2(v["pred"][:, j].astype(float),
                                                                       fit[x]["f_c_hz"], fit[x]["zeta"], step)
                                                              for j, x in enumerate(names)], axis=1).astype(np.float32)}
                                       for bag, v in rigid.items()})


def predict_networks(platform: str, file: str, bags: dict, jobs_wanted: tuple[str, ...]) -> None:
    """The deployed seed of each job, whole bags (sequence classes) or rows (the others), as ``evaluate_on``."""
    import torch

    sys.path.insert(0, str(NN))
    import r8_summary
    from dataset import CustomDataset
    from evaluate_on import load_checkpoint
    from utils import unpack_model_output

    deployed = {(e["job"], e["platform"]): e for e in
                r8_summary.deployed_checkpoints(SIM / "runs" / settings()["run_id"]).values()}
    todo = [job for job in jobs_wanted if (job, platform) in deployed and load(platform, file, job) is None]
    if not todo:
        return
    path = file_path(platform, file)
    first = torch.load(deployed[(todo[0], platform)]["checkpoint"], map_location="cpu", weights_only=False)
    normalization = first["normalization"]
    dataset = CustomDataset(csv_path=str(path), normalization=normalization.get("scheme", "train"), normalize=True,
                            normalize_target=bool(normalization.get("normalize_target", True)))
    dataset.apply_normalization_state(normalization)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bag_rows = {b: np.asarray(i) for b, i in dataset.bag_to_indices.items() if b in bags}
    for job in todo:
        entry = deployed[(job, platform)]
        blob = torch.load(entry["checkpoint"], map_location="cpu", weights_only=False)
        if json.dumps(blob["normalization"], default=str) != json.dumps(normalization, default=str):
            raise ValueError(f"{job} {platform}: another normalization than the file was loaded with")
        model, _, record = load_checkpoint(Path(entry["checkpoint"]), None, None, dataset.dof)
        model.to(device).eval()
        identified = bool(getattr(model, "is_identified_filter", False))
        sequence = bool(getattr(model, "is_sequence_model", False))
        if sequence and hasattr(model, "time_step") and abs(dataset.dt - model.time_step) > 1e-9 * max(1.0, model.time_step):
            if identified:
                model.set_sample_step(dataset.dt)
            else:
                model.time_step = float(dataset.dt)
        privileged = identified and model.filter_param == "privileged"
        if privileged:
            dataset.use_privileged_source(blob.get("settings", {}).get("privileged_source") or "sampled")
        out = {}
        with torch.no_grad():
            for bag, rows in bag_rows.items():
                index = torch.as_tensor(rows, dtype=torch.long)
                q, dq, ddq = (dataset.q[index].to(device), dataset.dq[index].to(device), dataset.ddq[index].to(device))
                if sequence:
                    if privileged:
                        truth = dataset.privileged_rows(rows)
                        zeta = truth.get("zeta") if getattr(model, "damping_model", "D") == "zeta" else None
                        model.set_privileged(truth["K"][None].to(device), truth["D"][None].to(device),
                                             zeta=None if zeta is None else zeta[None].to(device))
                    prediction, _ = unpack_model_output(model(q, dq, ddq, None))
                    if privileged:
                        model.set_privileged(None, None)
                    item = {"pred": dataset.denormalize(prediction.cpu(), "F", bag_ids=None).numpy()}
                    last = getattr(model, "_last", None) or {}
                    if last.get("delta") is not None and last.get("D") is not None:
                        delta = last["delta"][0].detach().cpu().double().numpy()
                        D = last["D"].detach().cpu().double().numpy().reshape(-1)[-dataset.dof:]
                        rate = np.gradient(delta, dataset.dt, axis=0)
                        item["delta"] = delta
                        item["S"] = item["pred"] - D * rate
                else:
                    tau_cmd = dataset.tau_cmd[index].to(device)
                    parts = []
                    for start in range(0, len(rows), 8192):
                        sl = slice(start, start + 8192)
                        prediction, _ = unpack_model_output(model(q[sl], dq[sl], ddq[sl], tau_cmd[sl]))
                        parts.append(dataset.denormalize(prediction.cpu(), "F", bag_ids=[bag] * len(prediction)).numpy())
                    item = {"pred": np.concatenate(parts)}
                out[bag] = {k: np.asarray(v, dtype=np.float32) for k, v in item.items()}
        save(platform, file, job, out)
        print(f"  {job} {platform} {file}: {len(out)} bags ({record['class_path']}, seed {entry['seed']})", flush=True)
        del model
        torch.cuda.empty_cache()


def save_truth(platform: str, file: str, bags: dict) -> None:
    if load(platform, file, "truth") is not None:
        return
    out = {}
    for bag, item in bags.items():
        delta, spring = true_spring(item, platform)
        out[bag] = {"t": item["t"], "ft": item["ft"].astype(np.float32), "tau_m": item["tau"].astype(np.float32),
                    "dq": item["dq"].astype(np.float32), "q": item["q"].astype(np.float32),
                    "delta": delta.astype(np.float32), "S": spring.astype(np.float32),
                    "f_n": item["f_n_filter"], "K": item["stiffness"], "regime": item["regime"],
                    "tier": item["tier"], "excited": item.get("excited"), "observable": item.get("observable"),
                    "te_amplitude": item.get("te_amplitude"), "te_order": item.get("te_order"),
                    "te_phase": item.get("te_phase"), "theta": item["q_motor_clean"].astype(np.float32)}
    save(platform, file, "truth", out)


def cmd_predict(args) -> None:
    for platform in args.platform:
        for file in args.files:
            print(f"== {platform} {file}", flush=True)
            bags = load_bags(platform, file)
            save_truth(platform, file, bags)
            predict_networks(platform, file, bags, tuple(args.jobs))
            predict_baselines(platform, file, bags, args.workers)
            predict_lpf(platform, file, args.workers)
            del bags


# ------------------------------------------------------------------------------------------------ metrics -----

def band_filter(x: np.ndarray, band: str, f_n: float, rate: float) -> np.ndarray | None:
    """Zero-phase 4th-order Butterworth on the band of ``f_n``; ``None`` without an ``f_n`` or where the band leaves
    (0, Nyquist)."""
    if not np.isfinite(f_n) or f_n <= 0:
        return None
    nyquist = 0.5 * rate
    low, high = 0.5 * f_n, 2.0 * f_n
    if band == "low":
        sos = signal.butter(4, low / nyquist, btype="low", output="sos")
    elif band == "mode":
        if high >= 0.95 * nyquist:
            return None
        sos = signal.butter(4, [low / nyquist, high / nyquist], btype="band", output="sos")
    else:
        if high >= 0.95 * nyquist:
            return None
        sos = signal.butter(4, high / nyquist, btype="high", output="sos")
    return signal.sosfiltfilt(sos, x, axis=0)


def welch_segments(n: int) -> int:
    """``nperseg`` for >= 8 segments at 50 % overlap."""
    return max(32, int(2 * n / 9))


def bag_metrics(truth: dict, pred: dict, rigid: dict, j: int, rate: float, burn_in: float) -> dict:
    keep = (truth["t"] - truth["t"][0]) >= burn_in - 1e-9
    y, yhat, r = (x[:, j].astype(float) for x in (truth["ft"], pred["pred"], rigid["pred"]))
    f_n = float(truth["f_n"][j])
    error, rigid_error = yhat - y, r - y
    out = {"rmse": float(np.sqrt(np.mean(error[keep] ** 2)))}
    nper = welch_segments(int(keep.sum()))
    for band in BANDS:
        e = band_filter(error, band, f_n, rate)
        out[f"rmse_{band}"] = float(np.sqrt(np.mean(e[keep] ** 2))) if e is not None else float("nan")
    f, pe = signal.welch(error[keep], fs=rate, nperseg=nper)
    _, pr = signal.welch(rigid_error[keep], fs=rate, nperseg=nper)
    masks = {"low": f < 0.5 * f_n, "mode": (f >= 0.5 * f_n) & (f <= 2 * f_n), "high": f > 2 * f_n}
    for band, mask in masks.items():
        out[f"psd_ratio_{band}"] = float(pe[mask].sum() / pr[mask].sum()) if mask.any() and pr[mask].sum() > 0 \
            else float("nan")
    f, coh = signal.coherence(y[keep], yhat[keep], fs=rate, nperseg=nper)
    _, syy = signal.welch(y[keep], fs=rate, nperseg=nper)
    _, syh = signal.csd(y[keep], yhat[keep], fs=rate, nperseg=nper)
    mask = (f >= 0.5 * f_n) & (f <= 2 * f_n)
    out["coherence"] = float(np.mean(coh[mask])) if mask.any() else float("nan")
    out["gain"] = float(np.abs(syh[mask]).sum() / syy[mask].sum()) if mask.any() else float("nan")
    out["phase_deg"] = float(np.degrees(np.angle(np.interp(f_n, f, syh.real) + 1j * np.interp(f_n, f, syh.imag))))
    if "delta" in pred and "delta" in truth:
        a, b = pred["delta"][keep, j].astype(float), truth["delta"][keep, j].astype(float)
        out["delta_corr"] = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
        out["delta_rms_ratio"] = float(np.sqrt(np.mean(a ** 2)) / np.sqrt(np.mean(b ** 2)))
        f, coh = signal.coherence(b, a, fs=rate, nperseg=nper)
        mask = (f >= 0.5 * f_n) & (f <= 2 * f_n)
        out["delta_coherence"] = float(np.mean(coh[mask])) if mask.any() else float("nan")
    return out


def frf_metrics(truth: dict, pred: dict, j: int, rate: float, burn_in: float) -> dict:
    """H1 of motor torque -> tau_s, model against truth: peak frequency, half-power zeta, peak gain."""
    keep = (truth["t"] - truth["t"][0]) >= burn_in - 1e-9
    u, y, yhat = truth["tau_m"][keep, j].astype(float), truth["ft"][keep, j].astype(float), pred["pred"][keep, j].astype(float)
    nper = min(len(u), int(round(rate * 8.0)))
    f, suu = signal.welch(u, fs=rate, nperseg=nper)
    _, suy = signal.csd(u, y, fs=rate, nperseg=nper)
    _, suh = signal.csd(u, yhat, fs=rate, nperseg=nper)
    f_n = float(truth["f_n"][j])
    window = (f >= 0.4 * f_n) & (f <= 2.5 * f_n)
    out = {}
    for label, h in (("true", np.abs(suy / suu)), ("model", np.abs(suh / suu))):
        idx = np.flatnonzero(window)
        peak = idx[np.argmax(h[idx])]
        half = h[peak] / math.sqrt(2.0)
        left = peak
        while left > 0 and h[left] > half:
            left -= 1
        right = peak
        while right < len(h) - 1 and h[right] > half:
            right += 1
        f1 = np.interp(half, [h[left], h[left + 1]], [f[left], f[left + 1]]) if left < peak else f[left]
        f2 = np.interp(half, [h[right], h[right - 1]], [f[right], f[right - 1]]) if right > peak else f[right]
        out[f"f_peak_{label}"] = float(f[peak])
        out[f"zeta_{label}"] = float((f2 - f1) / (2.0 * f[peak]))
        out[f"gain_{label}"] = float(h[peak])
    out["f_peak_ratio"] = out["f_peak_model"] / out["f_peak_true"]
    out["zeta_ratio"] = out["zeta_model"] / out["zeta_true"] if out["zeta_true"] > 0 else float("nan")
    out["peak_gain_ratio"] = out["gain_model"] / out["gain_true"]
    return out


def model_list(platform: str, file: str) -> list[str]:
    return [m for m in BASELINES + NN_JOBS + ("J16", "J17", "J14x", "J15x") if cache_path(platform, file, m).exists()]


def cmd_metrics(args) -> None:
    rows, frf, segments = [], [], []
    for platform in args.platform:
        priors = contract(platform)["transmission_priors"]
        rate, burn_in = 1.0 / float(priors["sample_time_step"]), float(priors["burn_in_s"])
        names = joint_names(platform)
        for file in args.files:
            truth, rigid = load(platform, file, "truth"), load(platform, file, "rigid")
            if truth is None or rigid is None:
                continue
            for model in model_list(platform, file):
                pred = load(platform, file, model)
                for bag, item in truth.items():
                    if bag not in pred:
                        continue
                    for name in MAIN[platform]:
                        j = names.index(name)
                        if file == "modal" and item["excited"] != name:
                            continue
                        rows.append({"platform": platform, "file": file, "model": model, "bag": bag, "joint": name,
                                     "regime": item["regime"], "K": float(item["K"][j]), "f_n": float(item["f_n"][j]),
                                     **bag_metrics(item, pred[bag], rigid[bag], j, rate, burn_in)})
                        if file == "modal":
                            frf.append({"platform": platform, "model": model, "bag": bag, "joint": name,
                                        **frf_metrics(item, pred[bag], j, rate, burn_in)})
                            segments += modal_segments(platform, model, bag, item, pred[bag], rigid[bag], j, rate,
                                                       burn_in, name)
            print(f"  metrics {platform} {file}: {len(rows)} rows", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "bag_metrics.csv", index=False)
    pd.DataFrame(frf).to_csv(OUT / "frf.csv", index=False)
    pd.DataFrame(segments).to_csv(OUT / "modal_segments.csv", index=False)


def instantaneous_frequency(dq: np.ndarray, rate: float, window_s: float = 1.0) -> np.ndarray:
    """Per sample, the dominant frequency of the excited joint's velocity in a sliding window (the chirp's)."""
    nper = int(round(window_s * rate))
    f, t, z = signal.stft(dq, fs=rate, nperseg=nper, noverlap=int(0.9 * nper), boundary="even")
    dominant = f[np.argmax(np.abs(z[1:]), axis=0) + 1]
    return np.interp(np.arange(len(dq)) / rate, t, dominant)


def modal_segments(platform, model, bag, item, pred, rigid, j, rate, burn_in, name) -> list[dict]:
    """A5: the modal-file error of one bag split by time segment, and its overlap with the TE torque K e(theta)."""
    t = item["t"] - item["t"][0]
    y, yhat = item["ft"][:, j].astype(float), pred["pred"][:, j].astype(float)
    error, rigid_error = yhat - y, rigid["pred"][:, j].astype(float) - y
    f_n = float(item["f_n"][j])
    inst = instantaneous_frequency(item["dq"][:, j].astype(float), rate)
    resonance = (inst >= 0.7 * f_n) & (inst <= 1.4 * f_n) & (t >= burn_in)
    labels = {"burn_in": t < burn_in, "chirp_start": (t >= burn_in) & (t < burn_in + 2.0) & ~resonance,
              "resonance": resonance, "rest": (t >= burn_in + 2.0) & ~resonance}
    out = []
    te = np.zeros_like(y)
    if item.get("te_amplitude") is not None and np.isfinite(item["te_amplitude"][j]):
        te = float(item["K"][j]) * float(item["te_amplitude"][j]) * np.sin(
            float(item["te_order"][j]) * item["theta"][:, j].astype(float) + float(np.nan_to_num(item["te_phase"][j])))
    mode_error = band_filter(error, "mode", f_n, rate)
    mode_te = band_filter(te, "mode", f_n, rate)
    for label, mask in labels.items():
        if not mask.any():
            continue
        row = {"platform": platform, "model": model, "bag": bag, "joint": name, "segment": label,
               "seconds": float(mask.sum() / rate), "mse": float(np.mean(error[mask] ** 2)),
               "mse_rigid": float(np.mean(rigid_error[mask] ** 2))}
        if mode_error is not None and mode_te is not None and mode_te[mask].std() > 0:
            row["corr_mode_error_te"] = float(np.corrcoef(mode_error[mask], mode_te[mask])[0, 1])
            row["rms_mode_te"] = float(np.sqrt(np.mean(mode_te[mask] ** 2)))
            row["rms_mode_error"] = float(np.sqrt(np.mean(mode_error[mask] ** 2)))
        out.append(row)
    return out


# ------------------------------------------------------------------------------------------------ figures -----

LINES = (("truth", "k", 1.6), ("rigid", "tab:gray", 1.0), ("best_residual", "tab:green", 1.0), ("J2", "tab:blue", 1.0),
         ("J14", "tab:purple", 1.0), ("J1", "tab:orange", 1.0), ("oracle_filter", "tab:red", 1.0))


def best_residual(platform: str) -> str:
    """The residual with the lowest validation RMSE of the deployed seeds (11 B3's rule)."""
    import r8_summary

    deployed = [e for e in r8_summary.deployed_checkpoints(SIM / "runs" / settings()["run_id"]).values()
                if e["platform"] == platform and e["job"] in RESIDUALS]
    return min(deployed, key=lambda e: e["best_val_rmse"])["job"]


def pick_bags(metrics: pd.DataFrame, truth: dict, rigid: dict, platform: str, file: str, joint: str, regime: str | None,
              rate: float) -> dict[str, str]:
    """The median and 90th-percentile rigid-error bags, and the highest-content bag (in-band rigid error, 05 D3)."""
    part = metrics[(metrics.platform == platform) & (metrics.file == file) & (metrics.model == "rigid")
                   & (metrics.joint == joint)]
    if regime is not None:
        part = part[part.regime == regime]
    if not len(part):
        return {}
    ranked = part.sort_values("rmse")
    chosen = {"median": ranked.iloc[len(ranked) // 2]["bag"], "p90": ranked.iloc[int(0.9 * (len(ranked) - 1))]["bag"]}
    chosen["content"] = part.sort_values("rmse_mode").iloc[-1]["bag"]
    return chosen


def cmd_figures(args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = pd.read_csv(OUT / "bag_metrics.csv")
    figures = OUT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    targets = [(p, "production", j, r) for (p, j, r) in sorted(GATED)] + \
        [(p, "modal", j, None) for p in args.platform for j in MAIN[p]]
    for platform, file, joint, regime in targets:
        if platform not in args.platform:
            continue
        priors = contract(platform)["transmission_priors"]
        rate, burn_in = 1.0 / float(priors["sample_time_step"]), float(priors["burn_in_s"])
        truth, rigid = load(platform, file, "truth"), load(platform, file, "rigid")
        if truth is None:
            continue
        j = joint_names(platform).index(joint)
        residual = best_residual(platform)
        models = {"rigid": rigid, "best_residual": load(platform, file, residual),
                  **{m: load(platform, file, m) for m in ("J2", "J14", "J1", "oracle_filter")}}
        for kind, bag in pick_bags(metrics, truth, rigid, platform, file, joint, regime, rate).items():
            item = truth[bag]
            t = item["t"] - item["t"][0]
            accel = np.abs(np.gradient(item["dq"][:, j].astype(float), 1.0 / rate))
            accel[t < burn_in] = 0.0
            peak = t[int(np.argmax(accel))]
            spans = {"full": (0.0, t[-1]), "zoom": (peak - 1.0, peak + 1.0), "ringing": (peak, peak + 0.5)}
            fig, axes = plt.subplots(2, 3, figsize=(17, 6.5), sharex="col", gridspec_kw={"height_ratios": [2, 1]})
            for col, (span, (a, b)) in enumerate(spans.items()):
                mask = (t >= a) & (t <= b)
                for label, color, width in LINES:
                    if label == "truth":
                        axes[0, col].plot(t[mask], item["ft"][mask, j], color=color, lw=width, label="tau_s true")
                        continue
                    data = models.get(label)
                    if data is None or bag not in data:
                        continue
                    name = f"{label} ({residual})" if label == "best_residual" else label
                    axes[0, col].plot(t[mask], data[bag]["pred"][mask, j], color=color, lw=width, label=name)
                    axes[1, col].plot(t[mask], data[bag]["pred"][mask, j] - item["ft"][mask, j], color=color, lw=width)
                axes[0, col].set_title(f"{span}" + ("" if span == "full" else f" ({a:.2f}-{b:.2f} s)"), fontsize=9)
                axes[1, col].set_xlabel("t [s]")
                axes[1, col].axhline(0, color="k", lw=0.5)
            axes[0, 0].set_ylabel("tau_s [N m]")
            axes[1, 0].set_ylabel("error [N m]")
            axes[0, 2].legend(fontsize=7, loc="upper right")
            fig.suptitle(f"{platform} {file} {joint} {regime or 'modal'}: {kind} bag {bag} "
                         f"(f_n {item['f_n'][j]:.1f} Hz)", fontsize=10)
            fig.tight_layout()
            fig.savefig(figures / f"ts_{platform}_{file}_{joint}_{regime or 'modal'}_{kind}.png", dpi=110)
            plt.close(fig)
            if kind == "median":
                states(plt, figures, platform, file, joint, regime, bag, item, j, models, burn_in)
    k_scatter(plt, metrics, figures)


def states(plt, figures, platform, file, joint, regime, bag, item, j, models, burn_in) -> None:
    """The 2M internal states against the truth: delta_hat vs theta + e(theta) - q, S_hat vs f(delta)."""
    t = item["t"] - item["t"][0]
    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    axes[0].plot(t, item["delta"][:, j] * 1e3, "k", lw=1.5, label="delta true")
    axes[1].plot(t, item["S"][:, j], "k", lw=1.5, label="S true f(delta)")
    for label, color in (("J2", "tab:blue"), ("J14", "tab:purple"), ("J1", "tab:orange"), ("oracle_filter", "tab:red")):
        data = models.get(label)
        if data is None or bag not in data or "delta" not in data[bag]:
            continue
        axes[0].plot(t, data[bag]["delta"][:, j] * 1e3, color=color, lw=1, label=label)
        axes[1].plot(t, data[bag]["S"][:, j], color=color, lw=1, label=label)
    axes[0].set_ylabel("delta [mrad]")
    axes[1].set_ylabel("S [N m]")
    axes[1].set_xlabel("t [s]")
    for ax in axes:
        ax.axvline(burn_in, color="k", ls=":", lw=0.8)
        ax.legend(fontsize=7)
    fig.suptitle(f"{platform} {file} {joint} {regime or 'modal'}: internal states, median bag {bag}", fontsize=10)
    fig.tight_layout()
    fig.savefig(figures / f"states_{platform}_{file}_{joint}_{regime or 'modal'}.png", dpi=110)
    plt.close(fig)


def k_scatter(plt, metrics: pd.DataFrame, figures: Path) -> None:
    """A4.7: per-bag RMSE ratio to rigid against K / K_pooled (production test, main joints)."""
    prod = metrics[metrics.file == "production"]
    rigid = prod[prod.model == "rigid"].set_index(["platform", "bag", "joint"])["rmse"]
    for platform in prod.platform.unique():
        joints = MAIN[platform]
        fig, axes = plt.subplots(1, len(joints), figsize=(5 * len(joints), 4), squeeze=False)
        for ax, joint in zip(axes[0], joints):
            for model, color in (("J2", "tab:blue"), ("J14", "tab:purple"), ("J16", "tab:green"), ("J5", "tab:brown")):
                part = prod[(prod.platform == platform) & (prod.model == model) & (prod.joint == joint)]
                if not len(part):
                    continue
                ratio = part["rmse"].to_numpy() / rigid.loc[[(platform, b, joint) for b in part.bag]].to_numpy()
                ax.semilogx(part["K"] / np.median(part["K"]), ratio, "o", ms=3, color=color, alpha=0.7, label=model)
            ax.axhline(1.0, color="k", lw=0.6)
            ax.set_xlabel("K / median K (test robots)")
            ax.set_title(f"{platform} {joint}", fontsize=9)
        axes[0][0].set_ylabel("RMSE / rigid RMSE (per bag)")
        axes[0][0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figures / f"k_scatter_{platform}.png", dpi=110)
        plt.close(fig)


# ------------------------------------------------------------------------------------------------- report -----

def _md(frame: pd.DataFrame, digits: int = 3) -> str:
    def cell(v):
        if isinstance(v, float):
            return "" if v != v else f"{v:.{digits}g}"
        return str(v)
    head = "| " + " | ".join(map(str, frame.columns)) + " |"
    return "\n".join([head, "|" + "---|" * len(frame.columns)] +
                     ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False)])


def pooled_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """Per platform, file, joint, regime and model: RMS-pooled RMSEs and band RMSEs, medians of the spectral ratios."""
    keys = ["platform", "file", "joint", "regime", "model"]
    rms = lambda x: float(np.sqrt(np.nanmean(np.square(x))))  # noqa: E731
    agg = {c: rms for c in ("rmse", "rmse_low", "rmse_mode", "rmse_high")}
    agg.update({c: "median" for c in ("psd_ratio_low", "psd_ratio_mode", "psd_ratio_high", "coherence", "gain",
                                      "phase_deg", "delta_corr", "delta_rms_ratio", "delta_coherence")
                if c in metrics})
    table = metrics.groupby(keys).agg({**agg, "bag": "count"}).rename(columns={"bag": "bags"}).reset_index()
    rigid = table[table.model == "rigid"].set_index(keys[:-1])["rmse"]
    table["vs_rigid"] = table["rmse"] / rigid.reindex(pd.MultiIndex.from_frame(table[keys[:-1]])).to_numpy()
    return table


MODEL_ORDER = ("rigid", "lpf_rigid", "zollo_id_rigid", "zollo_id_elastic_bounded", "J11", "J4", "J12", "J13", "J16",
               "J17", "J2", "J14", "J14x", "J5", "J15", "J15x", "J3", "J1", "nominal_filter", "oracle_filter",
               "zollo_id_inverse")
LABELS = {"rigid": "rigid URDF", "lpf_rigid": "LPF-rigid", "zollo_id_rigid": "Zollo ID-rigid",
          "zollo_id_elastic_bounded": "Zollo ID-elastic (bounded)", "J11": "J11 KALNN res. (friction-only)",
          "J4": "J4 KALNN res. (r7)", "J12": "J12 LNN res. (friction-only)", "J13": "J13 LNN res. (r7)",
          "J16": "J16 KALNN anchored res.", "J17": "J17 LNN anchored res.", "J2": "J2 2M-KALNN-PHYS",
          "J14": "J14 2M-LNN-PHYS", "J14x": "J14x 2M-LNN-PHYS (300 ep.)", "J5": "J5 2M-KALNN (r7)",
          "J15": "J15 2M-LNN (r7)", "J15x": "J15x 2M-LNN (r7, cont.)", "J3": "J3 2M-KALNN-ID",
          "J1": "J1 teacher (privileged)", "nominal_filter": "nominal filter (true friction)",
          "oracle_filter": "oracle filter (privileged)", "zollo_id_inverse": "Zollo id_inverse (privileged)"}


def ordered(table: pd.DataFrame) -> pd.DataFrame:
    rank = {m: i for i, m in enumerate(MODEL_ORDER)}
    table = table.assign(_rank=table["model"].map(rank).fillna(99)).sort_values(["_rank"]).drop(columns="_rank")
    return table.assign(model=table["model"].map(lambda m: LABELS.get(m, m)))


def memorisation_table() -> pd.DataFrame:
    rows = []
    for path in sorted((SIM / "runs" / settings()["run_id"] / "a0").glob("*/memorisation.csv")):
        job, platform = path.parent.name.split("-", 1)
        frame = pd.read_csv(path).pivot_table(index="epoch", columns="split", values="rmse").reset_index()
        best = int(frame.loc[frame["val"].idxmin(), "epoch"])
        for _, r in frame.iterrows():
            rows.append({"job": job, "platform": platform, "epoch": int(r["epoch"]), "train10": r["train10"],
                         "val": r["val"], "gap (val - train10)": r["val"] - r["train10"],
                         "selected": "best val" if int(r["epoch"]) == best else ""})
    return pd.DataFrame(rows)


def cmd_report(args) -> None:
    metrics = pd.read_csv(OUT / "bag_metrics.csv")
    metrics.loc[metrics.file == "modal", "regime"] = "modal"
    table = pooled_table(metrics)
    table.to_csv(OUT / "pooled.csv", index=False)
    frf = pd.read_csv(OUT / "frf.csv")
    frf_table = frf.groupby(["platform", "joint", "model"]).agg(
        bags=("bag", "count"), f_peak_ratio=("f_peak_ratio", "median"), zeta_ratio=("zeta_ratio", "median"),
        peak_gain_ratio=("peak_gain_ratio", "median"), f_peak_true=("f_peak_true", "median"),
        zeta_true=("zeta_true", "median")).reset_index()
    frf_table.to_csv(OUT / "frf_pooled.csv", index=False)
    segments = pd.read_csv(OUT / "modal_segments.csv")
    seg = segments.groupby(["platform", "joint", "model", "segment"]).agg(
        seconds=("seconds", "median"), rmse=("mse", lambda x: float(np.sqrt(np.mean(x)))),
        rmse_rigid=("mse_rigid", lambda x: float(np.sqrt(np.mean(x)))),
        corr_mode_error_te=("corr_mode_error_te", "median")).reset_index()
    seg["vs_rigid"] = seg["rmse"] / seg["rmse_rigid"]
    seg.to_csv(OUT / "modal_segments_pooled.csv", index=False)

    full = ["model", "bags", "rmse", "vs_rigid", "rmse_low", "rmse_mode", "rmse_high", "psd_ratio_low",
            "psd_ratio_mode", "psd_ratio_high", "coherence", "gain", "phase_deg", "delta_corr", "delta_rms_ratio",
            "delta_coherence"]
    short = ["model", "rmse", "vs_rigid", "rmse_low", "rmse_mode", "rmse_high", "coherence", "gain", "phase_deg"]
    notes = OUT / "notes.md"
    lines = ["# 11: time-series and spectral diagnostics of the r9 models", "",
             "Generated by `tools/r9_timeseries.py report` (tables) from `results/r9/timeseries/*.csv`; the findings "
             "are `timeseries/notes.md`, included first. Deployed seeds (validation selection); every model on the "
             "motor-side inputs the networks get; rows past the burn-in. Band RMSE: zero-phase 4th-order Butterworth "
             "on the error, bands < 0.5 f_n, [0.5, 2] f_n, > 2 f_n (f_n the bag's true filter-terms f_n). PSD ratio: "
             "Welch error power over the rigid error's, same band. Coherence, gain |S_yy^| / S_yy and the phase of "
             "S_yy^ at f_n: prediction against tau_s in [0.5, 2] f_n. Medians over bags for the spectral columns, "
             "RMS over bags for the RMSEs. Privileged rows (J1, the filters with true friction, Zollo id_inverse) "
             "are references, never competitors.", ""]
    if notes.exists():
        lines += [notes.read_text(encoding="utf-8").strip(), ""]
    memo = memorisation_table()
    lines += ["## A0. Memorisation: J11, J12 retrained (seed 0), every epoch on 10 train robots and the 30 val robots",
              "", _md(memo) if len(memo) else "pending", ""]
    fit = {p: json.loads((OUT / f"lpf_rigid_{p}.json").read_text()) for p in args.platform
           if (OUT / f"lpf_rigid_{p}.json").exists()}
    lines += ["## LPF-rigid (A4.5): the fitted low-pass per main joint (20 train robots)", "",
              _md(pd.DataFrame([{"platform": p, "joint": j, **v} for p, f in fit.items() for j, v in f["joints"].items()
                                if j in MAIN[p]])), ""]
    lines += ["## A4. Gated joint-regimes, production test", ""]
    for platform, joint, regime in sorted(GATED):
        part = table[(table.platform == platform) & (table.file == "production") & (table.joint == joint)
                     & (table.regime == regime)]
        lines += [f"### {platform} {joint} `{regime}`", "", _md(ordered(part)[full]), ""]
    lines += ["## A4. Non-gated main joint-regimes, production test", ""]
    for platform in args.platform:
        for joint in MAIN[platform]:
            for regime in ("full", "real_like", "off"):
                if (platform, joint, regime) in GATED:
                    continue
                part = table[(table.platform == platform) & (table.file == "production") & (table.joint == joint)
                             & (table.regime == regime)]
                if len(part):
                    lines += [f"### {platform} {joint} `{regime}`", "", _md(ordered(part)[short]), ""]
    lines += ["## A4. Gain-shift file, main joints (all regimes pooled)", ""]
    gain = pooled_table(metrics[metrics.file == "gainshift"].assign(regime="all"))
    for platform in args.platform:
        for joint in MAIN[platform]:
            part = gain[(gain.platform == platform) & (gain.joint == joint)]
            if len(part):
                lines += [f"### {platform} {joint}", "", _md(ordered(part)[short]), ""]
    lines += ["## A4.4 Modal (chirp) file: band errors and the FRF (motor torque -> tau_s, H1)", "",
              "f_peak, half-power zeta and peak gain of the model's FRF over the truth's (medians over the bags whose "
              "excited joint it is). Stop rule (11 A5): any non-oracle model with f_peak within +-15 % and the peak "
              "gain within [0.5, 2] on a gated joint.", ""]
    for platform in args.platform:
        for joint in MAIN[platform]:
            part = table[(table.platform == platform) & (table.file == "modal") & (table.joint == joint)]
            f = frf_table[(frf_table.platform == platform) & (frf_table.joint == joint)]
            if not len(part):
                continue
            merged = part.merge(f[["model", "f_peak_ratio", "zeta_ratio", "peak_gain_ratio"]], on="model", how="left")
            lines += [f"### {platform} {joint} (true peak {f['f_peak_true'].median():.2f} Hz, zeta "
                      f"{f['zeta_true'].median():.3f})", "",
                      _md(ordered(merged)[short + ["f_peak_ratio", "zeta_ratio", "peak_gain_ratio"]]), ""]
    lines += ["## A5. Modal-file error by time segment", "",
              "Segments: the burn-in, the 2 s after it, the resonance passage (the chirp's instantaneous frequency, "
              "from the excited joint's velocity, within [0.7, 1.4] f_n), the rest. `corr_mode_error_te`: median "
              "correlation of the in-band error with the in-band TE torque K e(theta).", ""]
    keep = ["rigid", "lpf_rigid", "J2", "J14", "J5", "J1", "oracle_filter", best_residual("ur10") if "ur10" in
            args.platform else "J11"]
    for platform in args.platform:
        for joint in MAIN[platform]:
            part = seg[(seg.platform == platform) & (seg.joint == joint) & seg.model.isin(keep)]
            if len(part):
                lines += [f"### {platform} {joint}", "",
                          _md(ordered(part)[["model", "segment", "seconds", "rmse", "rmse_rigid", "vs_rigid",
                                             "corr_mode_error_te"]]), ""]
    figures = sorted((OUT / "figures").glob("*.png")) if (OUT / "figures").exists() else []
    lines += ["## A3. Figures", ""] + [f"* `timeseries/figures/{f.name}`" for f in figures]
    (RESULTS / "11_timeseries.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {RESULTS / '11_timeseries.md'}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("predict", "metrics", "figures", "report", "all"))
    parser.add_argument("--platform", nargs="+", default=["iiwa_drive", "ur10"])
    parser.add_argument("--files", nargs="+", default=list(FILES), choices=FILES)
    parser.add_argument("--jobs", nargs="+", default=list(NN_JOBS), help="the networks to predict with")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    args = parser.parse_args(argv)
    steps = ("predict", "metrics", "figures", "report") if args.command == "all" else (args.command,)
    for step in steps:
        {"predict": cmd_predict, "metrics": cmd_metrics, "figures": cmd_figures, "report": cmd_report}[step](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
