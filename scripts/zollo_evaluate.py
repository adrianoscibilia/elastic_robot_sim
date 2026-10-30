#!/usr/bin/env python3
"""Score the Zollo-identified predictors next to the rigid URDF and the four LNNs.

Reads ``config/zollo/<platform>.yaml`` and one identification written by
``zollo_identify.py`` (variant B, pooled by default), then, on the
simulated test split and the real files:

1. **harness gate**: the nominal URDF scored by this script must reproduce
   the existing harness's rigid-URDF numbers (the training contract's
   ``baselines.two_mass.splits.<split>.rigid.rms`` and
   ``real_zero_shot_rows.csv``) to ``references.harness_tolerance``; the script
   stops otherwise (``--skip-gate`` overrides, and says so);
2. the predictors -- ``nominal`` (rigid URDF), ``id_rigid``, ``id_elastic``
   (as identified), ``id_elastic_zeta`` (labelled ablation: damping floored at
   the prior's zeta_min), ``id_elastic_bounded`` (K clipped to the two-mass
   models' stiffness bounds, then D floored with the clipped K;
   `04_variant_A_elastic.md` Sec 2), ``id_inverse`` (simulation only, true
   link state) -- for A-pc (`06_variant_A_consistent.md`) K, D are the
   medians of the per-robot K_r, D_r (C3) --
   on the inputs the two-mass models get, scored on the harness's rows;
3. a table beside the LNNs' numbers from the r7 result CSVs, the per-bag
   errors, ID-elastic's integration diagnostics, and one trace figure per
   file (``evaluation.plots``).

Writes ``<output_dir>/evaluation_<identification stem>.{csv,per_bag.csv,json}``
and the figures; ``--no-save`` prints only, ``--show`` opens the figures.

Examples
--------
::

    uv run python scripts/zollo_evaluate.py --config config/zollo/ur10.yaml
    uv run python scripts/zollo_evaluate.py --config config/zollo/iiwa_drive.yaml --files sim --show --no-save
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(_REPO / "src"))

from elastic_sim import identification as idn
from elastic_sim.assets import AssetRegistry
from elastic_sim.zollo.config import load_platform
from elastic_sim.zollo.data import load_real, load_sim, sidecar
from elastic_sim.zollo.identify import load_identification
from elastic_sim.zollo.predictors import (
    ElasticModel, ElasticSettings, LinkModel, TransmissionBounds, consumer_inputs, identified_transmission,
    is_positive_definite, per_robot_transmission, score,
)

LNN_MODELS = ("lnn_tau_res", "kalnn_tau_res", "lnn_twomass", "kalnn_twomass")
LNN_NAMES = {"lnn_tau_res": "LNN residual", "kalnn_tau_res": "KALNN residual", "lnn_twomass": "LNN two-mass",
             "kalnn_twomass": "KALNN two-mass"}
PREDICTORS = ("nominal", "id_rigid", "id_elastic", "id_elastic_zeta", "id_elastic_bounded", "id_inverse")
ELASTIC = ("id_elastic", "id_elastic_zeta", "id_elastic_bounded")
PREDICTOR_NAMES = {"nominal": "rigid URDF (nominal)", "id_rigid": "ID-rigid", "id_elastic": "ID-elastic",
                   "id_elastic_zeta": "ID-elastic, zeta >= zeta_min (ablation)",
                   "id_elastic_bounded": "ID-elastic, bounded (K clipped, D floored)",
                   "id_inverse": "ID-inverse (sim diagnostic)"}
# Load-bearing joints for the trace figures (02_NUMBERS.md N4b).
PLOT_JOINTS = {"iiwa_drive": (1, 3), "ur10": (1, 2)}


def lnn_reference(cfg, file: str) -> dict[str, dict[str, float]]:
    """mean, sd and deployed (best-validation seed) RMSE of the four LNNs on ``file``."""
    import pandas as pd

    out: dict[str, dict[str, float]] = {}
    if file == "sim":
        frame = pd.read_csv(cfg.resolve(cfg.references["lnn_sim"]))
        frame = frame[(frame.variant == cfg.platform) & frame.model.isin(LNN_MODELS)]
        for model, group in frame.groupby("model"):
            deployed = group.loc[group.best_val_rmse.idxmin()]
            out[model] = {"mean": float(group.test_rmse.mean()), "sd": float(group.test_rmse.std()),
                          "deployed": float(deployed.test_rmse)}
        return out
    frame = pd.read_csv(cfg.resolve(cfg.references["real_rows"]))
    frame = frame[(frame.robot == cfg.platform) & (frame.file == file.split("_", 1)[1]) & (frame.input == "raw")
                  & (frame.calib == "none") & (frame["rows"] == "all")]
    for model, group in frame[frame.model.isin(LNN_MODELS)].groupby("model"):
        out[model] = {"mean": float(group.rmse.mean()), "sd": float(group.rmse.std()),
                      "deployed": float(group[group.best_seed.astype(str) == "True"].rmse.iloc[0])}
    for model in ("rigid_URDF", "classical_ID_otherfile"):
        rows = frame[frame.model == model]
        if len(rows):
            out[model] = {"deployed": float(rows.rmse.iloc[0])}
    return out


def harness_reference(cfg, file: str, sim_contract: dict[str, Any], split: str) -> float:
    if file == "sim":
        return float(sim_contract["baselines"]["two_mass"]["splits"][split]["rigid"]["rms"])
    return lnn_reference(cfg, file)["rigid_URDF"]["deployed"]


def run_file(file: str, bags, contract, predictors, models, head_s, tolerance, zeta_min, verbose):
    preds: dict[str, dict[str, np.ndarray]] = {p: {} for p in predictors}
    diagnostics: dict[str, dict[str, Any]] = {}
    for index, bag in enumerate(bags):
        theta, dtheta, ddtheta = consumer_inputs(bag, contract)
        started = time.perf_counter()
        if "nominal" in predictors:
            preds["nominal"][bag.name] = models["nominal"].torque(theta, dtheta, ddtheta)
        if "id_rigid" in predictors:
            preds["id_rigid"][bag.name] = models["link"].torque(theta, dtheta, ddtheta)
        if "id_inverse" in predictors and bag.q is not None:
            from elastic_sim.diagnostics import savitzky_golay_acceleration

            diff = contract.get("differentiation") or {}
            ddq = savitzky_golay_acceleration(bag.dq, bag.step, window=int(diff.get("sg_window") or 11),
                                              order=int(diff.get("sg_poly") or 3))
            preds["id_inverse"][bag.name] = models["link"].torque(bag.q, bag.dq, ddq)
        probes = [theta[i] for i in np.linspace(0, len(theta) - 1, 20).astype(int)]
        eig = np.asarray([np.linalg.eigvalsh(models["link"].mass(q))[0] for q in probes])
        diag: dict[str, Any] = {"non_pd_samples_rigid": int(sum(not is_positive_definite(models["link"].mass(q))
                                                                 for q in probes)), "pd_probes": len(probes),
                                "min_eig_probes": float(eig.min())}
        for name in ELASTIC:
            if name not in predictors:
                continue
            if name == "id_elastic":
                model = models["elastic"]
            elif name == "id_elastic_zeta":
                model = models["elastic"].with_damping_floor(zeta_min, probes)
            else:
                model = models["bounded"].with_damping_floor(zeta_min, probes)
            run = model.run(theta, dtheta, bag.step)
            preds[name][bag.name] = run.tau
            elastic = run.tau - models["link"].torque(theta, dtheta, ddtheta)
            diag[name] = {"substeps": run.substeps, "omega_max_hz": run.omega_max / (2 * np.pi),
                          "time_step": bag.step / run.substeps, "diverged_at_sample": run.diverged_at,
                          "diverged_at_s": None if run.diverged_at is None else float(
                              bag.t[int(run.diverged_at)] - bag.t[0]),
                          "diverged_joint": None if run.diverged_joint is None else run.diverged_joint + 1,
                          "stiffness": model.K.tolist(),
                          "non_pd_samples": run.non_pd_samples, "equilibrium_residual": run.equilibrium_residual,
                          "elastic_part_rms": float(np.sqrt(np.nanmean(elastic ** 2))), "notes": run.notes,
                          "damping": model.D.tolist()}
            # A bounded but growing oscillation never reaches divergence_rad; flag a bag whose
            # elastic correction is larger than half the target (the arms' elastic content is <= 4 %).
            diag[name]["unstable"] = bool(diag[name]["elastic_part_rms"] > 0.5 * np.sqrt(np.mean(bag.tau_J ** 2)))
        diagnostics[bag.name] = diag
        if verbose:
            print(f"  [{file} {index + 1}/{len(bags)}] {bag.name} {time.perf_counter() - started:.1f} s", flush=True)
    scores = {p: score(preds[p], bags, head_s, tolerance) for p in predictors if preds[p]}
    return preds, scores, diagnostics


def plot(cfg, file: str, bag, preds, head_s: float, joints, show: bool, path: Path | None) -> None:
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = bag.t - bag.t[0]
    fig, axes = plt.subplots(len(joints), 1, figsize=(10, 3.2 * len(joints)), sharex=True)
    for ax, j in zip(np.atleast_1d(axes), joints):
        ax.plot(t, bag.tau_J[:, j], color="k", lw=1.2, label="measured $\\tau_J$")
        for name, style in (("nominal", ":"), ("id_rigid", "-"), ("id_elastic", "-"), ("id_elastic_bounded", "-"),
                            ("id_inverse", "--")):
            if name in preds and bag.name in preds[name]:
                ax.plot(t, preds[name][bag.name][:, j], ls=style, lw=1.0, label=PREDICTOR_NAMES[name])
        ax.axvspan(0, head_s, color="0.9", zorder=0)
        ax.set_ylabel(f"joint {j + 1} [N m]")
    np.atleast_1d(axes)[0].legend(fontsize=8, ncol=3, loc="upper right")
    np.atleast_1d(axes)[-1].set_xlabel("t [s] (grey: burn-in, not scored)")
    fig.suptitle(f"{cfg.platform} {file}: {bag.name}")
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=130)
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config/zollo/ur10.yaml")
    parser.add_argument("--identification", default=None,
                        help="an identification JSON (default <output_dir>/identification_B_pooled.json)")
    parser.add_argument("--files", nargs="+", default=None, help="sim, real_A, real_B (default all)")
    parser.add_argument("--predictors", nargs="+", default=list(PREDICTORS), choices=PREDICTORS)
    parser.add_argument("--max-bags", type=int, default=None, help="first N bags per file (quick look)")
    parser.add_argument("--skip-gate", action="store_true", help="report even if the harness gate fails")
    parser.add_argument("--show", action="store_true", help="open the trace figures")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    cfg = load_platform(args.config, _REPO)
    id_path = Path(args.identification) if args.identification else cfg.output_dir / "identification_B_pooled.json"
    result = load_identification(id_path)
    rigid_model = idn.build_model(AssetRegistry.for_repository(_REPO).load(result.meta["asset"]))
    ev = cfg.evaluation
    split = ev.get("sim_split", "test")
    train_contract = sidecar(cfg.dataset, "contract")
    priors = train_contract["transmission_priors"]
    head_s = float(ev.get("burn_in_s") or priors["burn_in_s"])
    settings = ElasticSettings.from_mapping(ev.get("elastic"))
    bounds = TransmissionBounds.from_two_mass(train_contract, ev.get("lnn_repo") and cfg.resolve(ev["lnn_repo"]))
    if not np.isclose(bounds.zeta_min, float(priors["zeta_min"])):
        raise SystemExit(f"zeta_min {bounds.zeta_min} differs from the contract's {priors['zeta_min']}")
    models = {"nominal": LinkModel(rigid_model, idn.standard_parameters(*rigid_model[:2])),
              "link": LinkModel.from_identification(result, rigid_model),
              "bounded": ElasticModel.bounded(result, rigid_model, bounds, settings)}
    no_elastic = None
    try:
        models["elastic"] = ElasticModel.from_identification(result, rigid_model, settings)
    except ValueError as error:
        # An identified K <= 0 has no forward model; that is a result, reported, not a crash.
        # The bounded model still runs: its K is clipped into the two-mass bounds.
        no_elastic = str(error)
        args.predictors = [p for p in args.predictors if p not in ("id_elastic", "id_elastic_zeta")]
    K_id, D_id = identified_transmission(result)
    per_robot = per_robot_transmission(result)
    files = args.files or ["sim"] + [f"real_{k}" for k in cfg.real_files()]
    tol = float(cfg.references.get("harness_tolerance", 1e-3))
    print(f"{cfg.platform}: {id_path.name} (variant {result.config.variant}, {result.config.transmission}); "
          f"burn-in {head_s} s; " + (f"no ID-elastic: {no_elastic}" if no_elastic else
                                      f"K {np.round(models['elastic'].K, 1).tolist()}, "
                                      f"D {np.round(models['elastic'].D, 2).tolist()}"))
    K_used = models["bounded"].K
    print(f"bounds ({bounds.source}): K_lo {np.round(bounds.k_low, 1).tolist()}, K_hi "
          f"{np.round(bounds.k_high, 1).tolist()}, zeta_min {bounds.zeta_min:.4f}")
    if per_robot:
        print(f"C3: K, D = median over {len(result.layout.robots)} training robots of the per-robot K_r, D_r")
    print(f"bounded K {np.round(K_used, 1).tolist()} (clipped joints "
          f"{[j + 1 for j in np.flatnonzero(K_used != K_id)]})")
    out: dict[str, Any] = {"identification": str(id_path), "burn_in_s": head_s, "settings": vars(settings),
                           "zeta_min": priors["zeta_min"], "no_elastic": no_elastic,
                           "bounds": {**bounds.describe(), "K_identified": K_id.tolist(),
                                      "D_identified": None if D_id is None else D_id.tolist(),
                                      "K_bounded": K_used.tolist(),
                                      "K_clipped": [bool(c) for c in K_used != K_id],
                                      "predictor_transmission": "median over the training robots (C3)" if per_robot
                                      else "pooled, as identified",
                                      **{f"{g}_per_robot_{f.__name__}": f(v, axis=0).tolist()
                                         for g, v in per_robot.items() for f in (np.min, np.max)}},
                           "files": {}}
    rows, bag_rows = [], []
    for file in files:
        if file == "sim":
            bags, info = load_sim(cfg.dataset, splits=(split,), truth=False)
            contract, tolerance = info["contract"], 1e-9          # round7.bag_baselines' row rule
            predictors = list(args.predictors)
        else:
            path = cfg.real_files()[file.split("_", 1)[1]]
            bags, info = load_real(path)
            contract, tolerance = info["contract"], 0.0          # real_eval/analyze3.py's row rule
            predictors = [p for p in args.predictors if p != "id_inverse"]
        if args.max_bags:
            bags = bags[:args.max_bags]
        print(f"\n== {file}: {len(bags)} bags ({Path(info['path']).name})")
        preds, scores, diagnostics = run_file(file, bags, contract, predictors, models, head_s, tolerance,
                                              float(priors["zeta_min"]), verbose=True)
        reference = harness_reference(cfg, file, train_contract, split)
        mine = scores["nominal"]["rmse"] if "nominal" in scores else float("nan")
        gate = (not args.max_bags) and abs(mine - reference) <= tol
        print(f"harness gate: nominal {mine:.4f} vs harness {reference:.4f} -> "
              f"{'PASS' if gate else ('n/a (--max-bags)' if args.max_bags else 'FAIL')}")
        if not gate and not args.max_bags and not args.skip_gate:
            raise SystemExit("the scorer does not reproduce the harness; not reporting (--skip-gate overrides)")
        lnn = lnn_reference(cfg, file)
        out["files"][file] = {"path": info["path"], "bags": [b.name for b in bags], "gate": {
            "mine": mine, "harness": reference, "pass": bool(gate)}, "scores": scores, "lnn": lnn,
            "diagnostics": diagnostics}
        print(f"{'model':42s} {'RMSE':>8s} {'NRMSE':>7s}  diverged/unstable")
        for p, s in scores.items():
            n_div = len(s["diverged"])
            print(f"{PREDICTOR_NAMES[p]:42s} {s['rmse']:8.3f} {s.get('nrmse_mean', np.nan):7.3f}  "
                  f"{n_div}/{len(bags)}" + (f" {s['diverged']}" if n_div else ""))
            rows.append({"platform": cfg.platform, "file": file, "model": p, "rmse": s["rmse"],
                         "nrmse_mean": s.get("nrmse_mean"), "rows": s["rows"], "n_bags": len(bags),
                         "diverged": ";".join(s["diverged"]),
                         **{f"rmse_j{j}": v for j, v in enumerate(s.get("rmse_joints", []))},
                         **{f"nrmse_j{j}": v for j, v in enumerate(s.get("nrmse_joints", []))}})
            bag_rows += [{"platform": cfg.platform, "file": file, "model": p, "bag": b, "rmse": v}
                         for b, v in s["per_bag"].items()]
        for model in LNN_MODELS:
            if model in lnn:
                v = lnn[model]
                print(f"{LNN_NAMES[model]:42s} {v['deployed']:8.3f}          (deployed; mean {v['mean']:.3f} "
                      f"± {v['sd']:.3f})")
        if "classical_ID_otherfile" in lnn:
            print(f"{'classical ID fitted on the other real file':42s} {lnn['classical_ID_otherfile']['deployed']:8.3f}")
        for name in ELASTIC:
            if name in predictors:
                d = [v[name] for v in diagnostics.values()]
                print(f"{name}: substeps {sorted({x['substeps'] for x in d})}, omega_max "
                      f"{max(x['omega_max_hz'] for x in d):.0f} Hz, non-PD samples {sum(x['non_pd_samples'] for x in d)}"
                      f", diverged {sum(x['diverged_at_sample'] is not None for x in d)}, unstable "
                      f"{[b for b, v in diagnostics.items() if v[name]['unstable']]}")
                for b, v in diagnostics.items():
                    x = v[name]
                    if x["diverged_at_sample"] is not None:
                        j = x["diverged_joint"]
                        print(f"  diverged: {b} at t = {x['diverged_at_s']:.3f} s, joint {j}, K {x['stiffness'][j - 1]:.4g}"
                              f", D {x['damping'][j - 1]:.4g}")
        min_eig = [v["min_eig_probes"] for v in diagnostics.values()]
        pd_probes = sum(v["pd_probes"] - v["non_pd_samples_rigid"] for v in diagnostics.values())
        all_probes = sum(v["pd_probes"] for v in diagnostics.values())
        out["files"][file]["mass_matrix"] = {"pd_probes": pd_probes, "probes": all_probes,
                                             "min_eig": float(min(min_eig))}
        print(f"M_hat at the probes: PD {pd_probes}/{all_probes}, min eig {min(min_eig):.4g}")
        if "id_elastic_bounded" in predictors:
            floors = np.asarray([v["id_elastic_bounded"]["damping"] for v in diagnostics.values()])
            D_ref = np.zeros_like(K_id) if D_id is None else D_id
            out["files"][file]["bounds"] = {"D_used_min": floors.min(axis=0).tolist(),
                                            "D_used_max": floors.max(axis=0).tolist(),
                                            "D_raised_bags": (floors > D_ref).sum(axis=0).tolist()}
            diverged = [b for b, v in diagnostics.items() if v["id_elastic_bounded"]["diverged_at_sample"] is not None]
            print(f"G3: M_hat PD at every probe {pd_probes == all_probes}; bounded ID-elastic diverged "
                  f"{len(diverged)}/{len(bags)} -> {'PASS' if pd_probes == all_probes and not diverged else 'FAIL'}")
            spans = {g: (v.min(axis=0), v.max(axis=0)) for g, v in per_robot.items()}
            head = f"{'K_r range':>19s} " if "stiffness" in spans else ""
            dhead = f"{'D_r range':>17s} " if "damping" in spans else ""
            print(f"{'joint':>5s} {head}{'K pred':>10s} {'K lo':>9s} {'K hi':>9s} {'K used':>10s} {dhead}{'D pred':>9s} "
                  f"{'D used (bags)':>19s} raised")
            for j in range(len(K_id)):
                k_span = (f"{spans['stiffness'][0][j]:9.4g}-{spans['stiffness'][1][j]:<9.4g} "
                          if "stiffness" in spans else "")
                d_span = f"{spans['damping'][0][j]:8.3g}-{spans['damping'][1][j]:<8.3g} " if "damping" in spans else ""
                print(f"{j + 1:5d} {k_span}{K_id[j]:10.4g} {bounds.k_low[j]:9.4g} {bounds.k_high[j]:9.4g} "
                      f"{K_used[j]:10.4g}{'*' if K_used[j] != K_id[j] else ' '}{d_span}{D_ref[j]:9.3g} "
                      f"{floors[:, j].min():9.3g}-{floors[:, j].max():<9.3g} "
                      f"{int((floors[:, j] > D_ref[j]).sum())}/{len(floors)}")
        plot_bag = (ev.get("plots") or {}).get(file)
        chosen = next((b for b in bags if b.name == plot_bag), None)
        if chosen is not None and (args.show or not args.no_save):
            path = None if args.no_save else cfg.output_dir / f"traces_{id_path.stem}_{file}_{chosen.name}.png"
            plot(cfg, file, chosen, preds, head_s, PLOT_JOINTS.get(cfg.platform, (0, 1)), args.show, path)
    if args.no_save:
        print("\n(--no-save: nothing written)")
        return
    stem = cfg.output_dir / f"evaluation_{id_path.stem}"
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix, table in ((".csv", rows), (".per_bag.csv", bag_rows)):
        with Path(str(stem) + suffix).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for r in table for k in r)))
            writer.writeheader()
            writer.writerows(table)
    Path(str(stem) + ".json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(f"\nwrote {stem}.csv, .per_bag.csv, .json and the trace figures in {cfg.output_dir}")


if __name__ == "__main__":
    main()
