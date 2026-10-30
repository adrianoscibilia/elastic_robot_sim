#!/usr/bin/env python3
"""Score the fair classical baselines beside the rigid URDF and the four LNNs (`03_fair_comparison_plan.md`).

Reads ``config/zollo/fair/<platform>.yaml`` and the models ``zollo_fair_fit.py``
wrote (``id_rigid_fair``, ``id_rigid_fair_consistent``, ``id_2m_fair``, and the
percentile-loss ``*_p90`` ones when present), then on the simulated test
split, the gain-shift file and the real files:

1. **FG3**: the URDF scored here reproduces the harness (the contract's
   ``baselines.two_mass.splits.test.rigid.rms``; ``real_zero_shot_rows.csv``)
   to ``references.harness_tolerance``; stops otherwise (``--skip-gates``);
2. **FG4**: the two-mass filter with the nominal URDF, zero friction, nominal
   K and the prior-centre zeta reproduces the contract's
   ``rigid_elastic_nominal_full`` test RMSE to the same tolerance;
3. **FG5 / F7**: every bag is scored; a non-finite bag makes the model
   "failed (n/N)", never an average over the survivors;
4. tables: pooled RMSE and mean NRMSE beside the LNNs (deployed and mean +-
   sd), per joint, per bag (paired, with the LNNs' deployed seeds), the
   gain-shift ratio, the identified K, zeta against the test robots' range;
   one trace figure per file in ``evaluation.plots``.

Inputs are the consumer's (recorded ``q``, ``dq``, SG derivative of ``dq`` with
the evaluated file's contract window); rows are kept when ``t - t_bag_start
>= burn-in`` (the training contract's).  Writes ``<output_dir>/evaluation.{json,csv}``,
``per_bag.csv``, ``paired.csv``, ``per_joint.csv`` and the figures; ``--no-save``
prints only, ``--show`` opens the figures.

Examples
--------
::

    uv run python scripts/zollo_fair_evaluate.py --config config/zollo/fair/ur10.yaml
    uv run python scripts/zollo_fair_evaluate.py --config config/zollo/fair/iiwa_drive.yaml --files sim --show --no-save
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(_REPO / "src"))

from elastic_sim import identification as idn
from elastic_sim.assets import AssetRegistry
from elastic_sim.zollo.config import load_fair_config
from elastic_sim.zollo.data import load_real, sidecar
from elastic_sim.zollo.fair import FairModel, FairPriors, load_fair, nominal_full
from elastic_sim.zollo.predictors import consumer_inputs, score

LNN_MODELS = ("lnn_tau_res", "kalnn_tau_res", "lnn_twomass", "kalnn_twomass")
NAMES = {"urdf": "rigid URDF", "urdf_2m_nominal": "two-mass nominal (physics, FG4)",
         "id_rigid_fair": "ID-rigid-fair", "id_rigid_fair_consistent": "ID-rigid-fair, consistent (2M stage 1)",
         "id_2m_fair": "ID-2M-fair", "id_rigid_fair_p90": "ID-rigid-fair, P90 loss",
         "id_2m_fair_p90": "ID-2M-fair, P90 loss", "lnn_tau_res": "LNN residual", "kalnn_tau_res": "KALNN residual",
         "lnn_twomass": "LNN two-mass", "kalnn_twomass": "KALNN two-mass"}
MODEL_FILES = ("id_rigid_fair", "id_rigid_fair_consistent", "id_2m_fair", "id_rigid_fair_p90", "id_2m_fair_p90")
PLOT_JOINTS = {"iiwa_drive": (1, 3), "ur10": (1, 2)}


def load_models(cfg, rigid_model, priors: FairPriors, n: int) -> dict[str, FairModel]:
    models = {"urdf": FairModel("urdf", n, nominal_full(rigid_model, n), 1.0),
              "urdf_2m_nominal": FairModel("urdf_2m_nominal", n, nominal_full(rigid_model, n), 1.0,
                                           priors.k_init, priors.zeta_init)}
    for name in MODEL_FILES:
        path = cfg.output_dir / f"{name}.json"
        if path.exists():
            models[name] = FairModel.load(path)
        for seed_path in sorted(cfg.output_dir.glob(f"{name}_s[0-9]*.json")):     # F8: every seed, for mean +- sd
            models[seed_path.stem] = FairModel.load(seed_path)
    return models


def label(name: str) -> str:
    base, _, seed = name.rpartition("_s")
    return f"{NAMES[base]}, seed {seed}" if base in NAMES and seed.isdigit() else NAMES[name]


def lnn_tables(cfg, n: int) -> dict[str, Any]:
    """The LNNs' deployed seeds and their sim / real / per-bag / per-joint numbers."""
    import pandas as pd

    sim = pd.read_csv(cfg.resolve(cfg.references["lnn_sim"]))
    sim = sim[(sim.variant == cfg.platform) & sim.model.isin(LNN_MODELS)]
    deployed = {m: int(g.loc[g.best_val_rmse.idxmin()].seed) for m, g in sim.groupby("model")}
    out: dict[str, Any] = {"deployed_seed": deployed, "files": {}}
    entry = {}
    per_bag = pd.read_csv(cfg.resolve(cfg.references["lnn_sim_per_bag"]))
    per_bag = per_bag[per_bag.variant == cfg.platform]
    for model, g in sim.groupby("model"):
        row = g[g.seed == deployed[model]].iloc[0]
        bags = per_bag[(per_bag.model == model) & (per_bag.seed == deployed[model])]
        sse_j = [float((bags.n * bags[f"rmse_joint_{j}"] ** 2).sum()) for j in range(n)]
        entry[model] = {"deployed": float(row.test_rmse), "mean": float(g.test_rmse.mean()),
                        "sd": float(g.test_rmse.std()), "nrmse_mean": float(row.mean_nrmse),
                        "nrmse_mean_mean": float(g.mean_nrmse.mean()),
                        "gs_ratio": float(row.gs_ratio), "gs_ratio_mean": float(g.gs_ratio.mean()),
                        "gainshift_rmse": float(row.gainshift_rmse),
                        "rmse_joints": [float(np.sqrt(s / bags.n.sum())) for s in sse_j],
                        "per_bag": dict(zip(bags.bag, bags.rmse.astype(float)))}
    out["files"]["sim"] = entry
    rows = pd.read_csv(cfg.resolve(cfg.references["real_rows"]))
    real_bag = pd.read_csv(cfg.resolve(cfg.references["real_per_bag"]))
    for name in cfg.real_files():
        sub = rows[(rows.robot == cfg.platform) & (rows.file == name) & (rows.input == "raw") & (rows.calib == "none")
                   & (rows["rows"] == "all")]
        bags = real_bag[(real_bag.robot == cfg.platform) & (real_bag.file == name) & (real_bag.input == "raw")
                        & (real_bag.calib == "none")]
        entry = {}
        for model, g in sub.groupby("model"):
            best = g[g.best_seed.astype(str) == "True"].iloc[0]
            seed = int(best.seed)
            mine = bags[(bags.model == model) & (bags.seed == seed)]
            entry[model] = {"deployed": float(best.rmse), "mean": float(g.rmse.mean()) if len(g) > 1 else None,
                            "sd": float(g.rmse.std()) if len(g) > 1 else None, "seed": seed,
                            "rmse_joints": [float(best[f"j{j}"]) for j in range(n)],
                            "per_bag": dict(zip(mine.bag, mine.rmse.astype(float)))}
        out["files"][f"real_{name}"] = entry
    return out


def transmission_table(cfg, models, lnn) -> list[dict[str, Any]]:
    """Identified K, zeta per joint against the test robots' range (FR_06 Sec 6.2's reading)."""
    import pandas as pd

    frame = pd.read_csv(cfg.resolve(cfg.references["lnn_eval_test"]))
    sim = pd.read_csv(cfg.resolve(cfg.references["lnn_sim"]))
    sim = sim[sim.variant == cfg.platform]
    rows = []
    first = frame[frame.model.str.contains("twomass")].iloc[0]
    n = models["urdf"].n
    for j in range(n):
        row = {"joint": j + 1, "true_K_min": float(first[f"true_K_min_joint_{j}"]),
               "true_K_max": float(first[f"true_K_max_joint_{j}"]),
               "true_zeta_min": float(first[f"true_zeta_min_joint_{j}"]),
               "true_zeta_max": float(first[f"true_zeta_max_joint_{j}"])}
        for name in ("id_2m_fair", "id_2m_fair_p90"):
            if name in models:
                row[f"{name}_K"] = float(models[name].stiffness[j])
                row[f"{name}_zeta"] = float(models[name].zeta[j])
        for model in ("lnn_twomass", "kalnn_twomass"):
            checkpoint = sim[(sim.model == model) & (sim.seed == lnn["deployed_seed"][model])].checkpoint.iloc[0]
            match = frame[frame.checkpoint == checkpoint]
            if len(match):
                row[f"{model}_K"] = float(match.iloc[0][f"learned_K_joint_{j}"])
                row[f"{model}_zeta"] = float(match.iloc[0][f"learned_zeta_joint_{j}"])
        rows.append(row)
    return rows


def run_file(bags, contract, models, rigid_model, head_s, tolerance):
    preds: dict[str, dict[str, np.ndarray]] = {m: {} for m in models}
    diagnostics: dict[str, dict[str, Any]] = {}
    for bag in bags:
        theta, dtheta, ddtheta = consumer_inputs(bag, contract)
        diag: dict[str, Any] = {}
        for name, model in models.items():
            with np.errstate(all="ignore"):
                out = model.predict(rigid_model, theta, dtheta, ddtheta, bag.step)
            preds[name][bag.name] = out["tau"]
            if model.is_two_mass:
                diag[name] = {"max_abs_delta": float(np.nanmax(np.abs(out["delta"]))),
                              "finite": bool(np.isfinite(out["tau"]).all()), "non_pd": out["non_pd"],
                              "damping": out["damping"].tolist()}
        diagnostics[bag.name] = diag
    scores = {m: score(preds[m], bags, head_s, tolerance) for m in models}
    return preds, scores, diagnostics


def pooled(entry: dict[str, Any], n_bags: int) -> str:
    """F7: a model with any non-finite bag has no pooled RMSE."""
    if entry["diverged"]:
        return f"failed ({len(entry['diverged'])}/{n_bags})"
    return f"{entry['rmse']:.3f}"


def paired(per_bag: dict[str, dict[str, float]], reference: str) -> list[dict[str, Any]]:
    from scipy.stats import wilcoxon

    out = []
    ref = per_bag[reference]
    for model, values in per_bag.items():
        if model == reference:
            continue
        common = [b for b in ref if b in values and np.isfinite(ref[b]) and np.isfinite(values[b])]
        if len(common) < 2:
            continue
        diff = np.asarray([ref[b] - values[b] for b in common])
        p = float(wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0
        out.append({"reference": reference, "other": model, "bags": len(common), "mean_reference": float(np.mean(
            [ref[b] for b in common])), "mean_other": float(np.mean([values[b] for b in common])),
            "mean_diff": float(diff.mean()), "reference_wins": int((diff < 0).sum()), "p_wilcoxon": p})
    return out


def plot(cfg, file, bag, preds, head_s, joints, show, path):
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = bag.t - bag.t[0]
    fig, axes = plt.subplots(len(joints), 1, figsize=(10, 3.2 * len(joints)), sharex=True)
    for ax, j in zip(np.atleast_1d(axes), joints):
        ax.plot(t, bag.tau_J[:, j], color="k", lw=1.2, label="measured $\\tau_J$")
        for name, style in (("urdf", ":"), ("id_rigid_fair", "-"), ("id_2m_fair", "-")):
            if name in preds and bag.name in preds[name]:
                ax.plot(t, preds[name][bag.name][:, j], ls=style, lw=1.0, label=label(name))
        ax.axvspan(0, head_s, color="0.9", zorder=0)
        ax.set_ylabel(f"joint {j + 1} [N m]")
    np.atleast_1d(axes)[0].legend(fontsize=8, ncol=4, loc="upper right")
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
    parser.add_argument("--config", default="config/zollo/fair/ur10.yaml")
    parser.add_argument("--files", nargs="+", default=None, help="sim, gainshift, real_A, real_B (default all)")
    parser.add_argument("--skip-gates", action="store_true", help="report even if FG3 / FG4 fail (says so)")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    cfg = load_fair_config(args.config, _REPO)
    rigid_model = idn.build_model(AssetRegistry.for_repository(_REPO).load(cfg.asset))
    train_contract = sidecar(cfg.dataset, "contract")
    n = int(train_contract["n_dof"])
    step = 1.0 / float(train_contract["differentiation"]["sample_rate_hz"])
    priors = FairPriors.from_contract(train_contract, step)
    head_s = priors.burn_in_s
    models = load_models(cfg, rigid_model, priors, n)
    lnn = lnn_tables(cfg, n)
    tol = float(cfg.references.get("harness_tolerance", 1e-3))
    print(f"{cfg.platform}: models {list(models)}; burn-in {head_s} s; LNN deployed seeds {lnn['deployed_seed']}")
    files = args.files or ["sim", "gainshift"] + [f"real_{k}" for k in cfg.real_files()]
    out: dict[str, Any] = {"platform": cfg.platform, "burn_in_s": head_s, "models": {
        k: {"epsilon": m.epsilon, "stiffness": None if m.stiffness is None else m.stiffness.tolist(),
            "zeta": None if m.zeta is None else m.zeta.tolist(), "info": {i: v for i, v in m.info.items()
                                                                           if i != "log_cholesky"}}
        for k, m in models.items()}, "lnn": {"deployed_seed": lnn["deployed_seed"]}, "files": {}}
    rows, bag_rows, joint_rows, pair_rows = [], [], [], []
    for file in files:
        if file in ("sim", "gainshift"):
            path = cfg.dataset if file == "sim" else cfg.resolve(cfg.evaluation["gainshift"])
            bags, info = load_fair(path, ("test",))
            contract, tolerance = info["contract"], 1e-9            # round7.bag_baselines' row rule
        else:
            bags, info = load_real(cfg.real_files()[file.split("_", 1)[1]])
            contract, tolerance = info["contract"], 0.0             # real_eval/analyze3.py's row rule
        print(f"\n== {file}: {len(bags)} bags ({Path(info['path']).name})")
        preds, scores, diagnostics = run_file(bags, contract, models, rigid_model, head_s, tolerance)
        gates: dict[str, Any] = {}
        if file == "sim":
            baselines = train_contract["baselines"]["two_mass"]["splits"]["test"]
            gates["FG3"] = {"mine": scores["urdf"]["rmse"], "harness": baselines["rigid"]["rms"]}
            gates["FG4"] = {"mine": scores["urdf_2m_nominal"]["rmse"],
                            "harness": baselines["rigid_elastic_nominal_full"]["rms"]}
        elif file.startswith("real"):
            gates["FG3"] = {"mine": scores["urdf"]["rmse"], "harness": lnn["files"][file]["rigid_URDF"]["deployed"]}
        for name, gate in gates.items():
            gate["pass"] = bool(abs(gate["mine"] - gate["harness"]) <= tol)
            print(f"{name}: {gate['mine']:.4f} vs harness {gate['harness']:.4f} -> {'PASS' if gate['pass'] else 'FAIL'}")
            if not gate["pass"] and not args.skip_gates:
                raise SystemExit(f"{name} failed; not reporting (--skip-gates overrides)")
        two_mass = [m for m, v in models.items() if v.is_two_mass]
        fg5 = {m: {"diverged": scores[m]["diverged"], "max_abs_delta": max(d[m]["max_abs_delta"] for d in
                                                                              diagnostics.values()),
                   "non_pd": sum(d[m]["non_pd"] for d in diagnostics.values())} for m in two_mass}
        print("FG5: " + "; ".join(f"{m} diverged {len(v['diverged'])}/{len(bags)}, max |delta| "
                                  f"{v['max_abs_delta']:.2e} rad, non-PD M samples {v['non_pd']}" for m, v in fg5.items()))
        reference = lnn["files"].get(file if file != "gainshift" else "none", {})
        print(f"{'model':40s} {'RMSE':>14s} {'NRMSE':>7s}")
        for m, s in scores.items():
            print(f"{label(m):40s} {pooled(s, len(bags)):>14s} {s.get('nrmse_mean', np.nan):7.3f}")
            rows.append({"platform": cfg.platform, "file": file, "model": m, "rmse": None if s["diverged"] else s["rmse"],
                         "status": pooled(s, len(bags)), "nrmse_mean": s.get("nrmse_mean"), "rows": s["rows"],
                         "n_bags": len(bags)})
            joint_rows.append({"platform": cfg.platform, "file": file, "model": m,
                               **{f"rmse_j{j + 1}": v for j, v in enumerate(s.get("rmse_joints", []))},
                               **{f"nrmse_j{j + 1}": v for j, v in enumerate(s.get("nrmse_joints", []))}})
            bag_rows += [{"platform": cfg.platform, "file": file, "model": m, "bag": b, "rmse": v}
                         for b, v in s["per_bag"].items()]
        targets = np.vstack([b.tau_J[(b.t - b.t[0]) >= head_s - tolerance] for b in bags])
        target_rms = np.sqrt(np.mean(targets ** 2, axis=0))
        for m in LNN_MODELS:
            if m in reference:
                v = reference[m]
                nrmse = float(np.mean(np.asarray(v["rmse_joints"]) / target_rms))
                spread = f"(mean {v['mean']:.3f} ± {v['sd']:.3f})" if v.get("mean") is not None else ""
                print(f"{NAMES[m]:40s} {v['deployed']:14.3f} {nrmse:7.3f}  deployed {spread}")
                rows.append({"platform": cfg.platform, "file": file, "model": m, "rmse": v["deployed"],
                             "status": f"{v['deployed']:.3f}", "nrmse_mean": nrmse, "mean": v.get("mean"),
                             "sd": v.get("sd"), "n_bags": len(bags)})
                joint_rows.append({"platform": cfg.platform, "file": file, "model": m,
                                   **{f"rmse_j{j + 1}": x for j, x in enumerate(v["rmse_joints"])},
                                   **{f"nrmse_j{j + 1}": x / r for j, (x, r) in enumerate(zip(v["rmse_joints"],
                                                                                              target_rms))}})
                bag_rows += [{"platform": cfg.platform, "file": file, "model": m, "bag": b, "rmse": x}
                             for b, x in v["per_bag"].items()]
        if file != "gainshift":
            per_bag = {m: s["per_bag"] for m, s in scores.items() if m != "urdf_2m_nominal"}
            per_bag.update({m: reference[m]["per_bag"] for m in LNN_MODELS if m in reference})
            for ref in ("id_2m_fair", "id_rigid_fair"):
                if ref in per_bag:
                    pair_rows += [{"platform": cfg.platform, "file": file, **p} for p in paired(per_bag, ref)]
        out["files"][file] = {"path": info["path"], "bags": [b.name for b in bags], "gates": gates, "fg5": fg5,
                              "scores": scores, "target_rms": target_rms.tolist(), "diagnostics": diagnostics}
        plot_bag = (cfg.evaluation.get("plots") or {}).get(file)
        chosen = next((b for b in bags if b.name == plot_bag), None)
        if chosen is not None and (args.show or not args.no_save):
            path = None if args.no_save else cfg.output_dir / f"traces_{file}_{chosen.name}.png"
            plot(cfg, file, chosen, preds, head_s, PLOT_JOINTS.get(cfg.platform, (0, 1)), args.show, path)
    if "sim" in out["files"] and "gainshift" in out["files"]:
        gs = {}
        for m in models:
            a, b = out["files"]["gainshift"]["scores"][m], out["files"]["sim"]["scores"][m]
            gs[m] = None if (a["diverged"] or b["diverged"]) else a["rmse"] / b["rmse"]
        gs.update({m: {"deployed": v["gs_ratio"], "mean": v["gs_ratio_mean"]} for m, v in lnn["files"]["sim"].items()})
        out["gain_shift_ratio"] = gs
        print("\ngain-shift ratio (gain-shift RMSE / test RMSE): " + ", ".join(
            f"{label(m)} {v['deployed'] if isinstance(v, dict) else v:.3f}" for m, v in gs.items() if v is not None))
    out["transmission"] = transmission_table(cfg, models, lnn)
    print("\nK, zeta vs the test robots' range:")
    for r in out["transmission"]:
        print("  " + ", ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in r.items()))
    out["paired"] = pair_rows
    if args.no_save:
        print("\n(--no-save: nothing written)")
        return
    import pandas as pd

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(cfg.output_dir / "evaluation.csv", index=False)
    pd.DataFrame(bag_rows).to_csv(cfg.output_dir / "per_bag.csv", index=False)
    pd.DataFrame(joint_rows).to_csv(cfg.output_dir / "per_joint.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(cfg.output_dir / "paired.csv", index=False)
    pd.DataFrame(out["transmission"]).to_csv(cfg.output_dir / "transmission.csv", index=False)
    (cfg.output_dir / "evaluation.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(f"\nwrote evaluation.{{json,csv}}, per_bag.csv, per_joint.csv, paired.csv, transmission.csv in {cfg.output_dir}")


if __name__ == "__main__":
    main()
