#!/usr/bin/env python3
"""Fit the fair classical baselines, ID-rigid-fair and ID-2M-fair (`03_fair_comparison_plan.md`).

Reads ``config/zollo/fair/<platform>.yaml``, then:

1. **F4 check**: the priors read from the contract (K bounds widened x1.5,
   zeta in [prior low, max(1.5 prior high, 1)], burn-in, step) must equal the
   LNN two-mass run's ``model_kwargs`` (``references.lnn_chain``);
2. **FG1**: the train/val row counts, the two-mass windows and the target
   scale must equal the LNN training code's (``scripts/lnn_row_counts.py``
   under ``lnn_pipeline.python``); stops otherwise (``--skip-fg1`` overrides
   and says so);
3. **ID-rigid-fair** for every point of the selection grid (Coulomb ``eps``
   x row weighting), the best validation RMSE over every val row kept (the
   residual LNNs' statistic);
4. **ID-2M-fair** by output error for every grid point, each started from
   that point's ID-rigid-fair, the best validation RMSE over whole val bags
   after the burn-in kept (the two-mass LNNs' statistic).

Only ``t``, ``bag``, ``split``, ``q``, ``dq``, ``ft`` of the train and val
splits are read (F1).  Writes ``<output_dir>/id_rigid_fair.json``,
``id_2m_fair.json`` and ``fit_summary.json``; ``--no-save`` prints only.

Examples
--------
::

    uv run python scripts/zollo_fair_fit.py --config config/zollo/fair/ur10.yaml
    uv run python scripts/zollo_fair_fit.py --config config/zollo/fair/iiwa_drive.yaml --models rigid --no-save
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(_REPO / "src"))

from elastic_sim import identification as idn
from elastic_sim.assets import AssetRegistry
from elastic_sim.zollo.config import load_fair_config
from elastic_sim.zollo.fair import (
    BagRows, FairPriors, OutputErrorSettings, check_dynamics, fit_rigid, fit_two_mass, load_fair, mass_regressor,
    non_pd_share, row_counts, target_scale,
)


def check_priors(cfg, priors: FairPriors) -> dict[str, Any]:
    """F4: the priors equal the LNN two-mass run's model_kwargs."""
    chain = yaml.safe_load(cfg.resolve(cfg.references["lnn_chain"]).read_text(encoding="utf-8"))
    kwargs = next(r for r in chain["runs"] if r["base_name"] == "lnn_twomass")["model_kwargs"]
    stiffness = np.asarray(kwargs["stiffness_range"], dtype=float)
    ratio = np.asarray(kwargs["damping_ratio_range"], dtype=float)
    checks = {
        "time_step": abs(float(kwargs["time_step"]) - priors.step) < 1e-9,
        "burn_in_s": abs(float(kwargs["burn_in_s"]) - priors.burn_in_s) < 1e-12,
        "stiffness_bounds": np.allclose(stiffness[:, 0] / 1.5, priors.k_low, rtol=1e-12)
        and np.allclose(stiffness[:, 1] * 1.5, priors.k_high, rtol=1e-12),
        "zeta_low": np.allclose(ratio[:, 0], priors.zeta_low, rtol=1e-12),
        "zeta_high": np.allclose(np.asarray(kwargs["zeta_max"], dtype=float), priors.zeta_high, rtol=1e-12),
        "coupling_full": kwargs.get("coupling") == "full" and kwargs.get("integrator") == "trapezoidal",
    }
    if not all(checks.values()):
        raise SystemExit(f"F4: the contract priors differ from the LNN two-mass run's: {checks}")
    return checks


def fg1(cfg, priors: FairPriors, mine: dict[str, int], scale: np.ndarray) -> dict[str, Any]:
    pipeline = cfg.lnn_pipeline
    command = [os.fspath(cfg.resolve(pipeline["python"])), os.fspath(_REPO / "scripts" / "lnn_row_counts.py"),
               "--repo", os.fspath(cfg.resolve(pipeline["repo"])), "--dataset", os.fspath(cfg.dataset),
               "--time-step", repr(priors.step), "--burn-in", repr(priors.burn_in_s)]
    done = subprocess.run(command, capture_output=True, text=True, check=True)
    theirs = json.loads(done.stdout.strip().splitlines()[-1])
    rows = {k: {"lnn": theirs[k], "id": mine[k], "equal": theirs[k] == mine[k]} for k in mine}
    scale_error = float(np.max(np.abs(np.asarray(theirs["target_scale"]) - scale) / scale))
    return {"rows": rows, "target_scale_rel_error": scale_error, "lnn_target_scale": theirs["target_scale"],
            "pass": all(r["equal"] for r in rows.values()) and scale_error < 1e-6, "lnn": theirs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config/zollo/fair/ur10.yaml")
    parser.add_argument("--models", nargs="+", default=["rigid", "two_mass"], choices=["rigid", "two_mass"])
    parser.add_argument("--epsilons", nargs="+", type=float, default=None, help="override selection.epsilons")
    parser.add_argument("--weighted", nargs="+", choices=["on", "off"], default=None,
                        help="override selection.weighted")
    parser.add_argument("--stage2-iterations", type=int, default=None)
    parser.add_argument("--stage3-iterations", type=int, default=None)
    parser.add_argument("--device", default=None, help="torch device of the output-error fit (default: config)")
    parser.add_argument("--skip-fg1", action="store_true", help="fit even if FG1 fails (says so)")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    cfg = load_fair_config(args.config, _REPO)
    started = time.perf_counter()
    rigid_model = idn.build_model(AssetRegistry.for_repository(_REPO).load(cfg.asset))
    train, info = load_fair(cfg.dataset, ("train",))
    val, _ = load_fair(cfg.dataset, ("val",))
    contract = info["contract"]
    step = float(np.median(np.diff(train[0].t)))
    priors = FairPriors.from_contract(contract, step)
    print(f"{cfg.platform}: {len(train)} train bags, {len(val)} val bags, step {step:g} s, "
          f"burn-in {priors.burn_in_s} s, windows {priors.window_s:g} s / stride {priors.stride_s:g} s")
    prior_checks = check_priors(cfg, priors)
    print(f"F4 priors = the LNN two-mass run's model_kwargs: {prior_checks}")
    scale = target_scale(train)
    counts = row_counts(train, val, priors)
    fg1_result = fg1(cfg, priors, counts, scale)
    print("FG1 coverage (LNN training code vs this identification):")
    for key, value in fg1_result["rows"].items():
        print(f"  {key:24s} LNN {value['lnn']:>9d}   ID {value['id']:>9d}   {'equal' if value['equal'] else 'DIFFERENT'}")
    print(f"  target scale s_j max relative difference {fg1_result['target_scale_rel_error']:.2e}")
    print(f"FG1 -> {'PASS' if fg1_result['pass'] else 'FAIL'}")
    if not fg1_result["pass"] and not args.skip_fg1:
        raise SystemExit("FG1 failed: the row sets are not the LNNs' (--skip-fg1 overrides)")

    t0 = time.perf_counter()
    train_rows = [BagRows.build(b, rigid_model, priors.sg_window, priors.sg_poly) for b in train]
    val_rows = [BagRows.build(b, rigid_model, priors.sg_window, priors.sg_poly) for b in val]
    print(f"regressors at the motor state: {time.perf_counter() - t0:.0f} s")
    selection = cfg.selection
    epsilons = args.epsilons or [float(e) for e in selection.get("epsilons", [1e-3, 1e-2, 1e-1])]
    weightings = ([w == "on" for w in args.weighted] if args.weighted else
                  [bool(w) for w in selection.get("weighted", [True, False])])
    tolerance = float(selection.get("rank_tolerance", 1e-6))
    summary: dict[str, Any] = {"platform": cfg.platform, "dataset": os.fspath(cfg.dataset), "priors": priors.describe(),
                               "prior_checks": prior_checks, "fg1": fg1_result, "target_scale": scale.tolist(),
                               "grid": {"epsilons": epsilons, "weighted": weightings}}

    print("\nID-rigid-fair (weighted LS on every train row; val RMSE over every val row):")
    rigid_fits = {}
    for eps in epsilons:
        for weighted in weightings:
            fit = fit_rigid(train_rows, val_rows, rigid_model, epsilon=eps, weighted=weighted, scale=scale,
                            tolerance=tolerance)
            rigid_fits[(eps, weighted)] = fit
            print(f"  eps {eps:7.0e} weighted {str(weighted):5s}: rank {fit.base.rank}, train {fit.train_rmse:.4f}, "
                  f"val {fit.val_rmse:.4f}")
    best_rigid = min(rigid_fits.values(), key=lambda f: f.val_rmse)
    mismatch = check_dynamics(best_rigid.model, rigid_model, val_rows[0])
    pd = non_pd_share(best_rigid.model, rigid_model, train_rows + val_rows)
    print(f"selected: eps {best_rigid.model.epsilon:g}, weighted {best_rigid.weighted}; val {best_rigid.val_rmse:.4f}; "
          f"rnea vs regressor {mismatch:.1e}; M_hat not positive definite at {100 * pd['non_pd_share']:.0f} % of "
          f"{pd['probes']} probed states (min eigenvalue {pd['min_eigenvalue']:.3g})")
    if mismatch > 1e-8:
        raise SystemExit("the model copy's rnea does not reproduce the identified regressor model")
    best_rigid.model.info.update(selection={f"{e:g}/{'w' if w else 'u'}": f.val_rmse for (e, w), f in rigid_fits.items()},
                                 dynamics_check=mismatch, base_rank=best_rigid.base.rank, inertia_pd=pd)
    summary["rigid"] = {"grid": [{"epsilon": e, "weighted": w, "rank": f.base.rank, "train_rmse": f.train_rmse,
                                  "val_rmse": f.val_rmse} for (e, w), f in rigid_fits.items()],
                        "selected": {"epsilon": best_rigid.model.epsilon, "weighted": best_rigid.weighted,
                                     "val_rmse": best_rigid.val_rmse}, "inertia_pd": pd}

    if "two_mass" in args.models:
        settings = OutputErrorSettings.from_mapping(cfg.output_error)
        overrides = {k: v for k, v in (("stage2_iterations", args.stage2_iterations),
                                       ("stage3_iterations", args.stage3_iterations), ("device", args.device))
                     if v is not None}
        settings = replace(settings, **overrides)
        print(f"\nID-2M-fair (output error, {settings}):")
        t1 = time.perf_counter()
        mass = [mass_regressor(rigid_model, rows.theta) for rows in train_rows]
        print(f"mass regressors of the train bags: {time.perf_counter() - t1:.0f} s")
        two_mass_fits = {}
        for (eps, weighted), rigid in rigid_fits.items():
            print(f" eps {eps:g}, weighted {weighted}:")
            t1 = time.perf_counter()
            fit = fit_two_mass(rigid, train_rows, val_rows, rigid_model, priors, scale, settings=settings,
                               mass_regressors=mass)
            mismatch = check_dynamics(fit.model, rigid_model, val_rows[0])
            if mismatch > 1e-8:
                raise SystemExit(f"rnea/crba of the consistent inertials differ from the regressor ({mismatch:.1e})")
            two_mass_fits[(eps, weighted)] = fit
            print(f"  -> val {fit.val_rmse:.4f} (stage {fit.model.info['selected_stage']}, iteration "
                  f"{fit.model.info['selected_iteration']}), K {np.round(fit.model.stiffness).astype(int).tolist()}, "
                  f"zeta {np.round(fit.model.zeta, 3).tolist()}, non-PD M_hat samples "
                  f"{fit.model.info['non_pd_train_val']} [{time.perf_counter() - t1:.0f} s]")
        best = min(two_mass_fits.values(), key=lambda f: f.val_rmse)
        best.model.info.update(selection={f"{e:g}/{'w' if w else 'u'}": f.val_rmse
                                          for (e, w), f in two_mass_fits.items()})
        best.stage1.info.update(epsilon_weighting="the selected ID-2M-fair's (its stage 1)")
        print(f"selected: eps {best.model.epsilon:g}, weighted {best.model.info['weighted']}; val {best.val_rmse:.4f}")
        summary["two_mass"] = {
            "settings": vars(settings),
            "grid": [{"epsilon": e, "weighted": w, "val_rmse": f.val_rmse, "cost": f.cost,
                      "stage1_val_rmse": f.stage1.info["val_rmse"],
                      "stage": f.model.info["selected_stage"], "iteration": f.model.info["selected_iteration"],
                      "stiffness": f.model.stiffness.tolist(), "zeta": f.model.zeta.tolist(), "history": f.history}
                     for (e, w), f in two_mass_fits.items()],
            "selected": {"epsilon": best.model.epsilon, "weighted": best.model.info["weighted"],
                         "val_rmse": best.val_rmse}, "counts": best.counts}
    summary["wall_s"] = time.perf_counter() - started
    if args.no_save:
        print("\n(--no-save: nothing written)")
        return
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    best_rigid.model.save(cfg.output_dir / "id_rigid_fair.json")
    if "two_mass" in args.models:
        best.model.save(cfg.output_dir / "id_2m_fair.json")
        best.stage1.save(cfg.output_dir / "id_rigid_fair_consistent.json")
    path = cfg.output_dir / ("fit_summary.json" if "two_mass" in args.models else "fit_summary_rigid.json")
    path.write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    print(f"\nwrote {cfg.output_dir}/id_rigid_fair.json" + (", id_2m_fair.json, id_rigid_fair_consistent.json"
                                                             if "two_mass" in args.models else "")
          + f", {path.name} ({summary['wall_s']:.0f} s)")


if __name__ == "__main__":
    main()
