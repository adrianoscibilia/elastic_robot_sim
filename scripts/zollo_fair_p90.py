#!/usr/bin/env python3
"""The percentile-loss (``percentile_mse@90``) variants of the fair baselines (`03_fair_comparison_plan.md` F6, F8).

Run after ``zollo_fair_fit.py``: uses its selected ``eps`` and weighting
(``id_rigid_fair.json``, ``id_2m_fair.json``) and ID-2M-fair's stage 1
(``id_rigid_fair_consistent.json``), then for every seed of
``percentile.seeds`` trains with Adam as the LNNs (``elastic_sim.zollo.fair_adam``)
and deploys the best-validation seed.  Writes ``id_rigid_fair_p90[_s<seed>].json``,
``id_2m_fair_p90[_s<seed>].json`` and ``p90_summary.json``; ``--no-save`` prints only.

Example
-------
::

    uv run python scripts/zollo_fair_p90.py --config config/zollo/fair/ur10.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(_REPO / "src"))

from elastic_sim import identification as idn
from elastic_sim.assets import AssetRegistry
from elastic_sim.zollo.config import load_fair_config
from elastic_sim.zollo.fair import BagRows, FairModel, FairPriors, fit_rigid, load_fair, mass_regressor, target_scale
from elastic_sim.zollo.fair_adam import AdamSettings, fit_rigid_p90, fit_two_mass_p90


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config/zollo/fair/ur10.yaml")
    parser.add_argument("--models", nargs="+", default=["rigid", "two_mass"], choices=["rigid", "two_mass"])
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="override percentile.epochs (quick look)")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    cfg = load_fair_config(args.config, _REPO)
    p90 = cfg.percentile
    seeds = args.seeds or [int(s) for s in p90.get("seeds", [0, 1, 2])]
    common = {"epochs": int(args.epochs or p90.get("epochs", 300)), "patience": int(p90.get("patience", 30)),
              "loss_percentile": float(p90.get("loss_percentile", 90.0)),
              "device": str(cfg.output_error.get("device", "auto"))}
    rigid_model = idn.build_model(AssetRegistry.for_repository(_REPO).load(cfg.asset))
    train, info = load_fair(cfg.dataset, ("train",))
    val, _ = load_fair(cfg.dataset, ("val",))
    priors = FairPriors.from_contract(info["contract"], float(np.median(np.diff(train[0].t))))
    scale = target_scale(train)
    train_rows = [BagRows.build(b, rigid_model, priors.sg_window, priors.sg_poly) for b in train]
    val_rows = [BagRows.build(b, rigid_model, priors.sg_window, priors.sg_poly) for b in val]
    summary: dict = {"platform": cfg.platform, "seeds": seeds, "settings": {**common, **p90}}
    saved: list[tuple[str, FairModel]] = []

    if "rigid" in args.models:
        selected = FairModel.load(cfg.output_dir / "id_rigid_fair.json")
        rigid = fit_rigid(train_rows, val_rows, rigid_model, epsilon=selected.epsilon, weighted=True, scale=scale,
                          tolerance=float(cfg.selection.get("rank_tolerance", 1e-6)))
        settings = AdamSettings(learning_rate=float(p90["residual"]["learning_rate"]),
                                batch_size=int(p90["residual"]["batch_size"]), **common)
        print(f"ID-rigid-fair P90 (eps {selected.epsilon:g}; {settings})")
        runs = {}
        for seed in seeds:
            t0 = time.perf_counter()
            model, record = fit_rigid_p90(rigid, train_rows, val_rows, scale, settings, seed)
            runs[seed] = (model, record)
            print(f" seed {seed}: val {record['val_rmse']:.4f} at epoch {record['best_epoch']} of "
                  f"{record['epochs_run']} [{time.perf_counter() - t0:.0f} s]")
        best_seed = min(runs, key=lambda s: runs[s][1]["val_rmse"])
        summary["rigid"] = {"deployed_seed": best_seed, "runs": {s: r for s, (_, r) in runs.items()}}
        saved += [(f"id_rigid_fair_p90_s{s}", m) for s, (m, _) in runs.items()]
        saved.append(("id_rigid_fair_p90", runs[best_seed][0]))

    if "two_mass" in args.models:
        selected = FairModel.load(cfg.output_dir / "id_2m_fair.json")
        stage1 = FairModel.load(cfg.output_dir / "id_rigid_fair_consistent.json")
        if stage1.epsilon != selected.epsilon:
            raise SystemExit("id_rigid_fair_consistent.json is not the selected ID-2M-fair's stage 1")
        settings = AdamSettings(learning_rate=float(p90["two_mass"]["learning_rate"]),
                                batch_size=int(p90["two_mass"]["batch_size"]), **common)
        t0 = time.perf_counter()
        mass = [mass_regressor(rigid_model, rows.theta) for rows in train_rows]
        print(f"\nID-2M-fair P90 (eps {selected.epsilon:g}; {settings}); mass regressors "
              f"{time.perf_counter() - t0:.0f} s")
        runs = {}
        for seed in seeds:
            t0 = time.perf_counter()
            model, record = fit_two_mass_p90(stage1, train_rows, val_rows, rigid_model, priors, scale, mass, settings,
                                             seed, cfg.resolve(cfg.lnn_pipeline["repo"]))
            runs[seed] = (model, record)
            print(f" seed {seed}: val {model.info['val_rmse']:.4f}, K {np.round(model.stiffness).astype(int).tolist()}, "
                  f"zeta {np.round(model.zeta, 3).tolist()} [{time.perf_counter() - t0:.0f} s]")
        best_seed = min(runs, key=lambda s: runs[s][0].info["val_rmse"])
        summary["two_mass"] = {"deployed_seed": best_seed, "runs": {s: r for s, (_, r) in runs.items()}}
        saved += [(f"id_2m_fair_p90_s{s}", m) for s, (m, _) in runs.items()]
        saved.append(("id_2m_fair_p90", runs[best_seed][0]))

    if args.no_save:
        print("\n(--no-save: nothing written)")
        return
    for name, model in saved:
        model.save(cfg.output_dir / f"{name}.json")
    path = cfg.output_dir / f"p90_summary{'' if len(args.models) == 2 else '_' + args.models[0]}.json"
    path.write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    print(f"\nwrote {', '.join(n + '.json' for n, _ in saved)} and {path.name} in {cfg.output_dir}")


if __name__ == "__main__":
    main()
