#!/usr/bin/env python3
"""Cross-check the A-pc constrained least squares with a second, independent solver.

`REFACTOR_SPECS/model_based_baseline/06_variant_A_consistent.md` Sec 1.2:
the SDP optimum of ``zollo_identify.py`` (cvxpy) against the same cost and
feasible set solved by the log-Cholesky parametrisation (Rucker & Wensing
2022) shifted by ``eps I`` and Levenberg-Marquardt, from the URDF
(:func:`elastic_sim.zollo.consistent.solve_log_cholesky`).  Reads the
identification JSON and its triangular factor (``.factor.npz``), then

* compares the two optima's cost (target: 1e-6 relative);
* scores both on the simulated test split -- ID-rigid and the bounded
  ID-elastic -- and reports the RMSE between their predictions (target:
  1e-3 N m) on the harness's rows.

Writes ``<output_dir>/consistency_check_<identification stem>.json``;
``--no-save`` prints only.

Example
-------
::

    uv run python scripts/zollo_consistency_check.py --config config/zollo/ur10.yaml
"""

from __future__ import annotations

import argparse
import copy
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
from elastic_sim.zollo.config import load_platform
from elastic_sim.zollo.consistent import body_eigenvalues, solve_log_cholesky
from elastic_sim.zollo.data import load_sim, sidecar
from elastic_sim.zollo.identify import load_factor, load_identification
from elastic_sim.zollo.predictors import ElasticModel, ElasticSettings, LinkModel, TransmissionBounds, consumer_inputs, kept_rows

COST_TOLERANCE = 1e-6          # relative
PREDICTION_TOLERANCE = 1e-3    # N m RMSE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config/zollo/ur10.yaml")
    parser.add_argument("--identification", default=None,
                        help="an identification JSON (default <output_dir>/identification_Apc_pooled.json)")
    parser.add_argument("--max-nfev", type=int, default=20000)
    parser.add_argument("--start", choices=("urdf", "sdp"), default="urdf",
                        help="LM start: the URDF (independent), or beside the SDP optimum (0.999 SDP + 0.001 URDF "
                             "inertials, strictly inside the feasible set): can LM still lower the cost?")
    parser.add_argument("--max-bags", type=int, default=None, help="first N sim test bags (quick look)")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    cfg = load_platform(args.config, _REPO)
    id_path = Path(args.identification) if args.identification else cfg.output_dir / "identification_Apc_pooled.json"
    sdp = load_identification(id_path)
    if sdp.solution.full is None:
        raise SystemExit(f"{id_path.name} is not a constrained solve")
    acc = load_factor(id_path.with_suffix(".factor.npz"))
    rigid_model = idn.build_model(AssetRegistry.for_repository(_REPO).load(sdp.meta["asset"]))
    urdf = idn.standard_parameters(*rigid_model[:2])
    epsilon = sdp.config.consistency_epsilon
    started = time.perf_counter()
    start = None
    if args.start == "sdp":
        start = sdp.full.copy()
        inertial = sdp.layout.inertial_slice()
        start[inertial] = 0.999 * start[inertial] + 0.001 * urdf
    fit = solve_log_cholesky(acc, sdp.base, sdp.layout, urdf, epsilon=epsilon, max_nfev=args.max_nfev, start=start)
    c_sdp, c_lm = sdp.meta["consistency"]["cost"], fit.cost
    cost_rel = abs(c_lm - c_sdp) / c_sdp
    increase_rel = abs(fit.increase - (c_sdp - fit.cost_unconstrained)) / max(abs(c_sdp - fit.cost_unconstrained), 1e-300)
    print(f"{cfg.platform}: {id_path.name}; eps {epsilon:g}")
    print(f"  SDP ({sdp.meta['consistency']['solver']}): cost {c_sdp:.12g}")
    print(f"  LM  (log-Cholesky): cost {c_lm:.12g}; {fit.report['status']} after {fit.report['nfev']} evaluations "
          f"({time.perf_counter() - started:.0f} s)")
    print(f"  unconstrained cost {fit.cost_unconstrained:.12g}")
    print(f"  cost agreement {cost_rel:.3g} relative (target {COST_TOLERANCE:g}) -> "
          f"{'PASS' if cost_rel <= COST_TOLERANCE else 'FAIL'}; the increases over unconstrained agree to "
          f"{increase_rel:.3g} relative")
    print(f"  LM active bodies {fit.report['active_bodies']}, friction at 0 {fit.report['active_friction']}; "
          f"masses {np.round(fit.report['masses'], 4).tolist()}")
    print(f"  SDP active bodies {sdp.meta['consistency']['active_bodies']}; masses "
          f"{np.round(sdp.meta['consistency']['masses'], 4).tolist()}")
    base_sdp, base_lm = sdp.base.base_of(sdp.full), sdp.base.base_of(fit.full)
    base_diff = np.abs(base_lm - base_sdp) / np.maximum(np.abs(base_sdp), 1e-6 * np.abs(base_sdp).max())
    print(f"  base vectors: max relative difference {base_diff.max():.3g} (at {sdp.layout.names[sdp.base.independent[int(np.argmax(base_diff))]]})")

    lm = copy.copy(sdp)
    lm.solution = copy.copy(sdp.solution)
    lm.solution.full = fit.full
    ev = cfg.evaluation
    train_contract = sidecar(cfg.dataset, "contract")
    priors = train_contract["transmission_priors"]
    head_s = float(ev.get("burn_in_s") or priors["burn_in_s"])
    settings = ElasticSettings.from_mapping(ev.get("elastic"))
    bounds = TransmissionBounds.from_two_mass(train_contract, ev.get("lnn_repo") and cfg.resolve(ev["lnn_repo"]))
    models = {name: (LinkModel.from_identification(r, rigid_model), ElasticModel.bounded(r, rigid_model, bounds, settings))
              for name, r in (("sdp", sdp), ("lm", lm))}
    bags, info = load_sim(cfg.dataset, splits=(ev.get("sim_split", "test"),), truth=False)
    if args.max_bags:
        bags = bags[:args.max_bags]
    diffs = {"id_rigid": [], "id_elastic_bounded": []}
    min_eig = {"sdp": np.inf, "lm": np.inf}
    diverged = {"sdp": [], "lm": []}
    for bag in bags:
        theta, dtheta, ddtheta = consumer_inputs(bag, info["contract"])
        keep = kept_rows(bag.t, head_s, 1e-9)
        probes = [theta[i] for i in np.linspace(0, len(theta) - 1, 20).astype(int)]
        out = {}
        for name, (link, bounded) in models.items():
            min_eig[name] = min(min_eig[name], min(np.linalg.eigvalsh(link.mass(q))[0] for q in probes))
            run = bounded.with_damping_floor(bounds.zeta_min, probes).run(theta, dtheta, bag.step)
            if run.diverged_at is not None:
                diverged[name].append(bag.name)
            out[name] = {"id_rigid": link.torque(theta, dtheta, ddtheta), "id_elastic_bounded": run.tau}
        for p in diffs:
            diffs[p].append((out["sdp"][p] - out["lm"][p])[keep])
        print(f"  [{bag.name}] done", flush=True)
    rmse = {p: float(np.sqrt(np.nanmean(np.vstack(v) ** 2))) for p, v in diffs.items()}
    for p, v in rmse.items():
        print(f"  {p}: RMSE between SDP and LM predictions {v:.3g} N m (target {PREDICTION_TOLERANCE:g}) -> "
              f"{'PASS' if v <= PREDICTION_TOLERANCE else 'FAIL'}")
    print(f"  min eig M_hat at the probes: SDP {min_eig['sdp']:.4g}, LM {min_eig['lm']:.4g}; diverged SDP "
          f"{diverged['sdp']}, LM {diverged['lm']}")
    record = {"identification": str(id_path), "epsilon": epsilon, "start": args.start, "max_nfev": args.max_nfev, "cost_sdp": c_sdp, "cost_lm": c_lm,
              "cost_unconstrained": fit.cost_unconstrained, "cost_relative_difference": cost_rel,
              "increase_relative_difference": increase_rel, "cost_pass": bool(cost_rel <= COST_TOLERANCE),
              "base_max_relative_difference": float(base_diff.max()), "prediction_rmse": rmse,
              "prediction_pass": {p: bool(v <= PREDICTION_TOLERANCE) for p, v in rmse.items()},
              "min_eig_probes": min_eig, "diverged": diverged, "bags": [b.name for b in bags],
              "lm": {k: v for k, v in fit.report.items()}, "lm_full": fit.full.tolist(),
              "lm_min_eig_J": float(body_eigenvalues(fit.full[sdp.layout.inertial_slice()]).min())}
    if args.no_save:
        print("\n(--no-save: nothing written)")
        return
    path = cfg.output_dir / f"consistency_check_{id_path.stem}{'' if args.start == 'urdf' else '_from_sdp'}.json"
    path.write_text(json.dumps(record, indent=1, default=float), encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
