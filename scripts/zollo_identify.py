#!/usr/bin/env python3
"""Identify the Zollo et al. (2015) elastic-joint model on a simulated training split.

Reads ``config/zollo/<platform>.yaml`` (CLI flags override it), builds
Zollo's rows from the training bags, selects the base parameters, solves
for them and prints the recovery against the simulator's ground truth
(gate G2 of `REFACTOR_SPECS/model_based_baseline/01_zollo_plan.md`).

Writes ``<output_dir>/identification_<variant>_<transmission>[_<tag>].json``
(variant A-pr, ``--variant A --transmission per_robot_motor``, as
``identification_Apr_pooled[_<tag>].json``; variant A-pc,
``--variant A --transmission per_robot_transmission``, as
``identification_Apc_pooled[_<tag>].json``)
(the parameter set ``zollo_evaluate.py`` reads), the recovery table as
CSV beside it and, for a constrained solve, the triangular factor
(``.factor.npz``, for ``zollo_consistency_check.py``); ``--no-save`` prints only.

Examples
--------
The predictors' parameter set (variant B, pooled)::

    uv run python scripts/zollo_identify.py --config config/zollo/ur10.yaml

The recovery check, every robot its own transmission, Zollo-faithful rows::

    uv run python scripts/zollo_identify.py --config config/zollo/ur10.yaml \\
        --variant A --transmission per_robot

Variant A-pr (`04_variant_A_elastic.md`): Zollo's rows, rotor inertia and
motor friction per robot, link model, K and D pooled; and its D-free ablation::

    uv run python scripts/zollo_identify.py --config config/zollo/ur10.yaml \\
        --variant A --transmission per_robot_motor
    uv run python scripts/zollo_identify.py --config config/zollo/ur10.yaml \\
        --variant A --transmission per_robot_motor --no-damping --tag noD

Variant A-pc (`06_variant_A_consistent.md`): every per-robot draw per robot,
link inertials and link friction pooled, LS constrained to physically
consistent inertials; and its ablations (unconstrained; A-pr groups +
constraint; no D column)::

    uv run python scripts/zollo_identify.py --config config/zollo/ur10.yaml \\
        --variant A --transmission per_robot_transmission
    uv run python scripts/zollo_identify.py --config config/zollo/ur10.yaml \\
        --variant A --transmission per_robot_transmission --unconstrained
    uv run python scripts/zollo_identify.py --config config/zollo/ur10.yaml \\
        --variant A --transmission per_robot_motor --consistent
    uv run python scripts/zollo_identify.py --config config/zollo/ur10.yaml \\
        --variant A --transmission per_robot_transmission --no-damping --tag noD

A quick look on three robots, nothing written::

    uv run python scripts/zollo_identify.py --config config/zollo/iiwa_drive.yaml --robots 3 --no-save
"""

from __future__ import annotations

import argparse
import csv
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
from elastic_sim.zollo.data import load_sim
from elastic_sim.zollo.identify import identify, recovery_table, with_overrides

INERTIAL_GROUPS = ("mass", "first_moment", "inertia")
PER_ROBOT_SUMMARY = ("stiffness", "damping", "rotor", "motor_viscous", "motor_coulomb")


def identification_path(output_dir: Path, label: str, tag: str | None) -> Path:
    return output_dir / f"identification_{label}{'_' + tag if tag else ''}.json"


def _print_recovery(rows: list[dict], verbose: bool) -> None:
    print(f"\n{'parameter':26s} {'category':13s} {'identified':>12s} {'std %':>8s} {'true':>12s} {'err %':>8s}"
          f" {'true range':>21s} {'effective':>11s} {'err eff %':>9s}")
    for r in rows:
        if r["group"] in INERTIAL_GROUPS and not verbose:
            continue
        span = (f"[{r['true_min']:.4g}, {r['true_max']:.4g}]" if "true_min" in r else "")
        eff = f"{r['true_effective']:.4g}" if "true_effective" in r else ""
        eff_err = f"{r['error_effective_pct']:+.1f}" if "error_effective_pct" in r else ""
        print(f"{r['column']:26s} {r['category']:13s} {r['identified']:+12.5g} {r['std_pct']:8.2f} "
              f"{r['true']:+12.5g} {r['error_pct']:+8.1f} {span:>21s} {eff:>11s} {eff_err:>9s}")
    inertial = [r for r in rows if r["group"] in INERTIAL_GROUPS and np.isfinite(r["true"])]
    if inertial:
        errors = np.asarray([abs(r["identified"] - r["true"]) for r in inertial])
        scale = max(abs(r["true"]) for r in inertial)
        print(f"\ninertial base parameters: {len(inertial)}; |error| median {np.median(errors):.3g}, "
              f"max {errors.max():.3g} (largest |true| {scale:.3g}); --verbose lists them")


def _print_per_robot(rows: list[dict]) -> list[dict]:
    """Per group and joint, median / max |error| % over robots: K, D against the effective spring, the rest against truth."""
    out = []
    for group in PER_ROBOT_SUMMARY:
        mine = [r for r in rows if r["group"] == group and r["robot"]]
        if not mine:
            continue
        print(f"\n{group} per robot, |error| % median / max over {len({r['robot'] for r in mine})} robots "
              f"({'effective linear spring' if group in ('stiffness', 'damping') else 'truth'}):")
        for joint in sorted({r["joint"] for r in mine}):
            key = "error_effective_pct" if group in ("stiffness", "damping") else "error_pct"
            errors = np.abs([r[key] for r in mine if r["joint"] == joint and np.isfinite(r.get(key, np.nan))])
            values = [r["identified"] for r in mine if r["joint"] == joint]
            if not len(errors):
                continue
            entry = {"group": group, "joint": joint, "median_abs_error_pct": float(np.median(errors)),
                     "max_abs_error_pct": float(errors.max()), "identified_min": float(np.min(values)),
                     "identified_median": float(np.median(values)), "identified_max": float(np.max(values))}
            out.append(entry)
            print(f"  joint {joint}: {entry['median_abs_error_pct']:7.1f} / {entry['max_abs_error_pct']:7.1f}   "
                  f"identified {entry['identified_min']:+.4g} .. {entry['identified_median']:+.4g} .. "
                  f"{entry['identified_max']:+.4g}")
    return out


def _mass_matrix_spectrum(result, rigid_model, samples: int, seed: int = 0) -> dict:
    """``min eig M_hat(q)`` over ``samples`` random states, uniform in the URDF joint limits (or +-pi)."""
    from elastic_sim.zollo.predictors import LinkModel

    link = LinkModel.from_identification(result, rigid_model) if result.config.pooled_predictor else None
    if link is None:
        return {}
    model = rigid_model[1]
    low = np.where(np.isfinite(model.lowerPositionLimit), model.lowerPositionLimit, -np.pi)
    high = np.where(np.isfinite(model.upperPositionLimit), model.upperPositionLimit, np.pi)
    low, high = np.maximum(low, -2 * np.pi), np.minimum(high, 2 * np.pi)
    rng = np.random.default_rng(seed)
    eig = np.asarray([np.linalg.eigvalsh(link.mass(rng.uniform(low, high))) for _ in range(samples)])
    diagonal = np.asarray([np.diag(link.mass(rng.uniform(low, high))) for _ in range(samples)])
    return {"samples": samples, "min_eig": float(eig[:, 0].min()), "median_min_eig": float(np.median(eig[:, 0])),
            "positive_definite": int((eig[:, 0] > 0).sum()),
            "joints_with_nonpositive_diagonal": [int(j + 1) for j in np.flatnonzero((diagonal <= 0).any(axis=0))]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config/zollo/ur10.yaml")
    parser.add_argument("--variant", choices=("A", "B"), default=None)
    parser.add_argument("--transmission", choices=("pooled", "per_robot", "per_robot_motor", "per_robot_transmission"),
                        default=None)
    consistency = parser.add_mutually_exclusive_group()
    consistency.add_argument("--consistent", dest="consistent", action="store_const", const=True, default=None,
                             help="constrain the LS to physically consistent inertials (C2; default for A-pc)")
    consistency.add_argument("--unconstrained", dest="consistent", action="store_const", const=False,
                             help="plain LS (A-pc ablation 1)")
    parser.add_argument("--epsilon", type=float, default=None, help="consistency_epsilon: J_i >= eps I")
    parser.add_argument("--solver", default=None, help="consistency_solver (cvxpy name: CLARABEL, SCS)")
    parser.add_argument("--random-states", type=int, default=50, help="random q for min eig M_hat (0 skips)")
    parser.add_argument("--no-damping", action="store_true", help="drop the D columns (Zollo's damping-free model)")
    parser.add_argument("--cutoff-hz", type=float, default=None)
    parser.add_argument("--decimate", type=int, default=None)
    parser.add_argument("--low-speed", type=float, default=None, help="low_speed_fraction (0 disables)")
    parser.add_argument("--difference", choices=("central", "forward"), default=None)
    parser.add_argument("--weighted", action="store_true", default=None)
    parser.add_argument("--ridge", type=float, default=None, help="ridge_to_nominal")
    parser.add_argument("--robots", type=int, default=None, help="use only the first N training robots (quick look)")
    parser.add_argument("--tag", default=None, help="suffix of the output file (ablations)")
    parser.add_argument("--verbose", action="store_true", help="list every inertial base parameter too")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    cfg = load_platform(args.config, _REPO)
    config = with_overrides(cfg.identification, variant=args.variant, transmission=args.transmission,
                            weighted=args.weighted, ridge_to_nominal=args.ridge, consistent=args.consistent,
                            consistency_epsilon=args.epsilon, consistency_solver=args.solver,
                            spring_damping=False if args.no_damping else None,
                            **{"preprocessing.cutoff_hz": args.cutoff_hz, "preprocessing.decimate": args.decimate,
                               "preprocessing.low_speed_fraction": args.low_speed,
                               "preprocessing.difference": args.difference})
    started = time.perf_counter()
    bags, info = load_sim(cfg.dataset, splits=cfg.splits)
    if args.robots is not None:
        keep = sorted({b.tier for b in bags if b.tier != "rigid"})[:args.robots] + ["rigid"]
        bags = [b for b in bags if b.tier in keep]
    asset = AssetRegistry.for_repository(_REPO).load(info["manifest"]["asset"])
    rigid_model = idn.build_model(asset)
    print(f"{cfg.platform}: {len(bags)} bags of {cfg.splits} from {cfg.dataset.name} "
          f"(privileged channels delayed {info['delay_samples']} sample(s)); asset {asset.name}")
    print(f"variant {config.variant}, transmission {config.transmission} (per robot: "
          f"{list(config.per_robot_groups) or 'none'}), D columns {config.spring_damping}, consistent inertials "
          f"{config.is_consistent}"
          f"{f' (eps {config.consistency_epsilon:g}, {config.consistency_solver})' if config.is_consistent else ''}"
          f", rigid tier "
          f"{'excluded' if config.variant == 'A' or not config.include_rigid_tier else 'link rows only'}, preprocessing "
          f"{config.preprocessing.describe()}")
    result = identify(bags, config, rigid_model, truth=info["truth"], verbose=True)
    result.meta.update({"platform": cfg.platform, "dataset": str(cfg.dataset), "asset": asset.name,
                        "delay_samples": info["delay_samples"], "step": info["step"],
                        "contract_differentiation": info["contract"].get("differentiation"),
                        "wall_time_s": time.perf_counter() - started})
    base = result.base
    print(f"\nrows {result.stats.rows} from {result.stats.samples} samples of {result.stats.bags} bags; "
          f"friction rows dropped at low speed per joint {result.stats.rows_dropped_low_speed}")
    print(f"base parameters {base.rank} of {result.layout.size}; unidentifiable {len(base.zero)}; "
          f"condition number scaled {base.condition_scaled:.4g}, raw {base.condition_raw:.4g}")
    names = result.layout.names
    for group in ("stiffness", "damping"):
        if result.layout.has(group):
            print(f"{group} categories: " + ", ".join(
                f"{names[i]} {base.categories[i]}" for i, c in enumerate(result.layout.columns) if c.group == group))
    print("residual RMS per equation: " + ", ".join(f"{k} {v:.3g}" for k, v in result.row_residual_rms.items()))
    if "consistency" in result.meta:
        c = result.meta["consistency"]
        print(f"\nconsistent LS: {c['solver']} {c['solver_version']} (cvxpy {c['cvxpy_version']}), status {c['status']}"
              f" after {c['solver_iterations']} iterations; attempts {[a['status'] for a in c['attempts']]}")
        print(f"  cost {c['cost']:.10g} against unconstrained {c['cost_unconstrained']:.10g}: increase "
              f"{c['cost_increase']:.6g} ({100 * c['cost_increase_relative']:.4g} %)")
        print(f"  eps {c['epsilon']:g}; active bodies {c['active_bodies']}; friction at 0 {c['active_friction']}")
        for b in c["bodies"]:
            print(f"  body {b['body']}: eig J {np.array2string(np.asarray(b['eig_J']), precision=4)}"
                  f"{'  ACTIVE' if b['active'] else ''}, dual norm {b.get('dual_norm', float('nan')):.3g}")
        print(f"  masses {np.round(c['masses'], 4).tolist()}; unidentifiable columns max |value| "
              f"{c['unidentifiable_max_abs']:.4g}")
    if args.random_states and config.pooled_predictor:
        result.meta["mass_matrix_random"] = _mass_matrix_spectrum(result, rigid_model, args.random_states)
        print(f"M_hat over {args.random_states} random q: {result.meta['mass_matrix_random']}")
    rows = recovery_table(result, info["truth"])
    _print_recovery(rows, args.verbose)
    result.meta["per_robot_recovery"] = _print_per_robot(rows)
    if args.no_save:
        print("\n(--no-save: nothing written)")
        return
    path = identification_path(cfg.output_dir, config.label, args.tag)
    result.save(path)
    with path.with_suffix(".recovery.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)
    if config.is_consistent or config.transmission == "per_robot_transmission":
        result.save_factor(path.with_suffix(".factor.npz"))
    print(f"\nwrote {path} and {path.with_suffix('.recovery.csv').name} ({time.perf_counter() - started:.0f} s)")


if __name__ == "__main__":
    main()
