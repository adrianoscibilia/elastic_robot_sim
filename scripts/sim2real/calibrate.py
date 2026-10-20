#!/usr/bin/env python3
"""Run one sim2real calibration stage (S2R_01 T-4.5; R0 in pass P-3).

Every setting comes from the stage config (``config/sim2real/iiwa_s*.yaml``);
the flags only override the optimizer for comparisons. The run folder is
``<config output>/<run id>/`` (default run id: the config's file stem, or
``<stem>_<method>_s<seed>`` when ``--method``/``--seed`` override it). The
working overlay ``working.yaml`` is rewritten at every val improvement;
nothing is published.

Examples
--------
S0, then S1 (S1 reads S0's best.yaml as its prior)::

    python scripts/sim2real/calibrate.py --config config/sim2real/iiwa_s0_static.yaml
    python scripts/sim2real/calibrate.py --config config/sim2real/iiwa_s1_rigid.yaml

S1 with CMA at the comparison budget, seed 2::

    python scripts/sim2real/calibrate.py --config config/sim2real/iiwa_s1_rigid.yaml --method cma --seed 2 --budget 300

Look before running (prints theta0's metrics and identifiability, writes nothing)::

    python scripts/sim2real/calibrate.py --config config/sim2real/iiwa_s1_rigid.yaml --dry-run

Rebuild a report from its folder::

    python scripts/sim2real/calibrate.py --report-only data/sim2real/kuka_lbr_iiwa_14_r820/iiwa_s1_rigid

S1 with S2R_06's freeze-then-refit instead of the A-12' label (the owner's
alternative, S2R_08 §3)::

    python scripts/sim2real/calibrate.py --config config/sim2real/iiwa_s1_rigid.yaml --mode freeze_refit \
        --run-id iiwa_s1_rigid_freeze_refit

P4a-4, a comparator on S1's calibrated set (its calibrated parameters
free, the rest frozen at its fitted values)::

    python scripts/sim2real/calibrate.py --config config/sim2real/iiwa_s1_rigid.yaml --method cma --seed 1 \
        --budget 300 --free-from data/sim2real/kuka_lbr_iiwa_14_r820/iiwa_s1_rigid
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.fspath(_REPO / "src"))

from elastic_sim.sim2real.config import OPTIMIZER_METHODS, load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, help="stage config")
    parser.add_argument("--method", choices=OPTIMIZER_METHODS, help="override optimizer.method")
    parser.add_argument("--seed", type=int, help="override optimizer.seed")
    parser.add_argument("--budget", type=int, help="override optimizer.budget (LS: Jacobian evaluations)")
    parser.add_argument("--prior-weight-factor", type=float,
                        help="override loss.prior_weight_factor (sensitivity runs; S2R_05 F-1)")
    parser.add_argument("--free-from", type=Path, metavar="RUN_DIR",
                        help="free set = that run's calibrated parameters (best.yaml), the rest frozen at its values; "
                             "freeze-then-refit off (S2R_08 P4a-4)")
    parser.add_argument("--mode", choices=("label", "freeze_refit"),
                        help="override identifiability.mode (S2R_08 A-12' label | S2R_06 A-12 freeze_refit)")
    parser.add_argument("--run-id", help="run folder name under the config's output")
    parser.add_argument("--resume", action="store_true", help="continue a run folder from its history.jsonl")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print theta0's metrics and identifiability only")
    parser.add_argument("--report-only", type=Path, metavar="RUN_DIR", help="rebuild report.md from a run folder")
    args = parser.parse_args(argv)

    if args.report_only is not None:
        from elastic_sim.sim2real.calibration import report_only

        print(report_only(args.report_only, figures=not args.no_figures))
        return 0
    if args.config is None:
        parser.error("--config is required (or --report-only)")
    config = load_config(args.config)
    if args.prior_weight_factor is not None:
        import dataclasses

        config = dataclasses.replace(config, loss=dataclasses.replace(
            config.loss, prior_weight=None, prior_weight_factor=args.prior_weight_factor))
        resolved = dict(config.resolved)
        resolved["loss"] = {**dict(resolved.get("loss") or {}), "prior_weight": "auto",
                            "prior_weight_factor": args.prior_weight_factor}
        config = dataclasses.replace(config, resolved=resolved)
    if args.dry_run:
        return dry_run(config)
    from elastic_sim.sim2real.calibration import calibrate

    run_id = args.run_id
    if run_id is None and (args.method or args.seed is not None or args.free_from is not None):
        method = args.method or config.optimizer.method
        seed = config.optimizer.seed if args.seed is None else args.seed
        run_id = f"{config.source.stem}{'_reduced' if args.free_from else ''}_{method}_s{seed}"
    folder = calibrate(config, method=args.method, seed=args.seed, budget=args.budget, run_id=run_id,
                       resume=args.resume, figures=not args.no_figures, verbose=args.verbose,
                       mode=args.mode, reduced_from=args.free_from,
                       command=" ".join(shlex.quote(a) for a in [Path(sys.argv[0]).name, *sys.argv[1:]]))
    print(f"run folder: {folder}")
    print((folder / "report.md").read_text(encoding="utf-8").split("## (b)")[0])
    return 0


def dry_run(config) -> int:
    from elastic_sim.sim2real import identifiability as ida
    from elastic_sim.sim2real.calibration import analysed_names, evaluate_theta, identifiability_at
    from elastic_sim.sim2real.parameters import Registry
    from elastic_sim.sim2real.plant import PlantBase
    from elastic_sim.sim2real.problem import CalibrationProblem, load_runs

    base = PlantBase.from_config(config)
    registry = Registry.from_config(config, base)
    runs = load_runs(config)
    problem = CalibrationProblem.from_config(config, registry, base, runs)
    theta0 = registry.values()
    print(f"{config.source.name}: {registry.n_free} free parameters, rows per split "
          f"{ {k: e.n_rows for k, e in problem.engines.items()} }, lambda {problem.loss.prior_weight:.3g}")
    block, _ = evaluate_theta(problem, theta0, {k: r.contract for k, r in runs.items()})
    for split in ("train", "val", "test"):
        torque = (block.get(split) or {}).get("torque")
        if torque:
            print(f"  {split:5s} RMSE {torque['rmse_median_over_bags']:.3f} N m, per joint "
                  + " ".join(f"{v:.2f}" for v in torque["per_joint_rmse"]))
    for run, slopes in (block["holds"]["hold_slopes"] or {}).items():
        if isinstance(slopes, dict) and "slope" in slopes:
            print(f"  hold slopes {run}: " + ", ".join(f"J{int(j) + 1} {v:.4f}" for j, v in slopes["slope"].items()))
    ident = identifiability_at(problem, theta0, analysed_names(registry, base))
    print(ida.to_markdown(ident, title="Identifiability at theta0"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
