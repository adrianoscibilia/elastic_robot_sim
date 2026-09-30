#!/usr/bin/env python3
"""Size a round-6 platform's excitation against its effort limits (`R6_02` P2-1, `R6_04` D-1/A-7).

Two sweeps on a preflight-sized build (the same robots, trajectories and
seeds at every point, the rigid reference included), each stopping at the
first point whose peak ``|tau_cmd| / effort`` exceeds ``--threshold`` (0.8,
the round-4 warning threshold):

* ``--sweep fraction`` (default): ``excitation.probe_acceleration_fraction``.
  With ``probe_budget: additive`` the probe is added on top of the main
  trajectory's full budget, so the peak grows with it; the sweep runs up to
  1.0 (probe acceleration = main budget).  With the split budget the peak
  *falls* as the fraction rises (R6_03 Sec 3.4) and the cap decides.
* ``--sweep acceleration``: ``excitation.max_acceleration`` together with
  the regime's upper bound (A-7, FMRR: up to 2.0 m/s^2, 4x the URDF's 0.5).

The chosen value is the largest point whose peak stays at or below the
threshold with no unstable bag; it is printed with the table and written to
``reports/round6/<sweep>_<stem>.json`` (``--out`` to override, ``--no-save``
to skip).

    python scripts/round6_probe_fraction.py --config config/identification/ur10_table_round6.yaml
    python scripts/round6_probe_fraction.py --config config/identification/fmrr_tecnobody_round6.yaml \\
        --sweep acceleration --values 0.5 0.75 1.0 1.5 2.0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(_REPO / "src"))
# One BLAS/OpenMP thread per worker: the build already runs one process per
# core, and threaded BLAS on top drove the load past 400 on 32 cores.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
sys.path.insert(0, os.fspath(_REPO / "scripts"))

import numpy as np

from diagnose_controller_modes import _load_asset, _shrink
from elastic_sim.dataset import default_jobs, generate, load_config


def with_value(config, sweep: str, value: float):
    """The config at one sweep point."""
    if sweep == "fraction":
        return replace(config, excitation=replace(config.excitation, probe_acceleration_fraction=float(value)))
    regime = config.regime
    if regime.enabled:
        regime = replace(regime, max_acceleration=(min(regime.max_acceleration[0], float(value)), float(value)))
    return replace(config, excitation=replace(config.excitation, max_acceleration=float(value)), regime=regime)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--sweep", choices=("fraction", "acceleration"), default="fraction")
    parser.add_argument("--values", type=float, nargs="+", default=None,
                        help="sweep points (default: 0.2-0.7 split, 0.25-1.0 additive; required for acceleration)")
    parser.add_argument("--fractions", type=float, nargs="+", default=None, help="alias of --values (fraction sweep)")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--robots", type=int, default=2)
    parser.add_argument("--trajectories", type=int, default=2)
    parser.add_argument("--jobs", type=int, default=None)
    parser.add_argument("--out", default=None, help="report path (default reports/round6/<sweep>_<stem>.json)")
    parser.add_argument("--no-save", action="store_true", help="print only; write no report file")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    base = load_config(args.config)
    asset = _load_asset(base.asset)
    base = _shrink(base, trajectories=args.trajectories, robots=args.robots, backends=tuple(base.backends))
    jobs = default_jobs(base.backends) if args.jobs is None else args.jobs
    values = args.values or args.fractions
    if values is None:
        if args.sweep == "acceleration":
            parser.error("--sweep acceleration needs --values")
        values = ([0.25, 0.5, 0.75, 1.0] if base.excitation.probe_budget == "additive"
                  else [0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    rows = []
    for value in sorted(values):
        config = with_value(base, args.sweep, value)
        _, manifest, _ = generate(config, asset, verbose=not args.quiet, jobs=jobs, trajectory_cache={})
        records = manifest["records"]
        peaks = [float(r["peak_torque_ratio"]) for r in records]
        worst = records[int(np.argmax(peaks))] if records else {}
        rows.append({"sweep": args.sweep, "value": float(value),
                     "peak_torque_ratio": max(peaks) if peaks else None,
                     "worst_bag": worst.get("bag"), "bags": len(records),
                     "unstable_bags": len(manifest.get("unstable_bags", [])),
                     "sag_max_fraction": max((r.get("sag_max_fraction") or 0.0) for r in records) if records else None})
        print(f"{args.sweep} {value:g}: peak |tau|/effort {rows[-1]['peak_torque_ratio']:.3f} "
              f"({rows[-1]['worst_bag']}), {rows[-1]['bags']} bags, {rows[-1]['unstable_bags']} unstable, "
              f"worst sag {rows[-1]['sag_max_fraction']:.3f}")
        if rows[-1]["peak_torque_ratio"] is not None and rows[-1]["peak_torque_ratio"] > args.threshold:
            break
    admissible = [r for r in rows if r["peak_torque_ratio"] is not None and r["peak_torque_ratio"] <= args.threshold
                  and r["unstable_bags"] == 0]
    chosen = max((r["value"] for r in admissible), default=None)
    print(f"\nchosen {args.sweep} for {base.asset}: {chosen} (threshold {args.threshold:g}, "
          f"swept {min(values):g}-{max(values):g})")
    if not args.no_save:
        out = (Path(args.out) if args.out else
               _REPO / "reports" / "round6" / f"{args.sweep}_{Path(base.output).stem}.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"config": args.config, "sweep": args.sweep,
                                   "probe_budget": base.excitation.probe_budget, "threshold": args.threshold,
                                   "rows": rows, "chosen": chosen, "robots": args.robots,
                                   "trajectories": args.trajectories}, indent=2), encoding="utf-8")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
