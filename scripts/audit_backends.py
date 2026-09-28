#!/usr/bin/env python3
"""MuJoCo vs Newton audit on a small matched build (`R5_10` T-4).

Newton left production in round 5 (`R5_10` T-3.2): it is 10-15x slower than
MuJoCo and on the elastic chain it falls back to ``SolverMuJoCo`` anyway, so
it adds no independent evidence to a training file.  What it still gives is a
cross-check of the simulator, and this is where that lives: 3 robots (plus
the rigid reference, the only genuinely independent pair), 1 trajectory, both
backends, compared on the **clean** columns.  It is not part of any launch.

    python scripts/audit_backends.py --config config/identification/ur10_table_round5.yaml

Writes ``<out>.backend_comparison.csv`` and ``<out>.summary.json``; exits 1
when a pair fails, so it can gate a release by hand.
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
sys.path.insert(0, os.fspath(_REPO / "scripts"))

from diagnose_controller_modes import _load_asset, _shrink
from elastic_sim.backend_comparison import compare_backends, format_report, summarize
from elastic_sim.dataset import load_config, generate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config/identification/ur10_table_round5.yaml")
    parser.add_argument("--robots", type=int, default=3)
    parser.add_argument("--trajectories", type=int, default=1)
    parser.add_argument("--mode", default=None, help="controller mode (default: the config's own)")
    parser.add_argument("--out", default="reports/audit_backends/audit",
                        help="output prefix for the comparison CSV and summary JSON")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    asset = _load_asset(config.asset)
    config = _shrink(config, trajectories=args.trajectories, robots=args.robots, backends=("mujoco", "newton"))
    # The comparison needs the clean columns; they are debug columns, so an
    # audit may switch them on whatever the config says.
    config = replace(config, signals=replace(config.signals, clean_columns=True))
    if args.mode:
        config = replace(config, controller=replace(config.controller, mode=args.mode))

    frame, manifest, _ = generate(config, asset, verbose=True, jobs=1)
    report = compare_backends(frame, manifest["records"], config.comparison)
    print("\n" + format_report(report, config.comparison))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out.with_name(out.name + ".backend_comparison.csv"), index=False)
    summary = {
        "config": str(args.config), "mode": config.controller.mode, "robots": args.robots,
        "trajectories": args.trajectories, "columns": sorted(set(report.get("columns", []))),
        "unstable_bags": manifest["unstable_bags"],
        **summarize(report, config.comparison),
    }
    out.with_name(out.name + ".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out}.backend_comparison.csv and {out}.summary.json")
    return 0 if not summary.get("failed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
