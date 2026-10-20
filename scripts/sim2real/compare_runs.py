#!/usr/bin/env python3
"""Rank finished calibration runs on the physical metrics (S2R_02 A-2).

Ranked by the test torque RMSE; the table also carries the hold slopes,
plausibility (largest prior distance, prior-determined count), the friction
signature, and sample efficiency (evaluations to reach 1 % of the run's
final best val loss) as a secondary column. Prints the table; ``--out``
writes it as markdown (+ JSON) as well.

Example::

    python scripts/sim2real/compare_runs.py data/sim2real/kuka_lbr_iiwa_14_r820/iiwa_s1_rigid*
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.fspath(_REPO / "src"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("folders", nargs="+", type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", type=Path, help="write <out>.md and <out>.json")
    args = parser.parse_args(argv)

    from elastic_sim.sim2real.parameters import check_write_target
    from elastic_sim.sim2real.report import compare_runs, write_json

    folders = [f for f in args.folders if (f / "metrics.json").is_file()]
    table, rows = compare_runs(folders, split=args.split)
    print(table)
    if args.out is not None:
        out = args.out.resolve()
        check_write_target(out.with_suffix(".md"))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.with_suffix(".md").write_text(table, encoding="utf-8")
        write_json(out.with_suffix(".json"), rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
