#!/usr/bin/env python3
"""Write a run's result as a candidate overlay (S2R_06 P4a-5).

``<run>/candidate_calibrated.yaml``: the stage's full parameter set, each
entry ``calibrated: true | false``, plus the provenance header. It stays
under ``data/sim2real/``; publishing to ``config/sim2real/calibrated/``
waits for the architect's review (A-3). ``--show`` prints it, writes nothing.

Example::

    python scripts/sim2real/candidate.py data/sim2real/kuka_lbr_iiwa_14_r820/iiwa_s1_rigid
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
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--show", action="store_true", help="print best.yaml's calibrated/not table only")
    args = parser.parse_args(argv)

    import yaml

    from elastic_sim.sim2real.report import write_candidate

    if args.show:
        entries = yaml.safe_load((args.run_dir / "best.yaml").read_text(encoding="utf-8"))["parameters"]
        for name, entry in entries.items():
            print(f"{'calibrated' if entry.get('calibrated', True) else 'NOT       '}  {name:45s} {entry['value']:.6g}")
        return 0
    print(write_candidate(args.run_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
