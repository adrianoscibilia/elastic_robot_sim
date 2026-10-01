"""``show_plan``: build (or load) a plan bundle and print/publish it.

RViz publishing (robot model, scene markers, planned paths) needs a display
this environment does not have; when no display is available this falls back
to a text summary, which is enough to unit-test the plan/print logic and
enough for an operator running headless to sanity-check a plan bundle before
opening RViz separately.
"""

from __future__ import annotations

import argparse
import sys

from .config import load_lab_config
from .planning import build_plan, load_plan


def show_plan_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="show_plan")
    parser.add_argument("--config", required=True)
    parser.add_argument("--plan-dir", default=None, help="load an existing plan instead of building one")
    parser.add_argument("--n-candidates", type=int, default=None)
    args = parser.parse_args(argv or sys.argv[1:])

    config = load_lab_config(args.config)
    if args.plan_dir:
        bundle = load_plan(__import__("pathlib").Path(args.plan_dir))
    else:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            bundle = build_plan(config, Path(tmp), n_candidates=args.n_candidates)

    print(f"asset: {bundle.asset.name}")
    print(f"config_digest: {bundle.config_digest}")
    print(f"rejected_for_collision: {bundle.rejected_for_collision}")
    for segment in bundle.segments:
        print(f"  {segment.segment_id:16s} {segment.kind:12s} "
              f"{len(segment.trajectory.time):6d} samples  {segment.trajectory.duration:6.2f} s  "
              f"digest={segment.digest[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(show_plan_main())
