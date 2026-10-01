"""Standalone ``plan`` stage entry point (no ``rclpy``).

Run as ``python3 -m erd_recording.plan_cli`` -- directly by an operator, or as
a subprocess from :mod:`erd_recording.pipeline` with
:func:`erd_recording.env_guard.clean_ros_subprocess_env`, so pinocchio never
loads in the same process as a sourced ROS environment (T1.0/T1.6).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .env_guard import assert_environment

assert_environment(require_ros=False, require_pinocchio=True)

from .config import load_lab_config  # noqa: E402
from .planning import build_plan  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="erd_recording.plan_cli")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-candidates", type=int, default=None)
    parser.add_argument("--driver-urdf", default=None,
                        help="the driver's own description, exported by the ROS side (RR_04 B-2 FK gate)")
    parser.add_argument("--driver-urdf-calibrated", default=None,
                        help="UR10 only: report-only deviation against --driver-urdf (RR_04 B-2)")
    parser.add_argument("--ladder", type=float, default=None,
                        help="RR_06 P3-3b: also build a commissioning-scaled copy of every excitation segment")
    args = parser.parse_args(argv or sys.argv[1:])

    config = load_lab_config(args.config)
    bundle = build_plan(
        config, Path(args.output_dir), n_candidates=args.n_candidates,
        driver_urdf_path=Path(args.driver_urdf) if args.driver_urdf else None,
        driver_urdf_calibrated_path=Path(args.driver_urdf_calibrated) if args.driver_urdf_calibrated else None,
        ladder_scale=args.ladder,
    )
    print(f"plan ok: {len(bundle.segments)} segments, config_digest={bundle.config_digest[:12]}, "
          f"rejected_for_collision={bundle.rejected_for_collision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
