#!/usr/bin/env python3
"""Generate a replayable joint-space excitation trajectory from an asset URDF."""

from __future__ import annotations

import argparse
import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_REPO, "src"))

from elastic_sim.serial_trajectory import generate_serial_arm_trajectory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", required=True, help="Robot URDF asset path")
    parser.add_argument("--output", required=True, help="Output trajectory JSON")
    parser.add_argument("--joints", default=None, help="Optional comma-separated URDF joint names")
    parser.add_argument("--waypoints", type=int, default=8)
    parser.add_argument("--max-velocity", type=float, default=1.0, help="Joint-space bound (rad/s or m/s)")
    parser.add_argument("--max-acceleration", type=float, default=2.0, help="Joint-space bound (rad/s^2 or m/s^2)")
    parser.add_argument("--limit-margin", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    joints = tuple(x.strip() for x in args.joints.split(",") if x.strip()) if args.joints else None
    trajectory = generate_serial_arm_trajectory(
        args.urdf,
        joint_names=joints,
        num_waypoints=args.waypoints,
        max_velocity=args.max_velocity,
        max_acceleration=args.max_acceleration,
        limit_margin=args.limit_margin,
        seed=args.seed,
    )
    trajectory.save(args.output)
    print(f"Saved {len(trajectory.waypoints)} waypoints for {len(trajectory.joint_names)} joints to {args.output}")


if __name__ == "__main__":
    main()
