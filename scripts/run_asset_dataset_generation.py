#!/usr/bin/env python3
"""Generate canonical synthetic CSV datasets for any portable asset backend."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", required=True, help="Asset name or asset.yaml path")
    parser.add_argument("--backend", choices=("newton", "mujoco"), default="newton")
    parser.add_argument("--dynamics", choices=("kinematic", "rigid", "elastic"), default="kinematic")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-step", type=float, default=0.004)
    parser.add_argument("--stiffness-min", type=float, default=1_000.0)
    parser.add_argument("--stiffness-max", type=float, default=30_000.0)
    parser.add_argument("--damping-min", type=float, default=10.0)
    parser.add_argument("--damping-max", type=float, default=500.0)
    parser.add_argument("--waypoints", type=int, default=8)
    args = parser.parse_args()
    if args.trials < 1 or args.stiffness_min <= 0 or args.stiffness_max < args.stiffness_min:
        parser.error("trials and stiffness bounds must be positive and ordered")
    if args.damping_min < 0 or args.damping_max < args.damping_min:
        parser.error("damping bounds must be non-negative and ordered")
    import numpy as np
    rng = np.random.default_rng(args.seed)
    script = Path(__file__).with_name("run_asset_simulation.py")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for trial in range(args.trials):
        stiffness = float(np.exp(rng.uniform(np.log(args.stiffness_min), np.log(args.stiffness_max))))
        damping = float(rng.uniform(args.damping_min, args.damping_max))
        output = output_dir / f"trial_{trial:04d}.csv"
        command = [
            sys.executable, str(script), "--asset", args.asset, "--output", str(output),
            "--backend", args.backend, "--dynamics", args.dynamics,
            "--seed", str(args.seed + trial), "--time-step", str(args.time_step),
            "--waypoints", str(args.waypoints), "--stiffness", str(stiffness), "--damping", str(damping),
        ]
        subprocess.run(command, check=True)
        manifest.append({
            "trial": trial, "seed": args.seed + trial, "backend": args.backend,
            "dynamics": args.dynamics, "stiffness": stiffness, "damping": damping,
            "file": output.name,
        })
    import json
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Generated {args.trials} {args.backend}/{args.dynamics} trials and {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
