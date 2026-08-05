#!/usr/bin/env python3
"""Simulate one portable robot asset along a replayable joint trajectory.

Examples
--------
Generate and execute a UR10 excitation::

    python scripts/run_asset_simulation.py --asset ur10 --output data/ur10_trial.csv --seed 42

Validate an asset and save its trajectory without requiring Newton::

    python scripts/run_asset_simulation.py --asset kuka_kr300_r2500_ultra_se --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(_REPO / "src"))

from elastic_sim.assets import AssetRegistry, load_asset_spec
from elastic_sim.elastic_settings import load_simulation_settings, merge_runner_settings
from elastic_sim.generic_newton_runner import (
    GenericNewtonRigidTrajectoryRunner,
    GenericNewtonTrajectoryRunner,
    GenericNewtonKinematicTrajectoryRunner,
)
from elastic_sim.generic_mujoco_runner import (
    GenericMujocoElasticTrajectoryRunner,
    GenericMujocoKinematicTrajectoryRunner,
    GenericMujocoRigidTrajectoryRunner,
)
from elastic_sim.serial_trajectory import SerialTrajectoryConfig, generate_serial_arm_trajectory


def _load_asset(reference: str):
    candidate = Path(reference)
    if candidate.is_file():
        return load_asset_spec(candidate)
    return AssetRegistry.for_repository(_REPO).load(reference)


def _frame_from_result(result):
    import pandas as pd
    frame = {"time": result["time"]}
    for prefix, key in (
        ("q_ref", "q_ref"), ("dq_ref", "dq_ref"), ("q", "q_link"), ("dq", "dq_link"),
        ("q_motor", "q_motor"), ("dq_motor", "dq_motor"), ("tau", "tau_motor"),
    ):
        values = result[key]
        for index in range(values.shape[1]):
            frame[f"{prefix}{index}"] = values[:, index]
    return pd.DataFrame(frame)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", required=True, help="Asset name or asset.yaml path")
    parser.add_argument("--trajectory", default=None, help="Saved trajectory JSON; generate one when omitted")
    parser.add_argument("--trajectory-output", default=None, help="Where to save a generated trajectory JSON")
    parser.add_argument("--output", default=None, help="Optional CSV output for the simulated canonical rollout")
    parser.add_argument("--settings", default=None, help="Optional YAML with elastic transmission and mass overrides")
    parser.add_argument("--backend", choices=("newton", "mujoco"), default="newton", help="Simulation backend")
    parser.add_argument("--time-step", type=float, default=0.004)
    parser.add_argument("--waypoints", type=int, default=8)
    parser.add_argument("--max-velocity", type=float, default=0.8)
    parser.add_argument("--max-acceleration", type=float, default=1.5)
    parser.add_argument("--limit-margin", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--stiffness", type=float, default=None, help="Elastic transmission stiffness for all active joints")
    parser.add_argument("--damping", type=float, default=None, help="Elastic transmission damping for all active joints")
    parser.add_argument("--motor-stiffness", type=float, default=None)
    parser.add_argument("--motor-damping", type=float, default=None)
    parser.add_argument("--intermediate-mass", type=float, default=None, help="Default fictional transmission-link mass")
    parser.add_argument("--intermediate-size", type=float, default=None, help="Newton proxy-inertia size for transmission links")
    parser.add_argument("--dynamics", choices=("kinematic", "rigid", "elastic"), default="kinematic", help="Kinematic playback is the stable default; rigid/elastic are physics modes")
    parser.add_argument("--visualize", action="store_true", help="Show the selected backend's viewer during simulation")
    parser.add_argument("--realtime-scale", type=float, default=1.0, help="Viewer playback rate; 1.0 is real time")
    parser.add_argument("--dry-run", action="store_true", help="Validate resources and create a trajectory only")
    args = parser.parse_args()
    asset = _load_asset(args.asset)
    asset.resolve_active_joints()
    resources = asset.validate_resources()
    trajectory = SerialTrajectoryConfig.load(args.trajectory) if args.trajectory else generate_serial_arm_trajectory(
        asset.urdf_path, joint_names=asset.joint_names, num_waypoints=args.waypoints,
        max_velocity=args.max_velocity, max_acceleration=args.max_acceleration,
        limit_margin=args.limit_margin, seed=args.seed,
    )
    if args.trajectory_output:
        trajectory.save(args.trajectory_output)
    print(f"Asset {asset.name}: {len(asset.joint_names)} active joints; {len(resources)} mesh resources; trajectory {trajectory.duration:.3f}s")
    if args.dry_run:
        return
    config = load_simulation_settings(args.settings, asset) if args.settings else {}
    config = merge_runner_settings(config, {
        "default_stiffness": args.stiffness, "default_damping": args.damping,
        "motor_stiffness": args.motor_stiffness, "motor_damping": args.motor_damping,
        "intermediate_mass": args.intermediate_mass, "intermediate_size": args.intermediate_size,
    })
    if args.backend == "newton" and args.dynamics == "kinematic":
        runner = GenericNewtonKinematicTrajectoryRunner(asset, config)
    elif args.backend == "newton" and args.dynamics == "rigid":
        runner = GenericNewtonRigidTrajectoryRunner(asset, config)
    elif args.backend == "newton":
        runner = GenericNewtonTrajectoryRunner(asset, config)
    elif args.dynamics == "kinematic":
        runner = GenericMujocoKinematicTrajectoryRunner(asset, config)
    elif args.dynamics == "rigid":
        runner = GenericMujocoRigidTrajectoryRunner(asset, config)
    else:
        runner = GenericMujocoElasticTrajectoryRunner(asset, config)
    run_kwargs = {"time_step": args.time_step}
    run_kwargs.update(visualize=args.visualize, realtime_scale=args.realtime_scale)
    result = runner.run(trajectory, **run_kwargs)
    if not args.output:
        print(f"Simulation completed ({len(result['time'])} samples); no output was requested.")
        return
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _frame_from_result(result).to_csv(output, index=False)
    print(f"Wrote {len(result['time'])} simulated samples to {output}")


if __name__ == "__main__":
    main()
