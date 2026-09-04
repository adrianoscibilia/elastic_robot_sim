#!/usr/bin/env python3
"""Validate, simulate, and/or record one asset-based experiment."""

from __future__ import annotations

import argparse
import platform
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from elastic_sim.experiment import (  # noqa: E402
    ExperimentStore,
    artifact_root,
    config_digest,
    generate_materialized_trajectory,
    load_experiment_config,
    resolve_asset,
    rollout_to_frame,
)
from elastic_sim.sim2real import run_simulation  # noqa: E402


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", required=True, help="sim-to-real asset YAML")
    parser.add_argument("--sim-only", action="store_true", help="write only below data/simulated/<robot>")
    parser.add_argument("--real-only", action="store_true", help="write only below data/recorded/<robot>")
    parser.add_argument("--dry-run", action="store_true", help="validate without creating artifacts or contacting ROS")
    parser.add_argument("--backends", nargs="+", choices=("newton", "mujoco"), default=None)
    parser.add_argument("--no-motor-control", action="store_true", help="skip configured motor lifecycle services")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-id", default=None, help="shared run directory name")
    parser.add_argument("--simulated-root", default=None, help="override root; final component must be simulated")
    parser.add_argument("--recorded-root", default=None, help="override root; final component must be recorded")
    return parser.parse_args()


def _validate(config: dict, asset) -> None:
    asset.resolve_active_joints()
    asset.validate_resources()
    trajectory = config.get("trajectory", {}) or {}
    if float(trajectory.get("duration", 0.0)) <= 0.0:
        raise ValueError("trajectory.duration must be positive")
    if float(trajectory.get("time_step", 0.01)) <= 0.0:
        raise ValueError("trajectory.time_step must be positive")


def _save_trajectories(store: ExperimentStore, trajectories) -> None:
    store.save_trajectory(trajectories[0])
    if len(trajectories) > 1:
        for index, trajectory in enumerate(trajectories):
            store.save_trajectory(trajectory, f"trajectories/trajectory_{index:03d}.json")


def _manifest(config: dict, asset, trajectories, backends, digest: str, kind: str) -> dict:
    return {
        "schema_version": 2,
        "artifact_kind": kind,
        "asset": asset.name,
        "asset_urdf": str(asset.urdf_path),
        "config_path": str(Path(config["_config_path"]).resolve()),
        "config_hash": digest,
        "trajectory": {"count": len(trajectories), "joint_names": list(asset.joint_names), "digests": [item.digest() for item in trajectories]},
        "backends": list(backends),
        "parameters": config.get("model", {}),
        "software": {"python": platform.python_version(), "platform": platform.platform()},
        "start_timestamp": datetime.now(timezone.utc).isoformat(),
        "completion_status": "incomplete",
    }


def main() -> int:
    args = _args()
    if args.sim_only and args.real_only:
        raise SystemExit("--sim-only and --real-only are mutually exclusive")
    config = load_experiment_config(args.config)
    asset = resolve_asset(config, args.config)
    _validate(config, asset)
    backends = tuple(args.backends or config.get("simulation", {}).get("backends", ("newton", "mujoco")))
    seed = int(args.seed if args.seed is not None else config.get("trajectory", {}).get("seed", 0))
    count = max(1, int(config.get("trajectory", {}).get("num_trajectories", 1)))
    digest = config_digest(config)
    trajectories = []
    for index in range(count):
        trajectory = generate_materialized_trajectory(asset, config, seed + index)
        trajectories.append(replace(trajectory, metadata=trajectory.metadata | {"config_hash": digest}))
    if args.dry_run:
        print(f"asset={asset.name} joints={list(asset.joint_names)}")
        print(f"trajectories={count} samples={[len(item.time) for item in trajectories]} backends={list(backends)}")
        print(f"config_hash={digest}")
        return 0

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sim_store = None if args.real_only else ExperimentStore(artifact_root(config, "simulated", _REPO, args.simulated_root), asset.name, run_id)
    real_store = None if args.sim_only else ExperimentStore(artifact_root(config, "recorded", _REPO, args.recorded_root), asset.name, run_id)
    sim_manifest = _manifest(config, asset, trajectories, backends, digest, "simulated") if sim_store else None
    real_manifest = _manifest(config, asset, trajectories, (), digest, "recorded") if real_store else None
    if sim_manifest is not None and real_store is not None:
        sim_manifest["recorded_counterpart"] = str(real_store.path)
    if real_manifest is not None and sim_store is not None:
        real_manifest["simulated_counterpart"] = str(sim_store.path)

    try:
        if sim_store is not None:
            _save_trajectories(sim_store, trajectories)
            for backend in backends:
                frames = []
                for index, trajectory in enumerate(trajectories):
                    frame = rollout_to_frame(trajectory, run_simulation(asset, config, trajectory, backend), source=f"sim_{backend}")
                    if not np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy(float)).all():
                        raise RuntimeError(f"{backend} produced non-finite output for trajectory {index}")
                    frame.insert(0, "trajectory_id", index)
                    frames.append(frame)
                sim_store.save_frame(pd.concat(frames, ignore_index=True), f"sim_{backend}.parquet")
            sim_manifest["completion_status"] = "simulation_complete"
            sim_manifest["end_timestamp"] = datetime.now(timezone.utc).isoformat()
            sim_store.save_manifest(sim_manifest)

        if real_store is not None:
            _save_trajectories(real_store, trajectories)
            from elastic_sim.ros_experiment import execute_real_trajectories
            real, ros_manifest = execute_real_trajectories(config, trajectories, real_store.path / "raw", no_motor_control=args.no_motor_control)
            real_store.save_frame(real, "observations.parquet")
            real_store.save_frame(real, "real.parquet")
            real_manifest["ros"] = ros_manifest
            real_manifest["raw_bag_path"] = str(real_store.path / "raw" / "rosbag2")
            real_manifest["completion_status"] = "complete"
            real_manifest["end_timestamp"] = datetime.now(timezone.utc).isoformat()
            real_store.save_manifest(real_manifest)
        print("Experiment complete: " + ", ".join(str(store.path) for store in (sim_store, real_store) if store is not None))
        return 0
    except Exception as exc:
        for store, manifest in ((sim_store, sim_manifest), (real_store, real_manifest)):
            if store is not None and manifest is not None and manifest.get("completion_status") == "incomplete":
                manifest["error"] = f"{type(exc).__name__}: {exc}"
                manifest["end_timestamp"] = datetime.now(timezone.utc).isoformat()
                store.save_manifest(manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
