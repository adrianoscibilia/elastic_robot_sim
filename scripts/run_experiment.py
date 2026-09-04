"""Materialize, simulate, and/or execute one sim-to-real experiment.

Examples::

    uv run python scripts/run_experiment.py --config config/assets/fmrr_tecnobody_sim2real.yaml
    uv run python scripts/run_experiment.py --config ... --sim-only --backends newton mujoco
    uv run python scripts/run_experiment.py --config ... --real-only --no-motor-control
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from elastic_sim.experiment import (  # noqa: E402
    ExperimentStore,
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
    parser.add_argument("--sim-only", action="store_true", help="generate and simulate, without ROS")
    parser.add_argument("--real-only", action="store_true", help="generate and execute, without simulation")
    parser.add_argument("--dry-run", action="store_true", help="validate and materialize metadata without running backends or ROS")
    parser.add_argument("--backends", nargs="+", choices=("newton", "mujoco"), default=None)
    parser.add_argument("--no-motor-control", action="store_true", help="do not call configured motor lifecycle services")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-id", default=None, help="override the UTC run directory name")
    return parser.parse_args()


def _root(config: dict) -> Path:
    value = config.get("output_root", "data/experiments")
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (_REPO / path).resolve()


def _validate(config: dict, asset) -> None:
    if asset.name != str(config.get("asset")) and Path(str(config.get("asset"))).name != asset.name:
        raise ValueError("resolved asset does not match config asset identifier")
    asset.resolve_active_joints()
    if bool(config.get("validate_meshes", False)):
        asset.validate_resources()
    trajectory = config.get("trajectory", {}) or {}
    if float(trajectory.get("duration", 0.0)) <= 0.0:
        raise ValueError("trajectory.duration must be positive")
    if float(trajectory.get("time_step", 0.01)) <= 0.0:
        raise ValueError("trajectory.time_step must be positive")


def main() -> int:
    args = _args()
    if args.sim_only and args.real_only:
        raise SystemExit("--sim-only and --real-only are mutually exclusive")
    config = load_experiment_config(args.config)
    asset = resolve_asset(config, args.config)
    _validate(config, asset)
    backends = tuple(args.backends or config.get("simulation", {}).get("backends", ("newton", "mujoco")))
    seed = int(args.seed if args.seed is not None else config.get("trajectory", {}).get("seed", 0))
    count = max(1, int(config.get("trajectory", {}).get("num_trajectories", config.get("trajectory", {}).get("number", 1))))
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

    store = ExperimentStore(_root(config), asset.name, args.run_id)
    status = "incomplete"
    manifest = {
        "schema_version": 1,
        "asset": asset.name,
        "asset_urdf": str(asset.urdf_path),
        "config_path": str(Path(args.config).resolve()),
        "config_hash": digest,
        "trajectory": {"count": count, "joint_names": list(asset.joint_names), "seed": seed, "digests": [item.digest() for item in trajectories]},
        "backends": list(backends),
        "parameters": config.get("model", {}),
        "ros": {
            "action_server": config.get("ros", {}).get("action_server", "/joint_trajectory_controller/follow_joint_trajectory"),
            "topics": config.get("ros", {}).get("topics", {}),
            "topic_types": {
                "joint_states": "sensor_msgs/msg/JointState",
                "flange_wrench": "geometry_msgs/msg/WrenchStamped",
                "controller_state": "control_msgs/msg/JointTrajectoryControllerState",
            },
        },
        "units": {"joint_position": "SI as declared by asset", "joint_velocity": "SI/s", "effort": "SI torque/force as declared by asset"},
        "software": {"python": platform.python_version(), "platform": platform.platform()},
        "start_timestamp": datetime.now(timezone.utc).isoformat(),
        "completion_status": status,
    }
    try:
        if count == 1:
            store.save_trajectory(trajectories[0])
        else:
            store.save_trajectory(trajectories[0])
            for index, trajectory in enumerate(trajectories):
                store.save_trajectory(trajectory, f"trajectories/trajectory_{index:03d}.json")

        if not args.real_only:
            for backend in backends:
                frames = []
                for index, trajectory in enumerate(trajectories):
                    result = run_simulation(asset, config, trajectory, backend)
                    frame = rollout_to_frame(trajectory, result, source=f"sim_{backend}")
                    numeric = frame.select_dtypes(include=[np.number])
                    if not np.isfinite(numeric.to_numpy(float)).all():
                        raise RuntimeError(f"{backend} produced non-finite output for trajectory {index}")
                    frame.insert(0, "trajectory_id", index)
                    frames.append(frame)
                store.save_frame(__import__("pandas").concat(frames, ignore_index=True), f"sim_{backend}.parquet")

        if not args.sim_only:
            from elastic_sim.ros_experiment import execute_real_trajectories
            real, ros_manifest = execute_real_trajectories(config, trajectories, store.path / "raw", no_motor_control=args.no_motor_control)
            store.save_frame(real, "observations.parquet")
            # Calibration-ready output keeps the complete normalized signal
            # set; observations remains the same joined frame for provenance.
            store.save_frame(real, "real.parquet")
            manifest["ros"] = ros_manifest
            manifest["raw_bag_path"] = str(store.path / "raw" / "rosbag2")
        status = "complete" if not args.sim_only else "simulation_complete"
        manifest["completion_status"] = status
        manifest["end_timestamp"] = datetime.now(timezone.utc).isoformat()
        store.save_manifest(manifest)
        print(f"Experiment {status}: {store.path}")
        return 0
    except Exception as exc:
        manifest["completion_status"] = "incomplete"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        manifest["end_timestamp"] = datetime.now(timezone.utc).isoformat()
        store.save_manifest(manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
