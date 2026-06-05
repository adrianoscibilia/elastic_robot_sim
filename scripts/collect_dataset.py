"""Collect real-robot calibration data: trajectory configs + real recordings.

Reads default parameters from config/settings.yaml (collection section).
Every CLI argument overrides the corresponding settings.yaml value.

The script generates random sinusoidal / PTP trajectories, saves their
configs (with seeds) for full reproducibility, then executes each one on
the real robot via ROS 2 and records the joint/force signals.

The resulting files are the ground truth used by run_calibration.py:

    data/recordings/<session>/
        trajectory_20240605_143022.json   ← seed + waypoints for reproducibility
        real_20240605_143022.parquet      ← recorded joint states + F/T
        trajectory_20240605_143058.json
        real_20240605_143058.parquet
        ...

Usage
-----
# Use settings from config/settings.yaml
python scripts/collect_dataset.py --output-dir data/recordings/session_01

# Override specific settings on the CLI
python scripts/collect_dataset.py \\
    --output-dir data/recordings/session_01 \\
    --num-trajectories 20 \\
    --modes ptp \\
    --sim-time 12.0

# Point at a different settings file
python scripts/collect_dataset.py \\
    --output-dir data/recordings/session_01 \\
    --settings config/settings_robot2.yaml
"""

from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
import time
from datetime import datetime

import yaml

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC = os.path.join(_REPO, "src")
_SCRIPTS = os.path.join(_REPO, "scripts")
for _p in (_SRC, _SCRIPTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from elastic_sim.trajectory import make_ptp_trajectory, make_sinusoidal_trajectory


# ---------------------------------------------------------------------------
# Settings loader
# ---------------------------------------------------------------------------

def _load_settings(path: str | None = None) -> dict:
    if path is None:
        path = os.path.join(_REPO, "config", "settings.yaml")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# CLI  (all args are optional — settings.yaml provides the defaults)
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Collect real-robot calibration recordings.  "
            "Defaults are read from config/settings.yaml (collection section)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--output-dir", required=True,
        help="Directory where trajectory JSON files and real.parquet files are saved.",
    )
    p.add_argument(
        "--settings", default=None, metavar="PATH",
        help="Path to a settings YAML file (default: config/settings.yaml).",
    )
    p.add_argument(
        "--num-trajectories", type=int, default=None, metavar="N",
        help="Number of trajectories to collect (0 = run until Ctrl-C).",
    )
    p.add_argument(
        "--modes", nargs="+", choices=["sin", "ptp"], default=None,
        help="Trajectory modes to randomly sample.",
    )
    p.add_argument(
        "--sim-time", type=float, default=None,
        help="Duration of each trajectory in seconds.",
    )
    p.add_argument(
        "--master-seed", type=int, default=None,
        help="Seed for the trajectory-seed RNG (overrides settings.yaml; null = random).",
    )
    p.add_argument(
        "--ros-python", default=None,
        help="Python executable used to launch real_robot/record_real_rollout.py.",
    )
    p.add_argument(
        "--ft-topic", default=None,
        help="ROS 2 force-torque topic name.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Robot subprocess
# ---------------------------------------------------------------------------

def _start_robot(
    traj_json: str, output_file: str, ros_python: str, ft_topic: str
) -> subprocess.Popen:
    script = os.path.join(_REPO, "real_robot", "record_real_rollout.py")
    cmd = [
        ros_python, script,
        "--traj-config", traj_json,
        "--output-file", output_file,
        "--ft-topic", ft_topic,
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    settings = _load_settings(args.settings)
    col = settings.get("collection", {})

    # Resolve effective configuration (CLI wins over settings.yaml)
    num_trajectories: int  = args.num_trajectories if args.num_trajectories is not None else col.get("num_trajectories", 10)
    modes: list[str]       = args.modes            if args.modes is not None            else col.get("modes", ["sin", "ptp"])
    sim_time: float        = args.sim_time          if args.sim_time is not None          else col.get("sim_time", 15.0)
    master_seed            = args.master_seed       if args.master_seed is not None       else col.get("master_seed", None)
    ros_python: str        = args.ros_python        if args.ros_python is not None        else col.get("ros_python", "python3")
    ft_topic: str          = args.ft_topic          if args.ft_topic is not None          else col.get("ft_topic", "/ft_sensor/wrench")

    raw_limits = col.get("joint_limits", {"x": [-1.8, 1.8], "y": [-1.8, 1.8], "z": [-1.0, 1.0]})
    joint_limits = {ax: tuple(v) for ax, v in raw_limits.items()}

    os.makedirs(args.output_dir, exist_ok=True)

    rng = random.Random(master_seed)
    infinite = num_trajectories == 0

    print(f"Output dir      : {args.output_dir}")
    print(f"Trajectory modes: {modes}")
    print(f"Duration        : {sim_time} s")
    print(f"Joint limits    : {joint_limits}")
    print(f"Master seed     : {master_seed if master_seed is not None else 'random'}")
    print(f"Trajectories    : {'unlimited (Ctrl-C to stop)' if infinite else num_trajectories}")

    count = 0
    robot_proc: subprocess.Popen | None = None

    try:
        while infinite or count < num_trajectories:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            seed = rng.randint(0, 2**31 - 1)
            mode_str = rng.choice(modes)

            print(f"\n{'='*60}")
            print(f"  Recording {count + 1}{'/' + str(num_trajectories) if not infinite else ''}"
                  f"  ts={ts}  mode={mode_str}  seed={seed}")

            # Generate and save trajectory
            if mode_str == "sin":
                traj = make_sinusoidal_trajectory(joint_limits, sim_time, seed)
            else:
                traj = make_ptp_trajectory(joint_limits, sim_time, seed)

            traj_json = os.path.join(args.output_dir, f"trajectory_{ts}.json")
            traj.config.save(traj_json)
            print(f"  Trajectory saved: trajectory_{ts}.json")

            # Execute on real robot
            real_out = os.path.join(args.output_dir, f"real_{ts}.parquet")
            print("  [real] Launching robot subprocess...", flush=True)
            robot_proc = _start_robot(traj_json, real_out, ros_python, ft_topic)

            # Wait for the robot to finish executing the full trajectory
            stdout, stderr = robot_proc.communicate()
            if robot_proc.returncode == 0:
                print(f"  [real] Saved: real_{ts}.parquet")
                if stdout.strip():
                    for line in stdout.strip().splitlines():
                        print(f"         {line}")
            else:
                print(f"  [real] FAILED (exit {robot_proc.returncode}) — trajectory config kept for debugging")
                if stderr.strip():
                    for line in stderr.strip().splitlines()[:5]:
                        print(f"         [stderr] {line}")
            robot_proc = None

            count += 1

    except KeyboardInterrupt:
        print(f"\nInterrupted after {count} completed recordings.")

    finally:
        if robot_proc is not None and robot_proc.poll() is None:
            print("  [real] Terminating robot subprocess...")
            robot_proc.terminate()
            try:
                robot_proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                robot_proc.kill()

    print(f"\nCollection complete: {count} trajectories saved to {args.output_dir}")
    print(
        "\nNext — run calibration:\n"
        f"  python scripts/run_calibration.py "
        f"--recordings-dir {args.output_dir}"
    )


if __name__ == "__main__":
    main()
