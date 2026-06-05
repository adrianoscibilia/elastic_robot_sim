"""Collect a synchronized calibration dataset: sim (Newton/MuJoCo) + real robot.

All backends share the same randomly-generated trajectory and the same
datetime timestamp per recording, so::

    newton_20240605_143022.parquet
    mujoco_20240605_143022.parquet
    real_20240605_143022.parquet
    trajectory_20240605_143022.json

are directly comparable.

The real robot is launched as a background subprocess (requires a ROS 2
environment); if it fails or is absent the sim files for that recording are
still valid.  Ctrl-C cleanly aborts between trajectories and terminates any
in-flight robot subprocess.

Usage examples
--------------
# Both sims + real robot, 10 random trajectories
python scripts/collect_dataset.py \\
    --backends newton mujoco real \\
    --num-trajectories 10 \\
    --output-dir data/recordings/session_01

# Sim-only, run until Ctrl-C
python scripts/collect_dataset.py \\
    --backends newton mujoco \\
    --num-trajectories 0 \\
    --output-dir data/recordings/sim_only

# Newton only, PTP trajectories, calibrated params
python scripts/collect_dataset.py \\
    --backends newton real \\
    --modes ptp \\
    --params calibrated_newton.yaml \\
    --num-trajectories 20 \\
    --output-dir data/recordings/newton_cal
"""

from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
import time
from datetime import datetime

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC = os.path.join(_REPO, "src")
_SCRIPTS = os.path.join(_REPO, "scripts")
for _p in (_SRC, _SCRIPTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from elastic_sim.params import RobotParams
from elastic_sim.rollout import RolloutResult
from elastic_sim.trajectory import (
    make_ptp_trajectory,
    make_sinusoidal_trajectory,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Record synchronized sim+real calibration data with timestamped files."
    )
    p.add_argument(
        "--backends", nargs="+",
        choices=["newton", "mujoco", "real"],
        default=["newton", "mujoco"],
        help="Backends to record (default: newton mujoco).",
    )
    p.add_argument(
        "--num-trajectories", type=int, default=10,
        metavar="N",
        help="How many trajectories to collect (0 = run until Ctrl-C, default: 10).",
    )
    p.add_argument(
        "--output-dir", required=True,
        help="Directory where all timestamped files are saved.",
    )
    p.add_argument(
        "--modes", nargs="+", choices=["sin", "ptp"], default=["sin", "ptp"],
        help="Trajectory modes to sample randomly (default: sin ptp).",
    )
    p.add_argument(
        "--sim-time", type=float, default=15.0,
        help="Trajectory duration in seconds (default: 15.0).",
    )
    p.add_argument(
        "--time-step", type=float, default=0.01,
        help="Sim integration step in seconds (default: 0.01).",
    )
    p.add_argument(
        "--params", default=None,
        help="Path to a YAML with RobotParams (default: config/settings.yaml).",
    )
    p.add_argument(
        "--master-seed", type=int, default=None,
        help="Seed for the trajectory-seed RNG (default: random).",
    )
    p.add_argument(
        "--noise", action="store_true",
        help="Add sensor noise to sim measurements (off by default).",
    )
    p.add_argument(
        "--cut-off-time", type=float, default=0.0,
        help="Skip the first N seconds of the recording (default: 0.0).",
    )
    p.add_argument(
        "--ros-python", default="python3",
        help=(
            "Python executable used to launch real_robot/record_real_rollout.py. "
            "Point this at the Python in your sourced ROS 2 workspace if needed."
        ),
    )
    p.add_argument(
        "--ft-topic", default="/ft_sensor/wrench",
        help="ROS 2 topic for force/torque data (forwarded to the real robot recorder).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Sim runners
# ---------------------------------------------------------------------------

def _run_newton(
    params: RobotParams, traj, *, noise: bool, cut_off_time: float, time_step: float
) -> RolloutResult:
    from elastic_sim.sim_runner import build_model, run_rollout
    model, dof_map, _ = build_model(params)
    return run_rollout(
        model, dof_map, traj,
        noise=noise, cut_off_time=cut_off_time, time_step=time_step,
    )


def _run_mujoco(
    params: RobotParams, traj, *, noise: bool, cut_off_time: float, time_step: float
) -> RolloutResult:
    from elastic_sim.mujoco_runner import build_model, run_rollout
    model, data, dof_map, act_map = build_model(params, time_step=time_step)
    return run_rollout(
        model, data, dof_map, act_map, traj,
        noise=noise, cut_off_time=cut_off_time,
    )


_SIM_RUNNERS = {"newton": _run_newton, "mujoco": _run_mujoco}


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------

def _save_rollout(rollout: RolloutResult, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    rollout.to_dataframe().to_parquet(path, index=False)


def _start_robot(
    traj_json: str, output_file: str, ros_python: str, ft_topic: str
) -> subprocess.Popen:
    """Launch real_robot/record_real_rollout.py as a non-blocking subprocess."""
    script = os.path.join(_REPO, "real_robot", "record_real_rollout.py")
    cmd = [
        ros_python, script,
        "--traj-config", traj_json,
        "--output-file", output_file,
        "--ft-topic", ft_topic,
    ]
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    params = RobotParams.from_yaml(args.params)

    # Joint limits from sim_common
    from sim_common import MOTOR_JOINT_LIMIT_XY, MOTOR_JOINT_LIMIT_Z
    joint_limits = {
        "x": (-MOTOR_JOINT_LIMIT_XY, MOTOR_JOINT_LIMIT_XY),
        "y": (-MOTOR_JOINT_LIMIT_XY, MOTOR_JOINT_LIMIT_XY),
        "z": (-MOTOR_JOINT_LIMIT_Z, MOTOR_JOINT_LIMIT_Z),
    }

    sim_backends = [b for b in args.backends if b != "real"]
    has_real = "real" in args.backends
    rng = random.Random(args.master_seed)

    count = 0
    robot_proc: subprocess.Popen | None = None
    infinite = args.num_trajectories == 0

    print(f"Output dir : {args.output_dir}")
    print(f"Backends   : {args.backends}")
    print(f"Modes      : {args.modes}")
    print(f"Sim time   : {args.sim_time}s  time-step={args.time_step}s")
    print(f"Trajectories: {'unlimited (Ctrl-C to stop)' if infinite else args.num_trajectories}")

    try:
        while infinite or count < args.num_trajectories:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            seed = rng.randint(0, 2**31 - 1)
            mode_str = rng.choice(args.modes)
            mode_id = 1 if mode_str == "sin" else 2

            print(f"\n{'='*60}")
            print(f"  Recording {count + 1}{'/' + str(args.num_trajectories) if not infinite else ''}"
                  f"  ts={ts}  mode={mode_str}  seed={seed}")

            # Generate trajectory
            if mode_str == "sin":
                traj = make_sinusoidal_trajectory(joint_limits, args.sim_time, seed)
            else:
                traj = make_ptp_trajectory(joint_limits, args.sim_time, seed)

            traj_json = os.path.join(args.output_dir, f"trajectory_{ts}.json")
            traj.config.save(traj_json)
            print(f"  Trajectory → {os.path.basename(traj_json)}")

            # ----------------------------------------------------------------
            # Launch real robot in background BEFORE running sims.
            # The robot takes real-time; sims finish in a few seconds.
            # ----------------------------------------------------------------
            robot_proc = None
            real_out = os.path.join(args.output_dir, f"real_{ts}.parquet")
            if has_real:
                print("  [real] Launching robot subprocess...")
                robot_proc = _start_robot(traj_json, real_out, args.ros_python, args.ft_topic)

            # ----------------------------------------------------------------
            # Run simulators sequentially (they run much faster than real-time)
            # ----------------------------------------------------------------
            for backend in sim_backends:
                print(f"  [{backend}] Running simulation...", end="", flush=True)
                t0 = time.monotonic()
                try:
                    rollout = _SIM_RUNNERS[backend](
                        params, traj,
                        noise=args.noise,
                        cut_off_time=args.cut_off_time,
                        time_step=args.time_step,
                    )
                    out = os.path.join(args.output_dir, f"{backend}_{ts}.parquet")
                    _save_rollout(rollout, out)
                    dt = time.monotonic() - t0
                    print(f" {len(rollout.time)} steps in {dt:.1f}s → {os.path.basename(out)}")
                except Exception as exc:
                    print(f" ERROR: {exc}")

            # ----------------------------------------------------------------
            # Wait for real robot to finish
            # ----------------------------------------------------------------
            if robot_proc is not None:
                print("  [real] Waiting for robot to finish...", end="", flush=True)
                stdout, stderr = robot_proc.communicate()
                if robot_proc.returncode == 0:
                    print(f" done → {os.path.basename(real_out)}")
                    if stdout.strip():
                        for line in stdout.strip().splitlines():
                            print(f"         {line}")
                else:
                    print(f" FAILED (exit {robot_proc.returncode})")
                    if stderr.strip():
                        # Print at most 5 lines of stderr
                        for line in stderr.strip().splitlines()[:5]:
                            print(f"         [stderr] {line}")
                    print("  [real] Sim files are still valid for this recording.")
                robot_proc = None

            count += 1

    except KeyboardInterrupt:
        print(f"\nInterrupted after {count} completed recordings.")

    finally:
        # Terminate any in-flight robot subprocess
        if robot_proc is not None and robot_proc.poll() is None:
            print("  [real] Terminating robot subprocess...")
            robot_proc.terminate()
            try:
                robot_proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                robot_proc.kill()

    print(f"\nDataset collection complete: {count} trajectories saved to {args.output_dir}")
    print(
        "\nTo calibrate, run:\n"
        f"  python scripts/run_calibration.py --backend newton "
        f"--recordings-dir {args.output_dir}"
    )


if __name__ == "__main__":
    main()
