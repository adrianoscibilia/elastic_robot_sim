"""Record sim rollouts for one or more backends with an identical trajectory.

All requested backends use the same seed, trajectory mode, and time grid so
the resulting rollouts are directly comparable — and also comparable to a
real-robot recording made from the same trajectory.json.

Usage examples
--------------
# Record both simulators at default settings
python scripts/record_rollouts.py --backends newton mujoco

# Record MuJoCo only, PTP trajectory, custom seed
python scripts/record_rollouts.py --backends mujoco --mode 2 --seed 99 --sim-time 15

# Record from an existing trajectory.json (re-run or compare different params)
python scripts/record_rollouts.py --backends newton mujoco \\
    --traj-config data/rollouts/traj_m2_s42/trajectory.json

After recording, run the real robot to add real.parquet:
    python real_robot/record_real_rollout.py \\
        --traj-config data/rollouts/<traj_id>/trajectory.json \\
        --output-dir  data/rollouts/<traj_id>/
"""

from __future__ import annotations

import argparse
import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC = os.path.join(_REPO, "src")
_SCRIPTS = os.path.join(_REPO, "scripts")
for p in (_SRC, _SCRIPTS):
    if p not in sys.path:
        sys.path.insert(0, p)

from elastic_sim.params import RobotParams
from elastic_sim.rollout import RolloutStore
from elastic_sim.trajectory import (
    TrajectoryConfig,
    _trajectory_from_config,
    make_ptp_trajectory,
    make_sinusoidal_trajectory,
    make_hold_trajectory,
)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Record sim rollouts (Newton / MuJoCo) with an identical trajectory."
    )
    p.add_argument(
        "--backends", nargs="+", choices=["newton", "mujoco"], default=["newton", "mujoco"],
        help="Which simulator backends to run (default: both).",
    )
    p.add_argument(
        "--mode", type=int, default=2, choices=[0, 1, 2],
        help="Trajectory mode: 0=hold, 1=sinusoidal, 2=PTP (default: 2).",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    p.add_argument(
        "--sim-time", type=float, default=15.0,
        help="Simulation duration in seconds (default: 15.0).",
    )
    p.add_argument(
        "--time-step", type=float, default=0.01,
        help="Integration step size in seconds (default: 0.01).",
    )
    p.add_argument(
        "--cut-off-time", type=float, default=0.0,
        help="Skip recording the first N seconds (default: 0.0).",
    )
    p.add_argument(
        "--noise", action="store_true",
        help="Add sensor noise to sim measurements (off by default for clean comparison).",
    )
    p.add_argument(
        "--traj-config", default=None,
        help="Path to an existing trajectory.json.  If given, --mode/--seed/--sim-time are ignored.",
    )
    p.add_argument(
        "--output-dir", default=None,
        help=(
            "Output directory for rollout files.  "
            "Defaults to data/rollouts/traj_m<mode>_s<seed>/."
        ),
    )
    p.add_argument(
        "--params", default=None,
        help="Path to a settings.yaml or calibrated_*.yaml to use as sim params. "
             "Defaults to config/settings.yaml.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Backend runners
# ---------------------------------------------------------------------------

def _run_newton(params: RobotParams, traj, *, noise: bool, cut_off_time: float, time_step: float):
    from elastic_sim.sim_runner import build_model, run_rollout
    model, dof_map, _ = build_model(params)
    return run_rollout(
        model, dof_map, traj,
        noise=noise, cut_off_time=cut_off_time, time_step=time_step,
    )


def _run_mujoco(params: RobotParams, traj, *, noise: bool, cut_off_time: float, time_step: float):
    from elastic_sim.mujoco_runner import build_model, run_rollout
    model, data, dof_map, act_map = build_model(params, time_step=time_step)
    return run_rollout(
        model, data, dof_map, act_map, traj,
        noise=noise, cut_off_time=cut_off_time,
    )


_RUNNER = {"newton": _run_newton, "mujoco": _run_mujoco}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # Load params
    params = RobotParams.from_yaml(args.params)

    # Build or load trajectory config
    if args.traj_config is not None:
        print(f"Loading trajectory from {args.traj_config}")
        traj_config = TrajectoryConfig.load(args.traj_config)
        traj_id = os.path.basename(os.path.dirname(args.traj_config))
    else:
        from sim_common import MOTOR_JOINT_LIMIT_XY, MOTOR_JOINT_LIMIT_Z
        joint_limits = {
            "x": (-MOTOR_JOINT_LIMIT_XY, MOTOR_JOINT_LIMIT_XY),
            "y": (-MOTOR_JOINT_LIMIT_XY, MOTOR_JOINT_LIMIT_XY),
            "z": (-MOTOR_JOINT_LIMIT_Z, MOTOR_JOINT_LIMIT_Z),
        }
        mode = args.mode
        if mode == 0:
            import numpy as np
            traj = make_hold_trajectory(np.zeros(3), args.sim_time, joint_limits)
        elif mode == 1:
            traj = make_sinusoidal_trajectory(joint_limits, args.sim_time, args.seed)
        else:
            traj = make_ptp_trajectory(joint_limits, args.sim_time, args.seed)
        traj_config = traj.config
        traj_id = f"traj_m{mode}_s{args.seed}"
        print(f"Generated trajectory: mode={mode}, seed={args.seed}, sim_time={args.sim_time}s")

    # Determine output directory
    if args.output_dir is not None:
        output_dir = args.output_dir
        store = RolloutStore(os.path.dirname(output_dir.rstrip("/\\")))
        traj_id = os.path.basename(output_dir.rstrip("/\\"))
    else:
        rollouts_dir = os.path.join(_REPO, "data", "rollouts")
        store = RolloutStore(rollouts_dir)

    # Save trajectory so the real-robot recorder can use it
    store.save_trajectory(traj_id, traj_config)
    traj_json = os.path.join(store.base_dir, traj_id, "trajectory.json")
    print(f"Trajectory saved to {traj_json}")

    # Reconstruct callable trajectory
    traj = _trajectory_from_config(traj_config)

    # Run each requested backend
    for backend in args.backends:
        print(f"\n[{backend}] Running simulation...")
        runner = _RUNNER[backend]
        rollout = runner(
            params, traj,
            noise=args.noise,
            cut_off_time=args.cut_off_time,
            time_step=args.time_step,
        )
        store.save_sim(traj_id, rollout, backend=backend)
        out = os.path.join(store.base_dir, traj_id, f"{backend}.parquet")
        print(f"[{backend}] Saved {len(rollout.time)} steps → {out}")

    print(f"\nAll backends done.  Rollout directory: {os.path.join(store.base_dir, traj_id)}")
    print(
        "\nNext step — record the real robot:\n"
        f"  python real_robot/record_real_rollout.py \\\n"
        f"      --traj-config {traj_json} \\\n"
        f"      --output-dir  {os.path.join(store.base_dir, traj_id)}\n"
    )


if __name__ == "__main__":
    main()
