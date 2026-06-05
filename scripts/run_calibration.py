"""Run the sim-to-real calibration optimisation loop.

Loads all rollouts that have a real.parquet file, splits them into
train/validation sets, then runs the chosen optimizer to minimise the
sim-vs-real fidelity loss.

Usage examples
--------------
# CMA-ES, Newton backend, default settings
python scripts/run_calibration.py --backend newton

# Bayesian optimisation, MuJoCo backend
python scripts/run_calibration.py --backend mujoco --optimizer bo

# Newton, only a specific rollout dir
python scripts/run_calibration.py --backend newton \\
    --rollouts-dir data/rollouts/traj_m2_s42

# Save calibrated params and validate
python scripts/run_calibration.py --backend newton \\
    --output calibrated_newton.yaml --max-evals 300
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC = os.path.join(_REPO, "src")
_SCRIPTS = os.path.join(_REPO, "scripts")
for p in (_SRC, _SCRIPTS):
    if p not in sys.path:
        sys.path.insert(0, p)

from elastic_sim.calibration import SimCalibrationProblem
from elastic_sim.compare import compare
from elastic_sim.params import RobotParams
from elastic_sim.rollout import RolloutStore
from elastic_sim.trajectory import _trajectory_from_config


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sim-to-real calibration: find RobotParams that match recorded real rollouts."
    )
    p.add_argument(
        "--backend", choices=["newton", "mujoco"], default="newton",
        help="Simulator backend to calibrate (default: newton).",
    )
    p.add_argument(
        "--optimizer", choices=["cma", "bo", "skrl"], default="cma",
        help="Optimization backend (default: cma).",
    )
    p.add_argument(
        "--rollouts-dir", default=None,
        help=(
            "Path to a rollout store directory containing <traj_id>/ sub-dirs "
            "(structured format).  Defaults to data/rollouts/."
        ),
    )
    p.add_argument(
        "--recordings-dir", default=None,
        help=(
            "Path to a flat recordings directory produced by collect_dataset.py "
            "(files: trajectory_TS.json + real_TS.parquet).  "
            "Mutually exclusive with --rollouts-dir."
        ),
    )
    p.add_argument(
        "--train-fraction", type=float, default=0.8,
        help="Fraction of rollouts used for optimisation (rest used for validation).",
    )
    p.add_argument(
        "--max-evals", type=int, default=200,
        help="Maximum number of loss evaluations (default: 200).",
    )
    p.add_argument(
        "--cut-off-time", type=float, default=2.0,
        help="Seconds of initial transient to skip in comparison (default: 2.0).",
    )
    p.add_argument(
        "--time-step", type=float, default=0.01,
        help="Simulation time step (default: 0.01).",
    )
    p.add_argument(
        "--output", default=None,
        help=(
            "Path to write the best calibrated params as YAML.  "
            "Defaults to calibrated_<backend>.yaml in the repo root."
        ),
    )
    p.add_argument(
        "--no-payload", action="store_true",
        help="Fix payload at 0 (exclude from optimisation).",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Print optimizer progress.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_rollouts(rollouts_dir: str, backend: str):
    """Load all (real_rollout, traj_config) pairs from a structured rollout store."""
    store = RolloutStore(rollouts_dir)
    pairs = []
    for traj_id in store.list_traj_ids():
        if not store.has_real(traj_id):
            continue
        try:
            real = store.load_real(traj_id)
            config = store.load_trajectory(traj_id)
            pairs.append((real, config))
        except Exception as exc:
            print(f"  Warning: could not load {traj_id}: {exc}")
    return pairs, store


def _load_rollouts_flat(recordings_dir: str):
    """Load (real_rollout, traj_config) pairs from a flat timestamped directory.

    Expects files named trajectory_TS.json + real_TS.parquet produced by
    collect_dataset.py.  Returns (pairs, timestamps) where timestamps[i] is the
    datetime string for pairs[i].
    """
    import glob
    import pandas as pd

    from elastic_sim.rollout import RolloutResult
    from elastic_sim.trajectory import TrajectoryConfig

    pairs: list[tuple] = []
    timestamps: list[str] = []

    pattern = os.path.join(recordings_dir, "trajectory_*.json")
    for traj_path in sorted(glob.glob(pattern)):
        fname = os.path.basename(traj_path)
        ts = fname[len("trajectory_"):-len(".json")]
        real_path = os.path.join(recordings_dir, f"real_{ts}.parquet")
        if not os.path.exists(real_path):
            continue
        try:
            real = RolloutResult.from_dataframe(pd.read_parquet(real_path))
            config = TrajectoryConfig.load(traj_path)
            pairs.append((real, config))
            timestamps.append(ts)
        except Exception as exc:
            print(f"  Warning: could not load {ts}: {exc}")

    return pairs, timestamps


def _make_optimizer(name: str, max_evals: int):
    if name == "cma":
        from elastic_sim.optimizers.cma_backend import CMAOptimizer
        return CMAOptimizer(sigma0=0.3, max_evals=max_evals)
    if name == "bo":
        from elastic_sim.optimizers.bo_backend import BayesianOptimizer
        return BayesianOptimizer()
    if name == "skrl":
        from elastic_sim.optimizers.skrl_backend import SkrlOptimizer
        return SkrlOptimizer()
    raise ValueError(f"Unknown optimizer: {name}")


def _validate(problem: SimCalibrationProblem, val_rollouts, cut_off_time: float):
    """Report fidelity on the validation set using the current best params."""
    best_theta, best_train_loss = problem.best
    if best_theta is None:
        return
    print(f"\nValidation ({len(val_rollouts)} rollouts):")
    include_payload = len(best_theta) > 6
    params = RobotParams.denormalize(best_theta, include_payload=include_payload)
    # Re-apply best params
    problem._ensure_model(params)
    val_losses = []
    for real_rollout, traj_config in val_rollouts:
        traj = _trajectory_from_config(traj_config)
        sim_rollout = problem._run_sim(traj)
        result = compare(sim_rollout, real_rollout, cut_off_time=cut_off_time)
        val_losses.append(result["metric"])
        print(f"  loss={result['metric']:.6f}  per_axis={result['per_axis']}")
    print(f"  Mean validation loss: {np.mean(val_losses):.6f}")
    print(f"  Mean train loss:      {best_train_loss:.6f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    if args.rollouts_dir is not None and args.recordings_dir is not None:
        print("ERROR: Provide only one of --rollouts-dir or --recordings-dir.")
        sys.exit(1)

    print(f"Backend:   {args.backend}")
    print(f"Optimizer: {args.optimizer}")

    # ---- Flat recordings directory (collect_dataset.py output) ----
    _timestamps: list[str] = []
    if args.recordings_dir is not None:
        print(f"Recordings: {args.recordings_dir}  (flat timestamped format)")
        all_rollouts, _timestamps = _load_rollouts_flat(args.recordings_dir)
        store = None  # no RolloutStore for flat format
        _flat_dir = args.recordings_dir
    else:
        # ---- Structured rollout store (legacy / record_rollouts.py output) ----
        rollouts_dir = args.rollouts_dir or os.path.join(_REPO, "data", "rollouts")
        print(f"Rollouts:  {rollouts_dir}  (structured sub-dir format)")
        all_rollouts, store = _load_rollouts(rollouts_dir, args.backend)
        _flat_dir = None

    if not all_rollouts:
        print("ERROR: No rollouts with real.parquet found. Record the real robot first.")
        sys.exit(1)
    print(f"Found {len(all_rollouts)} real rollout(s).")

    # Train / validation split
    n_train = max(1, int(len(all_rollouts) * args.train_fraction))
    train_rollouts = all_rollouts[:n_train]
    val_rollouts = all_rollouts[n_train:]
    train_timestamps = _timestamps[:n_train] if _timestamps else []
    print(f"Train: {len(train_rollouts)}, Validation: {len(val_rollouts)}")

    # Build calibration problem
    problem = SimCalibrationProblem(
        train_rollouts,
        backend=args.backend,
        noise=False,
        cut_off_time=args.cut_off_time,
        time_step=args.time_step,
    )

    include_payload = not args.no_payload
    n_dims = problem.n_dims(include_payload)
    bounds = RobotParams.bounds(include_payload)

    # Use current settings.yaml as initial guess
    x0 = RobotParams.from_yaml().normalize(include_payload)
    print(f"Parameter dims: {n_dims}  (payload {'included' if include_payload else 'fixed'})")

    # Run optimisation
    optimizer = _make_optimizer(args.optimizer, args.max_evals)
    print(f"\nRunning {args.optimizer.upper()} (max_evals={args.max_evals})...")
    best_theta, history = optimizer.minimize(
        problem.loss, bounds, x0=x0,
        max_evals=args.max_evals,
        verbose=args.verbose,
    )
    best_theta_prob, best_loss = problem.best
    print(f"\nBest loss (train): {best_loss:.6f}  after {len(history)} evaluations")

    # Decode and display best params
    best_params = RobotParams.denormalize(best_theta_prob, include_payload=include_payload)
    print("\nCalibrated parameters:")
    for ax_name, ax in [("x", best_params.drive_x), ("y", best_params.drive_y), ("z", best_params.drive_z)]:
        print(f"  {ax_name}: stiffness={ax.stiffness:.2f} N/m, damping_ratio={ax.damping_ratio:.4f}")
    if include_payload:
        print(f"  payload: {best_params.payload:.3f} kg")

    # Validate if we have held-out data
    if val_rollouts:
        _validate(problem, val_rollouts, args.cut_off_time)

    # Save calibrated params
    output_path = args.output or os.path.join(_REPO, f"calibrated_{args.backend}.yaml")
    best_params.to_yaml(output_path)
    print(f"\nCalibrated params saved to {output_path}")

    # Save best sim rollout for each training trajectory
    print("\nSaving calibrated sim rollouts for all training trajectories...")
    problem._ensure_model(best_params)
    for i, (real_rollout, traj_config) in enumerate(train_rollouts):
        traj = _trajectory_from_config(traj_config)
        sim_rollout = problem._run_sim(traj)
        if _flat_dir is not None:
            # Flat format: save as <backend>_calibrated_TS.parquet alongside the source files
            ts = train_timestamps[i] if i < len(train_timestamps) else f"idx{i:04d}"
            out_path = os.path.join(_flat_dir, f"{args.backend}_calibrated_{ts}.parquet")
            sim_rollout.to_dataframe().to_parquet(out_path, index=False)
            print(f"  Saved {os.path.basename(out_path)}")
        elif store is not None:
            traj_id = f"traj_m{traj_config.mode}_s{traj_config.seed}"
            store.save_sim(traj_id, sim_rollout, backend=args.backend, calibrated=True)
            print(f"  Saved {args.backend}_calibrated.parquet for {traj_id}")

    print("\nCalibration complete.")


if __name__ == "__main__":
    main()
