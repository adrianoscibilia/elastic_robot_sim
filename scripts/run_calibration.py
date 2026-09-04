"""Run all configured sim-to-real calibration optimizers on experiment artifacts."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from elastic_sim.experiment import artifact_root, config_digest, load_experiment_config, resolve_asset, rollout_to_frame  # noqa: E402
from elastic_sim.parameter_registry import ParameterRegistry  # noqa: E402
from elastic_sim.sim2real import (  # noqa: E402
    ReferenceTrajectoryCalibrationProblem,
    discover_experiment_records,
    run_simulation,
    split_records,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", required=True, help="sim-to-real asset configuration")
    parser.add_argument("--methods", nargs="+", choices=("cma", "bo", "skrl", "all"), default=None)
    parser.add_argument("--backends", nargs="+", choices=("newton", "mujoco"), default=None)
    parser.add_argument("--recorded-root", default=None, help="input root; final component must be recorded")
    parser.add_argument("--max-evals", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-root", default=None)
    return parser.parse_args()


def _dependency_check(methods: list[str]) -> None:
    modules = {"cma": "cma", "bo": "skopt", "skrl": "skrl"}
    missing = []
    for method in methods:
        try:
            __import__(modules[method])
        except ImportError:
            missing.append(f"{method} (uv sync --group calibration)")
    if missing:
        raise RuntimeError("requested calibration methods are unavailable: " + ", ".join(missing))


def _optimizer(method: str, config: dict, max_evals: int):
    if method == "cma":
        from elastic_sim.optimizers.cma_backend import CMAOptimizer
        cfg = config.get("calibration", {}).get("cma", {}) or {}
        return CMAOptimizer(float(cfg.get("sigma0", 0.3)), cfg.get("popsize"), float(cfg.get("tolfun", 1.0e-7)), float(cfg.get("tolx", 1.0e-7)))
    if method == "bo":
        from elastic_sim.optimizers.bo_backend import BayesianOptimizer
        cfg = config.get("calibration", {}).get("bo", {}) or {}
        return BayesianOptimizer(int(cfg.get("n_initial_points", min(10, max_evals))), str(cfg.get("acq_func", "EI")), cfg.get("noise", "gaussian"))
    if method == "skrl":
        from elastic_sim.optimizers.skrl_backend import SkrlOptimizer
        cfg = config.get("calibration", {}).get("skrl", {}) or {}
        return SkrlOptimizer(str(cfg.get("algorithm", "ppo")), context_dim=int(cfg.get("context_dim", 4)), max_steps=max_evals, device=str(cfg.get("device", "cpu")))
    raise ValueError(method)


def _write_yaml(path: Path, asset: str, backend: str, params: dict[str, float], registry: ParameterRegistry, validation_loss: float) -> None:
    path.write_text(yaml.safe_dump({
        "asset": asset, "backend": backend, "parameters": params,
        "parameter_registry": registry.to_dict(), "validation_loss": float(validation_loss),
    }, sort_keys=False), encoding="utf-8")


def main() -> int:
    args = _args()
    config = load_experiment_config(args.config)
    asset = resolve_asset(config, args.config)
    cal = config.get("calibration", {}) or {}
    methods = list(args.methods or cal.get("methods", ("cma", "bo", "skrl")))
    if "all" in methods:
        methods = ["cma", "bo", "skrl"]
    methods = list(dict.fromkeys(methods))
    _dependency_check(methods)
    backends = tuple(args.backends or config.get("simulation", {}).get("backends", ("newton", "mujoco")))
    root = artifact_root(config, "recorded", _REPO, args.recorded_root)
    expected_hash = config_digest(config)
    records = discover_experiment_records(root, asset.name, expected_hash)
    if not records:
        raise RuntimeError(f"no complete, hash-compatible real experiment runs found below {root / asset.name}")
    train_fraction = float(cal.get("train_fraction", 0.8))
    train, validation = split_records(records, train_fraction, int(args.seed if args.seed is not None else cal.get("split_seed", 0)))
    if not validation:
        raise RuntimeError("calibration requires at least one held-out validation trajectory; record at least two trajectories")
    registry = ParameterRegistry.from_config(config, asset.joint_names)
    max_evals = int(args.max_evals if args.max_evals is not None else cal.get("max_evals", 100))
    if max_evals < 1:
        raise ValueError("max-evals must be positive")
    output_root = artifact_root(config, "calibrations", _REPO, args.output_root)
    run_dir = output_root / asset.name / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "schema_version": 1, "asset": asset.name, "config_hash": expected_hash,
        "experiment_runs": [str(record.path) for record in records],
        "train_trajectories": len(train), "validation_trajectories": len(validation),
        "methods": methods, "backends": list(backends), "results": {},
        "software": {"python": platform.python_version(), "platform": platform.platform()},
    }
    for backend in backends:
        backend_results = {}
        for method in methods:
            print(f"{backend}/{method}: {len(train)} train trajectories, {len(registry.specs)} parameters")
            problem = ReferenceTrajectoryCalibrationProblem(asset, config, train, backend, registry)
            optimizer = _optimizer(method, config, max_evals)
            start = time.perf_counter()
            _, optimizer_history = optimizer.minimize(problem.loss, registry.bounds, x0=registry.initial, max_evals=max_evals, verbose=False)
            train_theta, train_loss = min(((np.asarray(item["theta"]), float(item["loss"])) for item in problem.history), key=lambda item: item[1])
            best_params = registry.decode(train_theta)
            validation_problem = ReferenceTrajectoryCalibrationProblem(asset, config, validation, backend, registry)
            validation_loss, validation_components = validation_problem.evaluate(train_theta)
            history_path = run_dir / f"history_{backend}_{method}.json"
            history_path.write_text(json.dumps({"optimizer_history": [(np.asarray(theta).tolist(), float(loss)) for theta, loss in optimizer_history], "evaluations": problem.history}, indent=2), encoding="utf-8")
            _write_yaml(run_dir / f"calibrated_{backend}_{method}.yaml", asset.name, backend, best_params, registry, validation_loss)
            backend_results[method] = {"train_loss": train_loss, "validation_loss": validation_loss, "validation_components": validation_components, "parameters": best_params, "history": str(history_path), "runtime_s": time.perf_counter() - start}
        winner = min(backend_results, key=lambda name: backend_results[name]["validation_loss"])
        backend_results["selected"] = winner
        backend_results["selected_parameters"] = backend_results[winner]["parameters"]
        _write_yaml(run_dir / f"calibrated_{backend}.yaml", asset.name, backend, backend_results[winner]["parameters"], registry, backend_results[winner]["validation_loss"])
        validation_frames = []
        for index, (trajectory, _) in enumerate(validation):
            simulated = run_simulation(asset, config, trajectory, backend, backend_results[winner]["parameters"])
            frame = rollout_to_frame(trajectory, simulated, source=f"calibrated_{backend}")
            frame.insert(0, "trajectory_id", index)
            validation_frames.append(frame)
        __import__("pandas").concat(validation_frames, ignore_index=True).to_parquet(
            run_dir / f"validation_{backend}.parquet", index=False
        )
        report["results"][backend] = backend_results
    (run_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Calibration complete: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
