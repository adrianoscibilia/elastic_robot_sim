#!/usr/bin/env python3
"""Calibrate an asset-driven simulator against canonical torque-replay data.

The runner is deliberately injected as ``package.module:factory``.  A robot
configuration only selects an asset and parameter limits; it contains no
robot-specific optimizer or metric code.  A factory must return an object
implementing ``run_torque_replay(params, rollout)`` from generic_calibration.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_REPO, "src"))

from elastic_sim.generic_calibration import (
    ParameterSpace,
    ParameterSpec,
    TorqueReplayCalibrationProblem,
    compare_torque_replay,
    load_torque_replay_rollouts,
)
from elastic_sim.asset_dataset import load_asset_rollouts


def _load_factory(path: str):
    module_name, separator, name = path.partition(":")
    if not separator:
        raise ValueError("runner_factory must have form package.module:factory")
    return getattr(importlib.import_module(module_name), name)


def _make_runner(factory_path: str, config: dict):
    factory = _load_factory(factory_path)
    # The keyword signature is the supported public convention.  The fallback
    # makes early adapters easy to write while preserving a single CLI.
    try:
        return factory(asset=config.get("asset"), config=config)
    except TypeError:
        return factory(config)


def _reports(problem: TorqueReplayCalibrationProblem, theta: np.ndarray) -> list[dict]:
    params = problem.parameter_space.decode(theta)
    return [
        compare_torque_replay(
            problem.runner.run_torque_replay(params, rollout), rollout,
            q_weight=problem.q_weight, dq_weight=problem.dq_weight,
            motor_q_weight=problem.motor_q_weight, motor_dq_weight=problem.motor_dq_weight,
        )
        for rollout in problem.rollouts
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Calibration YAML")
    parser.add_argument("--output", default=None, help="Output JSON report (default from config)")
    parser.add_argument("--max-evals", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    required = ("dataset", "runner_factory", "parameters")
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f"calibration config missing required keys: {missing}")
    # Keep robot configs portable: all resource paths are relative to their
    # YAML, never to the caller's current directory.
    for key in ("asset", "dataset"):
        value = Path(str(cfg[key])).expanduser()
        cfg[key] = str(value if value.is_absolute() else (config_path.parent / value).resolve())
    joint_names = tuple(cfg["joint_names"]) if cfg.get("joint_names") else None
    dataset_format = cfg.get("dataset_format")
    if dataset_format:
        all_rollouts = load_asset_rollouts(
            cfg["dataset"], joint_names=joint_names, dataset_format=str(dataset_format),
            sample_time_s=cfg.get("sample_time_s"),
        )
    else:
        all_rollouts = load_torque_replay_rollouts(cfg["dataset"], joint_names=joint_names)
    fraction = float(cfg.get("train_fraction", 0.8))
    if not 0.0 < fraction <= 1.0:
        raise ValueError("train_fraction must be in (0, 1]")
    n_train = max(1, int(len(all_rollouts) * fraction))
    train, validation = all_rollouts[:n_train], all_rollouts[n_train:]
    specs = [
        ParameterSpec(
            name=item["name"], lower=float(item["lower"]), upper=float(item["upper"]),
            log_scale=bool(item.get("log_scale", True)),
        )
        for item in cfg["parameters"]
    ]
    space = ParameterSpace(specs)
    weights = cfg.get("metric_weights", {})
    runner = _make_runner(cfg["runner_factory"], cfg)
    problem = TorqueReplayCalibrationProblem(
        runner, train, space,
        q_weight=float(weights.get("position", 1.0)),
        dq_weight=float(weights.get("velocity", 0.3)),
        motor_q_weight=float(weights.get("motor_position", 0.0)),
        motor_dq_weight=float(weights.get("motor_velocity", 0.0)),
    )
    optimizer_cfg = cfg.get("optimizer", {})
    name = optimizer_cfg.get("name", "cma")
    max_evals = args.max_evals or int(optimizer_cfg.get("max_evals", 300))
    if name == "cma":
        from elastic_sim.optimizers.cma_backend import CMAOptimizer
        optimizer = CMAOptimizer(sigma0=float(optimizer_cfg.get("sigma0", 0.3)))
    elif name == "bo":
        from elastic_sim.optimizers.bo_backend import BayesianOptimizer
        optimizer = BayesianOptimizer(n_initial_points=int(optimizer_cfg.get("n_initial_points", 15)))
    else:
        raise ValueError("generic dataset calibration currently supports optimizer.name cma or bo")
    initial = cfg.get("initial_params")
    x0 = space.encode(initial) if initial else np.zeros(len(specs))
    print(f"Calibrating {len(specs)} parameters against {len(train)} train trajectory(s) with {name.upper()}")
    _, history = optimizer.minimize(problem.loss, space.bounds, x0=x0, max_evals=max_evals, verbose=args.verbose)
    theta, train_loss = problem.best
    assert theta is not None
    report = {
        "asset": cfg.get("asset"),
        "dataset": cfg["dataset"],
        "runner_factory": cfg["runner_factory"],
        "parameters": space.decode(theta),
        "theta": theta.tolist(),
        "train_loss": train_loss,
        "train_reports": _reports(problem, theta),
        "history": [{"theta": item[0].tolist(), "loss": item[1]} for item in history],
    }
    if validation:
        validation_problem = TorqueReplayCalibrationProblem(runner, validation, space, q_weight=problem.q_weight, dq_weight=problem.dq_weight, motor_q_weight=problem.motor_q_weight, motor_dq_weight=problem.motor_dq_weight)
        reports = _reports(validation_problem, theta)
        report["validation_reports"] = reports
        report["validation_loss"] = float(np.mean([item["metric"] for item in reports]))
    output = Path(args.output or cfg.get("output", "dataset_calibration.json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Best train loss: {train_loss:.6f}")
    if "validation_loss" in report:
        print(f"Validation loss: {report['validation_loss']:.6f}")
    print(f"Saved calibration report to {output}")


if __name__ == "__main__":
    main()
