"""Unified sim-to-real simulation and calibration workflow."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from .assets import AssetSpec
from .experiment import rollout_to_frame
from .materialized import MaterializedTrajectory
from .parameter_registry import ParameterRegistry, apply_parameter_overrides


def model_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize the model subsection while accepting useful YAML defaults."""
    value = copy.deepcopy(dict(config.get("model", {})))
    if not isinstance(value, dict):
        raise ValueError("model must be a mapping")
    for key in ("transmissions", "default_stiffness", "default_damping", "motor_stiffness", "motor_damping", "intermediate_mass", "intermediate_size"):
        if key not in value and key in config:
            value[key] = copy.deepcopy(config[key])
    return value


def _relative_path(config: Mapping[str, Any], value: str | None) -> str | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    source = config.get("_config_path")
    return str((Path(source).resolve().parent / path).resolve()) if source else str(path)


def _fmrr_params(config: Mapping[str, Any], params: Mapping[str, float]) -> Any:
    from .params import EFFECTIVE_AXIS_MASS, RobotParams

    mc = model_config(config)
    base_path = _relative_path(config, mc.get("base_params"))
    base = RobotParams.from_yaml(base_path)
    axes = {"joint_x": "x", "joint_y": "y", "joint_z": "z"}
    for name, value in params.items():
        parts = name.split(".")
        if len(parts) != 3 or parts[0] != "transmission" or parts[1] not in axes:
            raise ValueError(f"FMRR model does not support parameter {name!r}")
        axis = axes[parts[1]]
        current = base.axis(axis)
        if parts[2] == "stiffness":
            current.stiffness = float(value)
        elif parts[2] == "damping":
            current.damping_ratio = float(value) / (2.0 * np.sqrt(current.stiffness * EFFECTIVE_AXIS_MASS[axis]))
        elif parts[2] == "damping_ratio":
            current.damping_ratio = float(value)
        elif parts[2] == "motor_stiffness":
            base.motor_stiffness = float(value)
        elif parts[2] == "motor_damping":
            base.motor_damping = float(value)
        elif parts[2] == "intermediate_mass":
            base.intermediate_mass = float(value)
        else:
            raise ValueError(f"FMRR model does not support parameter {name!r}")
    return base


def run_simulation(
    asset: AssetSpec,
    config: Mapping[str, Any],
    trajectory: MaterializedTrajectory,
    backend: str,
    params: Mapping[str, float] | None = None,
) -> Mapping[str, Any]:
    """Run one exact materialized trajectory on one configured backend."""
    params = dict(params or {})
    sim_cfg = config.get("simulation", {}) or {}
    time_step = float(sim_cfg.get("time_step", np.median(np.diff(trajectory.time))))
    if time_step <= 0.0:
        raise ValueError("simulation.time_step must be positive")
    if asset.name == "fmrr_tecnobody":
        if backend == "newton":
            from .sim_runner import build_model, run_rollout
            model_params = _fmrr_params(config, params)
            model, dof_map, _ = build_model(model_params, urdf_path=str(asset.urdf_path))
            return run_rollout(model, dof_map, trajectory, noise=False, time_step=time_step)
        if backend == "mujoco":
            from .mujoco_runner import build_model, run_rollout
            model_params = _fmrr_params(config, params)
            model, data, dof_map, act_map = build_model(model_params, time_step=time_step)
            return run_rollout(model, data, dof_map, act_map, trajectory, noise=False)
        raise ValueError(f"unsupported backend {backend!r}")

    effective_config = apply_parameter_overrides(config, params)
    mc = model_config(effective_config)
    mode = str(mc.get("mode", "elastic")).lower()
    if backend == "newton":
        from .generic_newton_runner import (
            GenericNewtonKinematicTrajectoryRunner,
            GenericNewtonRigidTrajectoryRunner,
            GenericNewtonTrajectoryRunner,
        )
        runner_type = {"elastic": GenericNewtonTrajectoryRunner, "rigid": GenericNewtonRigidTrajectoryRunner, "kinematic": GenericNewtonKinematicTrajectoryRunner}.get(mode)
    elif backend == "mujoco":
        from .generic_mujoco_runner import GenericMujocoTrajectoryRunner
        runner_type = GenericMujocoTrajectoryRunner
    else:
        raise ValueError(f"unsupported backend {backend!r}")
    if runner_type is None:
        raise ValueError(f"unsupported model.mode {mode!r}")
    runner = runner_type(asset, mc) if backend == "newton" else runner_type(asset, mc, mode=mode)
    return runner.run(trajectory, time_step=time_step, visualize=False)


def _column_matrix(frame: pd.DataFrame, prefix: str, names: Sequence[str]) -> np.ndarray | None:
    labels = tuple("xyz") if prefix in {"force_flange", "torque_flange", "force_link_side", "torque_link_side"} else tuple(names)
    columns = [f"{prefix}__{name}" for name in labels]
    if not all(column in frame.columns for column in columns):
        return None
    return frame[columns].to_numpy(float)


def _resample(values: np.ndarray, source_time: np.ndarray, target_time: np.ndarray) -> np.ndarray:
    return np.column_stack([np.interp(target_time, source_time, values[:, i]) for i in range(values.shape[1])])


def _nrmse(prediction: np.ndarray, reference: np.ndarray) -> float:
    scale = np.maximum(np.percentile(np.abs(reference), 95, axis=0), 1.0e-8)
    return float(np.mean(np.sqrt(np.mean((prediction - reference) ** 2, axis=0)) / scale))


class ReferenceTrajectoryCalibrationProblem:
    """Loss over paired materialized trajectories and normalized real frames."""

    def __init__(self, asset: AssetSpec, config: Mapping[str, Any], records: Sequence[tuple[MaterializedTrajectory, pd.DataFrame]], backend: str, registry: ParameterRegistry):
        if not records:
            raise ValueError("at least one complete trajectory/real recording is required")
        self.asset, self.config, self.records = asset, config, tuple(records)
        self.backend, self.registry = backend, registry
        loss_cfg = config.get("calibration", {}).get("loss_weights", {}) or {}
        self.weights = {
            "position": float(loss_cfg.get("position", 1.0)),
            "velocity": float(loss_cfg.get("velocity", 0.3)),
            "effort": float(loss_cfg.get("effort", 0.1)),
            "force": float(loss_cfg.get("force", 0.1)),
            "torque": float(loss_cfg.get("torque", 0.1)),
            "motor_state": float(loss_cfg.get("motor_state", 0.0)),
        }
        self.history: list[dict[str, Any]] = []

    def evaluate(self, theta: Sequence[float], *, records: Sequence[tuple[MaterializedTrajectory, pd.DataFrame]] | None = None) -> tuple[float, dict[str, float]]:
        named = self.registry.decode(theta)
        components: dict[str, list[float]] = {key: [] for key in self.weights}
        start = time.perf_counter()
        failure = None
        try:
            for trajectory, real in records or self.records:
                simulated = rollout_to_frame(trajectory, run_simulation(self.asset, self.config, trajectory, self.backend, named), source=f"sim_{self.backend}")
                sim_time = simulated["t"].to_numpy(float)
                real_time = real["t"].to_numpy(float)
                for key, sim_prefix, real_prefix in (
                    ("position", "q", "q"), ("velocity", "dq", "dq"),
                    ("effort", "tau_motor", "tau_joint_state"),
                    ("force", "force_link_side", "force_link_side"),
                    ("torque", "torque_link_side", "torque_link_side"),
                    ("motor_state", "q_motor", "q_motor"),
                ):
                    sim_values = _column_matrix(simulated, sim_prefix, trajectory.joint_names)
                    real_values = _column_matrix(real, real_prefix, trajectory.joint_names)
                    if sim_values is None and key in {"force", "torque"}:
                        sim_values = _column_matrix(simulated, sim_prefix.replace("_link_side", "_flange"), trajectory.joint_names)
                    if real_values is None and key in {"force", "torque"}:
                        real_values = _column_matrix(real, real_prefix.replace("_link_side", "_flange"), trajectory.joint_names)
                    if sim_values is None or real_values is None or self.weights[key] == 0.0:
                        continue
                    components[key].append(_nrmse(_resample(sim_values, sim_time, real_time), real_values))
            means = {key: float(np.mean(value)) for key, value in components.items() if value}
            active = [(self.weights[key], value) for key, value in means.items() if self.weights[key] > 0.0]
            if not active:
                raise ValueError("no configured loss signal is present in the real recording")
            loss = float(sum(weight * value for weight, value in active) / sum(weight for weight, _ in active))
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
            means = {}
            loss = 1.0e12
        self.history.append({"theta": np.asarray(theta, dtype=float).tolist(), "parameters": named, "loss": loss, "components": means, "runtime_s": time.perf_counter() - start, "error": failure})
        return loss, means

    def loss(self, theta: np.ndarray) -> float:
        return self.evaluate(theta)[0]


@dataclass(frozen=True)
class ExperimentRecord:
    path: Path
    trajectories: tuple[tuple[MaterializedTrajectory, pd.DataFrame], ...]
    manifest: Mapping[str, Any]


def discover_experiment_records(root: str | Path, asset_name: str, config_hash: str | None = None) -> list[ExperimentRecord]:
    """Discover only complete, hash-compatible paired experiment runs."""
    root = Path(root).expanduser().resolve()
    records: list[ExperimentRecord] = []
    for manifest_path in sorted(root.joinpath(asset_name).rglob("manifest.yaml")) if (root / asset_name).exists() else ():
        run_dir = manifest_path.parent
        try:
            manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
            if manifest.get("completion_status") != "complete":
                continue
            if config_hash and manifest.get("config_hash") != config_hash:
                continue
            real_path = run_dir / "real.parquet"
            if not real_path.exists():
                continue
            real = pd.read_parquet(real_path)
            trajectory_paths = sorted((run_dir / "trajectories").glob("trajectory_*.json")) if (run_dir / "trajectories").exists() else []
            if not trajectory_paths and (run_dir / "trajectory.json").exists():
                trajectory_paths = [run_dir / "trajectory.json"]
            if not trajectory_paths:
                continue
            pairs = []
            for index, path in enumerate(trajectory_paths):
                trajectory = MaterializedTrajectory.load(path)
                subset = real[real["trajectory_id"].astype(str) == str(index)].copy() if "trajectory_id" in real else real.copy()
                if len(subset) >= 2:
                    pairs.append((trajectory, subset.reset_index(drop=True)))
            if pairs:
                records.append(ExperimentRecord(run_dir, tuple(pairs), manifest))
        except Exception:
            continue
    return records


def split_records(records: Sequence[ExperimentRecord], train_fraction: float, seed: int) -> tuple[list[tuple[MaterializedTrajectory, pd.DataFrame]], list[tuple[MaterializedTrajectory, pd.DataFrame]]]:
    items = [pair for record in records for pair in record.trajectories]
    if not 0.0 < train_fraction <= 1.0:
        raise ValueError("train_fraction must be in (0, 1]")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(items))
    n_train = max(1, int(np.floor(len(items) * train_fraction)))
    train = [items[int(index)] for index in order[:n_train]]
    validation = [items[int(index)] for index in order[n_train:]]
    return train, validation


def configuration_hash(config: Mapping[str, Any]) -> str:
    value = {key: item for key, item in config.items() if not str(key).startswith("_")}
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode()).hexdigest()
