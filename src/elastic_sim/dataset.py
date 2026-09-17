"""Identification dataset generation and export.

One *condition* is a choice of model tier (rigid or a transmission stiffness),
a friction sample and a simulator backend.  One *bag* is one excitation
trajectory executed under one condition.  The generator writes the
repository's canonical wide parquet for every bag; the exporter flattens a run
into the single CSV that ``dynamic_model_nn`` consumes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from .assets import AssetSpec
from .excitation import FourierExcitationConfig, optimize_excitation
from .identification import FrictionModel
from .materialized import MaterializedTrajectory
from .torque_runners import (
    ComputedTorqueController,
    SeaMotorController,
    TransmissionSpec,
    run_mujoco_elastic_torque,
    run_mujoco_torque,
    run_newton_elastic_torque,
    run_newton_torque,
)

RIGID_TIER = "rigid"


@dataclass(frozen=True)
class Tier:
    """One model-fidelity level of the dataset."""

    name: str
    stiffness: float | None = None
    transmission_damping_ratio: float = 0.1
    rotor_inertia: float = 0.1

    @property
    def is_rigid(self) -> bool:
        return self.stiffness is None

    def transmission(self, n_dof: int) -> TransmissionSpec | None:
        if self.is_rigid:
            return None
        return TransmissionSpec.uniform(
            n_dof, float(self.stiffness),
            damping_ratio=self.transmission_damping_ratio,
            rotor_inertia=self.rotor_inertia,
        )


def default_tiers(stiffnesses: Sequence[float] = (1.0e6, 2.0e5, 4.0e4, 1.0e4)) -> tuple[Tier, ...]:
    """Rigid reference plus a stiffness ladder from near-rigid to compliant.

    The stiffest tier exists to be checked against the rigid one: if it does
    not reproduce it, the elastic path is wrong and nothing below it can be
    trusted.
    """
    return (Tier(RIGID_TIER), *(Tier(f"k{value:.0e}".replace("+", ""), stiffness=value) for value in stiffnesses))


DEFAULT_CONFIG_DIR = "config/identification"
DEFAULT_CONFIG = "config/identification/kuka_lbr_iiwa_14_r820_table.yaml"


@dataclass(frozen=True)
class DatasetConfig:
    """Everything that defines a dataset build."""

    asset: str = "kuka_lbr_iiwa_14_r820_table"
    backends: tuple[str, ...] = ("mujoco", "newton")
    tiers: tuple[Tier, ...] = field(default_factory=default_tiers)
    n_trajectories: int = 4
    n_friction_samples: int = 2
    friction_scale_range: tuple[float, float] = (0.5, 2.0)
    seed: int = 20260917
    excitation: FourierExcitationConfig = field(default_factory=FourierExcitationConfig)
    candidates: int = 48
    control_frequency: float = 25.0
    control_damping_ratio: float = 1.0
    rigid_time_step: float = 5.0e-4
    sample_time_step: float = 0.002
    max_time_step: float = 5.0e-4
    output: str = "data/identification/dataset.csv"
    visualize: bool = False
    realtime_scale: float = 1.0


def load_config(path: str | Path) -> DatasetConfig:
    """Read a YAML identification config into a :class:`DatasetConfig`."""
    import yaml

    source = Path(path).expanduser()
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"identification config must be a mapping: {source}")
    known = {"schema_version", "asset", "backends", "tiers", "excitation",
             "dataset", "simulation", "visualization"}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"unknown keys in {source}: {', '.join(unknown)}")

    tier_cfg = raw.get("tiers", {}) or {}
    stiffness = [float(value) for value in tier_cfg.get("stiffness", []) or []]
    tiers = default_tiers(stiffness)
    if not tier_cfg.get("rigid", True):
        tiers = tuple(tier for tier in tiers if not tier.is_rigid)
    tiers = tuple(
        tier if tier.is_rigid else Tier(
            tier.name, tier.stiffness,
            transmission_damping_ratio=float(tier_cfg.get("transmission_damping_ratio", 0.1)),
            rotor_inertia=float(tier_cfg.get("rotor_inertia", 0.1)),
        )
        for tier in tiers
    )
    if not tiers:
        raise ValueError(f"{source} selects no tiers: enable rigid or list stiffness values")

    exc_cfg = raw.get("excitation", {}) or {}
    sim_cfg = raw.get("simulation", {}) or {}
    data_cfg = raw.get("dataset", {}) or {}
    view_cfg = raw.get("visualization", {}) or {}
    scale = data_cfg.get("friction_scale", [0.5, 2.0])
    return DatasetConfig(
        asset=str(raw.get("asset", "kuka_lbr_iiwa_14_r820_table")),
        backends=tuple(raw.get("backends", ["mujoco"])),
        tiers=tiers,
        n_trajectories=int(data_cfg.get("trajectories", 4)),
        n_friction_samples=int(data_cfg.get("friction_samples", 2)),
        friction_scale_range=(float(scale[0]), float(scale[1])),
        seed=int(data_cfg.get("seed", 20260917)),
        excitation=FourierExcitationConfig(
            n_harmonics=int(exc_cfg.get("n_harmonics", 5)),
            base_frequency=float(exc_cfg.get("base_frequency", 0.1)),
            n_periods=int(exc_cfg.get("n_periods", 1)),
            time_step=float(sim_cfg.get("sample_time_step", 0.002)),
            limit_margin=float(exc_cfg.get("limit_margin", 0.12)),
            max_acceleration=float(exc_cfg.get("max_acceleration", 3.0)),
            velocity_fraction=float(exc_cfg.get("velocity_fraction", 0.6)),
        ),
        candidates=int(exc_cfg.get("candidates", 48)),
        control_frequency=float(sim_cfg.get("control_frequency", 25.0)),
        control_damping_ratio=float(sim_cfg.get("control_damping_ratio", 1.0)),
        rigid_time_step=float(sim_cfg.get("rigid_time_step", 5.0e-4)),
        sample_time_step=float(sim_cfg.get("sample_time_step", 0.002)),
        max_time_step=float(sim_cfg.get("max_time_step", 5.0e-4)),
        output=str(data_cfg.get("output", "data/identification/dataset.csv")),
        visualize=bool(view_cfg.get("enabled", False)),
        realtime_scale=float(view_cfg.get("realtime_scale", 1.0)),
    )


def sample_friction(base: FrictionModel, rng: np.random.Generator, scale_range: tuple[float, float]) -> FrictionModel:
    """Scale each joint's viscous and Coulomb coefficient log-uniformly."""
    low, high = np.log(scale_range[0]), np.log(scale_range[1])
    viscous = base.viscous * np.exp(rng.uniform(low, high, size=base.n_dof))
    coulomb = base.coulomb * np.exp(rng.uniform(low, high, size=base.n_dof))
    return FrictionModel(viscous, coulomb)


def run_condition(
    asset: AssetSpec,
    trajectory: MaterializedTrajectory,
    tier: Tier,
    backend: str,
    friction: FrictionModel,
    config: DatasetConfig,
) -> dict[str, Any]:
    """Execute one trajectory under one condition and return the rollout."""
    n_dof = len(asset.joint_names)
    transmission = tier.transmission(n_dof)
    view = {"visualize": config.visualize, "realtime_scale": config.realtime_scale}
    if transmission is None:
        controller = ComputedTorqueController(
            asset, trajectory, friction=friction,
            natural_frequency=config.control_frequency,
            damping_ratio=config.control_damping_ratio,
        )
        runner = run_mujoco_torque if backend == "mujoco" else run_newton_torque
        return runner(asset, trajectory, controller, time_step=config.rigid_time_step,
                      friction=friction, **view)
    time_step = min(config.max_time_step, transmission.required_time_step())
    controller = SeaMotorController(
        asset, trajectory, transmission, friction=friction,
        natural_frequency=config.control_frequency,
        damping_ratio=config.control_damping_ratio,
    )
    runner = run_mujoco_elastic_torque if backend == "mujoco" else run_newton_elastic_torque
    return runner(asset, trajectory, controller, transmission, time_step=time_step,
                  friction=friction, **view)


def iter_conditions(config: DatasetConfig) -> Iterator[tuple[int, Tier, int, str]]:
    """Yield ``(trajectory_index, tier, friction_index, backend)`` in bag order.

    Trajectory index varies slowest and backend fastest, so consecutive bags
    differ by condition rather than by regime.  ``dynamic_model_nn`` splits
    train/test as a contiguous half of the file, and a dataset ordered by tier
    would put whole tiers on one side of that split.
    """
    for trajectory_index in range(config.n_trajectories):
        for friction_index in range(config.n_friction_samples):
            for tier in config.tiers:
                for backend in config.backends:
                    yield trajectory_index, tier, friction_index, backend


def rollout_frame(
    asset: AssetSpec,
    trajectory: MaterializedTrajectory,
    result: Mapping[str, Any],
    *,
    bag: str,
    tier: Tier,
    backend: str,
    friction: FrictionModel,
    resample_step: float,
) -> pd.DataFrame:
    """Flatten one rollout onto a uniform grid in the consumer's schema.

    ``dynamic_model_nn`` differentiates positions with a Savitzky-Golay filter
    and only does so when the time step is uniform, so every bag is resampled
    onto the same grid regardless of the step its tier required.
    """
    names = tuple(asset.joint_names)
    time = np.asarray(result["time"], dtype=float)
    grid = np.arange(time[0], time[-1] + 0.5 * resample_step, resample_step)

    def _on_grid(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        return np.column_stack([np.interp(grid, time, values[:, i]) for i in range(values.shape[1])])

    q = _on_grid(result["q_link"])
    dq = _on_grid(result["dq_link"])
    q_motor = _on_grid(result["q_motor"])
    dq_motor = _on_grid(result["dq_motor"])
    tau_motor = _on_grid(result["tau_motor"])
    tau_link = _on_grid(result["tau_link"])

    frame = pd.DataFrame({"t": grid, "bag": bag})
    for index in range(len(names)):
        frame[f"q{index}"] = q[:, index]
    for index in range(len(names)):
        frame[f"dq{index}"] = dq[:, index]
    for index in range(len(names)):
        frame[f"tau{index}"] = tau_motor[:, index]
    # The learning target: link-side generalized force, one channel per joint.
    for index in range(len(names)):
        frame[f"ft{index}"] = tau_link[:, index]
    for index in range(len(names)):
        frame[f"q_motor{index}"] = q_motor[:, index]
        frame[f"dq_motor{index}"] = dq_motor[:, index]
        frame[f"q_link{index}"] = q[:, index]
        frame[f"dq_link{index}"] = dq[:, index]
    frame["experiment"] = bag
    frame["tier"] = tier.name
    frame["backend"] = backend
    frame["stiffness"] = np.nan if tier.is_rigid else float(tier.stiffness)
    for index, name in enumerate(names):
        frame[f"viscous__{name}"] = friction.viscous[index]
        frame[f"coulomb__{name}"] = friction.coulomb[index]
    return frame


def generate(config: DatasetConfig, asset: AssetSpec, *, verbose: bool = True) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the whole dataset and return ``(frame, manifest)``."""
    base_friction = FrictionModel.from_asset(asset)
    rng = np.random.default_rng(config.seed)
    frictions = [base_friction] + [
        sample_friction(base_friction, rng, config.friction_scale_range)
        for _ in range(max(0, config.n_friction_samples - 1))
    ]
    trajectories: dict[int, MaterializedTrajectory] = {}
    for index in range(config.n_trajectories):
        trajectories[index] = optimize_excitation(
            asset, config.excitation, seed=config.seed + index, n_candidates=config.candidates
        )
        if verbose:
            print(f"trajectory {index}: condition={trajectories[index].metadata['condition_number']:.0f}")

    frames: list[pd.DataFrame] = []
    records: list[dict[str, Any]] = []
    for bag_index, (traj_index, tier, friction_index, backend) in enumerate(iter_conditions(config)):
        trajectory = trajectories[traj_index]
        friction = frictions[friction_index]
        bag = f"t{traj_index}_{tier.name}_f{friction_index}_{backend}"
        result = run_condition(asset, trajectory, tier, backend, friction, config)
        frames.append(rollout_frame(asset, trajectory, result, bag=bag, tier=tier, backend=backend,
                                    friction=friction, resample_step=config.sample_time_step))
        records.append({
            "bag": bag, "bag_index": bag_index, "trajectory": traj_index, "tier": tier.name,
            "stiffness": None if tier.is_rigid else float(tier.stiffness),
            "friction_index": friction_index, "backend": backend,
            "solver": result.get("solver"), "wall_time": float(result.get("wall_time", 0.0)),
            "samples": int(len(result["time"])),
            "trajectory_digest": trajectory.digest(),
            "condition_number": float(trajectory.metadata["condition_number"]),
            "viscous": friction.viscous.tolist(), "coulomb": friction.coulomb.tolist(),
            "tracking_rms": float(np.sqrt(np.mean((np.asarray(result["q_link"]) - np.asarray(result["q_ref"])) ** 2))),
            "feedback_ratio": float(
                np.mean(np.abs(result["tau_feedback"])) / max(np.mean(np.abs(result["tau_feedforward"])), 1e-12)
            ),
        })
        if verbose:
            print(f"  [{bag_index + 1}] {bag:34s} {records[-1]['wall_time']:6.1f}s "
                  f"trk={records[-1]['tracking_rms']:.2e}")

    frame = pd.concat(frames, ignore_index=True)
    manifest = {
        "asset": asset.name,
        "n_bags": len(records),
        "n_samples": int(len(frame)),
        "n_dof": len(asset.joint_names),
        "joint_names": list(asset.joint_names),
        "sample_time_step": config.sample_time_step,
        "seed": config.seed,
        "backends": list(config.backends),
        "tiers": [t.name for t in config.tiers],
        "target": "ft0..ft{n-1} = link-side joint torque [Nm]",
        "input": "q0..q{n-1} [rad], dq0..dq{n-1} [rad/s], tau0..tau{n-1} = applied motor torque [Nm]",
        "records": records,
    }
    return frame, manifest


def write_dataset(frame: pd.DataFrame, manifest: Mapping[str, Any], output: str | Path) -> tuple[Path, Path]:
    """Write the flat CSV plus its manifest next to it."""
    csv_path = Path(output).expanduser().resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    manifest_path = csv_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(dict(manifest), indent=2), encoding="utf-8")
    return csv_path, manifest_path
