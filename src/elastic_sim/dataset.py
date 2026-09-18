"""Identification dataset generation and export.

One *condition* is a choice of model tier (the rigid reference or one sampled
elastic robot), a friction sample and a simulator backend.  One *bag* is one
excitation trajectory executed under one condition.  The exporter flattens a
run into the single CSV that ``dynamic_model_nn`` consumes.

Elastic robots are sampled rather than laddered: every joint draws its own
transmission stiffness log-uniformly from a physically plausible interval and
its own damping ratio uniformly, and the damping coefficient is derived from
both and the joint's inertia, so the config never states a damping value.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from .assets import AssetSpec
from .backend_comparison import ComparisonThresholds, compare_backends, format_report, summarize
from .excitation import FourierExcitationConfig, optimize_excitation
from .identification import FrictionModel
from .kinematics import PortableKinematics
from .materialized import MaterializedTrajectory
from .torque_runners import (
    ComputedTorqueController,
    SeaMotorController,
    TransmissionSpec,
    link_inertia_envelope,
    run_mujoco_elastic_torque,
    run_mujoco_torque,
    run_newton_elastic_torque,
    run_newton_torque,
)

RIGID_TIER = "rigid"

# (median, minimum) link-side inertia per joint, see ``link_inertia_envelope``.
LinkInertia = tuple[np.ndarray, np.ndarray]


def _per_joint(values: Sequence[float], n_dof: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if len(array) not in (1, n_dof):
        raise ValueError(f"{name} has {len(array)} values; expected 1 or {n_dof}")
    return np.broadcast_to(array, (n_dof,)).copy()


@dataclass(frozen=True)
class Tier:
    """One model-fidelity level: the rigid reference or one elastic robot.

    Elastic values are per joint; a single value applies to every joint.
    """

    name: str
    stiffness: tuple[float, ...] | None = None
    damping_ratio: tuple[float, ...] = (0.1,)
    rotor_inertia: tuple[float, ...] = (0.1,)

    def __post_init__(self) -> None:
        for name in ("stiffness", "damping_ratio", "rotor_inertia"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, tuple(float(v) for v in np.atleast_1d(np.asarray(value, dtype=float))))

    @property
    def is_rigid(self) -> bool:
        return self.stiffness is None

    def transmission(self, n_dof: int, link_inertia: LinkInertia | None = None) -> TransmissionSpec | None:
        """Build the transmission; without ``link_inertia`` damping uses the rotor alone."""
        if self.is_rigid:
            return None
        stiffness = _per_joint(self.stiffness, n_dof, "stiffness")
        zeta = _per_joint(self.damping_ratio, n_dof, "damping_ratio")
        rotor = _per_joint(self.rotor_inertia, n_dof, "rotor_inertia")
        if link_inertia is None:
            return TransmissionSpec(stiffness, 2.0 * zeta * np.sqrt(stiffness * rotor), rotor, damping_ratio=zeta)
        nominal, floor = link_inertia
        return TransmissionSpec.from_damping_ratio(stiffness, zeta, rotor, nominal, floor)


@dataclass(frozen=True)
class TransmissionSampling:
    """How elastic robots are drawn.

    ``stiffness`` holds one ``(min, max)`` interval per joint [Nm/rad], sampled
    log-uniformly so a wide interval is not dominated by its stiff end.
    ``damping_ratio`` is one ``(min, max)`` interval shared by all joints and
    sampled uniformly per joint.  ``rotor_inertia`` is the reflected rotor
    inertia [kg m^2], one value or one per joint; it is not sampled.
    """

    robots: int = 0
    stiffness: tuple[tuple[float, float], ...] = ()
    damping_ratio: tuple[float, float] = (0.05, 0.2)
    rotor_inertia: tuple[float, ...] = (0.1,)
    inertia_samples: int = 512

    def __post_init__(self) -> None:
        if self.robots < 0 or self.inertia_samples < 1:
            raise ValueError("transmission.robots must be >= 0 and inertia_samples positive")
        if self.robots and not self.stiffness:
            raise ValueError("transmission.stiffness needs one [min, max] interval per joint")
        for low, high in self.stiffness:
            if not 0.0 < low <= high:
                raise ValueError(f"stiffness interval [{low}, {high}] must satisfy 0 < min <= max")
        low, high = self.damping_ratio
        if not 0.0 <= low <= high:
            raise ValueError(f"damping_ratio interval [{low}, {high}] must satisfy 0 <= min <= max")
        if len(self.rotor_inertia) not in (1, max(1, len(self.stiffness))) or min(self.rotor_inertia) <= 0.0:
            raise ValueError("rotor_inertia needs one positive value or one per joint")


def sample_robots(sampling: TransmissionSampling, seed: int) -> tuple[Tier, ...]:
    """Draw ``sampling.robots`` elastic robots named ``e00, e01, ...``.

    Robots are drawn in order from their own random stream, so a robot's
    parameters depend only on the seed and its index: raising ``robots`` adds
    robots without changing the existing ones.
    """
    rng = np.random.default_rng((int(seed), 1))
    intervals = np.asarray(sampling.stiffness, dtype=float).reshape(-1, 2)
    robots = []
    for index in range(sampling.robots):
        stiffness = np.exp(rng.uniform(np.log(intervals[:, 0]), np.log(intervals[:, 1])))
        zeta = rng.uniform(sampling.damping_ratio[0], sampling.damping_ratio[1], size=len(intervals))
        robots.append(Tier(f"e{index:02d}", stiffness=tuple(stiffness), damping_ratio=tuple(zeta),
                           rotor_inertia=sampling.rotor_inertia))
    return tuple(robots)


def build_tiers(rigid_reference: bool, sampling: TransmissionSampling, seed: int) -> tuple[Tier, ...]:
    """The rigid reference, if kept, followed by the sampled elastic robots."""
    tiers = ((Tier(RIGID_TIER),) if rigid_reference else ()) + sample_robots(sampling, seed)
    if not tiers:
        raise ValueError("no tiers selected: keep the rigid reference or sample at least one robot")
    return tiers


DEFAULT_CONFIG_DIR = "config/identification"
DEFAULT_CONFIG = "config/identification/kuka_lbr_iiwa_14_r820_table.yaml"


@dataclass(frozen=True)
class DatasetConfig:
    """Everything that defines a dataset build.

    ``tiers`` is derived from ``rigid_reference``, ``transmission`` and
    ``seed``; rebuild it with :func:`build_tiers` after changing any of them.
    """

    asset: str = "kuka_lbr_iiwa_14_r820_table"
    backends: tuple[str, ...] = ("mujoco", "newton")
    rigid_reference: bool = True
    transmission: TransmissionSampling = field(default_factory=TransmissionSampling)
    tiers: tuple[Tier, ...] = (Tier(RIGID_TIER),)
    comparison: ComparisonThresholds = field(default_factory=ComparisonThresholds)
    n_trajectories: int = 4
    trajectories_per_robot: bool = False
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


_TRANSMISSION_KEYS = {"robots", "stiffness", "damping_ratio", "rotor_inertia", "inertia_samples"}


def load_config(path: str | Path) -> DatasetConfig:
    """Read a YAML identification config into a :class:`DatasetConfig`."""
    import yaml

    source = Path(path).expanduser()
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"identification config must be a mapping: {source}")
    if "tiers" in raw:
        raise ValueError(
            f"{source}: the `tiers` ladder was replaced by `rigid_reference` and a sampled "
            "`transmission` block (see config/identification/kuka_lbr_iiwa_14_r820_table.yaml)"
        )
    known = {"schema_version", "asset", "backends", "rigid_reference", "transmission", "comparison",
             "excitation", "dataset", "simulation", "visualization"}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"unknown keys in {source}: {', '.join(unknown)}")

    tr_cfg = raw.get("transmission", {}) or {}
    unknown = sorted(set(tr_cfg) - _TRANSMISSION_KEYS)
    if unknown:
        raise ValueError(f"unknown transmission keys in {source}: {', '.join(unknown)}")
    zeta = tr_cfg.get("damping_ratio", [0.05, 0.2])
    sampling = TransmissionSampling(
        robots=int(tr_cfg.get("robots", 0)),
        stiffness=tuple((float(low), float(high)) for low, high in tr_cfg.get("stiffness", []) or []),
        damping_ratio=(float(zeta[0]), float(zeta[1])),
        rotor_inertia=tuple(float(value) for value in np.atleast_1d(tr_cfg.get("rotor_inertia", 0.1))),
        inertia_samples=int(tr_cfg.get("inertia_samples", 512)),
    )

    exc_cfg = raw.get("excitation", {}) or {}
    sim_cfg = raw.get("simulation", {}) or {}
    data_cfg = raw.get("dataset", {}) or {}
    view_cfg = raw.get("visualization", {}) or {}
    scale = data_cfg.get("friction_scale", [0.5, 2.0])
    seed = int(data_cfg.get("seed", 20260917))
    rigid_reference = bool(raw.get("rigid_reference", True))
    try:
        tiers = build_tiers(rigid_reference, sampling, seed)
    except ValueError as exc:
        raise ValueError(f"{source}: {exc}") from exc
    return DatasetConfig(
        asset=str(raw.get("asset", "kuka_lbr_iiwa_14_r820_table")),
        backends=tuple(raw.get("backends", ["mujoco"])),
        rigid_reference=rigid_reference,
        transmission=sampling,
        tiers=tiers,
        comparison=ComparisonThresholds.from_mapping(raw.get("comparison")),
        n_trajectories=int(data_cfg.get("trajectories", 4)),
        trajectories_per_robot=bool(data_cfg.get("trajectories_per_robot", False)),
        n_friction_samples=int(data_cfg.get("friction_samples", 2)),
        friction_scale_range=(float(scale[0]), float(scale[1])),
        seed=seed,
        excitation=FourierExcitationConfig(
            n_harmonics=int(exc_cfg.get("n_harmonics", 5)),
            base_frequency=float(exc_cfg.get("base_frequency", 0.1)),
            n_periods=int(exc_cfg.get("n_periods", 1)),
            time_step=float(sim_cfg.get("sample_time_step", 0.002)),
            limit_margin=float(exc_cfg.get("limit_margin", 0.12)),
            max_acceleration=float(exc_cfg.get("max_acceleration", 3.0)),
            velocity_fraction=float(exc_cfg.get("velocity_fraction", 0.6)),
            centre_jitter=float(exc_cfg.get("centre_jitter", 0.0)),
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


def trajectory_seed(config: DatasetConfig, tier: Tier, index: int) -> int:
    """Seed of trajectory ``index``, per robot when they are not shared.

    Both scripts derive it the same way, so ``--tier e03`` reproduces the
    trajectory that robot ran in the dataset.  The per-robot offset comes from
    the robot's *name*, not its position, so dropping the rigid tier or adding
    robots leaves the others' trajectories untouched.
    """
    if not config.trajectories_per_robot:
        return config.seed + index
    digest = hashlib.blake2b(tier.name.encode("utf-8"), digest_size=4).digest()
    return config.seed + index + 1000 * (1 + int.from_bytes(digest, "big") % 100_000)


def sample_friction(base: FrictionModel, rng: np.random.Generator, scale_range: tuple[float, float]) -> FrictionModel:
    """Scale each joint's viscous and Coulomb coefficient log-uniformly."""
    low, high = np.log(scale_range[0]), np.log(scale_range[1])
    viscous = base.viscous * np.exp(rng.uniform(low, high, size=base.n_dof))
    coulomb = base.coulomb * np.exp(rng.uniform(low, high, size=base.n_dof))
    return FrictionModel(viscous, coulomb)


def elastic_time_step(transmission: TransmissionSpec, config: DatasetConfig) -> float:
    return min(config.max_time_step, transmission.required_time_step())


def run_condition(
    asset: AssetSpec,
    trajectory: MaterializedTrajectory,
    tier: Tier,
    backend: str,
    friction: FrictionModel,
    config: DatasetConfig,
    *,
    link_inertia: LinkInertia | None = None,
) -> dict[str, Any]:
    """Execute one trajectory under one condition and return the rollout.

    ``link_inertia`` is computed from the asset when not given; pass it when
    running many conditions, since it is the same for all of them.
    """
    n_dof = len(asset.joint_names)
    view = {"visualize": config.visualize, "realtime_scale": config.realtime_scale}
    if tier.is_rigid:
        controller = ComputedTorqueController(
            asset, trajectory, friction=friction,
            natural_frequency=config.control_frequency,
            damping_ratio=config.control_damping_ratio,
        )
        runner = run_mujoco_torque if backend == "mujoco" else run_newton_torque
        result = runner(asset, trajectory, controller, time_step=config.rigid_time_step,
                        friction=friction, **view)
        result.update(transmission=None, time_step=config.rigid_time_step)
        return result
    if link_inertia is None:
        link_inertia = link_inertia_envelope(asset, n_samples=config.transmission.inertia_samples)
    transmission = tier.transmission(n_dof, link_inertia)
    time_step = elastic_time_step(transmission, config)
    controller = SeaMotorController(
        asset, trajectory, transmission, friction=friction,
        natural_frequency=config.control_frequency,
        damping_ratio=config.control_damping_ratio,
    )
    runner = run_mujoco_elastic_torque if backend == "mujoco" else run_newton_elastic_torque
    result = runner(asset, trajectory, controller, transmission, time_step=time_step,
                    friction=friction, **view)
    result.update(transmission=transmission, time_step=time_step)
    return result


def describe_transmission(transmission: TransmissionSpec | None, time_step: float | None = None) -> dict[str, Any]:
    """JSON-serializable per-joint transmission parameters."""
    if transmission is None:
        return {"stiffness": None, "damping": None, "damping_ratio": None, "rotor_inertia": None,
                "mode_frequency_hz": None, "time_step": time_step}
    return {
        "stiffness": transmission.stiffness.tolist(),
        "damping": transmission.damping.tolist(),
        "damping_ratio": None if transmission.damping_ratio is None else transmission.damping_ratio.tolist(),
        "rotor_inertia": transmission.rotor_inertia.tolist(),
        "mode_frequency_hz": transmission.natural_frequency().tolist(),
        "time_step": time_step,
    }


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
    transmission = describe_transmission(result.get("transmission"))
    for index, name in enumerate(names):
        frame[f"viscous__{name}"] = friction.viscous[index]
        frame[f"coulomb__{name}"] = friction.coulomb[index]
        for key in ("stiffness", "damping", "damping_ratio", "rotor_inertia"):
            values = transmission[key]
            frame[f"{key}__{name}"] = np.nan if values is None else values[index]
    return frame


def generate(
    config: DatasetConfig, asset: AssetSpec, *, verbose: bool = True
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Build the whole dataset and return ``(frame, manifest, backend_comparison)``."""
    base_friction = FrictionModel.from_asset(asset)
    rng = np.random.default_rng(config.seed)
    frictions = [base_friction] + [
        sample_friction(base_friction, rng, config.friction_scale_range)
        for _ in range(max(0, config.n_friction_samples - 1))
    ]
    n_dof = len(asset.joint_names)
    link_inertia = None
    robots: list[dict[str, Any]] = []
    if any(not tier.is_rigid for tier in config.tiers):
        link_inertia = link_inertia_envelope(asset, n_samples=config.transmission.inertia_samples)
        if verbose:
            print(f"link inertia M_ii [kg m^2]: median {np.array2string(link_inertia[0], precision=4)}")
            print(f"                            floor  {np.array2string(link_inertia[1], precision=4)}")
        for tier in config.tiers:
            if tier.is_rigid:
                continue
            transmission = tier.transmission(n_dof, link_inertia)
            robots.append({"name": tier.name, **describe_transmission(transmission, elastic_time_step(transmission, config))})
            if verbose:
                print(f"robot {tier.name}: k [Nm/rad] {np.array2string(transmission.stiffness, precision=0, floatmode='fixed')}"
                      f"\n           zeta {np.array2string(transmission.damping_ratio, precision=3)}"
                      f" | top mode {transmission.natural_frequency().max():.0f} Hz"
                      f" | step {robots[-1]['time_step']:.1e} s")

    # Contacts are disabled during a rollout, so a trajectory that collides
    # would be simulated straight through the geometry: validate here, where
    # the trajectory is chosen, rather than trusting it afterwards.
    kinematics = PortableKinematics(asset)
    trajectories: dict[tuple[str, int], MaterializedTrajectory] = {}
    for tier in (config.tiers if config.trajectories_per_robot else config.tiers[:1]):
        key = tier.name if config.trajectories_per_robot else ""
        for index in range(config.n_trajectories):
            trajectories[(key, index)] = optimize_excitation(
                asset, config.excitation, seed=trajectory_seed(config, tier, index),
                n_candidates=config.candidates, kinematics=kinematics,
            )
            if verbose:
                metadata = trajectories[(key, index)].metadata
                label = f"trajectory {index}" + (f" for {tier.name}" if config.trajectories_per_robot else "")
                print(f"{label}: condition={metadata['condition_number']:.0f}"
                      f" ({metadata['collision_rejections']} candidates rejected for collision)")

    frames: list[pd.DataFrame] = []
    records: list[dict[str, Any]] = []
    for bag_index, (traj_index, tier, friction_index, backend) in enumerate(iter_conditions(config)):
        trajectory = trajectories[(tier.name if config.trajectories_per_robot else "", traj_index)]
        friction = frictions[friction_index]
        bag = f"t{traj_index}_{tier.name}_f{friction_index}_{backend}"
        result = run_condition(asset, trajectory, tier, backend, friction, config, link_inertia=link_inertia)
        frames.append(rollout_frame(asset, trajectory, result, bag=bag, tier=tier, backend=backend,
                                    friction=friction, resample_step=config.sample_time_step))
        records.append({
            "bag": bag, "bag_index": bag_index, "trajectory": traj_index, "tier": tier.name,
            **describe_transmission(result["transmission"], result["time_step"]),
            "friction_index": friction_index, "backend": backend,
            "solver": result.get("solver"), "wall_time": float(result.get("wall_time", 0.0)),
            "samples": int(len(result["time"])),
            "trajectory_digest": trajectory.digest(),
            "condition_number": float(trajectory.metadata["condition_number"]),
            "viscous": friction.viscous.tolist(), "coulomb": friction.coulomb.tolist(),
            "tracking_rms": float(np.sqrt(np.mean((np.asarray(result["q_link"]) - np.asarray(result["q_ref"])) ** 2))),
            "max_deflection": float(np.abs(np.asarray(result["q_motor"]) - np.asarray(result["q_link"])).max()),
            "feedback_ratio": float(
                np.mean(np.abs(result["tau_feedback"])) / max(np.mean(np.abs(result["tau_feedforward"])), 1e-12)
            ),
        })
        if verbose:
            print(f"  [{bag_index + 1}] {bag:30s} {records[-1]['wall_time']:6.1f}s "
                  f"trk={records[-1]['tracking_rms']:.2e} defl={records[-1]['max_deflection']:.2e}")

    frame = pd.concat(frames, ignore_index=True)
    comparison = compare_backends(frame, records, config.comparison)
    if verbose and len(config.backends) > 1:
        print("\n" + format_report(comparison, config.comparison))
    manifest = {
        "asset": asset.name,
        "n_bags": len(records),
        "n_samples": int(len(frame)),
        "n_dof": n_dof,
        "joint_names": list(asset.joint_names),
        "sample_time_step": config.sample_time_step,
        "seed": config.seed,
        "trajectories_per_robot": config.trajectories_per_robot,
        "distinct_trajectories": len(trajectories),
        "backends": list(config.backends),
        "tiers": [t.name for t in config.tiers],
        "rigid_reference": config.rigid_reference,
        "transmission_sampling": asdict(config.transmission),
        "link_inertia": None if link_inertia is None else {
            "median": link_inertia[0].tolist(), "floor": link_inertia[1].tolist(),
        },
        "robots": robots,
        "backend_comparison": summarize(comparison, config.comparison),
        "target": "ft0..ft{n-1} = link-side joint torque [Nm]",
        "input": "q0..q{n-1} [rad], dq0..dq{n-1} [rad/s], tau0..tau{n-1} = applied motor torque [Nm]",
        "records": records,
    }
    return frame, manifest, comparison


def write_dataset(
    frame: pd.DataFrame, manifest: Mapping[str, Any], output: str | Path,
    comparison: pd.DataFrame | None = None,
) -> tuple[Path, Path, Path | None]:
    """Write the flat CSV, its manifest and, if any pairs, the backend comparison."""
    csv_path = Path(output).expanduser().resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    manifest_path = csv_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(dict(manifest), indent=2), encoding="utf-8")
    comparison_path = None
    if comparison is not None and not comparison.empty:
        comparison_path = comparison_report_path(csv_path)
        comparison.to_csv(comparison_path, index=False)
    return csv_path, manifest_path, comparison_path


def comparison_report_path(csv_path: str | Path) -> Path:
    return Path(csv_path).with_suffix(".backend_comparison.csv")
