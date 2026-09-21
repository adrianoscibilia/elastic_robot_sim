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
import warnings
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from .assets import AssetSpec
from .payload import Payload, payload_asset
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


_PROVENANCE_CLASSES = frozenset("MPCDE")
_SAMPLING_METHODS = ("iid", "stratified", "sobol")


@dataclass(frozen=True)
class TransmissionSampling:
    """How elastic robots are drawn.

    ``stiffness`` holds one ``(min, max)`` interval per joint [Nm/rad] -- the
    legacy form, kept for backward compatibility.  ``stiffness_nominal`` +
    ``stiffness_factor`` is the preferred form: a per-joint nominal (with
    provenance, see ``docs/PARAMETER_PROVENANCE.md``) times a shared
    log-uniform multiplicative factor, split between a per-robot common scale
    and per-joint scatter by ``stiffness_common_fraction``.  Give exactly one
    of the two representations.

    ``damping_ratio`` is one ``(min, max)`` interval shared by all joints and
    sampled uniformly per joint.  ``rotor_inertia`` is the reflected rotor
    inertia [kg m^2], one value or one per joint, fixed (not sampled) unless
    ``rotor_inertia_nominal`` is given, in which case it is drawn the same way
    as ``stiffness_nominal`` from its own stream, independent of stiffness.

    ``sampling`` selects how the *marginal* draw is spread across
    ``robots``: ``iid`` (independent, the historical behaviour -- a robot's
    parameters depend only on the seed and its index, so raising ``robots``
    leaves existing robots unchanged), ``stratified`` (one draw per equal
    -width stratum in log space, permuted independently per joint -- full
    marginal coverage at the same robot count, but not prefix-stable: raising
    ``robots`` redefines the whole set) or ``sobol`` (a scrambled Sobol
    sequence via ``scipy.stats.qmc`` -- also full coverage, and prefix-stable
    like ``iid``).
    """

    robots: int = 0
    stiffness: tuple[tuple[float, float], ...] = ()
    damping_ratio: tuple[float, float] = (0.05, 0.2)
    rotor_inertia: tuple[float, ...] = (0.1,)
    inertia_samples: int = 512
    sampling: str = "iid"
    stiffness_nominal: tuple[float, ...] = ()
    stiffness_factor: tuple[float, float] = (1.0, 1.0)
    stiffness_common_fraction: float = 0.5
    stiffness_provenance: tuple[str, ...] = ()
    rotor_inertia_nominal: tuple[float, ...] = ()
    rotor_inertia_factor: tuple[float, float] = (1.0, 1.0)
    rotor_inertia_provenance: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.robots < 0 or self.inertia_samples < 1:
            raise ValueError("transmission.robots must be >= 0 and inertia_samples positive")
        if self.sampling not in _SAMPLING_METHODS:
            raise ValueError(f"transmission.sampling must be one of {_SAMPLING_METHODS}")
        if self.stiffness and self.stiffness_nominal:
            raise ValueError("give either transmission.stiffness or stiffness_nominal/stiffness_factor, not both")
        if self.robots and not self.stiffness and not self.stiffness_nominal:
            raise ValueError("transmission.stiffness or stiffness_nominal/stiffness_factor is required")
        for low, high in self.stiffness:
            if not 0.0 < low <= high:
                raise ValueError(f"stiffness interval [{low}, {high}] must satisfy 0 < min <= max")
        if self.stiffness_nominal:
            if min(self.stiffness_nominal) <= 0.0:
                raise ValueError("stiffness_nominal must be positive")
            lo, hi = self.stiffness_factor
            if not 0.0 < lo <= hi:
                raise ValueError(f"stiffness_factor [{lo}, {hi}] must satisfy 0 < min <= max")
            if not 0.0 <= self.stiffness_common_fraction <= 1.0:
                raise ValueError("stiffness_common_fraction must be in [0, 1]")
        low, high = self.damping_ratio
        if not 0.0 <= low <= high:
            raise ValueError(f"damping_ratio interval [{low}, {high}] must satisfy 0 <= min <= max")
        n_joints = len(self.stiffness) or len(self.stiffness_nominal)
        if len(self.rotor_inertia) not in (1, max(1, n_joints)) or min(self.rotor_inertia) <= 0.0:
            raise ValueError("rotor_inertia needs one positive value or one per joint")
        if self.rotor_inertia_nominal:
            if min(self.rotor_inertia_nominal) <= 0.0:
                raise ValueError("rotor_inertia_nominal must be positive")
            lo, hi = self.rotor_inertia_factor
            if not 0.0 < lo <= hi:
                raise ValueError(f"rotor_inertia_factor [{lo}, {hi}] must satisfy 0 < min <= max")
        for name, values in (("stiffness_provenance", self.stiffness_provenance),
                             ("rotor_inertia_provenance", self.rotor_inertia_provenance)):
            bad = sorted(set(values) - _PROVENANCE_CLASSES)
            if bad:
                raise ValueError(f"{name} has unknown provenance class(es) {bad}; use one of {sorted(_PROVENANCE_CLASSES)}")


def _stratified_unit(count: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """``(count, n)`` draws in ``[0, 1)``, one per equal-width stratum per column.

    Independent uniform draws leave large parts of a wide interval unvisited
    at the counts this dataset uses.  Stratifying guarantees full marginal
    coverage at identical cost; shuffling each column's strata independently
    keeps the columns from being correlated by construction.
    """
    edges = np.linspace(0.0, 1.0, count + 1)
    draws = np.empty((count, n))
    for column in range(n):
        u = rng.uniform(edges[:-1], edges[1:])
        rng.shuffle(u)
        draws[:, column] = u
    return draws


def _sobol_unit(count: int, n: int, seed: int) -> np.ndarray:
    """``(count, n)`` draws in ``[0, 1)`` from a scrambled Sobol sequence.

    Prefix-stable like ``iid`` (the first ``n`` points of a Sobol sequence are
    themselves a low-discrepancy set for every ``n``), but with the coverage
    of stratification.
    """
    from scipy.stats import qmc

    sampler = qmc.Sobol(d=max(n, 1), scramble=True, seed=int(seed))
    return sampler.random(count)[:, :n]


def _sample_unit(count: int, n: int, method: str, rng: np.random.Generator, seed: int) -> np.ndarray:
    """Dispatch a ``(count, n)`` draw in ``[0, 1)`` on ``method``."""
    if method == "stratified":
        return _stratified_unit(count, n, rng)
    if method == "sobol":
        return _sobol_unit(count, n, seed)
    return rng.uniform(0.0, 1.0, size=(count, n))


def _stratified_log_uniform(low: np.ndarray, high: np.ndarray, count: int,
                            rng: np.random.Generator) -> np.ndarray:
    """``count`` draws per joint, one per equal-width stratum in log space.

    Returns shape ``(count, n_joints)``.  See :func:`_stratified_unit`.
    """
    unit = _stratified_unit(count, len(low), rng)
    return np.exp(np.log(low)[None, :] + unit * (np.log(high) - np.log(low))[None, :])


def _sample_marginal(low: np.ndarray, high: np.ndarray, count: int, method: str,
                     rng: np.random.Generator, seed: int) -> np.ndarray:
    """Dispatch a ``(count, n_joints)`` log-uniform draw on ``method``."""
    unit = _sample_unit(count, len(low), method, rng, seed)
    return np.exp(np.log(low)[None, :] + unit * (np.log(high) - np.log(low))[None, :])


def _sample_stiffness_correlated(nominal: np.ndarray, factor: tuple[float, float], common_fraction: float,
                                 count: int, method: str, rng: np.random.Generator, seed: int) -> np.ndarray:
    """``count`` stiffness vectors as ``nominal x log-uniform factor``.

    A fraction ``common_fraction`` of the log-uncertainty is a single scalar
    shared by every joint of a robot, the rest is drawn per joint.  This
    encodes that an arm is built from one gearbox family: a robot is globally
    stiffer or softer than nominal, with joint-to-joint scatter on top of
    that.  ``common_fraction=0`` recovers fully independent per-joint draws;
    ``1`` gives a one-dimensional family.
    """
    n = len(nominal)
    span = np.log(factor[1]) - np.log(factor[0])
    centre = 0.5 * (np.log(factor[0]) + np.log(factor[1]))
    common_unit = _sample_unit(count, 1, method, rng, seed)[:, 0]
    joint_unit = _sample_unit(count, n, method, rng, seed + 1)
    common = (common_unit - 0.5) * span * common_fraction
    per_joint = (joint_unit - 0.5) * span * (1.0 - common_fraction)
    return nominal[None, :] * np.exp(centre + common[:, None] + per_joint)


def sample_robots(sampling: TransmissionSampling, seed: int) -> tuple[Tier, ...]:
    """Draw ``sampling.robots`` elastic robots named ``e00, e01, ...``.

    With ``sampling.sampling == "iid"`` and the legacy ``stiffness`` interval
    form, robots are drawn one at a time from their own random stream in the
    historical, bit-identical order: a robot's parameters depend only on the
    seed and its index, so raising ``robots`` adds robots without changing
    the existing ones.  Every other combination (stratified/Sobol coverage,
    or the ``stiffness_nominal x factor`` form) draws all robots at once from
    the same stream, since those methods are defined by the whole batch.

    ``rotor_inertia`` is fixed (not sampled) unless ``rotor_inertia_nominal``
    is given, in which case it is drawn from stream ``(seed, 4)``,
    independent of the stiffness draw -- correlating them would hide the
    ``k``/``J_rotor`` degeneracy rather than cover it (see ``R3_03 Sec 3``).
    """
    rng = np.random.default_rng((int(seed), 1))
    count = sampling.robots

    if sampling.sampling == "iid" and not sampling.stiffness_nominal:
        intervals = np.asarray(sampling.stiffness, dtype=float).reshape(-1, 2)
        stiffness_draws = np.empty((count, len(intervals)))
        zeta_draws = np.empty((count, len(intervals)))
        for index in range(count):
            stiffness_draws[index] = np.exp(rng.uniform(np.log(intervals[:, 0]), np.log(intervals[:, 1])))
            zeta_draws[index] = rng.uniform(sampling.damping_ratio[0], sampling.damping_ratio[1], size=len(intervals))
    elif sampling.stiffness_nominal:
        nominal = np.asarray(sampling.stiffness_nominal, dtype=float)
        stiffness_draws = _sample_stiffness_correlated(
            nominal, sampling.stiffness_factor, sampling.stiffness_common_fraction,
            count, sampling.sampling, rng, int(seed),
        )
        zeta_draws = rng.uniform(sampling.damping_ratio[0], sampling.damping_ratio[1], size=(count, len(nominal)))
    else:
        intervals = np.asarray(sampling.stiffness, dtype=float).reshape(-1, 2)
        stiffness_draws = _sample_marginal(intervals[:, 0], intervals[:, 1], count, sampling.sampling, rng, int(seed))
        zeta_draws = rng.uniform(sampling.damping_ratio[0], sampling.damping_ratio[1], size=(count, len(intervals)))

    if sampling.rotor_inertia_nominal:
        rotor_rng = np.random.default_rng((int(seed), 4))
        rotor_nominal = np.asarray(sampling.rotor_inertia_nominal, dtype=float)
        rotor_draws = _sample_stiffness_correlated(
            rotor_nominal, sampling.rotor_inertia_factor, 0.0, count, sampling.sampling, rotor_rng, int(seed) + 1,
        )
    else:
        rotor_draws = None

    robots = []
    for index in range(count):
        rotor_inertia = sampling.rotor_inertia if rotor_draws is None else tuple(rotor_draws[index])
        robots.append(Tier(f"e{index:02d}", stiffness=tuple(stiffness_draws[index]),
                           damping_ratio=tuple(zeta_draws[index]), rotor_inertia=rotor_inertia))
    return tuple(robots)


def build_tiers(rigid_reference: bool, sampling: TransmissionSampling, seed: int) -> tuple[Tier, ...]:
    """The rigid reference, if kept, followed by the sampled elastic robots."""
    tiers = ((Tier(RIGID_TIER),) if rigid_reference else ()) + sample_robots(sampling, seed)
    if not tiers:
        raise ValueError("no tiers selected: keep the rigid reference or sample at least one robot")
    return tiers


@dataclass(frozen=True)
class SplitPolicy:
    """How tiers are assigned to train / validation / test.

    ``contiguous`` reproduces the historical behaviour: no ``split`` label is
    assigned here at all, and a consumer that splits a contiguous half of the
    file puts every robot in both halves.  ``holdout_robots`` reserves whole
    robots, which is the only split that measures generalization to an
    unseen transmission -- the reason the transmission is randomized at all.
    """

    mode: str = "contiguous"
    test_robots: int = 0
    val_robots: int = 0
    test_trajectories: int = 0
    val_trajectories: int = 0

    def __post_init__(self) -> None:
        if self.mode not in ("contiguous", "holdout_trajectories", "holdout_robots"):
            raise ValueError(f"unknown split mode {self.mode!r}")
        for name in ("test_robots", "val_robots", "test_trajectories", "val_trajectories"):
            if getattr(self, name) < 0:
                raise ValueError(f"split.{name} must be >= 0")


def assign_splits(tiers: tuple[Tier, ...], policy: SplitPolicy) -> dict[str, str]:
    """Map each tier name to ``"train" | "val" | "test"``.

    ``holdout_robots`` reserves the softest and stiffest strata (by mean log
    stiffness) for test and the next-most-extreme for validation, so the held
    -out set is a measured extrapolation margin rather than another
    interpolation sample.  The rigid tier has no stiffness to hold out and is
    always ``train``.
    """
    labels = {tier.name: "train" for tier in tiers}
    if policy.mode != "holdout_robots":
        return labels
    elastic = [tier for tier in tiers if not tier.is_rigid]
    if policy.test_robots + policy.val_robots > len(elastic):
        raise ValueError(
            f"split.test_robots + split.val_robots ({policy.test_robots + policy.val_robots}) "
            f"exceeds the number of elastic robots ({len(elastic)})"
        )
    ranked = sorted(elastic, key=lambda tier: float(np.mean(np.log(tier.stiffness))))
    n_test = policy.test_robots
    n_soft_test = n_test // 2
    n_stiff_test = n_test - n_soft_test
    test_names = {tier.name for tier in ranked[:n_soft_test]} | (
        {tier.name for tier in ranked[len(ranked) - n_stiff_test:]} if n_stiff_test else set()
    )
    remaining = [tier for tier in ranked if tier.name not in test_names]
    n_val = policy.val_robots
    n_soft_val = n_val // 2
    n_stiff_val = n_val - n_soft_val
    val_names = {tier.name for tier in remaining[:n_soft_val]} | (
        {tier.name for tier in remaining[len(remaining) - n_stiff_val:]} if n_stiff_val else set()
    )
    for name in test_names:
        labels[name] = "test"
    for name in val_names:
        labels[name] = "val"
    return labels


@dataclass(frozen=True)
class PayloadSampling:
    """Randomized tool/payload rigidly attached to the flange.

    Without one, the last three link-side torque channels are numerically
    zero (the bare flange's ``M_77`` is ~3e-4 kg m^2) and per-channel
    normalization in the consumer amplifies their noise to unit variance
    (``R3_01`` F2).  ``per: "robot"`` draws one payload per elastic tier (the
    physical reading: a robot ships with a tool); ``"trajectory"`` draws one
    per bag (the reading: the same robot is used with many tools).
    """

    enabled: bool = False
    mass: tuple[float, float] = (0.0, 0.0)
    offset_x: tuple[float, float] = (0.0, 0.0)
    offset_y: tuple[float, float] = (0.0, 0.0)
    offset_z: tuple[float, float] = (0.0, 0.0)
    size: tuple[float, float] = (0.1, 0.1)
    per: str = "robot"

    def __post_init__(self) -> None:
        if self.per not in ("robot", "trajectory"):
            raise ValueError("payload.per must be 'robot' or 'trajectory'")
        for name in ("mass", "size"):
            low, high = getattr(self, name)
            if low < 0.0 or high < low:
                raise ValueError(f"payload.{name} must satisfy 0 <= min <= max")
        if self.enabled and self.size[0] <= 0.0:
            raise ValueError("payload.size min must be positive when enabled")


def sample_payloads(sampling: PayloadSampling, seed: int, count: int) -> tuple[Payload, ...]:
    """Draw ``count`` payloads from their own stream, in index order.

    Uses stream ``(seed, 2)`` so payloads are independent of the robot
    stream ``(seed, 1)``: adding payloads does not perturb the sampled
    stiffnesses, and vice versa.
    """
    if not sampling.enabled:
        return tuple(Payload() for _ in range(count))
    rng = np.random.default_rng((int(seed), 2))
    payloads = []
    for _ in range(count):
        mass = float(rng.uniform(sampling.mass[0], sampling.mass[1]))
        offset = (
            float(rng.uniform(sampling.offset_x[0], sampling.offset_x[1])),
            float(rng.uniform(sampling.offset_y[0], sampling.offset_y[1])),
            float(rng.uniform(sampling.offset_z[0], sampling.offset_z[1])),
        )
        size = float(rng.uniform(sampling.size[0], sampling.size[1]))
        payloads.append(Payload(mass=mass, offset=offset, size=size))
    return tuple(payloads)


@dataclass(frozen=True)
class RegimeSampling:
    """Per-trajectory dynamic-regime randomization.

    Without it every trajectory is amplitude-maximized against the same
    ``max_acceleration``/``velocity_fraction`` caps, so the acceleration and
    velocity distributions are near-identical bag to bag and the dataset
    spans a single dynamic regime (``R3_02`` F6).  Sampled log-uniformly per
    trajectory from stream ``(seed, 3, trajectory_seed)``, independent of the
    trajectory's own coefficient draw.  ``base_frequency`` is deliberately
    left fixed for now: randomizing it changes each bag's duration and
    length, which the declared ``split`` (not the historical contiguous one)
    no longer strictly needs but which is not worth the added risk in the
    first iteration of this feature.
    """

    enabled: bool = False
    max_acceleration: tuple[float, float] = (1.0, 1.0)
    velocity_fraction: tuple[float, float] = (1.0, 1.0)

    def __post_init__(self) -> None:
        for name in ("max_acceleration", "velocity_fraction"):
            low, high = getattr(self, name)
            if low <= 0.0 or high < low:
                raise ValueError(f"regime.{name} must satisfy 0 < min <= max")


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
    split: SplitPolicy = field(default_factory=SplitPolicy)
    payload: PayloadSampling = field(default_factory=PayloadSampling)
    excitation: FourierExcitationConfig = field(default_factory=FourierExcitationConfig)
    regime: RegimeSampling = field(default_factory=RegimeSampling)
    candidates: int = 48
    control_frequency: float = 25.0
    control_damping_ratio: float = 1.0
    rigid_time_step: float = 5.0e-4
    sample_time_step: float = 0.002
    max_time_step: float = 5.0e-4
    # Evaluate the controller every N physics steps and hold the torque
    # (zero-order hold) between updates, as a real drive does; 1 evaluates it
    # every step (unchanged behaviour).  The physics step is set by the
    # transmission mode (~640 Hz at A7), but the controller's closed-loop
    # bandwidth is ~4 Hz, so a real drive updates far less often than the
    # integrator steps -- decimating recovers most of that gap as speed.
    control_decimation: int = 1
    output: str = "data/identification/dataset.csv"
    metadata_columns: str = "inline"  # "inline" | "sidecar", see write_dataset
    visualize: bool = False
    realtime_scale: float = 1.0


_TRANSMISSION_KEYS = {
    "robots", "stiffness", "damping_ratio", "rotor_inertia", "inertia_samples", "sampling",
    "stiffness_nominal", "stiffness_factor", "stiffness_common_fraction", "stiffness_provenance",
    "rotor_inertia_nominal", "rotor_inertia_factor", "rotor_inertia_provenance",
}


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
             "excitation", "dataset", "simulation", "visualization", "payload"}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"unknown keys in {source}: {', '.join(unknown)}")

    tr_cfg = raw.get("transmission", {}) or {}
    unknown = sorted(set(tr_cfg) - _TRANSMISSION_KEYS)
    if unknown:
        raise ValueError(f"unknown transmission keys in {source}: {', '.join(unknown)}")
    zeta = tr_cfg.get("damping_ratio", [0.05, 0.2])
    if "stiffness" in tr_cfg and "stiffness_nominal" not in tr_cfg:
        warnings.warn(
            f"{source}: transmission.stiffness is deprecated, use stiffness_nominal/stiffness_factor "
            "(see config/identification/kuka_lbr_iiwa_14_r820_table.yaml)",
            DeprecationWarning, stacklevel=2,
        )
    stiffness_factor = tr_cfg.get("stiffness_factor", [1.0, 1.0])
    rotor_factor = tr_cfg.get("rotor_inertia_factor", [1.0, 1.0])
    sampling = TransmissionSampling(
        robots=int(tr_cfg.get("robots", 0)),
        stiffness=tuple((float(low), float(high)) for low, high in tr_cfg.get("stiffness", []) or []),
        damping_ratio=(float(zeta[0]), float(zeta[1])),
        rotor_inertia=tuple(float(value) for value in np.atleast_1d(tr_cfg.get("rotor_inertia", 0.1))),
        inertia_samples=int(tr_cfg.get("inertia_samples", 512)),
        sampling=str(tr_cfg.get("sampling", "stratified")),
        stiffness_nominal=tuple(float(v) for v in tr_cfg.get("stiffness_nominal", []) or []),
        stiffness_factor=(float(stiffness_factor[0]), float(stiffness_factor[1])),
        stiffness_common_fraction=float(tr_cfg.get("stiffness_common_fraction", 0.5)),
        stiffness_provenance=tuple(str(v) for v in tr_cfg.get("stiffness_provenance", []) or []),
        rotor_inertia_nominal=tuple(float(v) for v in tr_cfg.get("rotor_inertia_nominal", []) or []),
        rotor_inertia_factor=(float(rotor_factor[0]), float(rotor_factor[1])),
        rotor_inertia_provenance=tuple(str(v) for v in tr_cfg.get("rotor_inertia_provenance", []) or []),
    )

    exc_cfg = raw.get("excitation", {}) or {}
    sim_cfg = raw.get("simulation", {}) or {}
    data_cfg = raw.get("dataset", {}) or {}
    view_cfg = raw.get("visualization", {}) or {}
    scale = data_cfg.get("friction_scale", [0.5, 2.0])
    seed = int(data_cfg.get("seed", 20260917))
    rigid_reference = bool(raw.get("rigid_reference", True))
    split_cfg = data_cfg.get("split", {}) or {}
    split = SplitPolicy(
        mode=str(split_cfg.get("mode", "contiguous")),
        test_robots=int(split_cfg.get("test_robots", 0)),
        val_robots=int(split_cfg.get("val_robots", 0)),
        test_trajectories=int(split_cfg.get("test_trajectories", 0)),
        val_trajectories=int(split_cfg.get("val_trajectories", 0)),
    )
    payload_cfg = raw.get("payload", {}) or {}
    payload_mass = payload_cfg.get("mass", [0.0, 0.0])
    payload_size = payload_cfg.get("size", [0.1, 0.1])
    payload_x = payload_cfg.get("offset_x", [0.0, 0.0])
    payload_y = payload_cfg.get("offset_y", [0.0, 0.0])
    payload_z = payload_cfg.get("offset_z", [0.0, 0.0])
    payload = PayloadSampling(
        enabled=bool(payload_cfg.get("enabled", False)),
        mass=(float(payload_mass[0]), float(payload_mass[1])),
        offset_x=(float(payload_x[0]), float(payload_x[1])),
        offset_y=(float(payload_y[0]), float(payload_y[1])),
        offset_z=(float(payload_z[0]), float(payload_z[1])),
        size=(float(payload_size[0]), float(payload_size[1])),
        per=str(payload_cfg.get("per", "robot")),
    )
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
        split=split,
        payload=payload,
        excitation=FourierExcitationConfig(
            n_harmonics=int(exc_cfg.get("n_harmonics", 5)),
            base_frequency=float(exc_cfg.get("base_frequency", 0.1)),
            n_periods=int(exc_cfg.get("n_periods", 1)),
            time_step=float(sim_cfg.get("sample_time_step", 0.002)),
            limit_margin=float(exc_cfg.get("limit_margin", 0.12)),
            max_acceleration=float(exc_cfg.get("max_acceleration", 3.0)),
            velocity_fraction=float(exc_cfg.get("velocity_fraction", 0.6)),
            centre_jitter=float(exc_cfg.get("centre_jitter", 0.0)),
            probe_harmonics=tuple(int(v) for v in exc_cfg.get("probe_harmonics", []) or []),
            probe_acceleration_fraction=float(exc_cfg.get("probe_acceleration_fraction", 0.2)),
        ),
        regime=RegimeSampling(
            enabled=bool((exc_cfg.get("regime", {}) or {}).get("enabled", False)),
            max_acceleration=tuple(
                float(v) for v in (exc_cfg.get("regime", {}) or {}).get("max_acceleration", [1.0, 1.0])
            ),
            velocity_fraction=tuple(
                float(v) for v in (exc_cfg.get("regime", {}) or {}).get("velocity_fraction", [1.0, 1.0])
            ),
        ),
        candidates=int(exc_cfg.get("candidates", 48)),
        control_frequency=float(sim_cfg.get("control_frequency", 25.0)),
        control_damping_ratio=float(sim_cfg.get("control_damping_ratio", 1.0)),
        rigid_time_step=float(sim_cfg.get("rigid_time_step", 5.0e-4)),
        sample_time_step=float(sim_cfg.get("sample_time_step", 0.002)),
        max_time_step=float(sim_cfg.get("max_time_step", 5.0e-4)),
        control_decimation=int(sim_cfg.get("control_decimation", 1)),
        output=str(data_cfg.get("output", "data/identification/dataset.csv")),
        metadata_columns=str(data_cfg.get("metadata_columns", "inline")),
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


def regime_excitation(
    base: FourierExcitationConfig, regime: RegimeSampling, dataset_seed: int, trajectory_seed_value: int
) -> FourierExcitationConfig:
    """Derive one trajectory's excitation config from the regime's own stream.

    ``generate()`` and ``run_identification_simulation.py`` both call this
    the same way, so ``--tier eNN --trajectory k`` reproduces the dataset's
    regime exactly -- this is the single highest-risk regression point in
    this feature (``R3_02 Sec 3.2``).
    """
    if not regime.enabled:
        return base
    rng = np.random.default_rng((int(dataset_seed), 3, int(trajectory_seed_value)))
    max_acceleration = float(np.exp(rng.uniform(np.log(regime.max_acceleration[0]), np.log(regime.max_acceleration[1]))))
    velocity_fraction = float(np.exp(rng.uniform(np.log(regime.velocity_fraction[0]), np.log(regime.velocity_fraction[1]))))
    return replace(base, max_acceleration=max_acceleration, velocity_fraction=velocity_fraction)


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
    payload: Payload | None = None,
) -> dict[str, Any]:
    """Execute one trajectory under one condition and return the rollout.

    ``link_inertia`` is computed from the asset when not given; pass it when
    running many conditions, since it is the same for all of them.

    ``payload`` is welded to the asset's last link (see ``payload.py``)
    before anything downstream sees it -- Pinocchio, both simulators and the
    collision checker all consume the same modified asset by construction.
    Excitation trajectories are deliberately scored payload-free (their
    kinematic limits and regressor conditioning do not depend on the
    inertias a payload changes; only the required torque does, which is
    checked separately) -- do not "fix" this without re-reading
    ``R3_01 Sec 2.6``.
    """
    n_dof = len(asset.joint_names)
    view = {"visualize": config.visualize, "realtime_scale": config.realtime_scale}
    probe_top_hz = float(trajectory.metadata.get("probe_top_hz", 0.0))

    def _check_control_rate(time_step: float) -> None:
        if config.control_decimation <= 1 or probe_top_hz <= 0.0:
            return
        control_rate = 1.0 / (time_step * config.control_decimation)
        if control_rate <= 10.0 * probe_top_hz:
            raise ValueError(
                f"control_decimation={config.control_decimation} at time_step={time_step:g}s gives a "
                f"{control_rate:g} Hz control rate, not comfortably above the {probe_top_hz:g} Hz probe "
                "top frequency (need > 10x); lower control_decimation or the probe's top harmonic"
            )

    with payload_asset(asset, payload) as asset_p:
        if tier.is_rigid:
            _check_control_rate(config.rigid_time_step)
            controller = ComputedTorqueController(
                asset_p, trajectory, friction=friction,
                natural_frequency=config.control_frequency,
                damping_ratio=config.control_damping_ratio,
            )
            runner = run_mujoco_torque if backend == "mujoco" else run_newton_torque
            result = runner(asset_p, trajectory, controller, time_step=config.rigid_time_step,
                            friction=friction, control_decimation=config.control_decimation, **view)
            result.update(transmission=None, time_step=config.rigid_time_step, payload=payload)
            return result
        if link_inertia is None:
            link_inertia = link_inertia_envelope(asset_p, n_samples=config.transmission.inertia_samples)
        transmission = tier.transmission(n_dof, link_inertia)
        time_step = elastic_time_step(transmission, config)
        _check_control_rate(time_step)
        controller = SeaMotorController(
            asset_p, trajectory, transmission, friction=friction,
            natural_frequency=config.control_frequency,
            damping_ratio=config.control_damping_ratio,
        )
        runner = run_mujoco_elastic_torque if backend == "mujoco" else run_newton_elastic_torque
        result = runner(asset_p, trajectory, controller, transmission, time_step=time_step,
                        friction=friction, control_decimation=config.control_decimation, **view)
        result.update(transmission=transmission, time_step=time_step, payload=payload)
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


def _log_span_coverage(values: np.ndarray, declared_span: np.ndarray) -> dict[str, list[float]]:
    """Declared-vs-realized log-space coverage for a ``(n_robots, n_joints)`` sample."""
    realized = np.log(values.max(axis=0)) - np.log(values.min(axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        coverage = np.where(declared_span > 0, realized / np.maximum(declared_span, 1e-300), 1.0)
    return {
        "declared_log_span": declared_span.tolist(),
        "realized_log_span": realized.tolist(),
        "coverage_fraction": coverage.tolist(),
    }


def _linear_span_coverage(values: np.ndarray, declared_span: np.ndarray) -> dict[str, list[float]]:
    """Declared-vs-realized linear-space coverage for a ``(n_robots, n_joints)`` sample."""
    realized = values.max(axis=0) - values.min(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        coverage = np.where(declared_span > 0, realized / np.maximum(declared_span, 1e-300), 1.0)
    return {
        "declared_log_span": declared_span.tolist(),
        "realized_log_span": realized.tolist(),
        "coverage_fraction": coverage.tolist(),
    }


def sampling_coverage_report(config: DatasetConfig) -> dict[str, Any]:
    """Per-joint declared-vs-realized coverage of the sampled elastic robots.

    Six i.i.d. draws visit only a fraction of a wide declared interval
    (``R3_00`` F4); this is the artifact that makes the gap visible and that
    a reviewer would ask for.  Empty when there are no elastic robots.
    """
    sampling = config.transmission
    elastic = [tier for tier in config.tiers if not tier.is_rigid]
    report: dict[str, Any] = {}
    if not elastic:
        return report

    stiffness = np.asarray([tier.stiffness for tier in elastic])
    if sampling.stiffness_nominal:
        span = float(np.log(sampling.stiffness_factor[1] / sampling.stiffness_factor[0]))
        declared = np.full(stiffness.shape[1], span)
    else:
        intervals = np.asarray(sampling.stiffness, dtype=float).reshape(-1, 2)
        declared = np.log(intervals[:, 1] / intervals[:, 0])
    report["stiffness"] = {"method": sampling.sampling, **_log_span_coverage(stiffness, declared)}

    zeta = np.asarray([tier.damping_ratio for tier in elastic])
    lo, hi = sampling.damping_ratio
    report["damping_ratio"] = {"method": "iid", **_linear_span_coverage(zeta, np.full(zeta.shape[1], hi - lo))}

    if sampling.rotor_inertia_nominal:
        rotor = np.asarray([tier.rotor_inertia for tier in elastic])
        span = float(np.log(sampling.rotor_inertia_factor[1] / sampling.rotor_inertia_factor[0]))
        declared_rotor = np.full(rotor.shape[1], span)
        report["rotor_inertia"] = {"method": sampling.sampling, **_log_span_coverage(rotor, declared_rotor)}
    return report


def _print_coverage_warnings(report: Mapping[str, Any], joint_names: Sequence[str]) -> None:
    for key, entry in report.items():
        fractions = entry.get("coverage_fraction", [])
        for index, fraction in enumerate(fractions):
            if fraction < 0.8:
                name = joint_names[index] if index < len(joint_names) else f"joint {index}"
                print(f"warning: joint {name} {key} coverage {fraction:.2f} of its declared range; "
                      "raise `robots` or switch transmission.sampling to `stratified`.")


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
    split: str = "train",
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
    # Direct readout of the transmission deflection tau/k -- the cleanest
    # target for identifying the elastic parameters themselves (R3_01 Sec 3).
    for index in range(len(names)):
        frame[f"defl{index}"] = q_motor[:, index] - q[:, index]
    frame["experiment"] = bag
    frame["tier"] = tier.name
    frame["backend"] = backend
    frame["split"] = split
    transmission = describe_transmission(result.get("transmission"))
    for index, name in enumerate(names):
        frame[f"viscous__{name}"] = friction.viscous[index]
        frame[f"coulomb__{name}"] = friction.coulomb[index]
        for key in ("stiffness", "damping", "damping_ratio", "rotor_inertia"):
            values = transmission[key]
            frame[f"{key}__{name}"] = np.nan if values is None else values[index]
    payload: Payload | None = result.get("payload")
    frame["payload_mass"] = np.nan if payload is None else payload.mass
    frame["payload_offset_x"] = np.nan if payload is None else payload.offset[0]
    frame["payload_offset_y"] = np.nan if payload is None else payload.offset[1]
    frame["payload_offset_z"] = np.nan if payload is None else payload.offset[2]
    frame["payload_size"] = np.nan if payload is None else payload.size
    exc_metadata = trajectory.metadata
    frame["exc_base_frequency"] = exc_metadata.get("base_frequency")
    frame["exc_max_acceleration"] = exc_metadata.get("max_acceleration")
    frame["exc_velocity_fraction"] = exc_metadata.get("velocity_fraction")
    frame["exc_probe_top_hz"] = exc_metadata.get("probe_top_hz", 0.0)
    return frame


def _run_bag(args: tuple) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Execute one bag and return ``(frame, record)``.

    A plain top-level function taking one tuple argument, so it can be sent
    to a process pool: every argument is a simple dataclass/array (config,
    the asset spec, the tier, ...), never a built simulator model, which is
    what makes it picklable across the process boundary.  Bags are fully
    independent -- their RNG streams are keyed on ``(seed, index)``, not on
    call order -- so worker order never affects the result.
    """
    (asset, trajectory, tier, backend, friction, config, link_inertia, payload,
     split, bag, bag_index, traj_index, friction_index) = args
    result = run_condition(asset, trajectory, tier, backend, friction, config,
                           link_inertia=link_inertia, payload=payload)
    frame = rollout_frame(asset, trajectory, result, bag=bag, tier=tier, backend=backend,
                          friction=friction, resample_step=config.sample_time_step, split=split)
    record = {
        "bag": bag, "bag_index": bag_index, "trajectory": traj_index, "tier": tier.name, "split": split,
        **describe_transmission(result["transmission"], result["time_step"]),
        "payload": None if payload is None or payload.is_empty else payload.as_dict(),
        "friction_index": friction_index, "backend": backend,
        "solver": result.get("solver"), "wall_time": float(result.get("wall_time", 0.0)),
        "samples": int(len(result["time"])),
        "trajectory_digest": trajectory.digest(),
        "condition_number": float(trajectory.metadata["condition_number"]),
        "exc_base_frequency": float(trajectory.metadata["base_frequency"]),
        "exc_max_acceleration": float(trajectory.metadata["max_acceleration"]),
        "exc_velocity_fraction": float(trajectory.metadata["velocity_fraction"]),
        "exc_probe_top_hz": float(trajectory.metadata.get("probe_top_hz", 0.0)),
        "viscous": friction.viscous.tolist(), "coulomb": friction.coulomb.tolist(),
        "tracking_rms": float(np.sqrt(np.mean((np.asarray(result["q_link"]) - np.asarray(result["q_ref"])) ** 2))),
        "max_deflection": float(np.abs(np.asarray(result["q_motor"]) - np.asarray(result["q_link"])).max()),
        "feedback_ratio": float(
            np.mean(np.abs(result["tau_feedback"])) / max(np.mean(np.abs(result["tau_feedforward"])), 1e-12)
        ),
    }
    return frame, record


def generate(
    config: DatasetConfig, asset: AssetSpec, *, verbose: bool = True, jobs: int = 1,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Build the whole dataset and return ``(frame, manifest, backend_comparison)``."""
    base_friction = FrictionModel.from_asset(asset)
    rng = np.random.default_rng(config.seed)
    frictions = [base_friction] + [
        sample_friction(base_friction, rng, config.friction_scale_range)
        for _ in range(max(0, config.n_friction_samples - 1))
    ]
    n_dof = len(asset.joint_names)
    elastic_tiers = [tier for tier in config.tiers if not tier.is_rigid]

    # Payloads: one per elastic tier ("robot" -- a robot ships with a tool)
    # or one per (tier, trajectory) pair ("trajectory" -- the same robot used
    # with many tools).  The rigid tier always gets an empty payload, so the
    # rigid reference keeps validating against the unmodified URDF.
    payload_by_tier: dict[str, Payload] = {}
    payload_by_key: dict[tuple[str, int], Payload] = {}
    if config.payload.per == "robot":
        drawn = sample_payloads(config.payload, config.seed, len(elastic_tiers))
        payload_by_tier = {tier.name: p for tier, p in zip(elastic_tiers, drawn)}
    else:
        drawn = sample_payloads(config.payload, config.seed, config.n_trajectories * len(elastic_tiers))
        drawn_iter = iter(drawn)
        for traj_index in range(config.n_trajectories):
            for tier in elastic_tiers:
                payload_by_key[(tier.name, traj_index)] = next(drawn_iter)

    def _payload_for(tier: Tier, traj_index: int) -> Payload | None:
        if tier.is_rigid:
            return Payload()
        return payload_by_tier[tier.name] if config.payload.per == "robot" else payload_by_key[(tier.name, traj_index)]

    link_inertia = None
    robots: list[dict[str, Any]] = []
    if elastic_tiers:
        # Nominal (payload-free) numbers, for the manifest's "robots" summary
        # and the verbose print -- the ground truth per bag is the payload
        # columns rollout_frame writes, which do vary with the payload.
        link_inertia = link_inertia_envelope(asset, n_samples=config.transmission.inertia_samples)
        if verbose:
            print(f"link inertia M_ii [kg m^2]: median {np.array2string(link_inertia[0], precision=4)}")
            print(f"                            floor  {np.array2string(link_inertia[1], precision=4)}")
        for tier in elastic_tiers:
            transmission = tier.transmission(n_dof, link_inertia)
            entry = {"name": tier.name, **describe_transmission(transmission, elastic_time_step(transmission, config))}
            if config.payload.per == "robot":
                payload = payload_by_tier[tier.name]
                entry["payload"] = None if payload.is_empty else payload.as_dict()
            robots.append(entry)
            if verbose:
                print(f"robot {tier.name}: k [Nm/rad] {np.array2string(transmission.stiffness, precision=0, floatmode='fixed')}"
                      f"\n           zeta {np.array2string(transmission.damping_ratio, precision=3)}"
                      f" | top mode {transmission.natural_frequency().max():.0f} Hz"
                      f" | step {robots[-1]['time_step']:.1e} s")

    # link_inertia_envelope is asset-*and*-payload dependent, unlike the
    # nominal one above; cache it keyed on the payload so a "per: robot"
    # dataset does at most robots + 1 CRBA sweeps instead of one per bag.
    envelope_cache: dict[tuple, LinkInertia] = {}

    def _envelope_for(payload: Payload | None) -> LinkInertia:
        key = () if payload is None or payload.is_empty else (payload.mass, payload.offset, payload.size)
        if key not in envelope_cache:
            with payload_asset(asset, payload) as asset_p:
                envelope_cache[key] = link_inertia_envelope(asset_p, n_samples=config.transmission.inertia_samples)
        return envelope_cache[key]

    # Contacts are disabled during a rollout, so a trajectory that collides
    # would be simulated straight through the geometry: validate here, where
    # the trajectory is chosen, rather than trusting it afterwards.
    kinematics = PortableKinematics(asset)
    trajectories: dict[tuple[str, int], MaterializedTrajectory] = {}
    for tier in (config.tiers if config.trajectories_per_robot else config.tiers[:1]):
        key = tier.name if config.trajectories_per_robot else ""
        for index in range(config.n_trajectories):
            traj_seed = trajectory_seed(config, tier, index)
            excitation = regime_excitation(config.excitation, config.regime, config.seed, traj_seed)
            trajectories[(key, index)] = optimize_excitation(
                asset, excitation, seed=traj_seed,
                n_candidates=config.candidates, kinematics=kinematics,
            )
            if verbose:
                metadata = trajectories[(key, index)].metadata
                label = f"trajectory {index}" + (f" for {tier.name}" if config.trajectories_per_robot else "")
                print(f"{label}: condition={metadata['condition_number']:.0f}"
                      f" ({metadata['collision_rejections']} candidates rejected for collision)")

    split_labels = assign_splits(config.tiers, config.split)

    work: list[tuple] = []
    for bag_index, (traj_index, tier, friction_index, backend) in enumerate(iter_conditions(config)):
        trajectory = trajectories[(tier.name if config.trajectories_per_robot else "", traj_index)]
        friction = frictions[friction_index]
        bag = f"t{traj_index}_{tier.name}_f{friction_index}_{backend}"
        split = split_labels[tier.name]
        payload = _payload_for(tier, traj_index)
        link_inertia_for_bag = _envelope_for(payload) if not tier.is_rigid else None
        work.append((asset, trajectory, tier, backend, friction, config, link_inertia_for_bag, payload,
                    split, bag, bag_index, traj_index, friction_index))

    frames: list[pd.DataFrame] = []
    records: list[dict[str, Any]] = []
    if jobs > 1 and not config.visualize:
        import concurrent.futures

        with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as pool:
            results = list(pool.map(_run_bag, work))
    else:
        results = [_run_bag(item) for item in work]
    for bag_frame, record in results:
        frames.append(bag_frame)
        records.append(record)
        if verbose:
            print(f"  [{record['bag_index'] + 1}] {record['bag']:30s} {record['wall_time']:6.1f}s "
                  f"trk={record['tracking_rms']:.2e} defl={record['max_deflection']:.2e}")

    frame = pd.concat(frames, ignore_index=True)
    # ``dynamic_model_nn`` differentiates with a Savitzky-Golay filter that
    # assumes a constant step within a bag; a non-uniform grid would corrupt
    # ddq silently rather than raising, so verify it here instead.
    for bag_name, group in frame.groupby("bag"):
        step = np.diff(group["t"].to_numpy())
        if not np.allclose(step, config.sample_time_step, atol=1e-12):
            raise ValueError(f"bag {bag_name!r} does not have a uniform {config.sample_time_step}s time step")
    comparison = compare_backends(frame, records, config.comparison)
    if verbose and len(config.backends) > 1:
        print("\n" + format_report(comparison, config.comparison))
    sampling_coverage = sampling_coverage_report(config)
    if verbose:
        _print_coverage_warnings(sampling_coverage, asset.joint_names)
        if config.transmission.stiffness_provenance or config.transmission.rotor_inertia_provenance:
            print(f"provenance: stiffness {list(config.transmission.stiffness_provenance)}  "
                  f"rotor_inertia {list(config.transmission.rotor_inertia_provenance)}")
    manifest = {
        "asset": asset.name,
        "n_bags": len(records),
        "n_samples": int(len(frame)),
        "n_dof": n_dof,
        "joint_names": list(asset.joint_names),
        "sample_time_step": config.sample_time_step,
        "control_decimation": config.control_decimation,
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
        "sampling_coverage": sampling_coverage,
        "provenance": {
            "stiffness": list(config.transmission.stiffness_provenance) or None,
            "rotor_inertia": list(config.transmission.rotor_inertia_provenance) or None,
        },
        "target": "ft0..ft{n-1} = link-side joint torque [Nm]",
        "input": "q0..q{n-1} [rad], dq0..dq{n-1} [rad/s], tau0..tau{n-1} = applied motor torque [Nm]",
        "split": {
            "mode": config.split.mode,
            "train": sorted(name for name, label in split_labels.items() if label == "train"),
            "val": sorted(name for name, label in split_labels.items() if label == "val"),
            "test": sorted(name for name, label in split_labels.items() if label == "test"),
            "rationale": (
                "held-out robots measure generalization to an unseen transmission"
                if config.split.mode == "holdout_robots"
                else "contiguous/no holdout: every robot appears in both train and test"
            ),
        },
        "records": records,
    }
    return frame, manifest, comparison


_METADATA_COLUMN_PREFIXES = (
    "viscous__", "coulomb__", "stiffness__", "damping__", "damping_ratio__", "rotor_inertia__",
    "payload_", "exc_",
)


def write_dataset(
    frame: pd.DataFrame, manifest: Mapping[str, Any], output: str | Path,
    comparison: pd.DataFrame | None = None, *, metadata_columns: str = "inline",
) -> tuple[Path, Path, Path | None]:
    """Write the flat dataset, its manifest, a consumer contract sidecar and,
    if any pairs, the backend comparison.

    Dispatches on ``output``'s suffix: ``.csv`` (the default; written with
    ``float_format="%.7g"``, since the consumer casts to float32 on load
    anyway) or ``.parquet`` (typed, compressed, ~10x smaller; written as
    float32 directly).  With ``metadata_columns="sidecar"``, the ~42
    per-row-constant metadata columns (transmission, friction, payload,
    excitation-regime) are looked up by ``bag`` in a sidecar JSON instead of
    repeated on every row -- worth it once the robot count pushes past ~25
    (``R3_06``); ``"inline"`` (the default) keeps the historical, simpler
    single-file behaviour.
    """
    if metadata_columns not in ("inline", "sidecar"):
        raise ValueError("metadata_columns must be 'inline' or 'sidecar'")
    csv_path = Path(output).expanduser().resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if metadata_columns == "sidecar":
        meta_cols = [c for c in frame.columns if c.startswith(_METADATA_COLUMN_PREFIXES)]
        if meta_cols:
            per_bag = frame.groupby("bag", sort=False)[meta_cols].first()
            sidecar = {
                str(bag): {col: (None if pd.isna(value) else value) for col, value in row.items()}
                for bag, row in per_bag.to_dict(orient="index").items()
            }
            sidecar_path = csv_path.with_suffix(".bag_metadata.json")
            sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
            frame = frame.drop(columns=meta_cols)

    suffix = csv_path.suffix.lower()
    if suffix == ".parquet":
        float_cols = frame.select_dtypes(include="float64").columns
        frame.astype({col: "float32" for col in float_cols}).to_parquet(csv_path, index=False)
    elif suffix == ".csv":
        frame.to_csv(csv_path, index=False, float_format="%.7g")
    else:
        raise ValueError(f"unsupported dataset output suffix {suffix!r}; use .csv or .parquet")

    manifest_path = csv_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(dict(manifest), indent=2), encoding="utf-8")
    n_dof = int(manifest.get("n_dof", 0))
    contract = {
        "schema": "elastic_sim.identification/2",
        "n_dof": n_dof,
        "input_columns": [f"q0..q{n_dof - 1}", f"dq0..dq{n_dof - 1}", f"tau0..tau{n_dof - 1}"],
        "target_columns": [f"ft0..ft{n_dof - 1}"],
        "target_semantics": "link-side joint torque [Nm]",
        "requires_consumer": "dynamic_model_nn dataset.py with the general ft0..ft{dof-1} branch",
        "split": manifest.get("split"),
    }
    contract_path = csv_path.with_suffix(".contract.json")
    contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    comparison_path = None
    if comparison is not None and not comparison.empty:
        comparison_path = comparison_report_path(csv_path)
        comparison.to_csv(comparison_path, index=False)
    return csv_path, manifest_path, comparison_path


def comparison_report_path(csv_path: str | Path) -> Path:
    return Path(csv_path).with_suffix(".backend_comparison.csv")
