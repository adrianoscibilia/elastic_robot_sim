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

import contextlib
import hashlib
import json
import warnings
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from .assets import AssetSpec
from .controllers import (
    ControllerDraw,
    ControllerSpec,
    NominalModelSpec,
    VelocityLoopSpec,
    build_controller,
    describe_controller,
    effective_bandwidth,
    sample_velocity_loop,
)
from .measurement import IDEAL_MEASUREMENT, MeasurementModel, measure_bag, measure_channel
from .payload import Payload, payload_asset
from .plant_extras import NO_EXTRAS, PlantExtras, StiffnessNonlinearity, TorqueRipple
from .wrench import WRENCH_COLUMNS, ForceTorqueSensor, SensorSpec
from .backend_comparison import ComparisonThresholds, compare_backends, format_report, summarize
from .excitation import FourierExcitationConfig, effective_position_window, optimize_excitation
from .identification import FrictionModel
from .kinematics import PortableKinematics
from .materialized import MaterializedTrajectory
from .torque_runners import (
    ComputedTorqueController,
    SeaMotorController,
    TransmissionSpec,
    control_separation_ratio,
    link_inertia_envelope,
    link_inertia_max,
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
    file puts every robot in both halves. ``holdout_robots`` reserves whole
    robots, ranked by their *mean* log stiffness across joints, for
    combination generalization -- with independent per-joint sampling
    (``stiffness_common_fraction: 0``, the shipped default), the mean
    concentrates near nominal, so held-out robots are extreme only on
    average, not on every joint. It is still a real improvement over the
    contiguous split (no robot appears in both halves), but it is not a
    per-joint extrapolation margin; for that, draw train and test from
    disjoint ``stiffness_factor`` ranges instead (see
    ``scripts/run_range_sensitivity.py``'s S2/S3 variants, which do exactly
    this) (``R3_10 Sec 3.1``). ``holdout_trajectories`` is accepted for
    forward compatibility but not implemented yet -- ``assign_splits`` raises
    rather than silently no-op it.
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
    stiffness) for test and the next-most-extreme for validation -- see
    ``SplitPolicy``'s docstring for what this does and does not measure. The
    rigid tier has no stiffness to hold out and is always ``train``.
    """
    if policy.mode == "holdout_trajectories":
        raise NotImplementedError(
            "split.mode 'holdout_trajectories' is accepted for forward compatibility but not "
            "implemented; use 'contiguous' or 'holdout_robots'"
        )
    labels = {tier.name: "train" for tier in tiers}
    if policy.mode != "holdout_robots":
        return labels
    elastic = [tier for tier in tiers if not tier.is_rigid]
    if policy.test_robots + policy.val_robots > len(elastic):
        raise ValueError(
            f"split.test_robots + split.val_robots ({policy.test_robots + policy.val_robots}) "
            f"exceeds the number of elastic robots ({len(elastic)})"
        )
    if policy.test_robots + policy.val_robots == len(elastic) and elastic:
        raise ValueError(
            f"split.test_robots + split.val_robots ({policy.test_robots + policy.val_robots}) leaves zero "
            f"elastic robots for training out of {len(elastic)}; lower test_robots/val_robots or raise robots"
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


@dataclass(frozen=True)
class ControlGainSampling:
    """Randomized closed-loop feedback gains, one draw per trajectory.

    ``ComputedTorqueController``/``SeaMotorController`` linearize to an
    *extra* motor-side stiffness/damping of ``(M_ii + J_rotor) * kp`` /
    ``(M_ii + J_rotor) * kd`` on top of whatever the plant itself has, so a
    fixed gain bakes one specific closed-loop behaviour into every bag -- on
    a heavy joint that motor-side damper can dominate the transmission's own
    damping by an order of magnitude (``R3_12`` Sec 1), which a model trained
    on a single fixed gain would never see varied.  Left disabled at the
    dataclass level -- a caller that builds a ``DatasetConfig`` directly
    (most tests) keeps today's single fixed gain unless it opts in; the
    shipped YAML enables it explicitly, the same pattern ``payload``/
    ``regime`` already use.
    """

    enabled: bool = False
    natural_frequency: tuple[float, float] = (25.0, 25.0)
    damping_ratio: tuple[float, float] = (1.0, 1.0)

    def __post_init__(self) -> None:
        for name in ("natural_frequency", "damping_ratio"):
            low, high = getattr(self, name)
            if low <= 0.0 or high < low:
                raise ValueError(f"control_gains.{name} must satisfy 0 < min <= max")


_CONTROL_SEPARATION_ACTIONS = ("warn", "error")


@dataclass(frozen=True)
class ControlSeparationCheck:
    """Per-bag guard on ``sqrt(k / J_eff) / omega >= min_ratio`` (R4_02 Sec 7).

    Below ``min_ratio`` the ``SeaMotorController``'s feedback linearization
    and the plant's own open-loop transmission mode are not cleanly
    separated in frequency: on a joint with heavy link inertia and soft
    transmission stiffness -- true of the UR10's shoulder at the soft end of
    its prior, even at the UR10's own retuned gains -- a fixed control gain
    tuned for one robot's transmission can end up close to or above its
    resonance on another sampled robot.  ``action: "warn"`` reports it once,
    aggregated, at the end of the build; ``"error"`` raises before that bag
    is simulated.  Absent from a config, every config (iiwa included) still
    gets the default ``{min_ratio: 5.0, action: "warn"}`` check -- it changes
    nothing the iiwa build writes to the CSV, only two new manifest-record
    keys per bag (R4_06 Sec A1).
    """

    min_ratio: float = 5.0
    action: str = "warn"

    def __post_init__(self) -> None:
        if self.min_ratio <= 0.0:
            raise ValueError("control_separation.min_ratio must be positive")
        if self.action not in _CONTROL_SEPARATION_ACTIONS:
            raise ValueError(f"control_separation.action must be one of {_CONTROL_SEPARATION_ACTIONS}")


_POSITION_SIDES = ("link", "motor")
_TARGET_SIDES = ("link_torque", "motor_torque", "ee_wrench_joint")
#: Which side of the spring each target is measured on, for the collocation
#: test.  An end-effector wrench is a link-side quantity: the cell sits past
#: every transmission in the chain.
_TARGET_SPRING_SIDE = {
    "link_torque": "link", "ee_wrench_joint": "link", "motor_torque": "motor",
}


@dataclass(frozen=True)
class SignalPolicy:
    """Which physical signals the consumer's ``q``/``dq``/``tau``/``ft`` are.

    The owner's round-5 signal design (``R5_01`` Amendment 2) is *motor*-side
    position and its derivatives plus the motor effort as inputs, and the
    *link*-side torque as the target.  That pair is non-collocated -- position
    and torque are measured on opposite sides of the spring -- which is the
    only arrangement in which elasticity appears in the data at all
    (``R5_00`` Amendment 1).  A round-4 dataset pairs link position with link
    torque, for which the link equation gives
    ``tau_s = M(q) qdd + c + g`` exactly: rigid-body dynamics with no trace of
    the spring, however elastic the robot really is.

    The default is round 4's collocated pair, so no existing config changes
    meaning; the round-5 configs set ``position_side: motor`` explicitly.
    ``q_motor*``/``q_link*`` columns are written either way, so the physical
    side of ``q0..q{n-1}`` is never ambiguous in a file -- but a consumer
    reading ``q0..`` is reading whatever this policy selected, which is why it
    is recorded in the contract sidecar and not only in the manifest.

    ``clean_columns`` adds ``*_clean`` copies of every measured channel, which
    is what makes a noisy dataset still decomposable into physics and
    instrument (``R5_00`` Q-C.3).
    """

    position_side: str = "link"
    target: str = "link_torque"
    clean_columns: bool = False

    def __post_init__(self) -> None:
        if self.position_side not in _POSITION_SIDES:
            raise ValueError(f"dataset.signals.position_side must be one of {_POSITION_SIDES}")
        if self.target not in _TARGET_SIDES:
            raise ValueError(f"dataset.signals.target must be one of {_TARGET_SIDES}")

    @property
    def is_collocated(self) -> bool:
        """True when position and target sit on the same side of the spring.

        A collocated pair carries no elastic signature whatever the robot's
        stiffness, so the elastic-vs-residual comparison is meaningless on it.
        """
        return self.position_side == _TARGET_SPRING_SIDE[self.target]

    @property
    def needs_sensor(self) -> bool:
        """True when the target comes from a force/torque cell, not a joint."""
        return self.target == "ee_wrench_joint"

    def describe(self) -> dict[str, Any]:
        return {
            "position_side": self.position_side,
            "target": self.target,
            "target_kind": _TARGET_KINDS[self.target],
            "collocated": self.is_collocated,
            "clean_columns": bool(self.clean_columns),
        }


#: The link-friction background as a declared experimental axis (`R5_07` T-16).
LINK_FRICTION_TIERS = ("off", "fixed", "per_robot")


def _robot_stream(robot: str) -> int:
    """A stable integer per robot name, so the draw survives adding robots."""
    return int.from_bytes(hashlib.blake2b(str(robot).encode("utf-8"), digest_size=4).digest(), "big")


def _log_uniform(rng: np.random.Generator, bracket: tuple[float, float], size: int) -> np.ndarray:
    """One factor per joint, log-uniform over ``bracket`` (degenerate is 1.0)."""
    low, high = float(bracket[0]), float(bracket[1])
    if low == high:
        return np.full(size, low)
    return np.exp(rng.uniform(np.log(low), np.log(high), size=size))


@dataclass(frozen=True)
class PlantExtrasSampling:
    """Config-level description of the non-Lagrangian plant effects.

    Resolved per bag by :func:`plant_extras_for_bag`, which is what draws the
    torque ripple's phase.  Empty by default: a config that says nothing gets
    round 4's purely Lagrangian link side.
    """

    #: ``off`` | ``fixed`` | ``per_robot`` (`R5_07` T-16, defect D-12).  The
    #: link-friction background is an experimental axis, not an incidental
    #: setting, because it decides what the elastic-vs-residual comparison can
    #: even see: both model classes absorb friction, so whatever share of the
    #: residual it occupies is shared background that the one term being moved
    #: between them has to stand out against.
    #:
    #: * ``off`` -- no link friction.  The science tier: can the elastic class
    #:   beat the residual class on **elasticity alone**?
    #: * ``fixed`` -- one constant law for every bag of every robot, which is
    #:   round 5's shipped behaviour and stays the default.  It is the honest
    #:   setting for a dataset aimed at *one* machine: a real FMRR has one
    #:   friction law and memorizing that machine is the job.  What it is not
    #:   is a test of friction generalization, and ``split.mode:
    #:   holdout_robots`` must not be read as one -- the held-out robots have
    #:   exactly the friction the training ones do.
    #: * ``per_robot`` -- drawn per robot from ``*_factor`` around the nominal
    #:   and logged in the existing ``link_viscous__*``/``link_coulomb__*``
    #:   columns.  The robustness tier, and the only one where a
    #:   ``holdout_robots`` claim about friction is true.  It makes the
    #:   comparison *harder*, not easier: a pointwise model cannot infer a
    #:   per-robot coefficient from one sample, so the background turns from a
    #:   learnable constant into irreducible variance.
    link_friction_tier: str = "fixed"
    link_friction_viscous: tuple[float, ...] = ()
    link_friction_coulomb: tuple[float, ...] = ()
    #: Multiplicative bracket around the nominals, used by ``per_robot`` only.
    #: ``(1.0, 1.0)`` is the degenerate factor a scalar config gets, so a
    #: round-4 or round-5-pass-2 config loads and behaves unchanged.
    link_friction_viscous_factor: tuple[float, float] = (1.0, 1.0)
    link_friction_coulomb_factor: tuple[float, float] = (1.0, 1.0)
    stiffness_breakpoints: tuple[float, ...] = ()
    stiffness_factors: tuple[float, ...] = ()
    ripple_amplitude: float = 0.0
    ripple_order: float = 24.0
    ripple_random_phase: bool = True

    def __post_init__(self) -> None:
        if self.link_friction_tier not in LINK_FRICTION_TIERS:
            raise ValueError(
                f"plant_extras.link_friction.tier must be one of {LINK_FRICTION_TIERS}, "
                f"got {self.link_friction_tier!r}"
            )
        for name in ("link_friction_viscous_factor", "link_friction_coulomb_factor"):
            low, high = (float(v) for v in getattr(self, name))
            if low <= 0.0 or high < low:
                raise ValueError(
                    f"plant_extras.link_friction.{name.split('_')[-2]}.factor must be a positive "
                    "[low, high] bracket with low <= high"
                )
        viscous, coulomb = self.link_friction_viscous, self.link_friction_coulomb
        if viscous and coulomb and len(viscous) != len(coulomb) and 1 not in (len(viscous), len(coulomb)):
            raise ValueError(
                "plant_extras.link_friction.viscous and .coulomb must have the same length "
                "(or one of them a single value broadcast to every joint)"
            )
        if any(float(v) < 0.0 for v in viscous + coulomb):
            raise ValueError("plant_extras.link_friction coefficients must be non-negative")
        if bool(self.stiffness_breakpoints) != bool(self.stiffness_factors):
            raise ValueError(
                "plant_extras.stiffness_nonlinearity needs both breakpoints and factors "
                "(factors one longer than breakpoints)"
            )

    @property
    def has_link_friction(self) -> bool:
        """True when this config's tier actually puts friction on the plant."""
        return self.link_friction_tier != "off" and bool(
            self.link_friction_viscous or self.link_friction_coulomb
        )

    @property
    def is_empty(self) -> bool:
        return not (
            self.has_link_friction or self.stiffness_breakpoints or self.ripple_amplitude
        )


def plant_extras_for_bag(
    sampling: PlantExtrasSampling, n_dof: int, dataset_seed: int, trajectory_seed_value: int,
    *, robot: str | None = None,
) -> PlantExtras:
    """Resolve one bag's plant extras, drawing the ripple phase.

    Stream ``(seed, 9, trajectory_seed)``: a new index, so enabling this
    perturbs no existing draw.  The ripple's phase is per bag rather than per
    dataset for the same reason the drive's velocity-loop gains are: a single
    fixed phase is a map a network can memorize instead of learning that a
    motor-angle-periodic disturbance exists at all.

    ``robot`` is the tier name, needed only by the ``per_robot`` friction tier
    (`R5_07` T-16), whose draw is keyed on the *robot* rather than on the bag:
    a machine has one friction law, and drawing it per bag would make it a
    per-sample disturbance instead of a property of the machine the split
    holds out.  Its stream is ``(seed, 10, robot)``, again a new index.
    """
    if sampling.is_empty:
        return NO_EXTRAS
    link_friction = None
    if sampling.has_link_friction:
        viscous = _per_joint(sampling.link_friction_viscous or (0.0,), n_dof, "plant_extras.link_friction.viscous")
        coulomb = _per_joint(sampling.link_friction_coulomb or (0.0,), n_dof, "plant_extras.link_friction.coulomb")
        if sampling.link_friction_tier == "per_robot":
            if robot is None:
                raise ValueError(
                    "plant_extras.link_friction.tier: per_robot needs the robot (tier) name, so the "
                    "draw is keyed on the machine rather than on the bag; pass robot=tier.name"
                )
            rng = np.random.default_rng((int(dataset_seed), 10, _robot_stream(robot)))
            viscous = viscous * _log_uniform(rng, sampling.link_friction_viscous_factor, n_dof)
            coulomb = coulomb * _log_uniform(rng, sampling.link_friction_coulomb_factor, n_dof)
        link_friction = FrictionModel(viscous, coulomb)
    nonlinearity = None
    if sampling.stiffness_breakpoints:
        nonlinearity = StiffnessNonlinearity(
            breakpoints=tuple(sampling.stiffness_breakpoints), factors=tuple(sampling.stiffness_factors),
        )
    ripple = None
    if sampling.ripple_amplitude:
        ripple = TorqueRipple(amplitude=float(sampling.ripple_amplitude), order=float(sampling.ripple_order))
        if sampling.ripple_random_phase:
            rng = np.random.default_rng((int(dataset_seed), 9, int(trajectory_seed_value)))
            ripple = ripple.with_phase(rng, n_dof)
    return PlantExtras(
        link_friction=link_friction, stiffness_nonlinearity=nonlinearity, torque_ripple=ripple,
    )


DEFAULT_CONFIG_DIR = "config/identification"
DEFAULT_CONFIG = "config/identification/kuka_lbr_iiwa_14_r820_table.yaml"


@dataclass(frozen=True)
class DatasetConfig:
    """Everything that defines a dataset build.

    ``tiers`` is derived from ``rigid_reference``, ``transmission`` and
    ``seed``; rebuild it with :func:`build_tiers` after changing any of them.
    """

    #: Config-file generation.  1 is round 3/4's schema; 2 declares a config
    #: written in round 5 or later and, with it, the differentiation bound
    #: `probe_top_hz <= 0.25 * sample_rate` as a *load-time error* rather than
    #: a build-time warning (`R5_06` T-8; D-8 keeps the two round-4 files
    #: loading).
    schema_version: int = 1
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
    control_gains: ControlGainSampling = field(default_factory=ControlGainSampling)
    control_separation: ControlSeparationCheck = field(default_factory=ControlSeparationCheck)
    # Round 5: which controller closes the loop, what the recorder sees, which
    # non-Lagrangian effects the plant has, and which physical signals the
    # consumer's columns carry.  Every default reproduces round 4 exactly.
    controller: ControllerSpec = field(default_factory=ControllerSpec)
    measurement: MeasurementModel = field(default_factory=MeasurementModel)
    plant_extras: PlantExtrasSampling = field(default_factory=PlantExtrasSampling)
    signals: SignalPolicy = field(default_factory=SignalPolicy)
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
_EXCITATION_KEYS = {
    "n_harmonics", "base_frequency", "n_periods", "limit_margin", "max_acceleration", "velocity_fraction",
    "centre_jitter", "probe_harmonics", "probe_acceleration_fraction", "candidates", "regime", "position_window",
}
_REGIME_KEYS = {"enabled", "max_acceleration", "velocity_fraction"}
_PAYLOAD_KEYS = {"enabled", "mass", "offset_x", "offset_y", "offset_z", "size", "per"}
_DATASET_KEYS = {
    "trajectories", "trajectories_per_robot", "friction_samples", "friction_scale", "seed", "output",
    "metadata_columns", "split", "signals",
}
_SIGNALS_KEYS = {"position_side", "target", "clean_columns"}
_SPLIT_KEYS = {"mode", "test_robots", "val_robots", "test_trajectories", "val_trajectories"}
_SIMULATION_KEYS = {
    "control_frequency", "control_damping_ratio", "control_gains", "control_decimation",
    "allow_control_decimation", "rigid_time_step", "max_time_step", "sample_time_step", "control_separation",
    "controller", "measurement", "plant_extras",
}
_CONTROLLER_KEYS = {"mode", "randomize", "nominal", "velocity_loop"}
_NOMINAL_KEYS = {"knows_payload", "friction_scale", "rotor_inertia_scale", "inertia_scale"}
_VELOCITY_LOOP_KEYS = {"position_gain", "velocity_bandwidth", "integral_time"}
_MEASUREMENT_KEYS = {
    "encoder_resolution", "q_noise", "dq_noise", "tau_noise_rel", "tau_noise_abs",
    "tau_gain_error", "delay_samples", "ft_noise_rel", "ft_noise_abs",
}
_PLANT_EXTRAS_KEYS = {"link_friction", "stiffness_nonlinearity", "torque_ripple"}
_LINK_FRICTION_KEYS = {"tier", "viscous", "coulomb", "provenance"}
_STIFFNESS_NONLINEARITY_KEYS = {"breakpoints", "factors"}
_TORQUE_RIPPLE_KEYS = {"amplitude", "order", "random_phase"}
_CONTROL_GAINS_KEYS = {"enabled", "natural_frequency", "damping_ratio"}
_CONTROL_SEPARATION_KEYS = {"min_ratio", "action"}
_VISUALIZATION_KEYS = {"enabled", "realtime_scale"}


def _check_keys(mapping: Mapping[str, Any], allowed: set[str], block: str, source: Path) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"unknown {block} keys in {source}: {', '.join(unknown)}")


def _pair(mapping: Mapping[str, Any], key: str, default: tuple[float, float]) -> tuple[float, float]:
    """Read a ``[low, high]`` pair, accepting a single number as a fixed value."""
    value = mapping.get(key, default)
    if isinstance(value, (int, float)):
        return (float(value), float(value))
    return (float(value[0]), float(value[1]))


def _controller_spec(controller_cfg: Mapping[str, Any]) -> ControllerSpec:
    nominal_cfg = controller_cfg.get("nominal", {}) or {}
    loop_cfg = controller_cfg.get("velocity_loop", {}) or {}
    return ControllerSpec(
        mode=str(controller_cfg.get("mode", "exact_ct")),
        nominal=NominalModelSpec(
            knows_payload=bool(nominal_cfg.get("knows_payload", True)),
            friction_scale=float(nominal_cfg.get("friction_scale", 1.0)),
            rotor_inertia_scale=float(nominal_cfg.get("rotor_inertia_scale", 1.0)),
            inertia_scale=float(nominal_cfg.get("inertia_scale", 1.0)),
        ),
        velocity_loop=VelocityLoopSpec(
            position_gain=_pair(loop_cfg, "position_gain", (10.0, 10.0)),
            velocity_bandwidth=_pair(loop_cfg, "velocity_bandwidth", (100.0, 100.0)),
            integral_time=_pair(loop_cfg, "integral_time", (0.05, 0.05)),
        ),
        randomize=bool(controller_cfg.get("randomize", False)),
    )


def _friction_prior(raw: Any) -> tuple[tuple[float, ...], tuple[float, float]]:
    """``viscous``/``coulomb`` as either a plain vector or ``{nominal, factor}``.

    The plain vector is what every config before `R5_07` writes and keeps its
    exact meaning: a nominal with the degenerate factor ``[1, 1]``, which the
    ``per_robot`` tier then draws no spread from.  The mapping form is what
    makes the spread explicit.
    """
    if raw is None:
        return (), (1.0, 1.0)
    if isinstance(raw, Mapping):
        nominal = tuple(float(v) for v in np.atleast_1d(raw.get("nominal", []) or []))
        factor = raw.get("factor", [1.0, 1.0])
        return nominal, (float(factor[0]), float(factor[1]))
    return tuple(float(v) for v in np.atleast_1d(raw or [])), (1.0, 1.0)


def _plant_extras_sampling(extras_cfg: Mapping[str, Any]) -> PlantExtrasSampling:
    friction_cfg = extras_cfg.get("link_friction", {}) or {}
    spring_cfg = extras_cfg.get("stiffness_nonlinearity", {}) or {}
    ripple_cfg = extras_cfg.get("torque_ripple", {}) or {}
    viscous_nominal, viscous_factor = _friction_prior(friction_cfg.get("viscous"))
    coulomb_nominal, coulomb_factor = _friction_prior(friction_cfg.get("coulomb"))
    return PlantExtrasSampling(
        link_friction_tier=str(friction_cfg.get("tier", "fixed")),
        link_friction_viscous=viscous_nominal,
        link_friction_coulomb=coulomb_nominal,
        link_friction_viscous_factor=viscous_factor,
        link_friction_coulomb_factor=coulomb_factor,
        stiffness_breakpoints=tuple(float(v) for v in spring_cfg.get("breakpoints", []) or []),
        stiffness_factors=tuple(float(v) for v in spring_cfg.get("factors", []) or []),
        ripple_amplitude=float(ripple_cfg.get("amplitude", 0.0)),
        ripple_order=float(ripple_cfg.get("order", 24.0)),
        ripple_random_phase=bool(ripple_cfg.get("random_phase", True)),
    )


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
    _check_keys(tr_cfg, _TRANSMISSION_KEYS, "transmission", source)
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
    _check_keys(exc_cfg, _EXCITATION_KEYS, "excitation", source)
    _check_keys(exc_cfg.get("regime", {}) or {}, _REGIME_KEYS, "excitation.regime", source)
    _check_keys(data_cfg, _DATASET_KEYS, "dataset", source)
    _check_keys(sim_cfg, _SIMULATION_KEYS, "simulation", source)
    _check_keys(sim_cfg.get("control_gains", {}) or {}, _CONTROL_GAINS_KEYS, "simulation.control_gains", source)
    _check_keys(sim_cfg.get("control_separation", {}) or {}, _CONTROL_SEPARATION_KEYS,
                "simulation.control_separation", source)
    controller_cfg = sim_cfg.get("controller", {}) or {}
    _check_keys(controller_cfg, _CONTROLLER_KEYS, "simulation.controller", source)
    _check_keys(controller_cfg.get("nominal", {}) or {}, _NOMINAL_KEYS, "simulation.controller.nominal", source)
    _check_keys(controller_cfg.get("velocity_loop", {}) or {}, _VELOCITY_LOOP_KEYS,
                "simulation.controller.velocity_loop", source)
    measurement_cfg = sim_cfg.get("measurement", {}) or {}
    _check_keys(measurement_cfg, _MEASUREMENT_KEYS, "simulation.measurement", source)
    extras_cfg = sim_cfg.get("plant_extras", {}) or {}
    _check_keys(extras_cfg, _PLANT_EXTRAS_KEYS, "simulation.plant_extras", source)
    _check_keys(extras_cfg.get("link_friction", {}) or {}, _LINK_FRICTION_KEYS,
                "simulation.plant_extras.link_friction", source)
    _check_keys(extras_cfg.get("stiffness_nonlinearity", {}) or {}, _STIFFNESS_NONLINEARITY_KEYS,
                "simulation.plant_extras.stiffness_nonlinearity", source)
    _check_keys(extras_cfg.get("torque_ripple", {}) or {}, _TORQUE_RIPPLE_KEYS,
                "simulation.plant_extras.torque_ripple", source)
    signals_cfg = data_cfg.get("signals", {}) or {}
    _check_keys(signals_cfg, _SIGNALS_KEYS, "dataset.signals", source)
    _check_keys(view_cfg, _VISUALIZATION_KEYS, "visualization", source)
    scale = data_cfg.get("friction_scale", [0.5, 2.0])
    seed = int(data_cfg.get("seed", 20260917))
    rigid_reference = bool(raw.get("rigid_reference", True))
    split_cfg = data_cfg.get("split", {}) or {}
    _check_keys(split_cfg, _SPLIT_KEYS, "dataset.split", source)
    position_window_cfg = exc_cfg.get("position_window", {}) or {}
    if not isinstance(position_window_cfg, dict):
        raise ValueError(f"{source}: excitation.position_window must be a mapping of joint name -> [lo, hi]")
    position_window = tuple(
        (str(name), (float(bounds[0]), float(bounds[1]))) for name, bounds in position_window_cfg.items()
    )
    split = SplitPolicy(
        mode=str(split_cfg.get("mode", "contiguous")),
        test_robots=int(split_cfg.get("test_robots", 0)),
        val_robots=int(split_cfg.get("val_robots", 0)),
        test_trajectories=int(split_cfg.get("test_trajectories", 0)),
        val_trajectories=int(split_cfg.get("val_trajectories", 0)),
    )
    payload_cfg = raw.get("payload", {}) or {}
    _check_keys(payload_cfg, _PAYLOAD_KEYS, "payload", source)
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
    control_decimation = int(sim_cfg.get("control_decimation", 1))
    if control_decimation > 1 and not bool(sim_cfg.get("allow_control_decimation", False)):
        # A decimated command still cancels friction exactly (the applied
        # torque is held whole, subtraction included, for the window -- see
        # torque_runners.run_mujoco_torque's comment), but the residual
        # between the recorded label and rnea(achieved state) grows with
        # decimation and there is no guard yet relating it to a joint's
        # friction slope and inertia (R3_11 Sec 1, R3_12 Sec 2.5): a silently
        # unsafe value should not be one YAML edit away.
        raise ValueError(
            f"{source}: simulation.control_decimation={control_decimation} requires "
            "simulation.allow_control_decimation: true (no automatic stability guard exists yet; "
            "verify the residual stays acceptable for this asset before enabling it, see R3_12 Sec 1/2.5)"
        )
    config = DatasetConfig(
        schema_version=int(raw.get("schema_version", 1)),
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
            position_window=position_window,
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
        control_gains=ControlGainSampling(
            enabled=bool((sim_cfg.get("control_gains", {}) or {}).get("enabled", False)),
            natural_frequency=tuple(
                float(v) for v in (sim_cfg.get("control_gains", {}) or {}).get(
                    "natural_frequency", [sim_cfg.get("control_frequency", 25.0)] * 2
                )
            ),
            damping_ratio=tuple(
                float(v) for v in (sim_cfg.get("control_gains", {}) or {}).get(
                    "damping_ratio", [sim_cfg.get("control_damping_ratio", 1.0)] * 2
                )
            ),
        ),
        control_separation=ControlSeparationCheck(
            min_ratio=float((sim_cfg.get("control_separation", {}) or {}).get("min_ratio", 5.0)),
            action=str((sim_cfg.get("control_separation", {}) or {}).get("action", "warn")),
        ),
        controller=_controller_spec(controller_cfg),
        measurement=MeasurementModel(
            encoder_resolution=float(measurement_cfg.get("encoder_resolution", 0.0)),
            q_noise=float(measurement_cfg.get("q_noise", 0.0)),
            dq_noise=float(measurement_cfg.get("dq_noise", 0.0)),
            tau_noise_rel=float(measurement_cfg.get("tau_noise_rel", 0.0)),
            tau_noise_abs=float(measurement_cfg.get("tau_noise_abs", 0.0)),
            tau_gain_error=float(measurement_cfg.get("tau_gain_error", 0.0)),
            delay_samples=int(measurement_cfg.get("delay_samples", 0)),
            ft_noise_rel=(None if measurement_cfg.get("ft_noise_rel") is None
                          else float(measurement_cfg["ft_noise_rel"])),
            ft_noise_abs=(None if measurement_cfg.get("ft_noise_abs") is None
                          else float(measurement_cfg["ft_noise_abs"])),
        ),
        plant_extras=_plant_extras_sampling(extras_cfg),
        signals=SignalPolicy(
            position_side=str(signals_cfg.get("position_side", "link")),
            target=str(signals_cfg.get("target", "link_torque")),
            clean_columns=bool(signals_cfg.get("clean_columns", False)),
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
    require_probe_within_sampling_bound(config, source=source)
    return config


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


def sample_control_gains(
    sampling: ControlGainSampling, base_frequency: float, base_damping_ratio: float,
    dataset_seed: int, trajectory_seed_value: int,
) -> tuple[float, float]:
    """Derive one trajectory's ``(natural_frequency, damping_ratio)``.

    Stream ``(seed, 6, trajectory_seed)``, independent of every other stream
    (robots ``(seed, 1)``, payload ``(seed, 2)``, regime ``(seed, 3)``,
    rotor inertia ``(seed, 4)``, payload re-draws ``(seed, 5)``): enabling
    this does not perturb any of them.  ``generate()`` and
    ``run_identification_simulation.py`` must call this the same way so
    ``--tier eNN --trajectory k`` reproduces the dataset's gains exactly,
    same requirement as ``regime_excitation``.
    """
    if not sampling.enabled:
        return base_frequency, base_damping_ratio
    rng = np.random.default_rng((int(dataset_seed), 6, int(trajectory_seed_value)))
    natural_frequency = float(np.exp(
        rng.uniform(np.log(sampling.natural_frequency[0]), np.log(sampling.natural_frequency[1]))
    ))
    damping_ratio = float(rng.uniform(sampling.damping_ratio[0], sampling.damping_ratio[1]))
    return natural_frequency, damping_ratio


def sample_friction(base: FrictionModel, rng: np.random.Generator, scale_range: tuple[float, float]) -> FrictionModel:
    """Scale each joint's viscous and Coulomb coefficient log-uniformly."""
    low, high = np.log(scale_range[0]), np.log(scale_range[1])
    viscous = base.viscous * np.exp(rng.uniform(low, high, size=base.n_dof))
    coulomb = base.coulomb * np.exp(rng.uniform(low, high, size=base.n_dof))
    return FrictionModel(viscous, coulomb)


def elastic_time_step(transmission: TransmissionSpec, config: DatasetConfig) -> float:
    return min(config.max_time_step, transmission.required_time_step())


def _inertia_envelope_bounds(asset: AssetSpec, config: DatasetConfig) -> tuple[tuple[float, float], ...] | None:
    """Per-active-joint ``(lo, hi)`` for ``link_inertia_envelope``/``link_inertia_max``.

    ``None`` (the whole URDF range) unless ``config.excitation.position_window``
    is set, in which case the envelope is sampled over the same effective
    window the excitation trajectories live in, instead of the full URDF
    range (R4_03 Sec 1) -- for the iiwa (no window) this keeps the historical
    call exactly as it was.
    """
    if not config.excitation.position_window:
        return None
    lower, upper = effective_position_window(asset, config.excitation)
    return tuple(zip(lower.tolist(), upper.tolist()))


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
    natural_frequency: float | None = None,
    damping_ratio: float | None = None,
    draw: ControllerDraw | None = None,
    extras: PlantExtras | None = None,
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

    ``natural_frequency``/``damping_ratio`` default to
    ``config.control_frequency``/``config.control_damping_ratio`` when not
    given; pass the per-trajectory draw from ``sample_control_gains`` to use
    a randomized closed-loop gain instead (``config.control_gains``).

    ``draw`` carries the velocity-loop gains as well and supersedes those two
    arguments when given (``resolve_bag`` builds it); ``extras`` carries the
    round-5 non-Lagrangian plant effects.  Both default to round 4's
    behaviour: exact computed torque at the config's fixed gains, on a purely
    Lagrangian link side.

    Plant extras are rejected on the rigid tier on purpose.  That tier is the
    dataset's *analytic reference*: its recorded torque must keep satisfying
    ``rnea(achieved state)`` so that the Pinocchio cross-check and the
    MuJoCo/Newton-Featherstone backend comparison stay meaningful.  Link-side
    friction, a nonlinear spring and torque ripple all break that by
    construction, and on a rigid chain the first two have no separate link
    side to act on anyway.
    """
    natural_frequency = config.control_frequency if natural_frequency is None else natural_frequency
    damping_ratio = config.control_damping_ratio if damping_ratio is None else damping_ratio
    if draw is None:
        loop = config.controller.velocity_loop
        draw = ControllerDraw(
            natural_frequency=natural_frequency, damping_ratio=damping_ratio,
            position_gain=loop.position_gain[0], velocity_bandwidth=loop.velocity_bandwidth[0],
            integral_time=loop.integral_time[0],
        )
    natural_frequency, damping_ratio = draw.natural_frequency, draw.damping_ratio
    extras = NO_EXTRAS if extras is None else extras
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
        # What the controller is allowed to know.  `asset` here is always the
        # bare, payload-free asset; `asset_p` is the plant.
        nominal_asset = asset_p if config.controller.nominal.knows_payload else asset
        if tier.is_rigid:
            if not extras.is_empty:
                raise ValueError(
                    "simulation.plant_extras cannot be applied to the rigid reference tier "
                    "(its recorded torque must stay consistent with rnea(achieved state) for the "
                    "Pinocchio and backend cross-checks); set rigid_reference: false or drop the extras"
                )
            _check_control_rate(config.rigid_time_step)
            controller = build_controller(
                config.controller, draw, asset=asset_p, nominal_asset=nominal_asset,
                trajectory=trajectory, friction=friction,
            )
            runner = run_mujoco_torque if backend == "mujoco" else run_newton_torque
            result = runner(asset_p, trajectory, controller, time_step=config.rigid_time_step,
                            friction=friction, control_decimation=config.control_decimation, **view)
            result.update(transmission=None, time_step=config.rigid_time_step, payload=payload,
                         natural_frequency=natural_frequency, damping_ratio=damping_ratio,
                         controller_draw=draw, controller_mode=config.controller.mode,
                         extras=NO_EXTRAS)
            return result
        if link_inertia is None:
            link_inertia = link_inertia_envelope(
                asset_p, n_samples=config.transmission.inertia_samples,
                bounds=_inertia_envelope_bounds(asset_p, config),
            )
        transmission = tier.transmission(n_dof, link_inertia)
        time_step = elastic_time_step(transmission, config)
        _check_control_rate(time_step)
        controller = build_controller(
            config.controller, draw, asset=asset_p, nominal_asset=nominal_asset,
            trajectory=trajectory, friction=friction, transmission=transmission,
            # The PD and velocity-loop gains are sized from a nominal inertia.
            # A payload-*unaware* controller must not get it from the
            # payload-fitted envelope: that would hand it the payload back
            # through its gains.  Passing None makes `build_controller` derive
            # it from the nominal asset instead.
            joint_inertia=link_inertia[0] if config.controller.nominal.knows_payload else None,
        )
        runner = run_mujoco_elastic_torque if backend == "mujoco" else run_newton_elastic_torque
        result = runner(asset_p, trajectory, controller, transmission, time_step=time_step,
                        friction=friction, control_decimation=config.control_decimation,
                        extras=extras, **view)
        result.update(transmission=transmission, time_step=time_step, payload=payload,
                     natural_frequency=natural_frequency, damping_ratio=damping_ratio,
                     controller_draw=draw, controller_mode=config.controller.mode,
                     extras=extras)
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
    signals: SignalPolicy | None = None,
    measurement: MeasurementModel | None = None,
    measurement_seed: int = 0,
    sensor: ForceTorqueSensor | None = None,
) -> pd.DataFrame:
    """Flatten one rollout onto a uniform grid in the consumer's schema.

    ``dynamic_model_nn`` differentiates positions with a Savitzky-Golay filter
    and only does so when the time step is uniform, so every bag is resampled
    onto the same grid regardless of the step its tier required.

    ``signals`` selects which physical side ``q0..``/``dq0..`` and ``ft0..``
    carry (``SignalPolicy``, ``R5_01`` Amendment 2); ``measurement`` applies
    the sensor model to the recorded channels *after* resampling, since that
    is the rate a recorder runs at and the rate ``delay_samples`` counts in.
    Both default to round 4: the collocated link/link pair, perfect
    instruments.

    ``sensor`` is required when ``signals.target`` is ``ee_wrench_joint`` and is
    built from the asset when not given; building it costs one Pinocchio model,
    so a caller running many bags should build it once and pass it in.
    """
    names = tuple(asset.joint_names)
    signals = SignalPolicy() if signals is None else signals
    measurement = IDEAL_MEASUREMENT if measurement is None else measurement
    time = np.asarray(result["time"], dtype=float)
    grid = np.arange(time[0], time[-1] + 0.5 * resample_step, resample_step)

    def _on_grid(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        return np.column_stack([np.interp(grid, time, values[:, i]) for i in range(values.shape[1])])

    # The end-effector wrench, when that is the target.  Computed here rather
    # than inside the runners because it is a pure function of the achieved
    # link-side motion, which the rollout already records: the cell measures
    # what the bodies past it do, and nothing about it feeds back into the
    # simulation.  On the *resampled* grid, so it is the wrench a recorder at
    # that rate would log.
    wrench = wrench_target = None
    if signals.needs_sensor:
        if sensor is None:
            sensor_spec = SensorSpec.from_asset(asset)
            if sensor_spec is None:
                raise ValueError(
                    f"dataset.signals.target='{signals.target}' needs a force/torque cell, but asset "
                    f"{asset.name!r} declares no `force_torque_sensor: {{frame: ...}}`; add one to its "
                    "asset.yaml or choose another target"
                )
            sensor = ForceTorqueSensor(asset, sensor_spec)
        link_position = _on_grid(result["q_link"])
        link_velocity = _on_grid(result["dq_link"])
        # The consumer recomputes ddq from the recorded velocity, but the *cell*
        # experiences the true acceleration, so the simulated wrench uses the
        # rollout's own ddq rather than a differentiated copy of it.
        link_acceleration = _on_grid(result["ddq_link"])
        mapping = link_position if signals.position_side == "link" else _on_grid(result["q_motor"])
        wrench, wrench_target = sensor.rollout(
            link_position, link_velocity, link_acceleration, q_mapping=mapping,
        )

    clean = {
        "q_link": _on_grid(result["q_link"]),
        "dq_link": _on_grid(result["dq_link"]),
        "q_motor": _on_grid(result["q_motor"]),
        "dq_motor": _on_grid(result["dq_motor"]),
        "tau_motor": _on_grid(result["tau_motor"]),
        "tau_link": _on_grid(result["tau_link"]),
    }
    measured = measure_bag(
        measurement, seed=measurement_seed,
        q_motor=clean["q_motor"], dq_motor=clean["dq_motor"],
        q_link=clean["q_link"], dq_link=clean["dq_link"],
        tau_motor=clean["tau_motor"], tau_link=clean["tau_link"],
    )
    q = measured.q_link if signals.position_side == "link" else measured.q_motor
    dq = measured.dq_link if signals.position_side == "link" else measured.dq_motor
    if signals.target == "ee_wrench_joint":
        # One instrument, one realization (`R5_06` Sec 3).  The cell measures
        # the 6-axis wrench -- with its own noise and gain error, since it is
        # the same *instrument slot* as a link-side torque sensor but a
        # different transducer -- and the Jacobian is software applied after
        # it, at the configuration the encoders report.  Measuring the wrench
        # and the mapped target as two draws, as pass 2 did, gave a consumer
        # two independent realizations of one physical reading.
        measured_wrench = measure_channel(
            measurement, wrench, kind="force_cell", seed=measurement_seed,
        ).values
        target = sensor.joint_torques(q, measured_wrench)
    elif signals.target == "link_torque":
        target = measured.tau_link
    else:
        target = measured.tau_motor

    frame = pd.DataFrame({"t": grid, "bag": bag})
    for index in range(len(names)):
        frame[f"q{index}"] = q[:, index]
    for index in range(len(names)):
        frame[f"dq{index}"] = dq[:, index]
    for index in range(len(names)):
        frame[f"tau{index}"] = measured.tau_motor[:, index]
    # The learning target: whichever side `signals.target` names, one channel
    # per joint.  Its physical meaning is in the contract sidecar.
    for index in range(len(names)):
        frame[f"ft{index}"] = target[:, index]
    for index in range(len(names)):
        frame[f"q_motor{index}"] = measured.q_motor[:, index]
        frame[f"dq_motor{index}"] = measured.dq_motor[:, index]
        frame[f"q_link{index}"] = measured.q_link[:, index]
        frame[f"dq_link{index}"] = measured.dq_link[:, index]
    # Direct readout of the transmission deflection tau/k -- the cleanest
    # target for identifying the elastic parameters themselves (R3_01 Sec 3).
    # Taken from the *measured* positions, so it is the deflection an observer
    # could actually compute; the clean columns below hold the true one.
    for index in range(len(names)):
        frame[f"defl{index}"] = measured.q_motor[:, index] - measured.q_link[:, index]
    if wrench is not None:
        # The raw 6-axis reading beside the mapped target (R5_03 T-1): a
        # consumer may want the wrench itself, and a diagnostic needs it to
        # separate the mapping from the measurement.  `ft` is exactly
        # `J(q)^T w` of these two sets of columns (`R5_06` Sec 3).
        for index, column in enumerate(WRENCH_COLUMNS):
            frame[column] = measured_wrench[:, index]
        # How much of the wrench the mapping can still carry (`R5_07` T-15).
        # On a 6-DoF arm `J^T` is configuration dependent and near a
        # singularity whole wrench directions stop reaching joint space; this
        # is the per-sample record of that, so a bag can be scored on posture
        # instead of the excitation being assumed non-singular.
        frame["sigma_min_j"] = sensor.smallest_singular_value(q)
    if signals.clean_columns:
        # The exact simulator values on the same grid.  A noisy dataset is only
        # decomposable into physics and instrument if these survive (R5_00
        # Sec 6.2); they are debug columns and never model inputs.  The
        # reference is one of them: it is not a measurable signal at all, but
        # tracking error is the headline difference between controller modes and
        # a diagnostic reading a written file cannot recover it otherwise.
        clean = dict(clean, q_ref=_on_grid(result["q_ref"]), dq_ref=_on_grid(result["dq_ref"]))
        if wrench_target is not None:
            clean = dict(clean, ft=wrench_target)
        for key, values in clean.items():
            for index in range(len(names)):
                frame[f"{key}_clean{index}"] = values[:, index]
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
    draw: ControllerDraw | None = result.get("controller_draw")
    if draw is None:
        frame["control_natural_frequency"] = result.get("natural_frequency")
        frame["control_damping_ratio"] = result.get("damping_ratio")
    else:
        # Only the gains this mode uses; see ControllerDraw.as_row on why the
        # dataset columns and the manifest's record differ here.
        for column, value in draw.as_row(str(result.get("controller_mode", "exact_ct"))).items():
            frame[column] = value
    if not measurement.is_ideal:
        # The per-bag calibration errors, so a consumer or a diagnostic can
        # tell a gain error from a modelling error.
        for index, name in enumerate(names):
            frame[f"gain_motor__{name}"] = measured.gain_motor[index]
            frame[f"gain_link__{name}"] = measured.gain_link[index]
    extras: PlantExtras | None = result.get("extras")
    if extras is not None and not extras.is_empty:
        link_friction = extras.link_friction
        for index, name in enumerate(names):
            frame[f"link_viscous__{name}"] = 0.0 if link_friction is None else link_friction.viscous[index]
            frame[f"link_coulomb__{name}"] = 0.0 if link_friction is None else link_friction.coulomb[index]
        ripple = extras.torque_ripple
        frame["ripple_amplitude"] = 0.0 if ripple is None else float(ripple.amplitude)
        frame["ripple_order"] = np.nan if ripple is None else float(ripple.order)
    return frame


def control_separation_for_bag(
    asset: AssetSpec, tier: Tier, link_inertia: LinkInertia, link_inertia_max_value: np.ndarray,
    natural_frequency: float,
) -> tuple[float, str]:
    """Per-bag control/transmission separation ratio and its worst joint.

    Cheap (no simulation, just the tier's transmission preview), so
    ``generate()`` can call this for *every* bag while it still builds the
    work list -- before any bag is simulated -- and either raise immediately
    (``action: error``, R4_10 Sec 2.3: raising inside a worker meant bags
    already queued ahead of it, possibly minutes to hours of work with
    ``--jobs N``, would already have run) or record the ratio for the
    aggregated warning.
    """
    n_dof = len(asset.joint_names)
    preview = tier.transmission(n_dof, link_inertia)
    ratios = control_separation_ratio(
        preview.stiffness, preview.rotor_inertia, link_inertia_max_value, natural_frequency,
    )
    worst = int(np.argmin(ratios))
    return float(ratios[worst]), asset.joint_names[worst]


def raise_on_control_separation_violation(
    offending: Sequence[tuple[str, float, str]], min_ratio: float,
) -> None:
    """Raise once for every bag ``generate()`` found below ``min_ratio``.

    Split out from ``generate()`` so the "action: error raises before any bag
    is simulated, listing every offender" contract (R4_02 Sec 7, R4_10 Sec
    2.3) is unit-testable on a plain list of ``(bag, ratio, joint)`` tuples,
    with no trajectory optimisation (hence no Pinocchio) needed to reach it.
    """
    if not offending:
        return
    detail = "; ".join(f"{bag!r} ratio={ratio:.2f} joint={joint!r}" for bag, ratio, joint in offending)
    raise ValueError(
        f"{len(offending)} bag(s) violate simulation.control_separation.min_ratio="
        f"{min_ratio:g} before any bag was simulated: {detail}; "
        "lower control_gains.natural_frequency or raise the sampled stiffness"
    )


#: One force/torque cell per (asset name, URDF path, target) inside a worker
#: process.  `_run_bag` runs in a process pool, so this is per-worker and never
#: shared.
#:
#: The URDF path is part of the key because an asset's *name* does not identify
#: its inertias: `payload_asset` and `scripts/sweep_ft_payload.py` both yield an
#: asset with the same name and a rewritten URDF, and keying on the name alone
#: silently hands the second caller the first one's cell.  That cost the T-14
#: sweep its first run -- every payload mass was measured with the 0 kg cell,
#: which reads identically zero (`R5_08` D-13).
_SENSOR_CACHE: dict[tuple[str, str, str], ForceTorqueSensor] = {}


def _sensor_for(asset: AssetSpec, signals: SignalPolicy) -> ForceTorqueSensor | None:
    """The asset's force/torque cell, built once per process.

    Building it costs a Pinocchio model, and every bag of a build needs the same
    one, so it is cached on the asset name rather than rebuilt per bag.  Returns
    ``None`` unless the signal policy actually needs it, so nothing is built for
    a joint-torque target.
    """
    if not signals.needs_sensor:
        return None
    key = (asset.name, str(asset.urdf_path), signals.target)
    if key not in _SENSOR_CACHE:
        spec = SensorSpec.from_asset(asset)
        if spec is None:
            raise ValueError(
                f"dataset.signals.target='{signals.target}' needs a force/torque cell, but asset "
                f"{asset.name!r} declares no `force_torque_sensor: {{frame: ...}}` in its asset.yaml"
            )
        _SENSOR_CACHE[key] = ForceTorqueSensor(asset, spec)
    return _SENSOR_CACHE[key]


def _feedback_ratio(feedforward: np.ndarray, feedback: np.ndarray) -> float | None:
    """``mean|feedback| / mean|feedforward|``, or ``None`` when there is no feedforward."""
    reference = float(np.mean(np.abs(np.asarray(feedforward, dtype=float))))
    if reference <= 0.0:
        return None
    return float(np.mean(np.abs(np.asarray(feedback, dtype=float))) / reference)


def _run_bag(args: tuple) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Execute one bag and return ``(frame, record)``.

    A plain top-level function taking one tuple argument, so it can be sent
    to a process pool: every argument is a simple dataclass/array (config,
    the asset spec, the tier, ...), never a built simulator model, which is
    what makes it picklable across the process boundary.  Bags are fully
    independent -- their RNG streams are keyed on ``(seed, index)``, not on
    call order -- so worker order never affects the result.  The control-
    separation ratio/joint are computed by the caller (``generate()``)
    *before* any bag is dispatched, so an ``action: error`` violation is
    caught before the pool starts, not inside a worker (R4_10 Sec 2.3); this
    function only carries them through to the manifest record.
    """
    (asset, trajectory, tier, backend, friction, config, link_inertia, payload,
     split, bag, bag_index, traj_index, friction_index, control_gains, link_inertia_max_value,
     separation_ratio, separation_joint, draw, extras) = args
    natural_frequency, damping_ratio = control_gains
    result = run_condition(asset, trajectory, tier, backend, friction, config,
                           link_inertia=link_inertia, payload=payload,
                           natural_frequency=natural_frequency, damping_ratio=damping_ratio,
                           draw=draw, extras=None if tier.is_rigid else extras)
    frame = rollout_frame(asset, trajectory, result, bag=bag, tier=tier, backend=backend,
                          friction=friction, resample_step=config.sample_time_step, split=split,
                          signals=config.signals, measurement=config.measurement,
                          sensor=_sensor_for(asset, config.signals),
                          # Keyed on the bag index, so a bag's noise is the same
                          # however many bags ran before it and whatever the
                          # worker order (R4_10 Sec 2.3's reproducibility rule).
                          measurement_seed=config.seed + 1_000_003 * bag_index)
    # Effort limits come from the bare asset's URDF <limit effort=...> tags,
    # unaffected by payload injection (a payload changes link inertia, not
    # the motor's rated torque) -- a payload heavy/offset enough, combined
    # with a high regime acceleration, is the most likely way a bag becomes
    # physically meaningless without anything else catching it (R3_10 Sec 3.3).
    effort_limits = np.asarray(
        [joint.effort if joint.effort else np.inf for joint in asset.resolve_active_joints()]
    )
    peak_torque_ratio = float(np.max(np.abs(np.asarray(result["tau_motor"])) / effort_limits[None, :]))
    # The reference trajectory is validated collision-free at build time
    # (optimize_excitation), but tracking error means the *achieved* path
    # can differ, especially on a soft elastic robot -- flagged, not
    # dropped, same policy as the backend comparison (R4_14 Sec 2.3).
    margin = float(asset.metadata.get("collision", {}).get("margin", 0.0))
    max_joint_step = float(asset.metadata.get("collision", {}).get("max_joint_step", 0.05))
    with payload_asset(asset, payload) as asset_p:
        achieved_report = PortableKinematics(asset_p).validate_path(
            np.asarray(result["q_link"]), margin=margin, max_joint_step=max_joint_step,
        )
    record = {
        "bag": bag, "bag_index": bag_index, "trajectory": traj_index, "tier": tier.name, "split": split,
        **describe_transmission(result["transmission"], result["time_step"]),
        "payload": None if payload is None or payload.is_empty else payload.as_dict(),
        "friction_index": friction_index, "backend": backend,
        "solver": result.get("solver"), "wall_time": float(result.get("wall_time", 0.0)),
        "samples": int(len(result["time"])),
        "trajectory_digest": trajectory.digest(),
        "trajectory_signal_digest": trajectory.signal_digest(),
        "condition_number": float(trajectory.metadata["condition_number"]),
        "exc_base_frequency": float(trajectory.metadata["base_frequency"]),
        "exc_max_acceleration": float(trajectory.metadata["max_acceleration"]),
        "exc_velocity_fraction": float(trajectory.metadata["velocity_fraction"]),
        "exc_probe_top_hz": float(trajectory.metadata.get("probe_top_hz", 0.0)),
        "control_natural_frequency": float(natural_frequency),
        "control_damping_ratio": float(damping_ratio),
        "controller": describe_controller(config.controller, draw),
        "control_position_gain": float(draw.position_gain),
        "control_velocity_bandwidth": float(draw.velocity_bandwidth),
        "control_integral_time": float(draw.integral_time),
        "plant_extras": None if extras is None or extras.is_empty or tier.is_rigid else extras.describe(),
        "viscous": friction.viscous.tolist(), "coulomb": friction.coulomb.tolist(),
        "tracking_rms": float(np.sqrt(np.mean((np.asarray(result["q_link"]) - np.asarray(result["q_ref"])) ** 2))),
        "max_deflection": float(np.abs(np.asarray(result["q_motor"]) - np.asarray(result["q_link"])).max()),
        # Per-bag ringing measure (R5_Q Q-6): the velocity loop's range now
        # straddles the transmission mode on purpose, so some bags ring.  The
        # RMS -- not the peak -- is what compares across bags, and `generate()`
        # takes the ratio to the median over bags to flag the outliers.
        "deflection_rms": float(np.sqrt(np.mean(
            (np.asarray(result["q_motor"]) - np.asarray(result["q_link"])) ** 2
        ))),
        # None rather than a number divided by ~zero: a model-free PD
        # controller has no feedforward at all, and a ratio of 1e12 would read
        # as a measurement rather than as "not applicable" (R5_02 Sec 2).
        "feedback_ratio": _feedback_ratio(result["tau_feedforward"], result["tau_feedback"]),
        "peak_torque_ratio": peak_torque_ratio,
        "control_separation_min_ratio": separation_ratio,
        "control_separation_joint": separation_joint,
        "achieved_min_clearance_m": float(achieved_report.minimum_distance),
    }
    return frame, record


def sample_all_payloads(
    config: DatasetConfig, elastic_tiers: Sequence[Tier],
) -> tuple[dict[str, Payload], dict[tuple[str, int], Payload]]:
    """Draw every elastic tier's payload(s), in one deterministic batch.

    ``per: "robot"`` draws one payload per tier (a robot ships with a tool);
    ``"trajectory"`` draws one per ``(tier, index)`` pair, the same
    ``(index, tier)`` nesting order every caller must use for a draw to
    reproduce.  ``elastic_tiers`` must be built the same way by every caller
    (``[t for t in config.tiers if not t.is_rigid]``): the ``"trajectory"``
    stream is a single sequential draw over the whole batch, so it is only
    reproducible in isolation for ``per: "robot"`` (see ``resolve_bag``).
    """
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
    return payload_by_tier, payload_by_key


def payload_for(
    config: DatasetConfig, payload_by_tier: Mapping[str, Payload], payload_by_key: Mapping[tuple[str, int], Payload],
    tier: Tier, traj_index: int,
) -> Payload | None:
    if tier.is_rigid:
        return Payload()
    return payload_by_tier[tier.name] if config.payload.per == "robot" else payload_by_key[(tier.name, traj_index)]


@dataclass(frozen=True)
class ResolvedBag:
    """One ``(tier, trajectory index)`` bag's payload-aware trajectory and gains."""

    payload: Payload | None
    excitation: FourierExcitationConfig
    natural_frequency: float
    damping_ratio: float
    trajectory: MaterializedTrajectory
    # Round 5: the full controller draw (the two gains above plus the
    # velocity loop's three) and the bag's plant extras.  Kept on the same
    # object so that `run_identification_simulation.py` reproduces a bag's
    # controller by calling `resolve_bag`, exactly as it already reproduces
    # its trajectory and regime.
    draw: ControllerDraw | None = None
    extras: PlantExtras | None = None


def resolve_bag(
    config: DatasetConfig, asset: AssetSpec, tier: Tier, index: int, payload: Payload | None,
    *, kinematics_for: Callable[[Payload | None], PortableKinematics] | None = None,
) -> ResolvedBag:
    """Derive one bag's trajectory, excitation and control gains.

    The single source of truth for what a dataset bag actually is:
    ``generate()`` and ``run_identification_simulation.py`` both call this
    the same way, so ``--tier eNN --trajectory k`` reproduces the dataset's
    trajectory exactly -- scored against the *same payload-fitted* collision
    geometry, not the bare asset, whenever ``payload`` is non-empty (R3_14
    Sec 1.1: before this helper existed, the debug script always optimized
    against bare kinematics and ran the rollout with no payload at all,
    while ``generate()`` scores every trajectory against payload-fitted
    geometry whenever ``trajectories_per_robot`` is set, the shipped
    default -- silently reproducing the wrong trajectory, and a payload-free
    rollout, for every robot whose payload changed a collision rejection).

    ``payload`` is the caller's responsibility to resolve first (via
    ``sample_all_payloads``/``payload_for``, or ``Payload()`` when this bag
    is deliberately payload-free, e.g. a shared trajectory under
    ``trajectories_per_robot: false``) -- this function does not re-derive
    it, so it never disagrees with whatever the manifest says that bag's
    payload was. ``kinematics_for`` defaults to a fresh, uncached
    ``PortableKinematics`` per call, torn down before returning;
    ``generate()`` passes its cached, longer-lived version so a "per: robot"
    dataset does not rebuild the same payload-fitted kinematics once per
    trajectory.
    """
    traj_seed = trajectory_seed(config, tier, index)
    excitation = regime_excitation(config.excitation, config.regime, config.seed, traj_seed)
    natural_frequency, damping_ratio = sample_control_gains(
        config.control_gains, config.control_frequency, config.control_damping_ratio, config.seed, traj_seed,
    )
    position_gain, velocity_bandwidth, integral_time = sample_velocity_loop(
        config.controller, config.seed, traj_seed,
    )
    draw = ControllerDraw(
        natural_frequency=natural_frequency, damping_ratio=damping_ratio,
        position_gain=position_gain, velocity_bandwidth=velocity_bandwidth, integral_time=integral_time,
    )
    extras = plant_extras_for_bag(
        config.plant_extras, len(asset.joint_names), config.seed, traj_seed, robot=tier.name,
    )

    stack = contextlib.ExitStack()
    try:
        if kinematics_for is not None:
            kinematics = kinematics_for(payload)
        else:
            asset_p = stack.enter_context(payload_asset(asset, payload))
            kinematics = PortableKinematics(asset_p)
        trajectory = optimize_excitation(
            asset, excitation, seed=traj_seed, n_candidates=config.candidates, kinematics=kinematics,
        )
    finally:
        stack.close()

    return ResolvedBag(payload=payload, excitation=excitation, natural_frequency=natural_frequency,
                       damping_ratio=damping_ratio, trajectory=trajectory, draw=draw, extras=extras)


def generate(
    config: DatasetConfig, asset: AssetSpec, *, verbose: bool = True, jobs: int = 1,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Build the whole dataset and return ``(frame, manifest, backend_comparison)``."""
    warn_if_probe_outruns_differentiation(config)
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
    payload_by_tier, payload_by_key = sample_all_payloads(config, elastic_tiers)

    def _payload_for(tier: Tier, traj_index: int) -> Payload | None:
        return payload_for(config, payload_by_tier, payload_by_key, tier, traj_index)

    envelope_bounds = _inertia_envelope_bounds(asset, config)
    link_inertia = None
    # link_inertia_envelope is asset-*and*-payload dependent; cache it keyed
    # on the payload so a "per: robot" dataset does at most robots + 1 CRBA
    # sweeps instead of one per bag, and so the manifest's "robots" summary
    # below can report the *actual* payload-fitted damping/step per robot
    # instead of a payload-free number that would silently disagree with
    # that same robot's per-bag records (R3_10 task table, row C4).
    envelope_cache: dict[tuple, LinkInertia] = {}
    # Separate cache for the *maximum* link inertia the control/transmission
    # separation check needs (R4_03 Sec 2): the worst, softest-effective
    # corner, not the median/floor pair damping and the integration step use.
    envelope_max_cache: dict[tuple, np.ndarray] = {}

    def _envelope_for(payload: Payload | None) -> LinkInertia:
        key = () if payload is None or payload.is_empty else (payload.mass, payload.offset, payload.size)
        if key not in envelope_cache:
            with payload_asset(asset, payload) as asset_p:
                envelope_cache[key] = link_inertia_envelope(
                    asset_p, n_samples=config.transmission.inertia_samples, bounds=envelope_bounds,
                )
        return envelope_cache[key]

    def _envelope_max_for(payload: Payload | None) -> np.ndarray:
        key = () if payload is None or payload.is_empty else (payload.mass, payload.offset, payload.size)
        if key not in envelope_max_cache:
            with payload_asset(asset, payload) as asset_p:
                envelope_max_cache[key] = link_inertia_max(
                    asset_p, n_samples=config.transmission.inertia_samples, bounds=envelope_bounds,
                )
        return envelope_max_cache[key]

    robots: list[dict[str, Any]] = []
    if elastic_tiers:
        # per: "robot" has one stable payload per tier, so its summary can be
        # fully consistent with that tier's per-bag records; per:
        # "trajectory" does not (a robot's payload varies bag to bag), so
        # its summary stays nominal/payload-free, exactly as documented in
        # the per-bag payload_* columns being the ground truth there.
        link_inertia = link_inertia_envelope(asset, n_samples=config.transmission.inertia_samples, bounds=envelope_bounds)
        envelope_max_cache[()] = link_inertia_max(asset, n_samples=config.transmission.inertia_samples, bounds=envelope_bounds)
        if verbose:
            print(f"link inertia M_ii [kg m^2]: median {np.array2string(link_inertia[0], precision=4)}")
            print(f"                            floor  {np.array2string(link_inertia[1], precision=4)}")
        for tier in elastic_tiers:
            tier_payload = payload_by_tier.get(tier.name) if config.payload.per == "robot" else None
            tier_link_inertia = _envelope_for(tier_payload) if tier_payload is not None else link_inertia
            transmission = tier.transmission(n_dof, tier_link_inertia)
            entry = {"name": tier.name, **describe_transmission(transmission, elastic_time_step(transmission, config))}
            if tier_payload is not None:
                entry["payload"] = None if tier_payload.is_empty else tier_payload.as_dict()
            robots.append(entry)
            if verbose:
                print(f"robot {tier.name}: k [Nm/rad] {np.array2string(transmission.stiffness, precision=0, floatmode='fixed')}"
                      f"\n           zeta {np.array2string(transmission.damping_ratio, precision=3)}"
                      f" | top mode {transmission.natural_frequency().max():.0f} Hz"
                      f" | step {robots[-1]['time_step']:.1e} s")

    # Contacts are disabled during a rollout, so a trajectory that collides
    # would be simulated straight through the geometry: validate here, where
    # the trajectory is chosen, rather than trusting it afterwards.
    kinematics = PortableKinematics(asset)
    # A payload adds real geometry (a box, 5-25 cm on a side) that the bare
    # asset never saw; a trajectory validated collision-free against the bare
    # asset is not necessarily collision-free once one is fitted (R3_10 Sec
    # 3.2).  Cached by payload key, same bound as the link-inertia envelope
    # cache.  Each injected payload URDF is a temp file that ``payload_asset``
    # deletes when its ``with`` block exits; ``PortableKinematics`` re-parses
    # that path lazily (e.g. ``.joint_names``), so a cache entry built from a
    # closed ``with`` block would point at a file that no longer exists --
    # keep every unique payload's temp URDF alive for the duration of
    # ``generate()`` via an ExitStack, closed in the ``finally`` below.
    kinematics_cache: dict[tuple, PortableKinematics] = {(): kinematics}
    kinematics_stack = contextlib.ExitStack()

    def _kinematics_for(payload: Payload | None) -> PortableKinematics:
        key = () if payload is None or payload.is_empty else (payload.mass, payload.offset, payload.size)
        if key not in kinematics_cache:
            asset_p = kinematics_stack.enter_context(payload_asset(asset, payload))
            kinematics_cache[key] = PortableKinematics(asset_p)
        return kinematics_cache[key]

    trajectories: dict[tuple[str, int], MaterializedTrajectory] = {}
    control_gains: dict[tuple[str, int], tuple[float, float]] = {}
    controller_draws: dict[tuple[str, int], ControllerDraw] = {}
    bag_extras: dict[tuple[str, int], PlantExtras] = {}
    try:
        for tier in (config.tiers if config.trajectories_per_robot else config.tiers[:1]):
            key = tier.name if config.trajectories_per_robot else ""
            for index in range(config.n_trajectories):
                # When every tier gets its own trajectory, that tier's payload
                # (whether drawn "per: robot" -- constant across its
                # trajectories -- or "per: trajectory" -- one per (tier,
                # index)) is already known here: score candidates against the
                # *payload-fitted* geometry directly, so a collision just
                # rejects a candidate inside optimize_excitation instead of
                # aborting the whole build after the fact (R3_12 Sec 2.2).
                # With a single trajectory shared across every tier
                # (``trajectories_per_robot: false``), no one payload applies,
                # so this stays payload-free and the post-hoc check below is
                # the only guard.
                payload = _payload_for(tier, index) if config.trajectories_per_robot else Payload()
                resolved = resolve_bag(config, asset, tier, index, payload, kinematics_for=_kinematics_for)
                trajectories[(key, index)] = resolved.trajectory
                control_gains[(key, index)] = (resolved.natural_frequency, resolved.damping_ratio)
                controller_draws[(key, index)] = resolved.draw
                bag_extras[(key, index)] = resolved.extras
                if verbose:
                    metadata = resolved.trajectory.metadata
                    label = f"trajectory {index}" + (f" for {tier.name}" if config.trajectories_per_robot else "")
                    print(f"{label}: condition={metadata['condition_number']:.0f}"
                          f" ({metadata['collision_rejections']} candidates rejected for collision)")

        split_labels = assign_splits(config.tiers, config.split)

        # Post-hoc safety net for the case the loop above could not already
        # guarantee: a shared (``trajectories_per_robot: false``) trajectory
        # combined with a payload, which was necessarily scored payload-free
        # above.  A single bad payload draw should not abort a 60-bag build,
        # so re-draw a handful of alternative payloads for that (tier,
        # trajectory) before giving up.
        validated: dict[tuple[str, int], bool] = {}
        redraw_rng = np.random.default_rng((config.seed, 5))

        def _collision_free(trajectory: MaterializedTrajectory, payload: Payload) -> bool:
            margin = float(asset.metadata.get("collision", {}).get("margin", 0.0))
            max_joint_step = float(asset.metadata.get("collision", {}).get("max_joint_step", 0.05))
            report = _kinematics_for(payload).validate_path(trajectory.position, margin=margin, max_joint_step=max_joint_step)
            return report.valid

        work: list[tuple] = []
        offending: list[tuple[str, float, str]] = []
        for bag_index, (traj_index, tier, friction_index, backend) in enumerate(iter_conditions(config)):
            trajectory = trajectories[(tier.name if config.trajectories_per_robot else "", traj_index)]
            friction = frictions[friction_index]
            bag = f"t{traj_index}_{tier.name}_f{friction_index}_{backend}"
            split = split_labels[tier.name]
            payload = _payload_for(tier, traj_index)
            key = (tier.name, traj_index)
            if not config.trajectories_per_robot and payload is not None and not payload.is_empty and key not in validated:
                if not _collision_free(trajectory, payload):
                    for attempt in range(5):
                        candidate = sample_payloads(config.payload, int(redraw_rng.integers(0, 2**31 - 1)), 1)[0]
                        if _collision_free(trajectory, candidate):
                            payload = candidate
                            if config.payload.per == "robot":
                                payload_by_tier[tier.name] = candidate
                            else:
                                payload_by_key[key] = candidate
                            break
                    else:
                        raise ValueError(
                            f"tier {tier.name!r} trajectory {traj_index}: no payload in 6 draws (1 original + 5 "
                            f"retries) leaves this shared trajectory collision-free; this pairing is not usable"
                        )
                validated[key] = True
            link_inertia_for_bag = _envelope_for(payload) if not tier.is_rigid else None
            link_inertia_max_for_bag = _envelope_max_for(payload) if not tier.is_rigid else None
            resolved_key = (tier.name if config.trajectories_per_robot else "", traj_index)
            gains = control_gains[resolved_key]
            draw = controller_draws[resolved_key]
            extras = bag_extras[resolved_key]
            separation_ratio = separation_joint = None
            if not tier.is_rigid:
                separation_ratio, separation_joint = control_separation_for_bag(
                    asset, tier, link_inertia_for_bag, link_inertia_max_for_bag,
                    # The bandwidth to separate from the transmission mode is
                    # the *fastest* loop the controller closes, which for
                    # `velocity_pi` is the drive's inner velocity loop and not
                    # the error-dynamics frequency (R5_02 Sec 2.4).
                    effective_bandwidth(config.controller, draw),
                )
                # The bound exists because `SeaMotorController`'s feedback
                # linearization and the plant's own transmission mode have to
                # separate in frequency.  A model-free loop does no
                # linearization, and a real drive's velocity loop genuinely
                # does sit near the transmission mode -- that is why a real
                # series-elastic platform rings.  So the ratio is still
                # computed and recorded for every bag, but only enforced where
                # its rationale applies (R5_02 Sec 2.4).
                if (separation_ratio < config.control_separation.min_ratio
                        and not config.controller.is_model_free):
                    offending.append((bag, separation_ratio, separation_joint))
            work.append((asset, trajectory, tier, backend, friction, config, link_inertia_for_bag, payload,
                        split, bag, bag_index, traj_index, friction_index, gains, link_inertia_max_for_bag,
                        separation_ratio, separation_joint, draw, extras))
    finally:
        kinematics_stack.close()

    # Checked once the whole work list is built, *before* a single bag is
    # dispatched to a worker: with "action: error", raising from inside
    # _run_bag would let bags queued ahead of the first offender in the pool
    # already run, sometimes for minutes to hours (R4_10 Sec 2.3).
    if config.control_separation.action == "error":
        raise_on_control_separation_violation(offending, config.control_separation.min_ratio)

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
            if record["peak_torque_ratio"] > 0.8:
                print(f"    warning: bag {record['bag']!r} peak |tau| is "
                      f"{record['peak_torque_ratio']:.0%} of the effort limit on its worst joint")

    # Ringing: the bags whose deflection RMS stands far above the median for
    # this build.  Flagged, never dropped (R5_Q Q-6): a bag that rings is where
    # the damping ratio is observable at all, so it is the most informative bag
    # in the set, not a failure -- but a bag that rings 10x the median is more
    # likely an unstable loop draw than a rich one, and a build should say so.
    ringing = [r["deflection_rms"] for r in records if not np.isnan(r.get("deflection_rms", np.nan))]
    median_deflection = float(np.median(ringing)) if ringing else 0.0
    for record in records:
        value = record.get("deflection_rms")
        record["deflection_ratio_to_median"] = (
            None if not median_deflection or value is None else float(value / median_deflection)
        )
        record["diverged"] = bool(value is not None and not np.isfinite(value))
    loud = [r for r in records if (r["deflection_ratio_to_median"] or 0.0) > 10.0]
    if loud:
        worst = max(loud, key=lambda r: r["deflection_ratio_to_median"])
        message = (
            f"{len(loud)}/{len(records)} bags' deflection RMS is more than 10x this build's median "
            f"({median_deflection:.3g}); worst is bag {worst['bag']!r} at "
            f"{worst['deflection_ratio_to_median']:.1f}x with loop gains "
            f"kp={worst['control_position_gain']:.2f} omega_v={worst['control_velocity_bandwidth']:.1f}. "
            "Flagged, not dropped: a ringing bag is where the damping ratio becomes observable "
            "(R5_Q Q-6), but check it is ringing rather than diverging"
        )
        warnings.warn(message, stacklevel=2)
        if verbose:
            print(f"warning: {message}")

    below = [r for r in records if r["control_separation_min_ratio"] is not None
             and r["control_separation_min_ratio"] < config.control_separation.min_ratio]
    if below and config.controller.is_model_free:
        # Recorded, not warned about: see the enforcement comment above.
        below = []
    if below:
        worst = min(below, key=lambda r: r["control_separation_min_ratio"])
        message = (f"{len(below)}/{len(records)} bags have a control/transmission separation ratio "
                   f"below min_ratio={config.control_separation.min_ratio:g}; worst is bag {worst['bag']!r} "
                   f"at {worst['control_separation_min_ratio']:.2f} on joint {worst['control_separation_joint']!r}")
        # Emitted unconditionally (R4_10 Sec 2.2: a --quiet build previously
        # left this violation visible only in the manifest, with no trace on
        # stderr/stdout); the print is kept as well for verbose runs.
        warnings.warn(message, stacklevel=2)
        if verbose:
            print(f"warning: {message}")

    collision_margin = float(asset.metadata.get("collision", {}).get("margin", 0.0))
    too_close = [r for r in records if r["achieved_min_clearance_m"] < collision_margin]
    if too_close:
        worst = min(too_close, key=lambda r: r["achieved_min_clearance_m"])
        message = (f"{len(too_close)}/{len(records)} bags' achieved path (not the reference, which is "
                   f"validated collision-free at build time) comes closer than margin={collision_margin:g} m; "
                   f"worst is bag {worst['bag']!r} at {worst['achieved_min_clearance_m']:.4f} m "
                   "(R4_14 Sec 2.3: tracking error, not a code defect -- flagged, not dropped)")
        warnings.warn(message, stacklevel=2)
        if verbose:
            print(f"warning: {message}")

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
        "controller": describe_controller(config.controller),
        "measurement": config.measurement.describe(),
        "differentiation": differentiation_policy(config),
        "signals": config.signals.describe(),
        "force_torque_sensor": (
            None if not config.signals.needs_sensor
            else _sensor_for(asset, config.signals).describe()
        ),
        "plant_extras": None if config.plant_extras.is_empty else asdict(config.plant_extras),
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
        "target": f"ft0..ft{{n-1}} = {_TARGET_SEMANTICS[config.signals.target]}",
        "input": (
            f"q0..q{{n-1}}, dq0..dq{{n-1}} = {config.signals.position_side}-side position and velocity; "
            "tau0..tau{n-1} = commanded motor torque (motor effort)"
        ),
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
            # What the split does *not* test, said out loud (`R5_07` T-16,
            # defect D-12): the stiffness, damping and reflected inertia vary
            # per robot, so the holdout tests those; the link-friction
            # background only varies per robot on the `per_robot` tier, and on
            # `fixed` (the default) every held-out robot carries exactly the
            # training robots' friction law.
            "tests": sorted(
                ["transmission stiffness", "transmission damping", "reflected inertia"]
                + (["link friction"] if config.plant_extras.link_friction_tier == "per_robot" else [])
            ) if config.split.mode == "holdout_robots" else [],
            "does_not_test": (
                ["link friction: the same law on every robot "
                 f"(plant_extras.link_friction.tier: {config.plant_extras.link_friction_tier})"]
                if config.split.mode == "holdout_robots"
                and config.plant_extras.has_link_friction
                and config.plant_extras.link_friction_tier != "per_robot"
                else []
            ),
        },
        "records": records,
    }
    return frame, manifest, comparison


#: Physical meaning of each ``dataset.signals.target`` choice, for the
#: manifest and the consumer contract.
_TARGET_SEMANTICS = {
    "link_torque": "link-side joint torque [Nm]",
    "motor_torque": "motor-side (commanded) joint torque [Nm]",
    "ee_wrench_joint": (
        "end-effector force/torque sensor wrench mapped to joint space, "
        "tau = J(q_sensor)^T w [Nm]"
    ),
}
#: What kind of instrument produces each target, for the consumer contract.
#: The three round-5 platforms genuinely have three different ones: the iiwa's
#: joint torque sensors, FMRR's end-effector cell, and (nothing) on the UR10
#: (R5_03 T-1).
_TARGET_KINDS = {
    "link_torque": "joint_torque_sensor",
    "motor_torque": "motor_current",
    "ee_wrench_joint": "ee_wrench_joint",
}

#: Savitzky-Golay polynomial order the consumer uses.  Only the window is
#: chosen per dataset; the order is a shape choice, not a bandwidth one.
SG_POLY_ORDER = 3


def recommended_sg_window(sample_rate: float, probe_top_hz: float, *, poly: int = SG_POLY_ORDER) -> int:
    """Odd Savitzky-Golay window whose -3 dB point sits near 1.5x the probe top.

    Answers `R5_03` T-4 / `R5_02` H-4.  The consumer differentiates a noisy
    velocity with a Savitzky-Golay filter, and round 4's window was fixed at 11
    samples regardless of what the probe put in the data: at a 2 ms grid that is
    a 22 ms window, a low-pass well below the probe's top line, so the modal
    content the probe was added for was filtered straight out again and
    reappeared as an unexplainable residual (`R5_02` Sec 6.2 measures it at
    89-99 % of the rigid tier's residual).

    The artefact is *kept* -- on the real robot one also differentiates a noisy
    encoder, and removing it in simulation would break the owner's "same
    signals" rule -- but it is now explicit and consistent: the window is
    reported in the contract so the consumer uses one that passes the band the
    dataset actually contains.

    The cutoff of a Savitzky-Golay differentiator of order 3 is approximately
    ``f_c ~ 0.45 * rate / window`` (Schafer 2011's tabulation, close enough for
    choosing an odd integer).  Solving for ``f_c = 1.5 * probe_top`` gives the
    window below, clamped to at least ``poly + 2`` so the fit stays defined.
    """
    if sample_rate <= 0.0:
        raise ValueError("sample_rate must be positive")
    if probe_top_hz <= 0.0:
        # No probe: keep the historical window, which is what every round-3/4
        # dataset was consumed with.
        return 11
    window = int(round(0.45 * sample_rate / (1.5 * probe_top_hz)))
    window = max(window, poly + 2)
    return window if window % 2 == 1 else window + 1


def warn_if_probe_outruns_differentiation(config: DatasetConfig) -> None:
    """Warn when no Savitzky-Golay window can pass this config's probe band.

    `R5_03` T-4 asked for a hard config-load requirement,
    ``probe_top_hz <= 0.25 * sample_rate``.  It is a warning at *build* time
    instead, for one reason: **both shipped round-4 configs violate it** (iiwa
    190 Hz and UR10 150 Hz against a 500 Hz grid, 0.38 and 0.30 of the rate).
    Raising at load would stop them loading and `R4_06 B5` asserts they load
    warning-free, which would break the round-4 invariant this round has kept
    everywhere else; capping their probe is a numeric-prior change, and rule 1
    of `R4_00 Sec 7` reserves that for the architect (`R5_05` Sec 5, D-8).

    So the fact is surfaced where it costs something -- when a dataset is
    actually built -- and carried machine-readably in the contract's
    ``differentiation.probe_resolvable`` flag.
    """
    policy = differentiation_policy(config)
    if policy["probe_resolvable"]:
        return
    top = policy["probe_top_hz"]
    rate = policy["sample_rate_hz"]
    # The cap that would actually make it resolvable is the *filter's* floor,
    # 0.45 * rate / (poly + 2), not 0.25 * rate: pass 3 found the two bounds
    # disagree and the round had been quoting the looser one (`R5_08` Sec 2).
    resolvable_top = 0.45 * rate / (SG_POLY_ORDER + 2)
    warnings.warn(
        f"probe top frequency {top:g} Hz is above the widest passband a Savitzky-Golay "
        f"derivative of order {SG_POLY_ORDER} has at {rate:g} Hz ({resolvable_top:.1f} Hz, the "
        f"filter's five-sample floor), so no window both smooths the velocity and passes the "
        f"probe band: the recommended window is at that floor ({policy['sg_window']} samples, "
        f"cutoff {policy['sg_cutoff_hz']:.0f} Hz) and the probe's modal content will reappear in "
        f"the consumer's residual rather than in its ddq. Cap the top harmonic at "
        f"{int(resolvable_top / max(config.excitation.base_frequency, 1e-12))} or raise the "
        "sampling rate (R5_03 T-4; contract carries differentiation.probe_resolvable=false). "
        "This is a different bound from the schema_version: 2 load-time check, which is the "
        f"sampling one, 0.25 x rate = {0.25 * rate:g} Hz.",
        stacklevel=2,
    )


def require_probe_within_sampling_bound(config: DatasetConfig, *, source: Any = "config") -> None:
    """Refuse a round-5 config whose probe outruns its own sampling rate.

    `R5_03` T-4 asked for ``probe_top_hz <= 0.25 * sample_rate`` as a hard
    requirement and `R5_05` D-8 made it a build-time warning instead, because
    both shipped round-4 configs violate it and `R4_06 B5` asserts they load
    warning-free.  `R5_06` T-8 settles the two halves: the bound is an error
    for any config that declares ``schema_version: 2`` -- one written in round
    5 or later, which has no round-4 invariant to keep -- and stays a
    build-time warning for ``schema_version: 1``.

    **This bound is not the same as** :func:`differentiation_policy`'s
    ``probe_resolvable`` **flag, and pass 3 found that they part company**
    (`R5_08` Sec 2).  ``0.25 * rate`` is a sampling criterion: below it the
    probe's top line is recorded with four samples per period.  Resolvability
    is a *differentiation* criterion: a Savitzky-Golay derivative of order 3
    has a five-sample floor, so its widest passband is ``0.45 * rate / 5 =
    0.09 * rate``, and a probe between 0.09 and 0.25 of the rate is recorded
    faithfully and then filtered out again by the consumer.  Both facts matter
    and they are reported separately: this one refuses the config, the other
    is carried in the contract for the consumer to read.
    """
    if int(config.schema_version) < 2:
        return
    policy = differentiation_policy(config)
    rate = policy["sample_rate_hz"]
    top = policy["probe_top_hz"]
    if top <= 0.25 * rate:
        return
    cap = int(0.25 * rate / max(config.excitation.base_frequency, 1e-12))
    raise ValueError(
        f"{source}: probe top frequency {top:g} Hz is above 0.25 x the {rate:g} Hz sampling rate, so "
        f"the probe's top line is recorded with fewer than four samples per period. Cap the top "
        f"harmonic index at {cap} or raise the sampling rate. (R5_06 T-8; this is an error rather "
        "than a warning because the config declares schema_version: 2.)"
    )


def differentiation_policy(config: DatasetConfig) -> dict[str, Any]:
    """The `differentiation` block written into the consumer contract (T-4)."""
    sample_rate = 1.0 / float(config.sample_time_step)
    probe = config.excitation.probe_harmonics
    probe_top = float(probe[-1] * config.excitation.base_frequency) if probe else 0.0
    window = recommended_sg_window(sample_rate, probe_top)
    cutoff = 0.45 * sample_rate / window
    return {
        "sample_rate_hz": sample_rate,
        "probe_top_hz": probe_top,
        "sg_window": window,
        "sg_poly": SG_POLY_ORDER,
        "sg_cutoff_hz": cutoff,
        # False when the probe band cannot survive the differentiation at any
        # window: the filter's floor is poly + 2 samples, so above
        # probe_top = 0.25 * rate there is no window that both smooths and
        # passes the band (R5_03 T-4).  A consumer seeing False should expect
        # the probe's content in the residual, not in ddq.
        "probe_resolvable": bool(probe_top == 0.0 or cutoff >= probe_top),
        "rationale": (
            "the consumer recomputes ddq with a Savitzky-Golay derivative; this window's -3 dB "
            "point sits near 1.5x probe_top_hz, so the band the modal probe put in the data "
            "survives the differentiation instead of becoming an unexplainable residual (R5_03 T-4)"
        ),
    }


def _target_contains(manifest: Mapping[str, Any]) -> str:
    """One sentence on what the target channel physically includes.

    "whole arm" versus "the tool beyond the sensor only" is the distinction that
    decides whether a model is being asked to explain the robot's dynamics or a
    5 kg handle's, and it is invisible in the column names (R5_03 T-1).
    """
    target = str((manifest.get("signals") or {}).get("target", "link_torque"))
    if target == "ee_wrench_joint":
        sensor = manifest.get("force_torque_sensor") or {}
        measures = ", ".join(sensor.get("measures", [])) or "the bodies past the sensor"
        return (
            f"only the bodies mounted past the {sensor.get('frame', 'sensor')} frame "
            f"({measures}, {sensor.get('mass_kg', float('nan')):.3g} kg), plus any external "
            "contact force; NOT the arm's own dynamics"
        )
    if target == "motor_torque":
        return "the whole arm seen from the motor side, motor inertia and friction included"
    return "the whole arm seen from the link side, transmission excluded"


_METADATA_COLUMN_PREFIXES = (
    "viscous__", "coulomb__", "stiffness__", "damping__", "damping_ratio__", "rotor_inertia__",
    "payload_", "exc_",
    # Round 5, same reasoning: one value per bag repeated on every row.
    "gain_motor__", "gain_link__", "link_viscous__", "link_coulomb__", "ripple_",
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
        # Keep "t" at float64: it is what the uniform-time-step assertion
        # above and the consumer's Savitzky-Golay filter depend on, and
        # float32's ~1e-6 s spacing at 10 s degrades further on longer
        # trajectories (R3_10 Sec 3.5) -- everything else is a signal or
        # metadata value the consumer casts to float32 on load anyway.
        float_cols = frame.select_dtypes(include="float64").columns.drop("t", errors="ignore")
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
        # Machine-readable target kind, keyed on directly by the consumer at
        # dof == 6 (where "ft0..ft5" is ambiguous between this and a legacy
        # wrench) instead of pattern-matching target_columns' string form
        # (R3_14 Sec 2).
        "target_kind": "per_joint_torque",
        "target_semantics": _TARGET_SEMANTICS[str((manifest.get("signals") or {}).get("target", "link_torque"))],
        # Which instrument the target comes from, and what it physically
        # contains.  The three round-5 platforms have three different answers
        # and a consumer cannot infer it from the column names (R5_03 T-1).
        "target_instrument": str((manifest.get("signals") or {}).get("target_kind", "joint_torque_sensor")),
        "target_contains": _target_contains(manifest),
        "force_torque_sensor": manifest.get("force_torque_sensor"),
        # Diagnostic columns beside the target, when there is a cell (T-15).
        # `sigma_min_j` is sigma_min(J(q)) at the recorded configuration: near
        # a singularity whole wrench directions map to near-zero joint torque,
        # so the target stops carrying part of what the cell measured.  A
        # consumer weighting or filtering samples by posture reads this; it is
        # never a model input.
        "posture_columns": (
            None if not (manifest.get("force_torque_sensor")) else {
                "sigma_min_j": (
                    "smallest singular value of the 6xn frame Jacobian at the recorded "
                    "configuration; information the J^T mapping can still carry, not a "
                    "numerical-failure flag (R5_07 T-15)"
                ),
                "ft_equals": "ft = J(q)^T w on the recorded q* and w_* columns, exactly (R5_06 Sec 3)",
            }
        ),
        # How to differentiate this dataset's velocity (R5_03 T-4).  The
        # consumer reads it instead of hard-coding a window.
        "differentiation": manifest.get("differentiation"),
        # Which side of the spring each column is measured on.  A consumer
        # comparing an elastic model class against a residual one needs this:
        # on a collocated pair the link equation has no elastic term at all,
        # so the comparison measures input availability rather than elasticity
        # (R5_01 Sec 3, R5_00 Amendment 1).
        "signals": manifest.get("signals"),
        "input_semantics": manifest.get("input"),
        "measurement": manifest.get("measurement"),
        "controller": manifest.get("controller"),
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
