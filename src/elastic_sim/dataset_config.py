"""Identification dataset configuration: dataclasses, robot draws and YAML parsing.

Split out of ``dataset.py`` (`R5_10` T-7) with no behaviour change; import
from :mod:`elastic_sim.dataset`, which re-exports every name.  Elastic robots
are sampled rather than laddered: every joint draws its own transmission
stiffness log-uniformly from a physically plausible interval and its own
damping ratio uniformly, and the damping coefficient is derived from both and
the joint's inertia, so the config never states a damping value.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .backend_comparison import ComparisonThresholds
from .controllers import ControllerSpec, NominalModelSpec, VelocityLoopSpec
from .excitation import FourierExcitationConfig
from .measurement import MeasurementModel
from .torque_runners import TransmissionSpec


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
