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
from .controllers import ControllerSpec, IntegralSpec, NominalModelSpec, VelocityLoopSpec
from .excitation import FourierExcitationConfig
from .identification import FrictionModel
from .measurement import DQ_SOURCES, ChannelNoise, MeasurementModel, NoiseModel
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
class ProbeDesign:
    """A probe comb aimed at the link-side modes (`R6_02` P2-1).

    Log-spaced lines from ``min(lowest_hz, span[0] x f_link_low)`` up to
    ``span[1] x f_link_high`` of every joint whose band reaches into the
    resolvable band, capped at that band (:mod:`elastic_sim.link_modes`).
    """

    source: str = "link_modes"
    lowest_hz: float = 2.0
    lines: int = 30
    span: tuple[float, float] = (0.5, 1.5)

    def __post_init__(self) -> None:
        if self.source != "link_modes":
            raise ValueError("excitation.probe_design.source must be 'link_modes'")
        if self.lowest_hz <= 0.0 or self.lines < 2 or not 0.0 < self.span[0] < 1.0 < self.span[1]:
            raise ValueError("excitation.probe_design needs lowest_hz > 0, lines >= 2 and span [lo < 1 < hi]")

    def as_dict(self) -> dict[str, Any]:
        return {"source": self.source, "lowest_hz": self.lowest_hz, "lines": self.lines, "span": list(self.span)}


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
    #: Round 6 (`R6_00` Sec 2.2): the natural frequency as *fractions* of the
    #: discrete-time limit ``omega_max`` instead of absolute values, one or
    #: more ``(low, high)`` bands drawn log-uniformly over their union.  The
    #: production draw is one band, ``[0.5, 0.9]``; the gain-shift test file
    #: is the two outside it, ``[0.3, 0.5) U (0.9, 1.0]`` (Sec 7 item 4).
    #: Resolved at load time into :attr:`bands` (rad/s), which is what draws.
    natural_frequency_fraction: tuple[tuple[float, float], ...] = ()
    bands: tuple[tuple[float, float], ...] = ()
    omega_max: float | None = None

    def __post_init__(self) -> None:
        for name in ("natural_frequency", "damping_ratio"):
            low, high = getattr(self, name)
            if low <= 0.0 or high < low:
                raise ValueError(f"control_gains.{name} must satisfy 0 < min <= max")
        for name in ("natural_frequency_fraction", "bands"):
            rows = tuple((float(lo), float(hi)) for lo, hi in getattr(self, name))
            for low, high in rows:
                if low <= 0.0 or high < low:
                    raise ValueError(f"control_gains.{name} bands must satisfy 0 < min <= max")
            object.__setattr__(self, name, rows)

    def resolved(self, omega_max: float) -> "ControlGainSampling":
        """These gains with :attr:`natural_frequency_fraction` turned into rad/s."""
        from dataclasses import replace

        if not self.natural_frequency_fraction:
            return self
        bands = tuple((lo * omega_max, hi * omega_max) for lo, hi in self.natural_frequency_fraction)
        return replace(self, bands=bands, omega_max=float(omega_max),
                       natural_frequency=(min(b[0] for b in bands), max(b[1] for b in bands)))

    def with_fraction_bands(self, bands: Sequence[tuple[float, float]]) -> "ControlGainSampling":
        """The same sampling over other fractions of the same ``omega_max``."""
        from dataclasses import replace

        if self.omega_max is None:
            raise ValueError("with_fraction_bands needs gains resolved against omega_max first")
        return replace(self, natural_frequency_fraction=tuple(bands)).resolved(self.omega_max)


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


_TARGET_SOURCES = ("measured", "clean")


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
    #: Round 6 (`R6_00` Sec 1): ``measured`` routes the target through the
    #: platform's torque-measurement model (``simulation.noise.ft``), ``clean``
    #: writes the simulator's own ``tau_s``.  Only read when the config has a
    #: ``simulation.noise`` block; the noise ablation is the ``clean`` one.
    target_source: str = "measured"
    #: Keep the end-effector cell's ``w_*``/``sigma_min_j`` columns beside a
    #: joint-torque target (FMRR, for the post-training chain, Sec 5.5).
    wrench_columns: bool = False

    def __post_init__(self) -> None:
        if self.position_side not in _POSITION_SIDES:
            raise ValueError(f"dataset.signals.position_side must be one of {_POSITION_SIDES}")
        if self.target not in _TARGET_SIDES:
            raise ValueError(f"dataset.signals.target must be one of {_TARGET_SIDES}")
        if self.target_source not in _TARGET_SOURCES:
            raise ValueError(f"dataset.signals.target_source must be one of {_TARGET_SOURCES}")

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

    @property
    def needs_cell(self) -> bool:
        """True when the file carries the cell's wrench, as target or beside it."""
        return self.needs_sensor or self.wrench_columns

    def describe(self) -> dict[str, Any]:
        return {
            "position_side": self.position_side,
            "target": self.target,
            "target_kind": _TARGET_KINDS[self.target],
            "collocated": self.is_collocated,
            "clean_columns": bool(self.clean_columns),
            "target_source": self.target_source,
            "wrench_columns": bool(self.wrench_columns),
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
    #: One shared ``[K1/K2, 1, K3/K2]`` row, or one per joint (`R6_00` Sec 5.2).
    stiffness_factors: tuple = ()
    #: The knees in torque instead of deflection -- one shared ``[T1, T2]``
    #: row or one per joint -- converted per robot with its sampled stiffness
    #: (`StiffnessNonlinearity.from_torque_knees`).  Or as fractions of each
    #: joint's URDF effort limit (the iiwa: 20 % and 75 %).  At most one of the
    #: three breakpoint forms.
    stiffness_breakpoints_torque: tuple = ()
    stiffness_breakpoints_effort_fraction: tuple[float, ...] = ()
    ripple_amplitude: float = 0.0
    ripple_order: float | tuple[float, ...] = 24.0
    ripple_random_phase: bool = True
    #: The harmonic drive's kinematic error inside the spring (Sec 5.3).
    transmission_error_amplitude: tuple[float, ...] = ()
    transmission_error_order: tuple[float, ...] = (1.0,)
    transmission_error_random_phase: bool = True
    #: ``explicit`` (rounds 4-5) or ``implicit`` (`R6_00` Sec 3; see
    #: ``torque_runners._ImplicitLinkFriction``).  Same law, same magnitudes.
    link_friction_integration: str = "explicit"

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
        forms = [bool(self.stiffness_breakpoints), bool(self.stiffness_breakpoints_torque),
                 bool(self.stiffness_breakpoints_effort_fraction)]
        if sum(forms) > 1:
            raise ValueError(
                "plant_extras.stiffness_nonlinearity: give one of breakpoints, breakpoints_torque or "
                "breakpoints_effort_fraction"
            )
        if any(forms) != bool(self.stiffness_factors):
            raise ValueError(
                "plant_extras.stiffness_nonlinearity needs both breakpoints and factors "
                "(factors one longer than breakpoints)"
            )
        if self.link_friction_integration not in ("explicit", "implicit"):
            raise ValueError("plant_extras.link_friction.integration must be 'explicit' or 'implicit'")
        if any(float(v) < 0.0 for v in self.transmission_error_amplitude):
            raise ValueError("plant_extras.transmission_error.amplitude must be non-negative")

    @property
    def has_link_friction(self) -> bool:
        """True when this config's tier actually puts friction on the plant."""
        return self.link_friction_tier != "off" and bool(
            self.link_friction_viscous or self.link_friction_coulomb
        )

    @property
    def has_nonlinear_spring(self) -> bool:
        return bool(self.stiffness_breakpoints or self.stiffness_breakpoints_torque
                    or self.stiffness_breakpoints_effort_fraction)

    @property
    def has_transmission_error(self) -> bool:
        return any(float(v) > 0.0 for v in self.transmission_error_amplitude)

    @property
    def max_stiffness_factor(self) -> float:
        """The stiffest spring region's factor (1 for a linear spring)."""
        if not self.has_nonlinear_spring:
            return 1.0
        return float(max(1.0, np.max(np.asarray(self.stiffness_factors, dtype=float))))

    @property
    def is_empty(self) -> bool:
        return not (
            self.has_link_friction or self.has_nonlinear_spring or self.ripple_amplitude
            or self.has_transmission_error
        )


@dataclass(frozen=True)
class MotorFrictionPrior:
    """Motor-side friction re-derived per joint from the URDF (`R6_00` Sec 5.4).

    ``b_j = viscous_fraction * effort_j / velocity_j`` and ``c_j =
    coulomb_fraction * effort_j``, at the Coulomb smoothing ``epsilon``,
    replacing whatever ``<dynamics damping/friction>`` the URDF declares (the
    iiwa asset's blanket 10 Nm s/rad and 0.1 Nm was an asset artefact, not a
    prior; `R5_12` Sec 2).  It acts *before* the spring, so it lands in
    ``tau_cmd - tau_s`` and not in the target.  ``per_robot`` draws each
    robot's coefficients from ``dataset.friction_scale`` around these,
    independently per joint and coefficient, on its own stream; the rigid
    reference keeps the nominal.  Class E.
    """

    viscous_fraction: float = 0.05
    coulomb_fraction: float = 0.02
    epsilon: float = 1.0e-2
    per_robot: bool = True
    provenance: str = "E"

    def __post_init__(self) -> None:
        for name in ("viscous_fraction", "coulomb_fraction"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"simulation.motor_friction.{name} must be non-negative")
        if float(self.epsilon) <= 0.0:
            raise ValueError("simulation.motor_friction.epsilon must be positive")
        if self.provenance not in _PROVENANCE_CLASSES:
            raise ValueError(f"simulation.motor_friction.provenance must be one of {sorted(_PROVENANCE_CLASSES)}")

    def nominal(self, asset: Any) -> FrictionModel:
        joints = asset.resolve_active_joints()
        effort = np.asarray([np.nan if j.effort is None else float(j.effort) for j in joints])
        velocity = np.asarray([np.nan if j.velocity is None else float(j.velocity) for j in joints])
        if not (np.all(np.isfinite(effort)) and np.all(np.isfinite(velocity)) and np.all(velocity > 0.0)):
            raise ValueError(
                f"simulation.motor_friction needs a finite effort and velocity limit on every joint of "
                f"{asset.name!r}"
            )
        return FrictionModel(self.viscous_fraction * effort / velocity, self.coulomb_fraction * effort,
                             float(self.epsilon))

    def describe(self) -> dict[str, Any]:
        return {"viscous_fraction": float(self.viscous_fraction), "coulomb_fraction": float(self.coulomb_fraction),
                "epsilon": float(self.epsilon), "per_robot": bool(self.per_robot),
                "provenance": self.provenance,
                "rule": "b_j = viscous_fraction * effort_j / velocity_j, c_j = coulomb_fraction * effort_j (URDF limits)"}


@dataclass(frozen=True)
class GateSpec:
    """The controller-influence gates and hard checks a round-6 build must pass (`R6_00` Secs 7, 9)."""

    #: Sec 2.2: achieved-path RMS deviation over each joint's excitation span.
    sag_max_fraction: float = 0.05
    #: ``gate`` refuses a file over the limit; ``report`` records the sag and
    #: passes (`R6_02` Sec 7: FMRR on the bus, where the lag is real physics).
    sag_mode: str = "gate"
    sag_reason: str = ""
    #: Sec 7 item 3: round 5's ``pd`` level of the Q-C v2 statistic on the same
    #: platform, per tier kind; ``None`` where round 5 has none (the iiwa).
    qc_v2_reference_elastic: float | None = None
    qc_v2_reference_rigid: float | None = None
    qc_v2_reference_source: str = ""
    #: Sec 9: unstable bags allowed before a file is refused.
    unstable_max_fraction: float = 0.02
    #: Sec 7 item 4: the gain-shift file's bands, as fractions of omega_max.
    gainshift_fraction: tuple[tuple[float, float], ...] = ((0.3, 0.5), (0.9, 1.0))
    gainshift_max_ratio: float = 1.3

    def __post_init__(self) -> None:
        if self.sag_mode not in ("gate", "report"):
            raise ValueError("gates.sag_mode must be 'gate' or 'report'")
        if self.sag_mode == "report" and not self.sag_reason:
            raise ValueError("gates.sag_mode: report needs a sag_reason")

    def describe(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


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
    # Round 6 (`R6_00`, schema_version 3).  ``None`` keeps rounds 4-5 exactly.
    motor_friction: MotorFrictionPrior | None = None
    noise: NoiseModel | None = None
    # ``max_time_step: auto`` / ``rigid_time_step: auto`` (`R6_02` Sec 6): the
    # step is derived at load from the Sec 3 bound table rather than hand-set.
    auto_time_steps: tuple[str, ...] = ()
    # ``excitation.probe_design`` (`R6_02` P2-1): the probe comb is derived at
    # load from the link-side mode table instead of listed by hand.
    probe_design: ProbeDesign | None = None
    gates: GateSpec = field(default_factory=GateSpec)
    #: Build only these tiers' bags while every draw (payloads, splits, seeds)
    #: is still made over the whole tier list, so a bag here is the bag of the
    #: full build.  The gain-shift file is the test robots this way (Sec 7).
    only_tiers: tuple[str, ...] | None = None

    @property
    def is_round6(self) -> bool:
        """``schema_version: 3``: the round-6 loop, plant and instruments."""
        return int(self.schema_version) >= 3

    @property
    def loop_delay(self) -> int:
        """Samples between a measurement and the command that uses it (Sec 2.1)."""
        if self.noise is not None:
            return int(self.noise.delay_samples)
        return int(self.measurement.delay_samples)

    @property
    def control_period(self) -> float:
        """The PD's period: the sample period on the bus, the drive's own inside a drive (`R6_02` Sec 7)."""
        if self.controller.location == "drive":
            return 1.0 / float(self.controller.drive_rate)
        return float(self.sample_time_step)

    @property
    def control_delay(self) -> int:
        """Periods between a reading and the command that uses it: the bus delay, or 0 inside a drive."""
        return 0 if self.controller.location == "drive" else self.loop_delay

    @property
    def drive_ticks(self) -> int:
        """Drive periods per sample period (1 on the bus)."""
        return int(round(float(self.sample_time_step) / self.control_period))

    def omega_max(self) -> float:
        """``0.5 / (2 zeta_max T_c (1 + d))``, the Sec 2.2 limit, at the loop's own period and delay.

        On a drive loop with ``drive.bound: rotor`` this is the reference each
        axis's ``omega_max,j = omega_max J_j / (J_j + m_j)`` scales down from.
        """
        zeta_max = float(self.control_gains.damping_ratio[1])
        return 0.5 / (2.0 * zeta_max * self.control_period * (1.0 + self.control_delay))

    def nominal_rotor_inertia(self, n_dof: int) -> np.ndarray:
        """The nominal reflected rotor inertia per joint: what the controller and the rigid tier use."""
        values = self.transmission.rotor_inertia_nominal or self.transmission.rotor_inertia
        return _per_joint(values, n_dof, "rotor_inertia_nominal")


_TRANSMISSION_KEYS = {
    "robots", "stiffness", "damping_ratio", "rotor_inertia", "inertia_samples", "sampling",
    "stiffness_nominal", "stiffness_factor", "stiffness_common_fraction", "stiffness_provenance",
    "rotor_inertia_nominal", "rotor_inertia_factor", "rotor_inertia_provenance",
}


_EXCITATION_KEYS = {
    "n_harmonics", "base_frequency", "n_periods", "limit_margin", "max_acceleration", "velocity_fraction",
    "centre_jitter", "probe_harmonics", "probe_acceleration_fraction", "candidates", "regime", "position_window",
    "probe_design", "probe_budget",
}


_REGIME_KEYS = {"enabled", "max_acceleration", "velocity_fraction"}


_PAYLOAD_KEYS = {"enabled", "mass", "offset_x", "offset_y", "offset_z", "size", "per"}


_DATASET_KEYS = {
    "trajectories", "trajectories_per_robot", "friction_samples", "friction_scale", "seed", "output",
    "metadata_columns", "split", "signals",
}


_SIGNALS_KEYS = {"position_side", "target", "clean_columns", "target_source", "wrench_columns"}


_SPLIT_KEYS = {"mode", "test_robots", "val_robots", "test_trajectories", "val_trajectories"}


_SIMULATION_KEYS = {
    "control_frequency", "control_damping_ratio", "control_gains", "control_decimation",
    "allow_control_decimation", "rigid_time_step", "max_time_step", "sample_time_step", "control_separation",
    "controller", "measurement", "plant_extras", "motor_friction", "noise",
}


_CONTROLLER_KEYS = {"mode", "randomize", "nominal", "velocity_loop", "gain_sizing", "loop", "integral",
                    "location", "drive"}


_NOMINAL_KEYS = {"knows_payload", "friction_scale", "rotor_inertia_scale", "inertia_scale"}


_VELOCITY_LOOP_KEYS = {"position_gain", "velocity_bandwidth", "integral_time"}


_MEASUREMENT_KEYS = {
    "encoder_resolution", "q_noise", "dq_noise", "tau_noise_rel", "tau_noise_abs",
    "tau_gain_error", "delay_samples", "ft_noise_rel", "ft_noise_abs",
}


_PLANT_EXTRAS_KEYS = {"link_friction", "stiffness_nonlinearity", "torque_ripple", "transmission_error"}


_LINK_FRICTION_KEYS = {"tier", "viscous", "coulomb", "provenance", "integration"}


_STIFFNESS_NONLINEARITY_KEYS = {
    "breakpoints", "factors", "breakpoints_torque", "breakpoints_effort_fraction", "provenance",
}


_TORQUE_RIPPLE_KEYS = {"amplitude", "order", "random_phase", "provenance"}


_TRANSMISSION_ERROR_KEYS = {"amplitude", "order", "random_phase", "provenance"}


_MOTOR_FRICTION_KEYS = {"viscous_fraction", "coulomb_fraction", "epsilon", "per_robot", "provenance"}


_NOISE_KEYS = {"enabled", "delay_samples", "q", "dq", "ft", "wrench", "tag", "provenance"}


_NOISE_CHANNEL_KEYS = {"p", "sigma", "gain", "offset", "quantization", "sigma_quanta", "source"}


_GATES_KEYS = {
    "sag_max_fraction", "qc_v2_reference", "unstable_max_fraction", "gainshift_fraction", "gainshift_max_ratio",
    "sag_mode", "sag_reason",
}


_CONTROL_GAINS_KEYS = {"enabled", "natural_frequency", "damping_ratio", "natural_frequency_fraction"}


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
        gain_sizing=str(controller_cfg.get("gain_sizing", "link_plus_rotor")),
        loop=str(controller_cfg.get("loop", "physics_step")),
        integral=IntegralSpec(
            enabled=bool((controller_cfg.get("integral") or {}).get("enabled", False)),
            time_factor=float((controller_cfg.get("integral") or {}).get("time_factor", 10.0)),
            reason=str((controller_cfg.get("integral") or {}).get("reason", "")),
        ),
        location=str(controller_cfg.get("location", "bus")),
        drive_rate=float((controller_cfg.get("drive") or {}).get("rate", 0.0)),
        drive_bound=str((controller_cfg.get("drive") or {}).get("bound", "rotor")),
        setpoint_delay=int((controller_cfg.get("drive") or {}).get("setpoint_delay", 1)),
        velocity_cutoff_fraction=float((controller_cfg.get("drive") or {}).get("velocity_cutoff_fraction", 0.1)),
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


def _float_rows(raw: Any) -> tuple:
    """A flat float tuple, or a tuple of float tuples for a per-joint table."""
    values = list(raw or [])
    if values and isinstance(values[0], (list, tuple)):
        return tuple(tuple(float(v) for v in row) for row in values)
    return tuple(float(v) for v in values)


def _plant_extras_sampling(extras_cfg: Mapping[str, Any], *, round6: bool = False) -> PlantExtrasSampling:
    friction_cfg = extras_cfg.get("link_friction", {}) or {}
    spring_cfg = extras_cfg.get("stiffness_nonlinearity", {}) or {}
    ripple_cfg = extras_cfg.get("torque_ripple", {}) or {}
    error_cfg = extras_cfg.get("transmission_error", {}) or {}
    viscous_nominal, viscous_factor = _friction_prior(friction_cfg.get("viscous"))
    coulomb_nominal, coulomb_factor = _friction_prior(friction_cfg.get("coulomb"))
    order = ripple_cfg.get("order", 24.0)
    return PlantExtrasSampling(
        link_friction_tier=str(friction_cfg.get("tier", "fixed")),
        link_friction_viscous=viscous_nominal,
        link_friction_coulomb=coulomb_nominal,
        link_friction_viscous_factor=viscous_factor,
        link_friction_coulomb_factor=coulomb_factor,
        stiffness_breakpoints=tuple(float(v) for v in spring_cfg.get("breakpoints", []) or []),
        stiffness_factors=_float_rows(spring_cfg.get("factors")),
        stiffness_breakpoints_torque=_float_rows(spring_cfg.get("breakpoints_torque")),
        stiffness_breakpoints_effort_fraction=tuple(
            float(v) for v in spring_cfg.get("breakpoints_effort_fraction", []) or []
        ),
        ripple_amplitude=float(ripple_cfg.get("amplitude", 0.0)),
        ripple_order=tuple(float(v) for v in order) if isinstance(order, (list, tuple)) else float(order),
        ripple_random_phase=bool(ripple_cfg.get("random_phase", True)),
        transmission_error_amplitude=tuple(float(v) for v in np.atleast_1d(error_cfg.get("amplitude", []) or [])),
        transmission_error_order=tuple(float(v) for v in np.atleast_1d(error_cfg.get("order", 1.0))),
        transmission_error_random_phase=bool(error_cfg.get("random_phase", True)),
        link_friction_integration=str(friction_cfg.get("integration", "implicit" if round6 else "explicit")),
    )


def _noise_channel(raw: Any, name: str, source: Path) -> ChannelNoise:
    raw = raw or {}
    _check_keys(raw, _NOISE_CHANNEL_KEYS, f"simulation.noise.{name}", source)
    return ChannelNoise(
        p=float(raw.get("p", 0.0)), sigma=float(raw.get("sigma", 0.0)), gain=float(raw.get("gain", 0.0)),
        offset=float(raw.get("offset", 0.0)),
        quantization=tuple(float(v) for v in np.atleast_1d(raw.get("quantization", 0.0))),
        sigma_quanta=float(raw.get("sigma_quanta", 0.0)),
    )


def _noise_model(noise_cfg: Mapping[str, Any] | None, source: Path) -> NoiseModel | None:
    if noise_cfg is None:
        return None
    _check_keys(noise_cfg, _NOISE_KEYS, "simulation.noise", source)
    dq_cfg = dict(noise_cfg.get("dq", {}) or {})
    dq_source = str(dq_cfg.pop("source", "sensor"))
    if dq_source not in DQ_SOURCES:
        raise ValueError(f"{source}: simulation.noise.dq.source must be one of {DQ_SOURCES}")
    return NoiseModel(
        enabled=bool(noise_cfg.get("enabled", True)),
        q=_noise_channel(noise_cfg.get("q"), "q", source),
        dq=_noise_channel(dq_cfg, "dq", source),
        ft=_noise_channel(noise_cfg.get("ft"), "ft", source),
        wrench=_noise_channel(noise_cfg.get("wrench"), "wrench", source),
        dq_source=dq_source,
        delay_samples=int(noise_cfg.get("delay_samples", 1)),
        tag=str(noise_cfg.get("tag", "modelling_error")),
        provenance=str(noise_cfg.get("provenance", "E")),
    )


def _gate_spec(raw: Mapping[str, Any] | None, source: Path) -> GateSpec:
    raw = raw or {}
    _check_keys(raw, _GATES_KEYS, "gates", source)
    reference = raw.get("qc_v2_reference") or {}
    _check_keys(reference, {"elastic", "rigid", "source"}, "gates.qc_v2_reference", source)
    shift = raw.get("gainshift_fraction", [[0.3, 0.5], [0.9, 1.0]])
    return GateSpec(
        sag_max_fraction=float(raw.get("sag_max_fraction", 0.05)),
        sag_mode=str(raw.get("sag_mode", "gate")),
        sag_reason=str(raw.get("sag_reason", "")),
        qc_v2_reference_elastic=None if reference.get("elastic") is None else float(reference["elastic"]),
        qc_v2_reference_rigid=None if reference.get("rigid") is None else float(reference["rigid"]),
        qc_v2_reference_source=str(reference.get("source", "")),
        unstable_max_fraction=float(raw.get("unstable_max_fraction", 0.02)),
        gainshift_fraction=tuple((float(lo), float(hi)) for lo, hi in shift),
        gainshift_max_ratio=float(raw.get("gainshift_max_ratio", 1.3)),
    )


def load_config(path: str | Path, *, check_bounds: bool = True) -> DatasetConfig:
    """Read a YAML identification config into a :class:`DatasetConfig`.

    A ``schema_version: 3`` config is also held to round 6's load-time
    requirements (:func:`require_round6`); ``check_bounds`` includes the
    asset-dependent explicit-term bound, which needs the asset's inertia
    envelope and costs about a second.
    """
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
             "excitation", "dataset", "simulation", "visualization", "payload", "gates"}
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
    _check_keys(controller_cfg.get("drive", {}) or {}, {"rate", "bound", "setpoint_delay", "velocity_cutoff_fraction"},
                "simulation.controller.drive", source)
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
    _check_keys(extras_cfg.get("transmission_error", {}) or {}, _TRANSMISSION_ERROR_KEYS,
                "simulation.plant_extras.transmission_error", source)
    motor_friction_cfg = sim_cfg.get("motor_friction")
    if motor_friction_cfg is not None:
        _check_keys(motor_friction_cfg, _MOTOR_FRICTION_KEYS, "simulation.motor_friction", source)
    round6 = int(raw.get("schema_version", 1)) >= 3
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
    if round6 and "control_decimation" in sim_cfg:
        raise ValueError(
            f"{source}: schema_version 3 derives the control decimation from sample_time_step "
            "(one command per sample period, R6_00 Sec 2.1); remove simulation.control_decimation"
        )
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
            probe_budget=str(exc_cfg.get("probe_budget", "split")),
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
            natural_frequency_fraction=_fraction_bands(
                (sim_cfg.get("control_gains", {}) or {}).get("natural_frequency_fraction")
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
        plant_extras=_plant_extras_sampling(extras_cfg, round6=round6),
        signals=SignalPolicy(
            position_side=str(signals_cfg.get("position_side", "link")),
            target=str(signals_cfg.get("target", "link_torque")),
            clean_columns=bool(signals_cfg.get("clean_columns", False)),
            target_source=str(signals_cfg.get("target_source", "measured")),
            wrench_columns=bool(signals_cfg.get("wrench_columns", False)),
        ),
        candidates=int(exc_cfg.get("candidates", 48)),
        control_frequency=float(sim_cfg.get("control_frequency", 25.0)),
        control_damping_ratio=float(sim_cfg.get("control_damping_ratio", 1.0)),
        rigid_time_step=_time_step(sim_cfg.get("rigid_time_step"), "rigid_time_step", source),
        sample_time_step=float(sim_cfg.get("sample_time_step", 0.002)),
        max_time_step=_time_step(sim_cfg.get("max_time_step"), "max_time_step", source),
        control_decimation=int(sim_cfg.get("control_decimation", 1)),
        output=str(data_cfg.get("output", "data/identification/dataset.csv")),
        metadata_columns=str(data_cfg.get("metadata_columns", "inline")),
        visualize=bool(view_cfg.get("enabled", False)),
        realtime_scale=float(view_cfg.get("realtime_scale", 1.0)),
        motor_friction=None if motor_friction_cfg is None else MotorFrictionPrior(
            viscous_fraction=float(motor_friction_cfg.get("viscous_fraction", 0.05)),
            coulomb_fraction=float(motor_friction_cfg.get("coulomb_fraction", 0.02)),
            epsilon=float(motor_friction_cfg.get("epsilon", 1.0e-2)),
            per_robot=bool(motor_friction_cfg.get("per_robot", True)),
            provenance=str(motor_friction_cfg.get("provenance", "E")),
        ),
        noise=_noise_model(sim_cfg.get("noise"), source),
        gates=_gate_spec(raw.get("gates"), source),
        probe_design=_probe_design(exc_cfg.get("probe_design"), exc_cfg, source),
        auto_time_steps=tuple(key for key in ("max_time_step", "rigid_time_step")
                              if str(sim_cfg.get(key, "")).strip().lower() == "auto"),
    )
    if config.control_gains.natural_frequency_fraction:
        from dataclasses import replace

        config = replace(config, control_gains=config.control_gains.resolved(config.omega_max()))
    if config.probe_design is not None:
        from .link_modes import resolve_probe_design

        config = resolve_probe_design(config)
    require_probe_within_sampling_bound(config, source=source)
    require_round6(config, source=source)
    if config.auto_time_steps:
        from .dataset_bounds import resolve_auto_time_steps

        config = resolve_auto_time_steps(config, source=source)
    if check_bounds and config.is_round6:
        from .dataset_bounds import require_explicit_term_bound

        require_explicit_term_bound(config, source=source)
    return config


#: Ceiling for a step derived from the bound (``auto``): the schema default,
#: so a bound that allows more still steps no coarser than rounds 4-5 did.
AUTO_TIME_STEP_CEILING = 5.0e-4


def _probe_design(raw: Any, exc_cfg: Mapping[str, Any], source: Any) -> ProbeDesign | None:
    if raw is None:
        return None
    _check_keys(raw, {"source", "lowest_hz", "lines", "span"}, "excitation.probe_design", source)
    if exc_cfg.get("probe_harmonics"):
        raise ValueError(f"{source}: excitation.probe_harmonics and probe_design are exclusive")
    span = raw.get("span", [0.5, 1.5])
    return ProbeDesign(source=str(raw.get("source", "link_modes")), lowest_hz=float(raw.get("lowest_hz", 2.0)),
                       lines=int(raw.get("lines", 30)), span=(float(span[0]), float(span[1])))


def _time_step(raw: Any, key: str, source: Any) -> float:
    """A physics-step entry: a number, or ``auto`` (placeholder, resolved from the bound after load)."""
    if raw is None:
        return 5.0e-4
    if isinstance(raw, str):
        if raw.strip().lower() != "auto":
            raise ValueError(f"{source}: simulation.{key} must be a number or 'auto', got {raw!r}")
        return AUTO_TIME_STEP_CEILING
    return float(raw)


def _fraction_bands(raw: Any) -> tuple[tuple[float, float], ...]:
    """``[lo, hi]`` or ``[[lo, hi], ...]`` fractions of omega_max."""
    if not raw:
        return ()
    if isinstance(raw[0], (list, tuple)):
        return tuple((float(lo), float(hi)) for lo, hi in raw)
    return ((float(raw[0]), float(raw[1])),)


def physics_step(required: float, period: float) -> float:
    """The largest step no coarser than ``required`` that divides ``period`` exactly.

    Round 6 holds the command for one sample period (`R6_00` Sec 2.1), so a
    control instant has to land on a physics step: ``period / N`` with the
    smallest integer ``N`` that is fine enough.
    """
    if required <= 0.0 or period <= 0.0:
        raise ValueError("physics_step needs positive steps")
    count = int(np.ceil(period / required - 1e-9))
    return float(period) / max(count, 1)


def gain_bound(config: DatasetConfig) -> dict[str, Any]:
    """The Sec 2.2 discrete-time bound ``2 zeta omega T_c (1 + d) <= 0.5`` at the declared corner."""
    zeta = float(config.control_gains.damping_ratio[1])
    omega = float(config.control_gains.natural_frequency[1])
    period = config.control_period
    delay = config.control_delay
    value = 2.0 * zeta * omega * period * (1.0 + delay)
    return {"zeta_max": zeta, "omega_max_drawn": omega, "omega_max": config.omega_max(),
            "control_period": period, "delay_samples": delay, "value": value, "limit": 0.5,
            "ok": bool(value <= 0.5 * (1.0 + 1e-9))}


def require_round6(config: DatasetConfig, *, source: Any = "config") -> None:
    """Refuse a ``schema_version: 3`` config that breaks a round-6 rule (`R6_00` Sec 4).

    Every item of the list is an error, reported together: an unresolvable
    probe, a controller other than the model-free PD on the sampled loop and
    motor-inertia gains, no ``simulation.noise`` block (or a round-5
    ``measurement`` one), and a violated Sec 2.2 gain bound.  The Sec 3
    explicit-term bound needs the asset and is checked by
    :func:`elastic_sim.dataset_bounds.require_explicit_term_bound`.
    """
    if not config.is_round6:
        return
    errors: list[str] = []
    policy = differentiation_policy(config)
    if not policy["probe_resolvable"]:
        errors.append(
            f"probe top {policy['probe_top_hz']:g} Hz is above the {policy['sg_cutoff_hz']:.1f} Hz passband of "
            f"the consumer's Savitzky-Golay derivative at {policy['sample_rate_hz']:g} Hz "
            "(probe_resolvable: false)"
        )
    if config.controller.mode != "pd":
        errors.append(f"simulation.controller.mode is {config.controller.mode!r}; round 6 production is 'pd'")
    if config.controller.location == "bus" and config.controller.gain_sizing != "motor_inertia":
        errors.append("simulation.controller.gain_sizing must be 'motor_inertia' on the bus (Sec 2.2)")
    if config.controller.location == "drive":
        if config.controller.gain_sizing != "motor_plus_load":
            errors.append("simulation.controller.gain_sizing must be 'motor_plus_load' in a drive (R6_02 Sec 7)")
        ticks = float(config.sample_time_step) * float(config.controller.drive_rate)
        if abs(ticks - round(ticks)) > 1e-9 or round(ticks) < 1:
            errors.append(f"simulation.controller.drive.rate {config.controller.drive_rate:g} Hz is not an integer "
                          f"multiple of the {1.0 / config.sample_time_step:g} Hz sample rate")

    if config.controller.loop != "sample_rate":
        errors.append("simulation.controller.loop must be 'sample_rate' (Sec 2.1)")
    if config.noise is None:
        errors.append("simulation.noise is required (Sec 6)")
    elif config.noise.q.p > 0.0 or not any(v > 0.0 for v in config.noise.q.quantization):
        errors.append("simulation.noise.q must be quantization plus sigma_quanta steps, no percentage "
                      "of RMS (R6_02 Sec 3)")
    if not config.measurement.is_ideal:
        errors.append("simulation.measurement is the round-5 instrument; use simulation.noise")
    if config.motor_friction is None:
        errors.append("simulation.motor_friction is required (Sec 5.4)")
    if not config.control_gains.enabled or not config.control_gains.natural_frequency_fraction:
        errors.append("simulation.control_gains needs enabled: true and natural_frequency_fraction (Sec 2.2)")
    else:
        bound = gain_bound(config)
        if not bound["ok"]:
            errors.append(
                f"2 zeta omega T_c (1 + d) = {bound['value']:.3f} > 0.5 at zeta={bound['zeta_max']:g}, "
                f"omega={bound['omega_max_drawn']:.2f} rad/s, T_c={bound['control_period']:g} s, "
                f"d={bound['delay_samples']} (Sec 2.2; omega_max = {bound['omega_max']:.2f} rad/s)"
            )
    if errors:
        raise ValueError(f"{source}: schema_version 3 config refused:\n  - " + "\n  - ".join(errors))


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
