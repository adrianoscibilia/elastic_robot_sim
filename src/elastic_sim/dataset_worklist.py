"""Identification dataset work list: per-bag draws, splits, trajectory cache and ``generate``.

One *condition* is a choice of model tier (the rigid reference or one sampled
elastic robot), a friction sample and a simulator backend.  One *bag* is one
excitation trajectory executed under one condition.  Split out of
``dataset.py`` (`R5_10` T-7) with no behaviour change; import from
:mod:`elastic_sim.dataset`.
"""
from __future__ import annotations

import contextlib
import hashlib
import warnings
from dataclasses import asdict, dataclass, replace
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from .assets import AssetSpec
from .backend_comparison import compare_backends, format_report, summarize
from .controllers import ControllerDraw, describe_controller, effective_bandwidth, sample_velocity_loop
from .excitation import FourierExcitationConfig, optimize_excitation
from .identification import FrictionModel
from .kinematics import PortableKinematics
from .materialized import MaterializedTrajectory
from .payload import Payload, payload_asset
from .plant_extras import NO_EXTRAS, PlantExtras, StiffnessNonlinearity, TorqueRipple, TransmissionError
from .torque_runners import control_separation_ratio, link_inertia_envelope, link_inertia_max
from .dataset_bag import (
    _inertia_envelope_bounds,
    _run_bag,
    _sensor_for,
    describe_transmission,
    elastic_time_step,
)
from .dataset_config import (
    ControlGainSampling,
    MotorFrictionPrior,
    DatasetConfig,
    LinkInertia,
    PayloadSampling,
    PlantExtrasSampling,
    RegimeSampling,
    SplitPolicy,
    Tier,
    _TARGET_SEMANTICS,
    _log_uniform,
    _per_joint,
    differentiation_policy,
    warn_if_probe_outruns_differentiation,
)


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


def _robot_stream(robot: str) -> int:
    """A stable integer per robot name, so the draw survives adding robots."""
    return int.from_bytes(hashlib.blake2b(str(robot).encode("utf-8"), digest_size=4).digest(), "big")


def plant_extras_for_bag(
    sampling: PlantExtrasSampling, n_dof: int, dataset_seed: int, trajectory_seed_value: int,
    *, robot: str | None = None, stiffness: np.ndarray | None = None, effort: np.ndarray | None = None,
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

    Round 6: ``stiffness`` (the robot's sampled ``k``) converts torque knees
    into deflection breakpoints, and ``effort`` (the URDF limits) turns
    effort-fraction knees into torques (`R6_00` Sec 5.2).  Without a
    stiffness -- the rigid tier -- there is no spring to bend.  The
    transmission error's phase is drawn per bag from stream
    ``(seed, 12, trajectory_seed)`` (Sec 5.3).
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
    elif (sampling.stiffness_breakpoints_torque or sampling.stiffness_breakpoints_effort_fraction) \
            and stiffness is not None:
        if sampling.stiffness_breakpoints_torque:
            knees = np.asarray(sampling.stiffness_breakpoints_torque, dtype=float)
        else:
            if effort is None:
                raise ValueError("breakpoints_effort_fraction needs the joints' effort limits")
            knees = np.outer(np.asarray(effort, dtype=float),
                             np.asarray(sampling.stiffness_breakpoints_effort_fraction, dtype=float))
        nonlinearity = StiffnessNonlinearity.from_torque_knees(
            np.asarray(sampling.stiffness_factors, dtype=float), knees,
            _per_joint(stiffness, n_dof, "stiffness"),
        )
    ripple = None
    if sampling.ripple_amplitude:
        ripple = TorqueRipple(amplitude=float(sampling.ripple_amplitude), order=sampling.ripple_order)
        if sampling.ripple_random_phase:
            rng = np.random.default_rng((int(dataset_seed), 9, int(trajectory_seed_value)))
            ripple = ripple.with_phase(rng, n_dof)
    transmission_error = None
    if sampling.has_transmission_error and stiffness is not None:
        transmission_error = TransmissionError(
            amplitude=tuple(_per_joint(sampling.transmission_error_amplitude, n_dof, "transmission_error.amplitude")),
            order=tuple(_per_joint(sampling.transmission_error_order, n_dof, "transmission_error.order")),
        )
        if sampling.transmission_error_random_phase:
            rng = np.random.default_rng((int(dataset_seed), 12, int(trajectory_seed_value)))
            transmission_error = transmission_error.with_phase(rng, n_dof)
    return PlantExtras(
        link_friction=link_friction, stiffness_nonlinearity=nonlinearity, torque_ripple=ripple,
        transmission_error=transmission_error,
    )


def motor_friction_for(config: DatasetConfig, asset: AssetSpec, tier: Tier) -> FrictionModel:
    """The round-6 motor friction of one robot (`R6_00` Sec 5.4).

    The nominal rule (:class:`MotorFrictionPrior`) scaled per joint and per
    coefficient, log-uniformly over ``dataset.friction_scale``, from stream
    ``(seed, 11, robot)`` -- keyed on the robot's name, so a robot keeps its
    friction whatever else the build contains.  The rigid reference keeps the
    nominal: it is the stiff limit of the *nominal* robot.
    """
    prior = config.motor_friction
    base = prior.nominal(asset)
    if tier.is_rigid or not prior.per_robot:
        return base
    rng = np.random.default_rng((int(config.seed), 11, _robot_stream(tier.name)))
    low, high = np.log(config.friction_scale_range[0]), np.log(config.friction_scale_range[1])
    return base.scaled(np.exp(rng.uniform(low, high, size=base.n_dof)),
                       np.exp(rng.uniform(low, high, size=base.n_dof)))


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
    if sampling.bands:
        natural_frequency = _log_uniform_union(rng.uniform(), sampling.bands)
    else:
        natural_frequency = float(np.exp(
            rng.uniform(np.log(sampling.natural_frequency[0]), np.log(sampling.natural_frequency[1]))
        ))
    damping_ratio = float(rng.uniform(sampling.damping_ratio[0], sampling.damping_ratio[1]))
    return natural_frequency, damping_ratio


def _log_uniform_union(unit: float, bands: Sequence[tuple[float, float]]) -> float:
    """Map one ``U(0, 1)`` draw log-uniformly onto a union of ``(low, high)`` bands.

    One uniform number, as the single-band draw consumes, so the damping-ratio
    draw after it lands on the same stream position whichever bands a build
    uses: the gain-shift file then shares its bags' damping ratios with the
    production file (`R6_00` Sec 7 item 4, "same seeds").
    """
    widths = np.asarray([np.log(high) - np.log(low) for low, high in bands], dtype=float)
    position = float(unit) * float(widths.sum())
    for (low, high), width in zip(bands, widths):
        if position <= width or (low, high) == tuple(bands[-1]):
            return float(np.exp(np.log(low) + min(position, width)))
        position -= width
    raise AssertionError("unreachable")  # pragma: no cover


def sample_friction(base: FrictionModel, rng: np.random.Generator, scale_range: tuple[float, float]) -> FrictionModel:
    """Scale each joint's viscous and Coulomb coefficient log-uniformly."""
    low, high = np.log(scale_range[0]), np.log(scale_range[1])
    viscous = base.viscous * np.exp(rng.uniform(low, high, size=base.n_dof))
    coulomb = base.coulomb * np.exp(rng.uniform(low, high, size=base.n_dof))
    return FrictionModel(viscous, coulomb, base.epsilon)


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
                if config.only_tiers is not None and tier.name not in config.only_tiers:
                    continue
                for backend in config.backends:
                    yield trajectory_index, tier, friction_index, backend


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
    trajectory: MaterializedTrajectory | None = None,
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

    ``trajectory``, when given, is this bag's already-optimized trajectory
    (``generate()``'s cache, keyed by :func:`trajectory_key`) and skips the
    optimization; every other draw is still derived here.
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
        stiffness=None if tier.is_rigid else np.asarray(tier.stiffness, dtype=float),
        effort=np.asarray([np.inf if j.effort is None else j.effort for j in asset.resolve_active_joints()]),
    )

    if trajectory is None:
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


def trajectory_key(
    config: DatasetConfig, asset: AssetSpec, tier: Tier, index: int, payload: Payload | None,
) -> tuple:
    """Everything a bag's excitation trajectory depends on, and nothing else.

    The controller mode and the plant extras are not in it, so every mode x
    extras row of a matched build shares one optimization (`R5_10` T-3.3):
    the trajectory is a function of the asset, the seed, the regime-drawn
    excitation, the candidate count and the payload-fitted collision geometry.
    """
    traj_seed = trajectory_seed(config, tier, index)
    excitation = regime_excitation(config.excitation, config.regime, config.seed, traj_seed)
    payload_key = () if payload is None or payload.is_empty else (payload.mass, payload.offset, payload.size)
    return (asset.name, str(asset.urdf_path), tier.name if config.trajectories_per_robot else "", index,
            traj_seed, repr(excitation), int(config.candidates), payload_key)


def _optimize_trajectory(args: tuple) -> MaterializedTrajectory:
    """Process-pool entry point: one bag's trajectory, own kinematics."""
    config, asset, tier, index, payload = args
    return resolve_bag(config, asset, tier, index, payload).trajectory


#: The round-6 target and inputs, as the manifest and the contract state them (`R6_00` Sec 1).
ROUND6_TARGET_SEMANTICS = "transmission output torque applied to the link [N·m | N]"
ROUND6_INPUT_SEMANTICS = (
    "q0..q{{n-1}} = motor-side position as the encoder reports it (quantized, noisy, delay_samples old); "
    "dq0..dq{{n-1}} = motor-side velocity as the interface provides it: {dq}; "
    "tau0..tau{{n-1}} = commanded motor torque exactly as the model-free PD sent it, noise-free (not a "
    "sensor: the torque the plant receives differs from it by motor friction and ripple, which is physics)"
)


def _round6_manifest(config: DatasetConfig, asset: AssetSpec) -> dict[str, Any]:
    """What a round-6 file adds to its manifest (`R6_00` Secs 2, 5, 6; `R6_02` Secs 2, 7)."""
    from .dataset_config import gain_bound
    from .link_modes import link_mode_table

    modes = link_mode_table(config, asset)
    return {
        "schema_version": int(config.schema_version),
        # R6_02 P2-1: each joint's link-side mode band over the prior, and
        # which joints the budget gate may count as elastically observable.
        "link_modes": {
            "definition": ("f_link = sqrt(K / M_link) / (2 pi), K over the stiffness prior, M_link = M_jj(q) from "
                           "the bare arm's smallest to the largest with the heaviest payload; resolvable band "
                           "0.09 x sample rate (the SG derivative's five-sample passband)"),
            "joints": [mode.as_dict() for mode in modes],
            "observable_joints": [mode.joint for mode in modes if mode.observable],
            "mostly_observable_joints": [mode.joint for mode in modes if mode.mostly_observable],
        },
        "probe_design": None if config.probe_design is None else {
            **config.probe_design.as_dict(),
            "harmonics": list(config.excitation.probe_harmonics),
            "frequencies_hz": [h * config.excitation.base_frequency for h in config.excitation.probe_harmonics],
            "probe_acceleration_fraction": float(config.excitation.probe_acceleration_fraction),
        },
        "noise": config.noise.describe(),
        "motor_friction": {**config.motor_friction.describe(),
                           "nominal": {"viscous": config.motor_friction.nominal(asset).viscous.tolist(),
                                       "coulomb": config.motor_friction.nominal(asset).coulomb.tolist()},
                           "friction_scale": list(config.friction_scale_range)},
        "control_timing": {
            "location": config.controller.location,
            "control_period": config.control_period,
            "sample_period": float(config.sample_time_step),
            "delay_samples": int(config.control_delay),
            "recording_delay_samples": int(config.loop_delay),
            "zero_order_hold": True,
            "controller_reads": (
                "measured motor-side q/dq, delay_samples old: the recorded q*/dq* channels"
                if config.controller.location == "bus" else
                "the drive's own encoder (quantization + sigma_quanta counts) every drive period, undelayed, and "
                "its velocity = encoder difference through a first-order low-pass at "
                f"{config.controller.velocity_cutoff_fraction:g} x drive rate; setpoints from the bus, held, "
                f"{config.controller.setpoint_delay} sample(s) late; the bus records every "
                f"{config.drive_ticks}th reading, recording_delay_samples old, with the Sec 6 dq percentage on "
                "the recorded copy only (R6_02 Sec 7 amended, R6_04 A-4)"
            ),
            "tau": ("the command held over the sample period" if config.controller.location == "bus" else
                    "the drive's torque demand at the sample instant (0x6074 / target_current x k_t)"),
        },
        "gain_sizing": {
            "rule": ("kp_j = J_j,nom omega^2, kd_j = 2 zeta J_j,nom omega (nominal rotor inertia)"
                     if config.controller.gain_sizing == "motor_inertia" else
                     "kp_j = (J_j,nom + m_j,nom) omega_j^2, kd_j = 2 zeta (J_j,nom + m_j,nom) omega_j, "
                     "m_j,nom the nominal link inertia's mean diagonal along the reference (per bag: "
                     "records[].drive_load_inertia); omega_j = omega x J_j / (J_j + m_j,nom), the rotor-referred "
                     "bound 2 zeta omega_j T (1 + d) (J + m) / J <= 0.5 (R6_04 A-2)"),
            **({} if config.controller.location != "drive" else {"drive_bound": config.controller.drive_bound}),
            "nominal_rotor_inertia": config.nominal_rotor_inertia(len(asset.joint_names)).tolist(),
            "omega_max": config.omega_max(),
            "natural_frequency_fraction": [list(b) for b in config.control_gains.natural_frequency_fraction],
            "natural_frequency_bands": [list(b) for b in config.control_gains.bands],
            "bound": gain_bound(config),
        },
        "rigid_tier": (
            "stiff limit of the same robot: nominal rotor inertia as armature, motor friction on the same "
            "DOF, link friction and motor ripple; target rnea_link(q, dq, ddq) + f_link" if config.rigid_reference
            else None
        ),
        "gates": config.gates.describe(),
        "only_tiers": None if config.only_tiers is None else list(config.only_tiers),
    }


def default_jobs(backends: Sequence[str]) -> int:
    """``nproc - 1``, or 1 when Newton is a backend (`R5_10` T-3.4).

    Newton holds a GPU context that a forked worker must not inherit.
    """
    if "newton" in backends:
        return 1
    import os

    return max(1, (os.cpu_count() or 2) - 1)


def generate(
    config: DatasetConfig, asset: AssetSpec, *, verbose: bool = True, jobs: int = 1,
    trajectory_cache: dict[tuple, MaterializedTrajectory] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Build the whole dataset and return ``(frame, manifest, backend_comparison)``.

    ``trajectory_cache`` is shared across calls by a caller building several
    matched rows (modes x extras) of one platform; see :func:`trajectory_key`.
    ``jobs > 1`` runs trajectory optimization and bags in a process pool, and
    is forced to 1 when Newton is a backend.
    """
    if jobs > 1 and "newton" in config.backends:
        warnings.warn("jobs forced to 1: Newton is in backends (R5_10 T-3.4)", stacklevel=2)
        jobs = 1
    trajectory_cache = {} if trajectory_cache is None else trajectory_cache
    warn_if_probe_outruns_differentiation(config)
    base_friction = FrictionModel.from_asset(asset) if config.motor_friction is None else (
        config.motor_friction.nominal(asset)
    )
    if config.motor_friction is not None and config.n_friction_samples != 1:
        raise ValueError("simulation.motor_friction draws friction per robot; set dataset.friction_samples: 1")
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
    pool = None
    if jobs > 1 and not config.visualize:
        import concurrent.futures

        pool = concurrent.futures.ProcessPoolExecutor(max_workers=jobs)
    try:
        # Optimize every trajectory this build does not already have, in
        # parallel when there is a pool: it is most of a build's wall time.
        resolve_tiers = config.tiers if config.trajectories_per_robot else config.tiers[:1]
        if config.only_tiers is not None and config.trajectories_per_robot:
            resolve_tiers = tuple(t for t in resolve_tiers if t.name in config.only_tiers)
        pending: dict[tuple, tuple] = {}
        for tier in resolve_tiers:
            for index in range(config.n_trajectories):
                payload = _payload_for(tier, index) if config.trajectories_per_robot else Payload()
                cache_key = trajectory_key(config, asset, tier, index, payload)
                if cache_key not in trajectory_cache and cache_key not in pending:
                    pending[cache_key] = (config, asset, tier, index, payload)
        if pending:
            if pool is not None:
                built = list(pool.map(_optimize_trajectory, pending.values()))
            else:
                built = [
                    resolve_bag(config, asset, tier, index, payload, kinematics_for=_kinematics_for).trajectory
                    for (_, _, tier, index, payload) in pending.values()
                ]
            trajectory_cache.update(zip(pending, built))

        for tier in resolve_tiers:
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
                resolved = resolve_bag(
                    config, asset, tier, index, payload, kinematics_for=_kinematics_for,
                    trajectory=trajectory_cache[trajectory_key(config, asset, tier, index, payload)],
                )
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
            report = _kinematics_for(payload).validate_path(trajectory.position, margin=margin, max_joint_step=max_joint_step,
                                                            stop_at_first_invalid=True)
            return report.valid

        work: list[tuple] = []
        offending: list[tuple[str, float, str]] = []
        for bag_index, (traj_index, tier, friction_index, backend) in enumerate(iter_conditions(config)):
            trajectory = trajectories[(tier.name if config.trajectories_per_robot else "", traj_index)]
            friction = (frictions[friction_index] if config.motor_friction is None
                        else motor_friction_for(config, asset, tier))
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
    except BaseException:
        if pool is not None:
            pool.shutdown(cancel_futures=True)
        raise
    finally:
        kinematics_stack.close()

    # Checked once the whole work list is built, *before* a single bag is
    # dispatched to a worker: with "action: error", raising from inside
    # _run_bag would let bags queued ahead of the first offender in the pool
    # already run, sometimes for minutes to hours (R4_10 Sec 2.3).
    frames: list[pd.DataFrame] = []
    records: list[dict[str, Any]] = []
    try:
        if config.control_separation.action == "error":
            raise_on_control_separation_violation(offending, config.control_separation.min_ratio)
        if pool is not None:
            results = list(pool.map(_run_bag, work))
        else:
            results = [_run_bag(item) for item in work]
    finally:
        if pool is not None:
            pool.shutdown(cancel_futures=True)
    unstable_bags: list[dict[str, Any]] = []
    for bag_frame, record in results:
        if bag_frame is None:
            # `R5_10` T-2: excluded from the frame and from `records` (which
            # the backend comparison and the diagnostics pair with frame
            # bags), listed in the manifest with what diverged and the gains.
            unstable_bags.append(record)
            message = (f"bag {record['bag']!r} is unstable ({record['kind']} at t={record['time']:.4f}s, "
                       f"joint {record['joint']!r}); excluded from the dataset")
            warnings.warn(message, stacklevel=2)
            if verbose:
                print(f"  [{record['bag_index'] + 1}] {record['bag']:30s} UNSTABLE: {record['message']}")
            continue
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
    too_close = [r for r in records if r["achieved_min_clearance_m"] is not None
                 and r["achieved_min_clearance_m"] < collision_margin]
    if too_close:
        worst = min(too_close, key=lambda r: r["achieved_min_clearance_m"])
        message = (f"{len(too_close)}/{len(records)} bags' achieved path (not the reference, which is "
                   f"validated collision-free at build time) comes closer than margin={collision_margin:g} m; "
                   f"worst is bag {worst['bag']!r} at {worst['achieved_min_clearance_m']:.4f} m "
                   "(R4_14 Sec 2.3: tracking error, not a code defect -- flagged, not dropped)")
        warnings.warn(message, stacklevel=2)
        if verbose:
            print(f"warning: {message}")

    if not frames:
        raise RuntimeError(
            f"every bag of this build is unstable ({len(unstable_bags)}); first: {unstable_bags[0]['message']}"
        )
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
        "unstable_bags": unstable_bags,
        "n_dof": n_dof,
        "joint_names": list(asset.joint_names),
        "sample_time_step": config.sample_time_step,
        "control_decimation": config.control_decimation,
        "controller": describe_controller(config.controller),
        "measurement": config.measurement.describe(),
        **(_round6_manifest(config, asset) if config.is_round6 else {}),
        "differentiation": differentiation_policy(config),
        "signals": config.signals.describe(),
        "force_torque_sensor": (
            None if not config.signals.needs_cell
            else _sensor_for(asset, config.signals).describe()
        ),
        "plant_extras": None if config.plant_extras.is_empty else asdict(config.plant_extras),
        "seed": config.seed,
        "trajectories_per_robot": config.trajectories_per_robot,
        "distinct_trajectories": len(trajectories),
        # One digest per (tier, trajectory index): identical across every
        # matched mode x extras row of a platform (`R5_10` T-3.3).
        "trajectory_digests": {f"{name or 'shared'}/t{index}": trajectory.digest()
                               for (name, index), trajectory in trajectories.items()},
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
        "target": (f"ft0..ft{{n-1}} = {_TARGET_SEMANTICS[config.signals.target]}" if not config.is_round6
                   else f"ft0..ft{{n-1}} = {ROUND6_TARGET_SEMANTICS}, "
                        + ("through the platform's torque-measurement model (simulation.noise.ft)"
                           if config.signals.target_source == "measured" and config.noise.enabled
                           else "noise-free")),
        "input": (
            f"q0..q{{n-1}}, dq0..dq{{n-1}} = {config.signals.position_side}-side position and velocity; "
            "tau0..tau{n-1} = commanded motor torque (motor effort)"
        ) if not config.is_round6 else ROUND6_INPUT_SEMANTICS.format(
            dq=("the consumer's Savitzky-Golay derivative of the recorded q (the interface gives no velocity)"
                if config.noise.dq_source == "position_derivative" else "the drive's velocity estimate"),
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
