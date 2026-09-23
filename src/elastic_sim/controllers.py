"""Controller modes for identification rollouts (round 5).

Round 4 generated every dataset with one controller: exact computed torque,
built from the *payload-fitted* URDF with ``f_hat = f`` and ``J_hat = J``
(``torque_runners.ComputedTorqueController`` / ``SeaMotorController``).  That
controller knows the plant exactly, so the commanded torque, the state and the
link-side torque are all near-exact functions of the same reference: the data
cannot *attribute* the target to the right physical term, because rigid
Lagrangian, elastic potential, residual and "subtract motor inertia from
``tau_cmd``" all fit it equally well (``R5_00`` Sec 5.3 and its Amendment 1).
It also has no counterpart on any of the three real platforms: FMRR's drives
run Cyclic Synchronous Position with the loops closed inside the drive, and
neither arm is commanded by a controller that knows its payload
(``R5_01`` Amendment 1).

This module adds the alternatives, all with the same call signature as the
round-4 controllers so both torque runners accept them unchanged:

``exact_ct``
    Round 4's controller, bit-identical, still the default.
``nominal_ct``
    Computed torque built from a *nominal* model: payload-free, friction
    mis-scaled, rotor inertia mis-scaled.  What an industrial arm's own
    controller does.
``pd_gravity``
    Per-joint PD plus nominal gravity: no inertia model in the loop.
``pd``
    Per-joint PD alone, the legacy 3-DoF platform's decentralized law
    (``R5_00`` Sec 3).
``velocity_pi``
    The owner's choice for round 5 (``R5_01`` Amendment 2): a model-free outer
    position loop on a velocity interface, with a PI velocity loop underneath
    standing in for the drive's own inner loop, whose gains are randomized per
    bag so a network cannot memorize one drive tuning.

Every mode below ``exact_ct`` breaks the collinearity deliberately: the plant
model is either absent from the loop or wrong, so the tracking error, and with
it ``tau_cmd``, is driven by the dynamics rather than by the reference alone.
Whether that is enough is measured, not argued -- see
:mod:`elastic_sim.diagnostics` (``R5_00`` Q-C).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .assets import AssetSpec
from .identification import COULOMB_EPSILON, FrictionModel
from .materialized import MaterializedTrajectory
from .torque_runners import (
    ComputedTorqueController,
    SeaMotorController,
    TorqueCommand,
    TransmissionSpec,
)

#: Every mode ``simulation.controller.mode`` accepts.  ``exact_ct`` first: it
#: is the default and the round-4 invariant (``R5_02`` Sec 1).
CONTROLLER_MODES = ("exact_ct", "nominal_ct", "pd_gravity", "pd", "velocity_pi")

#: Modes whose loop contains no model of the plant at all.  Their commanded
#: torque is a function of the tracking error only, which is what makes
#: ``tau_cmd`` independent of the state regressor (``R5_00`` Sec 5.3).
MODEL_FREE_MODES = ("pd", "velocity_pi")


@dataclass(frozen=True)
class NominalModelSpec:
    """How wrong the controller's own model is allowed to be.

    Only meaningful for ``nominal_ct`` and ``pd_gravity`` (the two modes that
    carry a model at all).  The defaults are the *exact* model, so a config
    that names ``nominal_ct`` without a ``nominal`` block gets a controller
    that differs from ``exact_ct`` in nothing but its class -- deliberately,
    so that a build can separate "the model is nominal" from "the model is
    wrong by this much".

    ``knows_payload: false`` is the single most important one and the only
    one with no free parameter: a real controller is commissioned without the
    tool the robot is later run with, so the payload's inertia and its
    gravity torque are missing from the loop entirely.
    """

    knows_payload: bool = True
    friction_scale: float = 1.0
    rotor_inertia_scale: float = 1.0
    #: Multiplies every link mass and inertia the controller's model uses.
    #: A single scalar, not a per-link draw: the point is a loop whose inertia
    #: is systematically off, not an identification problem of its own.
    inertia_scale: float = 1.0

    def __post_init__(self) -> None:
        for name in ("friction_scale", "rotor_inertia_scale", "inertia_scale"):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"controller.nominal.{name} must be positive")

    @property
    def is_exact(self) -> bool:
        return (
            self.knows_payload
            and self.friction_scale == 1.0
            and self.rotor_inertia_scale == 1.0
            and self.inertia_scale == 1.0
        )


@dataclass(frozen=True)
class VelocityLoopSpec:
    """Gain ranges of the ``velocity_pi`` cascade, drawn once per bag.

    ``position_gain`` [1/s] is the outer, model-free loop
    ``v_cmd = dq_ref + kp (q_ref - theta)``; that is the part we would write
    ourselves on the real robot, over ``ur_robot_driver``'s velocity interface
    or FMRR's ``forward_velocity_controller``.  ``velocity_bandwidth``
    [rad/s] and ``integral_time`` [s] describe the drive's *own* inner loop,
    which is not ours and whose tuning we do not know -- hence a range, drawn
    per bag, rather than one number (``R5_01`` Amendment 2).

    The ranges are sampled log-uniformly, like every other per-trajectory
    draw in this pipeline.  ``position_gain`` must stay well below
    ``velocity_bandwidth`` for the cascade to separate; the constructor
    enforces the standard factor of two on the whole declared range rather
    than per draw, so an unusable config fails when it is loaded, not on the
    unlucky bag.
    """

    position_gain: tuple[float, float] = (10.0, 10.0)
    velocity_bandwidth: tuple[float, float] = (100.0, 100.0)
    integral_time: tuple[float, float] = (0.05, 0.05)

    def __post_init__(self) -> None:
        for name in ("position_gain", "velocity_bandwidth", "integral_time"):
            low, high = (float(v) for v in getattr(self, name))
            if low <= 0.0 or high < low:
                raise ValueError(f"controller.velocity_loop.{name} must satisfy 0 < min <= max")
            object.__setattr__(self, name, (low, high))
        if self.position_gain[1] > 0.5 * self.velocity_bandwidth[0]:
            raise ValueError(
                f"controller.velocity_loop.position_gain up to {self.position_gain[1]:g} 1/s is not "
                f"separated from a velocity loop as slow as {self.velocity_bandwidth[0]:g} rad/s "
                "(need position_gain <= velocity_bandwidth / 2 for the cascade to be stable)"
            )


@dataclass(frozen=True)
class ControllerSpec:
    """Which controller a dataset build uses, and how it is randomized."""

    mode: str = "exact_ct"
    nominal: NominalModelSpec = None  # type: ignore[assignment]
    velocity_loop: VelocityLoopSpec = None  # type: ignore[assignment]
    #: Draw ``velocity_loop`` gains per trajectory instead of taking the low
    #: end of each range.  Off at the dataclass level for the same reason
    #: ``payload``/``regime``/``control_gains`` are: a directly-built
    #: ``DatasetConfig`` (most tests) stays deterministic unless it opts in.
    randomize: bool = False

    def __post_init__(self) -> None:
        if self.mode not in CONTROLLER_MODES:
            raise ValueError(f"simulation.controller.mode must be one of {CONTROLLER_MODES}, got {self.mode!r}")
        if self.nominal is None:
            object.__setattr__(self, "nominal", NominalModelSpec())
        if self.velocity_loop is None:
            object.__setattr__(self, "velocity_loop", VelocityLoopSpec())

    @property
    def is_model_free(self) -> bool:
        return self.mode in MODEL_FREE_MODES

    @property
    def is_round4_default(self) -> bool:
        """True when this spec reproduces round 4's controller exactly."""
        return self.mode == "exact_ct"


@dataclass(frozen=True)
class ControllerDraw:
    """One bag's realized controller gains, recorded alongside the data.

    ``natural_frequency``/``damping_ratio`` are the existing
    ``simulation.control_gains`` draw and drive every mode but
    ``velocity_pi``; the three velocity-loop values drive only that one.  All
    five are written to the manifest and to per-row columns whatever the mode,
    with the unused ones held at the spec's own value, so a cross-mode
    comparison reads one schema.
    """

    natural_frequency: float
    damping_ratio: float
    position_gain: float
    velocity_bandwidth: float
    integral_time: float

    def as_dict(self) -> dict[str, float]:
        return {
            "control_natural_frequency": float(self.natural_frequency),
            "control_damping_ratio": float(self.damping_ratio),
            "control_position_gain": float(self.position_gain),
            "control_velocity_bandwidth": float(self.velocity_bandwidth),
            "control_integral_time": float(self.integral_time),
        }


def sample_velocity_loop(
    spec: ControllerSpec, dataset_seed: int, trajectory_seed_value: int,
) -> tuple[float, float, float]:
    """Draw ``(position_gain, velocity_bandwidth, integral_time)`` for one bag.

    Stream ``(seed, 7, trajectory_seed)``, a new stream index: enabling this
    perturbs none of the existing draws (robots ``1``, payload ``2``, regime
    ``3``, rotor inertia ``4``, payload re-draws ``5``, control gains ``6``).
    ``generate()`` and ``run_identification_simulation.py`` must call it the
    same way, the same requirement ``regime_excitation`` and
    ``sample_control_gains`` carry.
    """
    loop = spec.velocity_loop
    if not spec.randomize:
        return loop.position_gain[0], loop.velocity_bandwidth[0], loop.integral_time[0]
    rng = np.random.default_rng((int(dataset_seed), 7, int(trajectory_seed_value)))

    def _log_uniform(bounds: tuple[float, float]) -> float:
        return float(np.exp(rng.uniform(np.log(bounds[0]), np.log(bounds[1]))))

    # Drawn in declaration order so the stream stays reproducible.
    return (
        _log_uniform(loop.position_gain),
        _log_uniform(loop.velocity_bandwidth),
        _log_uniform(loop.integral_time),
    )


def effective_bandwidth(spec: ControllerSpec, draw: ControllerDraw) -> float:
    """The fastest loop this controller closes, in rad/s.

    The control/transmission separation guard (``dataset.ControlSeparationCheck``)
    compares a closed-loop bandwidth against ``sqrt(k / J_eff)``.  For every
    computed-torque and PD mode that bandwidth is the error dynamics'
    ``natural_frequency``; for ``velocity_pi`` it is the *inner* velocity
    loop, which is far faster than the outer position gain and is what would
    actually ring against the transmission mode.
    """
    if spec.mode == "velocity_pi":
        return float(draw.velocity_bandwidth)
    return float(draw.natural_frequency)


# ---------------------------------------------------------------------------
# Nominal model
# ---------------------------------------------------------------------------

def nominal_friction(friction: FrictionModel, spec: NominalModelSpec) -> FrictionModel:
    """The friction model the controller believes in."""
    if spec.friction_scale == 1.0:
        return friction
    scale = float(spec.friction_scale)
    return FrictionModel(friction.viscous * scale, friction.coulomb * scale)


def nominal_transmission(transmission: TransmissionSpec, spec: NominalModelSpec) -> TransmissionSpec:
    """The transmission the controller believes in (rotor inertia only).

    The spring is *not* nominalized: no mode in this module knows the spring
    at all.  ``SeaMotorController`` uses ``transmission`` solely for
    ``rotor_inertia``, which is the motor-side inertia a drive's commissioning
    does have a (possibly wrong) figure for.
    """
    if spec.rotor_inertia_scale == 1.0:
        return transmission
    return replace(transmission, rotor_inertia=transmission.rotor_inertia * float(spec.rotor_inertia_scale))


def nominal_joint_inertia(
    asset: AssetSpec, trajectory: MaterializedTrajectory, *, spec: NominalModelSpec | None = None,
) -> np.ndarray:
    """Diagonal of the nominal link-side inertia along the reference path.

    The PD and velocity-loop modes need a per-joint inertia to convert a
    target bandwidth into a torque gain (``K_p = (M_ii + J_rotor) omega^2``,
    ``R5_00`` Sec 6.1).  A real commissioning does this once, from a
    datasheet or a nominal model, not per configuration -- so this is the
    *mean* diagonal over the reference trajectory of the nominal (usually
    payload-free) model, evaluated once when the controller is built, and it
    stays fixed for the whole rollout even as the true ``M(q)`` swings by a
    factor of a few.  That mismatch is part of what the mode is for.
    """
    from . import identification as idn

    pin, model, data = idn.build_model(asset)
    if spec is not None:
        model, data = idn.scale_model_inertias(pin, model, spec.inertia_scale)
    positions = np.atleast_2d(np.asarray(trajectory.sample()["q"], dtype=float))
    # 32 configurations is enough for a mean: the diagonal varies smoothly
    # along a Fourier trajectory and the result only sizes a gain.
    stride = max(1, len(positions) // 32)
    diagonals = np.asarray([
        np.diag(np.asarray(pin.crba(model, data, positions[index]), dtype=float))
        for index in range(0, len(positions), stride)
    ])
    return np.asarray(np.mean(diagonals, axis=0), dtype=float)


def _effort_limit(asset: AssetSpec, effort_limit: np.ndarray | None) -> np.ndarray:
    if effort_limit is not None:
        return np.asarray(effort_limit, dtype=float)
    return np.asarray([
        joint.effort if joint.effort is not None else np.inf
        for joint in asset.resolve_active_joints()
    ], dtype=float)


# ---------------------------------------------------------------------------
# Model-free and partially-model-free controllers
# ---------------------------------------------------------------------------

class JointPdController:
    """Decentralized joint PD, optionally with nominal gravity feedforward.

    ``tau = kp (q_ref - x) + kd (dq_ref - v) [+ g_nominal(x)]`` with per-joint
    gains sized from a target bandwidth and the nominal inertia:
    ``kp_j = (M_jj + J_j) omega^2``, ``kd_j = 2 zeta (M_jj + J_j) omega``.
    ``x, v`` are the *motor*-side state on an elastic robot (collocated, hence
    stable as the transmission softens) and the joint state on a rigid one.

    Unlike computed torque this does not decouple the axes -- on an arm whose
    inertia matrix is badly conditioned the off-diagonal coupling shows up as
    tracking error, which is the point: the error, and with it the commanded
    torque, is then driven by the dynamics rather than by the reference
    (``R5_00`` Sec 5.3).  The trade is that the achieved motion is no longer
    the designed excitation; the recorded label stays exact regardless,
    because it is measured from the plant, not computed from the reference.

    Without gravity compensation (``mode: pd``) the steady-state error on a
    gravity-loaded joint is ``tau_g / kp``, tens of degrees on an arm.  That
    is faithful to the legacy platform (``R5_00`` Sec 3) and is why the
    *achieved* path, not the reference, is what a round-5 build must validate
    against the joint window and the collision geometry.
    """

    def __init__(
        self,
        asset: AssetSpec,
        trajectory: MaterializedTrajectory,
        *,
        joint_inertia: np.ndarray,
        natural_frequency: float = 25.0,
        damping_ratio: float = 1.0,
        gravity: bool = False,
        nominal: NominalModelSpec | None = None,
        effort_limit: np.ndarray | None = None,
    ) -> None:
        from . import identification as idn

        if natural_frequency <= 0.0 or damping_ratio <= 0.0:
            raise ValueError("natural_frequency and damping_ratio must be positive")
        self.asset = asset
        self.trajectory = trajectory
        self.natural_frequency = float(natural_frequency)
        self.damping_ratio = float(damping_ratio)
        self.nominal = nominal or NominalModelSpec()
        self.joint_inertia = np.asarray(joint_inertia, dtype=float).reshape(-1)
        if np.any(self.joint_inertia <= 0.0):
            raise ValueError("joint_inertia must be positive on every joint")
        # Torque gains, in Nm/rad and Nm/(rad/s) -- not the error-dynamics
        # gains ComputedTorqueController uses.
        self.kp = self.joint_inertia * self.natural_frequency**2
        self.kd = 2.0 * self.damping_ratio * self.joint_inertia * self.natural_frequency
        self.uses_gravity = bool(gravity)
        self.effort_limit = _effort_limit(asset, effort_limit)
        self._idn = idn
        if self.uses_gravity:
            pin, model, data = idn.build_model(asset)
            model, data = idn.scale_model_inertias(pin, model, self.nominal.inertia_scale)
            self._pin, self._model, self._data = pin, model, data

    def stability_margin(self, time_step: float) -> float:
        """``kd / (M + J) * dt``; explicit integration needs it below 2."""
        return float(np.max(self.kd / self.joint_inertia) * float(time_step))

    def __call__(
        self, t: float, q: np.ndarray, dq: np.ndarray, tau_spring: np.ndarray | None = None,
    ) -> TorqueCommand:
        q_ref, dq_ref, _ = self.trajectory(float(t))
        q = np.asarray(q, dtype=float)
        dq = np.asarray(dq, dtype=float)
        feedback = self.kp * (q_ref - q) + self.kd * (dq_ref - dq)
        if self.uses_gravity:
            feedforward = np.asarray(
                self._pin.computeGeneralizedGravity(self._model, self._data, q), dtype=float
            )
        else:
            feedforward = np.zeros_like(feedback)
        total = np.clip(feedforward + feedback, -self.effort_limit, self.effort_limit)
        # Reported split is of the *saturated* total, so feedforward +
        # feedback == total holds for every recorded sample, as it does for
        # the computed-torque controllers.
        return TorqueCommand(total=total, feedforward=feedforward, feedback=total - feedforward)


class VelocityLoopController:
    """Model-free position loop over a PI velocity loop (``R5_01`` Amendment 2).

    ::

        v_cmd = dq_ref + kp_pos (q_ref - x)                 (ours, model-free)
        e_v   = v_cmd - v
        tau   = kv (e_v + (1 / T_i) integral e_v dt)        (the drive's own)

    with ``kv = (M_jj + J_j) omega_v``, so ``omega_v`` is the inner loop's
    bandwidth in rad/s.  This is the round-5 controller the owner chose: it is
    what all three platforms can actually run (UR through
    ``ur_robot_driver``'s velocity interface, FMRR through the
    ``forward_velocity_controller`` that already exists in its launch file),
    it contains no model of the plant, and the inner loop's gains are the
    drive's, not ours -- so they are drawn per bag instead of fixed.

    The integral is what makes this usable where plain PD is not: it carries
    gravity and friction with no model, so the steady-state error goes to zero
    while the loop stays model-free.  It is also why the split reported in
    ``TorqueCommand`` puts the integral in ``feedforward`` and the
    proportional part in ``feedback``: in steady state the integral *is* the
    feedforward a model-based controller would have computed, and the ratio of
    the two stays the same diagnostic it is for the other modes.

    Integration is conditional (anti-windup): the error is accumulated only
    when the output is inside the effort limit or the error pushes it back in.
    Without that, the first saturated transient winds the integral up and the
    joint overshoots for the rest of the bag.
    """

    def __init__(
        self,
        asset: AssetSpec,
        trajectory: MaterializedTrajectory,
        *,
        joint_inertia: np.ndarray,
        position_gain: float,
        velocity_bandwidth: float,
        integral_time: float,
        effort_limit: np.ndarray | None = None,
    ) -> None:
        if position_gain <= 0.0 or velocity_bandwidth <= 0.0 or integral_time <= 0.0:
            raise ValueError("position_gain, velocity_bandwidth and integral_time must be positive")
        self.asset = asset
        self.trajectory = trajectory
        self.joint_inertia = np.asarray(joint_inertia, dtype=float).reshape(-1)
        if np.any(self.joint_inertia <= 0.0):
            raise ValueError("joint_inertia must be positive on every joint")
        self.position_gain = float(position_gain)
        self.velocity_bandwidth = float(velocity_bandwidth)
        self.integral_time = float(integral_time)
        self.kv = self.joint_inertia * self.velocity_bandwidth
        self.effort_limit = _effort_limit(asset, effort_limit)
        self.reset()

    def reset(self) -> None:
        self._integral = np.zeros(len(self.joint_inertia))
        self._last_time: float | None = None

    def stability_margin(self, time_step: float) -> float:
        """``omega_v * dt``; explicit integration of the loop needs it << 1."""
        return float(self.velocity_bandwidth * float(time_step))

    def __call__(
        self, t: float, q: np.ndarray, dq: np.ndarray, tau_spring: np.ndarray | None = None,
    ) -> TorqueCommand:
        t = float(t)
        q_ref, dq_ref, _ = self.trajectory(t)
        q = np.asarray(q, dtype=float)
        dq = np.asarray(dq, dtype=float)
        velocity_command = dq_ref + self.position_gain * (q_ref - q)
        error = velocity_command - dq
        step = 0.0 if self._last_time is None else max(t - self._last_time, 0.0)
        self._last_time = t

        proportional = self.kv * error
        candidate = self._integral + error * step
        integral_torque = self.kv / self.integral_time * candidate
        unsaturated = proportional + integral_torque
        # Conditional integration: accept the new integral only where the
        # command is within the limit, or where this error unwinds it.
        saturated = np.abs(unsaturated) > self.effort_limit
        winding_up = saturated & (np.sign(error) == np.sign(unsaturated))
        self._integral = np.where(winding_up, self._integral, candidate)
        feedforward = self.kv / self.integral_time * self._integral
        total = np.clip(feedforward + proportional, -self.effort_limit, self.effort_limit)
        return TorqueCommand(total=total, feedforward=feedforward, feedback=total - feedforward)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_controller(
    spec: ControllerSpec,
    draw: ControllerDraw,
    *,
    asset: AssetSpec,
    nominal_asset: AssetSpec,
    trajectory: MaterializedTrajectory,
    friction: FrictionModel,
    transmission: TransmissionSpec | None = None,
    epsilon: float = COULOMB_EPSILON,
    joint_inertia: np.ndarray | None = None,
) -> Any:
    """Return the controller ``spec`` names, ready for either torque runner.

    ``asset`` is the true, payload-fitted asset the plant is simulated from;
    ``nominal_asset`` is what the controller is allowed to know -- the same
    object for ``knows_payload: true``, the bare asset otherwise.  Resolving
    the two outside keeps this function free of the payload machinery
    (``payload_asset`` is a context manager whose temporary URDF must outlive
    the controller) and makes the distinction explicit at the call site.

    ``transmission`` is required for the elastic tier and ignored on the rigid
    one.  ``joint_inertia`` overrides the nominal diagonal the PD and
    velocity-loop gains are sized from; it is computed from ``nominal_asset``
    when not given.
    """
    if spec.mode == "exact_ct" and not spec.nominal.is_exact:
        raise ValueError(
            "simulation.controller.mode: 'exact_ct' with a non-exact `nominal` block is a "
            "contradiction; use mode: nominal_ct to make the controller's model wrong"
        )
    model_asset = nominal_asset if spec.mode in ("nominal_ct", "pd_gravity") else asset
    if spec.mode in ("exact_ct", "nominal_ct"):
        controller_friction = friction if spec.mode == "exact_ct" else nominal_friction(friction, spec.nominal)
        inertia_scale = 1.0 if spec.mode == "exact_ct" else spec.nominal.inertia_scale
        if transmission is None:
            return ComputedTorqueController(
                model_asset, trajectory, friction=controller_friction,
                natural_frequency=draw.natural_frequency, damping_ratio=draw.damping_ratio,
                epsilon=epsilon, effort_limit=_effort_limit(asset, None),
                inertia_scale=inertia_scale,
            )
        controller_transmission = (
            transmission if spec.mode == "exact_ct" else nominal_transmission(transmission, spec.nominal)
        )
        return SeaMotorController(
            model_asset, trajectory, controller_transmission, friction=controller_friction,
            natural_frequency=draw.natural_frequency, damping_ratio=draw.damping_ratio,
            epsilon=epsilon, effort_limit=_effort_limit(asset, None),
            inertia_scale=inertia_scale,
        )

    rotor = np.zeros(len(asset.joint_names)) if transmission is None else (
        nominal_transmission(transmission, spec.nominal).rotor_inertia
    )
    if joint_inertia is None:
        joint_inertia = nominal_joint_inertia(model_asset, trajectory, spec=spec.nominal)
    joint_inertia = np.asarray(joint_inertia, dtype=float).reshape(-1) + np.asarray(rotor, dtype=float)

    if spec.mode in ("pd", "pd_gravity"):
        return JointPdController(
            model_asset, trajectory, joint_inertia=joint_inertia,
            natural_frequency=draw.natural_frequency, damping_ratio=draw.damping_ratio,
            gravity=spec.mode == "pd_gravity", nominal=spec.nominal,
            effort_limit=_effort_limit(asset, None),
        )
    if spec.mode == "velocity_pi":
        return VelocityLoopController(
            asset, trajectory, joint_inertia=joint_inertia,
            position_gain=draw.position_gain, velocity_bandwidth=draw.velocity_bandwidth,
            integral_time=draw.integral_time, effort_limit=_effort_limit(asset, None),
        )
    raise ValueError(f"unhandled controller mode {spec.mode!r}")  # pragma: no cover - guarded above


def describe_controller(spec: ControllerSpec, draw: ControllerDraw | None = None) -> dict[str, Any]:
    """JSON-serializable record of the controller a build used."""
    record: dict[str, Any] = {
        "mode": spec.mode,
        "model_free": spec.is_model_free,
        "randomize": bool(spec.randomize),
        "nominal": {
            "knows_payload": bool(spec.nominal.knows_payload),
            "friction_scale": float(spec.nominal.friction_scale),
            "rotor_inertia_scale": float(spec.nominal.rotor_inertia_scale),
            "inertia_scale": float(spec.nominal.inertia_scale),
        },
        "velocity_loop": {
            "position_gain": list(spec.velocity_loop.position_gain),
            "velocity_bandwidth": list(spec.velocity_loop.velocity_bandwidth),
            "integral_time": list(spec.velocity_loop.integral_time),
        },
    }
    if draw is not None:
        record["draw"] = draw.as_dict()
    return record


def controller_columns() -> tuple[str, ...]:
    """Per-row column names :func:`ControllerDraw.as_dict` produces."""
    return tuple(ControllerDraw(1.0, 1.0, 1.0, 1.0, 1.0).as_dict())
