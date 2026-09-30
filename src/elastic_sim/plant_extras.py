"""Plant effects a rigid Lagrangian cannot express (round 5).

``R5_00`` Sec 5.2 (as corrected by its Amendment 1) is about what the residual
term ``r_phi(q, dq)`` of the residual model class has to learn.  In a round-4
dataset the answer is *nothing*: the link side of the simulated plant is purely
Lagrangian, so ``ft = tau_s = M(q) qdd + c + g`` holds exactly and the only
non-Lagrangian part of the target is an artefact of the consumer's own
Savitzky-Golay differentiation, which depends on time rather than on
``(q, dq)`` and which no residual can represent.

This module adds the three effects a real series-elastic joint has and the
Lagrangian does not (``R5_00`` Sec 5.4 item 4).  Each is off by default, so a
config that does not ask for them produces a round-4-identical plant:

* **link-side friction** -- viscous and Coulomb friction on the *output* of
  the transmission.  It enters the target directly
  (``tau_s = M qdd + c + g + f_link(dq)``), which is precisely the term the
  residual class exists for and which the elastic class must *not* absorb;
* **stiffness nonlinearity** -- the piecewise-linear ``K1/K2/K3`` spring
  characteristic of a harmonic drive datasheet, replacing the single ``K``;
* **torque ripple** -- a motor-angle-periodic disturbance on the applied
  torque, so the torque the plant receives is no longer the torque that was
  recorded as commanded.

All three are applied by the runners as generalized forces, not by editing the
simulator's model, so MuJoCo and Newton receive the same numbers and a rollout
stays comparable across backends.

Round 6 (`R6_00` Sec 5) adds a fourth, and changes how two are specified:

* **transmission error** -- the harmonic drive's kinematic error, measured on
  the iiwa as a motor-link deflection ripple twice per wave-generator turn
  ([S-1]).  It lives *inside* the spring: the deflection becomes
  ``(theta + e(theta)) - q`` with ``e = A sin(n theta + phi)``, so it reaches
  the target through the spring, not as a motor torque;
* the nonlinear spring's knees can be given **in torque**, as a datasheet
  does, and are converted per robot with that robot's sampled stiffness;
* the torque ripple's order can differ per joint (FMRR's two screw leads).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .identification import COULOMB_EPSILON, FrictionModel


@dataclass(frozen=True)
class StiffnessNonlinearity:
    """Piecewise-linear spring, in the shape a harmonic-drive datasheet gives.

    ``breakpoints`` are deflection magnitudes [rad] in ascending order and
    ``factors`` multiply the nominal stiffness ``k`` in each region, so
    ``len(factors) == len(breakpoints) + 1``.  The datasheet convention is a
    soft first region and a stiffer second and third
    (``factors: [0.6, 1.0, 1.4]`` around the nominal ``K2``), which is why
    ``k`` itself is not redefined: the sampled stiffness stays the reference
    and this only bends the curve around it.

    Either field may be one shared row or one row per joint (a tuple of
    tuples): the UR10's joints are three gearbox sizes with three different
    curves (`R6_00` Sec 5.2).  :meth:`from_torque_knees` builds the per-joint
    form from datasheet torques.

    The curve is continuous by construction (each region starts where the
    previous ended), so the spring has no force step and the integrator sees
    nothing discontinuous -- only a slope change.
    """

    breakpoints: tuple = ()
    factors: tuple = (1.0,)

    def __post_init__(self) -> None:
        breakpoints = _rows(self.breakpoints)
        factors = _rows(self.factors)
        per_joint = isinstance(breakpoints[0], tuple) if breakpoints else False
        per_joint_factors = isinstance(factors[0], tuple)
        break_rows = breakpoints if per_joint else (breakpoints,)
        factor_rows = factors if per_joint_factors else (factors,)
        if per_joint and per_joint_factors and len(break_rows) != len(factor_rows):
            raise ValueError(
                "plant_extras.stiffness_nonlinearity: per-joint breakpoints and factors need the same "
                f"number of joints (got {len(break_rows)} and {len(factor_rows)})"
            )
        widths = {len(row) for row in break_rows} | {len(row) - 1 for row in factor_rows}
        if len(widths) != 1:
            raise ValueError(
                "plant_extras.stiffness_nonlinearity needs one more factor than breakpoints "
                f"(got breakpoint rows of {sorted(len(r) for r in break_rows)} and factor rows of "
                f"{sorted(len(r) for r in factor_rows)})"
            )
        for row in break_rows:
            if any(value <= 0.0 for value in row):
                raise ValueError("plant_extras.stiffness_nonlinearity.breakpoints must be positive")
            if any(b <= a for a, b in zip(row, row[1:])):
                raise ValueError("plant_extras.stiffness_nonlinearity.breakpoints must be strictly increasing")
        for row in factor_rows:
            if any(value <= 0.0 for value in row):
                raise ValueError("plant_extras.stiffness_nonlinearity.factors must be positive")
        object.__setattr__(self, "breakpoints", breakpoints)
        object.__setattr__(self, "factors", factors)

    @classmethod
    def from_torque_knees(
        cls, factors: np.ndarray, knees: np.ndarray, stiffness: np.ndarray,
    ) -> "StiffnessNonlinearity":
        """Per-joint curve whose knees sit at datasheet torques ``T1, T2, ...``.

        ``k`` stays the middle-region (``K2``) reference, so region ``r`` has
        slope ``factors[r] * k`` and the deflection knees follow from the
        torque ones: ``theta_1 = T_1 / (f_1 k)``, ``theta_{r+1} = theta_r +
        (T_{r+1} - T_r) / (f_{r+1} k)`` (`R6_00` Sec 5.2).  This keeps each
        knee at its physical torque whatever stiffness a robot drew, which a
        fixed deflection breakpoint would not.
        """
        stiffness = np.asarray(stiffness, dtype=float).reshape(-1)
        n = len(stiffness)
        factors = np.broadcast_to(np.atleast_2d(np.asarray(factors, dtype=float)), (n, np.shape(np.atleast_2d(factors))[-1]))
        knees = np.broadcast_to(np.atleast_2d(np.asarray(knees, dtype=float)), (n, np.shape(np.atleast_2d(knees))[-1]))
        if factors.shape[1] != knees.shape[1] + 1:
            raise ValueError("breakpoints_torque needs one fewer entry than factors")
        if np.any(np.diff(knees, axis=1) <= 0.0) or np.any(knees <= 0.0):
            raise ValueError("breakpoints_torque must be positive and strictly increasing")
        deflection = np.empty_like(knees)
        previous_torque = np.zeros(n)
        previous_deflection = np.zeros(n)
        for index in range(knees.shape[1]):
            deflection[:, index] = previous_deflection + (knees[:, index] - previous_torque) / (
                factors[:, index] * stiffness
            )
            previous_torque, previous_deflection = knees[:, index], deflection[:, index]
        return cls(breakpoints=tuple(tuple(row) for row in deflection),
                   factors=tuple(tuple(row) for row in factors))

    @property
    def is_linear(self) -> bool:
        factors = np.asarray(self.factors, dtype=float)
        return not self.breakpoints or bool(np.all(factors == 1.0))

    @property
    def max_factor(self) -> float:
        """The stiffest region's factor, which sets the fastest transmission mode."""
        return float(np.max(np.asarray(self.factors, dtype=float)))

    def _tables(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        breakpoints = np.asarray(self.breakpoints, dtype=float)
        factors = np.asarray(self.factors, dtype=float)
        breakpoints = np.broadcast_to(np.atleast_2d(breakpoints), (n, breakpoints.shape[-1]))
        factors = np.broadcast_to(np.atleast_2d(factors), (n, factors.shape[-1]))
        return breakpoints, factors

    def torque(self, deflection: np.ndarray, stiffness: np.ndarray) -> np.ndarray:
        """Spring torque magnitude-curve ``K_nl(e)``, signed like ``e``.

        Returns the restoring torque as a function of the deflection with the
        same sign convention as the linear ``stiffness * deflection`` it
        replaces, so a caller can take the difference and apply it as a
        correction on top of a simulator's own linear joint stiffness.
        """
        deflection = np.asarray(deflection, dtype=float)
        stiffness = np.broadcast_to(np.asarray(stiffness, dtype=float), deflection.shape)
        magnitude = np.abs(deflection)
        if not self.breakpoints or not isinstance(self.breakpoints[0], tuple):
            # One shared curve: the historical arithmetic, operation for
            # operation, so a round-5 dataset rebuilds byte-identically.
            total = np.zeros_like(magnitude)
            lower = np.zeros_like(magnitude)
            factors = self.factors if not isinstance(self.factors[0], tuple) else None
            if factors is not None:
                for index, factor in enumerate(factors):
                    upper = (
                        np.full_like(magnitude, self.breakpoints[index])
                        if index < len(self.breakpoints) else
                        np.full_like(magnitude, np.inf)
                    )
                    width = np.clip(magnitude, lower, upper) - lower
                    total = total + stiffness * factor * width
                    lower = upper
                return np.sign(deflection) * total
        n = deflection.shape[-1]
        breakpoints, factors = self._tables(n)
        total = np.zeros_like(magnitude)
        lower = np.zeros_like(magnitude)
        for index in range(factors.shape[1]):
            upper = (np.broadcast_to(breakpoints[:, index], magnitude.shape)
                     if index < breakpoints.shape[1] else np.full_like(magnitude, np.inf))
            width = np.clip(magnitude, lower, upper) - lower
            total = total + stiffness * factors[:, index] * width
            lower = upper
        return np.sign(deflection) * total


def _rows(values: Any) -> tuple:
    """A flat tuple of floats, or a tuple of per-joint float tuples."""
    values = tuple(values)
    if values and isinstance(values[0], (tuple, list, np.ndarray)):
        return tuple(tuple(float(v) for v in row) for row in values)
    return tuple(float(v) for v in values)


@dataclass(frozen=True)
class TransmissionError:
    """Kinematic error of a harmonic drive, inside the spring (`R6_00` Sec 5.3).

    ``e(theta) = amplitude * sin(order * theta + phase)`` per joint, in the
    link coordinate's units, with ``order`` in cycles per radian of the
    *joint* angle (twice the gear ratio: two waves per wave-generator turn).
    The spring then transmits ``K(theta + e(theta) - q)``: what Chawda &
    Niemeyer measured on the iiwa's base joint as a motor-link deflection
    ripple ([S-1]).  It is geometry, not a force, which is why it acts through
    the stiffness and reaches the target, and why the motor feels the
    reaction ``-tau_s e'(theta)``.

    ``phase`` is drawn per bag, like the torque ripple's, so a network cannot
    learn one fixed map of it.
    """

    amplitude: tuple[float, ...] = (0.0,)
    order: tuple[float, ...] = (1.0,)
    phase: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        amplitude = tuple(float(v) for v in np.atleast_1d(self.amplitude))
        order = tuple(float(v) for v in np.atleast_1d(self.order))
        if any(v < 0.0 for v in amplitude):
            raise ValueError("plant_extras.transmission_error.amplitude must be non-negative")
        if any(v <= 0.0 for v in order):
            raise ValueError("plant_extras.transmission_error.order must be positive")
        object.__setattr__(self, "amplitude", amplitude)
        object.__setattr__(self, "order", order)
        object.__setattr__(self, "phase", tuple(float(v) for v in self.phase))

    @property
    def is_empty(self) -> bool:
        return all(v == 0.0 for v in self.amplitude)

    def with_phase(self, rng: np.random.Generator, n_dof: int) -> "TransmissionError":
        if self.is_empty:
            return self
        from dataclasses import replace

        return replace(self, phase=tuple(rng.uniform(0.0, 2.0 * np.pi, size=n_dof)))

    def _terms(self, motor_position: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        motor_position = np.asarray(motor_position, dtype=float)
        amplitude = np.broadcast_to(np.asarray(self.amplitude, dtype=float), motor_position.shape[-1:])
        order = np.broadcast_to(np.asarray(self.order, dtype=float), motor_position.shape[-1:])
        phase = np.asarray(self.phase, dtype=float) if self.phase else np.zeros(motor_position.shape[-1])
        return amplitude, order, order * motor_position + phase

    def error(self, motor_position: np.ndarray) -> np.ndarray:
        """``e(theta)``."""
        amplitude, _, angle = self._terms(motor_position)
        return amplitude * np.sin(angle)

    def slope(self, motor_position: np.ndarray) -> np.ndarray:
        """``de/dtheta``."""
        amplitude, order, angle = self._terms(motor_position)
        return amplitude * order * np.cos(angle)

    def max_slope(self) -> np.ndarray:
        """``A n``: the largest ``|de/dtheta|``, for the explicit-term bound."""
        return np.asarray(self.amplitude, dtype=float) * np.asarray(self.order, dtype=float)


@dataclass(frozen=True)
class TorqueRipple:
    """Motor-angle-periodic disturbance on the applied torque.

    ``tau_ripple = amplitude * limit * sin(order * theta + phase)`` per joint,
    with ``amplitude`` a fraction of that joint's rated torque and ``order``
    in cycles per radian of motor travel.  Cogging and current-loop ripple are
    what this stands in for.

    It is what makes the recorded ``tau_cmd`` an *imperfect* measure of the
    torque the plant received, which is the honest form of the relation
    ``R5_00`` Sec 5.3 assumes (``tau_cmd = ft + J thetadd + f``): with ripple
    that identity holds only up to a term periodic in the motor angle, and the
    residual is what can represent it.  ``phase`` is drawn per bag so a
    network cannot learn one fixed ripple map.
    """

    amplitude: float = 0.0
    #: Cycles per unit of motor travel; one value, or one per joint (FMRR's
    #: axes have two different screw leads, `R6_00` Sec 5.3).
    order: float | tuple[float, ...] = 24.0
    phase: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if float(self.amplitude) < 0.0:
            raise ValueError("plant_extras.torque_ripple.amplitude must be non-negative")
        order = self.order
        if isinstance(order, (tuple, list, np.ndarray)):
            order = tuple(float(v) for v in order)
            if len(order) == 1:
                order = order[0]
        else:
            order = float(order)
        if np.any(np.asarray(order, dtype=float) <= 0.0):
            raise ValueError("plant_extras.torque_ripple.order must be positive")
        object.__setattr__(self, "order", order)
        object.__setattr__(self, "phase", tuple(float(v) for v in self.phase))

    @property
    def is_empty(self) -> bool:
        return float(self.amplitude) == 0.0

    def with_phase(self, rng: np.random.Generator, n_dof: int) -> "TorqueRipple":
        """A copy whose per-joint phase is drawn from ``rng``."""
        if self.is_empty:
            return self
        from dataclasses import replace

        return replace(self, phase=tuple(rng.uniform(0.0, 2.0 * np.pi, size=n_dof)))

    def torque(self, motor_position: np.ndarray, effort_limit: np.ndarray) -> np.ndarray:
        if self.is_empty:
            return np.zeros_like(np.asarray(motor_position, dtype=float))
        motor_position = np.asarray(motor_position, dtype=float)
        limit = np.asarray(effort_limit, dtype=float)
        # An infinite rated torque (a URDF with no <limit effort=...>) has no
        # scale to hang a relative ripple on; those joints get none.
        limit = np.where(np.isfinite(limit), limit, 0.0)
        phase = np.asarray(self.phase, dtype=float) if self.phase else np.zeros_like(motor_position)
        order = np.asarray(self.order, dtype=float) if isinstance(self.order, tuple) else self.order
        return self.amplitude * limit * np.sin(order * motor_position + phase)


@dataclass(frozen=True)
class PlantExtras:
    """The non-Lagrangian plant effects one bag is simulated with."""

    link_friction: FrictionModel | None = None
    stiffness_nonlinearity: StiffnessNonlinearity | None = None
    torque_ripple: TorqueRipple | None = None
    epsilon: float = COULOMB_EPSILON
    transmission_error: TransmissionError | None = None

    @property
    def is_empty(self) -> bool:
        """True when this plant is round 4's plant exactly."""
        return (
            self.link_friction is None
            and (self.stiffness_nonlinearity is None or self.stiffness_nonlinearity.is_linear)
            and (self.torque_ripple is None or self.torque_ripple.is_empty)
            and not self.has_transmission_error
        )

    @property
    def has_transmission_error(self) -> bool:
        return self.transmission_error is not None and not self.transmission_error.is_empty

    @property
    def has_nonlinear_spring(self) -> bool:
        return self.stiffness_nonlinearity is not None and not self.stiffness_nonlinearity.is_linear

    def for_rigid(self) -> "PlantExtras":
        """What the rigid (K -> infinity) tier keeps: everything but the spring.

        Link friction and motor torque ripple are properties of the joint's
        two ends, which the rigid limit still has; the nonlinear spring and the
        transmission error live in the compliance it removes (`R6_00` Sec 2.3).
        """
        from dataclasses import replace

        return replace(self, stiffness_nonlinearity=None, transmission_error=None)

    def link_friction_torque(self, link_velocity: np.ndarray) -> np.ndarray:
        """Friction torque *resisting* the link-side motion.

        Applied by the runners to both the motor and the elastic degree of
        freedom: the link's rotation relative to its parent is
        ``theta + e``, so a torque on the link about the joint axis has unit
        partial derivative with respect to both coordinates.  Applying it to
        the elastic dof alone would instead make it a torque between the rotor
        and the link, i.e. a second transmission damper, which is not what
        output-bearing friction is.
        """
        if self.link_friction is None:
            return np.zeros_like(np.asarray(link_velocity, dtype=float))
        return -self.link_friction.torque(link_velocity, epsilon=self.epsilon)

    def link_friction_slope(self, link_velocity: np.ndarray) -> np.ndarray:
        """``d(friction)/d(link velocity)`` >= 0, for the implicit step (`R6_00` Sec 3)."""
        if self.link_friction is None:
            return np.zeros_like(np.asarray(link_velocity, dtype=float))
        return self.link_friction.slope(link_velocity, epsilon=self.epsilon)

    def spring_correction(
        self, deflection: np.ndarray, stiffness: np.ndarray,
    ) -> np.ndarray:
        """Torque to add to a *linear* spring to realize the nonlinear one.

        Returns ``-(K_nl(e) - k e)`` in the runners' sign convention, where
        ``e`` is the elastic joint's relative coordinate and the simulator
        already supplies ``-k e`` as its own joint stiffness.  Without a
        transmission error; :meth:`spring_forces` is the general form.
        """
        if self.stiffness_nonlinearity is None or self.stiffness_nonlinearity.is_linear:
            return np.zeros_like(np.asarray(deflection, dtype=float))
        deflection = np.asarray(deflection, dtype=float)
        stiffness = np.asarray(stiffness, dtype=float)
        nonlinear = self.stiffness_nonlinearity.torque(deflection, stiffness)
        return -(nonlinear - stiffness * deflection)

    def spring_forces(
        self, deflection: np.ndarray, deflection_rate: np.ndarray,
        motor_position: np.ndarray, motor_velocity: np.ndarray,
        stiffness: np.ndarray, damping: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """``(on_motor, on_elastic)`` generalized forces to add to the linear spring.

        The simulator supplies ``-k e - d edot`` on the elastic coordinate
        ``e = q - theta`` itself.  With a transmission error the spring works
        on ``delta = e - err(theta)`` instead, and transmits
        ``tau_s = -(K_nl(delta) + d delta_dot)`` to the link.  In the
        ``(theta, e)`` coordinates the link's ``tau_s`` loads both, and the
        motor's reaction ``-tau_s (1 + err')`` loads ``theta``, so

        * on the elastic coordinate: ``tau_s - (-k e - d edot)``;
        * on the motor coordinate: ``-tau_s err'(theta)``.

        With no transmission error the motor term is zero and the elastic one
        is :meth:`spring_correction`, computed the same way.
        """
        if not self.has_transmission_error:
            correction = self.spring_correction(deflection, stiffness)
            return np.zeros_like(correction), correction
        deflection = np.asarray(deflection, dtype=float)
        stiffness = np.asarray(stiffness, dtype=float)
        error = self.transmission_error.error(motor_position)
        slope = self.transmission_error.slope(motor_position)
        tau_s = self.spring_torque(deflection, deflection_rate, stiffness, damping,
                                   motor_position=motor_position, motor_velocity=motor_velocity)
        delta = deflection - error
        if self.has_nonlinear_spring:
            elastic = -(self.stiffness_nonlinearity.torque(delta, stiffness) - stiffness * deflection)
        else:
            elastic = stiffness * error
        elastic = elastic + np.asarray(damping, dtype=float) * slope * np.asarray(motor_velocity, dtype=float)
        return -tau_s * slope, elastic

    def spring_torque(
        self, deflection: np.ndarray, deflection_rate: np.ndarray,
        stiffness: np.ndarray, damping: np.ndarray,
        *, motor_position: np.ndarray | None = None, motor_velocity: np.ndarray | None = None,
    ) -> np.ndarray:
        """The full link-side torque ``tau_s``, nonlinear spring included.

        This is the recorded label: with a nonlinear spring the linear
        ``-(k e + d edot)`` is no longer what the joint transmits, so the
        runners must record this instead.  With a transmission error the
        spring acts on ``e - err(theta)`` and needs the motor state.
        """
        deflection = np.asarray(deflection, dtype=float)
        deflection_rate = np.asarray(deflection_rate, dtype=float)
        stiffness = np.asarray(stiffness, dtype=float)
        damping = np.asarray(damping, dtype=float)
        if self.has_transmission_error:
            if motor_position is None or motor_velocity is None:
                raise ValueError("a transmission error needs the motor state to evaluate the spring")
            deflection = deflection - self.transmission_error.error(motor_position)
            deflection_rate = deflection_rate - self.transmission_error.slope(motor_position) * np.asarray(
                motor_velocity, dtype=float
            )
        if self.stiffness_nonlinearity is None or self.stiffness_nonlinearity.is_linear:
            elastic = stiffness * deflection
        else:
            elastic = self.stiffness_nonlinearity.torque(deflection, stiffness)
        return -(elastic + damping * deflection_rate)

    def ripple_torque(self, motor_position: np.ndarray, effort_limit: np.ndarray) -> np.ndarray:
        if self.torque_ripple is None:
            return np.zeros_like(np.asarray(motor_position, dtype=float))
        return self.torque_ripple.torque(motor_position, effort_limit)

    def describe(self) -> dict[str, Any]:
        """JSON-serializable record of what this plant carries."""
        record = {
            "link_friction": None if self.link_friction is None else {
                "viscous": self.link_friction.viscous.tolist(),
                "coulomb": self.link_friction.coulomb.tolist(),
            },
            "stiffness_nonlinearity": None if self.stiffness_nonlinearity is None else {
                "breakpoints": _jsonable(self.stiffness_nonlinearity.breakpoints),
                "factors": _jsonable(self.stiffness_nonlinearity.factors),
            },
            "torque_ripple": None if self.torque_ripple is None else {
                "amplitude": float(self.torque_ripple.amplitude),
                "order": _jsonable(self.torque_ripple.order),
                "phase": list(self.torque_ripple.phase),
            },
        }
        if self.transmission_error is not None:
            record["transmission_error"] = {
                "amplitude": list(self.transmission_error.amplitude),
                "order": list(self.transmission_error.order),
                "phase": list(self.transmission_error.phase),
            }
        return record


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    return float(value)


#: A plant with nothing added: what every round-4 config gets.
NO_EXTRAS = PlantExtras()
