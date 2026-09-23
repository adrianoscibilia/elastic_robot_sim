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

    The curve is continuous by construction (each region starts where the
    previous ended), so the spring has no force step and the integrator sees
    nothing discontinuous -- only a slope change.
    """

    breakpoints: tuple[float, ...] = ()
    factors: tuple[float, ...] = (1.0,)

    def __post_init__(self) -> None:
        breakpoints = tuple(float(v) for v in self.breakpoints)
        factors = tuple(float(v) for v in self.factors)
        if len(factors) != len(breakpoints) + 1:
            raise ValueError(
                "plant_extras.stiffness_nonlinearity needs one more factor than breakpoints "
                f"(got {len(breakpoints)} breakpoints and {len(factors)} factors)"
            )
        if any(value <= 0.0 for value in breakpoints):
            raise ValueError("plant_extras.stiffness_nonlinearity.breakpoints must be positive")
        if any(b <= a for a, b in zip(breakpoints, breakpoints[1:])):
            raise ValueError("plant_extras.stiffness_nonlinearity.breakpoints must be strictly increasing")
        if any(value <= 0.0 for value in factors):
            raise ValueError("plant_extras.stiffness_nonlinearity.factors must be positive")
        object.__setattr__(self, "breakpoints", breakpoints)
        object.__setattr__(self, "factors", factors)

    @property
    def is_linear(self) -> bool:
        return not self.breakpoints or all(value == 1.0 for value in self.factors)

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
        total = np.zeros_like(magnitude)
        lower = np.zeros_like(magnitude)
        for index, factor in enumerate(self.factors):
            upper = (
                np.full_like(magnitude, self.breakpoints[index])
                if index < len(self.breakpoints) else
                np.full_like(magnitude, np.inf)
            )
            width = np.clip(magnitude, lower, upper) - lower
            total = total + stiffness * factor * width
            lower = upper
        return np.sign(deflection) * total


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
    order: float = 24.0
    phase: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if float(self.amplitude) < 0.0:
            raise ValueError("plant_extras.torque_ripple.amplitude must be non-negative")
        if float(self.order) <= 0.0:
            raise ValueError("plant_extras.torque_ripple.order must be positive")
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
        return self.amplitude * limit * np.sin(self.order * motor_position + phase)


@dataclass(frozen=True)
class PlantExtras:
    """The non-Lagrangian plant effects one bag is simulated with."""

    link_friction: FrictionModel | None = None
    stiffness_nonlinearity: StiffnessNonlinearity | None = None
    torque_ripple: TorqueRipple | None = None
    epsilon: float = COULOMB_EPSILON

    @property
    def is_empty(self) -> bool:
        """True when this plant is round 4's plant exactly."""
        return (
            self.link_friction is None
            and (self.stiffness_nonlinearity is None or self.stiffness_nonlinearity.is_linear)
            and (self.torque_ripple is None or self.torque_ripple.is_empty)
        )

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

    def spring_correction(
        self, deflection: np.ndarray, stiffness: np.ndarray,
    ) -> np.ndarray:
        """Torque to add to a *linear* spring to realize the nonlinear one.

        Returns ``-(K_nl(e) - k e)`` in the runners' sign convention, where
        ``e`` is the elastic joint's relative coordinate and the simulator
        already supplies ``-k e`` as its own joint stiffness.
        """
        if self.stiffness_nonlinearity is None or self.stiffness_nonlinearity.is_linear:
            return np.zeros_like(np.asarray(deflection, dtype=float))
        deflection = np.asarray(deflection, dtype=float)
        stiffness = np.asarray(stiffness, dtype=float)
        nonlinear = self.stiffness_nonlinearity.torque(deflection, stiffness)
        return -(nonlinear - stiffness * deflection)

    def spring_torque(
        self, deflection: np.ndarray, deflection_rate: np.ndarray,
        stiffness: np.ndarray, damping: np.ndarray,
    ) -> np.ndarray:
        """The full link-side torque ``tau_s``, nonlinear spring included.

        This is the recorded label: with a nonlinear spring the linear
        ``-(k e + d edot)`` is no longer what the joint transmits, so the
        runners must record this instead.
        """
        deflection = np.asarray(deflection, dtype=float)
        deflection_rate = np.asarray(deflection_rate, dtype=float)
        stiffness = np.asarray(stiffness, dtype=float)
        damping = np.asarray(damping, dtype=float)
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
        return {
            "link_friction": None if self.link_friction is None else {
                "viscous": self.link_friction.viscous.tolist(),
                "coulomb": self.link_friction.coulomb.tolist(),
            },
            "stiffness_nonlinearity": None if self.stiffness_nonlinearity is None else {
                "breakpoints": list(self.stiffness_nonlinearity.breakpoints),
                "factors": list(self.stiffness_nonlinearity.factors),
            },
            "torque_ripple": None if self.torque_ripple is None else {
                "amplitude": float(self.torque_ripple.amplitude),
                "order": float(self.torque_ripple.order),
                "phase": list(self.torque_ripple.phase),
            },
        }


#: A plant with nothing added: what every round-4 config gets.
NO_EXTRAS = PlantExtras()
