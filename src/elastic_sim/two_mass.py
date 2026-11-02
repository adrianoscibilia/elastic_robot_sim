"""The two-mass transmission filter of FR_04 Sec 0: reference implementation.

Pure NumPy and the standard library, so ``dynamic_model_nn`` can import this
file directly (its FR_04 tests cross-check its PyTorch filter against it) and
the producer's ``rigid_elastic_*`` baselines (FR_04 S-4) use the same code.

The link side, with ``q_link = theta - delta`` and linearised in ``delta``::

    M(theta) delta'' + D delta' + K delta = tau_r(theta, theta', theta'')
    tau_s = K delta + D delta'

``tau_r`` is the rigid link torque at the motor states, ``M`` the link inertia
(its diagonal per joint for ``coupling="diagonal"``, the full matrix for
``"full"``), ``K`` and ``D`` the transmission stiffness and damping.

Two discretisations, both unconditionally stable for positive ``K`` and ``D``:

* ``"trapezoidal"`` (the default): Newmark's average acceleration (gamma = 1/2,
  beta = 1/4), the trapezoidal rule on ``(delta, delta')``; second order, no
  numerical damping.  With ``a_k = delta''_k``::

      (M_{k+1} + h/2 D + h^2/4 K) a_{k+1} = tau_r,{k+1} - D (v_k + h/2 a_k) - K (delta_k + h v_k + h^2/4 a_k)
      v_{k+1}     = v_k + h/2 (a_k + a_{k+1})
      delta_{k+1} = delta_k + h v_k + h^2/4 (a_k + a_{k+1})

* ``"backward_euler"``: FR_04 Sec 3.3's equations, first order::

      (M_{k+1} + h D + h^2 K) v_{k+1} = M_{k+1} v_k + h (tau_r,{k+1} - K delta_k)
      delta_{k+1} = delta_k + h v_{k+1}

  Its phase error and numerical damping are ~ omega h / 2: at the data's
  sample steps that is 4-30 % on a chirp through the link modes, an order
  above FR_04 Sec 4's 1e-3 modal acceptance, which the trapezoidal rule meets
  (FR_04 implementation report).

``K`` may vary per sample (`identified_filter/05` D2): the recursion is then
linear in the state with time-varying coefficients, ``K_{k+1}`` in the new step.
``spring="total"`` is that spring, ``K_k delta_k``, which steps by ``dK delta`` when
``K`` moves faster than ``delta``.  ``spring="tangent"`` reads ``K`` as the slope
along the curve instead, ``S' = K delta'``, with ``S`` a state and ``tau_s = S + D delta'``:
no step, and with a constant ``K`` the same sequence as ``"total"``.  Each step takes the
step's mean stiffness, ``S_{k+1} = S_k + (K_k + K_{k+1}) / 2 (delta_{k+1} - delta_k)``
(`identified_filter/06` Sec 1.2): a step and its reversal then cancel exactly, where the new
sample's ``K_{k+1}`` alone ratchets ``S`` up the curve on every oscillation inside a knee's blend.

``tau_s,k = K delta_k + D v_k``.  The initial condition is quasi-static:
``delta_0 = tau_r,0 / K``, ``v_0 = a_0 = 0``, so ``tau_s,0 = tau_r,0``; exact
when the sequence starts at rest.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

COUPLINGS = ("diagonal", "full")
INTEGRATORS = ("trapezoidal", "backward_euler")
SPRINGS = ("total", "tangent")

#: Burn-in rounding step (FR_04 Sec 3.1: "rounded up to 0.05 s").
BURN_IN_QUANTUM_S = 0.05

#: Half-width of the blend across a catalogue knee, as a fraction of its torque (`identified_filter/05` D2).
KNEE_BLEND = 0.1


def burn_in_seconds(zeta_min: float, frequency_min_hz: float, quantum: float = BURN_IN_QUANTUM_S) -> float:
    """``max(3 / (zeta_min 2 pi f_min), 2 / f_min)``, rounded up to ``quantum`` (FR_04 Sec 3.1)."""
    if zeta_min <= 0.0 or frequency_min_hz <= 0.0:
        raise ValueError("zeta_min and frequency_min_hz must be positive")
    raw = max(3.0 / (zeta_min * 2.0 * math.pi * frequency_min_hz), 2.0 / frequency_min_hz)
    # A tolerance so an exact multiple is not pushed up one step by rounding.
    return round(math.ceil(raw / quantum - 1e-9) * quantum, 10)


def damping_from_ratio(stiffness: np.ndarray, zeta: np.ndarray, mean_inertia: np.ndarray) -> np.ndarray:
    """``D = 2 zeta sqrt(K M_bar)`` per joint, ``M_bar`` the mean diagonal link inertia."""
    return 2.0 * np.asarray(zeta, dtype=float) * np.sqrt(
        np.asarray(stiffness, dtype=float) * np.asarray(mean_inertia, dtype=float))


def mean_diagonal(inertia: np.ndarray) -> np.ndarray:
    """Mean over samples of the diagonal link inertia, from ``[T, n]`` diagonals or ``[T, n, n]`` matrices."""
    inertia = np.asarray(inertia, dtype=float)
    diagonal = np.diagonal(inertia, axis1=-2, axis2=-1) if inertia.ndim == 3 else inertia
    return diagonal.mean(axis=0)


def knee_factor(torque: np.ndarray, factors: np.ndarray, knees: np.ndarray, blend: float = KNEE_BLEND) -> np.ndarray:
    """``phi_j(|tau|)``: the catalogue spring's region factor at a torque, ``[T, n]`` (`identified_filter/05` D2).

    ``factors`` is ``[n, r + 1]`` or one shared row, ``knees`` ``[n, r]``.  A smoothstep over ``knee x (1 +- blend)``
    joins the regions; outside those intervals the factor is exactly the catalogue's.
    """
    magnitude = np.abs(np.asarray(torque, dtype=float))
    n = magnitude.shape[-1]
    knees = np.broadcast_to(np.atleast_2d(np.asarray(knees, dtype=float)), (n, np.shape(np.atleast_2d(knees))[-1]))
    factors = np.broadcast_to(np.atleast_2d(np.asarray(factors, dtype=float)),
                              (n, np.shape(np.atleast_2d(factors))[-1]))
    out = np.broadcast_to(factors[:, 0], magnitude.shape).copy()
    for r in range(knees.shape[1]):
        s = np.clip(0.5 + 0.5 * (magnitude - knees[:, r]) / (blend * knees[:, r]), 0.0, 1.0)
        out = out + (factors[:, r + 1] - factors[:, r]) * s * s * (3.0 - 2.0 * s)
    return out


def _curve_tables(stiffness: np.ndarray, factors: np.ndarray, knees: np.ndarray) -> tuple[np.ndarray, ...]:
    """Per joint: region slopes ``[n, r + 1]``, knee torques and knee deflections ``[n, r]``."""
    stiffness = np.asarray(stiffness, dtype=float).reshape(-1)
    n = len(stiffness)
    knees = np.broadcast_to(np.atleast_2d(np.asarray(knees, dtype=float)), (n, np.shape(np.atleast_2d(knees))[-1]))
    factors = np.broadcast_to(np.atleast_2d(np.asarray(factors, dtype=float)),
                              (n, np.shape(np.atleast_2d(factors))[-1]))
    slopes = factors * stiffness[:, None]
    widths = np.diff(np.concatenate([np.zeros((n, 1)), knees], axis=1), axis=1)
    return slopes, knees, np.cumsum(widths / slopes[:, :-1], axis=1)


def catalogue_torque(deflection: np.ndarray, stiffness: np.ndarray, factors: np.ndarray,
                     knees: np.ndarray) -> np.ndarray:
    """``f(delta)``: the catalogue spring with its knees at torques ``knees``, odd in ``delta``, ``[T, n]``.

    The plant's curve (``plant_extras.StiffnessNonlinearity.from_torque_knees``), unblended: region ``r`` has slope
    ``factors[r] K_0``.
    """
    slopes, knee_torque, knee_deflection = _curve_tables(stiffness, factors, knees)
    magnitude = np.abs(np.asarray(deflection, dtype=float))
    lower = np.concatenate([np.zeros((slopes.shape[0], 1)), knee_deflection], axis=1)
    base = np.concatenate([np.zeros((slopes.shape[0], 1)), knee_torque], axis=1)
    region = np.stack([np.searchsorted(lower[j], magnitude[..., j], side="right") - 1
                       for j in range(slopes.shape[0])], axis=-1)
    pick = lambda table: np.take_along_axis(np.broadcast_to(table, magnitude.shape + table.shape[-1:]),  # noqa: E731
                                            region[..., None], axis=-1)[..., 0]
    return np.sign(deflection) * (pick(base) + pick(slopes) * (magnitude - pick(lower)))


def catalogue_deflection(torque: np.ndarray, stiffness: np.ndarray, factors: np.ndarray,
                         knees: np.ndarray) -> np.ndarray:
    """``f^-1(tau)``: the deflection of the catalogue spring at a torque, ``[..., n]``."""
    slopes, knee_torque, knee_deflection = _curve_tables(stiffness, factors, knees)
    magnitude = np.abs(np.asarray(torque, dtype=float))
    lower = np.concatenate([np.zeros((slopes.shape[0], 1)), knee_torque], axis=1)
    base = np.concatenate([np.zeros((slopes.shape[0], 1)), knee_deflection], axis=1)
    region = np.stack([np.searchsorted(lower[j], magnitude[..., j], side="right") - 1
                       for j in range(slopes.shape[0])], axis=-1)
    pick = lambda table: np.take_along_axis(np.broadcast_to(table, magnitude.shape + table.shape[-1:]),  # noqa: E731
                                            region[..., None], axis=-1)[..., 0]
    return np.sign(torque) * (pick(base) + (magnitude - pick(lower)) / pick(slopes))


def filter_sequence(
    tau_r: np.ndarray, inertia: np.ndarray, stiffness: np.ndarray, damping: np.ndarray, time_step: float,
    *, coupling: str = "diagonal", integrator: str = "trapezoidal", spring: str = "total",
    initial_deflection: np.ndarray | None = None,
) -> dict[str, Any]:
    """Run the filter along one sequence; returns ``tau_s``, ``delta``, ``delta_dot`` (each ``[T, n]``).

    ``inertia`` is ``[T, n]`` (diagonals) or ``[T, n, n]``; ``coupling="diagonal"``
    reads only the diagonal of a full matrix, ``"full"`` needs the matrices.
    ``stiffness`` is ``[n]``, or ``[T, n]`` for a time-varying spring.
    ``initial_deflection`` replaces ``tau_r,0 / K_0`` (``spring="tangent"`` only, where
    ``delta`` never enters the dynamics, so ``tau_s`` does not depend on it): the
    catalogue curve's ``f^-1(tau_r,0)``, which keeps ``S = f(delta)`` at the start.
    """
    if coupling not in COUPLINGS:
        raise ValueError(f"coupling must be one of {COUPLINGS}, got {coupling!r}")
    if integrator not in INTEGRATORS:
        raise ValueError(f"integrator must be one of {INTEGRATORS}, got {integrator!r}")
    if spring not in SPRINGS:
        raise ValueError(f"spring must be one of {SPRINGS}, got {spring!r}")
    if spring == "tangent" and integrator != "trapezoidal":
        raise ValueError("spring='tangent' is implemented for the trapezoidal integrator")
    if initial_deflection is not None and spring != "tangent":
        raise ValueError("initial_deflection is for spring='tangent'; the total form starts at tau_r,0 / K_0")
    tau_r = np.asarray(tau_r, dtype=float)
    inertia = np.asarray(inertia, dtype=float)
    stiffness = np.asarray(stiffness, dtype=float)
    varying = stiffness.ndim == 2
    if not varying:
        stiffness = stiffness.reshape(-1)
    k_at = (lambda k: stiffness[k]) if varying else (lambda k: stiffness)
    damping = np.asarray(damping, dtype=float).reshape(-1)
    h = float(time_step)
    steps, n = tau_r.shape
    if np.any(stiffness <= 0.0) or np.any(damping < 0.0) or h <= 0.0:
        raise ValueError("stiffness and time_step must be positive, damping non-negative")
    if coupling == "full" and inertia.ndim != 3:
        raise ValueError("coupling='full' needs [T, n, n] inertia matrices")
    if coupling == "diagonal":
        diagonal = np.diagonal(inertia, axis1=-2, axis2=-1) if inertia.ndim == 3 else inertia
        mass = lambda k: diagonal[k]  # noqa: E731
        solve = lambda m, rhs: rhs / m  # noqa: E731
        apply = lambda m, x: m * x  # noqa: E731
        diag = lambda x: x  # noqa: E731
    else:
        mass = lambda k: inertia[k]  # noqa: E731
        solve = np.linalg.solve
        apply = lambda m, x: m @ x  # noqa: E731
        diag = np.diag

    delta = np.empty((steps, n))
    velocity = np.empty((steps, n))
    delta[0] = tau_r[0] / k_at(0) if initial_deflection is None else np.asarray(initial_deflection, float).reshape(-1)
    velocity[0] = 0.0
    if spring == "tangent":
        acceleration = np.zeros(n)
        force = np.asarray(tau_r[0], dtype=float).copy()
        forces = np.empty((steps, n))
        forces[0] = force
        for k in range(steps - 1):
            k_new = 0.5 * (k_at(k) + k_at(k + 1))
            system = mass(k + 1) + diag(0.5 * h * damping + 0.25 * h * h * k_new)
            predicted_v = velocity[k] + 0.5 * h * acceleration
            predicted_d = delta[k] + h * velocity[k] + 0.25 * h * h * acceleration
            new = solve(system, tau_r[k + 1] - damping * predicted_v
                        - (force + 0.5 * h * k_new * (velocity[k] + predicted_v)))
            velocity[k + 1] = predicted_v + 0.5 * h * new
            delta[k + 1] = predicted_d + 0.25 * h * h * new
            force = force + 0.5 * h * k_new * (velocity[k] + velocity[k + 1])
            forces[k + 1] = force
            acceleration = new
        return {"tau_s": forces + damping * velocity, "delta": delta, "delta_dot": velocity}
    if integrator == "trapezoidal":
        acceleration = np.zeros(n)
        for k in range(steps - 1):
            k_new = k_at(k + 1)
            system = mass(k + 1) + diag(0.5 * h * damping + 0.25 * h * h * k_new)
            predicted_v = velocity[k] + 0.5 * h * acceleration
            predicted_d = delta[k] + h * velocity[k] + 0.25 * h * h * acceleration
            new = solve(system, tau_r[k + 1] - damping * predicted_v - k_new * predicted_d)
            velocity[k + 1] = predicted_v + 0.5 * h * new
            delta[k + 1] = predicted_d + 0.25 * h * h * new
            acceleration = new
    else:
        for k in range(steps - 1):
            m = mass(k + 1)
            k_new = k_at(k + 1)
            velocity[k + 1] = solve(m + diag(h * damping + h * h * k_new),
                                    apply(m, velocity[k]) + h * (tau_r[k + 1] - k_new * delta[k]))
            delta[k + 1] = delta[k] + h * velocity[k + 1]
    return {"tau_s": stiffness * delta + damping * velocity, "delta": delta, "delta_dot": velocity}
