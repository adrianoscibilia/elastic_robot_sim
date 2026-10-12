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

#: Burn-in rounding step (FR_04 Sec 3.1: "rounded up to 0.05 s").
BURN_IN_QUANTUM_S = 0.05


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


def filter_sequence(
    tau_r: np.ndarray, inertia: np.ndarray, stiffness: np.ndarray, damping: np.ndarray, time_step: float,
    *, coupling: str = "diagonal", integrator: str = "trapezoidal",
) -> dict[str, Any]:
    """Run the filter along one sequence; returns ``tau_s``, ``delta``, ``delta_dot`` (each ``[T, n]``).

    ``inertia`` is ``[T, n]`` (diagonals) or ``[T, n, n]``; ``coupling="diagonal"``
    reads only the diagonal of a full matrix, ``"full"`` needs the matrices.
    """
    if coupling not in COUPLINGS:
        raise ValueError(f"coupling must be one of {COUPLINGS}, got {coupling!r}")
    if integrator not in INTEGRATORS:
        raise ValueError(f"integrator must be one of {INTEGRATORS}, got {integrator!r}")
    tau_r = np.asarray(tau_r, dtype=float)
    inertia = np.asarray(inertia, dtype=float)
    stiffness = np.asarray(stiffness, dtype=float).reshape(-1)
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
    delta[0] = tau_r[0] / stiffness
    velocity[0] = 0.0
    if integrator == "trapezoidal":
        acceleration = np.zeros(n)
        for k in range(steps - 1):
            system = mass(k + 1) + diag(0.5 * h * damping + 0.25 * h * h * stiffness)
            predicted_v = velocity[k] + 0.5 * h * acceleration
            predicted_d = delta[k] + h * velocity[k] + 0.25 * h * h * acceleration
            new = solve(system, tau_r[k + 1] - damping * predicted_v - stiffness * predicted_d)
            velocity[k + 1] = predicted_v + 0.5 * h * new
            delta[k + 1] = predicted_d + 0.25 * h * h * new
            acceleration = new
    else:
        for k in range(steps - 1):
            m = mass(k + 1)
            velocity[k + 1] = solve(m + diag(h * damping + h * h * stiffness),
                                    apply(m, velocity[k]) + h * (tau_r[k + 1] - stiffness * delta[k]))
            delta[k + 1] = delta[k] + h * velocity[k + 1]
    return {"tau_s": stiffness * delta + damping * velocity, "delta": delta, "delta_dot": velocity}
