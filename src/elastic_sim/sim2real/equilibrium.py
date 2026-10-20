"""Static equilibrium of an elastic chain at rest (S2R_01 T-3.1, §4.2).

With the motor held at ``theta`` and everything at rest, each transmission
carries the gravity load of the chain beyond it:

    tau_s(delta) = g_e(theta, delta)

``tau_s`` the spring law (``PlantExtras.spring_torque``: the linear ``K delta``
plus the nonlinearity and the transmission error, if any) and ``g_e`` the
gravity force on the elastic coordinates, MuJoCo's ``qfrc_bias`` at zero
velocity, evaluated on the built model itself. Coordinates as in the
runners: the elastic joint's ``e = q_link - theta = -delta``.

:func:`static_deflection` solves it by Newton with a finite-difference
Jacobian (seven columns; one ``mj_forward`` each). It is the
``InitialState.deflection`` callable R1 hands to
:func:`elastic_sim.torque_runners.run_mujoco_elastic_torque`.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np

#: Done-when of T-3.1: the equilibrium residual, N m.
RESIDUAL_TOLERANCE = 1.0e-9


def equilibrium_residual(mujoco: Any, model: Any, data: Any, indices: Mapping[str, np.ndarray],
                         spring: Callable[[np.ndarray, np.ndarray], np.ndarray], deflection: np.ndarray
                         ) -> np.ndarray:
    """``tau_s(delta) - g_e`` on the elastic coordinates, at rest, the motor
    where ``data`` has it. Leaves ``data`` at that state (``mj_forward``)."""
    data.qvel[:] = 0.0
    data.qpos[indices["elastic_qpos"]] = -np.asarray(deflection, dtype=float)
    mujoco.mj_forward(model, data)
    theta = np.array(data.qpos[indices["motor_qpos"]])
    return spring(np.asarray(deflection, dtype=float), theta) - np.array(data.qfrc_bias[indices["elastic_dof"]])


def static_deflection(mujoco: Any, model: Any, data: Any, indices: Mapping[str, np.ndarray],
                      spring: Callable[[np.ndarray, np.ndarray], np.ndarray], *, stiffness: np.ndarray,
                      tolerance: float = 0.01 * RESIDUAL_TOLERANCE, max_iterations: int = 30,
                      relative_step: float = 1.0e-6) -> tuple[np.ndarray, dict[str, Any]]:
    """Newton on :func:`equilibrium_residual` from ``delta = g_e(theta, 0) / K``.

    ``spring(delta, theta)`` is the transmitted torque at rest. Returns
    ``delta`` and a record (iterations, the residual's max, N m). Raises if
    the residual does not reach :data:`RESIDUAL_TOLERANCE`."""
    stiffness = np.asarray(stiffness, dtype=float)
    n = len(indices["elastic_qpos"])
    delta = np.zeros(n)
    equilibrium_residual(mujoco, model, data, indices, spring, delta)
    delta = np.array(data.qfrc_bias[indices["elastic_dof"]]) / stiffness
    history = []
    for iteration in range(1, max_iterations + 1):
        residual = equilibrium_residual(mujoco, model, data, indices, spring, delta)
        history.append(float(np.max(np.abs(residual))))
        if history[-1] <= tolerance:
            break
        jacobian = np.empty((n, n))
        for j in range(n):
            h = relative_step * max(abs(delta[j]), 1.0e-6)
            step = delta.copy()
            step[j] += h
            plus = equilibrium_residual(mujoco, model, data, indices, spring, step)
            step[j] -= 2.0 * h
            minus = equilibrium_residual(mujoco, model, data, indices, spring, step)
            jacobian[:, j] = (plus - minus) / (2.0 * h)
        delta = delta - np.linalg.solve(jacobian, residual)
    residual = equilibrium_residual(mujoco, model, data, indices, spring, delta)
    worst = float(np.max(np.abs(residual)))
    if worst > RESIDUAL_TOLERANCE:
        raise RuntimeError(f"static equilibrium: residual {worst:.3e} N m after {len(history)} Newton steps")
    return delta, {"iterations": len(history), "residual_max_nm": worst, "history": history,
                   "spring_torque": spring(delta, np.array(data.qpos[indices["motor_qpos"]])).tolist()}


def spring_at_rest(extras: Any, stiffness: np.ndarray, damping: np.ndarray) -> Callable[[np.ndarray, np.ndarray],
                                                                                        np.ndarray]:
    """``(delta, theta) -> tau_s`` at zero rates from a ``PlantExtras`` (the
    runners' spring law, so the equilibrium is the one the rollout starts in)."""
    def spring(delta: np.ndarray, theta: np.ndarray) -> np.ndarray:
        zeros = np.zeros_like(delta)
        if extras.has_transmission_error:
            return extras.spring_torque(-delta, zeros, stiffness, damping, motor_position=theta,
                                        motor_velocity=zeros)
        return extras.spring_torque(-delta, zeros, stiffness, damping)

    return spring


__all__ = ["RESIDUAL_TOLERANCE", "equilibrium_residual", "spring_at_rest", "static_deflection"]
