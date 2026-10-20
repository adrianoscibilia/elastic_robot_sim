"""KUKA position-loop emulation for R2 (S2R_01 §4.3, T-3.4).

:class:`KukaPositionEmulation` is the joint position loop with model-based
feedforward that R2 puts between the robot's command and the elastic
plant: :class:`~elastic_sim.torque_runners.SeaMotorController` on a
**nominal** model (the plant theta's inertials, the elastic build's
intermediate bodies, all scaled by ``controller.model_scale``), with
per-joint gains from ``controller.<joint>.bandwidth`` and
``damping_ratio``, run at ``controller.drive_rate`` (the runner's control
decimation) on the motor encoder (collocated). The command delay is in the
reference (:func:`command_reference`), not here.

``pd_gravity`` (the alternative of §4.3, for model comparison) drops the
inertia feedforward: ``tau = (J_r + M_ii(q)) (kp e + kd e') + g(q) + f_m``.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..materialized import MaterializedTrajectory


def command_reference(t: np.ndarray, command: np.ndarray, joint_names: Sequence[str], *, delay_cycles: int,
                      period: float, time_step: float) -> MaterializedTrajectory:
    """The FRI command echo as the loop's setpoint, ``delay_cycles`` periods
    late (S2R_04 A-10: ``commanded_position``; S2R_06 F-8: a fresh command
    every cycle, so no hold model): an interpolating cubic spline (C2) of
    the 1 kHz samples, its ``(q, dq, ddq)`` tabulated on the physics grid."""
    from scipy.interpolate import make_interp_spline

    t = np.asarray(t, dtype=float) - float(t[0])
    command = np.asarray(command, dtype=float)
    spline = make_interp_spline(t, command, k=3, axis=0)
    grid = np.arange(0.0, t[-1] + 0.5 * time_step, time_step)
    shifted = np.clip(grid - delay_cycles * period, 0.0, t[-1])
    table = [spline(shifted, nu) for nu in (0, 1, 2)]
    before = grid < delay_cycles * period
    table[1][before] = 0.0
    table[2][before] = 0.0

    def analytic(time_s: float, _table=table, _step=time_step, _n=len(grid)):
        k = min(int(round(time_s / _step)), _n - 1)
        return _table[0][k].copy(), _table[1][k].copy(), _table[2][k].copy()

    sample = np.clip(t - delay_cycles * period, 0.0, t[-1])
    return MaterializedTrajectory(time=t, position=spline(sample), velocity=spline(sample, 1),
                                  acceleration=spline(sample, 2), joint_names=tuple(joint_names),
                                  metadata={"source": "R2 command reference", "delay_cycles": int(delay_cycles)},
                                  analytic=analytic)


def kuka_position_emulation(plant: Any, trajectory: MaterializedTrajectory, transmission: Any, *,
                            intermediate: Any = None, emulation: str = "kuka_position") -> Any:
    """The R2 controller for ``plant`` (its ``controller`` block: bandwidth,
    damping ratio and model scale per joint)."""
    from ..torque_runners import SeaMotorController, TorqueCommand

    settings = plant.controller
    if settings is None:
        raise ValueError("R2 needs the controller.* parameters (PlantParameters.controller)")
    bandwidth = np.asarray(settings.bandwidth, dtype=float)
    zeta = np.asarray(settings.damping_ratio, dtype=float)
    controller = SeaMotorController(plant.base.asset, trajectory, transmission, natural_frequency=1.0,
                                    damping_ratio=1.0, friction=plant.motor_friction)
    plant.apply_to_pinocchio(controller._pin, controller._model)
    if intermediate is not None:
        intermediate.add_to(controller._model)
    if settings.model_scale != 1.0:
        pin = controller._pin
        for index in range(1, len(controller._model.inertias)):
            inertia = controller._model.inertias[index]
            controller._model.inertias[index] = pin.Inertia(inertia.mass * settings.model_scale, inertia.lever,
                                                            inertia.inertia * settings.model_scale)
    controller.kp = bandwidth ** 2
    controller.kd = 2.0 * zeta * bandwidth
    controller.natural_frequency = float(np.max(bandwidth))
    if emulation == "kuka_position":
        return controller
    if emulation != "pd_gravity":
        raise ValueError(f"R2 emulation {emulation!r}: kuka_position or pd_gravity")
    return _PdGravity(controller, TorqueCommand)


class _PdGravity:
    """``pd_gravity``: the same gains on the joint inertia, gravity and motor
    friction compensated, no inertia feedforward."""

    compensates_friction = True

    def __init__(self, inner: Any, command_type: Any):
        self.inner = inner
        self.command_type = command_type
        self.kp, self.kd = inner.kp, inner.kd
        self.effort_limit = inner.effort_limit

    def stability_margin(self, time_step: float) -> float:
        return float(np.max(self.kd) * float(time_step))

    def __call__(self, t: float, q_motor: np.ndarray, dq_motor: np.ndarray, tau_spring: np.ndarray):
        inner = self.inner
        q_ref, dq_ref, _ = inner.trajectory(float(t))
        q_motor = np.asarray(q_motor, dtype=float)
        dq_motor = np.asarray(dq_motor, dtype=float)
        pin, model, data = inner._pin, inner._model, inner._data
        gravity = pin.computeGeneralizedGravity(model, data, q_motor).copy()
        diagonal = np.diag(pin.crba(model, data, q_motor)).copy() + inner.transmission.rotor_inertia
        feedback = diagonal * (self.kp * (q_ref - q_motor) + self.kd * (dq_ref - dq_motor))
        feedforward = gravity + inner.friction.torque(dq_motor)
        total = np.clip(feedforward + feedback, -self.effort_limit, self.effort_limit)
        return self.command_type(total=total, feedforward=feedforward, feedback=total - feedforward)


__all__ = ["command_reference", "kuka_position_emulation"]
