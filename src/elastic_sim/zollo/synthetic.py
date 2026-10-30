"""A self-contained Spong-model elastic-joint simulator for the recovery checks.

Gate G1 of `01_zollo_plan.md`: before touching the producer's data, the
identification must recover the base parameters of a robot whose model is
exactly its own.  The plant here is that model::

    link:   M(q) ddq + c + g + f_l(dq) = tau_J = K (theta - q) + D (dtheta - dq)
    motor:  B ddtheta + f_m(dtheta) + tau_J = u

with Pinocchio's ABA for the link, RK4 at a fine step, ``u`` a motor-side PD
with gravity feedforward held over each step.  Each recorded sample carries
the accelerations evaluated at that sample's state and command, so the
equations hold to rounding and a noise-free identification can be checked
to numerical precision.  :func:`planar_two_link` is Zollo's 2-DoF planar
example (a = 0.5 m, l = 0.25 m, masses 20 and 10 kg).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .data import Bag
from .regressor import coulomb


def planar_two_link(link_length: float = 0.5, com: float = 0.25, masses: tuple[float, float] = (20.0, 10.0)) -> tuple:
    """Zollo's planar 2-DoF arm in a vertical plane: ``(pin, model, data)``."""
    import pinocchio as pin

    model = pin.Model()
    model.gravity.linear = np.array([0.0, -9.81, 0.0])
    parent = 0
    placement = pin.SE3.Identity()
    for index, mass in enumerate(masses):
        joint = model.addJoint(parent, pin.JointModelRZ(), placement, f"joint{index + 1}")
        rod = mass * link_length ** 2 / 12.0
        inertia = pin.Inertia(mass, np.array([com, 0.0, 0.0]), np.diag([1e-3 * rod, rod, rod]))
        model.appendBodyToJoint(joint, inertia, pin.SE3.Identity())
        parent = joint
        placement = pin.SE3(np.eye(3), np.array([link_length, 0.0, 0.0]))
    return pin, model, model.createData()


@dataclass
class SpongArm:
    rigid_model: tuple
    K: np.ndarray
    D: np.ndarray
    B: np.ndarray
    motor_viscous: np.ndarray
    motor_coulomb: np.ndarray
    link_viscous: np.ndarray
    link_coulomb: np.ndarray
    epsilon_link: float = 1.0e-3
    epsilon_motor: float = 1.0e-2

    def accelerations(self, q: np.ndarray, v: np.ndarray, th: np.ndarray, w: np.ndarray,
                      u: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pin, model, data = self.rigid_model
        tau = self.K * (th - q) + self.D * (w - v)
        f_l = self.link_viscous * v + self.link_coulomb * coulomb(v, self.epsilon_link)
        ddq = np.asarray(pin.aba(model, data, q, v, tau - f_l), dtype=float).copy()
        f_m = self.motor_viscous * w + self.motor_coulomb * coulomb(w, self.epsilon_motor)
        return ddq, (u - f_m - tau) / self.B, tau

    def truth(self) -> dict[str, list[float]]:
        return {"stiffness": list(self.K), "damping": list(self.D), "rotor": list(self.B),
                "motor_viscous": list(self.motor_viscous), "motor_coulomb": list(self.motor_coulomb),
                "link_viscous": list(self.link_viscous), "link_coulomb": list(self.link_coulomb)}


def simulate(arm: SpongArm, *, duration: float, step: float, sample_every: int, seed: int = 0,
             amplitude: float = 0.6, frequencies: tuple[float, ...] = (0.3, 0.7, 1.3), kp: float | np.ndarray = 4e3,
             kd: float | np.ndarray = 150.0, q0: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """One exciting run; returns the sampled signals (exact accelerations)."""
    pin, model, data = arm.rigid_model
    n = model.nv
    rng = np.random.default_rng(seed)
    q0 = np.zeros(n) if q0 is None else np.asarray(q0, dtype=float)
    amp = amplitude * rng.uniform(0.5, 1.0, (n, len(frequencies)))
    phase = rng.uniform(0.0, 2.0 * np.pi, (n, len(frequencies)))
    omega = 2.0 * np.pi * np.asarray(frequencies)

    def reference(t: float) -> tuple[np.ndarray, np.ndarray]:
        arg = omega[None, :] * t + phase
        return q0 + (amp * (np.sin(arg) - np.sin(phase))).sum(axis=1), (amp * omega * np.cos(arg)).sum(axis=1)

    def command(t: float, th: np.ndarray, w: np.ndarray) -> np.ndarray:
        ref, dref = reference(t)
        gravity = np.asarray(pin.computeGeneralizedGravity(model, data, ref), dtype=float)
        return kp * (ref - th) + kd * (dref - w) + gravity

    def rhs(state: np.ndarray, u: np.ndarray) -> np.ndarray:
        q, v, th, w = np.split(state, 4)
        ddq, ddth, _ = arm.accelerations(q, v, th, w, u)
        return np.concatenate([v, ddq, w, ddth])

    # Start in static equilibrium at q0: K (theta - q) = g(q).
    g0 = np.asarray(pin.computeGeneralizedGravity(model, data, q0), dtype=float)
    state = np.concatenate([q0, np.zeros(n), q0 + g0 / arm.K, np.zeros(n)])
    steps = int(round(duration / step))
    rows: dict[str, list] = {k: [] for k in ("t", "q", "dq", "ddq", "theta", "dtheta", "ddtheta", "tau_J", "u")}
    for i in range(steps + 1):
        t = i * step
        q, v, th, w = np.split(state, 4)
        u = command(t, th, w)
        if i % sample_every == 0:
            ddq, ddth, tau = arm.accelerations(q, v, th, w, u)
            for key, value in (("t", t), ("q", q), ("dq", v), ("ddq", ddq), ("theta", th), ("dtheta", w),
                               ("ddtheta", ddth), ("tau_J", tau), ("u", u)):
                rows[key].append(np.array(value, dtype=float, copy=True))
        k1 = rhs(state, u)
        k2 = rhs(state + 0.5 * step * k1, u)
        k3 = rhs(state + 0.5 * step * k2, u)
        k4 = rhs(state + step * k3, u)
        state = state + step / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        if not np.isfinite(state).all():
            raise RuntimeError(f"synthetic rollout diverged at t={t:.4f}s")
    return {k: np.asarray(v) for k, v in rows.items()}


def as_bag(run: dict[str, np.ndarray], name: str, tier: str = "e00", *, exact: bool = True,
           noise: dict[str, float] | None = None, seed: int = 0) -> Bag:
    """A :class:`Bag` of a run; with ``exact`` the identification uses the true derivatives.

    ``noise`` adds white Gaussian noise of the given standard deviation to
    ``theta``, ``q`` (positions) and ``tau_J``, ``u`` (torques); a noisy bag is
    never exact (it goes through Zollo's filter and differences).
    """
    rng = np.random.default_rng(seed)
    noisy = {}
    for key in ("theta", "q", "tau_J", "u"):
        sigma = float((noise or {}).get(key, 0.0))
        noisy[key] = run[key] + (rng.normal(0.0, sigma, run[key].shape) if sigma else 0.0)
    bag = Bag(name, tier, "train", run["t"], noisy["theta"], run["dtheta"], noisy["tau_J"], u=noisy["u"],
              q=noisy["q"], dq=run["dq"])
    if exact and not noise:
        bag.exact = {k: run[k] for k in ("theta", "dtheta", "ddtheta", "q", "dq", "ddq", "tau_J", "u")}
    return bag


def iiwa_arm(rigid_model: tuple, stiffness: Any, rotor: Any) -> SpongArm:
    """The iiwa with the simulator's nominal K and B, the r7 friction laws, a little spring damping."""
    K, B = np.asarray(stiffness, dtype=float), np.asarray(rotor, dtype=float)
    return SpongArm(rigid_model, K, 2.0 * 0.1 * np.sqrt(K * 0.3 * B), B,
                    motor_viscous=np.array([12.0, 12.0, 8.0, 8.0, 4.0, 2.5, 2.5]),
                    motor_coulomb=np.array([6.4, 6.4, 3.5, 3.5, 2.2, 0.8, 0.8]),
                    link_viscous=np.array([1.0, 1.0, 0.8, 0.8, 0.5, 0.3, 0.3]),
                    link_coulomb=np.array([4.5, 4.5, 3.0, 3.0, 1.7, 1.0, 1.0]))
