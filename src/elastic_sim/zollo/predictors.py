"""Predictors built from one identification, and the rows they are scored on.

All three predict the link-side transmission torque ``tau_J`` per bag:

* **ID-inverse** (simulation diagnostic only): the identified link model at
  the simulator's true link state, ``Y(q, dq, ddq) pi* + f_l(dq)``;
* **ID-rigid**: the same link model at the motor state, ``q ~ theta``, with
  exactly the inputs the two-mass models get (recorded ``q``, ``dq`` and the
  Savitzky-Golay derivative of ``dq``, contract window);
* **ID-elastic**: the identified elastic model driven by the motor-side
  signals only -- ``theta``, ``dtheta`` prescribe the motor, the link is
  integrated forward::

      M(q) ddq = K (theta - q) + D (dtheta - dq) - c(q, dq) - g(q) - f_l(dq)
      tau_J    = K (theta - q) + D (dtheta - dq)

  with ``M``, ``c``, ``g`` from the identified base parameters through the
  regressor (``M[:, k] = Y(q, 0, e_k) pi - Y(q, 0, 0) pi``).

The integrator is Newmark's average acceleration (the two-mass filter's,
`04_models_and_training.md` Sec 3.3: second order, no numerical damping),
with the spring, the damper and the link-friction slope implicit and the
rigid terms at the predicted state; the step is the sample step divided so
that ``omega_max h <= max_omega_h`` for the highest identified mode.  The
motor signals are interpolated linearly between samples.  A run that leaves
``divergence_rad`` of deflection or turns non-finite is stopped and reported,
never clipped.

**Bounded ID-elastic** (`04_variant_A_elastic.md` Sec 2): the identified K
clipped to the two-mass models' own stiffness bounds (the contract's prior
widened by ``sequence_models.STIFFNESS_WIDENING``, read through that code,
:meth:`TransmissionBounds.from_two_mass`), then D raised to the ``zeta_min``
floor with the clipped K (:meth:`ElasticModel.with_damping_floor`).  Nothing
is refitted; the identified values stay in the identification.

**A-pc** (`06_variant_A_consistent.md` C3): K and D are identified per
robot; the predictors' K_j, D_j are the medians over the training robots
(:func:`identified_transmission`), then as identified or bounded as above.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import os
import sys
from pathlib import Path

import numpy as np

from .data import Bag
from .identify import Identification
from .regressor import coulomb


class LinkModel:
    """``tau = Y(q, dq, ddq) pi + f_l(dq)`` for one identified (or nominal) inertial vector."""

    def __init__(self, rigid_model: tuple, inertial: np.ndarray, viscous: np.ndarray | None = None,
                 coulomb_: np.ndarray | None = None, epsilon: float = 1.0e-3) -> None:
        self.pin, self.model, self.data = rigid_model
        self.n = int(self.model.nv)
        self.inertial = np.asarray(inertial, dtype=float)
        self.viscous = np.zeros(self.n) if viscous is None else np.asarray(viscous, dtype=float)
        self.coulomb = np.zeros(self.n) if coulomb_ is None else np.asarray(coulomb_, dtype=float)
        self.epsilon = float(epsilon)
        self._zero = np.zeros(self.n)
        self._unit = np.eye(self.n)

    @classmethod
    def from_identification(cls, result: Identification, rigid_model: tuple) -> "LinkModel":
        layout = result.layout
        if not result.config.pooled_predictor:
            raise ValueError("predictors need a pooled identification (transmission: pooled, per_robot_motor or "
                             "per_robot_transmission)")
        full = result.full
        friction = [np.asarray([full[layout.index(g, j)] for j in range(layout.n)]) if layout.has(g) else None
                    for g in ("link_viscous", "link_coulomb")]
        return cls(rigid_model, full[layout.inertial_slice()], friction[0], friction[1], result.config.epsilon_link)

    def _Y(self, q: np.ndarray, dq: np.ndarray, ddq: np.ndarray) -> np.ndarray:
        return self.pin.computeJointTorqueRegressor(self.model, self.data, q, dq, ddq)

    def friction(self, dq: np.ndarray) -> np.ndarray:
        return self.viscous * dq + self.coulomb * coulomb(dq, self.epsilon)

    def friction_slope(self, dq: np.ndarray) -> np.ndarray:
        if self.epsilon <= 0.0:
            return self.viscous.copy()
        return self.viscous + self.coulomb / self.epsilon * (1.0 - np.tanh(dq / self.epsilon) ** 2)

    def torque(self, q: np.ndarray, dq: np.ndarray, ddq: np.ndarray) -> np.ndarray:
        """Over samples ``[T, n]``."""
        out = np.empty_like(np.asarray(q, dtype=float))
        for i in range(len(q)):
            out[i] = self._Y(q[i], dq[i], ddq[i]) @ self.inertial
        return out + self.friction(np.asarray(dq, dtype=float))

    def gravity(self, q: np.ndarray) -> np.ndarray:
        return self._Y(q, self._zero, self._zero) @ self.inertial

    def bias(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        """``c(q, dq) + g(q)`` (no friction)."""
        return self._Y(q, dq, self._zero) @ self.inertial

    def mass(self, q: np.ndarray) -> np.ndarray:
        g = self.gravity(q)
        M = np.column_stack([self._Y(q, self._zero, self._unit[k]) @ self.inertial - g for k in range(self.n)])
        return 0.5 * (M + M.T)


def is_positive_definite(M: np.ndarray) -> bool:
    try:
        np.linalg.cholesky(M)
        return True
    except np.linalg.LinAlgError:
        return False


@dataclass(frozen=True)
class ElasticSettings:
    max_omega_h: float = 0.5
    divergence_rad: float = 0.2
    max_substeps: int = 200
    rigid_update: str = "substep"       # "substep" | "sample": where M, c, g are re-evaluated

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "ElasticSettings":
        return cls(**dict(values or {}))


@dataclass
class ElasticRun:
    tau: np.ndarray
    q: np.ndarray
    substeps: int
    omega_max: float
    diverged_at: float | None = None
    diverged_joint: int | None = None
    non_pd_samples: int = 0
    equilibrium_residual: float = 0.0
    notes: list[str] = field(default_factory=list)


def per_robot_transmission(result: Identification) -> dict[str, np.ndarray]:
    """``{"stiffness": [robots, n], "damping": [robots, n]}`` of the groups identified per robot (else empty)."""
    layout = result.layout
    full = result.full
    out = {}
    for group in ("stiffness", "damping"):
        if layout.robots is not None and group in layout.per_robot_groups and layout.has(group):
            out[group] = np.asarray([[full[layout.index(group, j, r)] for j in range(layout.n)] for r in layout.robots])
    return out


def identified_transmission(result: Identification) -> tuple[np.ndarray, np.ndarray | None]:
    """``(K, D)`` per joint for the predictors (``D`` ``None`` without D columns).

    Pooled columns as identified; per-robot columns (A-pc) by their median
    over the training robots (C3).
    """
    layout = result.layout
    full = result.full
    per_robot = per_robot_transmission(result)

    def joint_values(group: str) -> np.ndarray:
        if group in per_robot:
            return np.median(per_robot[group], axis=0)
        return np.asarray([full[layout.index(group, j)] for j in range(layout.n)])

    return joint_values("stiffness"), joint_values("damping") if layout.spring_damping else None


@dataclass(frozen=True)
class TransmissionBounds:
    """The two-mass models' stiffness bounds and the contract's ``zeta_min``."""

    k_low: np.ndarray
    k_high: np.ndarray
    zeta_min: float
    source: str = ""

    @classmethod
    def from_two_mass(cls, contract: Mapping[str, Any], lnn_repo: str | Path | None = None) -> "TransmissionBounds":
        """Through ``sequence_models.TwoMassMixin._init_two_mass``, the code path the LNN two-mass runs use.

        ``lnn_repo`` is the ``dynamic_model_nn`` checkout (default ``$DMN_ROOT``,
        else the sibling of this repository).  The bounds are its float32 buffers.
        """
        repo = Path(lnn_repo or os.environ.get("DMN_ROOT") or Path(__file__).resolve().parents[4] / "dynamic_model_nn")
        if os.fspath(repo) not in sys.path:
            sys.path.insert(0, os.fspath(repo))
        import torch
        import sequence_models

        priors = contract["transmission_priors"]
        joints = priors["joints"]

        class _Bounds(sequence_models.TwoMassMixin, torch.nn.Module):
            def __init__(self) -> None:
                torch.nn.Module.__init__(self)
                self._init_two_mass(len(joints), [j["stiffness_range"] for j in joints],
                                    [j["damping_ratio_range"] for j in joints], "diagonal", "trapezoidal")

        probe = _Bounds()
        return cls(probe.stiffness_low.double().numpy(), probe.stiffness_high.double().numpy(),
                   float(priors["zeta_min"]),
                   f"{Path(sequence_models.__file__).resolve()} (STIFFNESS_WIDENING {sequence_models.STIFFNESS_WIDENING})")

    def clip(self, K: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(K, dtype=float), self.k_low, self.k_high)

    def describe(self) -> dict[str, Any]:
        return {"k_low": self.k_low.tolist(), "k_high": self.k_high.tolist(), "zeta_min": self.zeta_min,
                "source": self.source}


class ElasticModel:
    """ID-elastic: the identified link driven through the identified spring by the motor signals."""

    def __init__(self, link: LinkModel, stiffness: np.ndarray, damping: np.ndarray | None,
                 settings: ElasticSettings = ElasticSettings()) -> None:
        self.link = link
        self.K = np.asarray(stiffness, dtype=float)
        self.D = np.zeros_like(self.K) if damping is None else np.asarray(damping, dtype=float)
        self.settings = settings
        if np.any(self.K <= 0.0):
            raise ValueError(f"ID-elastic needs positive identified stiffness, got {self.K}")

    @classmethod
    def from_identification(cls, result: Identification, rigid_model: tuple,
                            settings: ElasticSettings = ElasticSettings()) -> "ElasticModel":
        K, D = identified_transmission(result)
        return cls(LinkModel.from_identification(result, rigid_model), K, D, settings)

    @classmethod
    def bounded(cls, result: Identification, rigid_model: tuple, bounds: TransmissionBounds,
                settings: ElasticSettings = ElasticSettings()) -> "ElasticModel":
        """The identified model with K clipped to ``bounds``; apply :meth:`with_damping_floor` per bag for D."""
        K, D = identified_transmission(result)
        return cls(LinkModel.from_identification(result, rigid_model), bounds.clip(K), D, settings)

    def with_damping_floor(self, zeta_min: float, qs: Sequence[np.ndarray]) -> "ElasticModel":
        """The labelled ablation of plan Sec 5.6: ``D_j >= 2 zeta_min sqrt(K_j M_jj)``, M_jj the mean over ``qs``.

        Not the identified model: a negative identified D (the unmodelled
        transmission error biases it, `02_results.md`) makes the link mode
        unstable, and this variant shows what the rest of the model does
        with the smallest damping the prior allows.
        """
        mean_diagonal = np.mean([np.diag(self.link.mass(q)) for q in qs], axis=0)
        floor = 2.0 * zeta_min * np.sqrt(self.K * np.abs(mean_diagonal))
        return ElasticModel(self.link, self.K, np.maximum(self.D, floor), self.settings)

    def omega_max(self, qs: Sequence[np.ndarray]) -> float:
        """Highest mode of the link against a clamped motor, ``max eig(M^-1 (K + D/h-free))``, over ``qs``."""
        best = 0.0
        for q in qs:
            eig = np.linalg.eigvals(np.linalg.solve(self.link.mass(q), np.diag(self.K)))
            best = max(best, float(np.sqrt(np.max(np.abs(eig)))))
        return best

    def equilibrium(self, theta: np.ndarray, iterations: int = 50) -> tuple[np.ndarray, float]:
        """``q`` with ``K (theta - q) = g(q)``, Newton from ``q = theta``."""
        q = np.asarray(theta, dtype=float).copy()
        for _ in range(iterations):
            residual = self.K * (theta - q) - self.link.gravity(q)
            if np.max(np.abs(residual)) < 1e-10:
                break
            jac = -np.diag(self.K) - _gravity_jacobian(self.link, q)
            q = q - np.linalg.solve(jac, residual)
        residual = self.K * (theta - q) - self.link.gravity(q)
        return q, float(np.max(np.abs(residual)))

    def run(self, theta: np.ndarray, dtheta: np.ndarray, step: float) -> ElasticRun:
        theta = np.asarray(theta, dtype=float)
        dtheta = np.asarray(dtheta, dtype=float)
        T, n = theta.shape
        s = self.settings
        probes = [theta[i] for i in np.linspace(0, T - 1, min(T, 20)).astype(int)]
        omega = self.omega_max(probes)
        m = int(min(max(1, np.ceil(step * omega / s.max_omega_h)), s.max_substeps))
        h = step / m
        run = ElasticRun(np.full((T, n), np.nan), np.full((T, n), np.nan), m, omega)
        if m == s.max_substeps:
            run.notes.append(f"substeps capped at {m}: omega_max h = {omega * h:.2f}")
        q, run.equilibrium_residual = self.equilibrium(theta[0])
        v = dtheta[0].copy()
        K, D = self.K, self.D
        M = self.link.mass(q)
        a = np.linalg.solve(M, K * (theta[0] - q) + D * (dtheta[0] - v) - self.link.bias(q, v) - self.link.friction(v))
        run.tau[0] = K * (theta[0] - q) + D * (dtheta[0] - v)
        run.q[0] = q
        for k in range(T - 1):
            M = self.link.mass(q)
            if not is_positive_definite(M):
                run.non_pd_samples += 1
            bias = self.link.bias(q, v) if s.rigid_update == "sample" else None
            for i in range(1, m + 1):
                w = i / m
                th = (1.0 - w) * theta[k] + w * theta[k + 1]
                dth = (1.0 - w) * dtheta[k] + w * dtheta[k + 1]
                q_p = q + h * v + 0.25 * h * h * a
                v_p = v + 0.5 * h * a
                if s.rigid_update == "substep":
                    M = self.link.mass(q_p)
                    bias = self.link.bias(q_p, v_p)
                S = self.link.friction_slope(v_p)
                lhs = M + np.diag(0.5 * h * (D + S) + 0.25 * h * h * K)
                rhs = K * (th - q_p) + D * (dth - v_p) - bias - self.link.friction(v_p)
                a = np.linalg.solve(lhs, rhs)
                q = q_p + 0.25 * h * h * a
                v = v_p + 0.5 * h * a
            run.tau[k + 1] = K * (theta[k + 1] - q) + D * (dtheta[k + 1] - v)
            run.q[k + 1] = q
            deflection = np.abs(theta[k + 1] - q)
            if not (np.isfinite(q).all() and np.isfinite(v).all()) or np.max(deflection) > s.divergence_rad:
                run.diverged_at = float(k + 1)
                bad = ~(np.isfinite(q) & np.isfinite(v))
                run.diverged_joint = int(np.argmax(bad) if bad.any() else np.argmax(deflection))
                run.tau[k + 1:] = np.nan
                run.q[k + 1:] = np.nan
                break
        return run


def _gravity_jacobian(link: LinkModel, q: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    base = link.gravity(q)
    jac = np.empty((len(q), len(q)))
    for k in range(len(q)):
        dq = np.zeros(len(q))
        dq[k] = eps
        jac[:, k] = (link.gravity(q + dq) - base) / eps
    return jac


def consumer_inputs(bag: Bag, contract: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(theta, dtheta, ddtheta)`` exactly as the two-mass models get them."""
    from ..diagnostics import savitzky_golay_acceleration

    diff = contract.get("differentiation") or {}
    ddq = savitzky_golay_acceleration(bag.dtheta, bag.step, window=int(diff.get("sg_window") or 11),
                                      order=int(diff.get("sg_poly") or 3))
    return bag.theta, bag.dtheta, ddq


def kept_rows(t: np.ndarray, head_s: float, tolerance: float) -> np.ndarray:
    """The harness's scored rows: ``t - t_bag_start >= burn-in`` (``tolerance`` as the harness has it)."""
    t = np.asarray(t, dtype=float)
    return (t - t[0]) >= head_s - tolerance


def score(predictions: Mapping[str, np.ndarray], bags: Sequence[Bag], head_s: float,
          tolerance: float) -> dict[str, Any]:
    """Pooled RMSE over joints and kept rows, per joint, NRMSE and per bag (diverged bags apart)."""
    errors, targets, per_bag, diverged = [], [], {}, []
    for bag in bags:
        pred = predictions[bag.name]
        keep = kept_rows(bag.t, head_s, tolerance)
        e = pred[keep] - bag.tau_J[keep]
        if not np.isfinite(e).all():
            diverged.append(bag.name)
            per_bag[bag.name] = float("nan")
            continue
        errors.append(e)
        targets.append(bag.tau_J[keep])
        per_bag[bag.name] = float(np.sqrt(np.mean(e ** 2)))
    if not errors:
        return {"rmse": float("nan"), "rows": 0, "per_bag": per_bag, "diverged": diverged}
    e, y = np.vstack(errors), np.vstack(targets)
    rmse_j = np.sqrt(np.mean(e ** 2, axis=0))
    nrmse_j = rmse_j / np.sqrt(np.mean(y ** 2, axis=0))
    return {"rmse": float(np.sqrt(np.mean(e ** 2))), "rmse_joints": rmse_j.tolist(),
            "nrmse_joints": nrmse_j.tolist(), "nrmse_mean": float(np.mean(nrmse_j)), "rows": int(len(e)),
            "per_bag": per_bag, "diverged": diverged}
