"""The fair classical baselines (`REFACTOR_SPECS/model_based_baseline/03_fair_comparison_plan.md`).

Two identified models that compete on exactly the LNNs' terms (plan Sec 2):

* **ID-rigid-fair** (plan Sec 3.1, the residual LNNs' counterpart)::

      tau = Y(theta, dtheta, ddtheta) pi + F_v dtheta + F_c tanh(dtheta / eps)

  weighted linear least squares on every train-split row, the residual
  LNNs' training rows (``dataset.declared_split_loaders``).
* **ID-2M-fair** (Sec 3.2, the two-mass LNNs' counterpart): that torque,
  ``tau_r``, through the two-mass filter of FR_06 Sec 3.3 with
  ``M_hat(theta)`` from the identified inertials, ``D = 2 zeta sqrt(K M_bar)``,
  ``K`` and ``zeta`` bounded by sigmoids exactly as ``TwoMassMixin``; fitted
  by output error on the two-mass training windows (``dataset.bag_windows``:
  ``burn_in + 1`` s, stride half a window, loss on each window's rows after
  its burn-in), Levenberg-Marquardt, stopped on the validation RMSE.

**Signals (F1).**  Only ``t``, ``bag``, ``split``, ``q``, ``dq``, ``ft`` are
read (:data:`FAIR_SIGNALS`, :func:`bags_from_frame` selects them before
anything else happens), read as float32 like the consumer.  ``ddtheta`` is
the Savitzky-Golay derivative of ``dq`` with the contract window.  The
contract supplies only what the two-mass LNNs get: the SG window, the burn-in
and the transmission priors (F4).

**The filter.**  :class:`BatchFilter` is ``elastic_sim.two_mass.filter_sequence``
(full coupling, trapezoidal) run over a batch of equal-length windows and
several input columns at once; the recursion is linear in its input, so one
pass gives the filtered regressor columns (the Jacobian in ``pi``, ``F``)
and a second the stiffness and damping sensitivities (the same recursion
driven by ``-E_j delta`` and ``-E_j delta_dot``).  The test suite checks it
against ``filter_sequence`` and the Jacobian against finite differences.
Predictions on whole bags use ``filter_sequence`` itself.

**Inertia.**  ID-rigid-fair's linear fit gives an ``M_hat(theta)`` that is
not positive definite (motor-side ``ddtheta`` carries the elastic modes, and
above them the transmission torque moves against the motor acceleration), so
ID-2M-fair takes the plan's in-class fix: physically consistent inertials,
``pi = pi(theta)`` per body by the log-Cholesky parametrisation of the
pseudo-inertia (Rucker & Wensing 2022), friction linear.  Its stage 1 is
ID-rigid-fair's problem (same rows, weights, loss) in that parametrisation,
from the URDF (:func:`fit_rigid_consistent`).  A model is stored as a full
layout-order vector; Pinocchio's ``rnea``/``crba`` on a copy of the model
carrying it give ``Y pi`` and ``M_hat(theta)`` (checked against the
regressor).  ID-rigid-fair's own vector puts every dependent and
unidentifiable column at its URDF value (:func:`consistent_full`), which
keeps ``rnea`` exact.  The output-error Jacobian treats ``M_hat`` and
``M_bar`` as constant in ``theta``; every residual uses them at the current
iterate, and a Levenberg-Marquardt step is accepted only if that exact cost
decreases.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .basis import Accumulator, Base, select_base, solve
from .data import Bag, sidecar
from .regressor import FIRST_MOMENT, INERTIA, INERTIAL, Column

#: F1: the only per-row columns a fair identification reads.
FAIR_SIGNALS = ("q", "dq", "ft")
#: ``sequence_models.STIFFNESS_WIDENING``: K bounds = the prior widened by this factor each way.
STIFFNESS_WIDENING = 1.5
#: ``dataset._HEAD_TOLERANCE_S``: the "t - t_start >= burn_in" rule's tolerance.
HEAD_TOLERANCE_S = 1e-9
#: Coulomb smoothing candidates, picked on validation (F5).
EPSILONS = (1e-3, 1e-2, 1e-1)


def fair_columns(n: int) -> list[str]:
    return ["t", "bag", "split"] + [f"{p}{i}" for p in FAIR_SIGNALS for i in range(n)]


def bags_from_frame(frame: Any, n: int, splits: Sequence[str] | None = None) -> list[Bag]:
    """The bags of ``splits`` from a frame, reading only :func:`fair_columns` (float32, as the consumer)."""
    columns = [c for c in fair_columns(n) if c != "split" or "split" in frame.columns]
    frame = frame[columns]
    if splits is not None:
        frame = frame[frame["split"].isin(list(splits))]
    block = lambda g, p: g[[f"{p}{i}" for i in range(n)]].to_numpy(dtype=np.float32).astype(float)  # noqa: E731
    out = []
    for name, group in frame.groupby("bag", sort=False):
        split = str(group["split"].iloc[0]) if "split" in group.columns else "test"
        out.append(Bag(str(name), "", split, group["t"].to_numpy(dtype=float), block(group, "q"),
                       block(group, "dq"), block(group, "ft")))
    return out


def load_fair(path: str | Path, splits: Sequence[str] | None = None) -> tuple[list[Bag], dict[str, Any]]:
    """A producer file's bags of ``splits`` (all with ``None``), :func:`bags_from_frame`'s columns only."""
    import pyarrow.parquet as pq

    path = Path(path)
    contract = sidecar(path, "contract")
    n = int(contract["n_dof"])
    schema = set(pq.read_schema(path).names)
    columns = [c for c in fair_columns(n) if c in schema]
    filters = [("split", "in", list(splits))] if splits is not None and "split" in schema else None
    frame = pq.read_table(path, columns=columns, filters=filters).to_pandas()
    if frame.empty:
        raise ValueError(f"{path}: no rows for splits {splits}")
    return bags_from_frame(frame, n, splits), {"path": str(path), "contract": contract, "n": n}


# --------------------------------------------------------------------------------------------- priors and rows


@dataclass(frozen=True)
class FairPriors:
    """F4: what the two-mass models receive from the contract, and nothing else."""

    step: float
    burn_in_s: float
    sg_window: int
    sg_poly: int
    k_low: np.ndarray
    k_high: np.ndarray
    zeta_low: np.ndarray
    zeta_high: np.ndarray
    zeta_prior_high: np.ndarray

    @classmethod
    def from_contract(cls, contract: Mapping[str, Any], step: float) -> "FairPriors":
        priors = contract["transmission_priors"]
        stiffness = np.asarray([j["stiffness_range"] for j in priors["joints"]], dtype=float)
        ratio = np.asarray([j["damping_ratio_range"] for j in priors["joints"]], dtype=float)
        diff = contract.get("differentiation") or {}
        return cls(float(step), float(priors["burn_in_s"]), int(diff.get("sg_window") or 11),
                   int(diff.get("sg_poly") or 3), stiffness[:, 0] / STIFFNESS_WIDENING,
                   stiffness[:, 1] * STIFFNESS_WIDENING, ratio[:, 0], np.maximum(1.5 * ratio[:, 1], 1.0),
                   ratio[:, 1])

    @property
    def window_s(self) -> float:
        return self.burn_in_s + 1.0

    @property
    def stride_s(self) -> float:
        return self.window_s / 2.0

    @property
    def k_init(self) -> np.ndarray:
        return np.sqrt(self.k_low * self.k_high)

    @property
    def zeta_init(self) -> np.ndarray:
        return 0.5 * (self.zeta_low + self.zeta_prior_high)

    def stiffness(self, kappa: np.ndarray) -> np.ndarray:
        return self.k_low + (self.k_high - self.k_low) * _sigmoid(kappa)

    def zeta(self, eta: np.ndarray) -> np.ndarray:
        return self.zeta_low + (self.zeta_high - self.zeta_low) * _sigmoid(eta)

    def logits(self, stiffness: np.ndarray, zeta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (_logit((np.asarray(stiffness) - self.k_low) / (self.k_high - self.k_low)),
                _logit((np.asarray(zeta) - self.zeta_low) / (self.zeta_high - self.zeta_low)))

    def describe(self) -> dict[str, Any]:
        return {"step": self.step, "burn_in_s": self.burn_in_s, "window_s": self.window_s,
                "stride_s": self.stride_s, "sg_window": self.sg_window, "sg_poly": self.sg_poly,
                "k_low": self.k_low.tolist(), "k_high": self.k_high.tolist(), "k_init": self.k_init.tolist(),
                "zeta_low": self.zeta_low.tolist(), "zeta_high": self.zeta_high.tolist(),
                "zeta_prior_high": self.zeta_prior_high.tolist(), "zeta_init": self.zeta_init.tolist()}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)    # TwoMassMixin's init clamp
    return np.log(p / (1.0 - p))


def head_mask(t: np.ndarray, head_s: float) -> np.ndarray:
    """``dataset.head_mask``: rows at or after ``head_s`` from the sequence's first sample."""
    t = np.asarray(t, dtype=float)
    return (t - t[0]) >= float(head_s) - HEAD_TOLERANCE_S


def sequence_windows(t: np.ndarray, priors: FairPriors) -> list[tuple[int, int]]:
    """``dataset.bag_windows`` for one bag: ``(start, stop)`` of each training window."""
    step = float(np.median(np.diff(t)))
    burn = int(math.ceil(priors.burn_in_s / step - 1e-6))
    length = burn + int(round((priors.window_s - priors.burn_in_s) / step))
    stride = int(round(priors.stride_s / step))
    if length > len(t):
        return []
    return [(s, s + length) for s in range(0, len(t) - length + 1, max(1, stride))]


def target_scale(bags: Sequence[Bag]) -> np.ndarray:
    """``dataset.target_scale``: 1 / the target's per-joint std over every train row (population)."""
    tau = np.vstack([bag.tau_J for bag in bags]).astype(np.float32)
    return 1.0 / np.maximum(tau.std(axis=0, dtype=np.float64), 1e-8)


def row_counts(train: Sequence[Bag], val: Sequence[Bag], priors: FairPriors) -> dict[str, int]:
    """FG1: the residual and two-mass row sets of this identification."""
    windows = [w for bag in train for w in sequence_windows(bag.t, priors)]
    loss = sum(int(head_mask(bag.t[a:b], priors.burn_in_s).sum()) for bag in train
               for a, b in sequence_windows(bag.t, priors))
    return {"train_rows": sum(len(b.t) for b in train), "val_rows": sum(len(b.t) for b in val),
            "train_windows": len(windows), "train_window_loss_rows": loss,
            "val_bag_loss_rows": sum(int(head_mask(b.t, priors.burn_in_s).sum()) for b in val)}


# ----------------------------------------------------------------------------------------------- regressor rows


@dataclass(frozen=True)
class FairLayout:
    """Columns: Zollo's inertial grouping (masses, first moments, inertias), then F_v, F_c per joint."""

    n: int

    @property
    def columns(self) -> list[Column]:
        n = self.n
        out = [Column("mass", b, "m") for b in range(n)]
        out += [Column("first_moment", b, INERTIAL[k]) for b in range(n) for k in FIRST_MOMENT]
        out += [Column("inertia", b, INERTIAL[k]) for b in range(n) for k in INERTIA]
        out += [Column("viscous", j) for j in range(n)] + [Column("coulomb", j) for j in range(n)]
        return out

    @property
    def size(self) -> int:
        return 12 * self.n

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.columns]

    def inertial_slice(self) -> np.ndarray:
        """Layout positions of the 10 n inertial columns, in Pinocchio's parameter order."""
        positions = np.empty(10 * self.n, dtype=int)
        for position, column in enumerate(self.columns[:10 * self.n]):
            positions[10 * column.joint + INERTIAL.index(column.component)] = position
        return positions

    def friction_slices(self) -> tuple[slice, slice]:
        return slice(10 * self.n, 11 * self.n), slice(11 * self.n, 12 * self.n)


def coulomb_feature(dtheta: np.ndarray, epsilon: float) -> np.ndarray:
    return np.tanh(np.asarray(dtheta, dtype=float) / epsilon)


@dataclass
class BagRows:
    """One bag's consumer inputs and its rigid regressor at the motor state (computed once)."""

    bag: Bag
    theta: np.ndarray
    dtheta: np.ndarray
    ddtheta: np.ndarray
    Y: np.ndarray                  # [T, n, 10 n], Pinocchio order

    @classmethod
    def build(cls, bag: Bag, rigid_model: tuple, sg_window: int, sg_poly: int) -> "BagRows":
        from ..diagnostics import savitzky_golay_acceleration
        from .regressor import rigid_regressor

        ddtheta = savitzky_golay_acceleration(bag.dtheta, bag.step, window=sg_window, order=sg_poly)
        return cls(bag, bag.theta, bag.dtheta, ddtheta, rigid_regressor(*rigid_model, bag.theta, bag.dtheta, ddtheta))

    def rows(self, layout: FairLayout, epsilon: float) -> np.ndarray:
        """``W [T, n, 12 n]`` in layout order."""
        T, n = self.theta.shape
        W = np.zeros((T, n, layout.size))
        W[:, :, layout.inertial_slice()] = self.Y
        viscous, coul = layout.friction_slices()
        idx = np.arange(n)
        W[:, idx, viscous.start + idx] = self.dtheta
        W[:, idx, coul.start + idx] = coulomb_feature(self.dtheta, epsilon)
        return W



# ---------------------------------------------------------------------------------- physically consistent inertials


def _inertia_matrix(pi: np.ndarray) -> np.ndarray:
    _, _, _, _, ixx, ixy, iyy, ixz, iyz, izz = pi
    return np.asarray([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]], dtype=float)


def inertial_from_log_cholesky(theta: np.ndarray) -> np.ndarray:
    """Rucker & Wensing (2022): ``J = U U^T`` with ``U`` upper triangular, ``J`` the pseudo-inertia.

    ``theta = (alpha, d1, d2, d3, s12, s23, s13, t1, t2, t3)``; returns Pinocchio's
    ``[m, mc_x, mc_y, mc_z, I_xx, I_xy, I_yy, I_xz, I_yz, I_zz]`` (about the body frame).
    Every ``theta`` is physically consistent (``J`` positive definite).
    """
    alpha, d1, d2, d3, s12, s23, s13, t1, t2, t3 = theta
    U = np.exp(alpha) * np.asarray([[np.exp(d1), s12, s13, t1], [0.0, np.exp(d2), s23, t2],
                                    [0.0, 0.0, np.exp(d3), t3], [0.0, 0.0, 0.0, 1.0]])
    J = U @ U.T
    sigma, h, m = J[:3, :3], J[:3, 3], J[3, 3]
    inertia = np.trace(sigma) * np.eye(3) - sigma
    return np.asarray([m, *h, inertia[0, 0], inertia[0, 1], inertia[1, 1], inertia[0, 2], inertia[1, 2],
                       inertia[2, 2]])


def log_cholesky_from_inertial(pi: np.ndarray) -> np.ndarray:
    """The inverse of :func:`inertial_from_log_cholesky`; ``pi`` must be physically consistent."""
    pi = np.asarray(pi, dtype=float)
    inertia = _inertia_matrix(pi)
    J = np.zeros((4, 4))
    J[:3, :3] = 0.5 * np.trace(inertia) * np.eye(3) - inertia
    J[:3, 3] = J[3, :3] = pi[1:4]
    J[3, 3] = pi[0]
    U = np.linalg.cholesky(J[::-1, ::-1])[::-1, ::-1]           # upper triangular, J = U U^T
    scale = U[3, 3]
    return np.asarray([np.log(scale), *np.log(np.diag(U)[:3] / scale), U[0, 1] / scale, U[1, 2] / scale,
                       U[0, 2] / scale, *(U[:3, 3] / scale)])


def consistent_inertials(theta: np.ndarray) -> np.ndarray:
    """``[10 n]`` Pinocchio-order inertials of ``n`` log-Cholesky blocks."""
    return np.concatenate([inertial_from_log_cholesky(b) for b in np.asarray(theta, dtype=float).reshape(-1, 10)])


def consistent_jacobian(theta: np.ndarray, step: float = 1e-6) -> np.ndarray:
    """``d pi / d theta`` (block diagonal, ``[10 n, 10 n]``), central differences per block."""
    theta = np.asarray(theta, dtype=float).reshape(-1, 10)
    out = np.zeros((theta.size, theta.size))
    for b, block in enumerate(theta):
        for k in range(10):
            plus, minus = block.copy(), block.copy()
            plus[k] += step
            minus[k] -= step
            out[10 * b:10 * b + 10, 10 * b + k] = (inertial_from_log_cholesky(plus)
                                                   - inertial_from_log_cholesky(minus)) / (2 * step)
    return out


def physically_consistent(pi: np.ndarray) -> bool:
    """Positive-definite pseudo-inertia (Wensing et al. 2017), every body."""
    for block in np.asarray(pi, dtype=float).reshape(-1, 10):
        inertia = _inertia_matrix(block)
        J = np.zeros((4, 4))
        J[:3, :3] = 0.5 * np.trace(inertia) * np.eye(3) - inertia
        J[:3, 3] = J[3, :3] = block[1:4]
        J[3, 3] = block[0]
        if np.linalg.eigvalsh(J)[0] <= 0.0:
            return False
    return True


# ---------------------------------------------------------------------------------------------------- the models


def consistent_full(base: Base, beta: np.ndarray, nominal: np.ndarray) -> np.ndarray:
    """A full vector with base vector ``beta``: dependent and zero columns at ``nominal``, the rest solved."""
    full = np.asarray(nominal, dtype=float).copy()
    full[base.independent] = np.asarray(beta, dtype=float) - (
        base.relation @ full[base.dependent] if len(base.dependent) else 0.0)
    return full


class LinkDynamics:
    """``rnea``/``crba`` on a copy of the rigid model carrying one inertial vector (Pinocchio order).

    ``rnea`` is exact for any vector with non-zero masses; ``crba`` composes the
    bodies' inertias and is used only for physically consistent ones.
    """

    def __init__(self, rigid_model: tuple, inertial: np.ndarray, consistent: bool | None = None) -> None:
        pin, model, _ = rigid_model
        self.pin = pin
        self.model = pin.Model(model)
        inertial = np.asarray(inertial, dtype=float)
        # ``consistent=True``: built from log-Cholesky blocks, consistent by construction (a nearly
        # degenerate body can miss the numerical eigenvalue test).
        self.consistent = physically_consistent(inertial) if consistent is None else bool(consistent)
        for body in range(1, self.model.njoints):
            values = inertial[10 * (body - 1):10 * body]
            if abs(values[0]) < 1e-9:
                raise ValueError(f"body {body}: mass {values[0]:.3g} cannot carry an inertial")
            self.model.inertias[body] = pin.Inertia.FromDynamicParameters(values)
        self.data = self.model.createData()
        self.n = int(self.model.nv)

    def torque(self, q: np.ndarray, dq: np.ndarray, ddq: np.ndarray) -> np.ndarray:
        rnea, model, data = self.pin.rnea, self.model, self.data
        return np.asarray([rnea(model, data, q[i], dq[i], ddq[i]) for i in range(len(q))])

    def inertia(self, q: np.ndarray) -> np.ndarray:
        if not self.consistent:
            raise ValueError("crba needs physically consistent inertials (use mass_by_regressor)")
        crba, model, data = self.pin.crba, self.model, self.data
        out = np.empty((len(q), self.n, self.n))
        for i in range(len(q)):
            M = crba(model, data, q[i])
            out[i] = np.triu(M) + np.triu(M, 1).T
        return out


def mass_by_regressor(rigid_model: tuple, inertial: np.ndarray, q: np.ndarray) -> np.ndarray:
    """``M(q)`` of any inertial vector: ``M[:, k] = Y(q, 0, e_k) pi - Y(q, 0, 0) pi``."""
    pin, model, data = rigid_model
    n = int(model.nv)
    zero = np.zeros(n)
    out = np.empty((len(q), n, n))
    for i, qi in enumerate(np.atleast_2d(q)):
        g = pin.computeJointTorqueRegressor(model, data, qi, zero, zero) @ inertial
        M = np.column_stack([pin.computeJointTorqueRegressor(model, data, qi, zero, e) @ inertial - g
                             for e in np.eye(n)])
        out[i] = 0.5 * (M + M.T)
    return out


def mass_regressor(rigid_model: tuple, q: np.ndarray) -> np.ndarray:
    """``[T, n, n, 10 n]`` float32: entry ``[t, i, j, c] = d (M(q_t) e_i)_j / d pi_c`` (Pinocchio order)."""
    pin, model, data = rigid_model
    n = int(model.nv)
    zero = np.zeros(n)
    out = np.empty((len(q), n, n, 10 * n), dtype=np.float32)
    for t, qt in enumerate(np.atleast_2d(q)):
        g = pin.computeJointTorqueRegressor(model, data, qt, zero, zero)
        for i in range(n):
            out[t, i] = pin.computeJointTorqueRegressor(model, data, qt, zero, np.eye(n)[i]) - g
    return out


@dataclass
class FairModel:
    """ID-rigid-fair (``stiffness`` ``None``) or ID-2M-fair, as a full layout-order vector."""

    name: str
    n: int
    full: np.ndarray
    epsilon: float
    stiffness: np.ndarray | None = None
    zeta: np.ndarray | None = None
    info: dict[str, Any] = field(default_factory=dict)
    _dynamics: LinkDynamics | None = field(default=None, repr=False, compare=False)

    @property
    def layout(self) -> FairLayout:
        return FairLayout(self.n)

    @property
    def is_two_mass(self) -> bool:
        return self.stiffness is not None

    @property
    def inertial(self) -> np.ndarray:
        return self.full[self.layout.inertial_slice()]

    @property
    def viscous(self) -> np.ndarray:
        return self.full[self.layout.friction_slices()[0]]

    @property
    def coulomb(self) -> np.ndarray:
        return self.full[self.layout.friction_slices()[1]]

    def dynamics(self, rigid_model: tuple) -> LinkDynamics:
        if self._dynamics is None:
            self._dynamics = LinkDynamics(rigid_model, self.inertial,
                                          consistent=True if self.info.get("log_cholesky") is not None else None)
        return self._dynamics

    def driving_torque(self, rigid_model: tuple, theta, dtheta, ddtheta) -> np.ndarray:
        """``tau_r = Y pi + F_v dtheta + F_c tanh(dtheta / eps)`` (ID-rigid-fair's prediction)."""
        return (self.dynamics(rigid_model).torque(theta, dtheta, ddtheta) + self.viscous * dtheta
                + self.coulomb * coulomb_feature(dtheta, self.epsilon))

    def damping(self, mean_inertia: np.ndarray) -> np.ndarray:
        from ..two_mass import damping_from_ratio

        return damping_from_ratio(self.stiffness, self.zeta, mean_inertia)

    def predict(self, rigid_model: tuple, theta, dtheta, ddtheta, step: float) -> dict[str, np.ndarray]:
        """Whole-sequence prediction from its first sample (the LNNs' full-bag rollout)."""
        from ..two_mass import filter_sequence, mean_diagonal

        tau_r = self.driving_torque(rigid_model, theta, dtheta, ddtheta)
        if not self.is_two_mass:
            return {"tau": tau_r, "tau_r": tau_r}
        inertia = self.dynamics(rigid_model).inertia(theta)
        damping = self.damping(mean_diagonal(inertia))
        out = filter_sequence(tau_r, inertia, self.stiffness, damping, step, coupling="full")
        return {"tau": out["tau_s"], "tau_r": tau_r, "delta": out["delta"], "damping": damping,
                "non_pd": int((~_pd_mask(inertia)).sum())}

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "n": self.n, "columns": self.layout.names, "full": self.full.tolist(),
                "epsilon": self.epsilon,
                "stiffness": None if self.stiffness is None else self.stiffness.tolist(),
                "zeta": None if self.zeta is None else self.zeta.tolist(), "info": self.info}

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_json(), indent=1, default=_json_default), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "FairModel":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        arr = lambda v: None if v is None else np.asarray(v, dtype=float)  # noqa: E731
        return cls(raw["name"], int(raw["n"]), np.asarray(raw["full"], dtype=float), float(raw["epsilon"]),
                   arr(raw["stiffness"]), arr(raw["zeta"]), raw.get("info") or {})


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(type(value))


def _pd_mask(M: np.ndarray) -> np.ndarray:
    return np.linalg.eigvalsh(M)[..., 0] > 0.0


def nominal_full(rigid_model: tuple, n: int) -> np.ndarray:
    """The URDF's inertials in layout order, zero friction (FG4's driving torque)."""
    from .. import identification as idn

    full = np.zeros(FairLayout(n).size)
    full[FairLayout(n).inertial_slice()] = idn.standard_parameters(*rigid_model[:2])
    return full


def check_dynamics(model: FairModel, rigid_model: tuple, rows: BagRows, samples: int = 5) -> float:
    """Max relative mismatch of the model copy's ``rnea`` (and ``crba``, if consistent) vs the regressor."""
    dyn = model.dynamics(rigid_model)
    worst = 0.0
    for i in np.linspace(0, len(rows.theta) - 1, samples).astype(int):
        tau = rows.Y[i] @ model.inertial
        tau_pin = dyn.torque(rows.theta[i:i + 1], rows.dtheta[i:i + 1], rows.ddtheta[i:i + 1])[0]
        worst = max(worst, float(np.abs(tau - tau_pin).max() / max(np.abs(tau).max(), 1e-9)))
        if dyn.consistent:
            M = mass_by_regressor(rigid_model, model.inertial, rows.theta[i:i + 1])[0]
            M_pin = dyn.inertia(rows.theta[i:i + 1])[0]
            worst = max(worst, float(np.abs(M - M_pin).max() / max(np.abs(M).max(), 1e-9)))
    return worst


def non_pd_share(model: FairModel, rigid_model: tuple, bags: Sequence[BagRows], every: int = 200) -> dict[str, float]:
    """Share of probed motor states where the model's ``M_hat(theta)`` is not positive definite."""
    probes = np.vstack([rows.theta[::every] for rows in bags])
    eig = np.linalg.eigvalsh(mass_by_regressor(rigid_model, model.inertial, probes))[:, 0]
    return {"probes": int(len(eig)), "non_pd_share": float(np.mean(eig <= 0.0)), "min_eigenvalue": float(eig.min())}


# ----------------------------------------------------------------------------------------------- ID-rigid-fair


@dataclass
class RigidFit:
    model: FairModel
    base: Base
    beta: np.ndarray
    weighted: bool
    val_rmse: float
    train_rmse: float
    acc: Accumulator
    val_acc: Accumulator               # unweighted val rows: ||W x - T|| = ||R_W x - z|| exactly


def fit_rigid(train: Sequence[BagRows], val: Sequence[BagRows], rigid_model: tuple, *, epsilon: float,
              weighted: bool, scale: np.ndarray, tolerance: float = 1e-6) -> RigidFit:
    """Plan Sec 3.1: weighted LS (rows scaled by ``s_j`` when ``weighted``) on every train row."""
    n = train[0].theta.shape[1]
    layout = FairLayout(n)
    acc = Accumulator(layout.size)
    weights = scale if weighted else np.ones(n)
    for rows in train:
        W = rows.rows(layout, epsilon)
        acc.add(W.reshape(-1, layout.size), rows.bag.tau_J.reshape(-1), weights=np.tile(weights, len(W)))
    base = select_base(acc, tolerance=tolerance)
    solution = solve(acc, base)
    full = consistent_full(base, solution.base, nominal_full(rigid_model, n))
    model = FairModel("id_rigid_fair", n, full, float(epsilon), info={
        "weighted": bool(weighted), "rank": base.rank, "condition_scaled": base.condition_scaled,
        "rows": acc.rows, "optimiser": "linear least squares on the base parameters (zollo/basis.py)"})
    val_acc = Accumulator(layout.size)
    for rows in val:
        val_acc.add(rows.rows(layout, epsilon).reshape(-1, layout.size), rows.bag.tau_J.reshape(-1))
    train_rmse = pointwise_rmse(model, train, layout)
    val_rmse = pointwise_rmse(model, val, layout)
    model.info.update(train_rmse=train_rmse, val_rmse=val_rmse)
    return RigidFit(model, base, solution.base, bool(weighted), val_rmse, train_rmse, acc, val_acc)


def pointwise_rmse(model: FairModel, bags: Sequence[BagRows], layout: FairLayout) -> float:
    """The residual LNNs' validation statistic: RMSE over every row and joint, physical units."""
    total, count = 0.0, 0
    for rows in bags:
        e = rows.rows(layout, model.epsilon) @ model.full - rows.bag.tau_J
        total += float((e ** 2).sum())
        count += e.size
    return math.sqrt(total / count)


@dataclass(frozen=True)
class OutputErrorSettings:
    """Levenberg-Marquardt on the F6 loss, stopped on the validation RMSE (FR_03 T-1's rule)."""

    stage1_iterations: int = 2000
    stage2_iterations: int = 40
    stage3_iterations: int = 80
    patience: int = 30                 # the LNNs' early_stopping_patience, in LM iterations
    min_delta: float = 1e-4            # model_workflow.VAL_MIN_DELTA
    lambda0: float = 1e-3
    lambda_max: float = 1e10
    cost_tolerance: float = 1e-10      # relative decrease below which the fit has converged
    device: str = "auto"
    chunk: int = 64                    # train windows per filter batch (device memory)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "OutputErrorSettings":
        return cls(**dict(values or {}))


def _lm_step(H: np.ndarray, g: np.ndarray, lam: float) -> np.ndarray:
    """Marquardt's step; the scaling's floor keeps directions the rows never excite at zero step."""
    diag = np.diag(H)
    floor = 1e-9 * max(float(diag.max()), 1e-300)
    return -np.linalg.solve(H + lam * np.diag(np.maximum(diag, floor)), g)


def levenberg_marquardt(x0: np.ndarray, evaluate: Callable, val_rmse: Callable, *, mask: slice | np.ndarray,
                        iterations: int, settings: OutputErrorSettings, history: list[dict[str, Any]],
                        stage: int, log: Callable[[str], None] | None, describe: Callable | None = None,
                        best: dict[str, Any] | None = None) -> dict[str, Any]:
    """LM on ``evaluate(x) -> cost`` / ``evaluate(x, jacobian=True) -> (cost, H, g)`` over ``x[mask]``.

    Keeps the iterate of best ``val_rmse`` (relative improvement ``min_delta``)
    and stops after ``patience`` iterations without one, on convergence or
    after ``iterations``.
    """
    x = np.asarray(x0, dtype=float).copy()
    if best is None:
        best = {"x": x.copy(), "val": val_rmse(x), "stage": stage, "iteration": 0}
    lam, patience, started = settings.lambda0, 0, time.perf_counter()
    for it in range(1, iterations + 1):
        cost, H, g = evaluate(x, jacobian=True)
        accepted = False
        while lam <= settings.lambda_max:
            trial = x.copy()
            trial[mask] += _lm_step(H, g, lam)
            try:
                with np.errstate(over="ignore", invalid="ignore"):
                    trial_cost = evaluate(trial)
            except (ValueError, RuntimeError, np.linalg.LinAlgError, FloatingPointError):
                trial_cost = np.inf            # e.g. a step that makes M_hat singular: rejected like any other
            if np.isfinite(trial_cost) and trial_cost < cost:
                accepted = True
                break
            lam *= 4.0
        if not accepted:
            if log:
                log(f"  stage {stage} it {it}: no decrease up to lambda {settings.lambda_max:.0e}; converged")
            break
        decrease = (cost - trial_cost) / cost
        x, cost, lam = trial, trial_cost, max(lam / 3.0, 1e-12)
        val = val_rmse(x)
        history.append({"stage": stage, "iteration": it, "cost": cost, "val_rmse": val, "lambda": lam})
        if val < best["val"] * (1.0 - settings.min_delta):
            best, patience = {"x": x.copy(), "val": val, "stage": stage, "iteration": it}, 0
        else:
            patience += 1
        if log:
            log(f"  stage {stage} it {it:3d}: cost {cost:.6g} val {val:.4f} (best {best['val']:.4f}) lambda "
                f"{lam:.1e}{'  ' + describe(x) if describe else ''}  [{time.perf_counter() - started:.0f} s]")
        if patience >= settings.patience or decrease < settings.cost_tolerance:
            break
    return best


def fit_rigid_consistent(rigid: RigidFit, val: Sequence[BagRows], rigid_model: tuple,
                         settings: OutputErrorSettings, log: Callable[[str], None] | None = print
                         ) -> tuple[FairModel, np.ndarray, list[dict[str, Any]]]:
    """ID-rigid-fair's problem (same rows, weights, loss) over physically consistent inertials.

    The plan's in-class fix (Sec 3.2) when ``M_hat`` of the linear fit is not
    positive definite: ``pi = pi(theta)`` per body (log-Cholesky), friction
    linear; Levenberg-Marquardt from the URDF on the triangular factor of the
    stacked rows (exactly the train-row loss), the iterate of best pointwise
    validation RMSE kept.  Returns the model, ``x = [theta (10 n), F (2 n)]``
    and the history.
    """
    n = rigid.model.n
    layout = FairLayout(n)
    inertial_at = layout.inertial_slice()
    friction_at = np.r_[layout.friction_slices()[0], layout.friction_slices()[1]]
    R, z, rows = rigid.acc.regressor, rigid.acc.target, rigid.acc.rows
    urdf = nominal_full(rigid_model, n)[inertial_at]

    def full_of(x: np.ndarray) -> np.ndarray:
        full = np.zeros(layout.size)
        full[inertial_at] = consistent_inertials(x[:10 * n])
        full[friction_at] = x[10 * n:]
        return full

    def evaluate(x: np.ndarray, jacobian: bool = False):
        e = R @ full_of(x) - z
        cost = float(e @ e) / rows
        if not jacobian:
            return cost
        G = np.zeros((layout.size, 12 * n))
        G[np.ix_(inertial_at, np.arange(10 * n))] = consistent_jacobian(x[:10 * n])
        G[friction_at, 10 * n + np.arange(2 * n)] = 1.0
        J = R @ G
        return cost, J.T @ J / rows, J.T @ e / rows

    R_val, z_val = rigid.val_acc.regressor, rigid.val_acc.target

    def val_rmse(x: np.ndarray) -> float:
        e = R_val @ full_of(x) - z_val
        return math.sqrt(float(e @ e) / rigid.val_acc.rows)

    theta0 = np.concatenate([log_cholesky_from_inertial(b) for b in urdf.reshape(-1, 10)])
    x0 = np.concatenate([theta0, rigid.model.full[friction_at]])
    history: list[dict[str, Any]] = []
    best = levenberg_marquardt(x0, evaluate, val_rmse, mask=slice(None), iterations=settings.stage1_iterations,
                               settings=replace(settings, patience=max(settings.patience, 1)), history=history,
                               stage=1, log=None)
    model = FairModel("id_rigid_fair_consistent", n, full_of(best["x"]), rigid.model.epsilon, info={
        "log_cholesky": best["x"][:10 * n].tolist(), "weighted": rigid.weighted, "val_rmse": best["val"], "iterations": len(history),
        "selected_iteration": best["iteration"],
        "optimiser": "Levenberg-Marquardt, log-Cholesky inertials (Rucker & Wensing 2022), from the URDF"})
    if log:
        log(f"  stage 1 (ID-rigid-fair, physically consistent): val {best['val']:.4f} after {len(history)} "
            f"iterations (selected {best['iteration']})")
    return model, best["x"], history


# ---------------------------------------------------------------------------------------------- the batch filter


class BatchFilter:
    """``filter_sequence`` (full coupling, trapezoidal) over ``B`` equal-length sequences, ``C`` inputs each.

    ``inertia`` ``[B, L, n, n]``, ``stiffness`` ``[n]``, ``damping`` ``[B, n]`` (one ``M_bar`` per
    sequence); torch tensors, float64.  :meth:`run` maps inputs ``U [B, L, n, C]`` to
    ``(delta, delta_dot)`` on rows ``keep_from:``.
    """

    def __init__(self, inertia, stiffness, damping, step: float) -> None:
        import torch

        self.torch = torch
        self.h = float(step)
        self.K = stiffness
        self.D = damping
        h = self.h
        diag = torch.diag_embed(0.5 * h * damping + 0.25 * h * h * stiffness[None, :])      # [B, n, n]
        self.S_inv = torch.linalg.inv(inertia[:, 1:] + diag[:, None])                      # [B, L-1, n, n]

    def run(self, U, keep_from: int = 0, accelerations: bool = False):
        """``(delta, delta_dot)`` on rows ``keep_from:``, and ``delta''`` on every row with ``accelerations``."""
        h = self.h
        B, L, n, C = U.shape
        K = self.K[None, :, None]
        D = self.D[:, :, None]
        delta = U[:, 0] / K
        velocity = self.torch.zeros_like(delta)
        acceleration = self.torch.zeros_like(delta)
        out_d = U.new_empty((B, L - keep_from, n, C))
        out_v = U.new_empty((B, L - keep_from, n, C))
        out_a = U.new_zeros((B, L, n, C)) if accelerations else None
        if keep_from == 0:
            out_d[:, 0], out_v[:, 0] = delta, velocity
        q = 0.25 * h * h
        for k in range(L - 1):
            pv = velocity + 0.5 * h * acceleration
            pd = delta + h * velocity + q * acceleration
            acceleration = self.S_inv[:, k] @ (U[:, k + 1] - D * pv - K * pd)
            velocity = pv + 0.5 * h * acceleration
            delta = pd + q * acceleration
            if k + 1 >= keep_from:
                out_d[:, k + 1 - keep_from], out_v[:, k + 1 - keep_from] = delta, velocity
            if accelerations:
                out_a[:, k + 1] = acceleration
        return (out_d, out_v, out_a) if accelerations else (out_d, out_v)


# ------------------------------------------------------------------------------------------------- ID-2M-fair


class TwoMassProblem:
    """The two-mass output-error problem on the train windows and the val bags, on one torch device.

    ``x = [theta (10 n, log-Cholesky), F_v, F_c (n each), kappa, eta (n each)]``.  The Jacobian is
    exact: besides ``tau_r``, ``K`` and ``D``, the inertials act through ``M_hat`` in the recursion
    (driven by ``-(dM/dpi_c) delta''``, from the mass regressor) and through ``M_bar`` in ``D``.
    """

    def __init__(self, train: Sequence[BagRows], val: Sequence[BagRows], rigid_model: tuple,
                 priors: FairPriors, scale: np.ndarray, *, epsilon: float, weighted: bool, device: str,
                 chunk: int = 64, mass_regressors: list[np.ndarray] | None = None) -> None:
        import torch

        self.torch = torch
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else
                                   ("cpu" if device == "auto" else device))
        self.rigid_model = rigid_model
        self.priors = priors
        self.n = n = train[0].theta.shape[1]
        self.epsilon = float(epsilon)
        self.chunk = int(chunk)
        self.layout = FairLayout(n)
        self.inertial_at = self.layout.inertial_slice()
        self.friction_at = np.r_[self.layout.friction_slices()[0], self.layout.friction_slices()[1]]
        self.weights = self._tensor(scale if weighted else np.ones(n))
        self.mass_reg = mass_regressors if mass_regressors is not None else [
            mass_regressor(rigid_model, rows.theta) for rows in train]
        U, targets, mbar, self.train_index, self.loss_from = [], [], [], [], None
        self.train_theta = [rows.theta for rows in train]
        for b, rows in enumerate(train):
            W = rows.rows(self.layout, self.epsilon)
            diagonal = np.einsum("tjjc->tjc", self.mass_reg[b], dtype=np.float64)
            for start, stop in sequence_windows(rows.bag.t, priors):
                keep = head_mask(rows.bag.t[start:stop], priors.burn_in_s)
                first = int(np.argmax(keep))
                if not keep[first:].all() or self.loss_from not in (None, first):
                    raise ValueError("every window's loss rows must be its tail, of one length")
                self.loss_from = first
                U.append(W[start:stop])
                targets.append(rows.bag.tau_J[start:stop][first:])
                mbar.append(diagonal[start:stop].mean(axis=0))
                self.train_index.append((b, start, stop))
        self.U = self._tensor(np.stack(U))                    # [B, L, n, 12 n]
        self.target = self._tensor(np.stack(targets))         # [B, Lk, n]
        self.mbar_reg = self._tensor(np.stack(mbar))          # [B, n, 10 n]: d M_bar_j / d pi
        self.val_theta = [rows.theta for rows in val]
        if len({len(rows.theta) for rows in val}) != 1:
            raise ValueError("validation bags of unequal length")
        self.val_keep = int(np.argmax(head_mask(val[0].bag.t, priors.burn_in_s)))
        if not all(head_mask(rows.bag.t, priors.burn_in_s)[self.val_keep:].all() for rows in val):
            raise ValueError("validation bags with different burn-in rows")
        self.U_val = self._tensor(np.stack([rows.rows(self.layout, self.epsilon) for rows in val]))
        self.target_val = self._tensor(np.stack([rows.bag.tau_J[self.val_keep:] for rows in val]))
        self._inertia_key = None
        self.non_pd = 0
        self.counts = {"windows": int(self.U.shape[0]), "window_length": int(self.U.shape[1]),
                       "loss_rows": int(self.target.shape[0] * self.target.shape[1]), "val_bags": len(val),
                       "val_loss_rows": int(self.target_val.shape[0] * self.target_val.shape[1])}

    def _tensor(self, x):
        return self.torch.as_tensor(np.ascontiguousarray(x), dtype=self.torch.float64, device=self.device)

    # ---- parameters

    def full_of(self, x: np.ndarray) -> np.ndarray:
        n = self.n
        full = np.zeros(self.layout.size)
        full[self.inertial_at] = consistent_inertials(x[:10 * n])
        full[self.friction_at] = x[10 * n:12 * n]
        return full

    def transmission(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = self.n
        return self.priors.stiffness(x[12 * n:13 * n]), self.priors.zeta(x[13 * n:14 * n])

    def model(self, x: np.ndarray, name: str = "id_2m_fair") -> FairModel:
        K, zeta = self.transmission(x)
        return FairModel(name, self.n, self.full_of(x), self.epsilon, K, zeta,
                         info={"log_cholesky": x[:10 * self.n].tolist()})

    def inertias(self, x: np.ndarray):
        """``M_hat`` on the train windows and the val bags (cached on the inertials)."""
        theta = np.asarray(x[:10 * self.n])
        key = theta.tobytes()
        if key != self._inertia_key:
            dyn = LinkDynamics(self.rigid_model, consistent_inertials(theta), consistent=True)
            per_bag = [dyn.inertia(q) for q in self.train_theta]
            val = np.stack([dyn.inertia(q) for q in self.val_theta])
            self.non_pd = int(sum((~_pd_mask(M)).sum() for M in per_bag) + (~_pd_mask(val)).sum())
            self.M = self._tensor(np.stack([per_bag[b][start:stop] for b, start, stop in self.train_index]))
            self.M_val = self._tensor(val)
            self._inertia_key = key
        return self.M, self.M_val

    # ---- residuals and Jacobian

    def evaluate(self, x: np.ndarray, *, jacobian: str | bool | None = None):
        """F6 cost at ``x``; with ``jacobian`` (``"transmission"`` or ``"all"``) also ``(H, g) = (J^T J, J^T e) / N``."""
        torch, n = self.torch, self.n
        K_np, zeta_np = self.transmission(x)
        M, _ = self.inertias(x)
        K, zeta = self._tensor(K_np), self._tensor(zeta_np)
        M_bar = torch.diagonal(M, dim1=-2, dim2=-1).mean(dim=1)                              # [B, n]
        D = 2.0 * zeta * torch.sqrt(K * M_bar)
        full = self._tensor(self.full_of(x))
        lf, everything = self.loss_from, jacobian == "all"
        if jacobian:
            eye = torch.eye(n, dtype=torch.float64, device=self.device)
            sig_k, sig_z = _sigmoid(x[12 * n:13 * n]), _sigmoid(x[13 * n:14 * n])
            dK_dkappa = self._tensor((self.priors.k_high - self.priors.k_low) * sig_k * (1 - sig_k))
            dz_deta = self._tensor((self.priors.zeta_high - self.priors.zeta_low) * sig_z * (1 - sig_z))
            width = (12 * n if everything else 0) + 2 * n
            H = torch.zeros((width, width), dtype=torch.float64, device=self.device)
            g = torch.zeros(width, dtype=torch.float64, device=self.device)
        if everything:
            G = np.zeros((self.layout.size, 12 * n))
            dpi = consistent_jacobian(x[:10 * n])
            G[np.ix_(self.inertial_at, np.arange(10 * n))] = dpi
            G[self.friction_at, 10 * n + np.arange(2 * n)] = 1.0
            G, dpi = self._tensor(G), self._tensor(dpi)
        total, N = 0.0, 0
        for lo in range(0, self.U.shape[0], self.chunk):
            c = slice(lo, lo + self.chunk)
            Kc, Dc, Dw = K[None, None, :, None], D[c][:, None, :, None], D[c]
            filt = BatchFilter(M[c], K, Dw, self.priors.step)
            out = filt.run((self.U[c] @ full)[..., None], accelerations=everything)
            delta, velocity = out[0][..., 0], out[1][..., 0]                                   # [b, L, n]
            e = (K * delta[:, lf:] + Dw[:, None] * velocity[:, lf:] - self.target[c]) * self.weights
            total += float((e ** 2).sum())
            N += e.numel()
            if not jacobian:
                continue
            # Sensitivities: the recursion driven by -E_j delta (stiffness) and -E_j delta_dot (damping).
            sd, sv = filt.run(torch.cat([-delta[..., None] * eye, -velocity[..., None] * eye], dim=-1), keep_from=lf)
            dK = Kc * sd[..., :n] + Dc * sv[..., :n] + delta[:, lf:, :, None] * eye        # d tau_s / d K_j, D fixed
            dD = Kc * sd[..., n:] + Dc * sv[..., n:] + velocity[:, lf:, :, None] * eye     # d tau_s / d D_j
            blocks = [(dK + dD * (Dw / (2.0 * K))[:, None, None, :]) * dK_dkappa,
                      dD * (Dw / zeta * dz_deta)[:, None, None, :]]
            if everything:
                fd, fv = filt.run(self.U[c], keep_from=lf)                                    # through tau_r
                J_x = (Kc * fd + Dc * fv) @ G
                # Through M_hat in the recursion: input -(dM/dpi_c) delta'' at every step.
                acceleration = out[2][..., 0].cpu()
                U3 = torch.stack([torch.einsum("li,lijc->ljc", acceleration[w].float(), torch.from_numpy(
                    self.mass_reg[b][start:stop])) for w, (b, start, stop) in
                    enumerate(self.train_index[c])]).to(device=self.device, dtype=torch.float64)
                md, mv = filt.run(-U3, keep_from=lf)
                J_pi = Kc * md + Dc * mv
                # Through M_bar in D = 2 zeta sqrt(K M_bar).
                J_pi = J_pi + torch.einsum("blij,bjc->blic", dD, (Dw / (2.0 * M_bar[c]))[..., None] * self.mbar_reg[c])
                J_x[..., :10 * n] += J_pi @ dpi
                blocks.insert(0, J_x)
            J = (torch.cat(blocks, dim=-1) * self.weights[None, None, :, None]).reshape(-1, width)
            H += J.T @ J
            g += J.T @ e.reshape(-1)
        cost = total / N
        if not jacobian:
            return cost
        return cost, (H / N).cpu().numpy(), (g / N).cpu().numpy()

    def val_rmse(self, x: np.ndarray) -> float:
        """The two-mass LNNs' selection statistic: whole val bags from their start, rows after the burn-in."""
        torch = self.torch
        K_np, zeta_np = self.transmission(x)
        _, M = self.inertias(x)
        K, zeta = self._tensor(K_np), self._tensor(zeta_np)
        D = 2.0 * zeta * torch.sqrt(K * torch.diagonal(M, dim1=-2, dim2=-1).mean(dim=1))
        tau_r = (self.U_val @ self._tensor(self.full_of(x)))[..., None]
        delta, velocity = BatchFilter(M, K, D, self.priors.step).run(tau_r, keep_from=self.val_keep)
        tau_s = K * delta[..., 0] + D[:, None] * velocity[..., 0]
        return float(torch.sqrt(((tau_s - self.target_val) ** 2).mean()))


@dataclass
class TwoMassFit:
    model: FairModel
    stage1: FairModel
    x: np.ndarray
    val_rmse: float
    cost: float
    history: list[dict[str, Any]]
    counts: dict[str, int]


def fit_two_mass(rigid: RigidFit, train: Sequence[BagRows], val: Sequence[BagRows], rigid_model: tuple,
                 priors: FairPriors, scale: np.ndarray, *, settings: OutputErrorSettings = OutputErrorSettings(),
                 mass_regressors: list[np.ndarray] | None = None,
                 log: Callable[[str], None] | None = print) -> TwoMassFit:
    """Plan Sec 3.2: (1) pi, F from ID-rigid-fair, K, zeta at the F4 init; (2) K, zeta only; (3) all jointly.

    Stage 1 is :func:`fit_rigid_consistent` (the in-class fix for a non-positive-definite ``M_hat``).
    ``mass_regressors`` (:func:`mass_regressor` of each train bag) may be shared between fits.
    """
    n = rigid.model.n
    stage1, x1, history = fit_rigid_consistent(rigid, val, rigid_model, settings, log)
    problem = TwoMassProblem(train, val, rigid_model, priors, scale, epsilon=rigid.model.epsilon,
                             weighted=rigid.weighted, device=settings.device, chunk=settings.chunk,
                             mass_regressors=mass_regressors)
    x = np.concatenate([x1, *priors.logits(priors.k_init, priors.zeta_init)])
    best = {"x": x.copy(), "val": problem.val_rmse(x), "stage": 1, "iteration": 0}
    cost = problem.evaluate(x)
    history.append({"stage": 1, "iteration": len(history), "cost_two_mass": cost, "val_rmse_two_mass": best["val"]})
    if log:
        log(f"  stage 1 through the filter (K, zeta at the F4 init): cost {cost:.6g}, val {best['val']:.4f}")

    def describe(xk: np.ndarray) -> str:
        K, zeta = problem.transmission(xk)
        return f"K {np.round(K).astype(int).tolist()} zeta {np.round(zeta, 3).tolist()}"

    for stage, iterations, mask, kind in ((2, settings.stage2_iterations, slice(12 * n, None), "transmission"),
                                          (3, settings.stage3_iterations, slice(None), "all")):
        best = levenberg_marquardt(best["x"], lambda xk, jacobian=False, kind=kind: problem.evaluate(
            xk, jacobian=kind if jacobian else None), problem.val_rmse, mask=mask, iterations=iterations,
            settings=settings, history=history, stage=stage, log=log, describe=describe, best=dict(best))
    model = problem.model(best["x"])
    final_cost = problem.evaluate(best["x"])
    problem.inertias(best["x"])
    model.info.update(weighted=rigid.weighted, val_rmse=best["val"], cost=final_cost, selected_stage=best["stage"],
                      selected_iteration=best["iteration"], non_pd_train_val=problem.non_pd,
                      stage1_val_rmse=stage1.info["val_rmse"],
                      optimiser="Levenberg-Marquardt (output error); log-Cholesky inertials; Jacobian by the "
                                "batched filter with M_hat held constant", counts=problem.counts,
                      device=str(problem.device))
    return TwoMassFit(model, stage1, best["x"], best["val"], final_cost, history, problem.counts)
