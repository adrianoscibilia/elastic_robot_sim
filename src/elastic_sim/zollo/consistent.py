"""Zollo's least squares constrained to physically consistent link inertials (variant A-pc).

`06_variant_A_consistent.md` C2: the cost is Zollo's own, ``||W Pi - T||^2``
on the stacked rows, read from the incremental-QR factor
(``||R_W Pi - z||^2``, :class:`.basis.Accumulator`), over the **full**
parameter vector, subject to

* per link body ``i``: the pseudo-inertia ``J_i(m_i, m_i c_i, I_i) >= eps I``
  (Wensing, Kim & Slotine 2017), which implies ``m_i > 0``, a positive
  definite inertia and the triangle inequalities;
* link viscous and Coulomb friction ``>= 0`` per joint;
* nothing else (motor parameters, K, D are free).

No prior and no tie-break: directions the rows never see stay wherever the
solver leaves them; they change no predicted torque and no ``M(q)``.

:func:`solve_consistent` is the convex SDP (cvxpy, Clarabel by default).
:func:`solve_log_cholesky` is the cross-check: the same cost and feasible
set by the log-Cholesky parametrisation (Rucker & Wensing 2022,
:func:`.fair.inertial_from_log_cholesky`) shifted by ``eps I``, and
Levenberg-Marquardt (MINPACK, :func:`scipy.optimize.least_squares`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .basis import Accumulator, Base, solve
from .regressor import Layout

LINK_FRICTION_GROUPS = ("link_viscous", "link_coulomb")


def pseudo_inertia(pi: np.ndarray) -> np.ndarray:
    """``J = [[Sigma, h], [h^T, m]]``, ``Sigma = tr(I)/2 1 - I``, of one body's 10 Pinocchio-order parameters."""
    m, hx, hy, hz, ixx, ixy, iyy, ixz, iyz, izz = np.asarray(pi, dtype=float)
    inertia = np.asarray([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
    J = np.zeros((4, 4))
    J[:3, :3] = 0.5 * np.trace(inertia) * np.eye(3) - inertia
    J[:3, 3] = J[3, :3] = (hx, hy, hz)
    J[3, 3] = m
    return J


# J is linear in pi: J(pi) = sum_k pi_k E_k.
PSEUDO_INERTIA_BASIS = np.stack([pseudo_inertia(e) for e in np.eye(10)])


def epsilon_inertial(epsilon: float) -> np.ndarray:
    """The body whose pseudo-inertia is ``eps I``: ``m = eps``, ``h = 0``, ``I = 2 eps 1``."""
    return np.asarray([epsilon, 0.0, 0.0, 0.0, 2 * epsilon, 0.0, 2 * epsilon, 0.0, 0.0, 2 * epsilon])


def body_eigenvalues(inertial: np.ndarray) -> np.ndarray:
    """``[n, 4]`` eigenvalues of every body's pseudo-inertia (ascending)."""
    return np.stack([np.linalg.eigvalsh(pseudo_inertia(b)) for b in np.asarray(inertial).reshape(-1, 10)])


def friction_positions(layout: Layout) -> np.ndarray:
    return np.asarray([i for i, c in enumerate(layout.columns) if c.group in LINK_FRICTION_GROUPS], dtype=int)


def _scales(acc: Accumulator) -> np.ndarray:
    norms = np.linalg.norm(acc.regressor, axis=0)
    return np.where(norms > 0.0, norms, 1.0)


def unconstrained_full(acc: Accumulator, base: Base, size: int) -> np.ndarray:
    """The unconstrained LS optimum as a full vector (dependent and zero columns 0, as :meth:`Base.full_of`)."""
    return base.full_of(solve(acc, base).base, size)


def cost(acc: Accumulator, full: np.ndarray) -> float:
    """Zollo's cost ``||W Pi - T||^2`` (the factor's ``||R_W Pi - z||^2`` plus nothing: ``R`` is square)."""
    e = acc.regressor @ np.asarray(full, dtype=float) - acc.target
    return float(e @ e)


@dataclass
class ConsistentSolution:
    full: np.ndarray
    cost: float
    cost_unconstrained: float
    report: dict[str, Any] = field(default_factory=dict)

    @property
    def increase(self) -> float:
        return self.cost - self.cost_unconstrained


def drift_report(layout: Layout, base: Base, full: np.ndarray) -> dict[str, Any]:
    """Where the solve left what the rows never see: the unidentifiable columns, and every body's mass.

    No minimiser need exist: when a binding body's mass is unidentifiable,
    raising it relaxes ``J_i >= eps I`` at no cost, and the infimum is reached
    only as it grows without bound.  These values show how far it went.
    """
    full = np.asarray(full, dtype=float)
    masses = full[layout.inertial_slice()][0::10]
    return {"unidentifiable": {layout.columns[i].name: float(full[i]) for i in base.zero},
            "unidentifiable_max_abs": float(np.abs(full[base.zero]).max(initial=0.0)),
            "masses": masses.tolist()}


def constraint_report(layout: Layout, full: np.ndarray, epsilon: float, *, duals: list | None = None,
                      friction_duals: np.ndarray | None = None, active_tolerance: float = 1e-2) -> dict[str, Any]:
    """Per body ``eig(J_i)`` and whether ``J_i >= eps I`` is active; link friction and whether it sits at 0.

    Active: the smallest eigenvalue within ``active_tolerance`` (relative) of
    ``eps`` -- or, with ``duals``, a dual matrix of non-negligible norm.
    """
    inertial = np.asarray(full)[layout.inertial_slice()]
    eig = body_eigenvalues(inertial)
    bodies = []
    for b in range(layout.n):
        entry = {"body": b + 1, "eig_J": eig[b].tolist(), "min_eig_minus_eps": float(eig[b, 0] - epsilon),
                 "active": bool(eig[b, 0] <= epsilon * (1.0 + active_tolerance))}
        if duals is not None:
            entry["dual_norm"] = float(np.linalg.norm(duals[b]))
        bodies.append(entry)
    friction = []
    positions = friction_positions(layout)
    scale = max(float(np.abs(np.asarray(full)[positions]).max(initial=0.0)), 1e-12)
    for k, p in enumerate(positions):
        value = float(full[p])
        entry = {"column": layout.columns[p].name, "value": value, "active": bool(value <= 1e-6 * scale)}
        if friction_duals is not None:
            entry["dual"] = float(friction_duals[k])
        friction.append(entry)
    return {"epsilon": epsilon, "bodies": bodies, "friction": friction,
            "active_bodies": [e["body"] for e in bodies if e["active"]],
            "active_friction": [e["column"] for e in friction if e["active"]]}


def solve_consistent(acc: Accumulator, base: Base, layout: Layout, *, epsilon: float = 1e-6,
                     solver: str = "CLARABEL", verbose: bool = False) -> ConsistentSolution:
    """The SDP: ``min ||R_W Pi - z||`` s.t. ``J_i(Pi) >= eps I``, link friction ``>= 0``.

    Solved in the column-scaled variable ``y = s * Pi`` (``s`` the column
    norms of ``R_W``, the scaling of :func:`.basis.solve`), as
    ``min ||A (y - y0)||`` with ``A = R_W / s`` and ``y0`` the unconstrained
    optimum: the same minimiser as Zollo's cost (``||A y - z||^2 = rss0 +
    ||A (y - y0)||^2``), without its constant, and the norm, not its square,
    so the solver's gap tolerance acts on the residual itself.
    """
    import cvxpy as cp

    RW = acc.regressor
    s = _scales(acc)
    A = RW / s[None, :]
    full0 = unconstrained_full(acc, base, layout.size)
    y0 = s * full0
    y = cp.Variable(layout.size)
    pi = cp.multiply(1.0 / s, y)
    inertial = layout.inertial_slice()
    constraints, lmis = [], []
    for b in range(layout.n):
        J = sum(pi[int(inertial[10 * b + k])] * PSEUDO_INERTIA_BASIS[k] for k in range(10))
        S = cp.Variable((4, 4), PSD=True)
        lmis.append(S == J - epsilon * np.eye(4))
        constraints.append(lmis[-1])
    friction = friction_positions(layout)
    if len(friction):
        positivity = pi[friction] >= 0.0
        constraints.append(positivity)
    problem = cp.Problem(cp.Minimize(cp.norm(A @ y - A @ y0, 2)), constraints)
    attempts: list[dict[str, Any]] = []
    kept: dict[str, Any] | None = None             # the best result so far: optimal beats optimal_inaccurate
    for name, options in _attempts(solver):
        try:
            problem.solve(solver=name, verbose=verbose, **options)
        except cp.error.SolverError as error:
            attempts.append({"solver": name, "options": options, "status": f"error: {error}"})
            continue
        attempts.append({"solver": name, "options": options, "status": problem.status,
                         "iterations": problem.solver_stats.num_iters, "solve_time_s": problem.solver_stats.solve_time})
        if y.value is not None and problem.status in ("optimal", "optimal_inaccurate") and (
                kept is None or problem.status == "optimal"):
            kept = {"attempt": len(attempts) - 1, "status": problem.status, "y": np.asarray(y.value, dtype=float),
                    "duals": [np.asarray(c.dual_value, dtype=float) for c in lmis],
                    "friction_duals": np.asarray(positivity.dual_value, dtype=float) if len(friction) else None,
                    "objective": float(problem.value)}
        if kept is not None and kept["status"] == "optimal":
            break
    if kept is None:
        raise RuntimeError(f"consistent LS: no solver succeeded: {attempts}")
    full = kept["y"] / s
    duals, friction_duals = kept["duals"], kept["friction_duals"]
    report = constraint_report(layout, full, epsilon, duals=duals, friction_duals=friction_duals)
    report.update(drift_report(layout, base, full))
    c0 = cost(acc, full0)
    c = cost(acc, full)
    used = attempts[kept["attempt"]]
    report.update({"solver": used["solver"], "status": kept["status"], "options": used["options"],
                   "attempts": attempts, "solver_iterations": used.get("iterations"),
                   "solve_time_s": used.get("solve_time_s"),
                   "cvxpy_version": cp.__version__, "solver_version": _solver_version(used["solver"]),
                   "cost": c, "cost_unconstrained": c0, "cost_increase": c - c0,
                   "cost_increase_relative": (c - c0) / c0 if c0 > 0 else float("nan"),
                   "objective_norm": kept["objective"],
                   "min_eig_J": float(body_eigenvalues(full[inertial]).min()),
                   "min_link_friction": float(full[friction].min()) if len(friction) else None})
    return ConsistentSolution(full, c, c0, report)


def _attempts(solver: str) -> list[tuple[str, dict[str, Any]]]:
    """Solver settings, tightest first; the problem never changes, only how it is solved."""
    tight = {"CLARABEL": {"tol_gap_abs": 1e-10, "tol_gap_rel": 1e-10, "tol_feas": 1e-10, "tol_ktratio": 1e-8,
                          "max_iter": 500},
             "SCS": {"eps_abs": 1e-9, "eps_rel": 1e-9, "max_iters": 200000}}
    first = solver.upper()
    ladder = [(first, tight.get(first, {})), (first, {})]
    if first != "SCS":
        ladder.append(("SCS", tight["SCS"]))
    return ladder


def _solver_version(solver: str) -> str:
    try:
        if solver.upper() == "CLARABEL":
            import clarabel
            return clarabel.__version__
        if solver.upper() == "SCS":
            import scs
            return scs.__version__
    except (ImportError, AttributeError):
        pass
    return "unknown"


def solve_log_cholesky(acc: Accumulator, base: Base, layout: Layout, rigid_inertial: np.ndarray, *,
                       epsilon: float = 1e-6, max_nfev: int = 20000, start: np.ndarray | None = None
                       ) -> ConsistentSolution:
    """The cross-check: the same cost and feasible set, by parametrisation, with Levenberg-Marquardt.

    Each body ``pi_i = pi(theta_i) + pi_eps`` (log-Cholesky, :func:`.fair.inertial_from_log_cholesky`;
    ``J(pi_eps) = eps I``, so ``J_i = U U^T + eps I``), each link friction
    ``f = phi^2``, every other parameter free.  The residual is
    ``R_W (Pi - Pi0)`` (``Pi0`` the unconstrained optimum, the same minimiser
    as the SDP's).  Starts from ``rigid_inertial`` (the URDF) for the bodies
    and the unconstrained optimum for the rest, unless ``start`` (a full
    vector) is given.
    """
    from scipy.optimize import least_squares

    from .fair import inertial_from_log_cholesky, log_cholesky_from_inertial

    RW = acc.regressor
    n = layout.n
    inertial = layout.inertial_slice()
    friction = friction_positions(layout)
    free = np.setdiff1d(np.arange(layout.size), np.concatenate([inertial, friction]))
    shift = epsilon_inertial(epsilon)
    full0 = unconstrained_full(acc, base, layout.size)
    origin = start if start is not None else full0
    bodies = np.asarray(start[inertial] if start is not None else rigid_inertial, dtype=float).reshape(n, 10)
    theta0 = np.concatenate([log_cholesky_from_inertial(b - shift) for b in bodies])
    phi0 = np.sqrt(np.maximum(origin[friction], 1e-2 if start is None else 1e-12))
    x0 = np.concatenate([theta0, phi0, origin[free]])
    k_theta, k_phi = 10 * n, 10 * n + len(friction)

    def full_of(x: np.ndarray) -> np.ndarray:
        out = np.empty(layout.size)
        out[inertial] = np.concatenate([inertial_from_log_cholesky(b) + shift for b in x[:k_theta].reshape(n, 10)])
        out[friction] = x[k_theta:k_phi] ** 2
        out[free] = x[k_phi:]
        return out

    def residual(x: np.ndarray) -> np.ndarray:
        with np.errstate(over="ignore", invalid="ignore"):
            r = RW @ (full_of(x) - full0)
        return np.where(np.isfinite(r), r, 1e150)          # an overflowing trial step: rejected

    def jacobian(x: np.ndarray) -> np.ndarray:
        G = np.zeros((layout.size, len(x)))
        for b, block in enumerate(x[:k_theta].reshape(n, 10)):
            rows = inertial[10 * b:10 * b + 10]
            for k in range(10):
                step = 1e-6 * max(1.0, abs(block[k]))
                plus, minus = block.copy(), block.copy()
                plus[k] += step
                minus[k] -= step
                G[rows, 10 * b + k] = (inertial_from_log_cholesky(plus) - inertial_from_log_cholesky(minus)) / (2 * step)
        G[friction, np.arange(k_theta, k_phi)] = 2.0 * x[k_theta:k_phi]
        G[free, np.arange(k_phi, len(x))] = 1.0
        return RW @ G

    fit = least_squares(residual, x0, jac=jacobian, method="lm", ftol=1e-15, xtol=1e-15, gtol=1e-15,
                        max_nfev=max_nfev, x_scale="jac")
    full = full_of(fit.x)
    report = constraint_report(layout, full, epsilon)
    report.update(drift_report(layout, base, full))
    c0 = cost(acc, full0)
    c = cost(acc, full)
    report.update({"solver": "Levenberg-Marquardt (scipy least_squares, MINPACK), log-Cholesky + eps I",
                   "status": fit.message, "nfev": int(fit.nfev), "njev": int(fit.njev or 0),
                   "cost": c, "cost_unconstrained": c0, "cost_increase": c - c0,
                   "cost_increase_relative": (c - c0) / c0 if c0 > 0 else float("nan"),
                   "min_eig_J": float(body_eigenvalues(full[inertial]).min()),
                   "start": "given full vector" if start is not None else "URDF inertials, unconstrained rest"})
    return ConsistentSolution(full, c, c0, report)
