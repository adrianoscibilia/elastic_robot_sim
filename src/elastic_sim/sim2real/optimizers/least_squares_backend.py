"""Bounded least squares (scipy TRF) in normalised coordinates.

``objective`` returns the stacked residual vector; the recorded loss is
``0.5 * ||r||^2``. The budget counts **Jacobian evaluations** (S2R_01 §7):
analytic when ``jacobian`` is given, forward differences in z otherwise (their
probe evaluations are not part of the history).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .base import History, Optimizer, best_of


class _BudgetExhausted(Exception):
    pass


class LeastSquaresOptimizer(Optimizer):
    requires_residuals = True

    def __init__(self, fd_step: float = 1e-6, ftol: float = 1e-10, xtol: float = 1e-10, gtol: float = 1e-10,
                 jacobian: Callable[[np.ndarray], np.ndarray] | None = None) -> None:
        self.fd_step = fd_step
        self.ftol = ftol
        self.xtol = xtol
        self.gtol = gtol
        self.jacobian = jacobian

    def minimize(
        self,
        objective: Callable[[np.ndarray], np.ndarray],
        bounds: list[tuple[float, float]],
        x0: np.ndarray | None = None,
        *,
        max_evals: int = 200,
        verbose: bool = False,
    ) -> tuple[np.ndarray, History]:
        from scipy.optimize import least_squares

        n = len(bounds)
        x0 = np.zeros(n) if x0 is None else np.clip(np.asarray(x0, dtype=float), -1.0, 1.0)
        history: History = []
        cache: dict[bytes, np.ndarray] = {}
        jacobian_calls = 0

        def residuals(z: np.ndarray) -> np.ndarray:
            key = np.asarray(z, dtype=float).tobytes()
            if key not in cache:
                r = np.asarray(objective(np.asarray(z, dtype=float).copy()), dtype=float).ravel()
                cache.clear()
                cache[key] = r
                loss = 0.5 * float(r @ r)
                history.append((np.asarray(z, dtype=float).copy(), loss))
                if verbose:
                    print(f"  LS eval {len(history):4d}: loss={loss:.6f}")
            return cache[key]

        def jacobian(z: np.ndarray) -> np.ndarray:
            nonlocal jacobian_calls
            if jacobian_calls >= max_evals:
                raise _BudgetExhausted
            jacobian_calls += 1
            z = np.asarray(z, dtype=float)
            if self.jacobian is not None:
                return np.asarray(self.jacobian(z.copy()), dtype=float)
            r0 = residuals(z)
            columns = []
            for i in range(n):
                # Step inward at the upper bound so the probe stays feasible.
                step = self.fd_step if z[i] + self.fd_step <= 1.0 else -self.fd_step
                probe = z.copy()
                probe[i] += step
                columns.append((np.asarray(objective(probe), dtype=float).ravel() - r0) / step)
            return np.column_stack(columns)

        try:
            least_squares(residuals, x0, jac=jacobian, bounds=(-np.ones(n), np.ones(n)), method="trf",
                          x_scale=1.0, ftol=self.ftol, xtol=self.xtol, gtol=self.gtol,
                          max_nfev=max(10 * max_evals, 100))
        except _BudgetExhausted:
            pass
        self.jacobian_evaluations = jacobian_calls
        return best_of(history, x0), history
