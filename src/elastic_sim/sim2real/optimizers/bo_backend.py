"""Bayesian optimisation via scikit-optimize (ported from
``legacy/.../optimizers/bo_backend.py``; adds a seed).

S2R_01 T-0.3: scikit-optimize 0.10.2 runs against numpy 2.4 (checked
2026-10-15), so no scipy/sklearn fallback is needed.
"""

from __future__ import annotations

import importlib.util
from typing import Callable

import numpy as np

from .base import History, Optimizer, OptimizerUnavailable, best_of


class BayesianOptimizer(Optimizer):
    def __init__(
        self,
        n_initial_points: int = 10,
        acq_func: str = "EI",
        noise: str | float = "gaussian",
        seed: int | None = None,
    ) -> None:
        self.n_initial_points = n_initial_points
        self.acq_func = acq_func
        self.noise = noise
        self.seed = seed

    @classmethod
    def is_available(cls) -> bool:
        return importlib.util.find_spec("skopt") is not None

    def minimize(
        self,
        objective: Callable[[np.ndarray], float],
        bounds: list[tuple[float, float]],
        x0: np.ndarray | None = None,
        *,
        max_evals: int = 50,
        verbose: bool = False,
    ) -> tuple[np.ndarray, History]:
        try:
            from skopt import gp_minimize
            from skopt.space import Real
        except ImportError as exc:
            raise OptimizerUnavailable("bo", "scikit-optimize") from exc

        n = len(bounds)
        history: History = []

        def _wrapped(x: list) -> float:
            theta = np.asarray(x, dtype=float)
            loss = float(objective(theta))
            history.append((theta.copy(), loss))
            if verbose:
                print(f"  BO eval {len(history):4d}: loss={loss:.6f}")
            return loss

        # An explicit x0 is evaluated inside n_calls; for budgets too short to
        # hold it after the initial points it is dropped so the budget holds.
        initial_points = min(self.n_initial_points, max_evals)
        x0_list = [np.asarray(x0, dtype=float).tolist()] if x0 is not None and max_evals >= initial_points + 1 \
            else None

        gp_minimize(
            _wrapped,
            [Real(-1.0, 1.0) for _ in range(n)],
            n_calls=max_evals,
            n_initial_points=initial_points,
            acq_func=self.acq_func,
            noise=self.noise,
            x0=x0_list,
            random_state=self.seed,
            verbose=verbose,
        )
        return best_of(history, np.zeros(n) if x0 is None else x0), history
