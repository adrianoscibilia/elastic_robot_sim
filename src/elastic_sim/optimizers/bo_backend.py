"""Bayesian optimization backend.

Requires: pip install scikit-optimize

Use this backend when each sim rollout is expensive (e.g. long simulation
time or many rollouts per evaluation). BO builds a surrogate GP model of the
objective and uses it to guide sampling, which can be more sample-efficient
than CMA-ES when the budget is very tight (<50 evaluations).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .base import Optimizer


class BayesianOptimizer(Optimizer):
    """Gaussian-process Bayesian optimisation via scikit-optimize.

    Args:
        n_initial_points: Number of quasi-random initial evaluations before
                          the GP surrogate is used.
        acq_func:         Acquisition function: "EI", "LCB", or "PI".
        noise:            Observation noise level passed to the GP kernel.
                          "gaussian" means auto-estimated.
    """

    def __init__(
        self,
        n_initial_points: int = 10,
        acq_func: str = "EI",
        noise: str | float = "gaussian",
    ) -> None:
        self.n_initial_points = n_initial_points
        self.acq_func = acq_func
        self.noise = noise

    def minimize(
        self,
        objective: Callable[[np.ndarray], float],
        bounds: list[tuple[float, float]],
        x0: np.ndarray | None = None,
        *,
        max_evals: int = 50,
        verbose: bool = False,
    ) -> tuple[np.ndarray, list[tuple[np.ndarray, float]]]:
        try:
            from skopt import gp_minimize
            from skopt.space import Real
        except ImportError as exc:
            raise ImportError(
                "Bayesian optimizer requires scikit-optimize: pip install scikit-optimize"
            ) from exc

        n = len(bounds)
        norm_bounds = [Real(-1.0, 1.0) for _ in range(n)]

        history: list[tuple[np.ndarray, float]] = []

        def _wrapped(x: list) -> float:
            theta = np.asarray(x)
            loss = objective(theta)
            history.append((theta.copy(), loss))
            if verbose:
                print(f"  BO eval {len(history):4d}: loss={loss:.6f}")
            return float(loss)

        x0_list = [x0.tolist()] if x0 is not None else None

        result = gp_minimize(
            _wrapped,
            norm_bounds,
            n_calls=max_evals,
            n_initial_points=self.n_initial_points,
            acq_func=self.acq_func,
            noise=self.noise,
            x0=x0_list,
            verbose=verbose,
        )

        best_x = np.asarray(result.x)
        return best_x, history
