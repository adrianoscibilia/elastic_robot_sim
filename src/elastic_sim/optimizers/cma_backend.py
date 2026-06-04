"""CMA-ES optimizer backend.

Requires: pip install cma

CMA-ES is the default backend for sim-to-real calibration: it handles
noisy, low-dimensional (~6-7 parameters) black-box objectives well and
typically converges in 100-500 evaluations for this problem size.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .base import Optimizer


class CMAOptimizer(Optimizer):
    """CMA-ES via the `cma` package.

    Args:
        sigma0:   Initial step-size in normalised [-1,1] space (default 0.3).
        popsize:  Population size. If None, uses the CMA default (4+floor(3*ln(n))).
        tolfun:   Convergence tolerance on function value change.
        tolx:     Convergence tolerance on variable change.
    """

    def __init__(
        self,
        sigma0: float = 0.3,
        popsize: int | None = None,
        tolfun: float = 1e-7,
        tolx: float = 1e-7,
    ) -> None:
        self.sigma0 = sigma0
        self.popsize = popsize
        self.tolfun = tolfun
        self.tolx = tolx

    def minimize(
        self,
        objective: Callable[[np.ndarray], float],
        bounds: list[tuple[float, float]],
        x0: np.ndarray | None = None,
        *,
        max_evals: int = 200,
        verbose: bool = False,
    ) -> tuple[np.ndarray, list[tuple[np.ndarray, float]]]:
        try:
            import cma
        except ImportError as exc:
            raise ImportError(
                "CMA-ES requires the 'cma' package: pip install cma"
            ) from exc

        n = len(bounds)
        norm_bounds = self._norm_bounds(n)

        if x0 is None:
            x0 = np.zeros(n)  # centre of normalised space

        history: list[tuple[np.ndarray, float]] = []

        def _wrapped(x: np.ndarray) -> float:
            loss = objective(np.asarray(x))
            history.append((np.asarray(x).copy(), loss))
            if verbose:
                print(f"  CMA eval {len(history):4d}: loss={loss:.6f}")
            return loss

        opts = cma.CMAOptions()
        opts["maxfevals"] = max_evals
        opts["bounds"] = [
            [lo for lo, _ in norm_bounds],
            [hi for _, hi in norm_bounds],
        ]
        opts["tolfun"] = self.tolfun
        opts["tolx"] = self.tolx
        opts["verbose"] = 1 if verbose else -9
        if self.popsize is not None:
            opts["popsize"] = self.popsize

        es = cma.CMAEvolutionStrategy(x0.tolist(), self.sigma0, opts)
        es.optimize(_wrapped)

        best_x = np.asarray(es.result.xbest)
        return best_x, history
