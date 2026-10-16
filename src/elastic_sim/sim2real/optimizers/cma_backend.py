"""CMA-ES backend (ported from ``legacy/.../optimizers/cma_backend.py``).

Ask/tell with an exact evaluation budget, as in the legacy backend; adds a
seed so runs repeat.
"""

from __future__ import annotations

import importlib.util
from typing import Callable

import numpy as np

from .base import History, Optimizer, OptimizerUnavailable, best_of


class CMAOptimizer(Optimizer):
    def __init__(
        self,
        sigma0: float = 0.3,
        popsize: int | None = None,
        tolfun: float = 1e-7,
        tolx: float = 1e-7,
        seed: int | None = None,
    ) -> None:
        self.sigma0 = sigma0
        self.popsize = popsize
        self.tolfun = tolfun
        self.tolx = tolx
        self.seed = seed

    @classmethod
    def is_available(cls) -> bool:
        return importlib.util.find_spec("cma") is not None

    def minimize(
        self,
        objective: Callable[[np.ndarray], float],
        bounds: list[tuple[float, float]],
        x0: np.ndarray | None = None,
        *,
        max_evals: int = 200,
        verbose: bool = False,
    ) -> tuple[np.ndarray, History]:
        try:
            import cma
        except ImportError as exc:
            raise OptimizerUnavailable("cma", "the 'cma' package") from exc

        n = len(bounds)
        norm_bounds = self._norm_bounds(n)
        x0 = np.zeros(n) if x0 is None else np.asarray(x0, dtype=float)
        history: History = []

        def _wrapped(x: np.ndarray) -> float:
            loss = float(objective(np.asarray(x)))
            history.append((np.asarray(x).copy(), loss))
            if verbose:
                print(f"  CMA eval {len(history):4d}: loss={loss:.6f}")
            return loss

        opts = cma.CMAOptions()
        opts["maxfevals"] = max_evals
        opts["bounds"] = [[lo for lo, _ in norm_bounds], [hi for _, hi in norm_bounds]]
        opts["tolfun"] = self.tolfun
        opts["tolx"] = self.tolx
        opts["verbose"] = 1 if verbose else -9
        if self.seed is not None:
            # cma treats seed 0 as "seed from the clock".
            opts["seed"] = int(self.seed) or 1
        if self.popsize is not None:
            opts["popsize"] = self.popsize

        es = cma.CMAEvolutionStrategy(x0.tolist(), self.sigma0, opts)
        # ``optimize`` can overshoot a small budget by one population; ask/tell
        # explicitly so every call is counted and max_evals is exact.
        while len(history) < max_evals and not es.stop():
            candidates = es.ask()
            if len(history) + len(candidates) > max_evals:
                # A partial population can't be told back: evaluate it and stop.
                for candidate in candidates[: max_evals - len(history)]:
                    _wrapped(np.asarray(candidate))
                break
            values = [_wrapped(np.asarray(candidate)) for candidate in candidates]
            es.tell(candidates, values)

        return best_of(history, x0), history
