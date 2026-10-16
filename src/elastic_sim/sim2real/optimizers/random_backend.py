"""Random-search baseline: uniform in [-1, 1]^n, seeded."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .base import History, Optimizer, best_of


class RandomOptimizer(Optimizer):
    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed

    def minimize(
        self,
        objective: Callable[[np.ndarray], float],
        bounds: list[tuple[float, float]],
        x0: np.ndarray | None = None,
        *,
        max_evals: int = 200,
        verbose: bool = False,
    ) -> tuple[np.ndarray, History]:
        n = len(bounds)
        rng = np.random.default_rng(self.seed)
        history: History = []
        for _ in range(max_evals):
            z = rng.uniform(-1.0, 1.0, size=n)
            loss = float(objective(z))
            history.append((z, loss))
            if verbose:
                print(f"  random eval {len(history):4d}: loss={loss:.6f}")
        return best_of(history, np.zeros(n) if x0 is None else x0), history
