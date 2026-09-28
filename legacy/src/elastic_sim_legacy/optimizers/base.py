"""Abstract Optimizer interface.

All backends implement minimize(objective, bounds, x0) and return
(best_theta, history).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np


class Optimizer(ABC):
    """Minimise a scalar objective over a bounded parameter space.

    The parameter space is always in normalised form ([-1, 1]^n) as produced
    by RobotParams.normalize() / denormalize().
    """

    @abstractmethod
    def minimize(
        self,
        objective: Callable[[np.ndarray], float],
        bounds: list[tuple[float, float]],
        x0: np.ndarray | None = None,
        *,
        max_evals: int = 200,
        verbose: bool = False,
    ) -> tuple[np.ndarray, list[tuple[np.ndarray, float]]]:
        """Run the optimisation.

        Args:
            objective:  Callable theta -> float; theta is in [-1, 1]^n.
                        *bounds* still refer to the underlying space and are
                        provided for backends that need them (e.g. BO).
            bounds:     List of (lo, hi) in the *physical* (not normalised)
                        space. Backends that operate in the normalised space
                        should transform bounds to [-1, 1] internally.
            x0:         Optional initial point in normalised space.
            max_evals:  Budget (number of objective evaluations).
            verbose:    Print progress if True.

        Returns:
            (best_theta, history)
            best_theta: 1-D ndarray, the best normalised theta found.
            history:    List of (theta, loss) tuples in evaluation order.
        """

    @staticmethod
    def _norm_bounds(n: int) -> list[tuple[float, float]]:
        """Bounds in normalised space: all axes in [-1, 1]."""
        return [(-1.0, 1.0)] * n
