"""Optimizer interface (ported from ``legacy/.../optimizers/base.py``).

Every backend works in the registry's normalised coordinates z in [-1, 1]^n
and returns ``(best_z, history)``, history being ``[(z, loss), ...]`` in
evaluation order.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

History = list[tuple[np.ndarray, float]]


class OptimizerUnavailable(ImportError):
    """The backend's optional dependency group is not installed."""

    def __init__(self, backend: str, packages: str) -> None:
        super().__init__(f"the {backend} optimizer needs {packages}: install the calibration group with "
                         f"`uv sync --group calibration`")


class Optimizer(ABC):
    """Minimise a scalar objective over [-1, 1]^n.

    ``requires_residuals`` backends (least squares) call ``objective`` for a
    1-D residual vector instead and minimise ``0.5 * ||r||^2``.
    """

    requires_residuals: bool = False

    @classmethod
    def is_available(cls) -> bool:
        return True

    @abstractmethod
    def minimize(
        self,
        objective: Callable[[np.ndarray], float],
        bounds: list[tuple[float, float]],
        x0: np.ndarray | None = None,
        *,
        max_evals: int = 200,
        verbose: bool = False,
    ) -> tuple[np.ndarray, History]:
        """Run the optimisation within ``max_evals`` objective evaluations.

        ``bounds`` are the physical bounds, one pair per coordinate; they set
        the dimension. Backends search the normalised box [-1, 1]^n.
        """

    @staticmethod
    def _norm_bounds(n: int) -> list[tuple[float, float]]:
        return [(-1.0, 1.0)] * n


def best_of(history: History, fallback: np.ndarray) -> np.ndarray:
    return np.asarray(min(history, key=lambda item: item[1])[0]) if history else np.asarray(fallback)
