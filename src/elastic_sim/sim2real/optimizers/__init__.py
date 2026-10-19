"""Optimizer backends behind one interface (S2R_01 T-0.2, T-4.4).

Backends import their optional dependency (``uv sync --group calibration``)
only when they run; ``is_available()`` checks without importing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import History, Optimizer, OptimizerUnavailable
from .bo_backend import BayesianOptimizer
from .cma_backend import CMAOptimizer
from .least_squares_backend import LeastSquaresOptimizer
from .random_backend import RandomOptimizer
from .skrl_backend import SkrlOptimizer

if TYPE_CHECKING:
    from ..config import OptimizerSettings

BACKENDS: dict[str, type[Optimizer]] = {
    "least_squares": LeastSquaresOptimizer,
    "cma": CMAOptimizer,
    "bo": BayesianOptimizer,
    "skrl_ppo": SkrlOptimizer,
    "random": RandomOptimizer,
}


def make_optimizer(settings: "OptimizerSettings") -> Optimizer:
    """The backend ``settings.method`` names, configured from its block and seed."""
    method = settings.method
    if method == "least_squares":
        return LeastSquaresOptimizer()
    if method == "cma":
        return CMAOptimizer(sigma0=float(settings.cma["sigma0"]), popsize=settings.cma.get("popsize"),
                            seed=settings.seed)
    if method == "bo":
        return BayesianOptimizer(n_initial_points=int(settings.bo["n_initial_points"]),
                                 acq_func=str(settings.bo["acq_func"]), seed=settings.seed)
    if method == "skrl_ppo":
        return SkrlOptimizer(rollouts=int(settings.skrl_ppo["rollouts"]),
                             learning_epochs=int(settings.skrl_ppo["learning_epochs"]),
                             device=str(settings.skrl_ppo["device"]), seed=settings.seed)
    if method == "random":
        return RandomOptimizer(seed=settings.seed)
    raise ValueError(f"unknown optimizer method {method!r}")


__all__ = ["BACKENDS", "BayesianOptimizer", "CMAOptimizer", "History", "LeastSquaresOptimizer", "Optimizer",
           "OptimizerUnavailable", "RandomOptimizer", "SkrlOptimizer", "make_optimizer"]
