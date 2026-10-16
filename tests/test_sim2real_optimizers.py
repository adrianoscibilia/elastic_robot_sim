"""S2R_01 T-0.2 / T-0.3: optimizer backends."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.path.join(_REPO, "src"))

from elastic_sim.sim2real.config import OptimizerSettings
from elastic_sim.sim2real.optimizers import (BACKENDS, BayesianOptimizer, CMAOptimizer, LeastSquaresOptimizer,
                                             RandomOptimizer, SkrlOptimizer, make_optimizer)

TARGET = np.array([0.3, -0.2, 0.5, 0.0, -0.6])
BOUNDS = [(0.0, 1.0)] * len(TARGET)


def _quadratic(z: np.ndarray) -> float:
    return float(np.sum((np.asarray(z) - TARGET) ** 2))


def _skip_unless(backend) -> None:
    if not backend.is_available():
        pytest.skip(f"{backend.__name__} needs the calibration group: `uv sync --group calibration`")


@pytest.mark.parametrize("budget", [1, 7, 23, 50])
def test_cma_budget_is_exact(budget):
    _skip_unless(CMAOptimizer)
    best, history = CMAOptimizer(seed=3, tolfun=0.0, tolx=0.0).minimize(_quadratic, BOUNDS, max_evals=budget)
    assert len(history) == budget
    assert _quadratic(best) == min(loss for _, loss in history)


def test_cma_is_seeded_and_converges():
    _skip_unless(CMAOptimizer)
    first = CMAOptimizer(seed=5).minimize(_quadratic, BOUNDS, max_evals=300)
    second = CMAOptimizer(seed=5).minimize(_quadratic, BOUNDS, max_evals=300)
    np.testing.assert_array_equal(first[0], second[0])
    assert _quadratic(first[0]) < 1e-4


def test_random_is_seeded_uniform_and_exact():
    a_best, a = RandomOptimizer(seed=11).minimize(_quadratic, BOUNDS, max_evals=40)
    _, b = RandomOptimizer(seed=11).minimize(_quadratic, BOUNDS, max_evals=40)
    _, c = RandomOptimizer(seed=12).minimize(_quadratic, BOUNDS, max_evals=40)
    assert len(a) == 40
    np.testing.assert_array_equal(np.array([z for z, _ in a]), np.array([z for z, _ in b]))
    assert not np.array_equal(a[0][0], c[0][0])
    samples = np.array([z for z, _ in a])
    assert samples.min() >= -1.0 and samples.max() <= 1.0
    assert _quadratic(a_best) == min(loss for _, loss in a)


def test_least_squares_solves_a_linear_problem():
    rng = np.random.default_rng(0)
    design = rng.normal(size=(30, 5))
    observed = design @ TARGET
    optimizer = LeastSquaresOptimizer()
    best, history = optimizer.minimize(lambda z: design @ z - observed, BOUNDS, max_evals=10)
    np.testing.assert_allclose(best, TARGET, atol=1e-8)
    assert optimizer.jacobian_evaluations <= 10


def test_least_squares_budget_counts_jacobians_and_respects_bounds():
    calls = []

    def jacobian(z):
        calls.append(z)
        return np.diag(np.cos(z))

    optimizer = LeastSquaresOptimizer(jacobian=jacobian)
    best, _ = optimizer.minimize(lambda z: np.sin(z) - np.sin(np.full(5, 2.0)), BOUNDS, max_evals=3)
    assert len(calls) == optimizer.jacobian_evaluations <= 3
    assert np.all(np.abs(best) <= 1.0)


def test_bo_minimises_within_budget():
    _skip_unless(BayesianOptimizer)
    best, history = BayesianOptimizer(n_initial_points=8, seed=1).minimize(_quadratic, BOUNDS, np.zeros(5),
                                                                          max_evals=25)
    assert len(history) == 25
    assert _quadratic(best) < _quadratic(np.zeros(5))


@pytest.mark.slow
def test_skrl_ppo_runs_its_exact_budget():
    _skip_unless(SkrlOptimizer)
    best, history = SkrlOptimizer(seed=1, rollouts=8).minimize(_quadratic, BOUNDS, max_evals=32)
    assert len(history) == 32
    assert np.all(np.abs(best) <= 1.0)


def test_unavailable_backend_raises_a_clear_message(monkeypatch):
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name == "skrl" else real(name, *a))
    assert not SkrlOptimizer.is_available()
    with pytest.raises(ImportError, match="uv sync --group calibration"):
        SkrlOptimizer().minimize(_quadratic, BOUNDS, max_evals=2)


def test_factory_builds_every_method():
    for method in BACKENDS:
        optimizer = make_optimizer(OptimizerSettings(method=method, seed=7))
        assert isinstance(optimizer, BACKENDS[method])
