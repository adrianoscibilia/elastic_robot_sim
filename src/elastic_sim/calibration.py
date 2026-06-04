"""SimCalibrationProblem: maps a parameter vector theta to an aggregate fidelity loss.

Usage::

    problem = SimCalibrationProblem(real_rollouts, weights={"position": 1.0})
    loss = problem.loss(theta)  # theta is the normalised parameter vector

The model is built once on the first call; subsequent calls use
apply_params_inplace for speed (falls back to rebuild if in-place mutation
fails, e.g. under the MuJoCo solver).
"""

from __future__ import annotations

import numpy as np

from .compare import compare
from .params import RobotParams
from .rollout import RolloutResult
from .trajectory import Trajectory, _trajectory_from_config, TrajectoryConfig


class SimCalibrationProblem:
    """Encapsulates the sim-vs-real loss function for a set of recorded rollouts.

    Args:
        real_rollouts:  List of (RolloutResult, TrajectoryConfig) pairs from the
                        real robot (Phase 1 data).
        weights:        Passed to compare(); controls position/velocity/force
                        trade-off in the loss.
        noise:          Whether to add sensor noise during sim rollouts.
                        Default False for calibration (noise masks the true
                        parameter sensitivity).
        cut_off_time:   Seconds of initial transient to skip in comparison.
        time_step:      Simulation integration step (s).
        rebuild_on_fail: If apply_params_inplace fails, rebuild model entirely.
    """

    def __init__(
        self,
        real_rollouts: list[tuple[RolloutResult, TrajectoryConfig]],
        *,
        weights: dict[str, float] | None = None,
        noise: bool = False,
        cut_off_time: float = 0.0,
        time_step: float = 0.01,
        rebuild_on_fail: bool = True,
    ) -> None:
        if not real_rollouts:
            raise ValueError("real_rollouts must be non-empty.")
        self._real_rollouts = real_rollouts
        self._weights = weights
        self._noise = noise
        self._cut_off_time = cut_off_time
        self._time_step = time_step
        self._rebuild_on_fail = rebuild_on_fail

        self._model = None
        self._dof_index_map: dict | None = None
        self._current_params: RobotParams | None = None

        self._n_evals: int = 0
        self._loss_history: list[tuple[np.ndarray, float]] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_model(self, params: RobotParams) -> tuple:
        from .sim_runner import build_model, apply_params_inplace

        if self._model is None:
            self._model, self._dof_index_map, _ = build_model(params)
            self._current_params = params
            return self._model, self._dof_index_map

        # Try cheap in-place update first
        ok = apply_params_inplace(self._model, params, self._dof_index_map)
        if not ok and self._rebuild_on_fail:
            self._model, self._dof_index_map, _ = build_model(params)
        self._current_params = params
        return self._model, self._dof_index_map

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def loss(self, theta: np.ndarray) -> float:
        """Evaluate the aggregate fidelity loss for normalised param vector theta.

        Steps:
            1. Denormalize theta → RobotParams
            2. Apply params (in-place or rebuild)
            3. For each real rollout, run sim and compare
            4. Return mean loss

        Args:
            theta: 1-D array in [-1, 1]^n (see RobotParams.bounds()).

        Returns:
            Scalar loss (lower = better).
        """
        from .sim_runner import run_rollout

        params = RobotParams.denormalize(
            theta, include_payload=len(theta) > 6
        )
        model, dof_index_map = self._ensure_model(params)

        losses = []
        for real_rollout, traj_config in self._real_rollouts:
            traj = _trajectory_from_config(traj_config)
            sim_rollout = run_rollout(
                model, dof_index_map, traj,
                noise=self._noise,
                cut_off_time=self._cut_off_time,
                time_step=self._time_step,
            )
            result = compare(
                sim_rollout, real_rollout,
                self._weights,
                cut_off_time=self._cut_off_time,
            )
            losses.append(result["metric"])

        mean_loss = float(np.mean(losses))
        self._n_evals += 1
        self._loss_history.append((theta.copy(), mean_loss))
        return mean_loss

    def n_dims(self, include_payload: bool = True) -> int:
        """Dimension of the normalised parameter vector."""
        return len(RobotParams.bounds(include_payload))

    @property
    def loss_history(self) -> list[tuple[np.ndarray, float]]:
        """List of (theta, loss) pairs from all previous loss() calls."""
        return list(self._loss_history)

    @property
    def best(self) -> tuple[np.ndarray | None, float]:
        """Best (theta, loss) seen so far, or (None, inf) if no evals yet."""
        if not self._loss_history:
            return None, float("inf")
        return min(self._loss_history, key=lambda x: x[1])
