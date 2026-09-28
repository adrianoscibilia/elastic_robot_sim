"""SimCalibrationProblem: maps a parameter vector theta to an aggregate fidelity loss.

Usage::

    # Newton backend (default)
    problem = SimCalibrationProblem(real_rollouts, backend="newton")
    loss = problem.loss(theta)

    # MuJoCo backend
    problem = SimCalibrationProblem(real_rollouts, backend="mujoco")
    loss = problem.loss(theta)

The model is built once on the first call; subsequent calls use
apply_params_inplace for speed.  MuJoCo always supports in-place mutation;
Newton falls back to a full rebuild if in-place fails (e.g. MuJoCo solver).
"""

from __future__ import annotations

import numpy as np

from .compare import compare
from .params import RobotParams
from .rollout import RolloutResult
from .trajectory import _trajectory_from_config, TrajectoryConfig

_SUPPORTED_BACKENDS = ("newton", "mujoco")


class SimCalibrationProblem:
    """Encapsulates the sim-vs-real loss function for a set of recorded rollouts.

    Args:
        real_rollouts:  List of (RolloutResult, TrajectoryConfig) pairs from the
                        real robot (Phase 1 data).
        backend:        Simulator backend: "newton" (default) or "mujoco".
        weights:        Passed to compare(); controls position/velocity/force
                        trade-off in the loss.
        noise:          Whether to add sensor noise during sim rollouts.
                        Default False for calibration (noise masks parameter sensitivity).
        cut_off_time:   Seconds of initial transient to skip in comparison.
        time_step:      Simulation integration step (s).
        rebuild_on_fail: (Newton only) If apply_params_inplace fails, rebuild model.
    """

    def __init__(
        self,
        real_rollouts: list[tuple[RolloutResult, TrajectoryConfig]],
        *,
        backend: str = "newton",
        weights: dict[str, float] | None = None,
        noise: bool = False,
        cut_off_time: float = 0.0,
        time_step: float = 0.01,
        rebuild_on_fail: bool = True,
    ) -> None:
        if not real_rollouts:
            raise ValueError("real_rollouts must be non-empty.")
        if backend not in _SUPPORTED_BACKENDS:
            raise ValueError(f"backend must be one of {_SUPPORTED_BACKENDS}, got '{backend}'.")

        self._real_rollouts = real_rollouts
        self._backend = backend
        self._weights = weights
        self._noise = noise
        self._cut_off_time = cut_off_time
        self._time_step = time_step
        self._rebuild_on_fail = rebuild_on_fail

        # Persistent simulator state (initialised on first loss() call)
        self._model = None
        self._aux = None          # dof_index_map (Newton) or (data, dof_map, act_map) (MuJoCo)
        self._current_params: RobotParams | None = None

        self._n_evals: int = 0
        self._loss_history: list[tuple[np.ndarray, float]] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_model(self, params: RobotParams) -> None:
        if self._backend == "newton":
            self._ensure_newton(params)
        else:
            self._ensure_mujoco(params)

    def _ensure_newton(self, params: RobotParams) -> None:
        from .sim_runner import build_model, apply_params_inplace
        if self._model is None:
            self._model, dof_map, _ = build_model(params)
            self._aux = dof_map
        else:
            ok = apply_params_inplace(self._model, params, self._aux)
            if not ok and self._rebuild_on_fail:
                self._model, self._aux, _ = build_model(params)
        self._current_params = params

    def _ensure_mujoco(self, params: RobotParams) -> None:
        from .mujoco_runner import build_model, apply_params_inplace
        if self._model is None:
            self._model, data, dof_map, act_map = build_model(params, time_step=self._time_step)
            self._aux = (data, dof_map, act_map)
        else:
            data, dof_map, act_map = self._aux
            apply_params_inplace(self._model, data, params)
        self._current_params = params

    def _run_sim(self, traj) -> RolloutResult:
        if self._backend == "newton":
            from .sim_runner import run_rollout
            return run_rollout(
                self._model, self._aux, traj,
                noise=self._noise,
                cut_off_time=self._cut_off_time,
                time_step=self._time_step,
            )
        else:
            from .mujoco_runner import run_rollout
            data, dof_map, act_map = self._aux
            return run_rollout(
                self._model, data, dof_map, act_map, traj,
                noise=self._noise,
                cut_off_time=self._cut_off_time,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def loss(self, theta: np.ndarray) -> float:
        """Evaluate the aggregate fidelity loss for normalised param vector theta.

        Args:
            theta: 1-D array in [-1, 1]^n (see RobotParams.bounds()).

        Returns:
            Scalar loss (lower = better).
        """
        params = RobotParams.denormalize(theta, include_payload=len(theta) > 6)
        self._ensure_model(params)

        losses = []
        for real_rollout, traj_config in self._real_rollouts:
            traj = _trajectory_from_config(traj_config)
            sim_rollout = self._run_sim(traj)
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

    @property
    def backend(self) -> str:
        return self._backend

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
