"""skrl-based RL optimizer backend.

Requires: pip install skrl gymnasium torch

This backend treats parameter calibration as a contextual bandit problem.
Each episode is a single-step interaction: the policy maps a context
(trajectory type, payload estimate) to a parameter vector, the simulator
evaluates fidelity, and the negative loss is returned as the reward.

When to prefer this over CMA-ES / BO:
  - You want an amortised policy: context → params (SimOpt-style).
  - You have many different trajectories / payloads and want a single
    policy that generalises across them.
  - You're running a closed-loop iterative refinement loop (Phase 5).

For non-contextual tuning (fixed robot, fixed trajectory) CMA-ES is almost
always more sample-efficient.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .base import Optimizer


# ---------------------------------------------------------------------------
# Gymnasium environment wrapping the calibration loss
# ---------------------------------------------------------------------------

class SimCalibrationEnv:
    """Single-step contextual bandit Gymnasium env for parameter calibration.

    Observation: context vector (trajectory mode one-hot + payload estimate).
    Action:      normalised parameter vector in [-1, 1]^n_params.
    Reward:      -loss (higher is better, matching RL convention).
    Episode length: 1 step (terminated after every step).

    Args:
        loss_fn:   Callable theta -> float (the calibration problem's loss).
        n_params:  Dimension of the action space.
        context_fn: Callable () -> ndarray giving the current context.
                    If None, a constant zero vector is used.
        context_dim: Dimension of the context vector.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        loss_fn: Callable[[np.ndarray], float],
        n_params: int,
        context_fn: Callable[[], np.ndarray] | None = None,
        context_dim: int = 4,
    ) -> None:
        try:
            import gymnasium as gym
            from gymnasium import spaces
        except ImportError as exc:
            raise ImportError(
                "skrl backend requires gymnasium: pip install gymnasium"
            ) from exc

        self._loss_fn = loss_fn
        self._n_params = n_params
        self._context_fn = context_fn or (lambda: np.zeros(context_dim))
        self._context_dim = context_dim

        obs_dim = context_dim + n_params  # concat context + last action
        self.observation_space = spaces.Box(
            low=-np.ones(obs_dim, dtype=np.float32),
            high=np.ones(obs_dim, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-np.ones(n_params, dtype=np.float32),
            high=np.ones(n_params, dtype=np.float32),
            dtype=np.float32,
        )
        self._last_action = np.zeros(n_params, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        self._last_action = np.zeros(self._n_params, dtype=np.float32)
        ctx = self._context_fn().astype(np.float32)
        obs = np.concatenate([ctx, self._last_action])
        return obs.astype(np.float32), {}

    def step(self, action):
        theta = np.asarray(action, dtype=float)
        loss = float(self._loss_fn(theta))
        reward = -loss
        self._last_action = theta.astype(np.float32)
        ctx = self._context_fn().astype(np.float32)
        obs = np.concatenate([ctx, self._last_action])
        terminated = True  # single-step episode
        truncated = False
        return obs.astype(np.float32), float(reward), terminated, truncated, {"loss": loss}

    def close(self):
        """Gymnasium lifecycle hook used by skrl trainers."""
        return None


# ---------------------------------------------------------------------------
# Optimizer using skrl PPO or SAC
# ---------------------------------------------------------------------------

class SkrlOptimizer(Optimizer):
    """RL-based calibration via skrl PPO or SAC.

    Args:
        algorithm:   "ppo" or "sac".
        context_fn:  Callable producing the context vector each episode.
        context_dim: Dimension of the context vector.
        max_steps:   Total environment steps for training.
        device:      Torch device string ("cpu", "cuda", ...).
    """

    def __init__(
        self,
        algorithm: str = "ppo",
        context_fn: Callable[[], np.ndarray] | None = None,
        context_dim: int = 4,
        max_steps: int = 10_000,
        device: str = "cpu",
    ) -> None:
        self.algorithm = algorithm.lower()
        self.context_fn = context_fn
        self.context_dim = context_dim
        self.max_steps = max_steps
        self.device = device

    def minimize(
        self,
        objective: Callable[[np.ndarray], float],
        bounds: list[tuple[float, float]],
        x0: np.ndarray | None = None,
        *,
        max_evals: int = 10_000,
        verbose: bool = False,
    ) -> tuple[np.ndarray, list[tuple[np.ndarray, float]]]:
        try:
            import torch
            import skrl
            from skrl.envs.wrappers.torch import wrap_env
            from skrl.agents.torch import ppo as _ppo_module
        except ImportError as exc:
            raise ImportError(
                "skrl backend requires: pip install skrl gymnasium torch"
            ) from exc

        n = len(bounds)
        skrl_v2 = hasattr(_ppo_module, "PPO_CFG")
        history: list[tuple[np.ndarray, float]] = []

        def _tracked(theta: np.ndarray) -> float:
            loss = objective(theta)
            history.append((theta.copy(), loss))
            return loss

        env = SimCalibrationEnv(
            _tracked, n,
            context_fn=self.context_fn,
            context_dim=self.context_dim,
        )
        wrapped_env = wrap_env(env, wrapper="gymnasium")

        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]

        from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
        import torch.nn as nn

        class Policy(GaussianMixin, Model):
            def __init__(self, obs_space, act_space, device, **kwargs):
                try:
                    Model.__init__(self, observation_space=obs_space, action_space=act_space, device=device, **kwargs)
                except TypeError:
                    Model.__init__(self, obs_space, act_space, device, **kwargs)
                GaussianMixin.__init__(self, clip_actions=True)
                self.net = nn.Sequential(
                    nn.Linear(obs_dim, 64), nn.Tanh(),
                    nn.Linear(64, 64), nn.Tanh(),
                    nn.Linear(64, act_dim),
                )
                self.log_std = nn.Parameter(torch.zeros(act_dim))

            def compute(self, inputs, role):
                states = inputs.get("states")
                if states is None:
                    states = inputs.get("observations")
                x = self.net(states)
                return (x, {"log_std": self.log_std}) if skrl_v2 else (x, self.log_std, {})

        class Value(DeterministicMixin, Model):
            def __init__(self, obs_space, act_space, device, **kwargs):
                try:
                    Model.__init__(self, observation_space=obs_space, action_space=act_space, device=device, **kwargs)
                except TypeError:
                    Model.__init__(self, obs_space, act_space, device, **kwargs)
                DeterministicMixin.__init__(self)
                self.net = nn.Sequential(
                    nn.Linear(obs_dim, 64), nn.Tanh(),
                    nn.Linear(64, 64), nn.Tanh(),
                    nn.Linear(64, 1),
                )

            def compute(self, inputs, role):
                states = inputs.get("states")
                if states is None:
                    states = inputs.get("observations")
                return self.net(states), {}

        models = {
            "policy": Policy(wrapped_env.observation_space, wrapped_env.action_space, self.device),
            "value": Value(wrapped_env.observation_space, wrapped_env.action_space, self.device),
        }

        if self.algorithm == "ppo":
            from skrl.agents.torch.ppo import PPO
            from skrl.memories.torch import RandomMemory
            try:
                from skrl.agents.torch.ppo import PPO_DEFAULT_CONFIG
                cfg = PPO_DEFAULT_CONFIG.copy()
                cfg["rollouts"] = 16
                cfg["learning_epochs"] = 4
            except ImportError:
                # skrl >= 2 exposes a typed PPO_CFG instead of the legacy
                # dictionary constant.
                from skrl.agents.torch.ppo import PPO_CFG
                cfg = PPO_CFG(rollouts=16, learning_epochs=4)
            agent = PPO(
                models=models,
                memory=RandomMemory(memory_size=16, num_envs=1, device=self.device),
                cfg=cfg,
                observation_space=wrapped_env.observation_space,
                action_space=wrapped_env.action_space,
                device=self.device,
            )
        else:
            raise NotImplementedError(f"Algorithm '{self.algorithm}' not yet wired up.")

        from skrl.trainers.torch import SequentialTrainer
        trainer_cfg = {"timesteps": max_evals, "headless": not verbose}
        trainer = SequentialTrainer(
            env=wrapped_env, agents=agent, cfg=trainer_cfg
        )
        trainer.train()

        # Extract best action seen during training
        if history:
            best_x, _ = min(history, key=lambda p: p[1])
        else:
            best_x = np.zeros(n)

        return best_x, history
