"""skrl PPO as a one-step bandit (ported from ``legacy/.../optimizers/skrl_backend.py``).

**Experimental** (S2R_01 D-5): each episode is one step, the action is z in
[-1, 1]^n and the reward is -loss. Kept as the RL comparator the owner
explored; for a fixed context CMA-ES is usually far more sample-efficient.
Changes from legacy: rollouts/learning epochs configurable, a seed, an exact
budget (the trainer runs ``max_evals`` timesteps, one evaluation each).
"""

from __future__ import annotations

import importlib.util
from typing import Callable

import numpy as np

from .base import History, Optimizer, OptimizerUnavailable, best_of


class SimCalibrationEnv:
    """Single-step contextual bandit with the gymnasium interface.

    Observation: context + last action; action: z in [-1, 1]^n; reward: -loss.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        loss_fn: Callable[[np.ndarray], float],
        n_params: int,
        context_fn: Callable[[], np.ndarray] | None = None,
        context_dim: int = 4,
    ) -> None:
        from gymnasium import spaces

        self._loss_fn = loss_fn
        self._n_params = n_params
        self._context_fn = context_fn or (lambda: np.zeros(context_dim))
        obs_dim = context_dim + n_params
        self.observation_space = spaces.Box(low=-np.ones(obs_dim, dtype=np.float32),
                                            high=np.ones(obs_dim, dtype=np.float32), dtype=np.float32)
        self.action_space = spaces.Box(low=-np.ones(n_params, dtype=np.float32),
                                       high=np.ones(n_params, dtype=np.float32), dtype=np.float32)
        self._last_action = np.zeros(n_params, dtype=np.float32)

    def _observation(self) -> np.ndarray:
        return np.concatenate([self._context_fn().astype(np.float32), self._last_action]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        self._last_action = np.zeros(self._n_params, dtype=np.float32)
        return self._observation(), {}

    def step(self, action):
        theta = np.clip(np.asarray(action, dtype=float).reshape(-1), -1.0, 1.0)
        loss = float(self._loss_fn(theta))
        self._last_action = theta.astype(np.float32)
        return self._observation(), -loss, True, False, {"loss": loss}

    def render(self):
        return None

    def close(self):
        return None


class SkrlOptimizer(Optimizer):
    def __init__(
        self,
        algorithm: str = "ppo",
        context_fn: Callable[[], np.ndarray] | None = None,
        context_dim: int = 4,
        rollouts: int = 16,
        learning_epochs: int = 4,
        device: str = "cpu",
        seed: int | None = None,
    ) -> None:
        self.algorithm = algorithm.lower()
        self.context_fn = context_fn
        self.context_dim = context_dim
        self.rollouts = rollouts
        self.learning_epochs = learning_epochs
        self.device = device
        self.seed = seed

    @classmethod
    def is_available(cls) -> bool:
        return all(importlib.util.find_spec(name) is not None for name in ("skrl", "torch", "gymnasium"))

    def minimize(
        self,
        objective: Callable[[np.ndarray], float],
        bounds: list[tuple[float, float]],
        x0: np.ndarray | None = None,
        *,
        max_evals: int = 10_000,
        verbose: bool = False,
    ) -> tuple[np.ndarray, History]:
        if not self.is_available():
            raise OptimizerUnavailable("skrl_ppo", "skrl, torch and gymnasium")
        if self.algorithm != "ppo":
            raise NotImplementedError(f"Algorithm '{self.algorithm}' not yet wired up.")
        import torch
        import torch.nn as nn
        from skrl.agents.torch import ppo as _ppo_module
        from skrl.envs.wrappers.torch import wrap_env
        from skrl.memories.torch import RandomMemory
        from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
        from skrl.trainers.torch import SequentialTrainer

        import logging

        logging.getLogger("skrl").setLevel(logging.INFO if verbose else logging.ERROR)
        if self.seed is not None:
            from skrl.utils import set_seed

            set_seed(int(self.seed))

        n = len(bounds)
        skrl_v2 = hasattr(_ppo_module, "PPO_CFG")
        history: History = []

        def _tracked(theta: np.ndarray) -> float:
            loss = float(objective(theta))
            history.append((theta.copy(), loss))
            if verbose:
                print(f"  PPO eval {len(history):4d}: loss={loss:.6f}")
            return loss

        env = SimCalibrationEnv(_tracked, n, context_fn=self.context_fn, context_dim=self.context_dim)
        # skrl's wrapper takes its device from the env and defaults to CUDA.
        env.device = self.device
        wrapped_env = wrap_env(env, wrapper="gymnasium")
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]

        def _model_init(model, obs_space, act_space, device, kwargs):
            try:
                Model.__init__(model, observation_space=obs_space, action_space=act_space, device=device, **kwargs)
            except TypeError:
                Model.__init__(model, obs_space, act_space, device, **kwargs)

        def _states(inputs):
            states = inputs.get("states")
            return inputs.get("observations") if states is None else states

        class Policy(GaussianMixin, Model):
            def __init__(self, obs_space, act_space, device, **kwargs):
                _model_init(self, obs_space, act_space, device, kwargs)
                GaussianMixin.__init__(self, clip_actions=True)
                self.net = nn.Sequential(nn.Linear(obs_dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(),
                                         nn.Linear(64, act_dim))
                self.log_std = nn.Parameter(torch.zeros(act_dim))

            def compute(self, inputs, role):
                x = self.net(_states(inputs))
                return (x, {"log_std": self.log_std}) if skrl_v2 else (x, self.log_std, {})

        class Value(DeterministicMixin, Model):
            def __init__(self, obs_space, act_space, device, **kwargs):
                _model_init(self, obs_space, act_space, device, kwargs)
                DeterministicMixin.__init__(self)
                self.net = nn.Sequential(nn.Linear(obs_dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(),
                                         nn.Linear(64, 1))

            def compute(self, inputs, role):
                return self.net(_states(inputs)), {}

        models = {
            "policy": Policy(wrapped_env.observation_space, wrapped_env.action_space, self.device),
            "value": Value(wrapped_env.observation_space, wrapped_env.action_space, self.device),
        }
        if skrl_v2:
            cfg = _ppo_module.PPO_CFG(rollouts=self.rollouts, learning_epochs=self.learning_epochs)
        else:
            cfg = _ppo_module.PPO_DEFAULT_CONFIG.copy()
            cfg["rollouts"] = self.rollouts
            cfg["learning_epochs"] = self.learning_epochs
        agent = _ppo_module.PPO(
            models=models,
            memory=RandomMemory(memory_size=self.rollouts, num_envs=1, device=self.device),
            cfg=cfg,
            observation_space=wrapped_env.observation_space,
            action_space=wrapped_env.action_space,
            device=self.device,
        )
        trainer = SequentialTrainer(env=wrapped_env, agents=agent,
                                    cfg={"timesteps": max_evals, "headless": True,
                                         "disable_progressbar": not verbose})
        trainer.train()
        return best_of(history, np.zeros(n)), history
