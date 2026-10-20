"""Loss (S2R_01 §6, T-4.1).

* **Normalised channel error** of bag b, joint j, channel c:
  ``e_cbj = RMS(r_bj) / s_cj``, r = sim - real on the kept rows, with
  ``s_cj = max(RMS of the real channel over the training bags,
  10 x noise sigma (FR_05 §6), 0.05 x max_j RMS)``. The floors stop A5-A7
  from dominating through tiny denominators.
* **Loss** ``J = sum_c w_c mean_b mean_j e_cbj + lambda * prior_penalty``.
  ``lambda`` defaults (``prior_weight: auto``) to ``1 / N``, N the number of
  scalar residual entries of one training evaluation (rows x joints x
  channels): the data term below is a mean over those entries, so the prior
  then weighs like one extra observation per parameter.
  ``prior_weight_factor`` scales the auto weight (S2R_05 F-1: the rows are
  normalised by the signal RMS, not the noise, so ``1 / N`` lets one prior
  sigma outweigh a residual of the full signal RMS on one row).
* **Least squares** gets the stacked residual
  ``rho = [ sqrt(w_c / (B n N_b)) r_bj[k] / s_cj ; sqrt(lambda) (f(x) - f(mu)) / sigma ]``
  so ``0.5 |rho|^2 = 0.5 sum_c w_c mean_b mean_j e_cbj^2 + lambda * prior_penalty``:
  the same normalisation, squared inside the mean (S2R_01 §6).

R0 has one channel, ``ft``; ``tracking`` and ``tau_cmd`` come with R2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .config import LossSettings
from .result import ReplayResult

#: S2R_01 §6 floors.
NOISE_FLOOR_FACTOR = 10.0
SHARE_FLOOR_FACTOR = 0.05


@dataclass(frozen=True)
class ChannelScale:
    """s_cj per joint and what set it."""

    channel: str
    scale: np.ndarray
    rms_train: np.ndarray
    noise_floor: np.ndarray
    share_floor: float

    def as_dict(self) -> dict[str, Any]:
        binding = np.where(self.scale == self.rms_train, "rms_train",
                           np.where(self.scale == self.noise_floor, "noise", "share"))
        return {"channel": self.channel, "scale": self.scale.tolist(), "rms_train": self.rms_train.tolist(),
                "noise_floor": self.noise_floor.tolist(), "share_floor": self.share_floor,
                "binding": binding.tolist()}


@dataclass(frozen=True)
class LossValue:
    total: float
    data: float                    # sum_c w_c mean_b mean_j e_cbj
    prior: float                   # lambda * prior_penalty
    prior_penalty: float
    per_joint: np.ndarray          # mean over bags of e_bj (ft)
    per_bag: Mapping[str, float]   # "run:bag" -> mean_j e_bj
    squared: float                 # 0.5 |rho|^2 (what least squares minimises)

    def as_dict(self) -> dict[str, Any]:
        return {"total": self.total, "data": self.data, "prior": self.prior, "prior_penalty": self.prior_penalty,
                "per_joint": self.per_joint.tolist(), "per_bag": dict(self.per_bag), "squared": self.squared}


@dataclass(frozen=True)
class Loss:
    scales: Mapping[str, ChannelScale]
    weights: Mapping[str, float]
    prior_weight: float
    prior_weight_auto: bool = True
    n_residuals: int = 0
    notes: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_training(cls, train: Sequence[ReplayResult], noise_sigma: np.ndarray, settings: LossSettings
                      ) -> "Loss":
        """Scales from the real channel over the training bags (theta-free),
        lambda from the training size."""
        if not train:
            raise ValueError("Loss.from_training: no training results")
        channels = {c: w for c, w in settings.channels.items() if w != 0.0}
        unsupported = sorted(set(channels) - {"ft"})
        if unsupported:
            raise ValueError(f"loss channels {unsupported} need R2 (tracking, tau_cmd); R0/R1 observe ft only")
        real = np.vstack([r.ft for r in train])
        rms = np.sqrt(np.mean(real ** 2, axis=0))
        noise = NOISE_FLOOR_FACTOR * np.asarray(noise_sigma, dtype=float)
        share = SHARE_FLOOR_FACTOR * float(np.max(rms))
        scale = np.maximum(np.maximum(rms, noise), share)
        n_residuals = int(sum(r.n_rows * r.n_dof for r in train)) * len(channels)
        auto = settings.prior_weight is None
        factor = float(settings.prior_weight_factor)
        weight = factor / n_residuals if auto else float(settings.prior_weight)
        return cls(scales={"ft": ChannelScale("ft", scale, rms, noise, share)}, weights=channels,
                   prior_weight=weight, prior_weight_auto=auto, n_residuals=n_residuals,
                   notes={"prior_weight_factor": factor})

    # -- scalar loss -------------------------------------------------------------

    def errors(self, results: Sequence[ReplayResult]) -> np.ndarray:
        """e_bj (ft): (bags, joints)."""
        scale = self.scales["ft"].scale
        return np.vstack([np.sqrt(np.mean(r.residual ** 2, axis=0)) / scale for r in results])

    def evaluate(self, results: Sequence[ReplayResult], prior_penalty: float = 0.0) -> LossValue:
        e = self.errors(results)
        data = self.weights.get("ft", 0.0) * float(np.mean(e))
        prior = self.prior_weight * float(prior_penalty)
        squared = 0.5 * self.weights.get("ft", 0.0) * float(np.mean(e ** 2)) + prior
        return LossValue(total=data + prior, data=data, prior=prior, prior_penalty=float(prior_penalty),
                         per_joint=e.mean(axis=0),
                         per_bag={f"{r.run_id}:{r.bag}": float(row.mean()) for r, row in zip(results, e)},
                         squared=squared)

    # -- least squares -----------------------------------------------------------

    def row_weights(self, results: Sequence[ReplayResult]) -> np.ndarray:
        """Per (row, joint) weight of the stacked data residual, flattened
        row-major in the order of ``results`` (= the engine's row order)."""
        scale = self.scales["ft"].scale
        n_bags = len(results)
        pieces = []
        for r in results:
            w = np.sqrt(self.weights.get("ft", 0.0) / (n_bags * r.n_dof * r.n_rows)) / scale
            pieces.append(np.broadcast_to(w, r.residual.shape).reshape(-1))
        return np.concatenate(pieces)

    def residual_vector(self, results: Sequence[ReplayResult], prior_rows: np.ndarray | None = None) -> np.ndarray:
        data = np.concatenate([r.residual.reshape(-1) for r in results]) * self.row_weights(results)
        if prior_rows is None:
            return data
        return np.concatenate([data, np.sqrt(self.prior_weight) * np.asarray(prior_rows, dtype=float)])

    def describe(self) -> dict[str, Any]:
        return {"weights": dict(self.weights), "prior_weight": self.prior_weight,
                "prior_weight_auto": self.prior_weight_auto, "n_residuals": self.n_residuals,
                "prior_weight_factor": self.notes.get("prior_weight_factor", 1.0),
                "scales": {c: s.as_dict() for c, s in self.scales.items()}}


__all__ = ["ChannelScale", "Loss", "LossValue", "NOISE_FLOOR_FACTOR", "SHARE_FLOOR_FACTOR"]
