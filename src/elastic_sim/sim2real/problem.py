"""The calibration problem: registry x bags x replay mode x loss (S2R_01 T-4.2).

:class:`CalibrationProblem` is what every optimizer backend sees: an
objective over the registry's box ``z in [-1, 1]^n`` (scalar for CMA, BO,
random; the stacked residual and its analytic Jacobian for least squares).

* **Splits.** One engine per split (``train`` drives the objective, ``val``
  selects, ``test`` is replayed once at the end by the caller). R0 needs no
  process pool; the engines are built once and updated in place, and
  :class:`CalibrationProblem` keeps the ``jobs`` field for R1/R2.
* **Selection.** Each evaluation also replays ``val``; the best theta is the
  lowest **val data loss** (the loss without the prior term, which is a
  regulariser of the fit, not a property of held-out data). ``on_improvement``
  is called on every new best (``calibrate.py`` rewrites ``working.yaml``
  there, S2R_02 A-3).
* **History.** Every evaluation appends one JSON line (z, the free theta,
  loss and components, val loss and RMSE, wall time, error) to
  ``history.jsonl``, flushed at once. **Resume**: the records are read back
  and an evaluation at a z already in the history returns its record without
  recomputing or re-logging, so a seeded optimizer restarted from scratch
  retraces its path for free and continues: same best theta as an
  uninterrupted run.
* **Non-physical theta** (S2R_01 §5): a finite loss of 1e12 with the reasons,
  logged; least squares gets a constant residual of the same 0.5|r|^2.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .config import Sim2RealConfig
from .loss import Loss
from .metrics import torque_errors
from .parameters import Registry
from .plant import NONPHYSICAL_LOSS, PlantBase, PlantParameters
from .real_runs import RealRun
from .replay_r0 import R0Engine, prepare_bags
from .result import ReplayResult

HISTORY_SCHEMA = "elastic_sim.sim2real.history/1"


@dataclass
class Evaluation:
    index: int
    z: np.ndarray
    theta: dict[str, float]          # free parameters
    loss: float
    data: float | None = None
    prior: float | None = None
    squared: float | None = None
    per_joint: list[float] | None = None
    val_loss: float | None = None
    val_rmse: float | None = None
    wall_s: float = 0.0
    error: str | None = None
    reasons: list[str] = field(default_factory=list)
    cached: bool = False

    def record(self) -> dict[str, Any]:
        return {"i": self.index, "z": [float(v) for v in self.z], "theta": self.theta, "loss": self.loss,
                "data": self.data, "prior": self.prior, "squared": self.squared, "per_joint": self.per_joint,
                "val_loss": self.val_loss, "val_rmse": self.val_rmse, "wall_s": self.wall_s, "error": self.error,
                "reasons": self.reasons}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "Evaluation":
        return cls(index=int(record["i"]), z=np.asarray(record["z"], dtype=float), theta=dict(record["theta"]),
                   loss=float(record["loss"]), data=record.get("data"), prior=record.get("prior"),
                   squared=record.get("squared"), per_joint=record.get("per_joint"),
                   val_loss=record.get("val_loss"), val_rmse=record.get("val_rmse"),
                   wall_s=float(record.get("wall_s", 0.0)), error=record.get("error"),
                   reasons=list(record.get("reasons") or []), cached=True)

    @property
    def selection(self) -> float:
        """What selects the best theta: val data loss, else the training loss."""
        if self.error is not None:
            return float("inf")
        return self.val_loss if self.val_loss is not None else self.loss


class CalibrationProblem:
    def __init__(self, registry: Registry, base: PlantBase, engines: Mapping[str, R0Engine], loss: Loss, *,
                 history_path: str | Path | None = None, resume: bool = True,
                 on_improvement: Callable[[Evaluation, dict[str, float]], None] | None = None,
                 runs: Sequence[str] = (), jobs: int = 1):
        if "train" not in engines:
            raise ValueError("CalibrationProblem needs a train engine")
        self.registry = registry
        self.base = base
        self.engines = dict(engines)
        self.loss = loss
        self.runs = tuple(runs)
        self.jobs = jobs
        self.on_improvement = on_improvement
        self.history_path = None if history_path is None else Path(history_path)
        self.evaluations: list[Evaluation] = []
        self._cache: dict[bytes, Evaluation] = {}
        self.best: Evaluation | None = None
        self.n_evaluations = 0  # calls this session, cached ones included
        self._last_train: tuple[bytes, list[ReplayResult]] | None = None
        if self.history_path is not None and self.history_path.is_file():
            if not resume:
                raise FileExistsError(f"{self.history_path} exists; resume or choose another run folder")
            for line in self.history_path.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                if record.get("schema") == HISTORY_SCHEMA:
                    continue
                evaluation = Evaluation.from_record(record)
                self.evaluations.append(evaluation)
                self._cache[evaluation.z.tobytes()] = evaluation
                self._track_best(evaluation, notify=False)
            if self.best is not None and self.on_improvement is not None:
                self.on_improvement(self.best, self.full_theta(self.best.theta))
        elif self.history_path is not None:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            self.history_path.write_text(json.dumps({"schema": HISTORY_SCHEMA, "free": list(registry.free_names)})
                                         + "\n", encoding="utf-8")

    # -- construction --------------------------------------------------------------

    @classmethod
    def from_config(cls, config: Sim2RealConfig, registry: Registry, base: PlantBase,
                    runs: Mapping[str, RealRun], **kwargs: Any) -> "CalibrationProblem":
        if config.replay.mode != "R0":
            raise NotImplementedError(f"replay mode {config.replay.mode}: P-4 (S2R_04 §5); P-3 is R0 only")
        engines = build_engines(config, base, runs)
        noise = noise_sigma(runs.values())
        theta0 = registry.values()
        plant0 = PlantParameters.from_theta(base, theta0, runs=tuple(runs))
        loss = Loss.from_training(engines["train"].replay(plant0), noise, config.loss)
        return cls(registry, base, engines, loss, runs=tuple(runs), jobs=config.jobs or 1, **kwargs)

    # -- theta -----------------------------------------------------------------------

    def full_theta(self, free: Mapping[str, float]) -> dict[str, float]:
        theta = self.registry.values()
        theta.update(free)
        return theta

    def theta(self, z: np.ndarray) -> dict[str, float]:
        return self.registry.decode(z)

    def plant(self, theta: Mapping[str, float]) -> PlantParameters:
        return PlantParameters.from_theta(self.base, theta, runs=self.runs)

    def replay(self, theta: Mapping[str, float], split: str) -> list[ReplayResult]:
        return self.engines[split].replay(self.plant(theta))

    # -- objective -----------------------------------------------------------------------

    def evaluate(self, z: np.ndarray) -> Evaluation:
        z = np.asarray(z, dtype=float).reshape(-1)
        self.n_evaluations += 1
        key = z.tobytes()
        if key in self._cache:
            return self._cache[key]
        start = time.perf_counter()
        theta = self.theta(z)
        free = {n: theta[n] for n in self.registry.free_names}
        plant = self.plant(theta)
        failure = plant.physical_failure()
        if failure is not None:
            evaluation = Evaluation(len(self.evaluations), z.copy(), free, NONPHYSICAL_LOSS, error=failure["error"],
                                    reasons=failure["reasons"])
        else:
            train = self.engines["train"].replay(plant)
            self._last_train = (key, train)
            value = self.loss.evaluate(train, self.registry.prior_penalty(theta))
            evaluation = Evaluation(len(self.evaluations), z.copy(), free, value.total, data=value.data,
                                    prior=value.prior, squared=value.squared, per_joint=value.per_joint.tolist())
            if "val" in self.engines:
                val = self.engines["val"].replay(plant)
                evaluation.val_loss = self.loss.evaluate(val).data
                dynamic = [r for r in val if r.kind != "hold"]
                if dynamic:
                    evaluation.val_rmse = torque_errors(dynamic)["rmse_median_over_bags"]
        evaluation.wall_s = time.perf_counter() - start
        self._log(evaluation)
        return evaluation

    def objective(self, z: np.ndarray) -> float:
        return self.evaluate(z).loss

    def residuals(self, z: np.ndarray) -> np.ndarray:
        """Stacked residual (data rows, then prior rows) for least squares;
        the evaluation is recorded like any other."""
        evaluation = self.evaluate(z)
        theta = self.theta(z)
        if evaluation.error is not None:
            size = self.loss.n_residuals + self.registry.n_free
            return np.full(size, np.sqrt(2.0 * NONPHYSICAL_LOSS / size))
        key = np.asarray(z, dtype=float).reshape(-1).tobytes()
        if self._last_train is not None and self._last_train[0] == key:
            train = self._last_train[1]
        else:
            train = self.engines["train"].replay(self.plant(theta))
        return self.loss.residual_vector(train, self.registry.prior_residuals(theta))

    def jacobian(self, z: np.ndarray) -> np.ndarray:
        """d residuals / d z (analytic in R0)."""
        theta = self.theta(z)
        plant = self.plant(theta)
        if plant.physical_failure() is not None:
            return np.zeros((self.loss.n_residuals + self.registry.n_free, self.registry.n_free))
        engine = self.engines["train"]
        names = self.registry.free_names
        weights = self.loss.row_weights(engine.results(plant, *engine.simulate(plant)))
        data = weights[:, None] * engine.jacobian(plant, names, theta=theta) * self.registry.dtheta_dz(theta)[None, :]
        prior = np.diag(np.sqrt(self.loss.prior_weight) * self.registry.prior_rows_dz())
        return np.vstack([data, prior])

    def data_jacobian_u(self, theta: Mapping[str, float], names: Sequence[str] | None = None,
                        split: str = "train") -> np.ndarray:
        """d (weighted data residual) / d u, u = f(theta) / sigma: the
        identifiability input. ``names`` may include frozen parameters with a
        prior sigma (e.g. K), analysed for the record."""
        names = list(self.registry.free_names if names is None else names)
        engine = self.engines[split]
        plant = self.plant(theta)
        weights = self.loss.row_weights(engine.results(plant, *engine.simulate(plant)))
        return weights[:, None] * engine.jacobian(plant, names, theta=theta) * \
            self.registry.dtheta_du(theta, names)[None, :]

    # -- bookkeeping ------------------------------------------------------------------------

    def _log(self, evaluation: Evaluation) -> None:
        self.evaluations.append(evaluation)
        self._cache[evaluation.z.tobytes()] = evaluation
        if self.history_path is not None:
            with open(self.history_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(evaluation.record()) + "\n")
                handle.flush()
        self._track_best(evaluation, notify=True)

    def _track_best(self, evaluation: Evaluation, *, notify: bool) -> None:
        if evaluation.error is not None:
            return
        if self.best is None or evaluation.selection < self.best.selection:
            self.best = evaluation
            if notify and self.on_improvement is not None:
                self.on_improvement(evaluation, self.full_theta(evaluation.theta))

    def best_train(self, key: str = "loss") -> Evaluation | None:
        """Lowest training objective: ``loss`` (J) or ``squared`` (0.5|r|^2,
        what least squares minimises)."""
        ok = [e for e in self.evaluations if e.error is None and getattr(e, key) is not None]
        return min(ok, key=lambda e: getattr(e, key)) if ok else None


def build_engines(config: Sim2RealConfig, base: PlantBase, runs: Mapping[str, RealRun],
                  splits: Sequence[str] = ("train", "val", "test", "zero")) -> dict[str, R0Engine]:
    """One R0 engine per non-empty split (A-9: intermediate bodies built once;
    ``zero``: the session-zero bags, S2R_06 A-13)."""
    from .plant import IntermediateBodies

    intermediate = IntermediateBodies.from_elastic_build(base)
    settings = config.replay.r0
    engines = {}
    for split in splits:
        bags = []
        for run in runs.values():
            chosen = [b for b in run.bags if config.split.split_of(b.run_id, b.bag) == split]
            bags += prepare_bags(chosen, settings, time_step=1.0 / run.rate_hz, split_of=config.split.split_of)
        if bags:
            engines[split] = R0Engine(base, bags, deflection=settings.deflection == "measured_ft",
                                      intermediate=intermediate)
    return engines


def noise_sigma(runs: Sequence[RealRun] | Any) -> np.ndarray:
    """FR_05 §6 sigma(ft) per joint: the sidecar's hold sigma, mean over runs."""
    sigmas = [np.asarray(run.sidecar["summary"]["holds"]["sigma_ft"], dtype=float) for run in runs
              if run.sidecar and "holds" in run.sidecar.get("summary", {})]
    if not sigmas:
        raise ValueError("no hold noise in the runs' sidecars (load with segments including 'hold')")
    return np.mean(sigmas, axis=0)


def load_runs(config: Sim2RealConfig) -> dict[str, RealRun]:
    from .real_runs import load_real_run

    return {run_id: load_real_run(spec.path, expected_sha256=spec.sha256, segments=config.segments, run_id=run_id)
            for run_id, spec in config.runs.items()}


__all__ = ["CalibrationProblem", "Evaluation", "HISTORY_SCHEMA", "build_engines", "load_runs", "noise_sigma"]
