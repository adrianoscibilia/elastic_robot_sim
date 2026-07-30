"""Dataset-to-simulator calibration primitives for arbitrary serial arms.

Unlike the Cartesian sim-to-real stack, this module takes the applied joint
torque sequence as its input and compares only channels actually observed in
the dataset.  End-effector F/T and joint torque are deliberately separate
channels: neither is silently substituted for the other.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd


def _vector_columns(frame: pd.DataFrame, prefix: str, n_dof: int, *, one_based: bool = False) -> list[str]:
    """Find a complete ordered vector using accepted canonical/legacy names."""
    offsets = (1, 0) if one_based else (0, 1)
    candidates = []
    for offset in offsets:
        candidates.extend([
            [f"{prefix}{i + offset}" for i in range(n_dof)],
            [f"{prefix}_{i + offset}" for i in range(n_dof)],
        ])
    for columns in candidates:
        if all(column in frame.columns for column in columns):
            return columns
    raise ValueError(f"missing {n_dof}-DoF '{prefix}' vector columns")


@dataclass(frozen=True)
class TorqueReplayRollout:
    """A measured joint-space rollout driven by applied torque samples."""

    time: np.ndarray
    q: np.ndarray
    dq: np.ndarray
    tau: np.ndarray
    joint_names: tuple[str, ...]
    q_next: np.ndarray | None = None
    dq_next: np.ndarray | None = None
    motor_q: np.ndarray | None = None
    motor_dq: np.ndarray | None = None
    link_q: np.ndarray | None = None
    link_dq: np.ndarray | None = None
    end_effector_ft: np.ndarray | None = None
    metadata: Mapping[str, object] | None = None

    @property
    def n_dof(self) -> int:
        return len(self.joint_names)

    @classmethod
    def from_dataframe(
        cls, frame: pd.DataFrame, *, joint_names: Sequence[str] | None = None
    ) -> "TorqueReplayRollout":
        if len(frame) < 2:
            raise ValueError("a calibration rollout needs at least two samples")
        time_col = "t" if "t" in frame.columns else "time" if "time" in frame.columns else None
        if time_col is None:
            raise ValueError("dataset must contain 't' or 'time'")
        if joint_names is None:
            # Prefer a declared count, otherwise infer legacy q0/q_0 columns.
            indexed = []
            for name in frame.columns:
                if name.startswith("q_") and name[2:].isdigit():
                    indexed.append(int(name[2:]))
                elif name.startswith("q") and name[1:].isdigit():
                    indexed.append(int(name[1:]))
            n_dof = max(indexed) + 1 if indexed else 0
            if not n_dof:
                raise ValueError("joint_names must be supplied when q columns cannot be inferred")
            joint_names = tuple(f"joint_{i + 1}" for i in range(n_dof))
        n_dof = len(joint_names)
        q_cols = _vector_columns(frame, "q", n_dof)
        dq_cols = _vector_columns(frame, "dq", n_dof)
        tau_cols = _vector_columns(frame, "tau", n_dof, one_based=True)
        def optional_vector(prefix: str) -> np.ndarray | None:
            try:
                return frame[_vector_columns(frame, prefix, n_dof)].to_numpy(float)
            except ValueError:
                return None
        motor_q, motor_dq = optional_vector("q_motor"), optional_vector("dq_motor")
        link_q, link_dq = optional_vector("q_link"), optional_vector("dq_link")
        q_next = None
        dq_next = None
        try:
            q_next = frame[_vector_columns(frame, "q_next", n_dof)].to_numpy(float)
            dq_next = frame[_vector_columns(frame, "dq_next", n_dof)].to_numpy(float)
        except ValueError:
            pass
        ft_names = ("fx", "fy", "fz", "tx", "ty", "tz")
        ft = frame[list(ft_names)].to_numpy(float) if all(x in frame for x in ft_names) else None
        time = frame[time_col].to_numpy(float)
        if not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0.0):
            raise ValueError("rollout time must be finite and strictly increasing")
        return cls(
            time=time,
            # q/dq always define the primary observable state.  For an
            # elastic robot that is the output/link side; retain the explicit
            # motor/link streams independently when a benchmark supplies them.
            q=link_q if link_q is not None else frame[q_cols].to_numpy(float),
            dq=link_dq if link_dq is not None else frame[dq_cols].to_numpy(float),
            tau=frame[tau_cols].to_numpy(float),
            joint_names=tuple(joint_names),
            q_next=q_next,
            dq_next=dq_next,
            motor_q=motor_q,
            motor_dq=motor_dq,
            link_q=link_q,
            link_dq=link_dq,
            end_effector_ft=ft,
        )


class TorqueReplayRunner(Protocol):
    """Simulator adapter required by :class:`TorqueReplayCalibrationProblem`."""

    def run_torque_replay(self, params: Mapping[str, float], rollout: TorqueReplayRollout) -> Mapping[str, np.ndarray]: ...


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    lower: float
    upper: float
    log_scale: bool = True

    def __post_init__(self) -> None:
        if self.lower <= 0.0 and self.log_scale:
            raise ValueError(f"log-scaled parameter '{self.name}' needs a positive lower bound")
        if self.upper <= self.lower:
            raise ValueError(f"invalid bounds for '{self.name}'")


class ParameterSpace:
    """Named physical parameters mapped to the optimizers' [-1, 1] space."""

    def __init__(self, specs: Sequence[ParameterSpec]) -> None:
        if not specs or len({item.name for item in specs}) != len(specs):
            raise ValueError("parameter specs must be non-empty and uniquely named")
        self.specs = tuple(specs)

    @property
    def bounds(self) -> list[tuple[float, float]]:
        return [(-1.0, 1.0)] * len(self.specs)

    def decode(self, theta: np.ndarray) -> dict[str, float]:
        if len(theta) != len(self.specs):
            raise ValueError("theta has the wrong dimension")
        result = {}
        for value, spec in zip(np.asarray(theta, dtype=float), self.specs):
            value = float(np.clip(value, -1.0, 1.0))
            if spec.log_scale:
                lo, hi = np.log(spec.lower), np.log(spec.upper)
                result[spec.name] = float(np.exp(lo + (value + 1.0) * (hi - lo) / 2.0))
            else:
                result[spec.name] = float(spec.lower + (value + 1.0) * (spec.upper - spec.lower) / 2.0)
        return result

    def encode(self, params: Mapping[str, float]) -> np.ndarray:
        """Map named physical values to normalized coordinates."""
        values = []
        for spec in self.specs:
            value = float(params[spec.name])
            if spec.log_scale:
                value, lo, hi = np.log(value), np.log(spec.lower), np.log(spec.upper)
            else:
                lo, hi = spec.lower, spec.upper
            values.append(2.0 * (value - lo) / (hi - lo) - 1.0)
        return np.asarray(values, dtype=float)


def _resample(values: np.ndarray, source_time: np.ndarray, target_time: np.ndarray) -> np.ndarray:
    return np.column_stack([np.interp(target_time, source_time, values[:, idx]) for idx in range(values.shape[1])])


def _normalised_rmse(prediction: np.ndarray, reference: np.ndarray) -> tuple[float, np.ndarray]:
    error = np.sqrt(np.mean((prediction - reference) ** 2, axis=0))
    scale = np.maximum(np.percentile(np.abs(reference), 95, axis=0), 1.0e-6)
    per_joint = error / scale
    return float(np.mean(per_joint)), per_joint


def compare_torque_replay(
    simulated: Mapping[str, np.ndarray], ground_truth: TorqueReplayRollout, *,
    q_weight: float = 1.0, dq_weight: float = 0.3,
    motor_q_weight: float = 0.0, motor_dq_weight: float = 0.0,
) -> dict[str, object]:
    """Compare observable output-side state; F/T remains diagnostic metadata."""
    sim_time = np.asarray(simulated.get("time", ground_truth.time), dtype=float)
    sim_q = np.asarray(simulated.get("q_link", simulated.get("q")), dtype=float)
    sim_dq = np.asarray(simulated.get("dq_link", simulated.get("dq")), dtype=float)
    if sim_q.ndim != 2 or sim_dq.shape != sim_q.shape or sim_q.shape[1] != ground_truth.n_dof:
        raise ValueError("simulator must return q/q_link and dq/dq_link with shape (N, n_dof)")
    q = _resample(sim_q, sim_time, ground_truth.time)
    dq = _resample(sim_dq, sim_time, ground_truth.time)
    q_error, q_per_joint = _normalised_rmse(q, ground_truth.q)
    dq_error, dq_per_joint = _normalised_rmse(dq, ground_truth.dq)
    weights = (q_weight, dq_weight, motor_q_weight, motor_dq_weight)
    if any(weight < 0.0 for weight in weights) or sum(weights) == 0.0:
        raise ValueError("at least one non-negative metric weight is required")
    weighted_errors = [(q_weight, q_error), (dq_weight, dq_error)]
    report: dict[str, object] = {
        "q_nrmse": q_error,
        "dq_nrmse": dq_error,
        "per_joint": {
            name: {"q_nrmse": float(q_per_joint[i]), "dq_nrmse": float(dq_per_joint[i])}
            for i, name in enumerate(ground_truth.joint_names)
        },
    }
    # Motor state is only a valid target when the source has distinct encoder
    # measurements.  Joint torque and EEF F/T remain separate observation
    # types and are never used here as a replacement for either state stream.
    if motor_q_weight or motor_dq_weight:
        if ground_truth.motor_q is None or ground_truth.motor_dq is None:
            raise ValueError("motor-state weights require motor_q and motor_dq in the dataset")
        if "q_motor" not in simulated or "dq_motor" not in simulated:
            raise ValueError("motor-state weights require q_motor and dq_motor from the simulator")
        motor_q = _resample(np.asarray(simulated["q_motor"], dtype=float), sim_time, ground_truth.time)
        motor_dq = _resample(np.asarray(simulated["dq_motor"], dtype=float), sim_time, ground_truth.time)
        motor_q_error, motor_q_per_joint = _normalised_rmse(motor_q, ground_truth.motor_q)
        motor_dq_error, motor_dq_per_joint = _normalised_rmse(motor_dq, ground_truth.motor_dq)
        weighted_errors.extend(((motor_q_weight, motor_q_error), (motor_dq_weight, motor_dq_error)))
        report["motor_q_nrmse"] = motor_q_error
        report["motor_dq_nrmse"] = motor_dq_error
        for index, name in enumerate(ground_truth.joint_names):
            report["per_joint"][name].update({  # type: ignore[index]
                "motor_q_nrmse": float(motor_q_per_joint[index]),
                "motor_dq_nrmse": float(motor_dq_per_joint[index]),
            })
    report["metric"] = float(sum(weight * error for weight, error in weighted_errors) / sum(weight for weight, _ in weighted_errors))
    return report


class TorqueReplayCalibrationProblem:
    """Aggregate a runner's open-loop torque-replay error across trajectories."""

    def __init__(self, runner: TorqueReplayRunner, rollouts: Sequence[TorqueReplayRollout], parameter_space: ParameterSpace, *, q_weight: float = 1.0, dq_weight: float = 0.3, motor_q_weight: float = 0.0, motor_dq_weight: float = 0.0) -> None:
        if not rollouts:
            raise ValueError("at least one calibration rollout is required")
        self.runner = runner
        self.rollouts = tuple(rollouts)
        self.parameter_space = parameter_space
        self.q_weight = q_weight
        self.dq_weight = dq_weight
        self.motor_q_weight = motor_q_weight
        self.motor_dq_weight = motor_dq_weight
        self.history: list[tuple[np.ndarray, float]] = []

    def loss(self, theta: np.ndarray) -> float:
        params = self.parameter_space.decode(theta)
        reports = [compare_torque_replay(self.runner.run_torque_replay(params, rollout), rollout, q_weight=self.q_weight, dq_weight=self.dq_weight, motor_q_weight=self.motor_q_weight, motor_dq_weight=self.motor_dq_weight) for rollout in self.rollouts]
        loss = float(np.mean([report["metric"] for report in reports]))
        self.history.append((np.asarray(theta, dtype=float).copy(), loss))
        return loss

    @property
    def best(self) -> tuple[np.ndarray | None, float]:
        if not self.history:
            return None, float("inf")
        return min(self.history, key=lambda item: item[1])


def load_torque_replay_rollouts(path: str, *, joint_names: Sequence[str] | None = None) -> list[TorqueReplayRollout]:
    """Load one canonical CSV/Parquet file, split on its optional ``bag`` field."""
    frame = pd.read_parquet(path) if path.lower().endswith((".parquet", ".pq")) else pd.read_csv(path)
    groups = [frame] if "bag" not in frame else [part for _, part in frame.groupby("bag", sort=True)]
    return [TorqueReplayRollout.from_dataframe(group.reset_index(drop=True), joint_names=joint_names) for group in groups]
