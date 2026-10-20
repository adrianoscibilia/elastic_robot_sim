"""Replay results on the real time grid (S2R_01 T-3.5).

A :class:`ReplayResult` is one bag replayed in one mode at one theta: the
real channels on the kept rows next to the simulated ones. Every consumer
(loss, metrics, report, ``replay.py``) reads these, so the residual has one
definition: ``sim - real`` on the recorded channel (S2R_01 §6).

:func:`replay_frame` flattens results into one table, the
``replay_<split>.parquet`` of a calibration run. Columns (``j`` = joint index):

==================  =======================================================
``run``, ``bag``    run id (config key) and bag name
``kind``            ``excitation`` | ``identify_excitation`` | ``hold``
``split``           ``train`` | ``val`` | ``test`` | ``""`` (in no split)
``mode``            replay mode (``R0``, ``R1``, ``R2``)
``row``             row index in the bag's 1 kHz cycle grid
``t``               s, from the bag's first cycle
``q{j}``            measured motor position, rad
``dq{j}``, ddq{j}`` the replay's derivatives (R0: SG of ``q``, 0 at holds; R1: the
                    reference spline's; R2: 0)
``q_link{j}``       link position the model was evaluated at (R1/R2: simulated), rad
``ft{j}``           measured joint torque sensor, N m, model sign
``tau_s{j}``        simulated spring (transmission) torque, N m
``ft_sim{j}``       the observation model's ``gain * tau_s + offset``
``res{j}``          ``ft_sim - ft``
``gain{j}``         the observation model's gain (constant per bag)
``offset{j}``       its offset for the bag's run (constant per bag)
==================  =======================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

_ARRAYS = ("q", "dq", "ddq", "q_link", "ft", "tau_s", "ft_sim")


@dataclass(frozen=True)
class ReplayResult:
    run_id: str
    bag: str
    kind: str
    split: str
    mode: str
    rows: np.ndarray       # (M,) indices into the bag's rows
    t: np.ndarray          # (M,)
    q: np.ndarray          # (M, n)
    dq: np.ndarray
    ddq: np.ndarray
    q_link: np.ndarray
    ft: np.ndarray         # real
    tau_s: np.ndarray      # simulated spring torque
    ft_sim: np.ndarray     # observed: gain * tau_s + offset
    gain: np.ndarray | None = None    # (n,) observation model of this bag
    offset: np.ndarray | None = None  # (n,)
    extra: Mapping[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str]:
        return self.run_id, self.bag

    @property
    def residual(self) -> np.ndarray:
        """sim - real on the recorded channel."""
        return self.ft_sim - self.ft

    @property
    def n_rows(self) -> int:
        return len(self.t)

    @property
    def n_dof(self) -> int:
        return self.q.shape[1]


def replay_frame(results: Iterable[ReplayResult]) -> pd.DataFrame:
    """One row per kept sample of every result (module docstring)."""
    frames = []
    for result in results:
        columns: dict[str, Any] = {"run": result.run_id, "bag": result.bag, "kind": result.kind,
                                   "split": result.split, "mode": result.mode, "row": result.rows.astype(np.int64),
                                   "t": result.t}
        for name in _ARRAYS:
            values = getattr(result, name)
            for j in range(result.n_dof):
                columns[f"{name}{j}"] = values[:, j]
        residual = result.residual
        for j in range(result.n_dof):
            columns[f"res{j}"] = residual[:, j]
        for name in ("gain", "offset"):
            values = getattr(result, name)
            if values is not None:
                for j in range(result.n_dof):
                    columns[f"{name}{j}"] = float(values[j])
        frames.append(pd.DataFrame(columns))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def write_replay(results: Sequence[ReplayResult], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    replay_frame(results).to_parquet(tmp, index=False)
    tmp.replace(path)
    return path


def read_replay(path: str | Path) -> list[ReplayResult]:
    """The results :func:`write_replay` wrote (same order)."""
    frame = pd.read_parquet(path)
    n = sum(1 for c in frame.columns if c.startswith("ft_sim"))
    results = []
    for (_, _), rows in frame.groupby(["run", "bag"], sort=False):
        block = {name: rows[[f"{name}{j}" for j in range(n)]].to_numpy(dtype=float) for name in _ARRAYS}
        first = rows.iloc[0]
        constants = {name: (first[[f"{name}{j}" for j in range(n)]].to_numpy(dtype=float)
                            if f"{name}0" in rows else None) for name in ("gain", "offset")}
        results.append(ReplayResult(run_id=str(first["run"]), bag=str(first["bag"]), kind=str(first["kind"]),
                                    split=str(first["split"]), mode=str(first["mode"]),
                                    rows=rows["row"].to_numpy(dtype=np.int64), t=rows["t"].to_numpy(dtype=float),
                                    **block, **constants))
    return results


def select(results: Iterable[ReplayResult], split: str | None = None, kinds: Sequence[str] | None = None
           ) -> list[ReplayResult]:
    """The results of one split (``None`` = all) and kinds."""
    return [r for r in results if (split is None or r.split == split) and (kinds is None or r.kind in kinds)]


__all__ = ["ReplayResult", "read_replay", "replay_frame", "select", "write_replay"]
