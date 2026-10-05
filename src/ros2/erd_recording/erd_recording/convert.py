"""raw.parquet -> dataset (RR_01 S8.2): per-robot column derivation + writing.

Both robots produce the same final columns (``t, bag, split, q.., dq.., tau..,
ft..``); only how ``dq``/``tau``/``ft`` are derived from the raw recording
differs (RR_01 S2).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from .identification import MotorSideFit, friction_torque


def _prefixed(frame: pd.DataFrame, prefix: str, n_dof: int) -> np.ndarray:
    return frame[[f"{prefix}{i}" for i in range(n_dof)]].to_numpy(dtype=float)


def _assign_prefixed(frame: pd.DataFrame, prefix: str, values: np.ndarray) -> None:
    for i in range(values.shape[1]):
        frame[f"{prefix}{i}"] = values[:, i]


def iiwa_dataset_frame(raw: pd.DataFrame, *, n_dof: int, sg_window: int, sg_poly: int, time_step: float) -> pd.DataFrame:
    """iiwa: ``tau``/``ft`` are already the right signals (FRI commanded/measured
    torque, raw); only ``dq`` is derived, per bag, exactly as
    ``dataset_bag.py`` does (RR_01 S1.4/S2)."""
    frame = raw[["t", "bag", "split"]].copy()
    q = _prefixed(raw, "q", n_dof)
    _assign_prefixed(frame, "q", q)
    dq = np.empty_like(q)
    for bag, group in raw.groupby("bag", sort=False):
        index = group.index
        q_bag = _prefixed(group, "q", n_dof)
        dq[index] = savgol_filter(q_bag, int(sg_window), int(sg_poly), deriv=1, delta=time_step, axis=0, mode="interp")
    _assign_prefixed(frame, "dq", dq)
    _assign_prefixed(frame, "tau", _prefixed(raw, "tau", n_dof))
    _assign_prefixed(frame, "ft", _prefixed(raw, "ft", n_dof))
    return frame


def ur10_dataset_frame(
    raw: pd.DataFrame, *, n_dof: int, k_tau: np.ndarray, motor_fit: MotorSideFit, epsilon: float,
    sg_window: int, sg_poly: int, time_step: float, tau_source_column: str = "i_cmd",
) -> pd.DataFrame:
    """UR10: ``tau = K_tau * i_cmd``; ``ft = K_tau * i_act - J_m ddq - f(dq)``
    (RR_02 CR-3, the Q-3a proxy). ``dq`` is the drive estimate (``actual_qd``,
    carried through unchanged); ``ddq`` (used only to build ``ft``, not
    written) is the SG derivative of that ``dq``."""
    frame = raw[["t", "bag", "split"]].copy()
    q = _prefixed(raw, "q", n_dof)
    dq = _prefixed(raw, "dq", n_dof)
    i_cmd = _prefixed(raw, tau_source_column, n_dof)
    i_act = _prefixed(raw, "i_act", n_dof)
    _assign_prefixed(frame, "q", q)
    _assign_prefixed(frame, "dq", dq)
    tau = k_tau[None, :] * i_cmd
    _assign_prefixed(frame, "tau", tau)

    ddq = np.empty_like(dq)
    for bag, group in raw.groupby("bag", sort=False):
        index = group.index
        dq_bag = _prefixed(group, "dq", n_dof)
        ddq[index] = savgol_filter(dq_bag, int(sg_window), int(sg_poly), deriv=1, delta=time_step, axis=0, mode="interp")
    friction = friction_torque(dq, {'viscous': motor_fit.viscous, 'coulomb': motor_fit.coulomb,
                                  'stribeck': motor_fit.stribeck,
                                  'stribeck_velocity': motor_fit.stribeck_velocity}, epsilon)
    ft = k_tau[None, :] * i_act - motor_fit.j_m[None, :] * ddq - friction
    _assign_prefixed(frame, "ft", ft)
    return frame


def write_real_dataset(
    frame: pd.DataFrame, manifest: Mapping[str, Any], contract: Mapping[str, Any], output: str | Path,
) -> tuple[Path, Path, Path]:
    """Write parquet + manifest via ``elastic_sim.dataset_io.write_dataset``
    with ``schema_version < 3`` (so ``_round6_contract`` is never invoked),
    then overwrite the contract with :func:`erd_recording.contract.
    write_real_contract` (RR_01 S8.2) -- **atomically** (RR_04 C-2): both
    writes happen in ``<output's parent>/.tmp/``, and only once *both* have
    succeeded are the three files renamed into their final location. A crash
    between the two (the pass-1 behaviour: `write_dataset` succeeds, then the
    process dies before the contract overwrite) previously left a real
    dataset with a misleading ``/2`` contract claiming simulation semantics;
    now nothing appears at the final path until the whole write has
    completed, since ``Path.replace`` is an atomic rename on the same
    filesystem.
    """
    import shutil

    from elastic_sim.dataset_io import write_dataset

    from .contract import write_real_contract

    manifest = dict(manifest)
    manifest.setdefault("schema_version", 2)
    if manifest["schema_version"] >= 3:
        raise ValueError("write_real_dataset: manifest schema_version must be < 3 (RR_02 CR-4)")

    output = Path(output).expanduser()
    tmp_dir = output.parent / ".tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    tmp_csv_path, tmp_manifest_path, _ = write_dataset(frame, manifest, tmp_dir / output.name)
    # The generic simulation writer downcasts every signal to float32.
    # Real encoder quantization can be finer than float32, and casting q
    # after deriving dq breaks the contract on the file itself (RR_08).
    if tmp_csv_path.suffix == ".parquet":
        frame.to_parquet(tmp_csv_path, index=False)
    tmp_contract_path = tmp_csv_path.with_suffix(".contract.json")
    write_real_contract(tmp_csv_path, contract)

    output.parent.mkdir(parents=True, exist_ok=True)
    final_paths = []
    for tmp_path in (tmp_csv_path, tmp_manifest_path, tmp_contract_path):
        final_path = output.parent / tmp_path.name
        tmp_path.replace(final_path)
        final_paths.append(final_path)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return tuple(final_paths)  # type: ignore[return-value]
