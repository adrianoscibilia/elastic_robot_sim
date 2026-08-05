"""Load the locally stored KUKA and Baxter ground-truth datasets.

All public loaders produce :class:`TorqueReplayRollout` objects in SI units
and keep separate motor and link encoders when the source exposes both.  This
is intentionally a narrow adapter layer: the generic calibration stack stays
robot-neutral and consumes only the canonical rollout protocol.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .generic_calibration import TorqueReplayRollout


def load_asset_rollouts(
    path: str | Path,
    *,
    joint_names: Sequence[str] | None = None,
    dataset_format: str | None = None,
    sample_time_s: float | None = None,
) -> list[TorqueReplayRollout]:
    """Load raw KUKA MAT or Baxter whitespace-CSV data from a file/directory."""
    source = Path(path).expanduser().resolve()
    kind = (dataset_format or _infer_format(source)).lower()
    if kind == "kuka_raw_mat":
        files = sorted(source.glob("recording_*.mat")) if source.is_dir() else [source]
        if not files:
            raise FileNotFoundError(f"No KUKA recording_*.mat files found in {source}")
        names = tuple(joint_names or ("a1", "a2", "a3", "a4", "a5", "a6"))
        if len(names) != 6:
            raise ValueError("KUKA KR300 data requires six joint names")
        return [_load_kuka_mat(item, names) for item in files]
    if kind == "baxter_csv":
        files = sorted(source.glob("*.csv")) if source.is_dir() else [source]
        if not files:
            raise FileNotFoundError(f"No Baxter CSV files found in {source}")
        if sample_time_s is None or sample_time_s <= 0.0:
            raise ValueError("Baxter CSV has no timestamps; sample_time_s must be supplied")
        names = tuple(joint_names or ("left_s0", "left_s1", "left_e0", "left_e1", "left_w0", "left_w1", "left_w2"))
        if len(names) != 7:
            raise ValueError("Baxter data requires seven joint names")
        return [_load_baxter_csv(item, names, sample_time_s) for item in files]
    raise ValueError(f"Unsupported asset dataset format {kind!r}")


def rollout_to_dataframe(rollout: TorqueReplayRollout, *, bag: str | None = None) -> pd.DataFrame:
    """Serialize one rollout to the canonical, calibration-ready CSV schema."""
    data: dict[str, np.ndarray | str] = {"time": rollout.time}
    for prefix, values in (("q", rollout.q), ("dq", rollout.dq), ("tau", rollout.tau)):
        for index in range(rollout.n_dof):
            data[f"{prefix}{index}"] = values[:, index]
    for prefix, values in (("q_motor", rollout.motor_q), ("dq_motor", rollout.motor_dq)):
        if values is not None:
            for index in range(rollout.n_dof):
                data[f"{prefix}{index}"] = values[:, index]
    if bag is not None:
        data["bag"] = bag
    return pd.DataFrame(data)


def _load_kuka_mat(path: Path, joint_names: tuple[str, ...]) -> TorqueReplayRollout:
    try:
        from scipy.io import loadmat
    except ImportError as exc:  # pragma: no cover - optional acquisition dependency
        raise ImportError("Reading KUKA MATLAB data requires scipy (pip install scipy)") from exc
    raw = loadmat(path, squeeze_me=True)
    required = ("q_mot_meas", "q_se_meas", "qd_mot_meas", "qd_se_meas", "tau_meas", "time")
    missing = [name for name in required if name not in raw]
    if missing:
        raise ValueError(f"{path} is not a KUKA raw recording; missing {missing}")
    radians = np.pi / 180.0
    time = np.asarray(raw["time"], dtype=float).reshape(-1)
    return TorqueReplayRollout(
        time=time,
        q=np.asarray(raw["q_se_meas"], dtype=float).T * radians,
        dq=np.asarray(raw["qd_se_meas"], dtype=float).T * radians,
        tau=np.asarray(raw["tau_meas"], dtype=float).T,
        joint_names=joint_names,
        motor_q=np.asarray(raw["q_mot_meas"], dtype=float).T * radians,
        motor_dq=np.asarray(raw["qd_mot_meas"], dtype=float).T * radians,
        link_q=np.asarray(raw["q_se_meas"], dtype=float).T * radians,
        link_dq=np.asarray(raw["qd_se_meas"], dtype=float).T * radians,
        metadata={"source": str(path), "format": "kuka_raw_mat", "position_units": "rad"},
    )


def _load_baxter_csv(path: Path, joint_names: tuple[str, ...], sample_time_s: float) -> TorqueReplayRollout:
    frame = pd.read_csv(path, sep=r"\s+")
    expected = [f"{prefix}_{name}" for prefix in ("ang", "vel", "torq") for name in ("s0", "s1", "e0", "e1", "w0", "w1", "w2")]
    missing = [name for name in expected if name not in frame]
    if missing:
        raise ValueError(f"{path} is not a Baxter dynamics CSV; missing {missing}")
    suffixes = ("s0", "s1", "e0", "e1", "w0", "w1", "w2")
    q = frame[[f"ang_{name}" for name in suffixes]].to_numpy(float)
    dq = frame[[f"vel_{name}" for name in suffixes]].to_numpy(float)
    tau = frame[[f"torq_{name}" for name in suffixes]].to_numpy(float)
    return TorqueReplayRollout(
        time=np.arange(len(frame), dtype=float) * sample_time_s,
        q=q,
        dq=dq,
        tau=tau,
        joint_names=joint_names,
        metadata={"source": str(path), "format": "baxter_csv", "sample_time_s": sample_time_s},
    )


def _infer_format(path: Path) -> str:
    if path.is_dir() or path.suffix.lower() == ".mat":
        return "kuka_raw_mat" if "kuka" in path.as_posix().lower() or path.is_dir() else "kuka_raw_mat"
    if path.suffix.lower() == ".csv":
        return "baxter_csv"
    raise ValueError(f"Cannot infer asset dataset format from {path}")
