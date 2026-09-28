"""RolloutResult schema and RolloutStore for saving/loading rollouts.

Both the sim and the real-robot recorder produce RolloutResult objects.
The store layout mirrors the plan:

    data/rollouts/<traj_id>/
        trajectory.json
        real.parquet
        sim_best.parquet
        meta.json
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .trajectory import TrajectoryConfig


@dataclass
class RolloutResult:
    """Time-series data from one simulation or real-robot execution."""

    time: np.ndarray       # (N,)
    ref_pos: np.ndarray    # (N, 3)  reference positions [x, y, z]
    ref_vel: np.ndarray    # (N, 3)  reference velocities
    q_motor: np.ndarray    # (N, 3)  motor-side positions
    q_link: np.ndarray     # (N, 3)  elastic-link positions
    dq_motor: np.ndarray   # (N, 3)
    dq_link: np.ndarray    # (N, 3)
    tau_motor: np.ndarray  # (N, 3)
    tau_link: np.ndarray   # (N, 3)
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # DataFrame conversion (matches sim_common.CSV_COLUMNS)
    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        data = {
            "t": self.time,
            "ref_x": self.ref_pos[:, 0],
            "ref_y": self.ref_pos[:, 1],
            "ref_z": self.ref_pos[:, 2],
            "vel_x": self.ref_vel[:, 0],
            "vel_y": self.ref_vel[:, 1],
            "vel_z": self.ref_vel[:, 2],
            "q_motor_x": self.q_motor[:, 0],
            "q_link_x": self.q_link[:, 0],
            "q_motor_y": self.q_motor[:, 1],
            "q_link_y": self.q_link[:, 1],
            "q_motor_z": self.q_motor[:, 2],
            "q_link_z": self.q_link[:, 2],
            "dq_motor_x": self.dq_motor[:, 0],
            "dq_link_x": self.dq_link[:, 0],
            "dq_motor_y": self.dq_motor[:, 1],
            "dq_link_y": self.dq_link[:, 1],
            "dq_motor_z": self.dq_motor[:, 2],
            "dq_link_z": self.dq_link[:, 2],
            "tau_motor_x": self.tau_motor[:, 0],
            "tau_link_x": self.tau_link[:, 0],
            "tau_motor_y": self.tau_motor[:, 1],
            "tau_link_y": self.tau_link[:, 1],
            "tau_motor_z": self.tau_motor[:, 2],
            "tau_link_z": self.tau_link[:, 2],
        }
        return pd.DataFrame(data)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, metadata: dict | None = None) -> RolloutResult:
        t = df["t"].to_numpy()
        n = len(t)

        def _col(name: str) -> np.ndarray:
            return df[name].to_numpy() if name in df.columns else np.zeros(n)

        return cls(
            time=t,
            ref_pos=np.column_stack([_col("ref_x"), _col("ref_y"), _col("ref_z")]),
            ref_vel=np.column_stack([_col("vel_x"), _col("vel_y"), _col("vel_z")]),
            q_motor=np.column_stack([_col("q_motor_x"), _col("q_motor_y"), _col("q_motor_z")]),
            q_link=np.column_stack([_col("q_link_x"), _col("q_link_y"), _col("q_link_z")]),
            dq_motor=np.column_stack([_col("dq_motor_x"), _col("dq_motor_y"), _col("dq_motor_z")]),
            dq_link=np.column_stack([_col("dq_link_x"), _col("dq_link_y"), _col("dq_link_z")]),
            tau_motor=np.column_stack([_col("tau_motor_x"), _col("tau_motor_y"), _col("tau_motor_z")]),
            tau_link=np.column_stack([_col("tau_link_x"), _col("tau_link_y"), _col("tau_link_z")]),
            metadata=metadata or {},
        )

    # ------------------------------------------------------------------
    # Resampling onto a common time grid
    # ------------------------------------------------------------------

    def resample(self, target_times: np.ndarray) -> RolloutResult:
        """Interpolate all signals onto target_times (1-D, increasing)."""
        arrays = [
            self.ref_pos, self.ref_vel,
            self.q_motor, self.q_link,
            self.dq_motor, self.dq_link,
            self.tau_motor, self.tau_link,
        ]
        resampled = [
            np.column_stack([
                np.interp(target_times, self.time, col[:, i])
                for i in range(col.shape[1])
            ])
            for col in arrays
        ]
        return RolloutResult(
            time=target_times,
            ref_pos=resampled[0],
            ref_vel=resampled[1],
            q_motor=resampled[2],
            q_link=resampled[3],
            dq_motor=resampled[4],
            dq_link=resampled[5],
            tau_motor=resampled[6],
            tau_link=resampled[7],
            metadata=dict(self.metadata),
        )


# ---------------------------------------------------------------------------
# RolloutStore
# ---------------------------------------------------------------------------

class RolloutStore:
    """Save and load rollouts under data/rollouts/<traj_id>/."""

    def __init__(self, base_dir: str | None = None) -> None:
        if base_dir is None:
            base_dir = os.path.join(
                os.path.dirname(__file__), "..", "..", "data", "rollouts"
            )
        self.base_dir = os.path.abspath(base_dir)

    def _rollout_dir(self, traj_id: str) -> str:
        return os.path.join(self.base_dir, traj_id)

    def _ensure_dir(self, traj_id: str) -> str:
        path = self._rollout_dir(traj_id)
        os.makedirs(path, exist_ok=True)
        return path

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_trajectory(self, traj_id: str, config: TrajectoryConfig) -> None:
        d = self._ensure_dir(traj_id)
        config.save(os.path.join(d, "trajectory.json"))

    def save_real(self, traj_id: str, rollout: RolloutResult) -> None:
        d = self._ensure_dir(traj_id)
        rollout.to_dataframe().to_parquet(os.path.join(d, "real.parquet"), index=False)
        if rollout.metadata:
            with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(rollout.metadata, f, indent=2)

    def save_sim(
        self,
        traj_id: str,
        rollout: RolloutResult,
        backend: str = "newton",
        *,
        calibrated: bool = False,
    ) -> None:
        """Save a sim rollout.  File name: <backend>[_calibrated].parquet."""
        d = self._ensure_dir(traj_id)
        suffix = "_calibrated" if calibrated else ""
        fname = f"{backend}{suffix}.parquet"
        rollout.to_dataframe().to_parquet(os.path.join(d, fname), index=False)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load_trajectory(self, traj_id: str) -> TrajectoryConfig:
        path = os.path.join(self._rollout_dir(traj_id), "trajectory.json")
        return TrajectoryConfig.load(path)

    def load_real(self, traj_id: str) -> RolloutResult:
        path = os.path.join(self._rollout_dir(traj_id), "real.parquet")
        df = pd.read_parquet(path)
        meta_path = os.path.join(self._rollout_dir(traj_id), "meta.json")
        meta: dict = {}
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
        return RolloutResult.from_dataframe(df, metadata=meta)

    def load_sim(
        self,
        traj_id: str,
        backend: str = "newton",
        *,
        calibrated: bool = False,
    ) -> RolloutResult:
        """Load a sim rollout.  backend="newton"|"mujoco"; calibrated=True for post-calibration."""
        suffix = "_calibrated" if calibrated else ""
        path = os.path.join(self._rollout_dir(traj_id), f"{backend}{suffix}.parquet")
        return RolloutResult.from_dataframe(pd.read_parquet(path))

    def list_traj_ids(self) -> list[str]:
        if not os.path.isdir(self.base_dir):
            return []
        return sorted(
            e for e in os.listdir(self.base_dir)
            if os.path.isdir(os.path.join(self.base_dir, e))
        )

    def has_real(self, traj_id: str) -> bool:
        return os.path.exists(
            os.path.join(self._rollout_dir(traj_id), "real.parquet")
        )

    def has_sim(self, traj_id: str, backend: str = "newton", *, calibrated: bool = False) -> bool:
        suffix = "_calibrated" if calibrated else ""
        return os.path.exists(
            os.path.join(self._rollout_dir(traj_id), f"{backend}{suffix}.parquet")
        )

    def available_backends(self, traj_id: str) -> list[str]:
        """Return which backend rollout files exist (excluding 'real')."""
        d = self._rollout_dir(traj_id)
        found = []
        for name in ("newton", "mujoco"):
            if os.path.exists(os.path.join(d, f"{name}.parquet")):
                found.append(name)
        return found
