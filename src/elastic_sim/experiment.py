"""Shared sim-to-real experiment configuration, trajectories, and artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from .assets import AssetSpec, load_asset_spec
from .materialized import MaterializedTrajectory


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    cfg = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"experiment config must be a mapping: {source}")
    cfg["_config_path"] = str(source)
    return cfg


def resolve_asset(config: Mapping[str, Any], config_path: str | Path | None = None) -> AssetSpec:
    raw = config.get("asset")
    if not raw:
        raise ValueError("experiment config requires asset")
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        base = Path(config_path or config.get("_config_path", Path.cwd())).resolve().parent
        path = (base / path).resolve()
    if path.is_file():
        return load_asset_spec(path)
    from .assets import AssetRegistry
    return AssetRegistry.for_repository().load(str(raw))


def _joint_limits(asset: AssetSpec, config: Mapping[str, Any]) -> dict[str, tuple[float, float]]:
    requested = config.get("trajectory", {}).get("workspace") or config.get("workspace")
    if requested is not None:
        if set(requested) != set(asset.joint_names):
            raise ValueError("trajectory.workspace must define exactly the asset active joints")
        limits = {name: (float(value[0]), float(value[1])) for name, value in requested.items()}
    else:
        from .serial_trajectory import _urdf_joint_limits
        names, lower, upper = _urdf_joint_limits(asset.urdf_path, asset.joint_names)
        limits = {name: (float(lower[i]), float(upper[i])) for i, name in enumerate(names)}
    for name, (lo, hi) in limits.items():
        if name not in asset.joint_names or not np.isfinite([lo, hi]).all() or lo >= hi:
            raise ValueError(f"invalid workspace for {name}: {(lo, hi)}")
    return limits


def _time_grid(duration: float, time_step: float) -> np.ndarray:
    if duration <= 0.0 or time_step <= 0.0:
        raise ValueError("duration and time_step must be positive")
    return np.arange(0.0, duration + 0.5 * time_step, time_step)


def generate_materialized_trajectory(asset: AssetSpec, config: Mapping[str, Any], seed: int) -> MaterializedTrajectory:
    """Generate a deterministic joint-space trajectory from YAML."""
    tc = config.get("trajectory", {})
    names = tuple(asset.joint_names)
    limits = _joint_limits(asset, config)
    duration = float(tc.get("duration", config.get("simulation", {}).get("sim_time", 15.0)))
    dt = float(tc.get("time_step", config.get("simulation", {}).get("time_step", 0.01)))
    mode = tc.get("mode", tc.get("modes", "ptp"))
    if isinstance(mode, (list, tuple)):
        mode = mode[int(np.random.default_rng(seed).integers(0, len(mode)))]
    mode = {0: "hold", 1: "sin", 2: "ptp"}.get(mode, mode)
    mode = str(mode).lower()
    if mode in {"sinusoidal", "sine"}:
        mode = "sin"
    rng = np.random.default_rng(seed)
    time = _time_grid(duration, dt)
    n = len(time)
    lower = np.asarray([limits[name][0] for name in names])
    upper = np.asarray([limits[name][1] for name in names])
    centre = (lower + upper) / 2.0
    span = upper - lower
    if mode == "hold":
        q = np.repeat(centre[None, :], n, axis=0)
        dq = np.zeros_like(q)
        ddq = np.zeros_like(q)
    elif mode == "sin":
        sin_cfg = tc.get("sinusoidal", {})
        amp_fraction = float(sin_cfg.get("amplitude_fraction", 0.25))
        amp_min = float(sin_cfg.get("amplitude_min", np.min(span) * 0.05))
        fmin = float(sin_cfg.get("frequency_min", sin_cfg.get("freq_min", 0.2)))
        fmax = float(sin_cfg.get("frequency_max", sin_cfg.get("freq_max", 1.5)))
        amp_upper = np.minimum(span * amp_fraction, 0.49 * span)
        amp_lower = np.minimum(np.maximum(0.0, amp_min), amp_upper)
        amp = rng.uniform(np.maximum(amp_lower, 0.25 * amp_upper), amp_upper)
        freq = rng.uniform(fmin, fmax, size=len(names))
        phase = rng.uniform(-np.pi, np.pi, size=len(names))
        q = centre + amp[None, :] * np.sin(time[:, None] * freq[None, :] + phase[None, :])
        dq = amp[None, :] * freq[None, :] * np.cos(time[:, None] * freq[None, :] + phase[None, :])
        ddq = -amp[None, :] * freq[None, :] ** 2 * np.sin(time[:, None] * freq[None, :] + phase[None, :])
    elif mode == "ptp":
        ptp_cfg = tc.get("ptp", {})
        count = max(2, int(ptp_cfg.get("waypoints", 1 + np.ceil(duration / float(ptp_cfg.get("step_duration", 2.0))))))
        margin = float(ptp_cfg.get("limit_margin", 0.08))
        if not 0.0 <= margin < 0.5:
            raise ValueError("trajectory.ptp.limit_margin must be in [0, 0.5)")
        points = rng.uniform(lower + margin * span, upper - margin * span, size=(count, len(names)))
        segment = duration / (count - 1)
        q = np.empty((n, len(names)))
        dq = np.empty_like(q)
        ddq = np.empty_like(q)
        for row, t in enumerate(time):
            index = min(int(t / segment), count - 2)
            u = np.clip((t - index * segment) / segment, 0.0, 1.0)
            s = 10 * u**3 - 15 * u**4 + 6 * u**5
            ds = (30 * u**2 - 60 * u**3 + 30 * u**4) / segment
            dds = (60 * u - 180 * u**2 + 120 * u**3) / segment**2
            delta = points[index + 1] - points[index]
            q[row] = points[index] + s * delta
            dq[row] = ds * delta
            ddq[row] = dds * delta
    else:
        raise ValueError(f"unsupported trajectory mode {mode!r}")
    cap = tc.get("max_velocity")
    if cap is not None and float(np.max(np.abs(dq))) > float(cap):
        factor = float(cap) / float(np.max(np.abs(dq)))
        dq *= factor
        ddq *= factor * factor
        # Position samples remain unchanged; the exact saved samples are the source of truth.
    metadata = {
        "asset": asset.name,
        "seed": int(seed),
        "mode": mode,
        "workspace": {name: list(limits[name]) for name in names},
        "time_step": dt,
        "duration": duration,
        "generator_config": dict(tc),
        "requested_speed": None if tc.get("max_velocity") is None else float(tc["max_velocity"]),
        "effective_speed": float(np.max(np.abs(dq))) if len(dq) else 0.0,
    }
    return MaterializedTrajectory(time, q, dq, names, ddq, metadata)


def rollout_to_frame(trajectory: MaterializedTrajectory, result: Any, *, source: str) -> pd.DataFrame:
    """Convert fixed or generic simulator output to the common wide schema."""
    if hasattr(result, "time"):
        values = {"time": result.time, "q": result.q_link, "dq": result.dq_link,
                  "q_motor": result.q_motor, "dq_motor": result.dq_motor,
                  "tau_motor": result.tau_motor, "tau_link": result.tau_link,
                  "force_flange": getattr(result, "force_flange", None),
                  "torque_flange": getattr(result, "torque_flange", None)}
    else:
        values = {"time": np.asarray(result["time"]), "q": np.asarray(result.get("q_link", result.get("q"))),
                  "dq": np.asarray(result.get("dq_link", result.get("dq"))),
                  "q_motor": result.get("q_motor"), "dq_motor": result.get("dq_motor"),
                  "tau_motor": result.get("tau_motor"), "tau_link": result.get("tau_link"),
                  "force_flange": result.get("force_flange"), "torque_flange": result.get("torque_flange"),
                  "force_link_side": result.get("force_link_side"), "torque_link_side": result.get("torque_link_side")}
    result_time = np.asarray(values["time"], dtype=float)
    if result_time.ndim != 1 or len(result_time) < 2 or not np.isfinite(result_time).all():
        raise ValueError("rollout result time must be a finite one-dimensional array")
    frame = pd.DataFrame({"t": result_time, "source": source})
    # FMRR is translational: its simulated link-side generalized force is the
    # XYZ flange-force channel under the explicit asset mapping.  Preserve
    # this mapping in the normalized schema rather than applying it to every
    # serial robot's joint torque.
    if tuple(trajectory.joint_names) == ("joint_x", "joint_y", "joint_z") and values.get("tau_link") is not None:
        if values.get("force_link_side") is None:
            values["force_link_side"] = values["tau_link"]
        if values.get("force_flange") is None:
            values["force_flange"] = values["tau_link"]
    planned = trajectory.sample()
    for prefix, array in (("q_ref", planned["q"]), ("dq_ref", planned["dq"]), ("ddq_ref", planned["ddq"]),
                          ("q", values["q"]), ("dq", values["dq"]), ("q_motor", values["q_motor"]),
                          ("dq_motor", values["dq_motor"]), ("tau_motor", values["tau_motor"]),
                          ("tau_link", values["tau_link"])):
        if array is None:
            continue
        array = np.asarray(array, dtype=float)
        if len(array) != len(frame):
            if array.ndim != 2 or array.shape[1] != trajectory.n_dof:
                raise ValueError(f"rollout field {prefix!r} must have shape (N, {trajectory.n_dof})")
            source_time = trajectory.time if prefix in {"q_ref", "dq_ref", "ddq_ref"} else result_time
            array = np.column_stack([np.interp(frame.t, source_time, array[:, i]) for i in range(array.shape[1])])
        for index, name in enumerate(trajectory.joint_names):
            frame[f"{prefix}__{name}"] = array[:, index]
    for prefix, array in (("force_flange", values.get("force_flange")), ("torque_flange", values.get("torque_flange")),
                          ("force_link_side", values.get("force_link_side")), ("torque_link_side", values.get("torque_link_side"))):
        if array is None:
            continue
        array = np.asarray(array, dtype=float)
        if array.ndim != 2 or array.shape[1] != 3:
            raise ValueError(f"{prefix} must have shape (N, 3)")
        if len(array) != len(frame):
            array = np.column_stack([np.interp(frame.t, result_time, array[:, i]) for i in range(3)])
        for index, axis in enumerate("xyz"):
            frame[f"{prefix}__{axis}"] = array[:, index]
    return frame


class ExperimentStore:
    """Filesystem layout for one asset/date/run experiment."""

    def __init__(self, root: str | Path, asset: str, run_id: str | None = None) -> None:
        now = datetime.now(timezone.utc)
        self.root = Path(root).expanduser().resolve()
        if self.root.name not in {"simulated", "recorded", "calibrations"}:
            raise ValueError("artifact root must end in simulated, recorded, or calibrations")
        if not asset or Path(asset).name != asset or asset in {".", ".."}:
            raise ValueError(f"invalid asset path component: {asset!r}")
        self.asset = asset
        self.date = now.strftime("%Y-%m-%d")
        self.run_id = run_id or now.strftime("%Y%m%dT%H%M%SZ")
        if Path(self.run_id).name != self.run_id or self.run_id in {".", ".."}:
            raise ValueError(f"invalid run-id path component: {self.run_id!r}")
        self.path = self.root / asset / self.date / self.run_id
        self.path.mkdir(parents=True, exist_ok=True)

    def save_trajectory(self, trajectory: MaterializedTrajectory, name: str = "trajectory.json") -> Path:
        path = self.path / name
        trajectory.save(path)
        return path

    def save_frame(self, frame: pd.DataFrame, name: str) -> Path:
        path = self.path / name
        frame.to_parquet(path, index=False)
        return path

    def save_manifest(self, manifest: Mapping[str, Any]) -> Path:
        path = self.path / "manifest.yaml"
        path.write_text(yaml.safe_dump(dict(manifest), sort_keys=False), encoding="utf-8")
        return path


def artifact_root(
    config: Mapping[str, Any], kind: str, repository_root: str | Path, override: str | Path | None = None,
) -> Path:
    """Resolve and validate one of the three non-overlapping artifact roots."""
    if kind not in {"simulated", "recorded", "calibrations"}:
        raise ValueError(f"unsupported artifact kind {kind!r}")
    configured = (config.get("paths", {}) or {}).get(f"{kind}_root", f"data/{kind}")
    value = Path(override if override is not None else configured).expanduser()
    root = value.resolve() if value.is_absolute() else (Path(repository_root).resolve() / value).resolve()
    if root.name != kind:
        raise ValueError(f"{kind} artifact root must end with '/{kind}', got {root}")
    return root

def config_digest(config: Mapping[str, Any]) -> str:
    clean = {key: value for key, value in config.items() if not str(key).startswith("_")}
    payload = json.dumps(clean, sort_keys=True, default=str, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def start_rosbag(output_dir: str | Path, topics: list[str]) -> subprocess.Popen:
    """Start rosbag2 without shell expansion and without pipe backpressure."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    command = ["ros2", "bag", "record", "-o", str(target / "rosbag2"), *topics]
    stdout = (target / "rosbag2.stdout.log").open("w", encoding="utf-8")
    stderr = (target / "rosbag2.stderr.log").open("w", encoding="utf-8")
    return subprocess.Popen(command, stdout=stdout, stderr=stderr, text=True)
