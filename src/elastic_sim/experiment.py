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
    physical = {joint.name: joint for joint in asset.resolve_active_joints()}
    for name, (lo, hi) in limits.items():
        joint = physical[name]
        if joint.lower is not None and lo < joint.lower - 1.0e-12:
            raise ValueError(f"workspace lower bound for {name} exceeds URDF limit {joint.lower}")
        if joint.upper is not None and hi > joint.upper + 1.0e-12:
            raise ValueError(f"workspace upper bound for {name} exceeds URDF limit {joint.upper}")
    return limits


def _time_grid(duration: float, time_step: float) -> np.ndarray:
    if duration <= 0.0 or time_step <= 0.0:
        raise ValueError("duration and time_step must be positive")
    return np.arange(0.0, duration + 0.5 * time_step, time_step)


def generate_materialized_trajectory(
    asset: AssetSpec, config: Mapping[str, Any], seed: int, *, _attempt: int = 0,
    _requested_seed: int | None = None,
) -> MaterializedTrajectory:
    """Generate and collision-validate a deterministic joint or Cartesian trajectory."""
    tc = config.get("trajectory", {})
    names = tuple(asset.joint_names)
    limits = _joint_limits(asset, config)
    duration = float(tc.get("duration", config.get("simulation", {}).get("sim_time", 15.0)))
    dt = float(tc.get("time_step", config.get("simulation", {}).get("time_step", 0.01)))
    space = str(tc.get("space", "joint")).lower()
    if space not in {"joint", "cartesian"}:
        raise ValueError("trajectory.space must be 'joint' or 'cartesian'")
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
    cartesian_reference: dict[str, np.ndarray] = {}
    ik_diagnostics: dict[str, Any] = {}
    explicit_cartesian = False
    if space == "cartesian":
        from .kinematics import PortableKinematics, kinematic_groups
        kin = PortableKinematics(asset)
        cc = tc.get("cartesian", {}) or {}
        requested_groups = cc.get("groups")
        if requested_groups is not None and not isinstance(requested_groups, list):
            raise ValueError("trajectory.cartesian.groups must be a list")
        groups = kinematic_groups(asset, requested_groups)
        initial_q = np.asarray(cc.get("initial_joint_positions", asset.metadata.get("default_configuration", centre)), dtype=float)
        if initial_q.shape != centre.shape:
            raise ValueError("trajectory.cartesian.initial_joint_positions has the wrong length")
        initial_poses = kin.forward(initial_q, groups)
        explicit = cc.get("waypoints")
        explicit_cartesian = explicit is not None
        waypoint_count = max(2, int((tc.get("ptp", {}) or {}).get("waypoints", 5)))
        targets: dict[str, np.ndarray] = {}
        for group in groups:
            start = initial_poses[group.name]
            if explicit is not None:
                raw_points = explicit.get(group.name) if isinstance(explicit, Mapping) else None
                if not isinstance(raw_points, list) or len(raw_points) < 1:
                    raise ValueError(f"Cartesian waypoints require a non-empty {group.name!r} list")
                points = np.asarray([_pose_value(item, start) for item in raw_points], dtype=float)
                if not np.allclose(points[0], start, atol=1.0e-9):
                    points = np.vstack((start, points))
            elif mode == "hold":
                points = np.vstack((start, start))
            elif mode == "sin":
                workspace = _cartesian_workspace(cc, group.name, start)
                amplitude = np.minimum((workspace[:, 1] - workspace[:, 0]) * 0.25, 0.05)
                frequency = rng.uniform(0.2, 0.6, 3)
                pose_rows = np.repeat(start[None, :], n, axis=0)
                pose_rows[:, :3] += amplitude[None, :] * np.sin(time[:, None] * frequency[None, :] + rng.uniform(-np.pi, np.pi, 3))
                targets[group.name] = pose_rows
                continue
            elif mode == "ptp":
                workspace = _cartesian_workspace(cc, group.name, start)
                xyz = rng.uniform(workspace[:, 0], workspace[:, 1], size=(waypoint_count - 1, 3))
                points = np.vstack((start, np.c_[xyz, np.repeat(start[None, 3:], waypoint_count - 1, axis=0)]))
            else:
                raise ValueError(f"unsupported Cartesian trajectory mode {mode!r}")
            targets[group.name] = _interpolate_pose_waypoints(points, time)
        collision_cfg = asset.metadata.get("collision", {}) or {}
        try:
            q, ik_diagnostics = kin.solve_pose_samples(
                targets, initial_q=initial_q, dt=dt,
                position_tolerance=float(cc.get("position_tolerance", 1.0e-3)),
                orientation_tolerance=float(cc.get("orientation_tolerance", 1.0e-2)),
                max_iterations=int(cc.get("max_iterations", 80)),
                collision_margin=float(cc.get("collision_margin", collision_cfg.get("margin", 0.0))),
                collision_avoidance=bool(cc.get("collision_avoidance", False)),
            )
        except (ValueError, RuntimeError) as exc:
            retries = int(cc.get("max_retries", 4))
            if mode in {"sin", "ptp"} and not explicit_cartesian and _attempt < retries:
                return generate_materialized_trajectory(
                    asset, config, seed + 1, _attempt=_attempt + 1,
                    _requested_seed=seed if _requested_seed is None else _requested_seed,
                )
            raise ValueError(f"Cartesian trajectory generation failed after {_attempt + 1} attempt(s): {exc}") from exc
        maximum_joint_increment = float(np.max(np.abs(np.diff(q, axis=0)))) if len(q) > 1 else 0.0
        allowed_joint_increment = float(cc.get("max_joint_increment", 0.35))
        if allowed_joint_increment <= 0.0:
            raise ValueError("trajectory.cartesian.max_joint_increment must be positive")
        if maximum_joint_increment > allowed_joint_increment:
            raise ValueError(
                "Cartesian IK produced a discontinuous joint path: "
                f"increment={maximum_joint_increment:.6g}, limit={allowed_joint_increment:.6g}"
            )
        ik_diagnostics["maximum_joint_increment"] = maximum_joint_increment
        cartesian_reference = targets
    elif mode == "hold":
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
    _validate_joint_samples(asset, q)
    time, dq, ddq, stretch = _time_parameterize(q, time, tc.get("max_velocity"), tc.get("max_acceleration"))
    collision_cfg = asset.metadata.get("collision", {}) or {}
    collision_diagnostics: dict[str, Any]
    from .kinematics import PortableKinematics
    kin = kin if space == "cartesian" else PortableKinematics(asset)
    report = kin.validate_path(
        q, margin=float((tc.get("cartesian", {}) or {}).get("collision_margin", collision_cfg.get("margin", 0.0))),
        max_joint_step=float(collision_cfg.get("max_joint_step", 0.05)),
    )
    collision_diagnostics = {
        "valid": report.valid, "minimum_distance": report.minimum_distance,
        "closest_pair": report.closest_pair, "checked_configurations": report.checked_configurations,
    }
    if not report.valid:
        retries = int((tc.get("cartesian", {}) or {}).get("max_retries", tc.get("max_retries", 4)))
        if mode in {"sin", "ptp"} and not explicit_cartesian and _attempt < retries:
            return generate_materialized_trajectory(
                asset, config, seed + 1, _attempt=_attempt + 1,
                _requested_seed=seed if _requested_seed is None else _requested_seed,
            )
        raise ValueError(
            f"Generated trajectory violates self-collision margin: minimum={report.minimum_distance:.6g}, "
            f"pair={report.closest_pair}"
        )
    metadata = {
        "asset": asset.name,
        "seed": int(seed),
        "requested_seed": int(seed if _requested_seed is None else _requested_seed),
        "generation_attempt": _attempt + 1,
        "mode": mode,
        "space": space,
        "workspace": {name: list(limits[name]) for name in names},
        "time_step": dt,
        "duration": float(time[-1]),
        "generator_config": dict(tc),
        "requested_speed": None if tc.get("max_velocity") is None else float(tc["max_velocity"]),
        "effective_speed": float(np.max(np.abs(dq))) if len(dq) else 0.0,
        "time_scale": stretch,
        "kinematic_groups": [group.name for group in kin.groups],
        "ik": ik_diagnostics,
        "collision": collision_diagnostics,
    }
    if cartesian_reference:
        metadata["cartesian_reference"] = {name: values.tolist() for name, values in cartesian_reference.items()}
    return MaterializedTrajectory(time, q, dq, names, ddq, metadata)


def _pose_value(value: Any, default: np.ndarray) -> np.ndarray:
    if isinstance(value, Mapping):
        position = value.get("position", default[:3])
        orientation = value.get("orientation", default[3:])
        pose = np.r_[np.asarray(position, dtype=float), np.asarray(orientation, dtype=float)]
    else:
        pose = np.asarray(value, dtype=float)
    if pose.shape not in {(3,), (7,)}:
        raise ValueError("Cartesian poses must contain XYZ or XYZ+quaternion")
    if pose.shape == (3,):
        pose = np.r_[pose, default[3:]]
    norm = np.linalg.norm(pose[3:])
    if not np.isfinite(pose).all() or norm < 1.0e-12:
        raise ValueError("Cartesian pose must be finite with a non-zero quaternion")
    pose[3:] /= norm
    return pose


def _cartesian_workspace(config: Mapping[str, Any], group: str, start: np.ndarray) -> np.ndarray:
    raw = config.get("workspace", {}) or {}
    value = raw.get(group, raw) if isinstance(raw, Mapping) else raw
    if isinstance(value, Mapping) and "position" in value:
        value = value["position"]
    if isinstance(value, Mapping) and all(axis in value for axis in "xyz"):
        bounds = np.asarray([value[axis] for axis in "xyz"], dtype=float)
    elif value:
        bounds = np.asarray(value, dtype=float)
    else:
        bounds = np.column_stack((start[:3] - 0.04, start[:3] + 0.04))
    if bounds.shape != (3, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError(f"Cartesian workspace for {group!r} must be x/y/z bounds")
    return bounds


def _interpolate_pose_waypoints(points: np.ndarray, time: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation, Slerp
    output = np.empty((len(time), 7), dtype=float)
    duration = float(time[-1])
    segment = duration / (len(points) - 1)
    for row, sample_time in enumerate(time):
        index = min(int(sample_time / segment), len(points) - 2)
        u = np.clip((sample_time - index * segment) / segment, 0.0, 1.0)
        smooth = 10 * u**3 - 15 * u**4 + 6 * u**5
        output[row, :3] = points[index, :3] + smooth * (points[index + 1, :3] - points[index, :3])
        rotations = Rotation.from_quat(points[index:index + 2, 3:])
        output[row, 3:] = Slerp([0.0, 1.0], rotations)([smooth]).as_quat()[0]
    return output


def _time_parameterize(q: np.ndarray, time: np.ndarray, velocity_cap: Any, acceleration_cap: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    edge_order = 2 if len(time) >= 3 else 1
    dq = np.gradient(q, time, axis=0, edge_order=edge_order)
    ddq = np.gradient(dq, time, axis=0, edge_order=edge_order)
    stretch = 1.0
    if velocity_cap is not None:
        cap = float(velocity_cap)
        if cap <= 0.0:
            raise ValueError("trajectory.max_velocity must be positive")
        stretch = max(stretch, float(np.max(np.abs(dq))) / cap)
    if acceleration_cap is not None:
        cap = float(acceleration_cap)
        if cap <= 0.0:
            raise ValueError("trajectory.max_acceleration must be positive")
        stretch = max(stretch, np.sqrt(float(np.max(np.abs(ddq))) / cap))
    scaled_time = time * stretch
    return scaled_time, dq / stretch, ddq / (stretch * stretch), stretch


def _validate_joint_samples(asset: AssetSpec, q: np.ndarray) -> None:
    values = np.asarray(q, dtype=float)
    for index, joint in enumerate(asset.resolve_active_joints()):
        if joint.lower is not None and float(np.min(values[:, index])) < joint.lower - 1.0e-8:
            raise ValueError(f"trajectory violates lower limit of {joint.name}")
        if joint.upper is not None and float(np.max(values[:, index])) > joint.upper + 1.0e-8:
            raise ValueError(f"trajectory violates upper limit of {joint.name}")


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
