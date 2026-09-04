"""Common native-viewer adapters with Cartesian trajectory overlays."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from .assets import AssetSpec
from .kinematics import PortableKinematics, cartesian_samples


_COLORS = ((0.15, 0.65, 1.0), (1.0, 0.45, 0.15), (0.4, 0.9, 0.35))
_FRAME_COLORS = ((0.95, 0.15, 0.15), (0.15, 0.9, 0.25), (0.15, 0.4, 1.0))
_WARNING_COLOR = (1.0, 0.05, 0.05)


class Visualizer(Protocol):
    """Backend-neutral lifecycle implemented by native viewer adapters."""

    def is_running(self) -> bool: ...
    def close(self) -> None: ...


class NewtonVisualizer:
    """Newton ViewerGL lifecycle used by every Newton runner."""

    def __init__(self, asset: AssetSpec, trajectory: Any) -> None:
        self.asset, self.trajectory = asset, trajectory
        self.kinematics = PortableKinematics(asset)
        self.reference = cartesian_samples(self.kinematics, _trajectory_positions(trajectory))
        self.actual: dict[str, list[np.ndarray]] = {name: [] for name in self.reference}
        self.viewer = None
        self._frame_index = 0
        self._warning = False

    def open(self, model: Any) -> "NewtonVisualizer":
        import newton
        cls = getattr(getattr(newton, "viewer", None), "ViewerGL", None)
        if cls is None:
            raise RuntimeError("Newton ViewerGL is unavailable; retry with --headless")
        try:
            self.viewer = cls()
            self.viewer.set_model(model)
        except Exception as exc:
            raise RuntimeError("Newton viewer could not start; retry with --headless") from exc
        return self

    def is_running(self) -> bool:
        return self.viewer is not None and (not hasattr(self.viewer, "is_running") or self.viewer.is_running())

    def render(self, sim_time: float, state: Any, q: np.ndarray) -> None:
        if self.viewer is None:
            return
        import warp as wp
        poses = self.kinematics.forward(q)
        warning = self._collision_warning(q)
        for name, pose in poses.items():
            self.actual[name].append(pose[:3].copy())
        self.viewer.begin_frame(float(sim_time))
        self.viewer.log_state(state)
        for index, (name, reference) in enumerate(self.reference.items()):
            ref = _thin(reference[:, :3])
            if len(ref) > 1:
                self.viewer.log_lines(
                    f"trajectory/reference/{name}", wp.array(ref[:-1], dtype=wp.vec3, device=self.viewer.device),
                    wp.array(ref[1:], dtype=wp.vec3, device=self.viewer.device), _COLORS[index % len(_COLORS)], width=0.006,
                )
            actual = np.asarray(self.actual[name])
            if len(actual) > 1:
                self.viewer.log_lines(
                    f"trajectory/actual/{name}", wp.array(actual[:-1], dtype=wp.vec3, device=self.viewer.device),
                    wp.array(actual[1:], dtype=wp.vec3, device=self.viewer.device),
                    _WARNING_COLOR if warning else (1.0, 0.9, 0.2), width=0.009,
                )
            target = wp.array([reference[min(len(reference) - 1, len(actual) - 1), :3]], dtype=wp.vec3, device=self.viewer.device)
            self.viewer.log_points(
                f"trajectory/target/{name}", target, 0.018,
                _WARNING_COLOR if warning else _COLORS[index % len(_COLORS)],
            )
            starts, ends = _frame_segments(poses[name])
            self.viewer.log_lines(
                f"trajectory/frame/{name}", wp.array(starts, dtype=wp.vec3, device=self.viewer.device),
                wp.array(ends, dtype=wp.vec3, device=self.viewer.device),
                wp.array(_FRAME_COLORS, dtype=wp.vec3, device=self.viewer.device), width=0.008,
            )
        self.viewer.end_frame()

    def _collision_margin(self) -> float:
        return float(self.asset.metadata.get("collision", {}).get("margin", 0.0))

    def _collision_warning(self, q: np.ndarray) -> bool:
        # Mesh distance queries are much costlier than rendering.  Ten-frame
        # polling keeps the overlay responsive while the complete reference
        # trajectory remains exhaustively validated before the viewer opens.
        if self._frame_index % 10 == 0:
            self._warning = not self.kinematics.collision_report((q,), margin=self._collision_margin()).valid
        self._frame_index += 1
        return self._warning

    def close(self) -> None:
        if self.viewer is not None:
            close = getattr(self.viewer, "close", None)
            if callable(close):
                close()
            self.viewer = None


class MujocoVisualizer:
    """MuJoCo passive-viewer lifecycle with user-scene path geometry."""

    def __init__(self, asset: AssetSpec, trajectory: Any) -> None:
        self.asset, self.trajectory = asset, trajectory
        self.kinematics = PortableKinematics(asset)
        self.reference = cartesian_samples(self.kinematics, _trajectory_positions(trajectory))
        self.actual: dict[str, list[np.ndarray]] = {name: [] for name in self.reference}
        self.viewer = None
        self._frame_index = 0
        self._warning = False

    def open(self, model: Any, data: Any) -> "MujocoVisualizer":
        try:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(model, data)
        except Exception as exc:
            raise RuntimeError("MuJoCo viewer could not start; retry with --headless") from exc
        return self

    def is_running(self) -> bool:
        return self.viewer is not None and self.viewer.is_running()

    def render(self, q: np.ndarray) -> None:
        if self.viewer is None:
            return
        import mujoco
        poses = self.kinematics.forward(q)
        warning = self._collision_warning(q)
        for name, pose in poses.items():
            self.actual[name].append(pose[:3].copy())
        scene = self.viewer.user_scn
        scene.ngeom = 0
        for index, (name, reference) in enumerate(self.reference.items()):
            _mujoco_polyline(mujoco, scene, _thin(reference[:, :3], 180), (*_COLORS[index % len(_COLORS)], 0.8), 0.006)
            actual_color = (*(_WARNING_COLOR if warning else (1.0, 0.9, 0.2)), 1.0)
            _mujoco_polyline(mujoco, scene, _thin(np.asarray(self.actual[name]), 180), actual_color, 0.009)
            target = reference[min(len(reference) - 1, len(self.actual[name]) - 1), :3]
            _mujoco_point(
                mujoco, scene, target,
                (*(_WARNING_COLOR if warning else _COLORS[index % len(_COLORS)]), 1.0), 0.018,
            )
            starts, ends = _frame_segments(poses[name])
            for axis, (start, end) in enumerate(zip(starts, ends)):
                _mujoco_polyline(mujoco, scene, np.asarray((start, end)), (*_FRAME_COLORS[axis], 1.0), 0.008)
        self.viewer.sync()

    def _collision_warning(self, q: np.ndarray) -> bool:
        if self._frame_index % 10 == 0:
            margin = float(self.asset.metadata.get("collision", {}).get("margin", 0.0))
            self._warning = not self.kinematics.collision_report((q,), margin=margin).valid
        self._frame_index += 1
        return self._warning

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


def _thin(points: np.ndarray, maximum: int = 300) -> np.ndarray:
    if len(points) <= maximum:
        return np.asarray(points, dtype=np.float32)
    indices = np.linspace(0, len(points) - 1, maximum).astype(int)
    return np.asarray(points[indices], dtype=np.float32)


def _trajectory_positions(trajectory: Any) -> np.ndarray:
    if hasattr(trajectory, "position"):
        return np.asarray(trajectory.position, dtype=float)
    from .serial_trajectory import SerialArmTrajectory
    evaluator = SerialArmTrajectory(trajectory)
    count = max(2, min(500, int(np.ceil(trajectory.duration / 0.02)) + 1))
    return np.asarray([evaluator(t)[0] for t in np.linspace(0.0, trajectory.duration, count)])


def _mujoco_polyline(mujoco: Any, scene: Any, points: np.ndarray, rgba: tuple[float, ...], width: float) -> None:
    for start, end in zip(points[:-1], points[1:]):
        if scene.ngeom >= scene.maxgeom:
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3), np.eye(3).reshape(-1), np.asarray(rgba, dtype=np.float32))
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, width, start, end)
        scene.ngeom += 1


def _mujoco_point(mujoco: Any, scene: Any, point: np.ndarray, rgba: tuple[float, ...], radius: float) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.asarray((radius, radius, radius)),
        np.asarray(point), np.eye(3).reshape(-1), np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def _frame_segments(pose: np.ndarray, scale: float = 0.07) -> tuple[np.ndarray, np.ndarray]:
    """Return three world-frame XYZ axes for an XYZW Cartesian pose."""
    x, y, z, w = np.asarray(pose[3:], dtype=float)
    rotation = np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    origin = np.asarray(pose[:3], dtype=np.float32)
    starts = np.repeat(origin[None, :], 3, axis=0)
    ends = starts + scale * rotation.T
    return starts.astype(np.float32), ends.astype(np.float32)
