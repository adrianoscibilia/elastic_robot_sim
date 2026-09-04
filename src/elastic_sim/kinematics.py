"""Backend-neutral forward/inverse kinematics and self-collision checks.

The final simulator input remains a :class:`MaterializedTrajectory`; this
module only materializes and validates its link-side joint references.  Heavy
robotics imports are intentionally lazy so ordinary config inspection remains
cheap and ROS-independent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .assets import AssetSpec


@dataclass(frozen=True)
class KinematicGroup:
    name: str
    joints: tuple[str, ...]
    tip_link: str
    translation_only: bool = False


@dataclass(frozen=True)
class CollisionReport:
    valid: bool
    minimum_distance: float
    closest_pair: tuple[str, str] | None
    checked_configurations: int


def kinematic_groups(asset: AssetSpec, selected: Sequence[str] | None = None) -> tuple[KinematicGroup, ...]:
    """Resolve leaf kinematic groups, expanding aggregate groups such as dual_arm."""
    raw = asset.metadata.get("kinematic_groups", {})
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError(f"Asset {asset.name!r} has no kinematic_groups metadata")
    requested = list(selected or ())
    if not requested:
        requested = [name for name, value in raw.items() if isinstance(value, Mapping) and "tip_link" in value]
    expanded: list[str] = []
    for name in requested:
        value = raw.get(name)
        if not isinstance(value, Mapping):
            raise ValueError(f"Unknown kinematic group {name!r} for {asset.name}")
        children = value.get("groups")
        if children is not None:
            if not isinstance(children, list) or not all(isinstance(item, str) for item in children):
                raise ValueError(f"kinematic group {name!r}.groups must be a list of names")
            expanded.extend(children)
        else:
            expanded.append(name)
    result: list[KinematicGroup] = []
    for name in dict.fromkeys(expanded):
        value = raw[name]
        joints, tip = value.get("joints"), value.get("tip_link")
        if not isinstance(joints, list) or not joints or not all(isinstance(item, str) for item in joints):
            raise ValueError(f"kinematic group {name!r} requires a non-empty joints list")
        if not isinstance(tip, str) or not tip:
            raise ValueError(f"kinematic group {name!r} requires tip_link")
        if not set(joints).issubset(asset.joint_names):
            raise ValueError(f"kinematic group {name!r} contains joints outside active_joints")
        result.append(KinematicGroup(name, tuple(joints), tip, bool(value.get("translation_only", False))))
    return tuple(result)


class PortableKinematics:
    """Pinocchio/Pink kinematics with Coal validation for one repository asset."""

    def __init__(self, asset: AssetSpec) -> None:
        self.asset = asset
        self.groups = kinematic_groups(asset)
        self._direct = all(group.translation_only for group in self.groups)
        self._checked = 0
        if self._direct:
            self.model = self.data = self.collision_model = self.collision_data = None
            self._q_indices = tuple(range(len(asset.joint_names)))
            return
        try:
            import pinocchio as pin
        except ImportError as exc:  # pragma: no cover - default dependency
            raise ImportError("Cartesian planning requires `uv sync` (Pinocchio is missing)") from exc
        self.pin = pin
        self.model, self.collision_model, _ = pin.buildModelsFromUrdf(
            str(asset.urdf_path.resolve()), package_dirs=[str(asset.urdf_path.parent.resolve())]
        )
        self.data = self.model.createData()
        self._q_indices = tuple(int(self.model.idx_qs[self.model.getJointId(name)]) for name in asset.joint_names)
        missing_frames = [group.tip_link for group in self.groups if not self.model.existFrame(group.tip_link)]
        if missing_frames:
            raise ValueError(f"Asset {asset.name!r} has unknown tip links: {missing_frames}")
        self._prepare_collision_pairs()
        self.collision_data = pin.GeometryData(self.collision_model)

    def neutral(self) -> np.ndarray:
        if self._direct:
            return np.zeros(len(self.asset.joint_names))
        q = self.pin.neutral(self.model)
        return np.asarray([q[index] for index in self._q_indices], dtype=float)

    def _model_q(self, q: Sequence[float]) -> np.ndarray:
        values = np.asarray(q, dtype=float)
        if values.shape != (len(self.asset.joint_names),):
            raise ValueError(f"Expected {len(self.asset.joint_names)} joints, got {values.shape}")
        if self._direct:
            return values.copy()
        model_q = self.pin.neutral(self.model)
        for source, target in enumerate(self._q_indices):
            model_q[target] = values[source]
        return model_q

    def _asset_q(self, model_q: Sequence[float]) -> np.ndarray:
        return np.asarray([model_q[index] for index in self._q_indices], dtype=float)

    def forward(self, q: Sequence[float], groups: Sequence[KinematicGroup] | None = None) -> dict[str, np.ndarray]:
        """Return poses as ``[x,y,z,qx,qy,qz,qw]`` in the asset base frame."""
        selected = tuple(groups or self.groups)
        values = np.asarray(q, dtype=float)
        if self._direct:
            xyz = np.zeros(3)
            for index, name in enumerate(self.asset.joint_names):
                if name.endswith("_x"): xyz[0] = values[index]
                elif name.endswith("_y"): xyz[1] = values[index]
                elif name.endswith("_z"): xyz[2] = values[index]
            return {group.name: np.r_[xyz, [0.0, 0.0, 0.0, 1.0]] for group in selected}
        model_q = self._model_q(values)
        self.pin.forwardKinematics(self.model, self.data, model_q)
        self.pin.updateFramePlacements(self.model, self.data)
        result: dict[str, np.ndarray] = {}
        for group in selected:
            placement = self.data.oMf[self.model.getFrameId(group.tip_link)]
            quat = self.pin.Quaternion(placement.rotation).coeffs()
            result[group.name] = np.r_[np.asarray(placement.translation), np.asarray(quat)]
        return result

    def solve_pose_samples(
        self,
        targets: Mapping[str, np.ndarray],
        *,
        initial_q: Sequence[float],
        dt: float,
        position_tolerance: float = 1.0e-3,
        orientation_tolerance: float = 1.0e-2,
        max_iterations: int = 80,
        collision_margin: float = 0.0,
        collision_avoidance: bool = False,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Solve synchronized pose samples using Pink differential IK."""
        groups = kinematic_groups(self.asset, tuple(targets))
        lengths = {len(np.asarray(value)) for value in targets.values()}
        if len(lengths) != 1:
            raise ValueError("All Cartesian target arrays must have the same length")
        count = lengths.pop()
        if self._direct:
            output = np.zeros((count, len(self.asset.joint_names)))
            target = np.asarray(targets[groups[0].name], dtype=float)
            output[:, :3] = target[:, :3]
            return output, {"solver": "direct_cartesian", "max_position_error": 0.0, "max_orientation_error": 0.0}
        try:
            import pink
            from pink.barriers import SelfCollisionBarrier
            from pink.tasks import FrameTask, PostureTask
        except ImportError as exc:  # pragma: no cover - default dependency
            raise ImportError("Cartesian IK requires `pin-pink`; run `uv sync`") from exc
        q_model = self._model_q(initial_q)
        configuration = pink.Configuration(
            self.model, self.data, q_model,
            collision_model=self.collision_model, collision_data=self.collision_data,
        )
        tasks: list[Any] = []
        frame_tasks: dict[str, Any] = {}
        for group in groups:
            task = FrameTask(group.tip_link, position_cost=1.0, orientation_cost=0.0 if group.translation_only else 0.2, lm_damping=1.0e-6)
            tasks.append(task)
            frame_tasks[group.name] = task
        posture = PostureTask(cost=1.0e-4)
        posture.set_target(q_model)
        tasks.append(posture)
        barriers = []
        if collision_avoidance and self.collision_model.collisionPairs and collision_margin > 0.0:
            barriers.append(SelfCollisionBarrier(min(16, len(self.collision_model.collisionPairs)), d_min=collision_margin))
        solved: list[np.ndarray] = []
        max_pos_error = max_rot_error = 0.0
        for row in range(count):
            for group in groups:
                pose = np.asarray(targets[group.name][row], dtype=float)
                rotation = self.pin.Quaternion(pose[6], pose[3], pose[4], pose[5]).matrix()
                frame_tasks[group.name].set_target(self.pin.SE3(rotation, pose[:3]))
            for _ in range(max_iterations):
                velocity = pink.solve_ik(
                    configuration, tasks, dt, solver="proxqp", damping=1.0e-8,
                    barriers=barriers, safety_break=True,
                )
                configuration.integrate_inplace(velocity, dt)
                errors = []
                for group in groups:
                    error = frame_tasks[group.name].compute_error(configuration)
                    errors.append((float(np.linalg.norm(error[:3])), float(np.linalg.norm(error[3:]))))
                if all(pe <= position_tolerance and (group.translation_only or re <= orientation_tolerance)
                       for group, (pe, re) in zip(groups, errors)):
                    break
            else:
                raise ValueError(f"IK did not converge for Cartesian sample {row}; errors={errors}")
            max_pos_error = max(max_pos_error, *(item[0] for item in errors))
            max_rot_error = max(max_rot_error, *(item[1] for item in errors))
            solved.append(self._asset_q(configuration.q))
        return np.asarray(solved), {
            "solver": "pink/proxqp", "max_position_error": max_pos_error,
            "max_orientation_error": max_rot_error, "samples": count,
        }

    def collision_report(self, configurations: Iterable[Sequence[float]], margin: float = 0.0) -> CollisionReport:
        if self._direct or not self.asset.self_collisions or not self.collision_model.collisionPairs:
            values = tuple(configurations)
            return CollisionReport(True, float("inf"), None, len(values))
        minimum, closest, checked = float("inf"), None, 0
        cache: dict[bytes, tuple[float, tuple[str, str] | None]] = {}
        for q in configurations:
            checked += 1
            values = np.asarray(q, dtype=float)
            key = values.tobytes()
            if key not in cache:
                model_q = self._model_q(values)
                self.pin.forwardKinematics(self.model, self.data, model_q)
                self.pin.updateGeometryPlacements(self.model, self.data, self.collision_model, self.collision_data, model_q)
                self.pin.computeDistances(self.model, self.data, self.collision_model, self.collision_data, model_q)
                state_minimum, state_pair = float("inf"), None
                for pair, distance in zip(self.collision_model.collisionPairs, self.collision_data.distanceResults):
                    value = float(distance.min_distance)
                    if value < state_minimum:
                        first = self.collision_model.geometryObjects[pair.first]
                        second = self.collision_model.geometryObjects[pair.second]
                        state_pair = (
                            self.model.frames[first.parentFrame].name,
                            self.model.frames[second.parentFrame].name,
                        )
                        state_minimum = value
                cache[key] = (state_minimum, state_pair)
            state_minimum, state_pair = cache[key]
            if state_minimum < minimum:
                minimum, closest = state_minimum, state_pair
        return CollisionReport(minimum >= margin, minimum, closest, checked)

    def validate_path(self, q: np.ndarray, *, margin: float = 0.0, max_joint_step: float = 0.05) -> CollisionReport:
        """Validate the path at a bounded joint-space arc-length resolution."""
        values = np.asarray(q, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(self.asset.joint_names):
            raise ValueError("Path shape does not match asset active joints")
        if max_joint_step <= 0.0:
            raise ValueError("collision.max_joint_step must be positive")
        # Include every materialized sample.  When adjacent samples are too
        # far apart, recursively bisect the segment (a power-of-two number of
        # subdivisions) until every checked increment is within the bound.
        expanded: list[np.ndarray] = [values[0]]
        for first, second in zip(values[:-1], values[1:]):
            distance = float(np.max(np.abs(second - first)))
            levels = max(0, int(np.ceil(np.log2(distance / max_joint_step)))) if distance > 0.0 else 0
            subdivisions = 2**levels
            for step in range(1, subdivisions + 1):
                expanded.append(first + (second - first) * (step / subdivisions))
        return self.collision_report(expanded, margin)

    def _prepare_collision_pairs(self) -> None:
        self.collision_model.addAllCollisionPairs()
        allowed = {
            frozenset(pair) for pair in self.asset.metadata.get("collision", {}).get("allowed_pairs", [])
            if isinstance(pair, (list, tuple)) and len(pair) == 2
        }
        retained = []
        for pair in list(self.collision_model.collisionPairs):
            first = self.collision_model.geometryObjects[pair.first]
            second = self.collision_model.geometryObjects[pair.second]
            j1, j2 = int(first.parentJoint), int(second.parentJoint)
            links = frozenset((self.model.frames[first.parentFrame].name, self.model.frames[second.parentFrame].name))
            adjacent = j1 == j2 or int(self.model.parents[j1]) == j2 or int(self.model.parents[j2]) == j1
            if not adjacent and links not in allowed:
                retained.append(self.pin.CollisionPair(int(pair.first), int(pair.second)))
        self.collision_model.removeAllCollisionPairs()
        for pair in retained:
            self.collision_model.addCollisionPair(pair)


def cartesian_samples(kinematics: PortableKinematics, q: np.ndarray) -> dict[str, np.ndarray]:
    """Forward-kinematics convenience for a complete sampled path."""
    rows = [kinematics.forward(row) for row in np.asarray(q)]
    return {group.name: np.asarray([row[group.name] for row in rows]) for group in kinematics.groups}


def enrich_rollout_frame(frame: Any, trajectory: Any, asset: AssetSpec) -> Any:
    """Append Cartesian tracking and per-sample collision-clearance columns."""
    kin = PortableKinematics(asset)
    reference = cartesian_samples(kin, trajectory.position)
    q_columns = [f"q__{name}" for name in trajectory.joint_names]
    if not all(column in frame for column in q_columns):
        return frame
    actual_q = frame[q_columns].to_numpy(float)
    actual = cartesian_samples(kin, actual_q)
    labels = ("x", "y", "z", "qx", "qy", "qz", "qw")
    configured_ref = trajectory.metadata.get("cartesian_reference", {})
    for group in kin.groups:
        ref = np.asarray(configured_ref.get(group.name, reference[group.name]), dtype=float)
        if len(ref) != len(frame):
            ref = reference[group.name]
        measured = actual[group.name]
        for index, label in enumerate(labels):
            frame[f"ee_ref__{group.name}__{label}"] = ref[:, index]
            frame[f"ee__{group.name}__{label}"] = measured[:, index]
        frame[f"ee_position_error__{group.name}"] = np.linalg.norm(ref[:, :3] - measured[:, :3], axis=1)
        quaternion_dot = np.clip(np.abs(np.sum(ref[:, 3:] * measured[:, 3:], axis=1)), 0.0, 1.0)
        frame[f"ee_orientation_error__{group.name}"] = 2.0 * np.arccos(quaternion_dot)
    if kin._direct or not asset.self_collisions:
        frame["self_collision_clearance"] = np.inf
    else:
        # Full mesh distance queries are intentionally sampled at a bounded
        # rate and interpolated for plotting.  The commanded reference has
        # already undergone the stricter arc-length validation above.
        count = min(len(actual_q), 100)
        sample_indices = np.unique(np.linspace(0, len(actual_q) - 1, count).astype(int))
        clearances = np.asarray([
            kin.collision_report((actual_q[index],)).minimum_distance for index in sample_indices
        ])
        frame["self_collision_clearance"] = np.interp(np.arange(len(actual_q)), sample_indices, clearances)
    return frame
