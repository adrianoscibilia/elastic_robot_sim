"""Known scene-coordinate frames per robot (RR_04 A-6).

RR_01 S4 declares every scene coordinate is in some named ``scene.frame``,
but until this pass nothing actually converted it: ``WorldKinematics`` added
every object's raw ``xyz``/``rpy`` straight onto the sim asset's universe
joint, silently assuming the declared frame **was** that asset's own root.
That's why pass 1 "had to push the fake scene clear of the built-in table"
instead of just naming its frame -- there was no conversion to rely on.

Each robot has a small, fixed table of known frames -> the **bare** sim
asset's own root frame (the model the excitation is planned against, RR_01
v3.1 S4/S5). An unknown frame is a config error, per RR_01 S4.
"""

from __future__ import annotations

import numpy as np

#: The bare sim assets' own root frame name (identity transform) and every
#: other frame this stack knows how to convert from, as
#: ``{robot: {frame_name: (translation_xyz, rpy)}}`` -- a point ``p`` in
#: ``frame_name`` is at ``R(rpy) @ p + translation`` in the asset's root.
KNOWN_FRAMES: dict[str, dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]]] = {
    "kuka_lbr_iiwa_14_r820": {
        # `iiwa_base` is the ICube driver description's own virtual root link
        # (`iiwa_description/urdf/iiwa.urdf.xacro`'s `${prefix}iiwa_base`),
        # connected to `link_0` -- the bare asset's actual root -- by a fixed
        # joint with `xyz="0 0 0" rpy="0 0 0"` (verified against that xacro,
        # RR_04 A-6): identity, not a coincidence of naming.
        "iiwa_base": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    },
    "ur10_cb3": {
        # `base_link` is the bare `ur10` asset's own root: identity.
        "base_link": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        # UR's `base` (what the pendant/PolyScope shows) is `base_link`
        # rotated pi about z (RR_01 S3.4 "Frames"; verified against
        # `ur_description`'s own `base` <-> `base_link` static transform).
        "base": ((0.0, 0.0, 0.0), (0.0, 0.0, np.pi)),
    },
}


class FrameError(ValueError):
    """An unknown scene frame was named (RR_01 S4: "An unknown frame is a
    config error")."""


def _rpy_to_matrix(rpy: tuple[float, float, float]) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return rz @ ry @ rx


def _matrix_to_rpy(rotation: np.ndarray) -> tuple[float, float, float]:
    sy = float(np.clip(-rotation[2, 0], -1.0, 1.0))
    pitch = np.arcsin(sy)
    if abs(sy) < 0.999999:
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = np.arctan2(-rotation[1, 2], rotation[1, 1])
        yaw = 0.0
    return float(roll), float(pitch), float(yaw)


def frame_transform_to_root(robot: str, frame: str) -> tuple[np.ndarray, np.ndarray]:
    """``(rotation, translation)`` taking a point/orientation declared in
    ``frame`` into ``robot``'s bare sim asset's own root frame."""
    frames = KNOWN_FRAMES.get(robot)
    if frames is None:
        raise FrameError(f"no known scene frames for robot {robot!r}")
    entry = frames.get(frame)
    if entry is None:
        raise FrameError(
            f"unknown scene frame {frame!r} for robot {robot!r}; known frames: {sorted(frames)}"
        )
    translation, rpy = entry
    return _rpy_to_matrix(rpy), np.asarray(translation, dtype=float)


def transform_xyz(robot: str, frame: str, xyz: tuple[float, float, float]) -> tuple[float, float, float]:
    rotation, translation = frame_transform_to_root(robot, frame)
    result = rotation @ np.asarray(xyz, dtype=float) + translation
    return (float(result[0]), float(result[1]), float(result[2]))


def transform_rpy(
    robot: str, frame: str, rpy: tuple[float, float, float],
) -> tuple[float, float, float]:
    rotation, _ = frame_transform_to_root(robot, frame)
    combined = rotation @ _rpy_to_matrix(rpy)
    return _matrix_to_rpy(combined)
