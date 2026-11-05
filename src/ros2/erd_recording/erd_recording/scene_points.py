"""Scene boundary walls from measured points (owner request 2026-11-05).

The lab files used to carry each virtual wall as a hand-computed ``box``
(centre, size, yaw). This module derives that box from what the owner
measures in the cell: two points ``p1``, ``p2`` on the boundary line, and
one point ``inside`` on the robot's (allowed) side.

The wall is vertical. Its face lies on the P1-P2 line, or ``offset`` metres
behind it (away from ``inside``); its local +y axis points to the allowed
side; its length is centred on the P1-P2 midpoint. The result is the same
``box`` the loader already handles, so the scene clearance inflates it
exactly as it inflates a hand-written box.

Standard library and numpy only: ``config.py`` imports it in the ROS
process, where pinocchio must not load.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

UNITS = {"m": 1.0, "mm": 1.0e-3}
#: An ``inside`` point closer than this to the P1-P2 line can't tell the sides apart.
MIN_INSIDE_DISTANCE_M = 0.01
#: P1 and P2 closer than this give no usable direction.
MIN_POINT_SEPARATION_M = 0.05
WALL_KEYS = {"id", "type", "p1", "p2", "inside", "unit", "offset", "thickness", "length", "z"}


class ScenePointsError(ValueError):
    """A wall_from_points entry that can't define a wall."""


def _xy(value: Any, name: str, scale: float) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) not in (2, 3):
        raise ScenePointsError(f"{name} must be [x, y] or [x, y, z]")
    return float(value[0]) * scale, float(value[1]) * scale


def wall_box_from_points(
    p1: Sequence[float], p2: Sequence[float], inside: Sequence[float], *,
    unit: str = "m", offset: float = 0.0, thickness: float = 0.2, length: float = 6.0,
    z: Sequence[float] = (-1.0, 3.0),
) -> dict[str, Any]:
    """The ``box`` (``size``, ``xyz``, ``rpy``) of a vertical wall through P1-P2.

    ``p1``, ``p2``, ``inside`` are in ``unit`` (``m`` or ``mm``; a third
    coordinate is ignored). ``offset``, ``thickness``, ``length`` and ``z``
    are always metres. Also returns the derived quantities for the record:
    ``normal_inside`` (unit vector to the allowed side), ``yaw``, the
    midpoint, and the signed distance of the scene origin from the line
    (positive = allowed side).
    """
    if unit not in UNITS:
        raise ScenePointsError(f"unit must be one of {sorted(UNITS)}, got {unit!r}")
    scale = UNITS[unit]
    x1, y1 = _xy(p1, "p1", scale)
    x2, y2 = _xy(p2, "p2", scale)
    xi, yi = _xy(inside, "inside", scale)
    dx, dy = x2 - x1, y2 - y1
    span = math.hypot(dx, dy)
    if span < MIN_POINT_SEPARATION_M:
        raise ScenePointsError(f"p1 and p2 are {span:.4f} m apart; need >= {MIN_POINT_SEPARATION_M} m")
    ux, uy = dx / span, dy / span                     # along the line
    nx, ny = -uy, ux                                  # one of the two normals
    mx, my = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    side = (xi - mx) * nx + (yi - my) * ny
    if abs(side) < MIN_INSIDE_DISTANCE_M:
        raise ScenePointsError(f"inside is {abs(side):.4f} m from the P1-P2 line; need >= {MIN_INSIDE_DISTANCE_M} m")
    if side < 0.0:
        nx, ny = -nx, -ny                             # normal now points to the allowed side
    offset, thickness, length = float(offset), float(thickness), float(length)
    if offset < 0.0:
        raise ScenePointsError("offset must be >= 0 (the wall face sits on the line or behind it)")
    if thickness <= 0.0 or length <= 0.0:
        raise ScenePointsError("thickness and length must be > 0")
    z_low, z_high = (float(v) for v in z)
    if not z_low < z_high:
        raise ScenePointsError("z must be [low, high] with low < high")
    depth = offset + 0.5 * thickness                  # box centre behind the line
    cx, cy = mx - nx * depth, my - ny * depth
    # local +y = (-sin yaw, cos yaw) must equal the allowed-side normal
    yaw = math.atan2(-nx, ny)
    origin_side = (0.0 - mx) * nx + (0.0 - my) * ny
    return {
        "size": (length, thickness, z_high - z_low),
        "xyz": (cx, cy, 0.5 * (z_low + z_high)),
        "rpy": (0.0, 0.0, yaw),
        "derived": {
            "normal_inside": (nx, ny), "yaw": yaw, "midpoint": (mx, my),
            "line_length_m": span, "origin_distance_m": origin_side,
        },
    }


def wall_box_from_entry(raw: Mapping[str, Any]) -> dict[str, Any]:
    """:func:`wall_box_from_points` from a lab-file ``scene.objects`` entry."""
    unknown = sorted(set(raw) - WALL_KEYS)
    if unknown:
        raise ScenePointsError(f"unknown key(s) {unknown}; allowed: {sorted(WALL_KEYS)}")
    for key in ("p1", "p2", "inside"):
        if key not in raw:
            raise ScenePointsError(f"missing {key!r}")
    return wall_box_from_points(
        raw["p1"], raw["p2"], raw["inside"], unit=str(raw.get("unit", "m")),
        offset=raw.get("offset", 0.0), thickness=raw.get("thickness", 0.2),
        length=raw.get("length", 6.0), z=raw.get("z", (-1.0, 3.0)),
    )
