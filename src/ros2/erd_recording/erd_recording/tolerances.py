"""Per-joint JTC path/goal tolerances, generated from the lab config at
launch time (RR_04 A-5).

``erd_iiwa/config/controllers.yaml`` and ``erd_ur10/config/controllers.yaml``
carry no per-joint ``constraints.<joint>`` entries (iiwa) or upstream's own
defaults (UR10, ``trajectory: 0.2, goal: 0.1``): neither aborts the JTC on a
real tracking deviation. The abort tracking radius is a per-lab safety
parameter (``limits.abort.tracking_rad``), not a robot default, so it is
never hand-edited into the checked-in YAML; instead the launch files call
:func:`write_tolerance_overrides` to generate a small parameter-override file
from whichever lab config the run uses, and load it as an extra
``parameters=[...]`` entry after the base controllers YAML (ROS 2 layers
later parameter files over earlier ones for the same node).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import yaml

from .config import LabConfig

#: RR_04 A-5: fixed goal tolerance and goal time on both robots; only the
#: per-joint trajectory (path) tolerance comes from the lab config.
GOAL_TOLERANCE_RAD = 0.01
GOAL_TIME_S = 0.5


def write_tolerance_overrides(
    config: LabConfig, controller_name: str, driver_joint_names: Sequence[str], output_path: str | Path,
) -> Path:
    """Write a ``<controller_name>: ros__parameters: constraints: ...`` YAML
    with ``trajectory: limits.abort.tracking_rad`` and ``goal:
    GOAL_TOLERANCE_RAD`` for every joint in ``driver_joint_names`` (the
    driver's own joint names, e.g. ``joint_a1``/``shoulder_pan_joint`` --
    what the controller's parameter server expects, not the sim-asset
    names)."""
    tracking_rad = config.limits.abort.tracking_rad
    constraints: dict[str, object] = {"goal_time": GOAL_TIME_S}
    for joint_name in driver_joint_names:
        constraints[joint_name] = {"trajectory": tracking_rad, "goal": GOAL_TOLERANCE_RAD}
    data = {controller_name: {"ros__parameters": {"constraints": constraints}}}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return output_path
