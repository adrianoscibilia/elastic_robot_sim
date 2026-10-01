"""In-memory export of the UR10 driver description, for the FK-identity gate
(RR_01 S5 `plan`, RR_04 B-2). No upstream file is patched; this only reads
``ur_description``'s own xacro with the arguments ``ur_control.launch.py``
itself uses.
"""

from __future__ import annotations

from pathlib import Path

import xacro
from ament_index_python.packages import get_package_share_directory

#: erd_ur10's own driver joint order (ur_robot_driver's `ur_type: ur10`).
JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
          "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")

#: The frame the FK-identity gate compares (RR_01 S5 "flange relative to each
#: model's own base"); both `ur_description` and the bare `ur10` asset name it
#: `tool0`.
TIP_LINK = "tool0"


def export_driver_urdf(output_path: str, *, kinematics_params_file: str | None = None) -> str:
    """Write the ``ur_description`` URDF for ``ur_type: ur10`` to
    ``output_path``. Nominal (default calibration) unless
    ``kinematics_params_file`` is given (RR_01 S5: "UR10: nominal and, when
    configured, calibrated")."""
    share = Path(get_package_share_directory("ur_description"))
    mappings = {"ur_type": "ur10", "name": "ur"}
    if kinematics_params_file:
        mappings["kinematics_params"] = kinematics_params_file
    xml = xacro.process_file(str(share / "urdf" / "ur.urdf.xacro"), mappings=mappings).toxml()
    Path(output_path).write_text(xml, encoding="utf-8")
    return output_path
