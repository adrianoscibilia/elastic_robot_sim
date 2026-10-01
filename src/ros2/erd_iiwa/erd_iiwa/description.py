# Copyright 2026 elastic_robot_sim contributors
# SPDX-License-Identifier: Apache-2.0
"""Derive the iiwa description in memory from the installed `iiwa_description`
upstream (RR_01 S3.3, S1.3 "what to take from iiwa_experiments"). No upstream
file is patched or shadowed; only the hardware plugin/state declarations
change, following ``iiwa_experiments/description.py``'s pattern.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import xacro
from ament_index_python.packages import get_package_share_directory

#: erd_iiwa's own GPIO interfaces + the extra joint state interfaces
#: FriPositionSystem exports, that the ICube description doesn't declare
#: (RR_01 S3.3).
_EXTRA_JOINT_STATES = ("commanded_effort", "external_effort", "commanded_position", "ipo_position")
_FRI_GPIO = ("time_sec", "time_nsec", "sample_time", "session_state", "command_mode",
            "connection_quality", "tracking_performance", "safety_state", "operation_mode",
            "drive_state", "cycle")


def descriptions(hardware: str = "mock", robot_ip: str = "192.170.10.2", fri_port: str = "30200", initial_positions=None):
    """Return ``(urdf_xml, srdf_xml)`` with erd_iiwa's ros2_control overlay.

    ``hardware``: ``mock`` (mock_components/GenericSystem) | ``emulator`` |
    ``real`` (both erd_iiwa/FriPositionSystem; the port/IP select which robot
    -- the on-loopback emulator or the real KONI network, RR_01 S3.5).
    """
    if hardware not in ("mock", "emulator", "real"):
        raise ValueError(f"hardware must be mock|emulator|real, got {hardware!r}")
    share = Path(get_package_share_directory("iiwa_description"))
    erd_share = Path(get_package_share_directory("erd_iiwa"))
    mappings = {
        "use_sim": "false",
        "use_fake_hardware": str(hardware == "mock").lower(),
        "prefix": "",
        "namespace": "/",
        "robot_ip": robot_ip,
        "robot_port": str(fri_port),
        "command_interface": "position",
        # erd_iiwa's own zero-offset file, not iiwa_description's (the
        # upstream default ships x: 1.0 as an example placement, which broke
        # the FK-identity gate, RR_06 P3-1 item 1 / RR_04 B-2 -- see the
        # comment in that file).
        "base_frame_file": str(erd_share / "config/base_frame.yaml"),
        "initial_positions_file": str(share / "config/initial_positions.yaml"),
    }
    root = ET.fromstring(xacro.process_file(str(share / "config/iiwa.config.xacro"), mappings=mappings).toxml())
    control = root.find("ros2_control")
    plugin = control.find("hardware/plugin")
    expected = "mock_components/GenericSystem" if hardware == "mock" else "iiwa_hardware/IiwaFRIHardwareInterface"
    if plugin.text.strip() != expected:
        raise RuntimeError("Unexpected upstream hardware description; review the iiwa_ros2 pin")
    if hardware != "mock":
        plugin.text = "erd_iiwa/FriPositionSystem"
        for param_name, value in (("robot_ip", robot_ip), ("fri_port", str(fri_port))):
            existing = control.find(f"hardware/param[@name='{param_name}']")
            if existing is None:
                ET.SubElement(control.find("hardware"), "param", name=param_name).text = value
            else:
                existing.text = value
    # The ICube upstream description declares a separate
    # `external_torque_sensor` <sensor> block, populated by
    # `iiwa_hardware/IiwaFRIHardwareInterface` (and `iiwa_experiments`'s
    # plugin, which mirrors its interface names). erd_iiwa/FriPositionSystem
    # exports `external_effort` per joint instead (RR_01 S3.3's own list);
    # remove the sensor block in every mode, mock included, or
    # `ros2 control list_hardware_interfaces` differs between mock and
    # emulator/real and mock runs gain an extra column real runs won't have
    # (RR_04 A-7; the emulator/real removal alone was T1.9's fix).
    for sensor in list(control.findall("sensor")):
        control.remove(sensor)

    for joint in control.findall("joint"):
        if hardware == "mock" and initial_positions is not None:
            initial = joint.find("state_interface[@name='position']/param[@name='initial_value']")
            if initial is None:
                initial = ET.SubElement(joint.find("state_interface[@name='position']"), "param", name="initial_value")
            initial.text = str(initial_positions[joint.get("name")])
        for name in _EXTRA_JOINT_STATES:
            if joint.find(f"state_interface[@name='{name}']") is not None:
                raise RuntimeError(f"Upstream already exports {name!r}; review integration")
            state = ET.SubElement(joint, "state_interface", name=name)
            if hardware == "mock":
                ET.SubElement(state, "param", name="initial_value").text = "0.0"

    gpio = ET.SubElement(control, "gpio", name="fri")
    for index, name in enumerate(_FRI_GPIO):
        state = ET.SubElement(gpio, "state_interface", name=name)
        if hardware == "mock" and name not in ("time_sec", "time_nsec"):
            # Finite and distinct (not all zero), so a tau==ft-style bug
            # can't hide behind indistinguishable defaults (RR_01 S3.3).
            # time_sec/time_nsec are left uninitialized (mock_components'
            # own NaN default): RR_01 S3.3 wants the converter to see NaN
            # there and fall back to header stamps, and GenericSystem's
            # initial_value parser rejects the literal string "nan".
            ET.SubElement(state, "param", name="initial_value").text = str({"sample_time": 0.001, "command_mode": 1.0}.get(name, float(index + 1)))

    semantic = ET.fromstring(
        xacro.process_file(str(share / "srdf/iiwa.srdf.xacro"), mappings={"name": "iiwa", "prefix": ""}).toxml()
    )
    for virtual in semantic.findall("virtual_joint"):
        if virtual.get("child_link") == "iiwa_base" and virtual.get("parent_frame") == "world":
            semantic.remove(virtual)
    return ET.tostring(root, encoding="unicode"), ET.tostring(semantic, encoding="unicode")


#: erd_iiwa/FriPositionSystem's own joint order (ICube driver names).
JOINTS = tuple(f"joint_a{i}" for i in range(1, 8))


def asset_snapshot(directory: str):
    """A driver-side :class:`elastic_sim.assets.AssetSpec` for the FK-identity
    gate (RR_01 S5 `plan`): the ICube xacro's own kinematics, not the
    simulation's bare asset."""
    from elastic_sim.assets import AssetSpec

    xml, srdf = descriptions()
    root = ET.fromstring(xml)
    for child in list(root):
        if child.tag not in ("joint", "link", "material"):
            root.remove(child)
    for mesh in root.findall(".//mesh"):
        mesh.set("filename", mesh.get("filename").removeprefix("file://"))
    path = Path(directory) / "iiwa_driver.urdf"
    path.write_text(ET.tostring(root, encoding="unicode"))
    allowed = [[e.get("link1"), e.get("link2")] for e in ET.fromstring(srdf).findall("disable_collisions")]
    return AssetSpec(
        name="erd_iiwa_driver", urdf_path=path, active_joints=JOINTS, self_collisions=True,
        metadata={
            "kinematic_groups": {"iiwa_arm": {"joints": list(JOINTS), "tip_link": "tool0"}},
            "collision": {"margin": 0.005, "max_joint_step": 0.01, "allowed_pairs": allowed},
        },
    )
