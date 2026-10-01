# Copyright 2026 elastic_robot_sim contributors
# SPDX-License-Identifier: Apache-2.0
"""RR_06 P3-1 item 3 / RR_04 A-7: mock, emulator and real must declare the
identical hardware-interface name set, so a recording stack verified at L1
transfers unchanged to L2/L3 -- `ros2 control list_hardware_interfaces`
reports exactly what's declared in the exported `<ros2_control>` block, so
comparing that block's own command/state interface names across hardware
modes is the same invariant a live launch-time comparison would check,
without needing a running controller_manager (mock) or a live FRI partner
(emulator/real) for every `colcon test` run.
"""

from xml.etree import ElementTree as ET

from erd_iiwa.description import descriptions


def _interface_names(urdf_xml: str) -> set[str]:
    control = ET.fromstring(urdf_xml).find("ros2_control")
    names: set[str] = set()
    for joint in control.findall("joint"):
        joint_name = joint.get("name")
        for interface in list(joint.findall("command_interface")) + list(joint.findall("state_interface")):
            names.add(f"{joint_name}/{interface.get('name')}")
    for gpio in control.findall("gpio"):
        gpio_name = gpio.get("name")
        for interface in list(gpio.findall("state_interface")) + list(gpio.findall("command_interface")):
            names.add(f"{gpio_name}/{interface.get('name')}")
    for sensor in control.findall("sensor"):
        sensor_name = sensor.get("name")
        for interface in sensor.findall("state_interface"):
            names.add(f"{sensor_name}/{interface.get('name')}")
    return names


def test_mock_and_emulator_declare_identical_interface_names():
    mock_xml, _ = descriptions(hardware="mock")
    emulator_xml, _ = descriptions(hardware="emulator")
    assert _interface_names(mock_xml) == _interface_names(emulator_xml)


def test_mock_and_real_declare_identical_interface_names():
    mock_xml, _ = descriptions(hardware="mock")
    real_xml, _ = descriptions(hardware="real")
    assert _interface_names(mock_xml) == _interface_names(real_xml)


def test_no_external_torque_sensor_block_in_any_mode():
    # RR_04 A-7's own finding: the ICube upstream's `external_torque_sensor`
    # <sensor> must be removed in every mode, not only emulator/real -- a
    # regression here is exactly what would make the two tests above fail
    # for the wrong reason (an extra mock-only interface), so it's asserted
    # directly too.
    for hardware in ("mock", "emulator", "real"):
        urdf_xml, _ = descriptions(hardware=hardware)
        control = ET.fromstring(urdf_xml).find("ros2_control")
        assert control.findall("sensor") == [], f"hardware={hardware} still declares a <sensor> block"
