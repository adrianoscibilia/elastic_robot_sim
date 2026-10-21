"""UR10 driver facts read from ``/robot_description`` (UR_01 U-2).

The ``<ros2_control>`` block of the description the driver is running holds
the servo parameters the contract records (``servoj_gain``,
``servoj_lookahead_time``) and the kinematics hash of the calibration it was
launched with. ``preflight`` parses it, compares it with the lab file and
writes the facts into ``preflight.json`` and the manifest, so a driver that
isn't what the lab file says is visible in every run. No ROS imports.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping
from xml.etree import ElementTree as ET

import yaml

#: ros2_control hardware ``<param>`` names read from the description.
DRIVER_PARAMS = ("servoj_gain", "servoj_lookahead_time", "robot_receive_timeout", "use_currents_as_efforts",
                 "kinematics/hash", "non_blocking_read", "headless_mode", "robot_ip", "reverse_ip")


def parse_driver_facts(urdf_text: str) -> dict[str, Any]:
    """``{param: value}`` of the first ``<ros2_control>`` block that has a
    ``<hardware>`` element (the description the UR driver publishes carries
    exactly one); missing parameters are ``None``. Numeric parameters come
    back as floats."""
    root = ET.fromstring(urdf_text)
    params: dict[str, str] = {}
    plugin = None
    for control in root.iter("ros2_control"):
        hardware = control.find("hardware")
        if hardware is None:
            continue
        plugin = (hardware.findtext("plugin") or "").strip()
        params = {p.get("name"): (p.text or "").strip() for p in hardware.findall("param")}
        break
    facts: dict[str, Any] = {"plugin": plugin}
    for name in DRIVER_PARAMS:
        value = params.get(name)
        if value is None or name in ("kinematics/hash", "robot_ip", "reverse_ip"):
            facts[name] = value
            continue
        try:
            facts[name] = float(value)
        except ValueError:
            facts[name] = value
    return facts


def read_kinematics_hash(calibration_file: str | Path | None) -> str | None:
    """``kinematics: hash:`` of a ``ur_calibration`` YAML, ``None`` without a file."""
    if not calibration_file:
        return None
    with open(calibration_file, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    kinematics = data.get("kinematics")
    value = kinematics.get("hash") if isinstance(kinematics, Mapping) else None
    return None if value is None else str(value)


def check_driver_facts(facts: Mapping[str, Any], *, servoj_gain: float, servoj_lookahead_time: float,
                       hardware: str, kinematics_hash: str | None = None) -> dict[str, Any]:
    """Compare the running description with the lab file. Gated on
    ``hardware: real`` (the contract records the lab file's servo values, so
    they must be the driver's, UR_00 D-5); L1/L2 report what would fail.
    ``kinematics_hash`` is the hash inside the lab file's calibration YAML
    (UR_07 F-3): the driver must be running with that calibration."""
    problems = []
    if kinematics_hash is not None and facts.get("kinematics/hash") != kinematics_hash:
        problems.append(f"kinematics/hash: driver description has {facts.get('kinematics/hash')!r}, "
                        f"the lab file's calibration has {kinematics_hash!r}")
    for name, expected in (("servoj_gain", servoj_gain), ("servoj_lookahead_time", servoj_lookahead_time)):
        actual = facts.get(name)
        if not isinstance(actual, float) or not math.isclose(actual, float(expected), rel_tol=1e-9, abs_tol=1e-12):
            problems.append(f"{name}: driver description has {actual!r}, lab file has {expected!r}")
    ok = not problems
    gated = hardware == "real"
    return {"ok": ok or not gated, "would_pass": ok, "gated": gated, "problems": problems, "facts": dict(facts)}
