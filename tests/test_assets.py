"""Pure-Python coverage for the generic asset contract (no Newton required)."""

import os
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(_REPO / "src"))

from elastic_sim.assets import AssetRegistry, AssetSpec, discover_urdf_joints, load_asset_spec
from elastic_sim.generic_newton_runner import ElasticTransmissionParams, GenericNewtonElasticBuilder


_URDF = """<?xml version='1.0'?>
<robot name='test_arm'>
  <link name='base'/><link name='link_1'/><link name='link_2'/><link name='tool'/>
  <joint name='joint_1' type='revolute'>
    <parent link='base'/><child link='link_1'/><axis xyz='0 0 1'/>
    <limit lower='-1.0' upper='1.0' effort='2.0' velocity='3.0'/>
  </joint>
  <joint name='joint_2' type='prismatic'>
    <parent link='link_1'/><child link='link_2'/><axis xyz='1 0 0'/>
    <limit lower='0.0' upper='0.5' effort='4.0' velocity='5.0'/>
  </joint>
  <joint name='mimic_joint' type='revolute'>
    <parent link='link_2'/><child link='tool'/><mimic joint='joint_2'/>
  </joint>
</robot>"""


def _write_asset(tmp_path: Path, active: str = "[]") -> Path:
    (tmp_path / "robot.urdf").write_text(_URDF, encoding="utf-8")
    spec = tmp_path / "arm.yaml"
    spec.write_text(
        "asset:\n  name: test_arm\n  urdf: robot.urdf\n  active_joints: " + active + "\n",
        encoding="utf-8",
    )
    return spec


def test_discovery_selects_only_non_mimic_one_dof_joints(tmp_path: Path):
    spec = load_asset_spec(_write_asset(tmp_path))
    joints = discover_urdf_joints(spec.urdf_path)
    assert [joint.name for joint in joints] == ["joint_1", "joint_2", "mimic_joint"]
    assert spec.joint_names == ("joint_1", "joint_2")
    assert joints[0].lower == -1.0
    assert joints[1].axis == (1.0, 0.0, 0.0)


def test_explicit_active_order_is_preserved(tmp_path: Path):
    spec = load_asset_spec(_write_asset(tmp_path, "[joint_2, joint_1]"))
    assert spec.joint_names == ("joint_2", "joint_1")


def test_invalid_active_joint_fails_before_simulator_import(tmp_path: Path):
    spec = load_asset_spec(_write_asset(tmp_path, "[missing]"))
    with pytest.raises(ValueError, match="unsupported joints"):
        spec.resolve_active_joints()


def test_registry_and_transmission_validation_are_generic(tmp_path: Path):
    _write_asset(tmp_path)
    registry = AssetRegistry((tmp_path,))
    spec = registry.load("test_arm")
    assert registry.available() == {"test_arm": tmp_path / "arm.yaml"}
    with pytest.raises(ValueError, match="missing=.*joint_2"):
        GenericNewtonElasticBuilder(spec, {"joint_1": ElasticTransmissionParams(1.0, 0.1)})


def test_transmission_parameters_reject_non_physical_values():
    with pytest.raises(ValueError, match="stiffness"):
        ElasticTransmissionParams(-1.0, 0.0)
