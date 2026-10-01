"""T1.2: config schema, loader, scene expansion, refusal rules."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from erd_recording.config import (
    ConfigError,
    REAL_ROS_DOMAIN_ID,
    SIMULATED_ROS_DOMAIN_ID,
    SceneBox,
    SceneCylinder,
    SceneSphere,
    load_lab_config,
)

FIXTURE = Path(__file__).resolve().parents[1] / "config" / "lab" / "iiwa_mock.yaml"


def _raw() -> dict:
    return yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))


def _write(tmp_path: Path, raw: dict) -> Path:
    path = tmp_path / "lab.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def test_loads_fixture():
    config = load_lab_config(FIXTURE)
    assert config.robot == "kuka_lbr_iiwa_14_r820"
    assert config.hardware == "mock"
    assert config.joint_order == (
        "iiwa_A1", "iiwa_A2", "iiwa_A3", "iiwa_A4", "iiwa_A5", "iiwa_A6", "iiwa_A7",
    )
    assert config.rate_hz == 1000.0


def test_bad_schema_refused(tmp_path):
    raw = _raw()
    raw["schema"] = "erd.lab/2"
    with pytest.raises(ConfigError, match="schema"):
        load_lab_config(_write(tmp_path, raw))


def test_bad_hardware_choice_refused(tmp_path):
    raw = _raw()
    raw["hardware"] = "ursim"  # valid for ur10_cb3, not for the iiwa
    with pytest.raises(ConfigError, match="hardware"):
        load_lab_config(_write(tmp_path, raw))


def test_domain_separation_real_refused_on_sim_domain(tmp_path):
    raw = _raw()
    raw["hardware"] = "real"
    raw["connection"]["ros_domain_id"] = SIMULATED_ROS_DOMAIN_ID
    with pytest.raises(ConfigError, match="ROS_DOMAIN_ID"):
        load_lab_config(_write(tmp_path, raw))


def test_domain_separation_mock_refused_off_sim_domain(tmp_path):
    raw = _raw()
    raw["connection"]["ros_domain_id"] = REAL_ROS_DOMAIN_ID
    with pytest.raises(ConfigError, match="ROS_DOMAIN_ID"):
        load_lab_config(_write(tmp_path, raw))


def test_real_hardware_refused_unreviewed_limits(tmp_path):
    raw = _raw()
    raw["hardware"] = "real"
    raw["connection"]["ros_domain_id"] = REAL_ROS_DOMAIN_ID
    raw["limits"]["reviewed"] = False
    with pytest.raises(ConfigError, match="not reviewed"):
        load_lab_config(_write(tmp_path, raw))


def test_real_hardware_refused_unreviewed_scene(tmp_path):
    raw = _raw()
    raw["hardware"] = "real"
    raw["connection"]["ros_domain_id"] = REAL_ROS_DOMAIN_ID
    raw["scene"]["reviewed"] = False
    with pytest.raises(ConfigError, match="not reviewed"):
        load_lab_config(_write(tmp_path, raw))


def test_real_hardware_refused_on_null_measured_value(tmp_path):
    raw = _raw()
    raw["hardware"] = "real"
    raw["connection"]["ros_domain_id"] = REAL_ROS_DOMAIN_ID
    raw["limits"]["position"]["iiwa_A1"] = None
    with pytest.raises(ConfigError, match="null value"):
        load_lab_config(_write(tmp_path, raw))


def test_real_hardware_accepts_reviewed_and_filled(tmp_path):
    raw = _raw()
    raw["hardware"] = "real"
    raw["connection"]["ros_domain_id"] = REAL_ROS_DOMAIN_ID
    config = load_lab_config(_write(tmp_path, raw))
    assert config.hardware == "real"


def test_position_limits_require_lo_lt_hi(tmp_path):
    raw = _raw()
    raw["limits"]["position"]["iiwa_A1"] = [1.0, -1.0]
    with pytest.raises(ConfigError, match="lo < hi"):
        load_lab_config(_write(tmp_path, raw))


def test_additive_probe_budget_refused_on_hardware(tmp_path):
    raw = _raw()
    raw["excitation"]["probe"]["budget"] = "additive"
    with pytest.raises(ConfigError, match="additive"):
        load_lab_config(_write(tmp_path, raw))


# ---------------------------------------------------------------------------
# Scene semantic-type expansion
# ---------------------------------------------------------------------------


def test_floor_expands_below_z(tmp_path):
    raw = _raw()
    raw["scene"]["objects"] = [{"id": "f", "type": "floor", "z": 0.0}]
    raw["scene"]["clearance"] = 0.0
    config = load_lab_config(_write(tmp_path, raw))
    (box,) = config.scene.objects
    assert isinstance(box, SceneBox)
    # The whole box lies on the far (below) side of z=0.
    assert box.xyz[2] + box.size[2] / 2.0 <= 1e-9


def test_ceiling_expands_above_z(tmp_path):
    raw = _raw()
    raw["scene"]["objects"] = [{"id": "c", "type": "ceiling", "z": 2.0}]
    raw["scene"]["clearance"] = 0.0
    config = load_lab_config(_write(tmp_path, raw))
    (box,) = config.scene.objects
    assert box.xyz[2] - box.size[2] / 2.0 >= 2.0 - 1e-9


def test_wall_normal_points_toward_robot(tmp_path):
    raw = _raw()
    raw["scene"]["objects"] = [
        {"id": "w", "type": "wall", "point": [1.0, 0.0, 0.0], "normal": "-x", "size": [2.0, 2.0]}
    ]
    raw["scene"]["clearance"] = 0.0
    config = load_lab_config(_write(tmp_path, raw))
    (box,) = config.scene.objects
    # normal -x points from the wall toward -x, so the wall's box sits at x >= 1.0.
    assert box.xyz[0] >= 1.0 - 1e-9


def test_table_expands_below_top(tmp_path):
    raw = _raw()
    raw["scene"]["objects"] = [
        {"id": "t", "type": "table", "top_z": 1.0, "centre_xy": [0.0, 0.0], "size_xy": [0.5, 0.5], "thickness": 0.1}
    ]
    raw["scene"]["clearance"] = 0.0
    config = load_lab_config(_write(tmp_path, raw))
    (box,) = config.scene.objects
    assert abs((box.xyz[2] + box.size[2] / 2.0) - 1.0) < 1e-9


def test_cylinder_and_sphere_expand(tmp_path):
    raw = _raw()
    raw["scene"]["objects"] = [
        {"id": "col", "type": "cylinder", "height": 1.0, "radius": 0.1, "xyz": [0, 0, 0]},
        {"id": "ball", "type": "sphere", "radius": 0.2, "xyz": [1, 1, 1]},
    ]
    raw["scene"]["clearance"] = 0.05
    config = load_lab_config(_write(tmp_path, raw))
    cyl, sphere = config.scene.objects
    assert isinstance(cyl, SceneCylinder) and cyl.radius == pytest.approx(0.15)
    assert isinstance(sphere, SceneSphere) and sphere.radius == pytest.approx(0.25)


def test_keep_out_is_virtual_box(tmp_path):
    raw = _raw()
    raw["scene"]["objects"] = [
        {"id": "zone", "type": "keep_out", "size": [1, 1, 1], "xyz": [0, 0, 0]}
    ]
    config = load_lab_config(_write(tmp_path, raw))
    (box,) = config.scene.objects
    assert box.virtual is True


def test_duplicate_object_id_refused(tmp_path):
    raw = _raw()
    raw["scene"]["objects"] = [
        {"id": "dup", "type": "sphere", "radius": 0.1, "xyz": [0, 0, 0]},
        {"id": "dup", "type": "sphere", "radius": 0.1, "xyz": [1, 1, 1]},
    ]
    with pytest.raises(ConfigError, match="duplicate"):
        load_lab_config(_write(tmp_path, raw))


# ---------------------------------------------------------------------------
# RR_04 A-6: scene.frame conversion into the bare asset's root frame
# ---------------------------------------------------------------------------

UR10_FIXTURE = Path(__file__).resolve().parents[1] / "config" / "lab" / "ur10_mock.yaml"


def _ur10_raw() -> dict:
    return yaml.safe_load(UR10_FIXTURE.read_text(encoding="utf-8"))


def test_iiwa_base_frame_is_identity(tmp_path):
    # iiwa_base -> the bare asset's root is a verified zero-offset fixed
    # joint (frames.py), so declaring the same object in `iiwa_base` must be
    # a no-op.
    raw = _raw()
    raw["scene"]["frame"] = "iiwa_base"
    raw["scene"]["objects"] = [{"id": "s", "type": "sphere", "radius": 0.1, "xyz": [0.3, -0.2, 0.5]}]
    config = load_lab_config(_write(tmp_path, raw))
    (sphere,) = config.scene.objects
    assert sphere.xyz == pytest.approx((0.3, -0.2, 0.5))


def test_ur10_base_frame_negates_xy_relative_to_base_link(tmp_path):
    # RR_04 A-6's own acceptance: "a UR obstacle at pendant coordinates (x, y)
    # in `base` sits at (-x, -y) in `base_link`" (base = Rz(pi) of base_link).
    raw = _ur10_raw()
    raw["scene"]["frame"] = "base"
    raw["scene"]["objects"] = [{"id": "b", "type": "box", "size": [0.1, 0.1, 0.1], "xyz": [0.4, 0.6, 0.2]}]
    (tmp_path / "a").mkdir()
    config_in_base = load_lab_config(_write(tmp_path / "a", raw))
    (box_in_base,) = config_in_base.scene.objects

    raw2 = _ur10_raw()
    raw2["scene"]["frame"] = "base_link"
    raw2["scene"]["objects"] = [{"id": "b", "type": "box", "size": [0.1, 0.1, 0.1], "xyz": [-0.4, -0.6, 0.2]}]
    (tmp_path / "b").mkdir()
    config_in_base_link = load_lab_config(_write(tmp_path / "b", raw2))
    (box_in_base_link,) = config_in_base_link.scene.objects

    assert box_in_base.xyz == pytest.approx(box_in_base_link.xyz, abs=1e-9)


def test_unknown_scene_frame_is_a_config_error(tmp_path):
    raw = _raw()
    raw["scene"]["frame"] = "not_a_real_frame"
    with pytest.raises(ConfigError, match="not_a_real_frame"):
        load_lab_config(_write(tmp_path, raw))
