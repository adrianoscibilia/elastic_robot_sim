"""UR_01 U-1.2 / U-2 / U-3 / U-4: the UR10 real-data hardening."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from elastic_sim.assets import AssetRegistry
from elastic_sim.materialized import MaterializedTrajectory
from erd_recording.config import ConfigError, load_lab_config
from erd_recording.driver_facts import check_driver_facts, parse_driver_facts
from erd_recording.planning import (
    PlanError,
    WorldKinematics,
    cartesian_speed_peaks,
    check_vendor_speed_limits,
    quintic_segment,
    relimited_asset,
)
from erd_recording.validate import check_position_side

LAB = Path(__file__).resolve().parents[1] / "config" / "lab"


def _raw() -> dict:
    return yaml.safe_load((LAB / "ur10_sim.yaml").read_text(encoding="utf-8"))


def _write(tmp_path: Path, raw: dict) -> Path:
    path = tmp_path / "lab.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


# --- U-1.2 -----------------------------------------------------------------


def test_kinematics_file_expands_repo_root(tmp_path, monkeypatch):
    monkeypatch.setenv("ERD_REPO_ROOT", "/some/clone")
    raw = _raw()
    raw["connection"]["kinematics_params_file"] = "${ERD_REPO_ROOT}/calibration.yaml"
    config = load_lab_config(_write(tmp_path, raw))
    assert config.connection.kinematics_params_file == "/some/clone/calibration.yaml"
    assert config.raw["connection"]["kinematics_params_file"] == "${ERD_REPO_ROOT}/calibration.yaml"


def test_kinematics_file_unset_variable_refused(tmp_path, monkeypatch):
    monkeypatch.delenv("ERD_NOT_SET_ANYWHERE", raising=False)
    raw = _raw()
    raw["connection"]["kinematics_params_file"] = "${ERD_NOT_SET_ANYWHERE}/c.yaml"
    with pytest.raises(ConfigError, match="not set"):
        load_lab_config(_write(tmp_path, raw))


def test_real_refuses_a_missing_kinematics_file(tmp_path):
    from test_config import _filled_real_lab_file

    raw = _filled_real_lab_file("ur10")
    raw["connection"]["kinematics_params_file"] = str(tmp_path / "missing.yaml")
    with pytest.raises(ConfigError, match="kinematics_params_file does not exist"):
        load_lab_config(_write(tmp_path, raw))
    present = tmp_path / "calibration.yaml"
    present.write_text("kinematics: {}\n")
    raw["connection"]["kinematics_params_file"] = str(present)
    assert load_lab_config(_write(tmp_path, raw)).connection.kinematics_params_file == str(present)


def test_blocking_read_defaults_to_the_rehearsed_value(tmp_path):
    assert load_lab_config(_write(tmp_path, _raw())).connection.blocking_read is False
    raw = _raw()
    raw["connection"]["blocking_read"] = True
    assert load_lab_config(_write(tmp_path, raw)).connection.blocking_read is True


# --- U-2 -------------------------------------------------------------------

URDF = """<robot name="ur"><ros2_control name="ur" type="system"><hardware>
<plugin>ur_robot_driver/URPositionHardwareInterface</plugin>
<param name="servoj_gain">2000</param><param name="servoj_lookahead_time">0.03</param>
<param name="robot_receive_timeout">0.04</param><param name="kinematics/hash">abc123</param>
<param name="robot_ip">192.168.56.10</param></hardware></ros2_control></robot>"""


def test_driver_facts_parse():
    facts = parse_driver_facts(URDF)
    assert facts["servoj_gain"] == 2000.0 and facts["servoj_lookahead_time"] == 0.03
    assert facts["robot_receive_timeout"] == 0.04 and facts["kinematics/hash"] == "abc123"
    assert facts["use_currents_as_efforts"] is None


def test_real_refuses_a_servoj_gain_that_differs_from_the_driver():
    facts = parse_driver_facts(URDF)
    ok = check_driver_facts(facts, servoj_gain=2000, servoj_lookahead_time=0.03, hardware="real")
    assert ok["ok"] is True and not ok["problems"]
    bad = check_driver_facts(facts, servoj_gain=1500, servoj_lookahead_time=0.03, hardware="real")
    assert bad["ok"] is False and "servoj_gain" in bad["problems"][0]
    sim = check_driver_facts(facts, servoj_gain=1500, servoj_lookahead_time=0.03, hardware="ursim")
    assert sim["ok"] is True and sim["would_pass"] is False


def test_driver_packages_are_in_the_code_digest_block():
    from erd_recording.code_digest import DRIVER_PACKAGES, installed_driver_packages

    versions = installed_driver_packages(("definitely-not-a-package",))
    assert versions == {"definitely-not-a-package": None}
    assert "ros-jazzy-ur-calibration" in DRIVER_PACKAGES


# --- U-3 -------------------------------------------------------------------


def test_position_side_keys_load(tmp_path):
    config = load_lab_config(_write(tmp_path, _raw()))
    assert config.identification.position_side is None
    raw = _raw()
    raw["identification"]["position_side"] = "link"
    raw["identification"]["position_side_evidence"] = "ur_eur4"
    config = load_lab_config(_write(tmp_path, raw))
    assert (config.identification.position_side, config.identification.position_side_evidence) == ("link", "ur_eur4")
    raw["identification"]["position_side"] = "both"
    with pytest.raises(ConfigError, match="position_side"):
        load_lab_config(_write(tmp_path, raw))


def test_position_side_checks():
    assert check_position_side("link", hardware="real")["ok"] is False
    assert "collocated" in check_position_side("link", hardware="real")["reason"]
    assert check_position_side("motor", hardware="real")["ok"] is True
    pending = check_position_side("pending", hardware="real")
    assert pending["ok"] is True and pending["status"] == "pending"


def test_link_side_makes_the_dataset_invalid_on_real():
    from erd_recording.validate import finalize_checks

    def verdict(side, hardware="real"):
        return finalize_checks([check_position_side(side, hardware=hardware)], hardware=hardware, robot="ur10_cb3")

    assert verdict("link")["errors"] == ["motor-side position"]
    assert verdict("pending")["ok"] and verdict("motor")["ok"]
    assert verdict("link", "ursim")["ok"]


# --- U-4 -------------------------------------------------------------------


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    config = load_lab_config(LAB / "ur10_sim.yaml")
    snapshot = relimited_asset(AssetRegistry.for_repository().load(config.description.sim_asset), config,
                               tmp_path_factory.mktemp("ur10_plan"))
    return WorldKinematics(snapshot, (), ())


def test_cartesian_peak_of_a_single_joint_move_is_radius_times_rate(world):
    # Shoulder pan alone at 0.5 rad/s: the tool moves on a circle about z.
    start = np.array([0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
    end = start + np.array([1.0, 0, 0, 0, 0, 0])
    trajectory = quintic_segment(start, end, velocity_limit=0.5, acceleration_limit=1.0, time_step=0.008)
    radius = float(np.linalg.norm(world.forward(start)["arm"][:2]))
    peaks = cartesian_speed_peaks(world, trajectory)
    assert peaks["tool0"] == pytest.approx(radius * np.max(np.abs(trajectory.velocity[:, 0])), rel=2e-2)
    assert 0 < peaks["elbow"] < peaks["tool0"]


def test_a_segment_above_08_of_the_limit_is_refused():
    report = {"segments": {"excitation_3": {"tool0": 0.85, "elbow": 0.2}, "return_0": {"tool0": 0.1, "elbow": 0.1}}}
    check_vendor_speed_limits(report, {"tool_speed": 1.1})  # 0.85 <= 0.88
    with pytest.raises(PlanError, match="excitation_3: tool0 peak speed 0.850"):
        check_vendor_speed_limits(report, {"tool_speed": 1.0})
    with pytest.raises(PlanError, match="elbow"):
        check_vendor_speed_limits(report, {"elbow_speed": 0.2})
    check_vendor_speed_limits(report, {})  # nothing declared, nothing checked


def test_peaks_name_the_measured_frame(world):
    start = np.array([0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
    trajectory = quintic_segment(start, start + 0.2, velocity_limit=0.5, acceleration_limit=1.0, time_step=0.008)
    assert world.groups[0].tip_link == "tool0"
    assert cartesian_speed_peaks(world, trajectory)["tool0"] > 0


@pytest.mark.slow  # a full real-config plan, about 3 minutes
def test_a_refused_plan_leaves_no_manifest(tmp_path, monkeypatch):
    """UR_07 F-1: a later stage loads whatever ``plan.manifest.json`` is in the
    plan folder, so the vendor-speed refusal must come before the bundle write."""
    from erd_recording.planning import build_plan
    from test_config import _filled_real_lab_file

    monkeypatch.setenv("ERD_REPO_ROOT", str(Path(__file__).resolve().parents[4]))
    calibration = tmp_path / "calibration.yaml"
    calibration.write_text("kinematics:\n  hash: calib_1\n", encoding="utf-8")
    raw = _filled_real_lab_file("ur10", calibration)
    raw["safety"]["vendor_limits"] = {"tool_speed": 0.01, "elbow_speed": 0.01}
    config = load_lab_config(_write(tmp_path, raw))
    out = tmp_path / "plan"
    with pytest.raises(PlanError, match="pendant"):
        build_plan(config, out, n_candidates=4)
    assert (out / "cartesian_peaks.json").is_file()
    assert not (out / "plan.manifest.json").exists()
    assert not list(out.glob("excitation_*.json"))


# --- UR_07 F-3 -------------------------------------------------------------


def test_real_refuses_a_null_calibration_file(tmp_path):
    from test_config import _filled_real_lab_file

    raw = _filled_real_lab_file("ur10")  # kinematics_params_file: null
    with pytest.raises(ConfigError, match="kinematics_params_file is null"):
        load_lab_config(_write(tmp_path, raw))
    del raw["connection"]["kinematics_params_file"]
    with pytest.raises(ConfigError, match="kinematics_params_file is null"):
        load_lab_config(_write(tmp_path, raw))


def test_sim_and_mock_still_load_without_a_calibration_file(tmp_path):
    assert load_lab_config(LAB / "ur10_sim.yaml").connection.kinematics_params_file is None
    assert load_lab_config(LAB / "ur10_mock.yaml").connection.kinematics_params_file is None


def test_driver_hash_must_match_the_calibration_file(tmp_path):
    from erd_recording.driver_facts import read_kinematics_hash

    calibration = tmp_path / "calibration.yaml"
    calibration.write_text("kinematics:\n  hash: abc123\n  shoulder: {x: 0}\n", encoding="utf-8")
    assert read_kinematics_hash(calibration) == "abc123" and read_kinematics_hash(None) is None
    facts = parse_driver_facts(URDF)
    kwargs = dict(servoj_gain=2000, servoj_lookahead_time=0.03)
    ok = check_driver_facts(facts, hardware="real", kinematics_hash="abc123", **kwargs)
    assert ok["ok"] and not ok["problems"]
    bad = check_driver_facts(facts, hardware="real", kinematics_hash="other", **kwargs)
    assert bad["ok"] is False and "kinematics/hash" in bad["problems"][0]
    sim = check_driver_facts(facts, hardware="ursim", kinematics_hash="other", **kwargs)
    assert sim["ok"] is True and sim["would_pass"] is False
    assert check_driver_facts(facts, hardware="real", **kwargs)["ok"]  # no file, nothing to compare
