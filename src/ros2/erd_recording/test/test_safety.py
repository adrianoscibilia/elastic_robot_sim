"""RR_04 A-1/A-2: the live abort monitor and start-state guard, unit-tested
without a ROS environment (see erd_recording/safety.py's module docstring)."""

from __future__ import annotations

import pytest

from erd_recording.safety import (
    FRI_COMMANDING_ACTIVE,
    MonitorSample,
    check_at_home_and_still,
    check_start_state,
    evaluate_abort_conditions,
)


# ---------------------------------------------------------------------------
# A-2: start-state guard
# ---------------------------------------------------------------------------


def test_start_state_guard_accepts_a_match_within_tolerance():
    measured = {"joint_a1": 0.001, "joint_a2": -0.5}
    first_point = {"joint_a1": 0.0, "joint_a2": -0.5 + 0.005}
    result = check_start_state(measured, first_point, tolerance_rad=0.01)
    assert result.ok


def test_start_state_guard_refuses_a_jump():
    measured = {"joint_a1": 0.0}
    first_point = {"joint_a1": 0.5}
    result = check_start_state(measured, first_point, tolerance_rad=0.01)
    assert not result.ok
    assert "joint_a1" in result.reason


def test_start_state_guard_refuses_on_missing_joint():
    result = check_start_state({}, {"joint_a1": 0.0}, tolerance_rad=0.01)
    assert not result.ok


def test_preflight_home_and_still_refuses_when_moving():
    measured_position = {"joint_a1": 0.0}
    measured_velocity = {"joint_a1": 0.2}
    home = {"joint_a1": 0.0}
    result = check_at_home_and_still(measured_position, measured_velocity, home)
    assert not result.ok
    assert "still" in result.reason


def test_preflight_home_and_still_passes_at_rest():
    measured_position = {"joint_a1": 0.001}
    measured_velocity = {"joint_a1": 0.0}
    home = {"joint_a1": 0.0}
    result = check_at_home_and_still(measured_position, measured_velocity, home)
    assert result.ok


# ---------------------------------------------------------------------------
# A-1: live abort monitor
# ---------------------------------------------------------------------------


def test_monitor_clean_sample_triggers_nothing():
    sample = MonitorSample(
        sample_age_s=0.001,
        measured_position={"joint_a1": 0.0}, reference_position={"joint_a1": 0.0},
        measured_effort={"joint_a1": 5.0}, effort_limit={"joint_a1": 320.0},
        fri_session_state=FRI_COMMANDING_ACTIVE,
    )
    assert evaluate_abort_conditions(sample, tracking_rad=0.05, torque_fraction=0.8) == []


def test_monitor_flags_stale_sample():
    sample = MonitorSample(sample_age_s=0.2)
    reasons = evaluate_abort_conditions(sample, tracking_rad=0.05)
    assert any("stale" in r for r in reasons)


def test_monitor_flags_tracking_error_and_names_the_joint():
    sample = MonitorSample(
        sample_age_s=0.001,
        measured_position={"joint_a1": 0.5, "joint_a2": 0.0},
        reference_position={"joint_a1": 0.0, "joint_a2": 0.0},
    )
    reasons = evaluate_abort_conditions(sample, tracking_rad=0.05)
    assert len(reasons) == 1
    assert "joint_a1" in reasons[0]


def test_monitor_flags_torque_over_fraction_iiwa_only():
    sample = MonitorSample(
        sample_age_s=0.001,
        measured_effort={"joint_a1": 300.0}, effort_limit={"joint_a1": 320.0},
    )
    reasons = evaluate_abort_conditions(sample, tracking_rad=0.05, torque_fraction=0.8)
    assert any("torque" in r for r in reasons)
    # No torque_fraction given (UR10 profile): the check is a no-op even with the same data.
    assert evaluate_abort_conditions(sample, tracking_rad=0.05, torque_fraction=None) == []


def test_monitor_flags_fri_session_loss():
    sample = MonitorSample(sample_age_s=0.001, fri_session_state=2.0)  # MONITORING_READY, not COMMANDING_ACTIVE
    reasons = evaluate_abort_conditions(sample, tracking_rad=0.05)
    assert any("session_state" in r for r in reasons)


def test_monitor_flags_speed_scaling_dip():
    sample = MonitorSample(sample_age_s=0.001, speed_scaling=0.9, target_speed_fraction=1.0)
    reasons = evaluate_abort_conditions(sample, tracking_rad=0.05)
    assert any("speed scaling" in r for r in reasons)


def test_monitor_flags_robot_and_safety_mode():
    sample = MonitorSample(sample_age_s=0.001, robot_mode_running=False, safety_mode_normal=False)
    reasons = evaluate_abort_conditions(sample, tracking_rad=0.05)
    assert any("robot mode" in r for r in reasons)
    assert any("safety mode" in r for r in reasons)


def test_monitor_flags_dead_bag_or_sidecar():
    sample = MonitorSample(sample_age_s=0.001, bag_alive=False, sidecar_alive=False)
    reasons = evaluate_abort_conditions(sample, tracking_rad=0.05)
    assert any("bag" in r for r in reasons)
    assert any("sidecar" in r for r in reasons)


# ---------------------------------------------------------------------------
# RR_12 A-3: preflight on hardware: real; A-4: the deliberate-abort override
# ---------------------------------------------------------------------------

from erd_recording.safety import (  # noqa: E402
    check_abort_override,
    check_active_controllers,
    check_software_version,
)

IIWA_EXPECTED = ("erd_arm_controller", "joint_state_broadcaster", "erd_state_broadcaster")


def test_active_controller_set_equal_to_expected_passes():
    states = {name: "active" for name in IIWA_EXPECTED}
    states["some_other_controller"] = "inactive"
    result = check_active_controllers(states, IIWA_EXPECTED)
    assert result["ok"], result
    assert result["active"] == sorted(IIWA_EXPECTED)


def test_an_extra_active_controller_refuses():
    states = {name: "active" for name in IIWA_EXPECTED}
    states["forward_position_controller"] = "active"
    result = check_active_controllers(states, IIWA_EXPECTED)
    assert not result["ok"]
    assert result["unexpected_active"] == ["forward_position_controller"]


def test_a_missing_expected_controller_refuses():
    states = {name: "active" for name in IIWA_EXPECTED}
    states["erd_state_broadcaster"] = "inactive"
    result = check_active_controllers(states, IIWA_EXPECTED)
    assert not result["ok"]
    assert result["missing"] == ["erd_state_broadcaster"]


def test_software_version_matches_on_the_configured_components():
    assert check_software_version((3, 15, 7, 106331), "3.15.7.106331")["ok"]
    assert check_software_version((3, 15, 8, 1), "3.15.8")["ok"]  # 3-part config ignores the build


def test_software_version_mismatch_refuses():
    result = check_software_version((3, 15, 8, 106331), "3.15.7.106331")
    assert not result["ok"]
    assert result["reported"] == "3.15.8.106331"
    assert not check_software_version(None, "3.15.7")["ok"]


FILE_RAD = 0.05


@pytest.mark.parametrize("tracking,ladder", [(0.005, None), (0.005, 0.25), (0.005, 1.0), (0.0, 0.1), (-0.01, 0.05)])
def test_abort_override_refused_without_small_ladder(tracking, ladder):
    with pytest.raises(ValueError):
        check_abort_override(tracking, ladder, file_rad=FILE_RAD, stage="run")


@pytest.mark.parametrize("ladder", [0.1, 0.05])
def test_abort_override_accepted_at_ladder_up_to_0_1(ladder):
    check_abort_override(0.005, ladder, file_rad=FILE_RAD, stage="run")
    check_abort_override(None, None, file_rad=FILE_RAD, stage="all")  # no override: nothing to check


# RR_16 Q-2 (F-1, F-2): the override may only lower the file's threshold, and only in `run`.

@pytest.mark.parametrize("tracking", [FILE_RAD, 0.6, 0.0500001])
def test_abort_override_refused_at_or_above_the_file_value(tracking):
    with pytest.raises(ValueError, match="below the lab file"):
        check_abort_override(tracking, 0.1, file_rad=FILE_RAD, stage="run")


@pytest.mark.parametrize("stage", ["all", "identify", "standstill", "preflight"])
def test_abort_override_refused_outside_run(stage):
    with pytest.raises(ValueError, match="accepted only with the stage"):
        check_abort_override(0.000624, 0.1, file_rad=FILE_RAD, stage=stage)


def test_abort_override_accepted_with_run_ladder_0_1_below_the_file_value():
    check_abort_override(0.000624, 0.1, file_rad=FILE_RAD, stage="run")


def test_cancel_reason_has_three_significant_digits():
    from erd_recording.reporting import _TRACKING
    from erd_recording.safety import MonitorSample, evaluate_abort_conditions

    sample = MonitorSample(measured_position={"A1": 0.100631}, reference_position={"A1": 0.1})
    reasons = evaluate_abort_conditions(sample, tracking_rad=0.000624)
    assert len(reasons) == 1
    assert "6.31e-04 rad > 6.24e-04 rad" in reasons[0], reasons[0]
    # report's cancel-latency reader still finds the threshold the cancel names
    assert float(_TRACKING.search(reasons[0]).group(1)) == pytest.approx(0.000624)
