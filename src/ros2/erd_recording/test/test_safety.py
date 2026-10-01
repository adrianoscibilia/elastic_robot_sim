"""RR_04 A-1/A-2: the live abort monitor and start-state guard, unit-tested
without a ROS environment (see erd_recording/safety.py's module docstring)."""

from __future__ import annotations

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
