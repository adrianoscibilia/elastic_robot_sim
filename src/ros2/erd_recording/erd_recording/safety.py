"""Pure (ROS-free) safety logic: the start-state guard (RR_04 A-2) and the
live abort-condition monitor (RR_04 A-1).

Deliberately separate from :mod:`erd_recording.pipeline`, which eagerly
imports ``rclpy`` at module scope (T1.0/T1.6): everything here is plain
dicts/dataclasses/numbers, so it is unit-testable without a ROS environment
sourced, the same reasoning that put ``erd_iiwa``'s session-loss decision in
a free function (RR_04 A-4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

#: RR_04 A-1: a monitor tick must reach a cancel decision within this long of
#: the triggering sample, measured end-to-end at 1 kHz.
MONITOR_LATENCY_BUDGET_S = 0.020
#: RR_04 A-1: "no new sample for > 50 ms" is itself an abort condition.
STALE_SAMPLE_BUDGET_S = 0.050
#: RR_04 A-1: speed scaling times the excitation's target fraction must stay
#: within this of 1.0 during an excitation segment.
SPEED_SCALING_FLOOR = 0.999


@dataclass(frozen=True)
class GuardResult:
    ok: bool
    reason: str = ""


def check_start_state(
    measured: Mapping[str, float], first_point: Mapping[str, float], *, tolerance_rad: float = 0.01,
) -> GuardResult:
    """RR_04 A-2: before every goal, compare the latest measured position
    (driver joint names, mapped **by name**, never by index) with the goal's
    first point. Refused above ``tolerance_rad`` on any joint."""
    problems = []
    for joint, target in first_point.items():
        current = measured.get(joint)
        if current is None:
            return GuardResult(False, f"no measured position for joint {joint!r}")
        delta = abs(current - target)
        if delta > tolerance_rad:
            problems.append(f"{joint}: |{current:.4f} - {target:.4f}| = {delta:.4f} rad > {tolerance_rad} rad")
    if problems:
        return GuardResult(False, "start-state guard refused a jump: " + "; ".join(problems))
    return GuardResult(True)


def check_at_home_and_still(
    measured_position: Mapping[str, float], measured_velocity: Mapping[str, float], home: Mapping[str, float],
    *, position_tolerance_rad: float = 0.01, velocity_tolerance_rad_s: float = 0.01,
) -> GuardResult:
    """RR_04 A-2 preflight: refuse unless the robot is at ``home`` within
    ``position_tolerance_rad`` on every joint **and** still (``|dq| <
    velocity_tolerance_rad_s``)."""
    problems = []
    for joint, target in home.items():
        current = measured_position.get(joint)
        if current is None:
            return GuardResult(False, f"no measured position for joint {joint!r}")
        if abs(current - target) > position_tolerance_rad:
            problems.append(f"{joint} not at home: |{current:.4f} - {target:.4f}| > {position_tolerance_rad}")
        velocity = measured_velocity.get(joint)
        if velocity is not None and abs(velocity) > velocity_tolerance_rad_s:
            problems.append(f"{joint} not still: |dq|={abs(velocity):.4f} rad/s > {velocity_tolerance_rad_s}")
    if problems:
        return GuardResult(False, "preflight home/still check failed: " + "; ".join(problems))
    return GuardResult(True)


# ---------------------------------------------------------------------------
# Live monitor (RR_04 A-1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MonitorSample:
    """One ``/dynamic_joint_states`` tick's worth of monitor inputs. Every
    field is optional: a field that doesn't apply to the running robot (e.g.
    ``fri_session_state`` on the UR10) is simply left ``None`` and its check
    is skipped -- the caller (``RecordingNode``) only fills in what its
    profile exports."""

    sample_age_s: float | None = None
    measured_position: Mapping[str, float] = field(default_factory=dict)
    reference_position: Mapping[str, float] | None = None
    measured_effort: Mapping[str, float] | None = None
    effort_limit: Mapping[str, float] | None = None
    fri_session_state: float | None = None
    speed_scaling: float | None = None
    target_speed_fraction: float | None = None
    robot_mode_running: bool | None = None
    safety_mode_normal: bool | None = None
    bag_alive: bool = True
    sidecar_alive: bool = True


#: FRI's ``COMMANDING_ACTIVE`` session state, as the GPIO interface encodes it
#: (RR_01 S3.3; same value KUKA::FRI::ESessionState uses).
FRI_COMMANDING_ACTIVE = 4


def evaluate_abort_conditions(
    sample: MonitorSample, *, tracking_rad: float, torque_fraction: float | None = None,
    stale_sample_budget_s: float = STALE_SAMPLE_BUDGET_S, speed_scaling_floor: float = SPEED_SCALING_FLOOR,
) -> list[str]:
    """RR_04 A-1's abort table. Returns the list of triggered reasons (empty
    means "no abort"). Every joint-wise check names the joint on failure, so
    a triggered abort is diagnosable from the reason string alone."""
    reasons: list[str] = []

    if sample.sample_age_s is not None and sample.sample_age_s > stale_sample_budget_s:
        reasons.append(f"stale sample: {sample.sample_age_s * 1000:.1f} ms > {stale_sample_budget_s * 1000:.0f} ms")

    if sample.reference_position is not None:
        for joint, reference in sample.reference_position.items():
            measured = sample.measured_position.get(joint)
            if measured is None:
                continue
            error = abs(measured - reference)
            if error > tracking_rad:
                reasons.append(f"tracking error {joint}: |{measured:.4f} - {reference:.4f}| "
                               f"= {error:.4f} rad > {tracking_rad} rad")

    if sample.measured_effort is not None and sample.effort_limit is not None and torque_fraction is not None:
        for joint, effort in sample.measured_effort.items():
            limit = sample.effort_limit.get(joint)
            if limit is None:
                continue
            bound = torque_fraction * limit
            if abs(effort) > bound:
                reasons.append(f"torque {joint}: |{effort:.2f}| > {torque_fraction} * {limit:.2f} = {bound:.2f} Nm")

    if sample.fri_session_state is not None and sample.fri_session_state != FRI_COMMANDING_ACTIVE:
        reasons.append(f"fri/session_state = {sample.fri_session_state:g}, expected {FRI_COMMANDING_ACTIVE} "
                       "(COMMANDING_ACTIVE)")

    if sample.speed_scaling is not None and sample.target_speed_fraction is not None:
        achieved = sample.speed_scaling * sample.target_speed_fraction
        if achieved < speed_scaling_floor:
            reasons.append(f"speed scaling: {sample.speed_scaling:g} * {sample.target_speed_fraction:g} "
                           f"= {achieved:.4f} < {speed_scaling_floor}")

    if sample.robot_mode_running is False:
        reasons.append("robot mode is not RUNNING")
    if sample.safety_mode_normal is False:
        reasons.append("safety mode is not NORMAL")

    if not sample.bag_alive:
        reasons.append("the bag recorder process has exited")
    if not sample.sidecar_alive:
        reasons.append("the RTDE sidecar process has exited")

    return reasons
