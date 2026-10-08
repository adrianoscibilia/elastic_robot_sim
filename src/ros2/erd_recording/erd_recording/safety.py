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


def format_rad(value: float) -> str:
    """A tracking error or threshold with 3 significant digits (RR_16 Q-2):
    ``6.31e-04``."""
    return f"{float(value):.2e}"


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
                # RR_16 Q-2 (F-3): 3 significant digits, so a rung-4a reason
                # never reads "0.0006 rad > 0.000624 rad".
                reasons.append(f"tracking error {joint}: |{measured:.6f} - {reference:.6f}| "
                               f"= {format_rad(error)} rad > {format_rad(tracking_rad)} rad")

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


# ---------------------------------------------------------------------------
# Preflight on hardware: real (RR_12 A-3) and the A-4 override
# ---------------------------------------------------------------------------


def check_active_controllers(states: Mapping[str, str], expected: Sequence[str]) -> dict:
    """RR_12 A-3a: the set of **active** controllers must equal ``expected``
    (the JTC and the broadcasters the recording needs). Anything else active
    -- e.g. a second commanding controller -- refuses, as does a missing one."""
    active = sorted(name for name, state in states.items() if state == "active")
    unexpected = sorted(set(active) - set(expected))
    missing = sorted(set(expected) - set(active))
    return {"ok": not unexpected and not missing, "active": active, "expected": sorted(expected),
            "unexpected_active": unexpected, "missing": missing,
            "all": {name: states[name] for name in sorted(states)}}


def parse_software_version(text: str) -> tuple[int, ...]:
    """``"3.15.7.106331"`` -> ``(3, 15, 7, 106331)``."""
    try:
        return tuple(int(part) for part in str(text).strip().split("."))
    except ValueError:
        raise ValueError(f"software version {text!r} is not dotted integers") from None


def check_software_version(reported: Sequence[int] | None, configured: str) -> dict:
    """RR_12 A-3b (UR): the controller's ``get_robot_software_version``
    (major, minor, bugfix, build) must equal ``connection.software_version``
    on every component the config states (a 3-part config ignores the build)."""
    expected = parse_software_version(configured)
    if reported is None:
        return {"ok": False, "reason": "get_robot_software_version did not answer", "configured": configured}
    reported = tuple(int(v) for v in reported)
    ok = reported[: len(expected)] == expected
    return {"ok": ok, "reported": ".".join(map(str, reported)), "configured": configured,
            "compared_components": len(expected)}


#: RR_12 A-4: the deliberate-abort override exists only at small amplitude.
ABORT_OVERRIDE_MAX_LADDER = 0.1

#: The plan segment kinds that excite the arm (everything else is an
#: approach, a return or a hold).
EXCITATION_KINDS = frozenset({"excitation", "excitation_commissioning", "identify_excitation", "identify_sweep"})


def segment_tracking_rad(kind: str | None, *, file_rad: float, excitation_override: float | None) -> float:
    """RR_14 P-3: the monitor's tracking threshold for one segment. The
    ``--abort-tracking-rad`` override applies inside excitation segments
    only; approaches, returns and holds keep the lab file's value."""
    if excitation_override is not None and kind in EXCITATION_KINDS:
        return float(excitation_override)
    return float(file_rad)


#: RR_16 Q-2 (F-2): the only stage the override may reach. `identify` runs
#: excitation kinds too, and `all` would include it.
ABORT_OVERRIDE_STAGES = frozenset({"run"})


def check_abort_override(tracking_rad: float | None, ladder: float | None, *, file_rad: float,
                         stage: str) -> None:
    """Refuse ``--abort-tracking-rad`` unless it *lowers* the lab file's
    ``limits.abort.tracking_rad`` (RR_16 Q-2, F-1), the stage is ``run``
    (F-2) and ``--ladder <= 0.1`` is also given (RR_12 A-4)."""
    if tracking_rad is None:
        return
    if not tracking_rad > 0:
        raise ValueError(f"--abort-tracking-rad must be positive, got {tracking_rad}")
    if not tracking_rad < file_rad:
        raise ValueError(f"--abort-tracking-rad {format_rad(tracking_rad)} rad must be below the lab file's "
                         f"limits.abort.tracking_rad {format_rad(file_rad)} rad; it may only lower the threshold")
    if stage not in ABORT_OVERRIDE_STAGES:
        raise ValueError(f"--abort-tracking-rad is accepted only with the stage(s) {sorted(ABORT_OVERRIDE_STAGES)}, "
                         f"not {stage!r}; RR_12 A-4: rung 4a is a `run`")
    if ladder is None or ladder > ABORT_OVERRIDE_MAX_LADDER:
        raise ValueError(f"--abort-tracking-rad is accepted only with --ladder <= {ABORT_OVERRIDE_MAX_LADDER} "
                         f"(got --ladder {ladder}); RR_12 A-4: rung 4a only")
