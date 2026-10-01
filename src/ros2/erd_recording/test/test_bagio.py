"""RR_04 A-8: the pure bag-decode/alignment/gap helpers, unit-tested with
fake message-shaped objects instead of a live bag (no ROS environment
needed -- see bagio.py's module docstring)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from erd_recording.bagio import assign_segments, count_gaps, decode_dynamic_joint_state, find_segment_start


@dataclass
class _FakeInterfaceValue:
    interface_names: list[str]
    values: list[float]


@dataclass
class _FakeDynamicJointState:
    joint_names: list[str] = field(default_factory=list)
    interface_values: list[_FakeInterfaceValue] = field(default_factory=list)


def test_decode_dynamic_joint_state_reads_by_name_not_index():
    # A shuffled joint order must still decode correctly (RR_01 S10).
    msg = _FakeDynamicJointState(
        joint_names=["joint_a2", "joint_a1", "fri"],
        interface_values=[
            _FakeInterfaceValue(["position", "effort"], [0.2, -2.0]),
            _FakeInterfaceValue(["position", "effort"], [0.1, -1.0]),
            _FakeInterfaceValue(["cycle", "session_state"], [42.0, 4.0]),
        ],
    )
    decoded = decode_dynamic_joint_state(msg)
    assert decoded["joint_a1"]["position"] == 0.1
    assert decoded["joint_a2"]["effort"] == -2.0
    assert decoded["fri"]["cycle"] == 42.0


def test_find_segment_start_on_a_synthetic_delayed_copy():
    # RR_01 S10: "segment alignment on a synthetic delayed copy" -- exact to
    # one sample.
    rng = np.random.default_rng(0)
    plan = np.sin(2 * np.pi * 0.3 * np.arange(200) * 0.01) + 0.01 * rng.normal(size=200)
    pre_roll = 37
    post_roll = 53
    reference = np.concatenate([rng.normal(scale=0.001, size=pre_roll), plan,
                                rng.normal(scale=0.001, size=post_roll)])
    start = find_segment_start(reference, plan)
    assert start == pre_roll


def test_find_segment_start_exact_when_reference_equals_plan():
    plan = np.linspace(0, 1, 50)
    assert find_segment_start(plan, plan) == 0


def test_count_gaps_detects_a_dropped_cycle():
    nominal_dt = 0.001
    times = np.arange(100) * nominal_dt
    ok = count_gaps(times, nominal_dt)
    assert ok == 0
    times_with_gap = times.copy()
    times_with_gap[50:] += 5 * nominal_dt  # one dropped-cycle-sized gap
    assert count_gaps(times_with_gap, nominal_dt) == 1


def test_assign_segments_labels_windows_and_leaves_gaps_unlabelled():
    labels = assign_segments(10, [("approach_0", 0, 3), ("excitation_0", 5, 8)])
    assert labels == ["approach_0", "approach_0", "approach_0", "", "",
                      "excitation_0", "excitation_0", "excitation_0", "", ""]
