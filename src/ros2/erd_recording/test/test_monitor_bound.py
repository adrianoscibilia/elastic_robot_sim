"""RR_18 R-6: the rung-3 monitor sample-age bound per robot."""

from __future__ import annotations

import pytest

from erd_recording.reporting import judge_monitor_age, monitor_age_bound

IIWA, UR10 = "kuka_lbr_iiwa_14_r820", "ur10_cb3"


def _entry(p99, peak):
    return {"evaluations": 100, "stale_budget_ms": 50.0, "sample_age_ms": {"p50": 1.0, "p99": p99, "max": peak}}


def test_iiwa_keeps_rr12s_10_ms_p99():
    bound = monitor_age_bound(IIWA)
    assert bound["p99_ms"] == 10.0 and bound["max_below_ms"] is None
    assert judge_monitor_age(_entry(1.4, 3.0), IIWA)["pass"]        # rr17 emulator: p99 1.3-1.4 ms
    assert not judge_monitor_age(_entry(10.5, 12.0), IIWA)["pass"]


def test_ur10_bound_is_two_nominal_periods_and_max_below_the_stale_budget():
    bound = monitor_age_bound(UR10)
    assert bound["p99_ms"] == pytest.approx(16.0) and bound["max_below_ms"] == 50.0
    ursim = judge_monitor_age(_entry(9.6, 14.6), UR10)              # F-19: rr17 URSim worst case
    assert ursim["pass"] and ursim["p99"] == 9.6
    # 12 ms passes the UR10 bound and fails the iiwa one (9.6 ms passed both: F-19 "by luck")
    assert judge_monitor_age(_entry(12.0, 20.0), UR10)["pass"] and not judge_monitor_age(_entry(12.0, 20.0), IIWA)["pass"]
    assert not judge_monitor_age(_entry(16.1, 20.0), UR10)["pass"]
    assert not judge_monitor_age(_entry(9.0, 50.0), UR10)["pass"]    # max must stay below 50 ms


def test_no_aged_samples_is_a_fail():
    assert not judge_monitor_age({"evaluations": 0}, UR10)["pass"]
