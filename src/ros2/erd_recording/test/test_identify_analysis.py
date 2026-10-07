"""RR_12 B-1 (identify fails on an incomplete segment) and B-3 (lag search
bound, gravity testability, UR anchor spacing), without a bag."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from erd_recording.identify_cli import (
    anchor_intervals,
    gravity_testability,
    lag_correlation,
    segment_health,
    windows,
)

DT = 0.001


def _segment(sid: str, kind: str, n: int):
    return SimpleNamespace(segment_id=sid, kind=kind, digest=f"d-{sid}",
                           trajectory=SimpleNamespace(time=np.arange(n) * DT, duration=(n - 1) * DT))


def _raw(n: int) -> pd.DataFrame:
    return pd.DataFrame({"stamp_ns": np.arange(n, dtype=np.int64) * 1_000_000, "fri_cycle": np.arange(n),
                         "t": np.arange(n) * DT})


def _events(sid: str, kind: str, start_ns: int, end_ns: int) -> list[dict]:
    return [{"stamp_ns": start_ns, "segment_id": sid, "kind": f"{kind}_start", "plan_digest": f"d-{sid}"},
            {"stamp_ns": end_ns, "segment_id": sid, "kind": f"{kind}_end", "plan_digest": f"d-{sid}"}]


def test_a_shortened_hold_makes_the_segment_invalid():
    raw = _raw(20_000)
    complete = _segment("identify_hold_0", "identify_hold", 5001)
    short = _segment("identify_hold_1", "identify_hold", 5001)
    events = (_events("identify_hold_0", "identify_hold", 0, 5_070_000_000)
              + _events("identify_hold_1", "identify_hold", 10_000_000_000, 12_551_000_000))
    frames, bounds = windows(raw, events, [complete, short])
    health = segment_health([complete, short], frames, bounds, losses=None)
    assert health["identify_hold_0"]["ok"]
    assert not health["identify_hold_1"]["ok"]
    assert health["identify_hold_1"]["robot_clock_samples"] == 2551
    assert abs(health["identify_hold_1"]["duration_s"] - 2.551) < 1e-9


def test_windows_use_publish_stamps_not_receive_stamps():
    raw = _raw(10_000)
    raw["bag_stamp_ns"] = raw["stamp_ns"] + 2_500_000_000  # a recorder 2.5 s late must not matter
    segment = _segment("identify_hold_0", "identify_hold", 5001)
    frames, _ = windows(raw, _events("identify_hold_0", "identify_hold", 1_000_000_000, 6_050_000_000), [segment])
    assert len(frames["identify_hold_0"]) == 5050


@pytest.mark.parametrize("lag_samples", [-37, -5, 0, 12, 49])
def test_lag_inside_the_50_ms_search_is_estimated(lag_samples):
    rng = np.random.default_rng(0)
    signal = np.convolve(rng.normal(size=4000), np.ones(20) / 20, mode="same")
    shifted = np.roll(signal, lag_samples)
    result = lag_correlation(shifted, signal, DT)
    assert not result["lag_at_bound"]
    assert abs(result["lag_s"] - lag_samples * DT) < 1e-12
    assert result["correlation"] > 0.99


@pytest.mark.parametrize("lag_samples", [80, -200])
def test_lag_outside_the_search_is_flagged_not_reported(lag_samples):
    # A slow signal: its correlation keeps rising towards the true lag, so
    # the in-window maximum sits on the bound (the l2 E-iiwa-1 pattern).
    rng = np.random.default_rng(1)
    signal = np.convolve(rng.normal(size=4000), np.ones(400) / 400, mode="same")
    result = lag_correlation(np.roll(signal, lag_samples), signal, DT)
    assert result["lag_at_bound"] is True
    assert result["lag_s"] is None


def test_gravity_slope_reported_only_where_testable():
    pose_gravity = [[0.0, 10.0, 0.0], [0.0, 40.0, 0.01], [0.0, 25.0, 0.02]]
    sigma = np.array([0.05, 0.2, 0.05])
    table = gravity_testability(pose_gravity, sigma, slope=np.array([0.3, 1.02, 7.0]),
                                r_squared=np.array([0.1, 0.999, 0.95]), joint_names=("A1", "A2", "A3"))
    assert [row["testable"] for row in table] == [False, True, False]
    assert table[1]["slope"] == 1.02
    assert table[0]["slope"] is None and "not testable at these poses" in table[0]["reason"]
    assert table[2]["slope"] is None and "gravity range" in table[2]["reason"]


def test_gravity_slope_needs_r_squared():
    table = gravity_testability([[0.0], [30.0]], np.array([0.1]), slope=np.array([1.0]),
                                r_squared=np.array([0.5]), joint_names=("A2",))
    assert not table[0]["testable"] and "r^2" in table[0]["reason"]


def test_ur_anchor_spacing_required_on_excitations_not_holds():
    excitation = _segment("identify_excitation_0", "identify_excitation", 100)
    hold = _segment("identify_hold_0", "identify_hold", 100)
    frames = {"identify_excitation_0": pd.DataFrame({"t": np.linspace(10.0, 20.0, 50), "dq0": 0.5}),
              "identify_hold_0": pd.DataFrame({"t": np.linspace(0.0, 5.0, 50)})}
    anchors = np.concatenate([np.arange(10.0, 14.0, 0.008), np.arange(14.5, 20.0, 0.008)])  # one 0.5 s hole
    table = anchor_intervals(anchors, [excitation, hold], frames)
    assert not table["identify_excitation_0"]["ok"]
    assert table["identify_excitation_0"]["max_anchor_interval_s"] == pytest.approx(0.508, abs=1e-6)
    assert table["identify_hold_0"]["ok"] and not table["identify_hold_0"]["required"]
    dense = anchor_intervals(np.arange(10.0, 20.0, 0.008), [excitation], frames)
    assert dense["identify_excitation_0"]["ok"]


def test_anchor_check_ignores_the_still_edges_of_the_window():
    excitation = _segment("identify_excitation_0", "identify_excitation", 100)
    t = np.arange(0.0, 9.4, 0.008)
    dq = np.where((t > 0.168) & (t < 9.3), 0.4, 0.0)  # standing for 0.168 s, as on URSim
    frames = {"identify_excitation_0": pd.DataFrame({"t": t, "dq0": dq})}
    anchors = t[(t > 0.168) & (t < 9.3)]
    table = anchor_intervals(anchors, [excitation], frames)
    assert table["identify_excitation_0"]["ok"]
    assert table["identify_excitation_0"]["max_anchor_interval_s"] < 0.02


def test_ursim_clock_ratio_scales_completeness():
    from erd_recording.bagio import segment_completeness
    assert not segment_completeness(594, 626)["ok"]               # real-time robot clock: 6 % short
    assert segment_completeness(594, 626, clock_ratio=0.937)["ok"]  # URSim at 0.937x
    assert not segment_completeness(300, 626, clock_ratio=0.937)["ok"]


def test_ursim_completeness_uses_the_window_clock_rate():
    """RR_15: rr15_ur10_sim_c's identify_hold_2 -- 581 robot samples in a
    5.02 s window at a local 0.926 x, against 585 required at the run's 0.939 x."""
    from erd_recording.bagio import segment_completeness
    from erd_recording.identify_cli import window_clock_ratio

    wall_s, local = 5.02, 0.926
    n = int(round(wall_s * local * 125)) + 1
    frame = pd.DataFrame({"timestamp": np.arange(n) * 0.008,
                          "stamp_ns": np.rint(np.arange(n) * 0.008 / local * 1e9).astype(np.int64)})
    assert not segment_completeness(n, 626, clock_ratio=0.939)["ok"]
    ratio = window_clock_ratio(frame, 0.939)
    assert abs(ratio - local) < 1e-9
    assert segment_completeness(n, 626, clock_ratio=ratio)["ok"]
    short = frame.iloc[: n // 2]  # a hold cut to half: still incomplete at its own rate
    assert not segment_completeness(len(short), 626, clock_ratio=window_clock_ratio(short, 0.939))["ok"]
    assert window_clock_ratio(frame, 1.0) == 1.0  # real/emulator/mock: never consulted
