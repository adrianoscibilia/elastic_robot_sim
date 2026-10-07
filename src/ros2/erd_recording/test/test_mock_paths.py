"""RR_15: the L1 (mock) data paths behind RR_14 P-2's green L1 set. Mock
hardware has no robot clock, no FRI echo and no RTDE; what it lacks is zero
or NaN and labelled, never invented."""

from __future__ import annotations

import numpy as np
import pandas as pd

from erd_recording.bagio import robot_clock_samples
from erd_recording.ur_bagio import MOCK_NAN_FIELDS, MOCK_ZERO_FIELDS, mock_sidecar

JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint",
          "wrist_3_joint")


def _decoded(n: int) -> list[dict]:
    rows = []
    for k in range(n):
        row = {name: {"position": 0.01 * k + j, "velocity": 1.25, "effort": 0.0} for j, name in enumerate(JOINTS)}
        row["gpio"] = {"robot_mode": 7.0, "safety_mode": 1.0}
        row["speed_scaling"] = {"speed_scaling_factor": 1.0}
        rows.append(row)
    return rows


def test_mock_sidecar_is_a_uniform_125_hz_frame_of_the_driver_state():
    stamps = np.arange(1000, dtype=np.int64) * 8_000_000 + 10**18
    stamps = np.delete(stamps, [400, 401])  # a controller-manager update gap
    decoded = [d for k, d in enumerate(_decoded(1000)) if k not in (400, 401)]
    frame, aligned, report = mock_sidecar(stamps, decoded, JOINTS)
    assert len(frame) == 1000 and np.allclose(np.diff(frame["timestamp"]), 0.008)
    assert np.allclose(frame["actual_q1"], 0.01 * np.arange(1000) + 1)  # interpolated through the gap
    assert (aligned[0], aligned[-1]) == (stamps[0], stamps[-1])
    for field in MOCK_ZERO_FIELDS:
        assert (frame[f"{field}0"] == 0).all()
    for field in MOCK_NAN_FIELDS:
        assert frame[f"{field}5"].isna().all()
    assert (frame["robot_mode"] == 7).all() and (frame["safety_status"] == 1).all()
    assert report["anchor_timestamps"] == [] and report["gaps"] == []
    assert report["mock_fields"]["constant"] == {"target_speed_fraction": 1.0}
    assert robot_clock_samples(frame) == 1000


def test_mock_iiwa_frame_counts_rows_not_a_frozen_cycle():
    frame = pd.DataFrame({"fri_cycle": np.full(5070, np.nan), "t": np.arange(5070) * 1e-3})
    assert robot_clock_samples(frame) == 5070
    frozen = pd.DataFrame({"fri_cycle": np.full(5070, 11.0)})  # the mock's placeholder, if not cleared
    assert robot_clock_samples(frozen) == 1
