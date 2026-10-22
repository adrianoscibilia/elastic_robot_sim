"""UR_11 (offline fix of the 2026-10-21 UR10 datasets): djs loss
attribution, the degenerate gravity testability, the nominal-model holdout
and the joint-size table, without a bag (``--keep-derived-raw``:
``test_keep_derived_raw.py``)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from erd_recording.bagio import data_topic_losses, ur_attributed_losses
from erd_recording.identification import friction_torque
from erd_recording.identify_cli import gravity_testability, nominal_model_holdout
from erd_recording.profile import IIWA_PROFILE, UR10_PROFILE

PERIOD = 8_000_000  # UR10, 125 Hz
SECOND = 1_000_000_000


def _djs_report(missing_ns, events, attribution="events"):
    return {"attribution": attribution, "nominal_period_ns": PERIOD, "event_window_s": 0.5,
            "topics": {"/dynamic_joint_states": {"data_topic": True, "missing": len(missing_ns),
                                                 "missing_stamps_ns": list(missing_ns),
                                                 "transport_stamps_ns": list(missing_ns),
                                                 "rosbag2_events": events}}}


def test_a_djs_skip_without_a_rosbag2_event_is_not_a_loss():
    # UR_10 S2: one skipped driver cycle, rosbag2 lost nothing.
    report = ur_attributed_losses(_djs_report([5 * SECOND], []))
    entry = report["topics"]["/dynamic_joint_states"]
    assert entry["transport_stamps_ns"] == [] and entry["skip_stamps_ns"] == [5 * SECOND]
    assert entry["publisher_skips"] == 1
    assert data_topic_losses(report, 0, 10 * SECOND) == 0


def test_a_djs_candidate_inside_a_rosbag2_event_window_is_a_loss():
    events = [{"t_ns": 5 * SECOND + 200_000_000}]  # the event follows the hole within its window
    report = ur_attributed_losses(_djs_report([5 * SECOND], events))
    entry = report["topics"]["/dynamic_joint_states"]
    assert entry["transport_stamps_ns"] == [5 * SECOND] and entry["skip_stamps_ns"] == []
    assert data_topic_losses(report, 0, 10 * SECOND) == 1
    late = ur_attributed_losses(_djs_report([5 * SECOND], [{"t_ns": 6 * SECOND}]))  # 1 s later: outside 0.5 s
    assert late["topics"]["/dynamic_joint_states"]["transport_stamps_ns"] == []
    startup = ur_attributed_losses(_djs_report([5 * SECOND], [dict(events[0], startup_artefact=True)]))
    assert startup["topics"]["/dynamic_joint_states"]["transport_stamps_ns"] == []


def test_count_fallback_reports_are_returned_unchanged():
    # fail closed: without the events attribution every candidate stays a loss
    report = _djs_report([5 * SECOND], [], attribution="count_fallback")
    assert ur_attributed_losses(report) is report
    assert ur_attributed_losses(None) is None


def test_zero_gravity_range_or_zero_sigma_is_not_testable():
    # UR_10 S3: wrist 3 (range 0, sigma 0, r^2 1 from ss_tot = 0) was judged
    # with slope 0 and failed the sign check; the iiwa mock's zero ft likewise.
    pose_gravity = [[0.0, 10.0, 0.0], [0.0, 40.0, 5.0], [0.0, 25.0, 9.0]]
    table = gravity_testability(pose_gravity, np.array([0.0, 0.2, 0.0]), slope=np.array([0.0, 1.0, 0.0]),
                                r_squared=np.array([1.0, 0.999, 1.0]), joint_names=("w3", "lift", "mock"))
    assert [row["testable"] for row in table] == [False, True, False]
    assert "degenerate" in table[0]["reason"] and table[0]["slope"] is None
    assert "degenerate" in table[2]["reason"]


def test_ur10_joint_sizes_follow_the_cited_table():
    # [S-6]: UR10 base/shoulder size 4, elbow size 3, wrists size 2.
    assert dict(zip(UR10_PROFILE.driver_joint_order, UR10_PROFILE.joint_sizes)) == {
        "shoulder_pan_joint": "size4", "shoulder_lift_joint": "size4", "elbow_joint": "size3",
        "wrist_1_joint": "size2", "wrist_2_joint": "size2", "wrist_3_joint": "size2"}
    assert "[S-6]" in UR10_PROFILE.joint_size_source
    assert IIWA_PROFILE.joint_sizes is None


def test_nominal_model_holdout_measures_the_proxy_against_the_bare_asset():
    from elastic_sim import identification as idn
    from elastic_sim.assets import AssetRegistry

    asset = AssetRegistry.for_repository().load("ur10")
    pin, model, data = idn.build_model(asset)
    rng = np.random.default_rng(0)
    q, dq, ddq = rng.uniform(-1, 1, (40, 6)), rng.uniform(-0.5, 0.5, (40, 6)), rng.uniform(-2, 2, (40, 6))
    k_tau = np.full(6, 12.0)
    fit = SimpleNamespace(j_m=np.full(6, 0.6), viscous=np.full(6, 3.0), coulomb=np.full(6, 2.0),
                          stribeck=np.zeros(6), stribeck_velocity=np.full(6, 0.03))
    rnea = np.asarray([pin.rnea(model, data, a, b, c).copy() for a, b, c in zip(q, dq, ddq)])
    torque = rnea + fit.j_m * ddq + friction_torque(dq, {"viscous": fit.viscous, "coulomb": fit.coulomb,
                                                         "stribeck": fit.stribeck,
                                                         "stribeck_velocity": fit.stribeck_velocity})
    mask = np.arange(40) < 10
    result = nominal_model_holdout(asset, q, dq, ddq, torque / k_tau, k_tau, fit, mask)
    assert result["rows"] == 10 and result["rms_nm"] < 1e-9
    assert result["target_rms_nm"] == pytest.approx(float(np.sqrt(np.mean(rnea[mask] ** 2))))
    offset = nominal_model_holdout(asset, q, dq, ddq, (torque + 1.0) / k_tau, k_tau, fit, mask)
    assert offset["rms_nm"] == pytest.approx(1.0)
    assert nominal_model_holdout(asset, q, dq, ddq, torque / k_tau, k_tau, fit, np.zeros(40, bool)) is None
