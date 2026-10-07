"""RR_12 B-1 (segment completeness) and B-2 (recorder loss accounting), on
synthetic streams shaped like the pass-5 bags (RR_13 S2)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from erd_recording.bagio import (
    data_topic_losses,
    event_delivery,
    missing_stamps,
    parse_rosbag2_log,
    publication_gap_stamps,
    MAX_SKIPS_PER_SEGMENT,
    recorder_loss_report,
    robot_clock_samples,
    segment_completeness,
    segment_skips,
    split_update_gaps,
    tick_gap_stamps,
)

PERIOD_NS = 1_000_000


def _updates(n: int, *, seed: int = 0) -> np.ndarray:
    """Controller-manager update stamps at 1 kHz with scheduling jitter and,
    every ~7 s, a 4 ms stall followed by a catch-up burst (as on l2_iiwa_a)."""
    rng = np.random.default_rng(seed)
    stamps = np.arange(n, dtype=np.int64) * PERIOD_NS + rng.integers(-30_000, 30_000, n)
    for start in range(1000, n - 10, 7000):
        stamps[start] += 3_000_000
        stamps[start + 1:start + 4] = stamps[start] + np.array([30_000, 60_000, 90_000])
    return np.sort(stamps)


def _topic_from(updates: np.ndarray, *, seed: int = 1) -> np.ndarray:
    """A per-update topic stamped by its own now(): 5-40 us earlier."""
    rng = np.random.default_rng(seed)
    return updates - rng.integers(5_000, 40_000, len(updates))


def test_parse_rosbag2_log_total_and_per_topic_lines():
    text = ("[INFO] [1.0] [rosbag2_recorder]: Recording stopped\n"
            "[DEBUG] [1.1] [rosbag2_recorder]: Messages lost on transport layer for topic '/joint_states'. Total lost: 3\n"
            "[DEBUG] [1.2] [rosbag2_recorder]: Messages lost on transport layer for topic '/joint_states'. Total lost: 175\n"
            "[WARN] [1.3] [rosbag2_recorder]: Number of messages lost on the transport layer: 351\n")
    parsed = parse_rosbag2_log(text)
    assert parsed["total"] == 351
    assert parsed["per_topic"] == {"/joint_states": 175}
    assert parse_rosbag2_log("no summary: recorder killed")["total"] is None
    assert parse_rosbag2_log("[INFO] [2.0] [rosbag2_recorder]: Recording stopped\n")["total"] == 0


def test_jitter_and_catch_up_bursts_are_not_losses():
    updates = _updates(60_000)
    assert len(missing_stamps(updates, _topic_from(updates))) == 0


def test_a_recorder_blackout_is_attributed_with_its_time():
    updates = _updates(60_000)
    topic = _topic_from(updates)
    lost = slice(20_000, 20_175)  # l2_iiwa_emustop: ~175 per depth-1 topic in ~0.18 s
    kept = np.delete(topic, np.arange(lost.start, lost.stop))
    missing = missing_stamps(updates, kept)
    assert len(missing) == 175
    assert missing.min() >= updates[lost.start - 30] and missing.max() <= updates[lost.stop + 30]


def test_single_skips_are_counted_one_each():
    updates = _updates(30_000)
    topic = np.delete(_topic_from(updates), [5_000, 12_000, 12_001])
    assert len(missing_stamps(updates, topic)) == 3


def test_publication_gaps_from_received_cycle():
    header = np.arange(10, dtype=np.int64) * PERIOD_NS
    received = np.array([1, 2, 3, 6, 7, 8, 9, 10, 12, 13], dtype=float)
    stamps = publication_gap_stamps(header, received)
    assert len(stamps) == 3  # 2 after cycle 3, 1 after cycle 10
    assert list(stamps) == [3 * PERIOD_NS, 3 * PERIOD_NS, 8 * PERIOD_NS]


def test_tick_gaps_without_a_counter():
    stamps = np.array([0, 8, 16, 40, 48], dtype=np.int64) * 1_000_000  # 125 Hz, 2 ticks missing
    assert len(tick_gap_stamps(stamps, 0.008)) == 2


def test_losses_on_data_topics_inside_a_segment_invalidate_it():
    updates = _updates(20_000)
    controller = np.delete(_topic_from(updates, seed=3), [8_500, 8_501])  # a run of 2: a transport loss
    joint_states = np.delete(_topic_from(updates, seed=4), [2_000])
    report = recorder_loss_report(
        rosbag2_log_text="Number of messages lost on the transport layer: 2", djs_header_ns=updates,
        djs_missing_ns=np.array([], dtype=np.int64), djs_method="test",
        stamped_topics={"/joint_states": joint_states, "/c/controller_state": controller},
        controller_state_topic="/c/controller_state")
    assert report["unattributed"] == 0  # 2 lost on controller_state; the /joint_states single is a skip
    segment = (int(updates[7_000]), int(updates[9_000]))
    assert data_topic_losses(report, *segment) == 2          # the controller_state loss
    assert data_topic_losses(report, int(updates[1_000]), int(updates[3_000])) == 0  # /joint_states: reported only


def test_isolated_controller_state_misses_are_skips_runs_are_losses():
    updates = _updates(20_000)
    controller = np.delete(_topic_from(updates, seed=3), [8_500] + list(range(12_000, 12_010)))
    report = recorder_loss_report(
        rosbag2_log_text="Number of messages lost on the transport layer: 12", djs_header_ns=updates,
        djs_missing_ns=np.array([], dtype=np.int64), djs_method="test",
        stamped_topics={"/c/controller_state": controller}, controller_state_topic="/c/controller_state")
    entry = report["topics"]["/c/controller_state"]
    assert entry["missing"] == 11 and entry["publisher_skips"] == 1
    assert data_topic_losses(report, int(updates[8_000]), int(updates[9_000])) == 0    # the single: a skip
    assert data_topic_losses(report, int(updates[11_000]), int(updates[13_000])) == 10  # the run: a loss
    assert report["unattributed"] == 2


def test_completeness_requires_the_planned_robot_clock_samples():
    planned = 5001  # identify_hold_0: 5.0 s at 1 kHz
    full = pd.DataFrame({"fri_cycle": np.arange(5070)})
    shortened = pd.DataFrame({"fri_cycle": np.arange(2552)})  # the l2_iiwa_emustop window
    assert segment_completeness(robot_clock_samples(full), planned)["ok"]
    result = segment_completeness(robot_clock_samples(shortened), planned)
    assert not result["ok"] and result["required_samples"] == 5000
    duplicated = pd.DataFrame({"fri_cycle": np.repeat(np.arange(4000), 2)})  # re-published cycles count once
    assert robot_clock_samples(duplicated) == 4000


def test_event_delivery_reports_late_events():
    events = [{"stamp_ns": 0, "recv_stamp_ns": 2_571_910_000, "segment_id": "identify_pose_0_approach",
               "kind": "identify_approach_start"},
              {"stamp_ns": 5_000_000_000, "recv_stamp_ns": 5_000_080_000, "segment_id": "identify_hold_0",
               "kind": "identify_hold_end"}]
    delivery = event_delivery(events)
    assert abs(delivery["max_delay_s"] - 2.57191) < 1e-6
    assert delivery["events_later_than_50ms"] == 1


def _report(updates, controller, total):
    return recorder_loss_report(
        rosbag2_log_text=f"Number of messages lost on the transport layer: {total}", djs_header_ns=updates,
        djs_missing_ns=np.array([], dtype=np.int64), djs_method="test",
        stamped_topics={"/c/controller_state": controller}, controller_state_topic="/c/controller_state")


def test_negative_unattributed_voids_the_skip_exclusion():
    """RR_14 P-4a, rr13_link_300's shape: rosbag2 counted fewer transport
    losses than the bag's own runs, so every single counts."""
    updates = _updates(20_000)
    controller = np.delete(_topic_from(updates, seed=3), [8_500, 12_000, 12_001])
    report = _report(updates, controller, total=0)
    assert report["unattributed"] == -2 and report["skip_exclusion_void"]
    assert data_topic_losses(report, int(updates[8_000]), int(updates[9_000])) == 1  # the single now counts
    balanced = _report(updates, controller, total=2)
    assert balanced["unattributed"] == 0 and not balanced["skip_exclusion_void"]
    assert data_topic_losses(balanced, int(updates[8_000]), int(updates[9_000])) == 0


def test_void_is_derived_for_a_report_written_before_p4():
    updates = _updates(20_000)
    report = _report(updates, np.delete(_topic_from(updates, seed=3), [8_500, 12_000, 12_001]), total=0)
    report.pop("skip_exclusion_void")
    for entry in report["topics"].values():
        entry.pop("skip_stamps_ns")
    entry = report["topics"]["/c/controller_state"]
    entry["transport_stamps_ns"] = entry["missing_stamps_ns"][1:]  # as written then: the single excluded
    assert data_topic_losses(report, int(updates[8_000]), int(updates[9_000])) == 1


def test_more_than_three_singles_in_a_segment_invalidate_it():
    """RR_14 P-4b."""
    updates = _updates(30_000)
    singles = [2_000, 4_000, 6_000, 8_000]  # 4 isolated singles in [1000, 9000]
    controller = np.delete(_topic_from(updates, seed=5), singles + [20_000])
    report = _report(updates, controller, total=0)  # all singles: nothing attributed, unattributed 0
    assert report["unattributed"] == 0 and not report["skip_exclusion_void"]
    window = (int(updates[1_000]), int(updates[9_000]))
    assert segment_skips(report, *window) == 4 > MAX_SKIPS_PER_SEGMENT
    assert data_topic_losses(report, *window) == 4
    three = (int(updates[1_000]), int(updates[7_000]))
    assert segment_skips(report, *three) == 3
    assert data_topic_losses(report, *three) == 0


def test_mock_update_gaps_are_not_losses():
    """rr15_iiwa_mock: 4 gaps of 1-2 ticks on /dynamic_joint_states with
    /joint_states silent in the same gaps (the update never ran)."""
    djs = np.arange(20_000, dtype=np.int64) * PERIOD_NS
    js = djs - 20_000
    skipped = [5_000, 5_001, 9_000]  # two controller-manager update gaps
    lost = [12_000]                  # /joint_states has this update: a real loss
    stall = list(range(15_000, 15_175))  # both topics gone for 175 updates: a recorder stall
    header = np.delete(djs, skipped + lost + stall)
    companion = np.delete(js, skipped + stall)
    lost_ticks, gaps = split_update_gaps(header, 0.001, companion)
    assert len(gaps) == 3
    assert len(lost_ticks) == 1 + 175
    assert np.all(np.abs(lost_ticks[:1] - djs[12_000]) < PERIOD_NS / 2)
