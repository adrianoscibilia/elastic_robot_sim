"""RR_16 Q-1(b)-(g): transport losses from rosbag2's own per-topic events
(``rosbag2_recorder`` at DEBUG), with the count heuristics as a fail-closed
fallback; RR_18 R-1 (artefacts anchored at the first delivery), R-2 (whole
runs charged) and R-3 (skips judged by the hole they leave)."""

from __future__ import annotations

import numpy as np
import pytest

from erd_recording.bag_record import PRODUCTION_LOG_LEVEL, recorder_log_level
from erd_recording.bagio import (
    MAX_SKIPS_PER_SEGMENT,
    STARTUP_ARTEFACT_S,
    TRANSPORT_EVENT_WINDOW_S,
    classify_startup_artefacts,
    data_topic_losses,
    parse_rosbag2_events,
    recorder_loss_report,
    segment_skip_verdict,
    segment_skips,
    startup_artefact_checks,
)

PERIOD_NS = 1_000_000
T0_NS = 1_791_381_770_000_000_000  # wall clock, like the logs
CS = "/c/controller_state"
JS = "/joint_states"
W_S = 3.5


def _updates(n: int, period_ns: int = PERIOD_NS) -> np.ndarray:
    return T0_NS + np.arange(n, dtype=np.int64) * period_ns


def _stamp(ns: int) -> str:
    return f"{ns // 10**9}.{ns % 10**9:09d}"


def _log(*, subscribed: dict[str, int], events: list[tuple[str, int, int]], total: int | None,
         debug: bool = True) -> str:
    """A rosbag2 0.26.11 stderr log: "Subscribed" lines (with their QoS
    block), loss events ``(topic, t_ns, running total)``, the summary."""
    lines = [f"[INFO] [{_stamp(T0_NS - 10**9)}] [rosbag2_recorder]: Recording..."]
    entries = []
    if debug:
        entries += [(t, f"[DEBUG] [{_stamp(t)}] [rosbag2_recorder]: Subscribed to topic '{topic}' with QoS:\n"
                        "history: keep_last\ndepth: 2000\nreliability: reliable") for topic, t in subscribed.items()]
        entries += [(t, f"[DEBUG] [{_stamp(t)}] [rosbag2_recorder]: Messages lost on transport layer for topic "
                        f"'{topic}'. Total lost: {running}") for topic, t, running in events]
        entries.append((T0_NS - 1, f"[DEBUG] [{_stamp(T0_NS - 1)}] [rcl]: never written at the production level"))
    lines += [text for _, text in sorted(entries)]
    if total is not None:
        lines.append(f"[INFO] [{_stamp(T0_NS + 10**11)}] [rosbag2_recorder]: Recording stopped")
        if total:
            lines.append(f"[WARN] [{_stamp(T0_NS + 10**11)}] [rosbag2_recorder]: "
                         f"Number of messages lost on the transport layer: {total}")
    return "\n".join(lines) + "\n"


def _report(updates, controller, log, *, joint_states=None, djs_missing=(), first=None, window_s=W_S,
            first_segment=None):
    stamped = {CS: controller}
    if joint_states is not None:
        stamped[JS] = joint_states
    return recorder_loss_report(
        rosbag2_log_text=log, djs_header_ns=updates, djs_missing_ns=np.asarray(djs_missing, dtype=np.int64),
        djs_method="test", stamped_topics=stamped, controller_state_topic=CS,
        first_message_ns=first if first is not None else {CS: int(controller[0]), JS: int(updates[0])},
        first_segment_ns=first_segment, event_window_s=window_s)


def _segment(updates, a, b):
    return int(updates[a]), int(updates[b])


def test_production_level_is_the_recorder_logger_only():
    assert PRODUCTION_LOG_LEVEL == "rosbag2_recorder:=debug"
    assert recorder_log_level(None) == PRODUCTION_LOG_LEVEL
    assert recorder_log_level("debug") == "debug"  # --recorder-log-level: diagnostic, global


def test_parse_reads_subscriptions_events_increments_and_total():
    log = _log(subscribed={JS: T0_NS, CS: T0_NS + 5_000_000},
               events=[(JS, T0_NS + 370_000, 1), (CS, T0_NS + 5_520_000, 1), (CS, T0_NS + 9 * 10**9, 176)],
               total=177)
    parsed = parse_rosbag2_events(log)
    assert parsed["debug"] and parsed["total"] == 177
    assert parsed["subscribed_ns"] == {JS: T0_NS, CS: T0_NS + 5_000_000}
    assert [(e["topic"], e["increment"]) for e in parsed["events"]] == [(JS, 1), (CS, 1), (CS, 175)]
    assert parsed["events"][0]["t_ns"] == T0_NS + 370_000  # ns-exact from the log stamp


def test_a_start_up_artefact_is_excluded():
    """RR_15 P-4(c): 1 lost per depth-1 topic, < 0.5 ms after rosbag2 subscribed."""
    updates = _updates(20_000)
    controller = updates.copy()
    log = _log(subscribed={CS: T0_NS - 2 * 10**9}, events=[(CS, T0_NS - 2 * 10**9 + 520_000, 1)], total=1)
    report = _report(updates, controller, log)
    assert report["attribution"] == "events"
    assert report["startup_artefacts"] == 1 and report["transport_events"] == 0 and report["unattributed"] == 0
    assert report["topics"][CS]["rosbag2_events"][0]["startup_artefact"]


def test_an_artefact_needs_the_window_and_no_early_counted_miss():
    subscribed = {CS: 0}
    late = [{"topic": CS, "t_ns": int(STARTUP_ARTEFACT_S * 1e9) + 1, "increment": 1}]
    assert classify_startup_artefacts(late, subscribed, {}) == [False]
    early = [{"topic": CS, "t_ns": 500_000, "increment": 1}]
    assert classify_startup_artefacts(early, subscribed, {}) == [True]
    # rosbag2 logs the event after taking the first message (19-97 us in
    # rr15_iiwa_sim_debug); only a miss *inside* the recorded stream that
    # early makes it a loss
    assert classify_startup_artefacts(early, subscribed, {CS: np.array([5_000_000])}) == [False]
    assert classify_startup_artefacts(early, subscribed, {CS: np.array([2 * 10**9])}) == [True]
    assert classify_startup_artefacts(early, {}, {}) == [False]  # no subscription line
    # the event may be logged just before the "Subscribed" line (32 us, rr17_iiwa_sim)
    assert classify_startup_artefacts([{"topic": CS, "t_ns": -32_000, "increment": 1}], subscribed, {}) == [True]
    assert classify_startup_artefacts([{"topic": CS, "t_ns": -20_000_000, "increment": 1}], subscribed, {}) == [False]


def test_no_transport_event_means_every_miss_is_a_publisher_skip():
    """T = 0: runs (F-9's catch-up 2-runs) and singles alike are skips."""
    updates = _updates(30_000)
    misses = [5_000, 5_001, 12_000]
    controller = np.delete(updates, misses)
    log = _log(subscribed={CS: T0_NS - 2 * 10**9}, events=[(CS, T0_NS - 2 * 10**9 + 500_000, 1)], total=1)
    report = _report(updates, controller, log)
    entry = report["topics"][CS]
    assert entry["missing"] == 3 and entry["publisher_skips"] == 3 and entry["transport_stamps_ns"] == []
    assert data_topic_losses(report, *_segment(updates, 4_000, 13_000)) == 0  # 3 skips <= MAX


UR_PERIOD_NS = 8_000_000


def _ur_segment_report(skip_indices, *, debug=True):
    """A UR10-like bag at 125 Hz, 3,000 updates; the segment is updates
    1,000-2,249 (1,250 expected updates, F-13's 10 s identify segment)."""
    updates = _updates(3_000, UR_PERIOD_NS)
    controller = np.delete(updates, skip_indices)
    report = _report(updates, controller, _log(subscribed={CS: T0_NS - 10**9}, events=[], total=0, debug=debug),
                     first={CS: int(controller[0])})
    return report, _segment(updates, 1_000, 2_249)


def test_r3_four_singles_in_1250_updates_are_valid():
    """F-13: rr17_ur10_sim_4's identify_excitation_2 held 4 singles."""
    report, window = _ur_segment_report([1_100, 1_400, 1_700, 2_000])
    verdict = segment_skip_verdict(report, *window)
    assert verdict["skips"] == 4 > MAX_SKIPS_PER_SEGMENT and verdict["expected_updates"] == 1_250
    assert verdict["largest_gap_periods"] == pytest.approx(2.0) and verdict["largest_gap_ms"] == pytest.approx(16.0)
    assert not verdict["invalid"] and data_topic_losses(report, *window) == 0


def test_r3_thirteen_singles_in_1250_updates_are_invalid():
    report, window = _ur_segment_report(list(range(1_050, 2_250, 90))[:13])
    verdict = segment_skip_verdict(report, *window)
    assert verdict["skips"] == 13 and verdict["skip_fraction"] > 0.01 and verdict["invalid"]
    assert data_topic_losses(report, *window) == 13


def test_r3_one_four_period_gap_is_invalid_and_a_three_period_gap_is_not():
    four, window = _ur_segment_report([1_500, 1_501, 1_502])    # 3 consecutive skips: 4 periods
    verdict = segment_skip_verdict(four, *window)
    assert verdict["skips"] == 3 and verdict["largest_gap_periods"] == pytest.approx(4.0) and verdict["invalid"]
    assert data_topic_losses(four, *window) == 3
    three, window = _ur_segment_report([1_500, 1_501])           # a 2-run: 3 periods
    assert segment_skip_verdict(three, *window)["largest_gap_periods"] == pytest.approx(3.0)
    assert data_topic_losses(three, *window) == 0


def test_r3_one_single_skip_in_a_short_sweep_is_valid_two_are_not():
    """RR_19 D-12: a 96-update sweep (identify_sweep_*) with one single skip
    is 1.04 %; it is judged by its 2-period hole, not by the fraction."""
    updates = _updates(3_000, UR_PERIOD_NS)
    window = _segment(updates, 1_000, 1_095)
    log = _log(subscribed={CS: T0_NS - 10**9}, events=[], total=0)
    one = _report(updates, np.delete(updates, [1_050]), log)
    assert segment_skip_verdict(one, *window)["skip_fraction"] > 0.01
    assert data_topic_losses(one, *window) == 0
    two = _report(updates, np.delete(updates, [1_020, 1_070]), log)
    assert data_topic_losses(two, *window) == 2


def test_r3_count_fallback_keeps_three_skips():
    report, window = _ur_segment_report([1_100, 1_400, 1_700, 2_000], debug=False)
    assert report["attribution"] == "count_fallback"
    verdict = segment_skip_verdict(report, *window)
    assert verdict["invalid"] and verdict["rule"].startswith("count")
    assert data_topic_losses(report, *window) == 4
    three, window = _ur_segment_report([1_100, 1_400, 1_700], debug=False)
    assert data_topic_losses(three, *window) == 0


def test_r2_a_stall_longer_than_w_is_charged_whole():
    """F-16: a 12 s stall at 1 kHz, its event logged at the end (lag = the
    stall, RR_17 1.1). Before R-2 its first ~3.5 s became publisher skips."""
    updates = _updates(60_000)
    stall = list(range(30_000, 42_000))
    single = [2_000]                                  # a single skip 30 s earlier
    controller = np.delete(updates, stall + single)
    event_ns = int(updates[42_000]) + 15_000_000
    log = _log(subscribed={CS: T0_NS - 2 * 10**9}, events=[(CS, event_ns, 12_000)], total=12_000)
    report = _report(updates, controller, log, window_s=TRANSPORT_EVENT_WINDOW_S)
    entry = report["topics"][CS]
    assert report["attribution"] == "events" and report["unattributed"] == 0
    assert len(entry["transport_stamps_ns"]) == 12_000 and entry["unlocated_windows_ns"] == []
    assert entry["skip_stamps_ns"] == [int(updates[2_000])] and entry["publisher_skips"] == 1
    # the stall's first 3.5 s lie outside [t_e - W, t_e] and are still transport
    assert data_topic_losses(report, *_segment(updates, 30_000, 33_000)) == 3_001
    assert data_topic_losses(report, *_segment(updates, 1_000, 3_000)) == 0


def test_r1_ur_first_delivery_long_after_subscribing_is_an_artefact():
    """F-12: rr17_ur10_sim_3, first message 0.8 s after the "Subscribed"
    line, its event 0.5 ms after that message."""
    subscribed, first = {CS: 0}, {CS: 800_000_000}
    event = {"topic": CS, "t_ns": 800_500_000, "increment": 1}
    (check,) = startup_artefact_checks([event], subscribed, {}, first_message_ns=first)
    assert check == {"first_event_of_one": True, "near_start": True, "no_early_miss": True,
                     "before_first_segment": True}
    assert classify_startup_artefacts([event], subscribed, {}, first_message_ns=first) == [True]
    # without the first-delivery anchor (RR_17 D-9) it was transport
    assert classify_startup_artefacts([event], subscribed, {}) == [False]
    # increment 2
    assert classify_startup_artefacts([{**event, "increment": 2}], subscribed, {}, first_message_ns=first) == [False]
    # the topic's second event, even if it qualifies otherwise
    second = {**event, "t_ns": 800_600_000}
    assert classify_startup_artefacts([event, second], subscribed, {}, first_message_ns=first) == [True, False]
    # logged after the bag's first segment started
    assert classify_startup_artefacts([event], subscribed, {}, first_message_ns=first,
                                      first_segment_ns=800_400_000) == [False]
    # before the first message is not "after" it, and 10 ms + 1 ns is too late
    assert classify_startup_artefacts([{**event, "t_ns": 799_999_000}], subscribed, {}, first_message_ns=first) == [False]
    late = {**event, "t_ns": 800_000_000 + int(STARTUP_ARTEFACT_S * 1e9) + 1}
    assert classify_startup_artefacts([late], subscribed, {}, first_message_ns=first) == [False]


def test_r1_in_a_report_the_ur_artefact_is_excluded_and_a_later_event_is_transport():
    updates = _updates(3_000, UR_PERIOD_NS)
    controller = np.delete(updates, [2_000, 2_001])
    first_ns = int(controller[0])
    log = _log(subscribed={CS: first_ns - 830_000_000},
               events=[(CS, first_ns + 500_000, 1), (CS, int(updates[2_010]), 3)], total=3)
    report = _report(updates, controller, log, first={CS: first_ns}, first_segment=int(updates[100]))
    assert report["startup_artefacts"] == 1 and report["transport_events"] == 2 and report["unattributed"] == 0
    assert [e["startup_artefact"] for e in report["topics"][CS]["rosbag2_events"]] == [True, False]
    assert report["topics"][CS]["rosbag2_events"][0]["after_first_message_us"] == pytest.approx(500.0)


def test_an_event_charges_only_its_window():
    updates = _updates(60_000)
    stall = list(range(10_000, 10_175))   # a recorder stall at +10 s
    burst = [40_000, 40_001]              # a publisher-skip 2-run at +40 s
    controller = np.delete(updates, stall + burst)
    event_ns = int(updates[10_175]) + 800_000_000  # detected 0.8 s after the stall ended
    log = _log(subscribed={CS: T0_NS - 2 * 10**9}, events=[(CS, event_ns, 175)], total=175)
    report = _report(updates, controller, log)
    assert report["attribution"] == "events" and report["unattributed"] == 0
    entry = report["topics"][CS]
    assert len(entry["transport_stamps_ns"]) == 175 and entry["publisher_skips"] == 2
    assert entry["unlocated_windows_ns"] == []
    assert data_topic_losses(report, *_segment(updates, 9_000, 11_000)) == 175
    assert data_topic_losses(report, *_segment(updates, 39_000, 41_000)) == 0
    # the same stall detected later than W: nothing is in the window, so the loss is unlocated
    late = _report(updates, controller, _log(subscribed={CS: T0_NS - 2 * 10**9},
                                             events=[(CS, int(updates[10_175]) + 5 * 10**9, 175)], total=175))
    assert late["topics"][CS]["unlocated_windows_ns"]
    lo, hi, n = late["topics"][CS]["unlocated_windows_ns"][0]
    assert n == 175 and hi - lo == int(W_S * 1e9)
    assert data_topic_losses(late, int(updates[10_175]) + 3 * 10**9, int(updates[10_175]) + 4 * 10**9) >= 175


def test_a_dynamic_joint_states_event_with_no_counted_gap_is_unlocated():
    updates = _updates(30_000)
    event_ns = int(updates[20_000])
    log = _log(subscribed={"/dynamic_joint_states": T0_NS - 10**9},
               events=[("/dynamic_joint_states", event_ns, 2)], total=2)
    report = _report(updates, updates.copy(), log)
    entry = report["topics"]["/dynamic_joint_states"]
    assert entry["transport_lost"] == 2 and entry["unlocated_windows_ns"]
    assert data_topic_losses(report, *_segment(updates, 19_000, 21_000)) == 2


def test_joint_states_events_are_reported_but_never_invalidate():
    updates = _updates(30_000)
    joint_states = np.delete(updates, [15_000, 15_001])
    log = _log(subscribed={JS: T0_NS - 10**9}, events=[(JS, int(updates[15_100]), 2)], total=2)
    report = _report(updates, updates.copy(), log, joint_states=joint_states)
    assert len(report["topics"][JS]["transport_stamps_ns"]) == 2
    assert data_topic_losses(report, *_segment(updates, 14_000, 16_000)) == 0


def test_increments_that_do_not_reconcile_fall_back_to_counting():
    updates = _updates(30_000)
    controller = np.delete(updates, [8_000, 8_001, 20_000])
    log = _log(subscribed={CS: T0_NS - 10**9}, events=[(CS, int(updates[8_100]), 2)], total=5)
    report = _report(updates, controller, log)
    assert report["attribution"] == "count_fallback" and "reconcile" in report["fallback_reason"]
    # P-4 rules: the 2-run is transport, the single a skip; unattributed 5 - 2 = 3
    assert report["unattributed"] == 3 and not report["skip_exclusion_void"]
    assert data_topic_losses(report, *_segment(updates, 7_000, 9_000)) == 2


def test_no_debug_lines_fall_back_to_counting():
    updates = _updates(30_000)
    controller = np.delete(updates, [8_000, 8_001])
    report = _report(updates, controller, _log(subscribed={}, events=[], total=2, debug=False))
    assert report["attribution"] == "count_fallback" and "DEBUG" in report["fallback_reason"]


def test_a_missing_total_voids_the_skip_exclusion():
    updates = _updates(30_000)
    controller = np.delete(updates, [12_000])
    log = _log(subscribed={CS: T0_NS - 10**9}, events=[], total=None)  # recorder killed before its summary
    report = _report(updates, controller, log)
    assert report["attribution"] == "count_fallback" and "total" in report["fallback_reason"]
    assert report["skip_exclusion_void"]
    assert data_topic_losses(report, *_segment(updates, 11_000, 13_000)) == 1  # the single counts


def test_skip_runs_report_their_update_spacing():
    """Q-1(g), F-9's shape: a 2-run inside a catch-up burst ~70 us apart."""
    updates = _updates(30_000)
    updates[10_001:] += 3_000_000                     # a 4 ms stall ...
    updates[10_001:10_004] = updates[10_001] + np.array([0, 70_000, 140_000])  # ... and a catch-up burst
    updates[10_004:] = updates[10_003] + np.cumsum(np.full(len(updates) - 10_004, PERIOD_NS))
    controller = np.delete(updates, [10_002, 10_003])
    report = _report(updates, controller, _log(subscribed={CS: T0_NS}, events=[], total=0))
    (run,) = report["publisher_skip_runs"][CS]
    assert run["length"] == 2
    assert run["median_update_step_us"] == pytest.approx(70.0, abs=1.0)
    assert run["nominal_us"] == pytest.approx(1000.0, abs=1.0)
