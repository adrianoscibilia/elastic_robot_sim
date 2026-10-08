"""RR_18 R-5: the recorder start wait reads rosbag2's per-topic
"Subscribed to topic" lines, leaves the sidecar's topic out, and fails on a
topic that never appears (RR_17 F-14: it used to warn and continue)."""

from __future__ import annotations

from erd_recording.bag_record import (
    RECORDER_STARTUP_S,
    SIDECAR_STATUS_TOPIC,
    SUBSCRIBE_TIMEOUT_S,
    bag_start_state,
    wait_for_bag_start,
)

T0 = 1_791_407_315.452186619
TOPICS = ["/dynamic_joint_states", "/joint_states", "/erd/events",
          "/scaled_joint_trajectory_controller/controller_state"]


def _log(subscribed, *, recording=True, all_line=False):
    lines = [f"[INFO] [{T0:.9f}] [rosbag2_recorder]: Recording..."] if recording else []
    lines += [f"[DEBUG] [{T0 + 0.01 * k:.9f}] [rosbag2_recorder]: Subscribed to topic '{topic}' with QoS:"
              for k, topic in enumerate(subscribed)]
    if all_line:
        lines.append(f"[INFO] [{T0 + 15.2:.9f}] [rosbag2_recorder]: All requested topics are subscribed. Stopping discovery...")
    return "\n".join(lines) + "\n"


class _Clock:
    def __init__(self, start):
        self.now = start

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def test_ready_two_seconds_after_recording_once_every_topic_is_subscribed():
    log = _log(TOPICS)
    assert not bag_start_state(log, TOPICS, T0 + 1.0)["ready"]
    state = bag_start_state(log, TOPICS, T0 + RECORDER_STARTUP_S)
    assert state["ready"] and state["missing"] == [] and state["recording_s"] == T0


def test_a_ur_bag_does_not_wait_for_the_sidecar_topic():
    """The sidecar starts after the bag: the wait is over at ~2 s, not 15 s."""
    clock = _Clock(T0)
    log = _log(TOPICS)  # no /erd/rtde_status yet, and no "All requested topics" line
    state = wait_for_bag_start(lambda: log, [t for t in TOPICS + [SIDECAR_STATUS_TOPIC] if t != SIDECAR_STATUS_TOPIC],
                               timeout_s=SUBSCRIBE_TIMEOUT_S, alive=lambda: True, clock=clock, sleep=clock.sleep)
    assert state["ok"] and clock.now - T0 < RECORDER_STARTUP_S + 0.1


def test_a_topic_that_never_appears_fails_at_the_timeout():
    clock = _Clock(T0)
    log = _log(TOPICS)
    state = wait_for_bag_start(lambda: log, TOPICS + ["/erd/never_published"], timeout_s=SUBSCRIBE_TIMEOUT_S,
                               alive=lambda: True, clock=clock, sleep=clock.sleep)
    assert not state["ok"] and state["missing"] == ["/erd/never_published"]
    assert "15 s" in state["reason"] and clock.now - T0 >= SUBSCRIBE_TIMEOUT_S


def test_a_recorder_that_exits_fails_at_once():
    clock = _Clock(T0)
    state = wait_for_bag_start(lambda: "", TOPICS, timeout_s=SUBSCRIBE_TIMEOUT_S, alive=lambda: False,
                               clock=clock, sleep=clock.sleep)
    assert not state["ok"] and state["reason"] == "the recorder exited" and state["missing"] == TOPICS
    assert clock.now == T0


def test_the_sidecar_topic_is_confirmed_on_its_own_line():
    clock = _Clock(T0 + 20.0)
    log = _log(TOPICS + [SIDECAR_STATUS_TOPIC])
    state = wait_for_bag_start(lambda: log, [SIDECAR_STATUS_TOPIC], timeout_s=5.0, startup_s=0.0,
                               alive=lambda: True, clock=clock, sleep=clock.sleep)
    assert state["ok"]
    missing = wait_for_bag_start(lambda: _log(TOPICS), [SIDECAR_STATUS_TOPIC], timeout_s=5.0, startup_s=0.0,
                                 alive=lambda: True, clock=clock, sleep=clock.sleep)
    assert not missing["ok"] and missing["missing"] == [SIDECAR_STATUS_TOPIC]
