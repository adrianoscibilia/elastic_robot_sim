"""Cancel latency for every abort (RR_12 A-2b), read back from a run's bags.

The trigger is the first sample, after the segment's start event, that crosses
the condition the cancel names; the latency runs from that sample's stamp to
the ``cancel`` event's publish stamp. Needs ``rosbag2_py`` (ROS sourced).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from .bagio import decode_dynamic_joint_state, read_bag_topic, read_events, read_header_stamped
from .safety import EXCITATION_KINDS, FRI_COMMANDING_ACTIVE, SPEED_SCALING_FLOOR, STALE_SAMPLE_BUDGET_S

_TRACKING = re.compile(r"tracking error .*?> ([0-9.eE+-]+) rad")


def classify_cancel(reason: str) -> str:
    """The monitor condition a cancel's reason string names."""
    if "SIGINT" in reason or "SIGTERM" in reason:
        return "signal"
    for key, name in (("tracking error", "tracking"), ("speed scaling", "speed_scaling"),
                      ("stale sample", "stale"), ("fri/session_state", "fri_session"), ("torque", "torque"),
                      ("timed out", "timeout")):
        if key in reason:
            return name
    return "other"


def first_crossing(stamps_ns: np.ndarray, values: np.ndarray, threshold: float, *, after_ns: int,
                   before_ns: int, above: bool = True) -> int | None:
    """Stamp of the first sample in ``[after_ns, before_ns]`` beyond ``threshold``."""
    stamps_ns = np.asarray(stamps_ns, dtype=np.int64)
    values = np.asarray(values, dtype=float)
    mask = (stamps_ns >= after_ns) & (stamps_ns <= before_ns) & ((values > threshold) if above else (values < threshold))
    hits = np.flatnonzero(mask)
    return int(stamps_ns[hits[0]]) if len(hits) else None


def cancel_latencies(bag_dir: Path, *, controller_name: str) -> list[dict[str, Any]]:
    """One row per ``cancel`` event in the bag."""
    from control_msgs.msg import DynamicJointState, JointTrajectoryControllerState

    events = read_events(bag_dir)
    cancels = [e for e in events if e["kind"] == "cancel"]
    if not cancels:
        return []
    rows = []
    controller = None
    djs = None
    speed = None
    for cancel in cancels:
        reason = cancel.get("detail", "")
        kind = classify_cancel(reason)
        starts = [e for e in events if e["kind"].endswith("_start") and e["stamp_ns"] <= cancel["stamp_ns"]]
        segment_start = starts[-1] if starts else None
        after = segment_start["stamp_ns"] if segment_start else 0
        trigger = None
        if kind == "tracking":
            match = _TRACKING.search(reason)
            threshold = float(match.group(1)) if match else None
            if controller is None:
                stamps, _, messages = read_header_stamped(bag_dir, f"/{controller_name}/controller_state",
                                                          JointTrajectoryControllerState)
                error = np.asarray([np.max(np.abs(np.asarray(m.reference.positions) - np.asarray(m.feedback.positions)))
                                    if m.reference.positions and m.feedback.positions else np.nan for m in messages])
                controller = (stamps, error)
            if threshold is not None:
                trigger = first_crossing(controller[0], controller[1], threshold, after_ns=after,
                                         before_ns=cancel["stamp_ns"])
        elif kind in ("fri_session", "stale"):
            if djs is None:
                stamps, _, messages = read_header_stamped(bag_dir, "/dynamic_joint_states", DynamicJointState)
                state = np.asarray([decode_dynamic_joint_state(m).get("fri", {}).get("session_state", np.nan)
                                    for m in messages])
                djs = (stamps, state)
            if kind == "fri_session":
                trigger = first_crossing(djs[0], np.abs(djs[1] - FRI_COMMANDING_ACTIVE), 0.5, after_ns=after,
                                         before_ns=cancel["stamp_ns"])
            else:
                before = djs[0][djs[0] <= cancel["stamp_ns"]]
                trigger = int(before[-1] + STALE_SAMPLE_BUDGET_S * 1e9) if len(before) else None
        elif kind == "speed_scaling":
            if speed is None:
                from std_msgs.msg import Float64

                data = [(stamp, msg.data / 100.0) for stamp, msg in
                        read_bag_topic(bag_dir, "/speed_scaling_state_broadcaster/speed_scaling", Float64)]
                speed = (np.asarray([d[0] for d in data], dtype=np.int64), np.asarray([d[1] for d in data]))
            # Float64 has no header: the recorder's receive stamp is the trigger time.
            trigger = first_crossing(speed[0], speed[1], SPEED_SCALING_FLOOR, after_ns=after,
                                     before_ns=cancel["recv_stamp_ns"], above=False)
        latency_ms = None if trigger is None else (cancel["stamp_ns"] - trigger) * 1e-6
        rows.append({"segment": segment_start["segment_id"] if segment_start else None, "reason": reason,
                     "condition": kind, "trigger_stamp_ns": trigger, "cancel_stamp_ns": cancel["stamp_ns"],
                     "latency_ms": latency_ms,
                     "note": "not a monitor trigger" if kind in ("signal", "timeout", "other") else
                             ("trigger is the last fresh sample + 50 ms" if kind == "stale" else None)})
    return rows


def latency_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = np.asarray([r["latency_ms"] for r in rows if r["latency_ms"] is not None], dtype=float)
    if not len(values):
        return {"n": 0}
    return {"n": int(len(values)), "min_ms": float(values.min()), "mean_ms": float(values.mean()),
            "max_ms": float(values.max())}


#: RR_14 P-3 (amends RR_12 S4.6 rung 4a): the rung-4a monitor threshold is
#: half the largest tracking error of the rung-4 `--ladder 0.1` run, and at
#: least this many encoder counts.
SUGGESTED_TRACKING_FRACTION = 0.5
SUGGESTED_TRACKING_MIN_COUNTS = 4

_EXCITATION_EVENT_KINDS = tuple(sorted(EXCITATION_KINDS))


def excitation_windows(events: list[dict[str, Any]]) -> list[tuple[str, int, int]]:
    """``(segment_id, start_ns, end_ns)`` of every completed excitation
    segment, from the publish stamps of its start/end events."""
    starts = {e["segment_id"]: e["stamp_ns"] for e in events
              if e["kind"] in tuple(f"{k}_start" for k in _EXCITATION_EVENT_KINDS)}
    return [(e["segment_id"], starts[e["segment_id"]], e["stamp_ns"]) for e in events
            if e["kind"] in tuple(f"{k}_end" for k in _EXCITATION_EVENT_KINDS) and e["segment_id"] in starts]


def max_tracking_error(stamps_ns: np.ndarray, error_rad: np.ndarray,
                       windows: list[tuple[str, int, int]]) -> dict[str, Any] | None:
    """Largest ``|reference - feedback|`` (any joint) inside the windows."""
    stamps_ns = np.asarray(stamps_ns, dtype=np.int64)
    error_rad = np.asarray(error_rad, dtype=float)
    best = None
    for segment, lo, hi in windows:
        mask = (stamps_ns >= lo) & (stamps_ns <= hi) & np.isfinite(error_rad)
        if not mask.any():
            continue
        value = float(error_rad[mask].max())
        if best is None or value > best["max_abs_error_rad"]:
            best = {"max_abs_error_rad": value, "segment": segment}
    return best


def suggest_abort_tracking_rad(max_abs_error_rad: float, count_rad: float) -> dict[str, Any]:
    """RR_14 P-3: ``0.5 x`` the largest error, floored at 4 encoder counts."""
    half = SUGGESTED_TRACKING_FRACTION * max_abs_error_rad
    floor = SUGGESTED_TRACKING_MIN_COUNTS * count_rad
    return {"suggested_rad": float(max(half, floor)), "half_max_error_rad": float(half),
            "floor_rad": float(floor), "floored": bool(floor > half),
            "rule": f"{SUGGESTED_TRACKING_FRACTION} x max |e| over excitation segments, "
                    f">= {SUGGESTED_TRACKING_MIN_COUNTS} encoder counts ({count_rad:.4g} rad each)"}


def tracking_suggestion(bag_dir: Path, *, controller_name: str, count_rad: float) -> dict[str, Any] | None:
    """The rung-4a suggestion from a run bag (``None`` without excitation)."""
    from control_msgs.msg import JointTrajectoryControllerState

    windows = excitation_windows(read_events(bag_dir))
    if not windows:
        return None
    stamps, _, messages = read_header_stamped(bag_dir, f"/{controller_name}/controller_state",
                                              JointTrajectoryControllerState)
    error = np.asarray([np.max(np.abs(np.asarray(m.reference.positions) - np.asarray(m.feedback.positions)))
                        if m.reference.positions and m.feedback.positions else np.nan for m in messages])
    worst = max_tracking_error(stamps, error, windows)
    if worst is None:
        return None
    return {**worst, "segments": [w[0] for w in windows],
            **suggest_abort_tracking_rad(worst["max_abs_error_rad"], count_rad)}
