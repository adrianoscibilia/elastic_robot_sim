"""Minimal MCAP bag reading (RR_01 S3.6, T1.6's "commanded reference in the
bag = plan to 1e-9 rad" check) and raw-telemetry extraction (RR_01 S8.1,
RR_04 A-8: "bagio builds raw.parquet from the bag").

The pure pieces (message-by-name decoding, segment-start alignment, gap
counting) are plain numpy/dict functions, unit-tested without a bag or a ROS
environment; only :func:`read_bag_topic`/:func:`bag_topics` and the
``build_*_raw_frame`` wrappers that call them need ``rosbag2_py`` sourced.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd


def read_bag_topic(bag_path: str | Path, topic: str, msg_type: type) -> Iterator[tuple[int, Any]]:
    """Yield ``(stamp_ns, message)`` for every message of ``topic`` in an MCAP
    rosbag2 bag directory, in recorded order."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message

    storage_options = rosbag2_py.StorageOptions(uri=str(bag_path), storage_id="mcap")
    converter_options = rosbag2_py.ConverterOptions(input_serialization_format="cdr",
                                                    output_serialization_format="cdr")
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
    while reader.has_next():
        read_topic, data, stamp_ns = reader.read_next()
        if read_topic != topic:
            continue
        yield stamp_ns, deserialize_message(data, msg_type)


def first_message_stamps(bag_path: str | Path, topics: Sequence[str]) -> dict[str, int | None]:
    """The bag (receive) stamp of each topic's first message; ``None`` for a
    topic with none."""
    import rosbag2_py

    result: dict[str, int | None] = {}
    for topic in topics:
        reader = rosbag2_py.SequentialReader()
        reader.open(rosbag2_py.StorageOptions(uri=str(bag_path), storage_id="mcap"),
                    rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"))
        reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
        result[topic] = None
        while reader.has_next():
            read_topic, _, stamp_ns = reader.read_next()
            if read_topic == topic:
                result[topic] = int(stamp_ns)
                break
    return result


def bag_topics(bag_path: str | Path) -> dict[str, str]:
    """``{topic_name: type_name}`` recorded in a bag, from its metadata."""
    import rosbag2_py

    storage_options = rosbag2_py.StorageOptions(uri=str(bag_path), storage_id="mcap")
    converter_options = rosbag2_py.ConverterOptions(input_serialization_format="cdr",
                                                    output_serialization_format="cdr")
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    return {info.name: info.type for info in reader.get_all_topics_and_types()}


# ---------------------------------------------------------------------------
# Pure decode/alignment/gap helpers (RR_04 A-8) -- unit-testable without ROS
# ---------------------------------------------------------------------------


def decode_dynamic_joint_state(msg: Any) -> dict[str, dict[str, float]]:
    """``{joint_name: {interface_name: value}}`` from one
    ``control_msgs/msg/DynamicJointState``-shaped object, decoded **by name**
    (never by index: RR_01 S10 "a shuffled /joint_states order converts
    correctly"). Works on the real message or any object exposing the same
    ``joint_names``/``interface_values[].{interface_names,values}`` shape
    (what the unit tests use, so this needs no live bag or ROS import)."""
    result: dict[str, dict[str, float]] = {}
    for joint_name, interface_value in zip(msg.joint_names, msg.interface_values):
        result[joint_name] = {
            name: float(value)
            for name, value in zip(interface_value.interface_names, interface_value.values)
        }
    return result


def find_segment_start(reference: np.ndarray, plan: np.ndarray) -> int:
    """The sample index in ``reference`` (the executed reference signal, e.g.
    iiwa ``commanded_position`` or UR ``target_q``, one joint's worth or a
    stacked/summed multi-joint signal) at which ``plan`` (the same segment's
    planned signal) best aligns, by cross-correlation (RR_01 S3.6
    "segmentation", "exact to one sample" -- RR_01 S10's synthetic-delayed-
    copy test).

    Both are 1-D. ``reference`` must be at least as long as ``plan``, and
    should extend somewhat before and after the true window so the
    correlation peak isn't an edge artifact.
    """
    reference = np.asarray(reference, dtype=float)
    plan = np.asarray(plan, dtype=float)
    if len(reference) < len(plan):
        raise ValueError("find_segment_start: reference must be at least as long as plan")
    # Cross-correlation via 'valid' convolution with the reversed plan signal,
    # both de-meaned so a DC offset between the two clocks' zero points
    # doesn't bias the peak.
    ref = reference - reference.mean()
    sig = plan - plan.mean()
    correlation = np.correlate(ref, sig, mode="valid")
    return int(np.argmax(correlation))


def count_gaps(times: np.ndarray, nominal_dt: float, *, rtol: float = 0.5) -> int:
    """Number of samples whose gap to the previous one exceeds
    ``(1 + rtol) * nominal_dt`` -- a dropped-cycle count from the robot's own
    clock (FRI ``cycle``/timestamp, or RTDE ``timestamp``), separate from any
    bag-side gap (RR_01 S3.6 "Drop policy": counted separately)."""
    times = np.asarray(times, dtype=float)
    if len(times) < 2:
        return 0
    dt = np.diff(times)
    return int(np.sum(dt > (1.0 + rtol) * nominal_dt))


def assign_segments(
    n_samples: int, segment_windows: Sequence[tuple[str, int, int]],
) -> list[str]:
    """A ``segment`` label per raw sample from ``[(segment_id, start, end), ...]``
    half-open index windows (RR_01 S8.1's ``segment`` column); samples outside
    every window are labelled ``""`` (e.g. inter-segment settle time)."""
    labels = [""] * n_samples
    for segment_id, start, end in segment_windows:
        for i in range(max(0, start), min(n_samples, end)):
            labels[i] = segment_id
    return labels


# ---------------------------------------------------------------------------
# Bag -> raw.parquet (iiwa): needs a live bag + ROS message types
# ---------------------------------------------------------------------------

#: iiwa raw columns decoded from `/dynamic_joint_states` (RR_01 S2/S8.1),
#: `{column_prefix: interface_name}` (state interfaces exported by
#: erd_iiwa/FriPositionSystem, RR_01 S3.3).
_IIWA_JOINT_INTERFACES = {
    "q": "position", "tau": "commanded_effort", "ft": "effort",
    "commanded_position": "commanded_position", "external_effort": "external_effort",
    "ipo_position": "ipo_position",
}
_IIWA_FRI_GPIO = ("time_sec", "time_nsec", "sample_time", "session_state", "command_mode",
                  "connection_quality", "tracking_performance", "safety_state", "operation_mode",
                  "drive_state", "cycle", "received_cycle")


def build_iiwa_raw_frame(
    bag_path: str | Path, *, driver_joint_order: Sequence[str], driver_to_sim: Mapping[str, str],
    joint_order: Sequence[str],
) -> pd.DataFrame:
    """Decode one bag's `/dynamic_joint_states` into the iiwa raw frame (RR_01
    S2: `tau` = FRI commanded torque raw, `ft` = FRI measured torque raw,
    `t` = the FRI timestamp). Columns are indexed in the **sim-asset** joint
    order (`joint_order`), mapped from the driver's names via
    `driver_to_sim` -- never by position (RR_01 S2's own warning: "the UR
    driver's /joint_states order isn't the URDF order"; the iiwa driver
    order need not match either)."""
    from control_msgs.msg import DynamicJointState

    n_dof = len(joint_order)
    sim_index = {name: index for index, name in enumerate(joint_order)}
    rows: list[dict[str, float]] = []
    for _, msg in read_bag_topic(bag_path, "/dynamic_joint_states", DynamicJointState):
        decoded = decode_dynamic_joint_state(msg)
        fri = decoded.get("fri", {})
        row: dict[str, float] = {}
        for driver_name in driver_joint_order:
            sim_name = driver_to_sim.get(driver_name)
            if sim_name is None or sim_name not in sim_index:
                continue
            index = sim_index[sim_name]
            interfaces = decoded.get(driver_name, {})
            for column_prefix, interface_name in _IIWA_JOINT_INTERFACES.items():
                row[f"{column_prefix}{index}"] = interfaces.get(interface_name, float("nan"))
        for name in _IIWA_FRI_GPIO:
            row[f"fri_{name}"] = fri.get(name, float("nan"))
        row["t"] = row.get("fri_time_sec", float("nan")) + 1e-9 * row.get("fri_time_nsec", 0.0)
        rows.append(row)
    frame = pd.DataFrame(rows)
    ordered = ["t"] + [f"{p}{i}" for p in _IIWA_JOINT_INTERFACES for i in range(n_dof)] + \
        [f"fri_{name}" for name in _IIWA_FRI_GPIO]
    return frame[ordered]


def read_header_stamped(bag_path: str | Path, topic: str, msg_type: type) -> tuple[np.ndarray, np.ndarray, list[Any]]:
    """``(header_stamp_ns, bag_receive_ns, messages)`` for one topic."""
    header, received, messages = [], [], []
    for stamp_ns, msg in read_bag_topic(bag_path, topic, msg_type):
        header.append(msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec)
        received.append(stamp_ns)
        messages.append(msg)
    return np.asarray(header, dtype=np.int64), np.asarray(received, dtype=np.int64), messages


def read_events(bag_path: str | Path) -> list[dict[str, Any]]:
    """``/erd/events`` with both clocks. ``stamp_ns`` is the **publish** time
    (header, the orchestrator's clock); ``recv_stamp_ns`` is when the recorder
    took it. RR_13 B-1: segment windows use ``stamp_ns``, because the recorder
    can deliver an event seconds late (``l2_iiwa_emustop``: 2.57 s)."""
    from erd_msgs.msg import RunEvent

    events = []
    for stamp, event in read_bag_topic(bag_path, "/erd/events", RunEvent):
        header = event.header.stamp.sec * 10**9 + event.header.stamp.nanosec
        events.append({"stamp_ns": int(header), "recv_stamp_ns": int(stamp), "segment_id": event.segment_id,
                       "kind": event.kind, "plan_digest": event.plan_digest, "detail": event.detail})
    events.sort(key=lambda e: e["stamp_ns"])
    return events


def event_delivery(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """How late the recorder received ``/erd/events`` (diagnostic only)."""
    delays = [(e["recv_stamp_ns"] - e["stamp_ns"]) * 1e-9 for e in events if "recv_stamp_ns" in e]
    if not delays:
        return {"n_events": 0}
    worst = int(np.argmax(delays))
    return {"n_events": len(delays), "max_delay_s": float(delays[worst]),
            "worst_event": f"{events[worst]['segment_id']} {events[worst]['kind']}",
            "events_later_than_50ms": int(sum(d > 0.05 for d in delays))}


def prepare_iiwa_conversion(config, run_dir: Path) -> None:
    """Read ROS messages here, before launching the ROS-free numerical stage."""
    import json
    from control_msgs.msg import DynamicJointState
    from .profile import profile_for

    profile = profile_for(config.robot)
    raw = build_iiwa_raw_frame(
        run_dir / "bag", driver_joint_order=profile.driver_joint_order,
        driver_to_sim={v: k for k, v in config.description.sim_to_driver().items()},
        joint_order=config.joint_order,
    )
    header, received, _ = read_header_stamped(run_dir / "bag", "/dynamic_joint_states", DynamicJointState)
    events = read_events(run_dir / "bag")
    if not np.any(header):
        # No stamp at all: fall back to the recorder clock for both streams.
        header = received.copy()
        for event in events:
            event["stamp_ns"] = event["recv_stamp_ns"]
    if config.hardware == "mock":
        # GenericSystem supplies no FRI clock. Preserve the controller-manager
        # clock; uniform resampling is explicitly synthetic and never used on FRI.
        raw["t"] = (header - header[0]) * 1e-9
        # Its fri/cycle and fri/received_cycle are constant placeholders
        # (the mock's initial values): no robot clock, so completeness
        # counts rows (robot_clock_samples), not a frozen cycle counter.
        for column in ("fri_cycle", "fri_received_cycle"):
            if column in raw:
                raw[column] = np.nan
    raw["stamp_ns"] = header       # controller-manager update time (same clock as event stamps)
    raw["bag_stamp_ns"] = received  # recorder receive time, diagnostic only
    raw.to_parquet(run_dir / "raw.parquet", index=False)
    (run_dir / "events.json").write_text(json.dumps(events, indent=2))
    write_recorder_losses(config, run_dir / "bag", run_dir / "raw" / "bag.rosbag2.stderr.log",
                          run_dir / "recorder_losses.json", djs_header=header, djs_frame=raw)
    write_controller_state(config, run_dir / "bag", header, run_dir / "controller_state.parquet")


def write_controller_state(config, bag_dir: Path, update_stamps_ns: np.ndarray, output: Path) -> None:
    """The JTC's ``controller_state.reference`` per message, in sim-asset
    joint order, for RR_01 S8.3's reference-tracking check. The topic is
    stamped by its own ``now()`` call (5-40 us off the update), so each
    message also gets the controller-manager update stamp it belongs to
    (``update_stamp_ns``: the nearest ``/dynamic_joint_states`` stamp within
    0.2 ms; 0 when ambiguous, e.g. inside a catch-up burst)."""
    from control_msgs.msg import JointTrajectoryControllerState
    from .profile import profile_for

    profile = profile_for(config.robot)
    topic = f"/{profile.controller_name}/controller_state"
    header, _, messages = read_header_stamped(bag_dir, topic, JointTrajectoryControllerState)
    sim_index = {sim: i for i, sim in enumerate(config.joint_order)}
    driver_to_sim = {driver: sim for sim, driver in config.description.sim_to_driver().items()}
    rows = []
    for msg in messages:
        row = {}
        for name, value in zip(msg.joint_names, msg.reference.positions):
            sim = driver_to_sim.get(name)
            if sim in sim_index:
                row[f"ref{sim_index[sim]}"] = float(value)
        rows.append(row)
    frame = pd.DataFrame(rows, columns=[f"ref{i}" for i in range(len(config.joint_order))])
    frame.insert(0, "stamp_ns", header)
    updates = np.asarray(update_stamps_ns, dtype=np.int64)
    if len(updates) and len(header):
        index = np.clip(np.searchsorted(updates, header), 1, len(updates) - 1)
        nearest = np.where(np.abs(updates[index] - header) < np.abs(updates[index - 1] - header),
                           updates[index], updates[index - 1])
        frame["update_stamp_ns"] = np.where(np.abs(nearest - header) <= 200_000, nearest, 0)
    else:
        frame["update_stamp_ns"] = 0
    frame.to_parquet(output, index=False)


# ---------------------------------------------------------------------------
# Segment completeness (RR_12 B-1) and recorder loss accounting (RR_12 B-2)
# ---------------------------------------------------------------------------


def event_window(events: Sequence[Mapping[str, Any]], segment_id: str) -> tuple[dict, dict] | None:
    """The first ``*_start`` and last ``*_end`` event of ``segment_id``."""
    starts = [e for e in events if e["segment_id"] == segment_id and e["kind"].endswith("_start")]
    ends = [e for e in events if e["segment_id"] == segment_id and e["kind"].endswith("_end")]
    if not starts or not ends:
        return None
    return starts[0], ends[-1]


def segment_completeness(n_recorded: int, n_planned: int, *, clock_ratio: float = 1.0) -> dict[str, Any]:
    """RR_12 B-1: a hold/excitation/sweep segment is complete when its robot
    clock carries at least ``n_planned - 1`` samples between its start and
    end events. A goal that ends early (abort, stop, a cut-short hold) fails
    this however healthy the samples it did record are.

    ``clock_ratio`` (robot seconds per wall second) is 1 on real hardware.
    URSim's controller runs slow (0.937, RR_11) while the JTC executes in ROS
    time, so its robot clock legitimately carries fewer samples; the
    requirement scales with the measured ratio there (L2 only)."""
    required = max(int(n_planned) - 1, 0)
    if clock_ratio != 1.0:
        # The ratio drifts over a 0.7 s sweep: one more sample of slack.
        required = max(int(np.floor(required * clock_ratio)) - 1, 0)
    return {"ok": int(n_recorded) >= required, "robot_clock_samples": int(n_recorded),
            "planned_samples": int(n_planned), "required_samples": required, "clock_ratio": float(clock_ratio)}


def robot_clock_samples(frame: pd.DataFrame) -> int:
    """Distinct robot-clock samples in a window: FRI cycles on the iiwa
    (duplicates, i.e. re-published cycles, count once), RTDE timestamps on
    the UR10, rows on mock (no robot clock)."""
    if "fri_cycle" in frame and frame["fri_cycle"].notna().any():
        return int(frame["fri_cycle"].dropna().nunique())
    if "timestamp" in frame:
        return int(frame["timestamp"].nunique())
    return int(len(frame))


_ROSBAG2_TOTAL = re.compile(r"Number of messages lost on the transport layer: (\d+)")
_ROSBAG2_TOPIC = re.compile(r"Messages lost on transport layer for topic '([^']+)'\. Total lost: (\d+)")


def parse_rosbag2_log(text: str) -> dict[str, Any]:
    """rosbag2 (Jazzy 0.26) end-of-recording loss summary. ``total`` is
    ``None`` when the line is missing (recorder killed before its summary).
    The per-topic line only exists at ``--log-level debug``; Jazzy has no
    lost-message statistics topic (RR_13 B-2)."""
    totals = [int(m.group(1)) for m in _ROSBAG2_TOTAL.finditer(text)]
    if not totals and "Recording stopped" in text:
        totals = [0]  # the summary line is a WARN printed only when something was lost
    per_topic: dict[str, int] = {}
    for match in _ROSBAG2_TOPIC.finditer(text):
        per_topic[match.group(1)] = max(per_topic.get(match.group(1), 0), int(match.group(2)))
    return {"total": totals[-1] if totals else None, "per_topic": per_topic}


_ROSBAG2_LINE = re.compile(r"^\[(DEBUG|INFO|WARN|ERROR|FATAL)\] \[(\d+)\.(\d+)\] \[rosbag2_recorder\]: (.*)$",
                           re.MULTILINE)
_SUBSCRIBED = re.compile(r"Subscribed to topic '([^']+)'")
_LOST_EVENT = re.compile(r"Messages lost on transport layer for topic '([^']+)'\. Total lost: (\d+)")

#: RR_16 Q-1(b), RR_18 R-1: a topic's first lost-message event this soon
#: after its first recorded message (or its "Subscribed" line) is the
#: depth-1 writer's last pre-match sample (RR_15 P-4(c)), not a loss --
#: unless the bag's own count shows a miss inside the recorded stream that
#: early, or a segment had already started.
STARTUP_ARTEFACT_S = 0.010


def _log_ns(seconds: str, fraction: str) -> int:
    return int(seconds) * 10**9 + int(fraction[:9].ljust(9, "0"))


def parse_rosbag2_events(text: str) -> dict[str, Any]:
    """RR_16 Q-1(b): rosbag2's own per-topic transport-loss events.

    ``debug`` is true when the log holds the recorder's DEBUG lines (its
    "Subscribed to topic" lines; RR_16 Q-1(a) sets that logger alone to
    DEBUG). Each event carries the wall time it was logged, the
    subscription's running total and the increment over its previous event."""
    subscribed: dict[str, int] = {}
    events: list[dict[str, Any]] = []
    last_total: dict[str, int] = {}
    debug = False
    for level, sec, frac, message in _ROSBAG2_LINE.findall(text):
        if level != "DEBUG":
            continue
        match = _SUBSCRIBED.match(message)
        if match:
            debug = True
            subscribed.setdefault(match.group(1), _log_ns(sec, frac))
            continue
        match = _LOST_EVENT.match(message)
        if match:
            topic, total = match.group(1), int(match.group(2))
            events.append({"topic": topic, "t_ns": _log_ns(sec, frac), "total": total,
                           "increment": total - last_total.get(topic, 0)})
            last_total[topic] = total
    return {"debug": debug, "subscribed_ns": subscribed, "events": events,
            "total": parse_rosbag2_log(text)["total"]}


def startup_artefact_checks(events: Sequence[Mapping[str, Any]], subscribed_ns: Mapping[str, int],
                            candidates_ns: Mapping[str, np.ndarray], *,
                            first_message_ns: Mapping[str, int | None] | None = None,
                            first_segment_ns: int | None = None,
                            window_s: float = STARTUP_ARTEFACT_S) -> list[dict[str, bool]]:
    """Per event, RR_18 R-1's four conditions for a start-up artefact (RR_16
    Q-1(b), amended by RR_17 D-9 and RR_18 R-1); all must hold:

    * ``first_event_of_one``: the topic's first event, with increment 1;
    * ``near_start``: logged within ``window_s`` *after* the topic's first
      recorded message, or within ``window_s`` either side of its
      "Subscribed to topic" line. rosbag2 takes the newest sample, then
      reports the older one the depth-1 writer overwrote, so the event
      follows the first message (>= 6.4 us in all 75 rr17 events); on the
      UR10 the first message itself can come 0.14-0.84 s after subscribing
      (RR_18 F-12), and the event can precede its own "Subscribed" line
      (32 us in rr17_iiwa_sim);
    * ``no_early_miss``: the bag's own count finds no miss on that topic up
      to ``window_s`` after the event (the lost sample precedes the recorded
      stream);
    * ``before_first_segment``: logged before the bag's first segment starts
      (``first_segment_ns``; no segment in the bag: holds).

    An unknown subscription time and first message is never an artefact."""
    first_message_ns = dict(first_message_ns or {})
    window_ns = window_s * 1e9
    seen: set[str] = set()
    checks = []
    for event in events:
        topic, t_ns = event["topic"], event["t_ns"]
        subscribed = subscribed_ns.get(topic)
        first = first_message_ns.get(topic)
        early = np.asarray(candidates_ns.get(topic, []), dtype=np.int64)
        checks.append({
            "first_event_of_one": topic not in seen and event["increment"] == 1,
            "near_start": bool((first is not None and 0 <= t_ns - first <= window_ns)
                               or (subscribed is not None and abs(t_ns - subscribed) <= window_ns)),
            "no_early_miss": not bool(np.any(early <= t_ns + window_ns)),
            "before_first_segment": first_segment_ns is None or t_ns < first_segment_ns,
        })
        seen.add(topic)
    return checks


def classify_startup_artefacts(events: Sequence[Mapping[str, Any]], subscribed_ns: Mapping[str, int],
                               candidates_ns: Mapping[str, np.ndarray], *,
                               first_message_ns: Mapping[str, int | None] | None = None,
                               first_segment_ns: int | None = None,
                               window_s: float = STARTUP_ARTEFACT_S) -> list[bool]:
    """Per event: is it a start-up artefact? (:func:`startup_artefact_checks`, all true.)"""
    return [all(check.values()) for check in startup_artefact_checks(
        events, subscribed_ns, candidates_ns, first_message_ns=first_message_ns,
        first_segment_ns=first_segment_ns, window_s=window_s)]


def missing_stamps(reference_ns: np.ndarray, topic_ns: np.ndarray, *, half_window: int = 25) -> np.ndarray:
    """Reference stamps (``/dynamic_joint_states``: one per controller-manager
    update) for which ``topic_ns`` has no message, inside the span both
    topics cover.

    ``/joint_states`` and the JTC ``controller_state`` are also published once
    per update, but stamped by their own ``now()`` call (5-40 us off the
    broadcaster's stamp), and after a scheduling stall the controller manager
    runs several catch-up updates within ~100 us, where those offsets reorder
    the streams. Pairing stamps one by one then invents losses. Instead:
    ``D_k`` = updates up to reference stamp ``k`` minus topic messages up to
    it. Jitter moves ``D`` by +-1 for a few samples; a lost message raises it
    for good. Each step of ``D``'s rolling median (``2 * half_window + 1``
    samples) is one loss, placed at that update's stamp.
    """
    reference_ns = np.asarray(reference_ns, dtype=np.int64)
    topic_ns = np.sort(np.asarray(topic_ns, dtype=np.int64))
    if len(reference_ns) < 2 or len(topic_ns) == 0:
        return np.asarray([], dtype=np.int64)
    period = float(np.median(np.diff(reference_ns)))
    inside = reference_ns[(reference_ns >= topic_ns[0] - 0.5 * period) & (reference_ns <= topic_ns[-1] + 0.5 * period)]
    if len(inside) == 0:
        return inside
    topic = topic_ns[(topic_ns >= inside[0] - 0.5 * period) & (topic_ns <= inside[-1] + 0.5 * period)]
    offset = np.median(topic[: min(len(topic), 1000)] - inside[np.clip(
        np.searchsorted(inside, topic[: min(len(topic), 1000)]), 0, len(inside) - 1)]) if len(topic) else 0.0
    counted = np.searchsorted(topic - offset, inside + 0.5 * period, side="right")
    difference = np.arange(1, len(inside) + 1) - counted
    padded = np.pad(difference, half_window, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * half_window + 1)
    baseline = np.median(windows, axis=1).astype(np.int64)
    baseline = baseline - baseline[0]
    steps = np.diff(np.concatenate([[0], baseline]))
    stamps: list[int] = []
    for index in np.flatnonzero(steps > 0):
        stamps += [int(inside[index])] * int(steps[index])
    return np.asarray(stamps, dtype=np.int64)


def tick_gap_stamps(stamps_ns: np.ndarray, nominal_dt_s: float) -> np.ndarray:
    """Approximate stamps of missing ticks in a fixed-rate stream without a
    sequence counter (UR ``/dynamic_joint_states``): a step over 1.5 periods
    loses ``round(step / dt) - 1`` ticks, placed evenly inside the step."""
    stamps_ns = np.asarray(stamps_ns, dtype=np.int64)
    dt_ns = nominal_dt_s * 1e9
    missing = []
    for left, step in zip(stamps_ns[:-1], np.diff(stamps_ns)):
        if step > 1.5 * dt_ns:
            count = int(round(step / dt_ns)) - 1
            missing += [int(left + k * step / (count + 1)) for k in range(1, count + 1)]
    return np.asarray(missing, dtype=np.int64)


#: RR_15: on ``hardware: mock`` a tick gap shared by every per-update topic is
#: a controller-manager update that never ran (non-realtime host), not a
#: lost message -- unless it is longer than this (a recorder stall loses
#: every topic at once: ~175 updates in l2_iiwa_emustop).
MAX_UPDATE_GAP_TICKS = 10


def split_update_gaps(header_ns: np.ndarray, nominal_dt_s: float,
                      companion_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(lost, update_gaps)`` tick stamps of a counter-less per-update stream.

    A gap in which ``companion_ns`` (another topic published from the same
    controller-manager update, ``/joint_states``) has messages is a loss on
    this topic; a gap where the companion is silent too is an update that
    never ran, unless it spans more than :data:`MAX_UPDATE_GAP_TICKS` ticks."""
    header_ns = np.asarray(header_ns, dtype=np.int64)
    companion_ns = np.sort(np.asarray(companion_ns, dtype=np.int64))
    dt_ns = nominal_dt_s * 1e9
    lost, gaps = [], []
    for left, right in zip(header_ns[:-1], header_ns[1:]):
        step = right - left
        if step <= 1.5 * dt_ns:
            continue
        count = int(round(step / dt_ns)) - 1
        ticks = [int(left + k * step / (count + 1)) for k in range(1, count + 1)]
        inside = np.searchsorted(companion_ns, right - 0.5 * dt_ns) - np.searchsorted(companion_ns, left + 0.5 * dt_ns)
        (lost if inside > 0 or count > MAX_UPDATE_GAP_TICKS else gaps).extend(ticks)
    return np.asarray(lost, dtype=np.int64), np.asarray(gaps, dtype=np.int64)


def publication_gap_stamps(header_ns: np.ndarray, received_cycle: np.ndarray) -> np.ndarray:
    """iiwa ``/dynamic_joint_states``: hardware reads that never reached the
    bag (``fri/received_cycle`` gaps), stamped at the following sample."""
    header_ns = np.asarray(header_ns, dtype=np.int64)
    received = np.asarray(received_cycle, dtype=float)
    if len(received) < 2 or not np.isfinite(received).all():
        return np.asarray([], dtype=np.int64)
    steps = np.diff(received)
    stamps: list[int] = []
    for index in np.flatnonzero(steps > 1):
        stamps += [int(header_ns[index + 1])] * int(steps[index] - 1)
    return np.asarray(stamps, dtype=np.int64)


#: RR_12 B-2: losses on these invalidate a segment; elsewhere they are reported.
DATA_TOPIC_KINDS = ("dynamic_joint_states", "controller_state")


def isolated_misses(reference_ns: np.ndarray, missing_ns: np.ndarray) -> np.ndarray:
    """RR_13 B-2: which missing per-update messages are isolated singles (no
    other miss within 3 update periods).

    Two loss signatures were measured on the per-update publishers (JSB
    ``/joint_states``, JTC ``controller_state``), whose writers keep only the
    last message: a recorder stall loses a *run* of ~175 consecutive messages
    (6 bags), while every isolated miss was a single message (23 on the iiwa
    and the UR10, rr13_*): the realtime publisher's worker still held the
    previous message (at a controller-manager catch-up on the iiwa, at an
    8 ms period with a late worker on the UR10) and the message was never
    sent. Singles are therefore reported as publisher skips, not recorder
    losses."""
    reference_ns = np.asarray(reference_ns, dtype=np.int64)
    missing_ns = np.sort(np.asarray(missing_ns, dtype=np.int64))
    if len(missing_ns) == 0:
        return np.zeros(0, dtype=bool)
    period = float(np.median(np.diff(reference_ns))) if len(reference_ns) > 1 else 1.0
    gaps = np.diff(missing_ns)
    near_previous = np.concatenate([[False], gaps <= 3 * period])
    near_next = np.concatenate([gaps <= 3 * period, [False]])
    return ~(near_previous | near_next)


#: RR_16 Q-1(c, d): a transport-loss event at ``t_e`` charges the candidate
#: misses in ``[t_e - W, t_e]``. W = the largest detection lag measured on a
#: forced stall + 0.5 s, and at least Fast DDS's heartbeat period (3 s,
#: S-29) + 0.5 s. Measured (RR_17 Q-1(d), domain 88, recorder SIGSTOPped
#: 0.3-8 s under a 1 kHz depth-1 writer): rosbag2 logs the event <= 15 ms
#: after the recorder resumes, so the lag from the first missing message is
#: the stall's length; the longest, 8.0063 s, gives W. A longer stall leaves
#: its earliest misses outside W: they become unlocated, which fails closed.
TRANSPORT_EVENT_WINDOW_S = 8.51


def _nearest_index(reference_ns: np.ndarray, stamps_ns: np.ndarray) -> np.ndarray:
    index = np.clip(np.searchsorted(reference_ns, stamps_ns), 1, len(reference_ns) - 1)
    return np.where(stamps_ns - reference_ns[index - 1] <= reference_ns[index] - stamps_ns, index - 1, index)


def contiguous_runs(stamps_ns: np.ndarray, reference_ns: np.ndarray) -> list[tuple[int, int]]:
    """RR_18 R-2: ``(start, stop)`` slices of the sorted ``stamps_ns`` (missing
    per-update messages, placed on reference update stamps) that form
    contiguous runs: consecutive missing updates with no recorded message
    between. Every update between two candidate misses is either recorded or
    itself a candidate, so a run is a stretch of candidates on consecutive
    update indices (the same index twice is the same run)."""
    stamps_ns = np.asarray(stamps_ns, dtype=np.int64)
    reference_ns = np.asarray(reference_ns, dtype=np.int64)
    if len(stamps_ns) == 0:
        return []
    if len(reference_ns) < 2:
        return [(0, len(stamps_ns))]
    breaks = (np.flatnonzero(np.diff(_nearest_index(reference_ns, stamps_ns)) > 1) + 1).tolist()
    return list(zip([0] + breaks, breaks + [len(stamps_ns)]))


def skip_holes(skip_ns: np.ndarray, reference_ns: np.ndarray) -> list[list[int]]:
    """RR_18 R-3: per contiguous run of publisher skips, ``[first_ns,
    last_ns, n, hole_ns]``: ``hole_ns`` is the gap the run leaves between the
    recorded updates either side of it, in update stamps (a single skip at a
    steady step: 2 periods; inside a catch-up burst: microseconds)."""
    skip_ns = np.sort(np.asarray(skip_ns, dtype=np.int64))
    reference_ns = np.asarray(reference_ns, dtype=np.int64)
    if len(skip_ns) == 0 or len(reference_ns) < 2:
        return []
    index = _nearest_index(reference_ns, skip_ns)
    holes = []
    for start, stop in contiguous_runs(skip_ns, reference_ns):
        before = max(int(index[start]) - 1, 0)
        after = min(int(index[stop - 1]) + 1, len(reference_ns) - 1)
        holes.append([int(skip_ns[start]), int(skip_ns[stop - 1]), int(stop - start),
                      int(reference_ns[after] - reference_ns[before])])
    return holes


def _skip_runs(stamps_ns: np.ndarray, reference_ns: np.ndarray) -> list[dict[str, Any]]:
    """RR_16 Q-1(g): consecutive publisher skips (no more than 3 update
    periods apart) and the update-stamp spacing inside each run: the median
    step of the reference stamps from the update before the run's first
    miss to the update after its last, against the nominal period."""
    stamps_ns = np.sort(np.asarray(stamps_ns, dtype=np.int64))
    reference_ns = np.asarray(reference_ns, dtype=np.int64)
    if len(stamps_ns) == 0 or len(reference_ns) < 2:
        return []
    period = float(np.median(np.diff(reference_ns)))
    breaks = np.flatnonzero(np.diff(stamps_ns) > 3 * period) + 1
    runs = []
    for run in np.split(stamps_ns, breaks):
        lo = max(int(np.searchsorted(reference_ns, run[0])) - 1, 0)
        hi = min(int(np.searchsorted(reference_ns, run[-1], side="right")) + 1, len(reference_ns))
        steps = np.diff(reference_ns[lo:hi])
        runs.append({"length": int(len(run)), "first_ns": int(run[0]), "last_ns": int(run[-1]),
                     "median_update_step_us": float(np.median(steps)) * 1e-3 if len(steps) else None,
                     "nominal_us": period * 1e-3})
    return runs


def _count_fallback(topics: dict[str, Any], djs_header_ns: np.ndarray, total: int | None) -> tuple[int, int | None, bool]:
    """RR_13 D-1 / RR_14 P-4 rules: isolated singles are publisher skips,
    runs of >= 2 are transport losses; a negative balance -- or (Q-1(f)) no
    rosbag2 total at all -- voids the skip exclusion for the whole bag."""
    for topic, entry in topics.items():
        if topic == "/dynamic_joint_states":
            continue
        missing = np.asarray(entry["missing_stamps_ns"], dtype=np.int64)
        skip = isolated_misses(djs_header_ns, missing)
        entry.update(publisher_skips=int(skip.sum()), skip_stamps_ns=missing[skip].tolist(),
                     transport_stamps_ns=missing[~skip].tolist())
    attributed = sum(entry["missing"] - entry["publisher_skips"] for entry in topics.values())
    unattributed = None if total is None else int(total - attributed)
    void = total is None or unattributed < 0
    if void:
        for entry in topics.values():
            entry["transport_stamps_ns"] = list(entry["missing_stamps_ns"])
    return attributed, unattributed, void


def recorder_loss_report(
    *, rosbag2_log_text: str | None, djs_header_ns: np.ndarray, djs_missing_ns: np.ndarray,
    djs_method: str, stamped_topics: Mapping[str, np.ndarray], controller_state_topic: str,
    first_message_ns: Mapping[str, int | None] | None = None,
    first_segment_ns: int | None = None, nominal_period_s: float | None = None,
    event_window_s: float = TRANSPORT_EVENT_WINDOW_S,
) -> dict[str, Any]:
    """Per-topic loss attribution for one bag (RR_12 B-2).

    ``missing`` (candidates) = per-update messages absent from the bag, by
    counting against ``/dynamic_joint_states``. Which of them were lost in
    transport is decided (RR_16 Q-1):

    * ``attribution: "events"``: from rosbag2's own per-topic events. Start-up
      artefacts (RR_18 R-1, :func:`startup_artefact_checks`) are excluded;
      with no other event on a topic every candidate is a publisher skip
      (never sent); otherwise each event at ``t_e`` charges, as transport,
      every contiguous run of candidates (:func:`contiguous_runs`) that
      overlaps ``[t_e - W, t_e]`` -- the whole run, so a stall longer than W
      is charged whole (RR_18 R-2) -- and the rest are skips. A topic whose
      events lost more than its charged candidates gets its event windows as
      ``unlocated_windows_ns`` (they invalidate any overlapping segment).
    * ``attribution: "count_fallback"`` (fail closed) when the log has no
      DEBUG lines, no rosbag2 total, or increments that don't sum to it:
      RR_13 D-1 / RR_14 P-4.

    ``/dynamic_joint_states`` candidates are always losses."""
    first_message_ns = dict(first_message_ns or {})
    topics: dict[str, Any] = {"/dynamic_joint_states": {
        "method": djs_method, "missing": int(len(djs_missing_ns)),
        "missing_stamps_ns": np.asarray(djs_missing_ns, dtype=np.int64).tolist(), "data_topic": True}}
    for topic, stamps in stamped_topics.items():
        missing = missing_stamps(djs_header_ns, stamps)
        topics[topic] = {"method": "per-update count vs /dynamic_joint_states",
                         "missing": int(len(missing)), "missing_stamps_ns": missing.tolist(),
                         "data_topic": topic == controller_state_topic}
    for entry in topics.values():
        entry.update(publisher_skips=0, skip_stamps_ns=[], transport_stamps_ns=list(entry["missing_stamps_ns"]))

    parsed = parse_rosbag2_events(rosbag2_log_text) if rosbag2_log_text is not None else \
        {"debug": False, "subscribed_ns": {}, "events": [], "total": None}
    total = parsed["total"]
    increments = sum(event["increment"] for event in parsed["events"])
    if rosbag2_log_text is None:
        fallback = "no rosbag2 log"
    elif not parsed["debug"]:
        fallback = "no rosbag2_recorder DEBUG lines in the log"
    elif total is None:
        fallback = "no rosbag2 transport-loss total (recorder killed before its summary)"
    elif increments != total or any(event["increment"] <= 0 for event in parsed["events"]):
        fallback = f"per-topic increments ({increments}) don't reconcile with rosbag2's total ({total})"
    else:
        fallback = None

    reference = np.asarray(djs_header_ns, dtype=np.int64)
    if nominal_period_s is None:
        nominal_period_s = float(np.median(np.diff(reference))) * 1e-9 if len(reference) > 1 else None
    report: dict[str, Any] = {"rosbag2_transport_lost": total, "event_window_s": float(event_window_s),
                              "startup_artefact_s": STARTUP_ARTEFACT_S, "first_segment_ns": first_segment_ns,
                              "nominal_period_ns": None if nominal_period_s is None else int(round(nominal_period_s * 1e9))}
    if fallback is not None:
        attributed, unattributed, void = _count_fallback(topics, djs_header_ns, total)
        report.update(attribution="count_fallback", fallback_reason=fallback, attributed_missing=attributed,
                      unattributed=unattributed, skip_exclusion_void=void,
                      rosbag2_events=parsed["events"])
    else:
        checks = startup_artefact_checks(parsed["events"], parsed["subscribed_ns"],
                                         {t: np.asarray(e["missing_stamps_ns"]) for t, e in topics.items()},
                                         first_message_ns=first_message_ns, first_segment_ns=first_segment_ns)
        window_ns = int(round(event_window_s * 1e9))
        per_topic_events: dict[str, list] = {}
        for event, check in zip(parsed["events"], checks):
            first = first_message_ns.get(event["topic"])
            subscribed = parsed["subscribed_ns"].get(event["topic"])
            per_topic_events.setdefault(event["topic"], []).append(
                {**event, "startup_artefact": all(check.values()), "artefact_checks": check,
                 "after_first_message_us": None if first is None else (event["t_ns"] - first) * 1e-3,
                 "after_subscribed_us": None if subscribed is None else (event["t_ns"] - subscribed) * 1e-3})
        for topic, events in per_topic_events.items():
            entry = topics.setdefault(topic, {"method": "rosbag2 events only (no per-update count)", "missing": 0,
                                              "missing_stamps_ns": [], "publisher_skips": 0, "skip_stamps_ns": [],
                                              "transport_stamps_ns": [], "data_topic": False})
            entry["rosbag2_events"] = events
        artefacts = transport = 0
        for topic, entry in topics.items():
            events = entry.get("rosbag2_events", [])
            real = [e for e in events if not e["startup_artefact"]]
            entry["startup_artefacts"] = sum(e["increment"] for e in events if e["startup_artefact"])
            entry["transport_lost"] = sum(e["increment"] for e in real)
            artefacts += entry["startup_artefacts"]
            transport += entry["transport_lost"]
            candidates = np.sort(np.asarray(entry["missing_stamps_ns"], dtype=np.int64))
            if topic == "/dynamic_joint_states":
                charged = np.ones(len(candidates), dtype=bool)  # B-2: always losses
                in_window = [(candidates >= e["t_ns"] - window_ns) & (candidates <= e["t_ns"]) for e in real]
                # an event with no counted miss in its window is a loss the count can't place
                unlocated = sum(e["increment"] for e, hit in zip(real, in_window) if not hit.any())
            else:
                charged = np.zeros(len(candidates), dtype=bool)
                for start, stop in contiguous_runs(candidates, reference):  # RR_18 R-2: whole runs
                    first_ns, last_ns = candidates[start], candidates[stop - 1]
                    if any(first_ns <= e["t_ns"] and last_ns >= e["t_ns"] - window_ns for e in real):
                        charged[start:stop] = True
                unlocated = entry["transport_lost"] - int(charged.sum())
            entry.update(transport_stamps_ns=candidates[charged].tolist(), skip_stamps_ns=candidates[~charged].tolist(),
                         publisher_skips=int((~charged).sum()))
            entry["unlocated_windows_ns"] = [[e["t_ns"] - window_ns, e["t_ns"], e["increment"]] for e in real] \
                if unlocated > 0 else []
        attributed = sum(len(entry["transport_stamps_ns"]) for entry in topics.values())
        report.update(attribution="events", startup_artefacts=artefacts, transport_events=transport,
                      attributed_missing=attributed, unattributed=int(total - artefacts - transport),
                      skip_exclusion_void=False)
    report["publisher_skips"] = sum(entry["publisher_skips"] for entry in topics.values())
    for entry in topics.values():  # RR_18 R-3: the hole each run of skips leaves
        entry["skip_holes_ns"] = skip_holes(np.asarray(entry["skip_stamps_ns"]), reference)
    report["publisher_skip_runs"] = {topic: _skip_runs(np.asarray(entry["skip_stamps_ns"]), reference)
                                     for topic, entry in topics.items() if entry["skip_stamps_ns"]}
    report["topics"] = topics
    report["note"] = ("missing = per-update messages absent from the bag (count vs /dynamic_joint_states); "
                      "transport_stamps_ns = those lost in transport (events: charged by a rosbag2 event within "
                      "event_window_s, whole contiguous runs (RR_18 R-2); count_fallback: runs of >= 2, or all "
                      "when skip_exclusion_void); skip_stamps_ns = publisher skips (never sent); skip_holes_ns = "
                      "[first, last, n, gap between the recorded updates around the run] per run of skips; "
                      "startup_artefacts = a topic's first event, increment 1, at its first message or "
                      "subscription and before the first segment (RR_18 R-1), excluded")
    return report


def losses_in_window(report: Mapping[str, Any] | None, lo_ns: int, hi_ns: int, *,
                     key: str = "missing_stamps_ns") -> dict[str, int]:
    """Missing samples per topic with a stamp inside ``[lo_ns, hi_ns]``
    (``key="transport_stamps_ns"``: publisher skips excluded)."""
    if not report:
        return {}
    result = {}
    for topic, entry in report.get("topics", {}).items():
        stamps = np.asarray(entry.get(key, entry.get("missing_stamps_ns", [])), dtype=np.int64)
        result[topic] = int(np.sum((stamps >= lo_ns) & (stamps <= hi_ns)))
    return result


#: RR_14 P-4b: more isolated ``controller_state`` singles than this inside
#: one segment invalidate it (they all count as losses). Since RR_18 R-3,
#: ``count_fallback`` only.
MAX_SKIPS_PER_SEGMENT = 3
#: RR_18 R-3, ``attribution: "events"``: a segment's publisher skips
#: invalidate it iff they exceed this fraction of its expected updates, and
#: more than one (RR_19 D-12: a 96-update UR10 sweep would otherwise fail on
#: one single skip, 1.04 %; literally applied, R-3 failed 6 sweeps and 3 of 4
#: rr17 URSim identify stages; one skip's hole is judged by the gap rule) ...
MAX_SKIP_FRACTION = 0.01
#: ... or any run of them leaves a gap between recorded updates longer than
#: this many nominal periods (iiwa 3 ms, UR10 24 ms). Judged against
#: ``MAX_SKIP_GAP_PERIODS + 0.5`` periods so that update-stamp jitter can't
#: decide it: a 2-run at the steady step leaves exactly 3 periods (valid),
#: a 3-run 4 periods (invalid).
MAX_SKIP_GAP_PERIODS = 3


def segment_skip_verdict(report: Mapping[str, Any] | None, lo_ns: int, hi_ns: int) -> dict[str, Any]:
    """Do the data-topic publisher skips inside ``[lo_ns, hi_ns]`` invalidate
    the segment?

    * ``attribution: "events"`` with a nominal period (RR_18 R-3): iff skips
      > max(1, :data:`MAX_SKIP_FRACTION` of the segment's expected updates),
      or a run of them leaves a gap > :data:`MAX_SKIP_GAP_PERIODS` nominal
      periods;
    * otherwise (``count_fallback``, reports before R-3): iff skips >
      :data:`MAX_SKIPS_PER_SEGMENT`.

    ``largest_gap_ms`` is the largest gap a run of skips leaves between
    recorded updates in the segment (what the JTC tracking fit bridges)."""
    skips = segment_skips(report, lo_ns, hi_ns)
    holes = [hole for entry in (report or {}).get("topics", {}).values() if entry.get("data_topic")
             for hole in entry.get("skip_holes_ns", []) if hole[0] <= hi_ns and hole[1] >= lo_ns]
    largest = max((hole[3] for hole in holes), default=0)
    period = (report or {}).get("nominal_period_ns")
    verdict: dict[str, Any] = {"skips": skips, "largest_gap_ms": largest * 1e-6}
    if (report or {}).get("attribution") == "events" and period:
        expected = int(round((hi_ns - lo_ns) / period)) + 1
        fraction = skips / expected if expected > 0 else 0.0
        gap_periods = largest / period
        verdict.update(rule=f"events: skips > max(1, {MAX_SKIP_FRACTION:.0%} of expected updates) or a gap > "
                            f"{MAX_SKIP_GAP_PERIODS} nominal periods (RR_18 R-3, RR_19 D-12)",
                       expected_updates=expected, skip_fraction=fraction, largest_gap_periods=gap_periods,
                       invalid=bool((skips > 1 and fraction > MAX_SKIP_FRACTION)
                                    or gap_periods > MAX_SKIP_GAP_PERIODS + 0.5))
    else:
        verdict.update(rule=f"count: skips > {MAX_SKIPS_PER_SEGMENT}", invalid=skips > MAX_SKIPS_PER_SEGMENT,
                       largest_gap_periods=largest / period if period else None)
    return verdict


def data_topic_losses(report: Mapping[str, Any] | None, lo_ns: int, hi_ns: int) -> int:
    """RR_12 B-2's rule: losses on ``/dynamic_joint_states`` or the JTC
    ``controller_state`` inside a segment invalidate it.

    ``attribution: "events"`` (RR_16 Q-1(e)): transport losses on a data
    topic (charged candidates, plus any unlocated event window that overlaps
    the segment) count; publisher skips on the data topics, singles and runs
    alike, count only when :func:`segment_skip_verdict` says so (RR_18 R-3:
    > 1 % of the expected updates or a gap > 3 nominal periods).

    ``count_fallback`` (and reports written before Q-1): RR_13 D-1 with RR_14
    P-4: isolated singles are skips unless the bag's skip exclusion is void
    (negative balance or no rosbag2 total); more than
    :data:`MAX_SKIPS_PER_SEGMENT` of them count. ``/dynamic_joint_states``
    losses always count."""
    if not report:
        return 0
    if report.get("attribution") == "events":
        counts = losses_in_window(report, lo_ns, hi_ns, key="transport_stamps_ns")
        losses = int(sum(count for topic, count in counts.items() if report["topics"][topic].get("data_topic")))
        for entry in report["topics"].values():
            if entry.get("data_topic"):
                losses += sum(int(n) for lo, hi, n in entry.get("unlocated_windows_ns", []) if lo <= hi_ns and hi >= lo_ns)
        verdict = segment_skip_verdict(report, lo_ns, hi_ns)
        return losses + (max(verdict["skips"], 1) if verdict["invalid"] else 0)
    unattributed = report.get("unattributed")
    void = report.get("skip_exclusion_void", unattributed is not None and unattributed < 0)
    counts = losses_in_window(report, lo_ns, hi_ns, key="missing_stamps_ns" if void else "transport_stamps_ns")
    losses = int(sum(count for topic, count in counts.items() if report["topics"][topic].get("data_topic")))
    if void:
        return losses
    skips = segment_skips(report, lo_ns, hi_ns)
    return losses + (skips if skips > MAX_SKIPS_PER_SEGMENT else 0)


def segment_skips(report: Mapping[str, Any] | None, lo_ns: int, hi_ns: int) -> int:
    """Isolated singles on the data topics inside ``[lo_ns, hi_ns]``."""
    if not report:
        return 0
    total = 0
    for entry in report.get("topics", {}).values():
        if not entry.get("data_topic"):
            continue
        if "skip_stamps_ns" in entry:
            stamps = np.asarray(entry["skip_stamps_ns"], dtype=np.int64)
        else:  # a report written before P-4: skips = missing minus transport
            stamps = np.setdiff1d(np.asarray(entry.get("missing_stamps_ns", []), dtype=np.int64),
                                  np.asarray(entry.get("transport_stamps_ns", []), dtype=np.int64))
        total += int(np.sum((stamps >= lo_ns) & (stamps <= hi_ns)))
    return total


def write_recorder_losses(config, bag_dir: Path, log_path: Path, output: Path, *,
                          djs_header: np.ndarray | None = None, djs_frame: pd.DataFrame | None = None) -> dict[str, Any]:
    """ROS side: read the bag's stamped topics and write ``output``."""
    import json
    from control_msgs.msg import DynamicJointState, JointTrajectoryControllerState
    from sensor_msgs.msg import JointState
    from .profile import profile_for

    profile = profile_for(config.robot)
    received_cycle = None
    if djs_header is None or (profile.has_fri_gpio and djs_frame is None):
        djs_header, _, messages = read_header_stamped(bag_dir, "/dynamic_joint_states", DynamicJointState)
        if profile.has_fri_gpio:
            received_cycle = [decode_dynamic_joint_state(m).get("fri", {}).get("received_cycle", float("nan"))
                              for m in messages]
    elif djs_frame is not None and "fri_received_cycle" in djs_frame:
        received_cycle = djs_frame["fri_received_cycle"].to_numpy()
    if profile.has_fri_gpio and received_cycle is not None and config.hardware != "mock":
        djs_missing = publication_gap_stamps(djs_header, received_cycle)
        method = "fri/received_cycle gaps (hardware reads not recorded)"
    else:
        djs_missing = tick_gap_stamps(djs_header, 1.0 / profile.rate_hz)
        method = "header-stamp steps over 1.5 nominal periods"
    controller_topic = f"/{profile.controller_name}/controller_state"
    stamped, first_message = {}, {}
    for topic, msg_type in (("/joint_states", JointState), (controller_topic, JointTrajectoryControllerState)):
        try:
            stamped[topic], received, _ = read_header_stamped(bag_dir, topic, msg_type)
            first_message[topic] = int(received.min()) if len(received) else None
        except Exception:  # noqa: BLE001 -- a topic the bag never recorded is reported as such
            stamped[topic] = np.asarray([], dtype=np.int64)
            first_message[topic] = None
    update_gaps = None
    if config.hardware == "mock" and len(stamped["/joint_states"]):
        djs_missing, update_gaps = split_update_gaps(djs_header, 1.0 / profile.rate_hz, stamped["/joint_states"])
        method += "; on mock, gaps /joint_states shares are controller-manager update gaps, not losses"
    text = Path(log_path).read_text(errors="replace") if Path(log_path).is_file() else None
    if text is not None:  # RR_16 Q-1(b): the first receive stamp of every other topic with a loss event
        others = {e["topic"] for e in parse_rosbag2_events(text)["events"]} - set(first_message)
        first_message.update(first_message_stamps(bag_dir, sorted(others)))
    try:  # RR_18 R-1(iv): an artefact is logged before the bag's first segment starts
        starts = [e["stamp_ns"] for e in read_events(bag_dir) if e["kind"].endswith("_start")]
    except Exception:  # noqa: BLE001 -- a bag without /erd/events (link test) has no segment
        starts = []
    report = recorder_loss_report(rosbag2_log_text=text, djs_header_ns=djs_header, djs_missing_ns=djs_missing,
                                  djs_method=method, stamped_topics=stamped, controller_state_topic=controller_topic,
                                  first_message_ns=first_message, first_segment_ns=min(starts) if starts else None,
                                  nominal_period_s=1.0 / profile.rate_hz)
    if update_gaps is not None:
        report["update_gaps"] = {"ticks": int(len(update_gaps)), "stamps_ns": update_gaps.tolist(),
                                 "max_ticks_per_gap": MAX_UPDATE_GAP_TICKS}
    report["bag"] = Path(bag_dir).name
    output.write_text(json.dumps(report, indent=1))
    return report


def iiwa_cycle_health(frame: pd.DataFrame) -> dict[str, Any]:
    """Distinguish remote packet loss from losses after the hardware read."""
    cycles = frame['fri_cycle'].to_numpy(dtype=float)
    remote = np.diff(cycles)
    # FRI sequence is uint32. Its wrap is not a lost or reordered cycle.
    remote[remote < -(2**31)] += 2**32
    finite = bool(np.isfinite(remote).all())
    lost = int(np.maximum(remote - 1, 0).sum()) if finite else -1
    result = {'ok': finite and bool(np.all(remote == 1)), 'lost_cycles': lost,
              'duplicates': int(np.sum(remote == 0)), 'reordered': int(np.sum(remote < 0))}
    if 'fri_received_cycle' in frame and frame['fri_received_cycle'].notna().all():
        received = np.diff(frame['fri_received_cycle'].to_numpy(dtype=float))
        result['lost_fri_cycles'] = int(np.maximum(remote - received, 0).sum())
        result['lost_publication_cycles'] = int(np.maximum(received - 1, 0).sum())
    return result
