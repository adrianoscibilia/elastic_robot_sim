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


def recorder_loss_report(
    *, rosbag2_log_text: str | None, djs_header_ns: np.ndarray, djs_missing_ns: np.ndarray,
    djs_method: str, stamped_topics: Mapping[str, np.ndarray], controller_state_topic: str,
) -> dict[str, Any]:
    """Per-topic loss attribution for one bag (RR_12 B-2), from the bag's own
    content, reconciled against rosbag2's transport-layer total."""
    rosbag2 = parse_rosbag2_log(rosbag2_log_text) if rosbag2_log_text is not None else {"total": None, "per_topic": {}}
    topics: dict[str, Any] = {"/dynamic_joint_states": {
        "method": djs_method, "missing": int(len(djs_missing_ns)), "missing_stamps_ns": np.asarray(djs_missing_ns).tolist(),
        "data_topic": True}}
    for topic, stamps in stamped_topics.items():
        missing = missing_stamps(djs_header_ns, stamps)
        skip = isolated_misses(djs_header_ns, missing)
        topics[topic] = {"method": "per-update count vs /dynamic_joint_states",
                         "missing": int(len(missing)), "missing_stamps_ns": missing.tolist(),
                         "publisher_skips": int(skip.sum()), "skip_stamps_ns": missing[skip].tolist(),
                         "transport_stamps_ns": missing[~skip].tolist(),
                         "data_topic": topic == controller_state_topic}
    topics["/dynamic_joint_states"]["transport_stamps_ns"] = topics["/dynamic_joint_states"]["missing_stamps_ns"]
    topics["/dynamic_joint_states"]["publisher_skips"] = 0
    topics["/dynamic_joint_states"]["skip_stamps_ns"] = []
    attributed = sum(entry["missing"] - entry["publisher_skips"] for entry in topics.values())
    total = rosbag2["total"]
    unattributed = None if total is None else int(total - attributed)
    # RR_14 P-4a: a negative balance means the bag's own accounting doesn't
    # reconcile with rosbag2's; the skip exclusion is then void for the whole
    # bag and every single miss counts as a loss.
    void = unattributed is not None and unattributed < 0
    if void:
        for entry in topics.values():
            entry["transport_stamps_ns"] = list(entry["missing_stamps_ns"])
    return {
        "rosbag2_transport_lost": total, "rosbag2_per_topic": rosbag2["per_topic"],
        "lost_message_topic": "not published by rosbag2 0.26 (Jazzy); only events/write_split exists",
        "topics": topics, "attributed_missing": attributed,
        "unattributed": unattributed,
        "skip_exclusion_void": void,
        "note": "missing = per-update messages absent from the bag; publisher_skips = isolated single misses "
                "(never sent); attributed_missing excludes them; unattributed = rosbag2's total minus that (topics "
                "without a per-update stamp: rosout, diagnostics, tf_static, events, action; negative when some "
                "'skips' were in fact single transport losses)",
    }


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
#: one segment invalidate it (they all count as losses).
MAX_SKIPS_PER_SEGMENT = 3


def data_topic_losses(report: Mapping[str, Any] | None, lo_ns: int, hi_ns: int) -> int:
    """RR_12 B-2's rule: losses on ``/dynamic_joint_states`` or the JTC
    ``controller_state`` inside a segment invalidate it.

    D-1 (RR_13, accepted in RR_14 with two safeguards): an isolated single
    ``controller_state`` miss is a realtime-publisher skip (see
    :func:`isolated_misses`): reported, not counted, unless (P-4a) the bag's
    ``unattributed`` is negative (every single then counts, see
    :func:`recorder_loss_report`) or (P-4b) the segment holds more than
    :data:`MAX_SKIPS_PER_SEGMENT` of them. ``/dynamic_joint_states`` losses
    always count."""
    if not report:
        return 0
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
    stamped = {}
    for topic, msg_type in (("/joint_states", JointState), (controller_topic, JointTrajectoryControllerState)):
        try:
            stamped[topic], _, _ = read_header_stamped(bag_dir, topic, msg_type)
        except Exception:  # noqa: BLE001 -- a topic the bag never recorded is reported as such
            stamped[topic] = np.asarray([], dtype=np.int64)
    update_gaps = None
    if config.hardware == "mock" and len(stamped["/joint_states"]):
        djs_missing, update_gaps = split_update_gaps(djs_header, 1.0 / profile.rate_hz, stamped["/joint_states"])
        method += "; on mock, gaps /joint_states shares are controller-manager update gaps, not losses"
    text = Path(log_path).read_text(errors="replace") if Path(log_path).is_file() else None
    report = recorder_loss_report(rosbag2_log_text=text, djs_header_ns=djs_header, djs_missing_ns=djs_missing,
                                  djs_method=method, stamped_topics=stamped, controller_state_topic=controller_topic)
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
