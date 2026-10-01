"""Minimal MCAP bag reading (RR_01 S3.6, T1.6's "commanded reference in the
bag = plan to 1e-9 rad" check) and raw-telemetry extraction (RR_01 S8.1,
RR_04 A-8: "bagio builds raw.parquet from the bag").

The pure pieces (message-by-name decoding, segment-start alignment, gap
counting) are plain numpy/dict functions, unit-tested without a bag or a ROS
environment; only :func:`read_bag_topic`/:func:`bag_topics` and the
``build_*_raw_frame`` wrappers that call them need ``rosbag2_py`` sourced.
"""

from __future__ import annotations

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
                  "drive_state", "cycle")


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


def prepare_iiwa_conversion(config, run_dir: Path) -> None:
    """Read ROS messages here, before launching the ROS-free numerical stage."""
    import json
    from control_msgs.msg import DynamicJointState
    from erd_msgs.msg import RunEvent
    from .profile import profile_for

    profile = profile_for(config.robot)
    raw = build_iiwa_raw_frame(
        run_dir / "bag", driver_joint_order=profile.driver_joint_order,
        driver_to_sim={v: k for k, v in config.description.sim_to_driver().items()},
        joint_order=config.joint_order,
    )
    stamps = np.asarray([stamp for stamp, _ in read_bag_topic(run_dir / "bag", "/dynamic_joint_states", DynamicJointState)])
    if config.hardware == "mock":
        # GenericSystem supplies no FRI clock. Preserve the actual bag clock;
        # uniform resampling is explicitly synthetic and never used on FRI.
        raw["t"] = (stamps - stamps[0]) * 1e-9
    raw["bag_stamp_ns"] = stamps
    raw.to_parquet(run_dir / "raw.parquet", index=False)
    events = []
    for stamp, event in read_bag_topic(run_dir / "bag", "/erd/events", RunEvent):
        events.append({"stamp_ns": stamp, "segment_id": event.segment_id, "kind": event.kind,
                       "plan_digest": event.plan_digest})
    (run_dir / "events.json").write_text(json.dumps(events, indent=2))
