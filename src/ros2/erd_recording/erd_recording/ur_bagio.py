"""UR RTDE/driver exact-position alignment and explicit per-cycle gap classes."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from .bagio import decode_dynamic_joint_state, read_bag_topic
from .profile import profile_for


def align_rtde(sidecar: pd.DataFrame, driver_q: np.ndarray, driver_stamps_ns: np.ndarray, *, dt=0.008):
    q = sidecar[[f'actual_q{i}' for i in range(6)]].to_numpy()
    times = sidecar['timestamp'].to_numpy()
    lookup = {}
    for i, row in enumerate(driver_q):
        lookup.setdefault(tuple(row), []).append(i)
    anchors = [(i, lookup[tuple(row)][0]) for i, row in enumerate(q)
               if len(lookup.get(tuple(row), [])) == 1]
    if len(anchors) < 10:
        raise ValueError('RTDE alignment requires at least 10 unique moving actual_q anchors')
    ri, di = np.asarray(anchors).T
    if np.any(np.diff(driver_stamps_ns) <= 0) or np.any(np.diff(di) <= 0):
        raise ValueError('RTDE/driver clock uncertainty: nonmonotonic exact anchors')
    # URSim's controller time can run slower than wall time. A fixed offset
    # (or even one affine fit) is insufficient at individual ROS cycles.
    # Interpolate between exact moving-position anchors on the two clocks.
    anchor_t = times[ri]
    anchor_ns = driver_stamps_ns[di]
    slope = float(np.polyfit(anchor_t-anchor_t[0], (anchor_ns-anchor_ns[0])*1e-9, 1)[0])
    aligned_ns = np.rint(np.interp(times, anchor_t, anchor_ns)).astype(np.int64)
    before, after = times < anchor_t[0], times > anchor_t[-1]
    aligned_ns[before] = anchor_ns[0] + np.rint((times[before]-anchor_t[0])*slope*1e9).astype(np.int64)
    aligned_ns[after] = anchor_ns[-1] + np.rint((times[after]-anchor_t[-1])*slope*1e9).astype(np.int64)
    matched = np.full(len(q), -1, dtype=int)
    for i, (row, stamp) in enumerate(zip(q, aligned_ns)):
        candidates = lookup.get(tuple(row), [])
        if candidates:
            j = min(candidates, key=lambda j: abs(driver_stamps_ns[j]-stamp))
            if abs(driver_stamps_ns[j]-stamp) < dt * slope * 1e9:
                matched[i] = j
    gaps = []
    for i, delta in enumerate(np.diff(times)):
        if abs(delta - dt) <= 0.0005:
            continue
        count = max(0, int(round(delta / dt)) - 1)
        if count == 0:
            gaps.append({'after_timestamp': float(times[i]), 'timestamp': float(times[i+1]),
                         'class': 'duplicate_or_irregular', 'missing_cycles': 0})
            continue
        left, right = matched[i], matched[i+1]
        distinct = []
        if 0 <= left < right and not np.array_equal(q[i], q[i+1]):
            for row in driver_q[left+1:right]:
                key = tuple(row)
                if key not in (tuple(q[i]), tuple(q[i+1])) and key not in distinct:
                    distinct.append(key)
            reader = len(distinct)
            classification = 'controller_skipped' if reader == 0 else ('sidecar_only' if reader == count else 'unclassified')
        else:
            # At standstill identical positions cannot prove which robot
            # cycles the driver's fixed-rate publisher reused. Do not label
            # an ambiguous gap a controller skip just to pass L2.
            classification = 'unclassified'
        for k in range(1, count + 1):
            gaps.append({'after_timestamp': float(times[i]), 'timestamp': float(times[i] + k * dt),
                         'class': classification, 'missing_cycles': 1,
                         'driver_left_row': int(left), 'driver_right_row': int(right)})
    matches = int(np.sum(matched >= 0))
    affine_residual = (anchor_ns-anchor_ns[0])*1e-9 - slope*(anchor_t-anchor_t[0])
    report = {'sidecar_rows': len(q), 'exact_matched_rows': matches, 'exact_match_fraction': matches / len(q),
              'unique_anchors': len(anchors), 'clock_mapping': 'piecewise exact actual_q anchors',
              'wall_seconds_per_robot_second': slope,
              'affine_residual_p99_s': float(np.percentile(np.abs(affine_residual),99)),
              'max_anchor_interval_s': float(np.max(np.diff(anchor_t))),
              'gaps': gaps, 'reader_missing_cycles': sum(g['class'] == 'sidecar_only' for g in gaps),
              'unclassified_cycles': sum(g['class'] not in ('sidecar_only','controller_skipped') for g in gaps),
              'controller_missing_cycles': sum(g['class'] == 'controller_skipped' for g in gaps)}
    return aligned_ns, report


def prepare_ur_conversion(config, run_dir: Path):
    from control_msgs.msg import DynamicJointState
    from erd_msgs.msg import RunEvent
    profile = profile_for(config.robot)
    sidecar = pd.read_parquet(run_dir / 'rtde.parquet')
    q, stamps = [], []
    for _, msg in read_bag_topic(run_dir / 'bag', '/dynamic_joint_states', DynamicJointState):
        decoded = decode_dynamic_joint_state(msg)
        q.append([decoded[j]['position'] for j in profile.driver_joint_order])
        stamps.append(msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec)
    aligned, report = align_rtde(sidecar, np.asarray(q), np.asarray(stamps, dtype=np.int64))
    raw = sidecar.copy()
    raw['t'] = raw['timestamp']
    raw['bag_stamp_ns'] = aligned
    for i, sim in enumerate(config.joint_order):
        driver = config.description.sim_to_driver()[sim]
        j = profile.driver_joint_order.index(driver)
        for prefix, field in [('q','actual_q'),('dq','actual_qd'),('i_act','actual_current'),
                              ('commanded_position','target_q')]:
            raw[f'{prefix}{i}'] = sidecar[f'{field}{j}']
    raw.to_parquet(run_dir / 'raw.parquet', index=False)
    events = [{'stamp_ns': e.header.stamp.sec * 10**9 + e.header.stamp.nanosec,
               'segment_id': e.segment_id, 'kind': e.kind, 'plan_digest': e.plan_digest}
              for _, e in read_bag_topic(run_dir / 'bag', '/erd/events', RunEvent)]
    (run_dir / 'events.json').write_text(json.dumps(events, indent=2))
    (run_dir / 'rtde_alignment.json').write_text(json.dumps(report, indent=2))
