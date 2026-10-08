"""Link and monitor measurement tools (RR_12 A-2a; RR_08 S4 rows 4 and 7).

``erd_link_test --config <lab.yaml> <seconds>`` records the robot's state
stream for ``<seconds>`` with no motion command (rung 1) and writes
``link_test.json`` into a normal run folder:

* iiwa: lost FRI cycles (``fri/cycle``), lost publication cycles
  (``fri/received_cycle``), the stamp-step histogram, Sunrise connection
  quality and session state, and the expected valid fraction of 12 s windows;
* UR10: driver and sidecar rates, sidecar gaps, the fraction of sidecar
  ``actual_q`` rows found exactly in the driver's stream;
* both: the recorder's per-topic losses (RR_12 B-2).

``erd_monitor_test --config <lab.yaml> [--injections 20]`` (L1/L2 only) is the
row-7 harness: a fast joint step with the monitor threshold lowered to 5 mrad
makes the live monitor cancel, then a slow return; it repeats, then reports
the cancel latency of every injection from the bag and the monitor's sample
ages (``monitor.json``).
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

#: RR_12 S4.7: stamp-step histogram edges (ms).
STEP_EDGES_MS = (0.0, 0.5, 0.9, 1.1, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, math.inf)


def stamp_step_histogram(stamps_ns: np.ndarray) -> dict[str, Any]:
    steps = np.diff(np.asarray(stamps_ns, dtype=np.int64)) * 1e-6
    if not len(steps):
        return {"n": 0}
    counts, _ = np.histogram(steps, bins=np.asarray(STEP_EDGES_MS))
    labels = [f"{lo:g}-{hi:g}" for lo, hi in zip(STEP_EDGES_MS[:-1], STEP_EDGES_MS[1:])]
    return {"n": int(len(steps)), "p50_ms": float(np.percentile(steps, 50)), "p99_ms": float(np.percentile(steps, 99)),
            "max_ms": float(steps.max()), "histogram_ms": dict(zip(labels, counts.tolist()))}


def iiwa_link_stats(frame: pd.DataFrame, *, rate_hz: float = 1000.0) -> dict[str, Any]:
    """``frame``: ``stamp_ns``, ``cycle``, ``received_cycle``,
    ``connection_quality``, ``session_state`` per ``/dynamic_joint_states``."""
    cycles = frame["cycle"].to_numpy(dtype=float)
    received = frame["received_cycle"].to_numpy(dtype=float)
    remote = np.diff(cycles)
    remote[remote < -(2**31)] += 2**32  # FRI's sequence counter is uint32
    local = np.diff(received)
    lost_fri = int(np.maximum(remote - local, 0).sum())
    lost_publication = int(np.maximum(local - 1, 0).sum())
    duration_s = float((frame["stamp_ns"].iloc[-1] - frame["stamp_ns"].iloc[0]) * 1e-9)
    loss_rate = lost_fri / duration_s if duration_s > 0 else float("nan")
    return {
        "samples": int(len(frame)), "duration_s": duration_s, "cycle_span": int(cycles[-1] - cycles[0] + 1),
        "lost_fri_cycles": lost_fri, "lost_publication_cycles": lost_publication,
        "duplicate_cycles": int(np.sum(remote == 0)), "reordered_cycles": int(np.sum(remote < 0)),
        "fri_loss_rate_per_s": loss_rate,
        # RR_12 S4.7: expected fraction of 12 s windows with no lost cycle.
        "expected_valid_12s_window_fraction": float(math.exp(-12.0 * loss_rate)) if np.isfinite(loss_rate) else None,
        "rate_from_cycles_hz": float((cycles[-1] - cycles[0]) / duration_s) if duration_s > 0 else None,
        "stamp_steps": stamp_step_histogram(frame["stamp_ns"].to_numpy()),
        "connection_quality": {str(int(k)): int(v) for k, v in frame["connection_quality"].value_counts().items()},
        "session_state": {str(int(k)): int(v) for k, v in frame["session_state"].value_counts().items()},
        "pass_rung1": bool(lost_fri <= 5 * duration_s / 300.0 and
                           np.percentile(np.diff(frame["stamp_ns"].to_numpy()) * 1e-6, 99) <= 2.0),
        "rung1_rule": "<= 5 lost FRI cycles per 300 s and stamp-step p99 <= 2 ms (RR_12 S4.7)",
    }


def ur_link_stats(sidecar: pd.DataFrame, driver_q: np.ndarray, driver_stamps_ns: np.ndarray,
                  *, rtde_period_s: float = 0.008) -> dict[str, Any]:
    times = sidecar["timestamp"].to_numpy(dtype=float)
    steps = np.diff(times)
    gaps = int(np.sum(np.abs(steps - rtde_period_s) > 0.0005))
    missing = int(np.sum(np.maximum(np.rint(steps / rtde_period_s) - 1, 0)))
    q = sidecar[[f"actual_q{i}" for i in range(driver_q.shape[1])]].to_numpy()
    driver_rows = {tuple(row) for row in driver_q}
    exact = float(np.mean([tuple(row) in driver_rows for row in q])) if len(q) else 0.0
    driver_duration = (driver_stamps_ns[-1] - driver_stamps_ns[0]) * 1e-9 if len(driver_stamps_ns) > 1 else 0.0
    return {
        "sidecar_rows": int(len(sidecar)), "sidecar_rate_hz": float((len(times) - 1) / (times[-1] - times[0]))
        if len(times) > 1 else None,
        "driver_rate_hz": float((len(driver_stamps_ns) - 1) / driver_duration) if driver_duration > 0 else None,
        "sidecar_gaps": gaps, "sidecar_missing_cycles": missing,
        "sidecar_max_step_ms": float(steps.max() * 1e3) if len(steps) else None,
        "exact_actual_q_match_fraction": exact,
        "match_rule": "fraction of sidecar actual_q rows present bit-exactly in the driver's /dynamic_joint_states",
    }


def _bag_iiwa_frame(bag_dir: Path) -> pd.DataFrame:
    from control_msgs.msg import DynamicJointState
    from .bagio import decode_dynamic_joint_state, read_header_stamped

    stamps, _, messages = read_header_stamped(bag_dir, "/dynamic_joint_states", DynamicJointState)
    fri = [decode_dynamic_joint_state(m).get("fri", {}) for m in messages]
    return pd.DataFrame({"stamp_ns": stamps, **{key: [f.get(key, np.nan) for f in fri] for key in
                                                ("cycle", "received_cycle", "connection_quality", "session_state")}})


def _spin_for(node: Any, seconds: float, stop: dict[str, bool]) -> None:
    import rclpy

    deadline = time.time() + seconds
    while time.time() < deadline and not stop["requested"]:
        rclpy.spin_once(node, timeout_sec=0.1)


def _installed_code_digest() -> dict:
    """RR_16 Q-4(i), RR_18 R-4: the run's code digest; a missing or stale
    build stamp, or a stale installed module, refuses."""
    from .code_digest import code_digest, refusal

    digest = code_digest()
    if refusal(digest):
        raise SystemExit(refusal(digest))
    return digest


def link_test_main(argv: list[str] | None = None) -> int:
    from .env_guard import assert_environment

    assert_environment(require_ros=True, require_pinocchio=False)
    import rclpy
    from rclpy.signals import SignalHandlerOptions

    from .bagio import write_recorder_losses
    from .config import check_environment_domain, load_lab_config
    from .pipeline import RecordingNode, make_run_dir, _write_manifest_fields
    from .profile import profile_for

    parser = argparse.ArgumentParser(prog="erd_link_test")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("duration_s", type=float)
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    config = load_lab_config(args.config)
    check_environment_domain(config)
    profile = profile_for(config.robot)
    digest = _installed_code_digest()
    paths = make_run_dir(config, run_id=args.run_id or time.strftime("link_test_%H%M%S"))
    _write_manifest_fields(paths, stage="link_test", hardware=config.hardware, code_digest=digest, duration_s=args.duration_s,
                           safety_vendor_checksum=config.safety.vendor_checksum)
    stop = {"requested": False}
    previous = signal.signal(signal.SIGINT, lambda *_: stop.update(requested=True))
    rclpy.init(args=None, signal_handler_options=SignalHandlerOptions.NO)
    node = RecordingNode(profile, config)
    try:
        sidecar = node.launch_sidecar(paths.rtde_parquet, robot_ip=config.connection.robot_ip)
        node.start_bag(paths.bag_dir, sidecar=sidecar)
        if sidecar:
            node.wait_for_sidecar_rows()
        _spin_for(node, args.duration_s, stop)
    finally:
        node.stop_bag()
        node.stop_sidecar()
        node.destroy_node()
        rclpy.shutdown()
        signal.signal(signal.SIGINT, previous)

    result: dict[str, Any] = {"robot": config.robot, "hardware": config.hardware, "requested_s": args.duration_s,
                              "interrupted": stop["requested"]}
    if profile.has_fri_gpio:
        frame = _bag_iiwa_frame(paths.bag_dir)
        result["iiwa"] = iiwa_link_stats(frame, rate_hz=config.rate_hz) if config.hardware != "mock" else {
            "samples": int(len(frame)), "stamp_steps": stamp_step_histogram(frame["stamp_ns"].to_numpy()),
            "note": "mock has no FRI counters"}
    else:
        from control_msgs.msg import DynamicJointState
        from .bagio import decode_dynamic_joint_state, read_header_stamped

        stamps, _, messages = read_header_stamped(paths.bag_dir, "/dynamic_joint_states", DynamicJointState)
        driver_q = np.asarray([[decode_dynamic_joint_state(m)[j]["position"] for j in profile.driver_joint_order]
                               for m in messages])
        if paths.rtde_parquet.exists():
            result["ur"] = ur_link_stats(pd.read_parquet(paths.rtde_parquet), driver_q, stamps)
        else:
            result["ur"] = {"note": "no sidecar output (mock has no RTDE server)",
                            "driver_rate_hz": float((len(stamps) - 1) / ((stamps[-1] - stamps[0]) * 1e-9))}
    losses = write_recorder_losses(config, paths.bag_dir, paths.raw_log_dir / "bag.rosbag2.stderr.log",
                                   paths.root / "recorder_losses.json")
    result["recorder_losses"] = {"rosbag2_transport_lost": losses["rosbag2_transport_lost"],
                                 "attribution": losses["attribution"],
                                 "startup_artefacts": losses.get("startup_artefacts"),
                                 "transport_events": losses.get("transport_events"),
                                 "missing_by_topic": {t: e["missing"] for t, e in losses["topics"].items()},
                                 "unattributed": losses["unattributed"]}
    (paths.root / "link_test.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


def _quintic_move(start: np.ndarray, target: np.ndarray, duration: float, dt: float, joint_names):
    """A minimum-jerk (quintic) move from ``start`` to ``target``, analytic pos/vel/acc."""
    from elastic_sim.materialized import MaterializedTrajectory

    t = np.arange(int(round(duration / dt)) + 1) * dt
    s = t / duration
    delta = np.asarray(target, dtype=float) - np.asarray(start, dtype=float)
    q = start + np.outer(10 * s**3 - 15 * s**4 + 6 * s**5, delta)
    dq = np.outer((30 * s**2 - 60 * s**3 + 30 * s**4) / duration, delta)
    ddq = np.outer((60 * s - 180 * s**2 + 120 * s**3) / duration**2, delta)
    return MaterializedTrajectory(time=t, position=q, velocity=dq, acceleration=ddq, joint_names=tuple(joint_names),
                                  metadata={"generator": "erd_monitor_test quintic"})


def _move_to(node: Any, target: np.ndarray, config: Any, duration: float, paths: Any, label: str) -> dict[str, Any]:
    sim_to_driver = config.description.sim_to_driver()
    current = node.measured_positions()
    start = np.asarray([current[sim_to_driver[j]] for j in config.joint_order])
    trajectory = _quintic_move(start, target, duration, 1.0 / config.rate_hz, config.joint_order)
    node.publish_event(paths.root.name, label, "return_start")
    goal = node.build_goal(trajectory, node.profile.driver_joint_order, sim_to_driver)
    result = node.send_goal_and_wait(goal, timeout_s=duration + 10.0, run_paths=paths)
    node.publish_event(paths.root.name, label, "return_end")
    return result


def monitor_test_main(argv: list[str] | None = None) -> int:
    from .env_guard import assert_environment

    assert_environment(require_ros=True, require_pinocchio=False)
    import rclpy
    from rclpy.signals import SignalHandlerOptions

    from .config import check_environment_domain, load_lab_config
    from .pipeline import RecordingNode, make_run_dir, stage_preflight, _write_manifest_fields
    from .profile import profile_for
    from .reporting import cancel_latencies, latency_summary

    parser = argparse.ArgumentParser(prog="erd_monitor_test")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--injections", type=int, default=20)
    parser.add_argument("--joint", type=int, default=0, help="sim-order joint index to step")
    parser.add_argument("--step-rad", type=float, default=0.15)
    parser.add_argument("--step-s", type=float, default=0.3)
    parser.add_argument("--threshold-rad", type=float, default=0.005)
    parser.add_argument("--return-s", type=float, default=4.0)
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    config = load_lab_config(args.config)
    if config.hardware == "real":
        print("erd_monitor_test is L1/L2 only: it commands deliberately aggressive steps", file=sys.stderr)
        return 2
    check_environment_domain(config)
    profile = profile_for(config.robot)
    digest = _installed_code_digest()
    paths = make_run_dir(config, run_id=args.run_id or time.strftime("monitor_test_%H%M%S"))
    _write_manifest_fields(paths, stage="monitor_test", hardware=config.hardware, code_digest=digest, injections=args.injections,
                           step_rad=args.step_rad, step_s=args.step_s, threshold_rad=args.threshold_rad)
    rclpy.init(args=None, signal_handler_options=SignalHandlerOptions.NO)
    node = RecordingNode(profile, config)
    previous = signal.signal(signal.SIGINT, lambda *_: setattr(node, "abort_requested", True))
    results = []
    try:
        report = stage_preflight(node, config, paths)
        if not report["ok"]:
            print(json.dumps(report, indent=2), file=sys.stderr)
            return 1
        sim_to_driver = config.description.sim_to_driver()
        measured = node.measured_positions()
        home = np.asarray([measured[sim_to_driver[j]] for j in config.joint_order])
        node.start_bag(paths.bag_dir)
        for k in range(args.injections):
            if node.abort_requested:
                break
            sign = 1.0 if k % 2 == 0 else -1.0
            target = home.copy()
            target[args.joint] += sign * args.step_rad
            trajectory = _quintic_move(home, target, args.step_s, 1.0 / config.rate_hz, config.joint_order)
            node.monitor_tracking_rad = args.threshold_rad
            node.publish_event(paths.root.name, f"inject_{k}", "inject_start")
            goal = node.build_goal(trajectory, profile.driver_joint_order, sim_to_driver)
            result = node.send_goal_and_wait(goal, timeout_s=args.step_s + 5.0, run_paths=paths)
            node.publish_event(paths.root.name, f"inject_{k}", "inject_end")
            node.monitor_tracking_rad = config.limits.abort.tracking_rad
            results.append(result)
            back = _move_to(node, home, config, args.return_s, paths, f"return_{k}")
            if not back.get("ok"):
                results.append({"return_failed": back})
                break
    finally:
        node.stop_bag()
        node.write_monitor_json(paths.root / "monitor.json", "monitor_test")
        node.destroy_node()
        rclpy.shutdown()
        signal.signal(signal.SIGINT, previous)
    rows = cancel_latencies(paths.bag_dir, controller_name=profile.controller_name)
    monitor = json.loads((paths.root / "monitor.json").read_text())
    summary = {"injections": args.injections, "cancels": sum(1 for r in results if r.get("reason")),
               "stop_confirmed": sum(1 for r in results if r.get("stop_confirmed")),
               "cancel_latency": latency_summary([r for r in rows if r["condition"] == "tracking"]),
               "sample_age_ms": monitor["monitor_test"].get("sample_age_ms"),
               "evaluations": monitor["monitor_test"]["evaluations"], "rows": rows}
    monitor["cancel_latency"] = {"monitor_test": {"cancels": rows, "summary": summary["cancel_latency"]}}
    (paths.root / "monitor.json").write_text(json.dumps(monitor, indent=2))
    (paths.root / "monitor_test.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    return 0
