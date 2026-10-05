"""Standalone ``convert`` + ``validate`` stage entry point (no ``rclpy``).

Run as a subprocess from :mod:`erd_recording.pipeline` with
:func:`erd_recording.env_guard.clean_ros_subprocess_env`, for the same reason
``plan_cli`` is (T1.0/T1.6): :func:`erd_recording.contract.real_baselines`
and the sign-convention gravity fit both import Pinocchio (RR_04 A-8).

The UR10 path requires the exact-position-aligned RTDE raw frame and the
recorded E-ur-1/3 results in ``identification.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .env_guard import assert_environment

assert_environment(require_ros=False, require_pinocchio=True)

from .bagio import build_iiwa_raw_frame, find_segment_start, read_bag_topic, iiwa_cycle_health  # noqa: E402
from .config import LabConfig, load_lab_config  # noqa: E402
from .contract import differentiation_from_reference, load_reference, real_baselines, real_contract, write_real_contract  # noqa: E402
from .convert import iiwa_dataset_frame  # noqa: E402
from .planning import PlanBundle, load_plan  # noqa: E402
from .profile import profile_for  # noqa: E402
from .validate import validate_dataset  # noqa: E402

#: RR_01 S3.6: settle time padded onto each excitation window before/after
#: the cross-correlation refinement, so the SG differentiation window (used
#: downstream by iiwa_dataset_frame) never runs off the edge of the segment.
_SEGMENT_PAD_S = 0.5


def _event_segment_windows(
    bag_dir: Path, djs_stamps_ns: np.ndarray, excitation_segment_ids: list[str],
) -> dict[str, tuple[int, int]]:
    """Approximate ``{segment_id: (start_index, end_index)}`` in the raw
    frame's row order, from ``/erd/events``' own bag timestamps matched to
    the nearest ``/dynamic_joint_states`` row (both read from the same bag,
    so they share one clock -- the bag recording time, not either topic's own
    header stamp, RR_04 A-8)."""
    from erd_msgs.msg import RunEvent

    starts: dict[str, int] = {}
    ends: dict[str, int] = {}
    for stamp_ns, event in read_bag_topic(bag_dir, "/erd/events", RunEvent):
        if event.segment_id not in excitation_segment_ids:
            continue
        index = int(np.searchsorted(djs_stamps_ns, stamp_ns))
        if event.kind.endswith("_start"):
            starts[event.segment_id] = index
        elif event.kind.endswith("_end"):
            ends[event.segment_id] = index
    return {sid: (starts[sid], ends[sid]) for sid in excitation_segment_ids if sid in starts and sid in ends}


def _refine_window(
    raw: pd.DataFrame, window: tuple[int, int], plan_position: np.ndarray, n_dof: int, *, pad_samples: int,
) -> tuple[int, int]:
    """Cross-correlate the segment's planned ``commanded_position`` (summed
    over joints, RR_01 S3.6) against the same window of the raw frame, padded
    by ``pad_samples`` on each side, to find the exact-to-one-sample start
    (RR_01 S10's "segment alignment on a synthetic delayed copy" test, here
    applied live)."""
    start, end = window
    padded_start = max(0, start - pad_samples)
    padded_end = min(len(raw), end + pad_samples)
    reference_columns = [f"commanded_position{i}" for i in range(n_dof)]
    reference = raw.loc[padded_start:padded_end - 1, reference_columns].to_numpy().sum(axis=1)
    plan_signal = plan_position.sum(axis=1)
    if len(reference) < len(plan_signal):
        return window  # not enough padding available (segment at the very edge): keep the event-based window
    offset = find_segment_start(reference, plan_signal)
    refined_start = padded_start + offset
    return refined_start, refined_start + len(plan_signal)


def _convert_iiwa(config: LabConfig, run_dir: Path, reference_contract_path: str) -> dict[str, Any]:
    profile = profile_for(config.robot)
    sim_to_driver = config.description.sim_to_driver()
    driver_to_sim = {driver: sim for sim, driver in sim_to_driver.items()}
    n_dof = len(config.joint_order)

    bundle: PlanBundle = load_plan(run_dir / "plan")
    raw = pd.read_parquet(run_dir / "raw.parquet")
    excitation_segments = [s for s in bundle.segments if s.kind == "excitation"]
    excitation_ids = [s.segment_id for s in excitation_segments]
    stamps = raw["bag_stamp_ns"].to_numpy()
    events = json.loads((run_dir / "events.json").read_text())
    starts, ends = {}, {}
    for event in events:
        sid = event["segment_id"]
        if sid not in excitation_ids:
            continue
        index = int(np.searchsorted(stamps, event["stamp_ns"]))
        if event["kind"].endswith("_start"):
            starts[sid] = index
        elif event["kind"].endswith("_end"):
            ends[sid] = index
    event_windows = {sid: (starts[sid], ends[sid]) for sid in excitation_ids if sid in starts and sid in ends}

    pad_samples = int(round(_SEGMENT_PAD_S * config.rate_hz))
    rows = []
    segment_checks = {}
    for segment in excitation_segments:
        if segment.segment_id not in event_windows:
            continue  # a segment /erd/events didn't bracket cleanly: excluded, not guessed at
        start, end = event_windows[segment.segment_id]
        if config.hardware != "mock":
            start, end = _refine_window(
                raw, (start, end), segment.trajectory.position, n_dof, pad_samples=pad_samples,
            )
        window = raw.iloc[start:end].copy()
        if config.hardware == "mock":
            # No robot clock exists in GenericSystem. Resample its recorded
            # positions on a nominal grid, explicitly marked synthetic.
            times = window["t"].to_numpy()
            grid = np.arange(times[0], times[-1], 1.0 / config.rate_hz)
            window = pd.DataFrame({column: np.interp(grid, times, window[column])
                                   for column in window if column != "t"})
            window["t"] = grid - grid[0]
            lost = 0
        else:
            window = window.drop_duplicates("fri_cycle").copy()
            cycles = window["fri_cycle"].to_numpy()
            health = iiwa_cycle_health(window)
            lost = health["lost_cycles"] if health["ok"] else max(1, health["lost_cycles"])
            # Robot sequence supplies the uniform sampling grid. Wall-clock
            # FRI timestamps remain in raw.parquet for jitter diagnosis.
            window["t"] = (cycles - cycles[0]) / config.rate_hz
        segment_checks[segment.segment_id] = {"ok": lost == 0, "lost_cycles": lost,
                                               "samples": len(window)}
        if config.hardware != "mock":
            segment_checks[segment.segment_id].update(health)
        if lost:
            continue
        window["bag"] = segment.segment_id
        window["split"] = "test"
        rows.append(window)
    if not rows:
        (run_dir / "validation.json").write_text(json.dumps({"ok": False, "status": "invalid",
                                                            "segments": segment_checks}, indent=2))
        raise RuntimeError("convert_cli: no excitation segment's window could be recovered from the bag/events")
    windowed = pd.concat(rows, ignore_index=True)

    reference_contract = load_reference(reference_contract_path, robot=config.robot, joint_order=config.joint_order)
    differentiation = differentiation_from_reference(
        reference_contract, rate_real=config.rate_hz, probe_top_real=0.0,
        reference_path=reference_contract_path,
    )
    frame = iiwa_dataset_frame(
        windowed, n_dof=n_dof, sg_window=differentiation["sg_window"], sg_poly=differentiation["sg_poly"],
        time_step=1.0 / config.rate_hz,
    )

    from elastic_sim.assets import AssetRegistry

    asset = AssetRegistry.for_repository().load(config.description.sim_asset)
    baselines = real_baselines(frame, asset=asset, n_dof=n_dof, time_step=1.0 / config.rate_hz,
                               sg_window=differentiation["sg_window"], sg_poly=differentiation["sg_poly"])

    arrays = config.limits.as_arrays(config.joint_order)
    manifest = {
        "schema_version": 2, "n_dof": n_dof, "joint_names": list(config.joint_order),
        "signals": {"target": "link_torque", "target_source": "measured", "target_kind": "joint_torque_sensor"},
        "differentiation": {"sg_window": differentiation["sg_window"], "sg_poly": differentiation["sg_poly"]},
        "split": "test", "sample_time_step": 1.0 / config.rate_hz,
        "noise": {"dq": {"source": "position_derivative"}},
    }
    contract = real_contract(
        manifest, n_dof=n_dof, target_instrument="iiwa_joint_torque_sensor_raw",
        target_semantics="joint torque sensor, raw, gravity included [Nm]",
        q_side="motor", dq_side="motor", dq_source="position_derivative",
        tau_instrument="fri_commanded_torque",
        controller={"location": "drive", "vendor": "kuka_position_control", "model_free": False,
                   "reference": "jtc_quintic", "command_latency_cycles": 1},
        differentiation=differentiation, baselines=baselines,
        real_block={"robot_id": config.robot, "sunrise_version": config.connection.sunrise_version},
    )

    contract["reference_synthetic"] = bool(reference_contract.get("synthetic", False))

    from .convert import write_real_dataset

    output = run_dir / "dataset" / f"{config.robot}.parquet"
    csv_path, manifest_path, contract_path = write_real_dataset(frame, manifest, contract, output)

    plan_digests = {s.segment_id: s.digest for s in excitation_segments}
    bag_digests = {event["segment_id"]: event["plan_digest"] for event in events
                   if event["segment_id"] in event_windows and event["kind"].endswith("_start")}
    validation = validate_dataset(
        frame, n_dof=n_dof, nominal_dt=1.0 / config.rate_hz,
        position_lower=arrays["position_lower"], position_upper=arrays["position_upper"],
        quantization=5.989e-8, tau_source="fri_commanded_torque", ft_source="iiwa_joint_torque_sensor_raw",
        effort_limit=arrays["effort"], bag_digests=bag_digests, plan_digests=plan_digests,
    )
    validation["segments"] = segment_checks
    if config.hardware == "mock":
        # The zero-torque GenericSystem is not a physical torque fixture.
        # Keep the failed diagnostic visible, but do not certify it as real.
        for check in validation["checks"]:
            if check["check"] == "tau != ft":
                check["measured_ok"] = check["ok"]
                check["ok"] = None
                check["note"] = "GenericSystem has no physical torque channels"
        validation["ok"] = not any(c["ok"] is False for c in validation["checks"])
    validation["status"] = "synthetic" if contract["reference_synthetic"] or config.hardware != "real" else ("valid" if validation["ok"] else "invalid")
    validation["reference_synthetic"] = contract["reference_synthetic"]
    (run_dir / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    return {"ok": validation["ok"], "dataset": str(csv_path), "contract": str(contract_path),
            "validation_ok": validation["ok"], "status": validation["status"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="erd_recording.convert_cli")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--reference-contract", default=None)
    args = parser.parse_args(argv or sys.argv[1:])

    config = load_lab_config(args.config)
    reference_contract = args.reference_contract or config.consumer.reference_contract
    if not reference_contract:
        print(json.dumps({"ok": False, "reason": "no --reference-contract given and "
                          "consumer.reference_contract is null (RR_01 S8.2)"}))
        return 2

    reference_contract = str((Path(args.config).resolve().parent / reference_contract).resolve())

    if config.robot == "kuka_lbr_iiwa_14_r820":
        result = _convert_iiwa(config, Path(args.run_dir), reference_contract)
    else:
        from .ur_convert import convert_ur
        result = convert_ur(config, Path(args.run_dir), reference_contract)
    print(json.dumps(result))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
