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
from .contract import (differentiation_from_reference, load_reference, real_baselines, real_contract,  # noqa: E402
                       resolve_reference, write_real_contract)
from .convert import iiwa_dataset_frame  # noqa: E402
from .planning import PlanBundle, load_plan  # noqa: E402
from .profile import profile_for  # noqa: E402
from .validate import (check_envelopes, check_noise, check_reference_tracking, check_session_health_iiwa,  # noqa: E402
                       not_applicable, validate_dataset)

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


# ---------------------------------------------------------------------------
# RR_12 B-1/B-2/C-2 inputs shared by both robots
# ---------------------------------------------------------------------------


def _relative_seconds(stamps_ns: np.ndarray, base_ns: int) -> np.ndarray:
    """Integer-ns stamps to float seconds after ``base_ns`` (float64 epoch
    seconds only resolve 0.24 us, too coarse for the 1e-9 rad check)."""
    return (np.asarray(stamps_ns, dtype=np.int64) - int(base_ns)) * 1e-9


def tracking_for_segment(segment: Any, *, base_ns: int, commanded: tuple[np.ndarray, np.ndarray] | None,
                         jtc: tuple[np.ndarray, np.ndarray] | None,
                         measured: tuple[np.ndarray, np.ndarray] | None) -> dict[str, Any]:
    """Fit the plan's start time to the executed JTC reference and to the
    drive-side command, and report RMS(q - reference) per joint."""
    from .validate import fit_start_time, quintic_hermite

    plan = segment.trajectory
    acceleration = plan.acceleration if plan.acceleration is not None else np.zeros_like(plan.position)
    rel = plan.time - plan.time[0]
    entry: dict[str, Any] = {"planned_samples": int(len(plan.time))}
    if jtc is not None and len(jtc[0]) > 10:
        entry["jtc"] = fit_start_time(_relative_seconds(jtc[0], base_ns), jtc[1], plan.time, plan.position,
                                      plan.velocity, acceleration, timing_slack_s=JTC_TIMING_SLACK_S)
    if commanded is not None and len(commanded[0]) > 10:
        entry["commanded"] = fit_start_time(_relative_seconds(commanded[0], base_ns), commanded[1], plan.time,
                                            plan.position, plan.velocity, acceleration,
                                            timing_slack_s=COMMAND_TIMING_SLACK_S)
    start = entry.get("jtc", entry.get("commanded", {})).get("start_s")
    if measured is not None and start is not None:
        t = _relative_seconds(measured[0], base_ns)
        mask = (t >= start) & (t <= start + rel[-1])
        reference = quintic_hermite(rel, plan.position, plan.velocity, acceleration, t[mask] - start)
        entry["rms_q_minus_reference_rad"] = np.sqrt(np.mean((measured[1][mask] - reference) ** 2, axis=0)).tolist()
    return entry


#: RR_13: the JTC samples its trajectory at its own clock read, which its
#: ``controller_state`` stamp follows by microseconds (measured: p99 0.2 us,
#: max 5 us on rr13_iiwa_all_1) ...
JTC_TIMING_SLACK_S = 50e-6
#: ... and the drive command is one controller cycle behind it (+ jitter).
COMMAND_TIMING_SLACK_S = 1.5e-3


def controller_state_window(run_dir: Path, lo_ns: int, hi_ns: int, n_dof: int) -> tuple[np.ndarray, np.ndarray] | None:
    """The JTC reference inside a segment, on the topic's own stamps (closest
    to when the JTC sampled; the update stamp is up to 70 us off)."""
    path = run_dir / "controller_state.parquet"
    if not path.is_file():
        return None
    frame = pd.read_parquet(path)
    frame = frame[(frame["stamp_ns"] >= lo_ns) & (frame["stamp_ns"] <= hi_ns)]
    return frame["stamp_ns"].to_numpy(dtype=np.int64), frame[[f"ref{i}" for i in range(n_dof)]].to_numpy(dtype=float)


def standstill_holds(config: LabConfig, run_dir: Path, bundle: PlanBundle) -> list[pd.DataFrame]:
    """The identification holds (``identify_hold_*``, standstill_s each at
    the standstill poses) from ``identification/raw.parquet``, minus their
    first 0.5 s: the standstill source for the sign-convention and noise
    checks (RR_12 C-2a)."""
    from .identify_cli import windows

    recording = run_dir / "identification"
    if not (recording / "raw.parquet").is_file() or not bundle.identify_segments:
        return []
    raw = pd.read_parquet(recording / "raw.parquet")
    events = json.loads((recording / "events.json").read_text())
    holds = [s for s in bundle.identify_segments if s.kind == "identify_hold"]
    frames, _ = windows(raw, events, holds)
    skip = int(round(0.5 * config.rate_hz))
    result = []
    for segment in holds:
        frame = frames[segment.segment_id].iloc[skip:].reset_index(drop=True)
        frame["bag"] = segment.segment_id
        frame["split"] = "test"
        result.append(frame)
    return result


def standstill_gravity(config: LabConfig, holds: list[pd.DataFrame]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Per hold: Pinocchio gravity torque at every sample, and at the mean pose."""
    from elastic_sim import identification as idn
    from elastic_sim.assets import AssetRegistry

    pin, model, data = idn.build_model(AssetRegistry.for_repository().load(config.description.sim_asset))
    n = len(config.joint_order)
    per_sample, per_pose = [], []
    for frame in holds:
        q = frame[[f"q{i}" for i in range(n)]].to_numpy(dtype=float)
        per_sample.append(np.asarray([pin.computeGeneralizedGravity(model, data, row).copy() for row in q]))
        per_pose.append(pin.computeGeneralizedGravity(model, data, q.mean(axis=0)).copy())
    return per_sample, per_pose


def sign_convention_check(config: LabConfig, measured: list[np.ndarray], holds: list[pd.DataFrame]) -> dict[str, Any]:
    """RR_01 S8.3 sign convention with RR_12 B-3's testability rule."""
    from .identification import fit_linear_per_joint
    from .identify_cli import detrended_sigma, gravity_testability
    from .validate import check_sign_convention

    if not holds:
        return {"check": "sign convention", "ok": None, "status": "not_available", "note": "no standstill holds"}
    per_sample, per_pose = standstill_gravity(config, holds)
    n = len(config.joint_order)
    stacked_measured, stacked_gravity = np.vstack(measured), np.vstack(per_sample)
    fit = fit_linear_per_joint(stacked_measured, stacked_gravity)
    columns = [f"_m{j}" for j in range(n)]
    sigma = detrended_sigma([pd.DataFrame(m, columns=columns) for m in measured], columns)
    testability = gravity_testability(per_pose, sigma, fit.slope, fit.r_squared, config.joint_order)
    result = check_sign_convention(stacked_measured, stacked_gravity, testability=testability,
                                   joint_names=config.joint_order)
    for entry, test in zip(result["joints"], testability):
        entry.update(gravity_range_nm=test["gravity_range_nm"], standstill_sigma_nm=test["standstill_sigma_nm"],
                     r_squared=test["r_squared"])
    result["offset_nm"] = fit.intercept.tolist()
    return result


def rate_gap_warning(differentiation: dict[str, Any]) -> dict[str, Any] | None:
    """RR_12 C-1: the real file cannot carry the training cutoff."""
    if differentiation.get("cutoff_matched", True):
        return None
    return {"warning": "rate gap",
            "detail": (f"achieved SG cutoff {differentiation['sg_cutoff_hz']:.4g} Hz at "
                       f"{differentiation['sample_rate_hz']:g} Hz (W {differentiation['sg_window']}) vs the "
                       f"reference's {differentiation['reference_cutoff_hz']:.4g} Hz at "
                       f"{differentiation['reference_sample_rate_hz']:g} Hz: declared mismatch pending RR_02 CR-2")}


def excitation_windows(raw: pd.DataFrame, events: list[dict[str, Any]], segments: list[Any]) -> dict[str, tuple[int, int, int, int]]:
    """``{segment_id: (start_index, end_index, start_ns, end_ns)}`` from the
    publish-side stamps (RR_13 B-1)."""
    from .bagio import event_window
    from .identify_cli import stamp_column

    stamps = raw[stamp_column(raw)].to_numpy()
    result = {}
    for segment in segments:
        bracket = event_window(events, segment.segment_id)
        if bracket is None:
            continue
        lo_ns, hi_ns = int(bracket[0]["stamp_ns"]), int(bracket[1]["stamp_ns"])
        lo, hi = np.searchsorted(stamps, [lo_ns, hi_ns])
        result[segment.segment_id] = (int(lo), int(hi), lo_ns, hi_ns)
    return result


def _prefixed(frame: pd.DataFrame, prefix: str, n_dof: int) -> np.ndarray:
    return frame[[f"{prefix}{i}" for i in range(n_dof)]].to_numpy(dtype=float)


def load_losses(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "recorder_losses.json"
    return json.loads(path.read_text()) if path.is_file() else None


def recorder_loss_summary(run_dir: Path) -> dict[str, Any]:
    """Every bag of the run, per topic (RR_12 B-2), for validation.json."""
    summary = {}
    for name, path in (("bag", run_dir / "recorder_losses.json"),
                       ("identification/bag", run_dir / "identification" / "recorder_losses.json"),
                       ("bag_standstill", run_dir / "recorder_losses_standstill.json")):
        if path.is_file():
            report = json.loads(path.read_text())
            summary[name] = {"rosbag2_transport_lost": report.get("rosbag2_transport_lost"),
                             "unattributed": report.get("unattributed"),
                             "missing_by_topic": {t: e["missing"] for t, e in report.get("topics", {}).items()}}
    return summary


def _convert_iiwa(config: LabConfig, run_dir: Path, reference_contract: dict[str, Any]) -> dict[str, Any]:
    from .bagio import data_topic_losses, losses_in_window, robot_clock_samples, segment_completeness, segment_skips

    reference_contract_path = reference_contract["_path"]
    n_dof = len(config.joint_order)

    bundle: PlanBundle = load_plan(run_dir / "plan")
    raw = pd.read_parquet(run_dir / "raw.parquet")
    excitation_segments = [s for s in bundle.segments if s.kind == "excitation"]
    events = json.loads((run_dir / "events.json").read_text())
    bounds = excitation_windows(raw, events, excitation_segments)
    losses = load_losses(run_dir)
    stamp_ns = raw["stamp_ns"].to_numpy() if "stamp_ns" in raw else raw["bag_stamp_ns"].to_numpy()

    pad_samples = int(round(_SEGMENT_PAD_S * config.rate_hz))
    rows = []
    segment_checks = {}
    session_windows, tracking = {}, {}
    for segment in excitation_segments:
        if segment.segment_id not in bounds:
            segment_checks[segment.segment_id] = {"ok": False, "reason": "missing bracketing events"}
            continue  # a segment /erd/events didn't bracket cleanly: excluded, not guessed at
        start, end, lo_ns, hi_ns = bounds[segment.segment_id]
        bracketed = raw.iloc[start:end]
        complete = segment_completeness(robot_clock_samples(bracketed), len(segment.trajectory.time))
        data_losses = data_topic_losses(losses, lo_ns, hi_ns)
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
        ok = lost == 0 and complete["ok"] and data_losses == 0
        segment_checks[segment.segment_id] = {"ok": ok, "lost_cycles": lost, "samples": len(window),
                                               "completeness": complete, "data_topic_losses": data_losses,
                                               "data_topic_skips": segment_skips(losses, lo_ns, hi_ns),
                                               "losses_by_topic": losses_in_window(losses, lo_ns, hi_ns),
                                               "duration_s": (hi_ns - lo_ns) * 1e-9}
        if config.hardware != "mock":
            segment_checks[segment.segment_id].update({k: v for k, v in health.items() if k != "ok"},
                                                      fri_cycles_ok=health["ok"])
        if not ok:
            continue
        session_windows[segment.segment_id] = bracketed
        bracket_stamps = stamp_ns[bounds[segment.segment_id][0]:bounds[segment.segment_id][1]]
        tracking[segment.segment_id] = tracking_for_segment(
            segment, base_ns=lo_ns,
            # mock: GenericSystem has no FRI command echo (commanded_position
            # is a constant 0 placeholder), so only the JTC part is judged.
            commanded=None if config.hardware == "mock" else (
                bracket_stamps, _prefixed(bracketed, "commanded_position", n_dof)),
            jtc=controller_state_window(run_dir, lo_ns, hi_ns, n_dof),
            measured=(bracket_stamps, _prefixed(bracketed, "q", n_dof)),
        )
        window["bag"] = segment.segment_id
        window["split"] = "test"
        rows.append(window)
    if not rows:
        (run_dir / "validation.json").write_text(json.dumps({"ok": False, "status": "invalid",
                                                            "segments": segment_checks}, indent=2))
        raise RuntimeError("convert_cli: no excitation segment's window could be recovered from the bag/events")
    windowed = pd.concat(rows, ignore_index=True)

    differentiation = differentiation_from_reference(
        reference_contract, rate_real=config.rate_hz, probe_top_real=0.0,
        reference_path=reference_contract_path,
    )
    time_step = 1.0 / config.rate_hz
    frame = iiwa_dataset_frame(
        windowed, n_dof=n_dof, sg_window=differentiation["sg_window"], sg_poly=differentiation["sg_poly"],
        time_step=time_step,
    )

    from elastic_sim.assets import AssetRegistry

    asset = AssetRegistry.for_repository().load(config.description.sim_asset)
    baselines = real_baselines(frame, asset=asset, n_dof=n_dof, time_step=time_step,
                               sg_window=differentiation["sg_window"], sg_poly=differentiation["sg_poly"])

    # RR_12 C-2a: standstill-based checks from the identification holds.
    holds = standstill_holds(config, run_dir, bundle)
    hold_frames = [iiwa_dataset_frame(h, n_dof=n_dof, sg_window=differentiation["sg_window"],
                                      sg_poly=differentiation["sg_poly"], time_step=time_step) for h in holds]
    sign = sign_convention_check(config, [h[[f"ft{j}" for j in range(n_dof)]].to_numpy(dtype=float) for h in holds],
                                 holds)
    noise = check_noise(hold_frames, frame, n_dof=n_dof, ft_offset=sign.get("offset_nm"),
                        source="identification/raw.parquet identify_hold_* (first 0.5 s dropped)")

    arrays = config.limits.as_arrays(config.joint_order)
    manifest = {
        "schema_version": 2, "n_dof": n_dof, "joint_names": list(config.joint_order),
        "signals": {"target": "link_torque", "target_source": "measured", "target_kind": "joint_torque_sensor"},
        "differentiation": {"sg_window": differentiation["sg_window"], "sg_poly": differentiation["sg_poly"]},
        "split": "test", "sample_time_step": time_step,
        "noise": {"dq": {"source": "position_derivative"}},
    }
    contract = real_contract(
        manifest, hardware=config.hardware, n_dof=n_dof, target_instrument="iiwa_joint_torque_sensor_raw",
        target_semantics="joint torque sensor, raw, gravity included [Nm]",
        q_side="motor", dq_side="motor", dq_source="position_derivative",
        tau_instrument="fri_commanded_torque",
        controller={"location": "drive", "vendor": "kuka_position_control", "model_free": False,
                   "reference": "jtc_quintic", "command_latency_cycles": 1},
        differentiation=differentiation, baselines=baselines,
        real_block={"robot_id": config.robot, "sunrise_version": config.connection.sunrise_version,
                    "safety_vendor_checksum": config.safety.vendor_checksum,
                    "reference": _reference_block(reference_contract)},
    )

    contract["reference_synthetic"] = bool(reference_contract.get("synthetic", False))

    from .convert import write_real_dataset

    output = run_dir / "dataset" / f"{config.robot}.parquet"
    csv_path, manifest_path, contract_path = write_real_dataset(frame, manifest, contract, output)

    plan_digests = {s.segment_id: s.digest for s in excitation_segments}
    bag_digests = {event["segment_id"]: event["plan_digest"] for event in events
                   if event["segment_id"] in session_windows and event["kind"].endswith("_start")}
    validation = validate_dataset(
        frame, n_dof=n_dof, nominal_dt=time_step,
        position_lower=arrays["position_lower"], position_upper=arrays["position_upper"],
        quantization=5.989e-8, tau_source="fri_commanded_torque", ft_source="iiwa_joint_torque_sensor_raw",
        effort_limit=arrays["effort"], bag_digests=bag_digests, plan_digests=plan_digests,
        session_health=check_session_health_iiwa(session_windows),
        identification_freshness=not_applicable("identification freshness", config.robot),
        reference_tracking=_mock_tracking_note(config, check_reference_tracking(tracking, commanded_tolerance=1e-6)),
        envelopes=check_envelopes(frame, n_dof=n_dof, effort_limit=arrays["effort"],
                                  sg_window=differentiation["sg_window"], sg_poly=differentiation["sg_poly"],
                                  time_step=time_step),
        noise=noise, sign_convention=sign, hardware=config.hardware, robot=config.robot,
    )
    validation["segments"] = segment_checks
    return _finish_validation(config, run_dir, validation, contract, differentiation, events,
                              {"dataset": str(csv_path), "contract": str(contract_path)})


def _mock_tracking_note(config: LabConfig, check: dict[str, Any]) -> dict[str, Any]:
    if config.hardware == "mock":
        check["note"] = "mock: no FRI command echo (commanded_position is a constant placeholder); JTC part only"
    return check


def _reference_block(reference: dict[str, Any]) -> dict[str, Any]:
    from .contract import portable_path

    return {"path": portable_path(reference["_path"]), "sha256": reference["_sha256"],
            "asset": reference.get("_asset"), "synthetic": bool(reference.get("synthetic", False)),
            "arm_check": reference.get("_arm_check")}


def _finish_validation(config: LabConfig, run_dir: Path, validation: dict[str, Any], contract: dict[str, Any],
                       differentiation: dict[str, Any], events: list[dict[str, Any]],
                       outputs: dict[str, Any]) -> dict[str, Any]:
    from .bagio import event_delivery

    warnings = [w for w in (rate_gap_warning(differentiation),) if w]
    for check in validation["checks"]:
        if check.get("warning"):
            warnings.append({"warning": check["check"], "detail": "see the check entry"})
    validation["warnings"] = warnings
    validation["differentiation"] = differentiation
    validation["reference"] = _reference_block_from_contract(contract)
    validation["recorder_losses"] = recorder_loss_summary(run_dir)
    validation["event_delivery"] = event_delivery(events)
    validation["hardware"] = config.hardware
    # RR_14 P-2: `synthetic` only when every check passed or is on the
    # simulator-limits list; a failing simulator run is `invalid` like a real one.
    if not validation["ok"]:
        validation["status"] = "invalid"
    else:
        validation["status"] = ("synthetic" if contract["reference_synthetic"] or config.hardware != "real"
                                else "valid")
    validation["simulator_limits"] = validation.get("simulator_limits", [])
    validation["reference_synthetic"] = contract["reference_synthetic"]
    (run_dir / "validation.json").write_text(json.dumps(validation, indent=2, default=_json_scalar), encoding="utf-8")
    return {"ok": validation["ok"], **outputs, "validation_ok": validation["ok"], "status": validation["status"],
            "nulls": validation["nulls"], "simulator_limits": validation["simulator_limits"],
            "warnings": [w["warning"] for w in warnings]}


def _json_scalar(value: Any) -> Any:
    """numpy scalars in check entries -> plain JSON values."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def _reference_block_from_contract(contract: dict[str, Any]) -> dict[str, Any]:
    return dict((contract.get("real") or {}).get("reference") or {})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="erd_recording.convert_cli")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--reference-contract", default=None)
    parser.add_argument("--reference-sha256", default=None,
                        help="pin for --reference-contract (the config's pin only covers its own reference)")
    args = parser.parse_args(argv or sys.argv[1:])

    config = load_lab_config(args.config)
    try:
        path, sha = resolve_reference(args.config, config.consumer.reference_contract,
                                      config.consumer.reference_sha256, args.reference_contract, args.reference_sha256)
        if config.hardware == "real" and not sha:
            raise ValueError("hardware: real needs a sha256-pinned reference contract (RR_12 C-1)")
        reference = load_reference(path, robot=config.robot, joint_order=config.joint_order, expected_sha256=sha,
                                   hardware=config.hardware, sim_asset=config.description.sim_asset)
    except ValueError as exc:
        print(json.dumps({"ok": False, "reason": str(exc)}))
        return 2

    if config.robot == "kuka_lbr_iiwa_14_r820":
        result = _convert_iiwa(config, Path(args.run_dir), reference)
    else:
        from .ur_convert import convert_ur
        result = convert_ur(config, Path(args.run_dir), reference)
    print(json.dumps(result))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
