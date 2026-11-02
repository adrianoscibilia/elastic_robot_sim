"""RR_23 C6a: the in-lab quick-look of one run (``quicklook.json`` / ``.md``).

Reads the run folder as ``convert`` leaves it (``plan/``, ``raw.parquet``,
``events.json``, ``controller_state.parquet``; UR10 also ``identification.json``
and ``identification/``), never the dataset, so it works on ``--ladder`` and
modal-ladder runs too, which never enter one.

Every run: peak ``|tau|``/effort (iiwa ``ft``; UR10 the dataset's current
proxy, from the run's ``identification.json``), max
tracking error against the abort, iiwa ``ft`` RMS above 15 Hz (and its ratio
to ``--previous`` scaled by amplitude), UR10 9.5-12 Hz content (and its ratio
to ``--reference``), and the pass/stop verdict of RR_20 / spec S4.4.
Excitations: T(f_k) on the probe lines. Chirps: peak ``|tau|`` and the measured
Q, the input to ``plan --modal-previous``. ``--mount-check`` (UR10): the
3-25 Hz content of the identify sweeps against ``--reference``'s.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .bagio import event_window
from .config import LabConfig
from .planning import load_plan
from .safety import EXCITATION_KINDS

SCHEMA = "erd.quicklook/1"
#: spec S4.4: peak |tau| above this fraction of effort stops the ladder.
TORQUE_STOP = {"kuka_lbr_iiwa_14_r820": 0.5, "ur10_cb3": 0.6}
#: spec S4.4: iiwa ft RMS above 15 Hz, rung over (previous rung x amplitude ratio).
FT_HF_HZ = 15.0
FT_HF_RATIO_STOP = 3.0
#: UR10 mounting mode (10.7 Hz) band, and the growth that stops (= the mount-check bar).
MOUNT_BAND_HZ = (9.5, 12.0)
MOUNT_GROWTH_STOP = 1.5
#: spec S1 mount check: 3-25 Hz over the identify sweeps, pass <= 1.5 x the reference.
MOUNT_CHECK_BAND_HZ = (3.0, 25.0)
MOUNT_CHECK_PASS = 1.5
#: P1 criterion (reported, not a stop): T >= 0.5 on lines <= 15 Hz.
TRANSMISSION_MIN, TRANSMISSION_TOP_HZ = 0.5, 15.0
#: chirp static torque: the median over the leading hold, skipping its first this many seconds.
HOLD_SKIP_S = 0.2


# ---------------------------------------------------------------------------
# Run access
# ---------------------------------------------------------------------------


class Recording:
    """One recording folder (a run, or its ``identification/``): raw frame, events, windows."""

    def __init__(self, root: Path, segments) -> None:
        self.root = Path(root)
        raw = pd.read_parquet(self.root / "raw.parquet")
        if "fri_cycle" in raw.columns and raw["fri_cycle"].notna().all():  # the mock's counter is NaN
            raw = raw.drop_duplicates("fri_cycle", keep="first").reset_index(drop=True)
        self.raw = raw
        self.events = json.loads((self.root / "events.json").read_text(encoding="utf-8"))
        stamps = raw["stamp_ns" if "stamp_ns" in raw else "bag_stamp_ns"].to_numpy()
        self.windows: dict[str, tuple[int, int, int, int]] = {}
        self.segments = {}
        for segment in segments:
            bracket = event_window(self.events, segment.segment_id)
            if bracket is None:
                continue
            lo_ns, hi_ns = int(bracket[0]["stamp_ns"]), int(bracket[1]["stamp_ns"])
            lo, hi = (int(i) for i in np.searchsorted(stamps, [lo_ns, hi_ns]))
            if hi - lo >= 2:
                self.windows[segment.segment_id] = (lo, hi, lo_ns, hi_ns)
                self.segments[segment.segment_id] = segment

    def frame(self, segment_id: str) -> pd.DataFrame:
        lo, hi = self.windows[segment_id][:2]
        return self.raw.iloc[lo:hi]

    def block(self, segment_id: str, prefix: str, n: int) -> np.ndarray | None:
        columns = [f"{prefix}{i}" for i in range(n)]
        frame = self.frame(segment_id)
        if not set(columns) <= set(frame.columns):
            return None
        return frame[columns].to_numpy(dtype=float)

    def refined_start(self, segment_id: str, plan_q: np.ndarray, n: int, rate_hz: float) -> int:
        """Row index where the segment's measured ``q`` best matches its plan
        (de-meaned cross-correlation of the joint sum, +-0.5 s around the events)."""
        from .bagio import find_segment_start

        lo, hi = self.windows[segment_id][:2]
        pad = int(round(0.5 * rate_hz))
        start, end = max(0, lo - pad), min(len(self.raw), lo + len(plan_q) + pad)
        reference = self.raw.iloc[start:end][[f"q{i}" for i in range(n)]].to_numpy(dtype=float).sum(axis=1)
        if len(reference) < len(plan_q):
            return lo
        return start + find_segment_start(reference, plan_q.sum(axis=1))


def _identification(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "identification.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _k_tau(run_dir: Path) -> np.ndarray | None:
    value = (_identification(run_dir) or {}).get("k_tau")
    return None if value is None else np.asarray(value, dtype=float)


def _torque(config: LabConfig, recording: Recording, segment_id: str,
            identification: Mapping[str, Any] | None) -> np.ndarray | None:
    """iiwa: raw ``ft``. UR10: the dataset's own current proxy (``convert.
    ur10_dataset_frame``: ``K_tau i_act - J_m ddq - friction``) from this run's
    ``identification.json``; ``None`` without one."""
    n = len(config.joint_order)
    if config.robot != "ur10_cb3":
        return recording.block(segment_id, "ft", n)
    if not identification or identification.get("motor_fit") is None:
        return None
    from .convert import ur10_dataset_frame
    from .ur_convert import _motor_fit

    frame = recording.frame(segment_id).reset_index(drop=True).copy()
    if len(frame) < 8 or not {f"i_act{i}" for i in range(n)} <= set(frame.columns):
        return None
    frame["bag"], frame["split"] = segment_id, "quicklook"
    differentiation = identification.get("differentiation") or {}
    proxy = ur10_dataset_frame(frame, n_dof=n, k_tau=np.asarray(identification["k_tau"], dtype=float),
                               motor_fit=_motor_fit(identification), epsilon=identification["epsilon"],
                               sg_window=int(differentiation.get("sg_window", 5)),
                               sg_poly=int(differentiation.get("sg_poly", 3)), time_step=1.0 / config.rate_hz,
                               tau_source_column=identification.get("tau_source", "i_act"))
    return proxy[[f"ft{i}" for i in range(n)]].to_numpy(dtype=float)


def _filtered(x: np.ndarray, band: tuple[float, float | None], rate_hz: float) -> np.ndarray | None:
    """Zero-phase Butterworth (order 4) high-pass ``(lo, None)`` or band-pass ``(lo, hi)``."""
    from scipy.signal import butter, sosfiltfilt

    if len(x) < 64:
        return None
    nyquist = 0.5 * rate_hz
    lo, hi = band
    if hi is None:
        sos = butter(4, lo / nyquist, btype="high", output="sos")
    else:
        sos = butter(4, [lo / nyquist, min(hi, 0.95 * nyquist) / nyquist], btype="band", output="sos")
    return sosfiltfilt(sos, x, axis=0)


def _band_rms(recording: Recording, segment_ids, prefix: str, n: int, band, rate_hz: float,
              scale: np.ndarray | None = None) -> list[float] | None:
    """RMS per joint of ``prefix`` filtered to ``band``, over the given segments together."""
    chunks = []
    for sid in segment_ids:
        block = recording.block(sid, prefix, n)
        if block is None:
            return None
        filtered = _filtered(block * (1.0 if scale is None else scale[None, :]), band, rate_hz)
        if filtered is not None:
            chunks.append(filtered)
    if not chunks:
        return None
    stacked = np.vstack(chunks)
    return np.sqrt(np.mean(stacked ** 2, axis=0)).tolist()


def _ratio(now: list[float] | None, before: list[float] | None, scale: float = 1.0) -> list[float | None] | None:
    if now is None or before is None:
        return None
    return [None if b * scale <= 0 or not math.isfinite(b) else a / (b * scale) for a, b in zip(now, before)]


def _max_defined(values) -> float | None:
    values = [v for v in (values or []) if v is not None and math.isfinite(v)]
    return max(values) if values else None


def _tracking(config: LabConfig, recording: Recording, segment_ids) -> list[float] | None:
    """Max ``|q - JTC reference|`` per joint over the segments (the monitor's own error)."""
    path = recording.root / "controller_state.parquet"
    if not path.is_file():
        return None
    n = len(config.joint_order)
    state = pd.read_parquet(path)
    ref_t = state["stamp_ns"].to_numpy(dtype=np.int64)
    ref = state[[f"ref{i}" for i in range(n)]].to_numpy(dtype=float)
    stamps = recording.raw["stamp_ns" if "stamp_ns" in recording.raw else "bag_stamp_ns"].to_numpy(dtype=np.int64)
    worst = np.zeros(n)
    for sid in segment_ids:
        lo, hi, lo_ns, hi_ns = recording.windows[sid]
        inside = (ref_t >= lo_ns) & (ref_t <= hi_ns)
        if inside.sum() < 2:
            continue
        q = recording.block(sid, "q", n)
        t = stamps[lo:hi]
        for j in range(n):
            interpolated = np.interp(t, ref_t[inside], ref[inside, j])
            worst[j] = max(worst[j], float(np.max(np.abs(q[:, j] - interpolated))))
    return worst.tolist()


# ---------------------------------------------------------------------------
# Amplitude of a run (for the rung-to-rung ratio)
# ---------------------------------------------------------------------------


def run_amplitude(bundle, executed: Mapping[str, Any]) -> float:
    """The run's amplitude scale: the ladder scale of an excitation ladder, the
    chirp rung of a modal plan, 1.0 for a full excitation run or sweeps."""
    for segment in executed.values():
        meta = segment.trajectory.metadata
        if segment.kind == "excitation_commissioning":
            return float(meta.get("ladder_scale", 1.0))
        if segment.kind.startswith("modal_chirp"):
            return float(meta.get("rung", 1.0))
    return 1.0


# ---------------------------------------------------------------------------
# P1: transmission on the probe lines
# ---------------------------------------------------------------------------


def transmission(config: LabConfig, recording: Recording, plan_dir: Path, executed) -> dict[str, Any] | None:
    """T(f_k) = |theta_meas(f_k)| / |theta_cmd(f_k)| per joint and probe line, on the
    excitation's whole periods (rectangular window, exact 0.2 Hz bins); median and
    min over the executed excitations."""
    manifest = json.loads((plan_dir / "plan.manifest.json").read_text(encoding="utf-8"))
    probe = manifest.get("probe")
    excitations = [s for s in executed.values() if s.kind in ("excitation", "excitation_commissioning")]
    if not probe or not excitations:
        return None
    n = len(config.joint_order)
    rate = config.rate_hz
    lines = np.asarray(probe["lines_hz"], dtype=float)
    per_excitation = {}
    for segment in excitations:
        meta = segment.trajectory.metadata
        duration = float(meta.get("n_periods", config.excitation.n_periods)) / float(meta["base_frequency"])
        n_samples = int(round(duration * rate))
        plan_q = segment.trajectory.position[:n_samples]
        start = recording.refined_start(segment.segment_id, plan_q, n, rate)
        measured = recording.raw.iloc[start:start + n_samples][[f"q{i}" for i in range(n)]].to_numpy(dtype=float)
        if len(measured) < n_samples or len(plan_q) < n_samples:
            continue
        bins = np.rint(lines * duration).astype(int)
        commanded = np.fft.rfft(plan_q - plan_q.mean(axis=0), axis=0)[bins]
        observed = np.fft.rfft(measured - measured.mean(axis=0), axis=0)[bins]
        with np.errstate(divide="ignore", invalid="ignore"):
            per_excitation[segment.segment_id] = np.abs(observed) / np.abs(commanded)
    if not per_excitation:
        return None
    stack = np.stack(list(per_excitation.values()))          # excitation x line x joint
    median, minimum = np.nanmedian(stack, axis=0), np.nanmin(stack, axis=0)
    low = lines <= TRANSMISSION_TOP_HZ
    joints = list(config.joint_order)
    return {"lines_hz": lines.tolist(), "excitations": list(per_excitation),
            "median": {j: median[:, i].tolist() for i, j in enumerate(joints)},
            "min": {j: minimum[:, i].tolist() for i, j in enumerate(joints)},
            "criterion": f"T >= {TRANSMISSION_MIN} on lines <= {TRANSMISSION_TOP_HZ:g} Hz (reported, not a stop)",
            "below_criterion": {j: [float(f) for f, t in zip(lines[low], median[low, i]) if t < TRANSMISSION_MIN]
                                for i, j in enumerate(joints)}}


# ---------------------------------------------------------------------------
# P2: chirps
# ---------------------------------------------------------------------------


def chirps(config: LabConfig, recording: Recording, plan_dir: Path, executed, identification) -> dict[str, Any]:
    """Per chirp: peak ``|tau_j|``/effort, the static torque (leading hold), the
    dynamic peak ``max |tau_j - static|`` over the chirp, and ``Q_measured`` =
    dynamic peak / (``M_jj`` a)."""
    manifest = json.loads((plan_dir / "plan.manifest.json").read_text(encoding="utf-8"))
    details = (manifest.get("modal") or {}).get("segments", {})
    arrays = config.limits.as_arrays(config.joint_order)
    order = list(config.joint_order)
    rate = config.rate_hz
    result = {}
    for sid, segment in executed.items():
        if not segment.kind.startswith("modal_chirp") or sid not in details:
            continue
        info = details[sid]
        j = order.index(info["joint"])
        torque = _torque(config, recording, sid, identification)
        entry = {"joint": info["joint"], "pose_name": info["pose_name"], "repeat": info["repeat"],
                 "rung": info["rung"], "a": info["a_max"], "M_jj": info["M_jj"], "Q_used": info["Q_used"],
                 "kind": segment.kind, "peak_nm": None, "peak_fraction": None, "static_nm": None,
                 "peak_dynamic_nm": None, "Q_measured": None}
        if torque is not None and np.isfinite(torque[:, j]).all():
            tau = torque[:, j]
            hold = int(round(config.modal.hold_s * rate))
            skip = int(round(HOLD_SKIP_S * rate))
            static = float(np.median(tau[skip:hold])) if hold > skip + 2 else float(np.median(tau))
            dynamic = float(np.max(np.abs(tau - static)))
            peak = float(np.max(np.abs(tau)))
            entry.update(peak_nm=peak, peak_fraction=peak / float(arrays["effort"][j]), static_nm=static,
                         peak_dynamic_nm=dynamic,
                         Q_measured=dynamic / (info["M_jj"] * info["a_max"]) if dynamic > 0 else None)
        result[sid] = entry
    return result


# ---------------------------------------------------------------------------
# The quick-look
# ---------------------------------------------------------------------------


def _previous_quicklook(run: str | Path | None) -> dict[str, Any] | None:
    if run is None:
        return None
    path = Path(run) / "quicklook.json"
    if not path.is_file():
        raise FileNotFoundError(f"{run}: no quicklook.json (run `quicklook` on it first)")
    return json.loads(path.read_text(encoding="utf-8"))


def _identification_recording(config: LabConfig, run_dir: Path, bundle) -> Recording | None:
    root = run_dir / "identification"
    if not (root / "raw.parquet").is_file() or not (root / "events.json").is_file():
        return None
    return Recording(root, bundle.identify_segments)


def mount_content(config: LabConfig, run_dir: Path, band: tuple[float, float]) -> dict[str, Any] | None:
    """RMS of ``q`` and ``K_tau i`` (raw current when K_tau is unknown) in ``band``
    over a run's identify sweeps."""
    bundle = load_plan(run_dir / "plan")
    recording = _identification_recording(config, run_dir, bundle)
    if recording is None:
        return None
    sweeps = [sid for sid, s in recording.segments.items() if s.kind == "identify_sweep"]
    if not sweeps:
        return None
    n = len(config.joint_order)
    k_tau = _k_tau(run_dir)
    return {"segments": len(sweeps), "band_hz": list(band),
            "q_rad": _band_rms(recording, sweeps, "q", n, band, config.rate_hz),
            "current": _band_rms(recording, sweeps, "i_act", n, band, config.rate_hz, scale=k_tau),
            "current_unit": "N*m (K_tau i)" if k_tau is not None else "A"}


def quicklook(config: LabConfig, run_dir: Path, *, previous: Path | None = None, reference: Path | None = None,
              mount_check: bool = False) -> dict[str, Any]:
    run_dir = Path(run_dir)
    bundle = load_plan(run_dir / "plan")
    n = len(config.joint_order)
    joints = list(config.joint_order)
    rate = config.rate_hz
    arrays = config.limits.as_arrays(config.joint_order)
    manifest_path = run_dir / "manifest.yaml"
    run_manifest = {}
    if manifest_path.is_file():
        import yaml

        run_manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    report: dict[str, Any] = {"schema": SCHEMA, "run": run_dir.name, "robot": config.robot,
                              "hardware": config.hardware, "status": run_manifest.get("status"),
                              "joint_names": joints, "rules": [], "verdict": None}
    rules = report["rules"]
    identification = _identification(run_dir) if config.robot == "ur10_cb3" else None
    k_tau = None if identification is None or identification.get("k_tau") is None else \
        np.asarray(identification["k_tau"], dtype=float)

    recording = None
    if (run_dir / "raw.parquet").is_file() and (run_dir / "events.json").is_file():
        recording = Recording(run_dir, bundle.segments)
    executed = recording.segments if recording else {}
    motion = [sid for sid, s in executed.items() if s.kind in EXCITATION_KINDS]
    report["executed_segments"] = list(executed)
    report["amplitude"] = run_amplitude(bundle, executed) if executed else None
    report["plan_mode"] = json.loads((run_dir / "plan" / "plan.manifest.json").read_text()).get("mode", "excitation")

    failed = run_manifest.get("status") == "failed"
    rules.append({"rule": "run completed (no protective stop, FRI drop or abort)", "ok": not failed,
                  "value": run_manifest.get("status"),
                  "detail": str(run_manifest.get("reason") or run_manifest.get("error"))[:300] if failed else None})

    if executed:
        torque_rows = [t for t in (_torque(config, recording, sid, identification) for sid in executed) if t is not None]
        if torque_rows:
            peak = np.nanmax(np.abs(np.vstack(torque_rows)), axis=0)
            fraction = peak / arrays["effort"]
            report["torque"] = {"source": "ft (raw)" if config.robot != "ur10_cb3" else
                                "current proxy K_tau i_act - J_m ddq - friction (identification.json)",
                                "peak_nm": dict(zip(joints, peak.tolist())),
                                "peak_fraction": dict(zip(joints, fraction.tolist()))}
            limit = TORQUE_STOP[config.robot]
            worst = float(np.nanmax(fraction))
            rules.append({"rule": f"peak |tau|/effort <= {limit}", "ok": bool(worst <= limit), "value": worst})
        else:
            report["torque"] = None
            rules.append({"rule": "peak |tau|/effort", "ok": None, "value": None,
                          "detail": "no torque channel (mock, or UR10 without identification.json K_tau)"})

        tracking = _tracking(config, recording, list(executed))
        abort = float(run_manifest.get("abort_tracking_rad_override") or config.limits.abort.tracking_rad)
        report["tracking"] = None if tracking is None else {"max_rad": dict(zip(joints, tracking)), "abort_rad": abort}
        rules.append({"rule": "max tracking error < abort", "ok": None if tracking is None else bool(max(tracking) < abort),
                      "value": None if tracking is None else max(tracking), "limit": abort})

    if config.robot == "kuka_lbr_iiwa_14_r820" and motion:
        hf = _band_rms(recording, motion, "ft", n, (FT_HF_HZ, None), rate)
        report["ft_hf_rms_nm"] = None if hf is None else dict(zip(joints, hf))
        before = _previous_quicklook(previous)
        if before is not None and hf is not None and before.get("ft_hf_rms_nm"):
            scale = (report["amplitude"] or 1.0) / (before.get("amplitude") or 1.0)
            ratio = _ratio(hf, [before["ft_hf_rms_nm"][j] for j in joints], scale)
            report["ft_hf_ratio"] = {"previous": before["run"], "amplitude_scale": scale, "ratio": dict(zip(joints, ratio))}
            worst = _max_defined(ratio)
            rules.append({"rule": f"ft > {FT_HF_HZ:g} Hz RMS <= {FT_HF_RATIO_STOP} x previous rung (amplitude-scaled)",
                          "ok": None if worst is None else bool(worst <= FT_HF_RATIO_STOP), "value": worst})
        else:
            rules.append({"rule": f"ft > {FT_HF_HZ:g} Hz RMS vs previous rung", "ok": None, "value": None,
                          "detail": "first rung (no --previous)" if before is None else "no ft channel"})

    if config.robot == "ur10_cb3":
        content = {"identify_sweeps": mount_content(config, run_dir, MOUNT_BAND_HZ)}
        if motion:
            content["motion"] = {"q_rad": _band_rms(recording, motion, "q", n, MOUNT_BAND_HZ, rate),
                                 "current": _band_rms(recording, motion, "i_act", n, MOUNT_BAND_HZ, rate, scale=k_tau)}
        report["mount_mode_content"] = content
        before = None if mount_check else _previous_quicklook(reference)
        if before is not None and before.get("mount_mode_content"):
            ratios = {}
            for key in ("identify_sweeps", "motion"):
                now, ref = content.get(key), before["mount_mode_content"].get(key)
                if now and ref:
                    ratios[key] = {"q": _ratio(now["q_rad"], ref["q_rad"]), "current": _ratio(now["current"], ref["current"])}
            report["mount_mode_growth"] = {"reference": before["run"], "ratios": ratios}
            # like with like: the identify sweeps when both runs have them
            basis = ratios.get("identify_sweeps") or ratios.get("motion")
            worst = None if basis is None else _max_defined((basis["q"] or []) + (basis["current"] or []))
            rules.append({"rule": f"{MOUNT_BAND_HZ[0]:g}-{MOUNT_BAND_HZ[1]:g} Hz content <= {MOUNT_GROWTH_STOP} x the "
                          "mount-check run", "ok": None if worst is None else bool(worst <= MOUNT_GROWTH_STOP),
                          "value": worst, "basis": "identify_sweeps" if ratios.get("identify_sweeps") else "motion",
                          # RR_24 D-4: on URSim the same sweeps scatter 0.6-1.9 x run to run, so this
                          # ratio is shown for the operator, not an automatic stop
                          "stop": False})
        else:
            rules.append({"rule": "10.7 Hz content vs the mount-check run", "ok": None, "value": None,
                          "detail": "this is the mount check" if mount_check else "no --reference"})

    if mount_check and config.robot == "ur10_cb3":
        now = mount_content(config, run_dir, MOUNT_CHECK_BAND_HZ)
        ref = mount_content(config, Path(reference), MOUNT_CHECK_BAND_HZ) if reference else None
        ratio_q = _ratio(now["q_rad"], ref["q_rad"]) if now and ref else None
        ratio_i = _ratio(now["current"], ref["current"]) if now and ref else None
        worst = _max_defined((ratio_q or []) + (ratio_i or []))
        report["mount_check"] = {"reference": None if reference is None else Path(reference).name, "this": now,
                                 "reference_content": ref, "ratio_q": ratio_q, "ratio_current": ratio_i,
                                 "pass_max": MOUNT_CHECK_PASS}
        rules.append({"rule": f"mount check: {MOUNT_CHECK_BAND_HZ[0]:g}-{MOUNT_CHECK_BAND_HZ[1]:g} Hz over the identify "
                      f"sweeps <= {MOUNT_CHECK_PASS} x reference", "ok": None if worst is None else bool(worst <= MOUNT_CHECK_PASS),
                      "value": worst})

    if recording is not None:
        report["transmission"] = transmission(config, recording, run_dir / "plan", executed)
        report["chirps"] = chirps(config, recording, run_dir / "plan", executed, identification)
        q_values = [c["Q_measured"] for c in report["chirps"].values() if c["Q_measured"] is not None]
        if report["chirps"]:
            rules.append({"rule": "measured Q <= 10 (reasonable)", "ok": None if not q_values else bool(max(q_values) <= 10.0),
                          "value": max(q_values) if q_values else None, "stop": False})

    stops = [r for r in rules if r["ok"] is False and r.get("stop", True)]
    report["verdict"] = "stop" if stops else "pass"
    report["stop_reasons"] = [r["rule"] for r in stops]
    return report


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def markdown(report: Mapping[str, Any]) -> str:
    joints = report["joint_names"]
    lines = [f"# Quick-look: {report['run']}", "",
             f"{report['robot']} ({report['hardware']}), plan mode `{report['plan_mode']}`, amplitude "
             f"{_fmt(report.get('amplitude'))}, run status `{report.get('status')}`.", "",
             f"**Verdict: {report['verdict'].upper()}**"
             + (f" ({'; '.join(report['stop_reasons'])})" if report["stop_reasons"] else ""), "",
             "| rule | ok | value |", "|---|---|---|"]
    for rule in report["rules"]:
        lines.append(f"| {rule['rule']} | {_fmt(rule['ok'])} | {_fmt(rule.get('value'))} |")
    per_joint = []
    if report.get("torque"):
        per_joint.append(("peak \\|tau\\|/effort", report["torque"]["peak_fraction"]))
    if report.get("tracking"):
        per_joint.append(("max tracking [mrad]", {j: v * 1e3 for j, v in report["tracking"]["max_rad"].items()}))
    if report.get("ft_hf_rms_nm"):
        per_joint.append(("ft > 15 Hz RMS [N m]", report["ft_hf_rms_nm"]))
    if report.get("ft_hf_ratio"):
        per_joint.append(("ft > 15 Hz ratio", report["ft_hf_ratio"]["ratio"]))
    if per_joint:
        lines += ["", "| joint | " + " | ".join(name for name, _ in per_joint) + " |",
                  "|---|" + "---|" * len(per_joint)]
        for j in joints:
            lines.append(f"| {j} | " + " | ".join(_fmt(values.get(j)) for _, values in per_joint) + " |")
    t = report.get("transmission")
    if t:
        lines += ["", "## Transmission T(f_k), median (min) over excitations", "",
                  "| f [Hz] | " + " | ".join(joints) + " |", "|---|" + "---|" * len(joints)]
        for k, f in enumerate(t["lines_hz"]):
            lines.append(f"| {f:g} | " + " | ".join(f"{t['median'][j][k]:.2f} ({t['min'][j][k]:.2f})" for j in joints) + " |")
    if report.get("chirps"):
        lines += ["", "## Chirps", "", "| segment | rung | peak \\|tau\\| [N m] | /effort | static | dynamic | Q used | Q measured |",
                  "|---|---|---|---|---|---|---|---|"]
        for sid, c in report["chirps"].items():
            lines.append(f"| {sid} | {c['rung']:g} | {_fmt(c['peak_nm'])} | {_fmt(c['peak_fraction'])} | "
                         f"{_fmt(c['static_nm'])} | {_fmt(c['peak_dynamic_nm'])} | {c['Q_used']:.2f} | {_fmt(c['Q_measured'])} |")
    if report.get("mount_check"):
        m = report["mount_check"]
        lines += ["", f"## Mount check vs {m['reference']}", "",
                  "| joint | q ratio | current ratio |", "|---|---|---|"]
        for i, j in enumerate(joints):
            lines.append(f"| {j} | {_fmt((m['ratio_q'] or [None] * len(joints))[i])} | "
                         f"{_fmt((m['ratio_current'] or [None] * len(joints))[i])} |")
    return "\n".join(lines) + "\n"


def write(report: Mapping[str, Any], run_dir: Path) -> None:
    def scalar(value):
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)

    (Path(run_dir) / "quicklook.json").write_text(json.dumps(report, indent=2, default=scalar), encoding="utf-8")
    (Path(run_dir) / "quicklook.md").write_text(markdown(report), encoding="utf-8")


def print_summary(report: Mapping[str, Any]) -> None:
    print(f"quicklook {report['run']}: {report['verdict'].upper()}"
          + (f" -- {'; '.join(report['stop_reasons'])}" if report["stop_reasons"] else ""))
    for rule in report["rules"]:
        print(f"  [{ {True: 'ok', False: 'STOP' if rule.get('stop', True) else 'note', None: '--'}[rule['ok']]:4s}] "
              f"{rule['rule']}: {_fmt(rule.get('value'))}" + (f" ({rule['detail']})" if rule.get("detail") else ""))
    if report.get("chirps"):
        for sid, c in report["chirps"].items():
            print(f"  chirp {sid}: peak {_fmt(c['peak_nm'])} N*m ({_fmt(c['peak_fraction'])} effort), "
                  f"Q used {c['Q_used']:.2f}, measured {_fmt(c['Q_measured'])}")
    t = report.get("transmission")
    if t:
        for j, below in t["below_criterion"].items():
            if below:
                print(f"  T < {TRANSMISSION_MIN} on {j} at {', '.join(f'{f:g}' for f in below)} Hz")
