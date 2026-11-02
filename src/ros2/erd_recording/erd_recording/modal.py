"""RR_23 C4/C5: P2 single-joint modal chirps and the TE constant-velocity sweeps.

Nothing here touches ROS. :func:`build_modal_segments` and
:func:`build_te_sweep_segments` are called by ``planning.build_plan`` (``plan
--modal`` / ``plan --te-sweeps``); :func:`measured_q` reads a previous rung's
run folder for the torque rule.

**Chirp law** (RR_20 S4, spec S3): one joint moves, the others hold the pose,
``theta_j = q_pose + A(f) w(t) sin phi(t)``, a log chirp ``f0 -> f1`` over ``T``,
``A(f) = min(a/(2 pi f)^2, position_amplitude_max, v/(2 pi f), j/(2 pi f)^3)``,
``a = rung x a_max``, ``w`` a cosine fade, holds before and after. It is r9's
own :class:`elastic_sim.round8.ChirpEvaluator`, so the real and the simulated
chirps are one function. ``v`` and ``j`` are :data:`CAP_MARGIN` x the lab
limits: the fade and the sweep add a few per cent to the peaks the clip
alone predicts, and the caps are checked on the analytic signal afterwards.

**Torque rule** (RR_20 S4.3, RR_23 C4): predicted peak ``|g_j| + M_jj a Q``,
``Q`` = :data:`Q_DEFAULT` unless a previous rung of the same (joint, pose) was
measured (``--modal-previous``). A chirp predicted above
``torque_fraction_max x effort`` is left out of the plan and listed.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from elastic_sim.materialized import MaterializedTrajectory

from .config import LabConfig

#: Resonance amplification assumed when no previous rung was measured (RR_20 S4.3).
Q_DEFAULT = 10.0
#: A measured Q below this is raised to it (the rigid response alone is Q = 1).
Q_FLOOR = 1.0
#: The chirp law's velocity and jerk clips are this fraction of the lab limits.
CAP_MARGIN = 0.95
#: Peaks and the clip table are evaluated on the analytic signal at least this fine.
FINE_RATE_HZ = 2000.0

SPLIT_MODAL = "modal_reference"
SPLIT_CALIBRATION = "calibration"
SPLIT_COMMISSIONING = "commissioning"
KIND_CHIRP = "modal_chirp"
KIND_CHIRP_COMMISSIONING = "modal_chirp_commissioning"
KIND_SWEEP = "calibration_sweep"


class ModalError(ValueError):
    """A modal or sweep plan cannot be built as asked."""


# ---------------------------------------------------------------------------
# Chirp law
# ---------------------------------------------------------------------------


def round8_spec(config: LabConfig, joint: str, pose_index: int, a: float):
    """The :class:`elastic_sim.round8.ModalSpec` of one (joint, pose) at acceleration ``a``."""
    from elastic_sim.round8 import ModalSpec as Round8Modal

    modal = config.modal
    band = modal.joints[joint]
    return Round8Modal(
        joints=((joint, band.f0, band.f1, band.T),),
        poses=((modal.pose_name(pose_index), tuple(config.poses.standstill[pose_index])),),
        a_max=float(a), position_amplitude_max=modal.position_amplitude_max,
        velocity_max=CAP_MARGIN * float(config.limits.velocity[joint]),
        jerk_max=CAP_MARGIN * float(config.limits.jerk[joint]),
        repeats=1, hold_s=modal.hold_s, fade_s=modal.fade_s,
    )


def chirp_trajectory(config: LabConfig, joint_names: Sequence[str], joint: str, pose_index: int, a: float,
                     time_step: float) -> MaterializedTrajectory:
    """One chirp segment (holds, fades and chirp), from r9's ``round8.modal_trajectory``."""
    from elastic_sim.round8 import modal_trajectory

    class _Names:  # modal_trajectory reads only ``asset.joint_names``
        pass

    names = _Names()
    names.joint_names = tuple(joint_names)
    return modal_trajectory(round8_spec(config, joint, pose_index, a), names, 0, time_step)


def bind_ranges(evaluator, n: int = 4000) -> dict[str, list[list[float]]]:
    """``{clip: [[f_lo, f_hi], ...]}``: where each term of ``A(f)`` is the minimum."""
    spec = evaluator.spec
    f = np.geomspace(evaluator.f0, evaluator.f1, n)
    w = 2.0 * math.pi * f
    terms = np.vstack([spec.a_max / w ** 2, np.full_like(w, spec.position_amplitude_max),
                       spec.velocity_max / w, spec.jerk_max / w ** 3])
    names = ("acceleration", "position", "velocity", "jerk")
    which = np.argmin(terms, axis=0)
    ranges: dict[str, list[list[float]]] = {name: [] for name in names}
    start = 0
    for i in range(1, n + 1):
        if i == n or which[i] != which[start]:
            ranges[names[which[start]]].append([round(float(f[start]), 3), round(float(f[i - 1]), 3)])
            start = i
    return ranges


def _derivatives(evaluator, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``x, dx, ddx`` of the moving joint at ``t``: ``ChirpEvaluator.__call__``'s
    central differences, vectorised."""
    h = 1e-5
    xm, x0, xp = evaluator.offset(t - h), evaluator.offset(t), evaluator.offset(t + h)
    return x0, (xp - xm) / (2.0 * h), (xp - 2.0 * x0 + xm) / (h * h)


def grid_jerk(evaluator, time_step: float) -> float:
    """Peak jerk on the plan's own sample grid (what the JTC is sent).

    The r9 cosine fade is C1: at both ends of each fade the acceleration steps
    by ``A pi^2 / (2 fade_s^2) |sin phi|`` (<= 0.05 rad/s^2 on these bands). A
    finite difference at 20x oversampling turns that step into a spike that
    grows as 1/step, so the jerk cap is judged here instead (RR_24 D-1)."""
    t = np.arange(0.0, evaluator.duration + 0.5 * time_step, time_step)
    return float(np.max(np.abs(np.diff(_derivatives(evaluator, t)[2])))) / time_step


def chirp_peaks(evaluator, time_step: float, *, rate_hz: float = FINE_RATE_HZ) -> dict[str, float]:
    """Peak ``|q - q_pose|``, ``|dq|``, ``|ddq|`` of the moving joint on a fine grid,
    and the jerk on the plan's grid (:func:`grid_jerk`)."""
    step = 1.0 / rate_hz
    x, dq, ddq = _derivatives(evaluator, np.arange(0.0, evaluator.duration + 0.5 * step, step))
    return {"position": float(np.max(np.abs(x))), "velocity": float(np.max(np.abs(dq))),
            "acceleration": float(np.max(np.abs(ddq))), "jerk": grid_jerk(evaluator, time_step)}


def pose_dynamics(asset, q: Sequence[float], joint: int) -> dict[str, float]:
    """URDF ``M_jj``, ``g_j`` and ``dg_j/dq_j`` at a pose (bare asset, no payload)."""
    from elastic_sim import identification as idn

    pin, model, data = idn.build_model(asset)
    q = np.asarray(q, dtype=float)
    m_jj = float(np.asarray(pin.crba(model, data, q))[joint, joint])
    g = pin.computeGeneralizedGravity(model, data, q).copy()
    eps = 1e-6
    dq = np.zeros_like(q)
    dq[joint] = eps
    plus = pin.computeGeneralizedGravity(model, data, q + dq).copy()
    minus = pin.computeGeneralizedGravity(model, data, q - dq).copy()
    return {"m_jj": m_jj, "g_j": float(g[joint]), "dg_dq_jj": float((plus[joint] - minus[joint]) / (2 * eps))}


# ---------------------------------------------------------------------------
# Targets and the torque rule
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModalOptions:
    """``plan --modal`` options (RR_23 C4 CLI)."""
    rung: float
    only: tuple[tuple[str, str], ...] | None = None     # (joint, pose name or index)
    repeats: int | None = None
    previous: tuple[str, ...] = ()                      # run folders of earlier rungs


def parse_only(text: str | None) -> tuple[tuple[str, str], ...] | None:
    """``"iiwa_A2:home,iiwa_A4:1"`` -> ``(("iiwa_A2", "home"), ("iiwa_A4", "1"))``."""
    if not text:
        return None
    pairs = []
    for item in text.split(","):
        joint, sep, pose = item.strip().partition(":")
        if not sep or not joint or not pose:
            raise ModalError(f"--modal-only expects <joint>:<pose_name>[,...], got {item!r}")
        pairs.append((joint, pose))
    return tuple(pairs)


def add_mode_arguments(parser) -> None:
    """RR_23 C4/C5: ``plan --modal --rung <r> [--modal-only ...] [--repeats n] [--modal-previous ...]``
    and ``plan --te-sweeps`` (shared by ``record_<robot>`` and this entry point)."""
    parser.add_argument("--modal", action="store_true", help="P2: plan the `modal:` chirps instead of the excitations")
    parser.add_argument("--rung", type=float, default=None, help="--modal: the chirp rung (a = rung x modal.a_max)")
    parser.add_argument("--modal-only", default=None, metavar="JOINT:POSE[,...]",
                        help="--modal: only these chirps, e.g. iiwa_A2:home (pose name from modal.pose_names, or index)")
    parser.add_argument("--repeats", type=int, default=None,
                        help="--modal: repeats per chirp (default 1 below rung 1.0, modal.repeats at a dataset rung)")
    parser.add_argument("--modal-previous", action="append", default=[], metavar="RUN_DIR",
                        help="--modal: an earlier rung's run folder (with quicklook.json); its measured Q replaces "
                             "Q = 10 for the chirps it measured. Repeat or comma-separate for several runs")
    parser.add_argument("--te-sweeps", action="store_true", help="plan the `te_sweeps:` sweeps instead of the excitations")


def modal_options(args):
    """``ModalOptions`` from :func:`add_mode_arguments`' flags, or ``None`` without ``--modal``."""
    if not args.modal:
        for flag in ("rung", "modal_only", "repeats"):
            if getattr(args, flag) is not None:
                raise SystemExit(f"--{flag.replace('_', '-')} needs --modal")
        if args.modal_previous:
            raise SystemExit("--modal-previous needs --modal")
        return None
    if args.rung is None:
        raise SystemExit("--modal needs --rung <r>")
    previous = tuple(p for item in args.modal_previous for p in item.split(",") if p)
    return ModalOptions(rung=args.rung, only=parse_only(args.modal_only), repeats=args.repeats, previous=previous)


def chirp_targets(config: LabConfig, options: ModalOptions) -> list[dict[str, Any]]:
    """The (joint, pose) chirps of one rung: ``modal.poses`` x ``modal.joints``,
    plus each extra pose up to its ``max_rung`` (D5), restricted by ``--modal-only``.

    A chirp enters a dataset (``kind: modal_chirp``) at rung 1.0, and an extra
    pose at its own ``max_rung``; any other rung is commissioning."""
    modal = config.modal
    if modal is None:
        raise ModalError(f"{config.source_path}: no `modal:` block")
    if not 0.0 < options.rung <= 1.0:
        raise ModalError(f"--rung must be in (0, 1], got {options.rung}")
    poses = [(index, None) for index in modal.poses]
    poses += [(extra.index, extra.max_rung) for extra in modal.extra_poses if options.rung <= extra.max_rung + 1e-12]
    targets = []
    for pose_index, max_rung in poses:
        for joint in modal.joints:
            dataset = abs(options.rung - (1.0 if max_rung is None else max_rung)) < 1e-9
            targets.append({"joint": joint, "pose_index": pose_index, "pose_name": modal.pose_name(pose_index),
                            "extra_pose": max_rung is not None, "dataset": dataset})
    if options.only is not None:
        wanted = set()
        for joint, pose in options.only:
            if joint not in modal.joints:
                raise ModalError(f"--modal-only: {joint!r} is not a `modal.joints` entry {list(modal.joints)}")
            index = int(pose) if pose.isdigit() else (modal.pose_names.index(pose) if pose in modal.pose_names else None)
            if index is None:
                raise ModalError(f"--modal-only: no pose named {pose!r} (pose_names {list(modal.pose_names)})")
            wanted.add((joint, index))
        targets = [t for t in targets if (t["joint"], t["pose_index"]) in wanted]
        missing = wanted - {(t["joint"], t["pose_index"]) for t in targets}
        if missing:
            raise ModalError(f"--modal-only {sorted(missing)}: not a chirp of rung {options.rung:g} "
                             "(pose not in modal.poses, or an extra pose above its max_rung)")
    if not targets:
        raise ModalError(f"rung {options.rung:g}: no chirp to plan")
    for target in targets:
        default = modal.repeats if target["dataset"] else 1
        target["repeats"] = int(options.repeats) if options.repeats is not None else default
        target["kind"] = KIND_CHIRP if target["dataset"] else KIND_CHIRP_COMMISSIONING
        target["split"] = SPLIT_MODAL if target["dataset"] else SPLIT_COMMISSIONING
    return targets


def measured_q(run_dirs: Iterable[str | Path]) -> dict[tuple[str, str], dict[str, Any]]:
    """``{(joint, pose_name): {"Q", "rung", "a", "peak_dynamic_nm", "run"}}`` from earlier
    rungs' ``quicklook.json`` (``chirps``). Where several runs measured the same
    chirp, the highest rung wins (then the later run on the command line)."""
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for run_dir in run_dirs:
        run_dir = Path(run_dir).expanduser().resolve()
        path = run_dir / "quicklook.json"
        if not path.is_file():
            raise ModalError(f"--modal-previous {run_dir}: no quicklook.json (run `quicklook` on it first)")
        report = json.loads(path.read_text(encoding="utf-8"))
        chirps = report.get("chirps") or {}
        if not chirps:
            raise ModalError(f"--modal-previous {run_dir}: quicklook.json has no measured chirp")
        best: dict[tuple[str, str], dict[str, Any]] = {}
        for entry in chirps.values():
            if entry.get("Q_measured") is None:
                continue
            key = (entry["joint"], entry["pose_name"])
            if key not in best or entry["Q_measured"] > best[key]["Q"]:  # worst repeat of this run
                best[key] = {"Q": float(entry["Q_measured"]), "rung": float(entry["rung"]), "a": float(entry["a"]),
                             "peak_dynamic_nm": float(entry["peak_dynamic_nm"]), "run": run_dir.name}
        for key, value in best.items():
            if key not in result or value["rung"] >= result[key]["rung"]:
                result[key] = value
    return result


def torque_rule(dynamics: Mapping[str, float], a: float, effort: float, fraction_max: float,
                previous: Mapping[str, Any] | None) -> dict[str, Any]:
    """Predicted peak ``|g_j| + M_jj a Q`` and the refusal (RR_20 S4.3)."""
    if previous is None:
        q_used, source = Q_DEFAULT, "default"
    else:
        q_used = max(Q_FLOOR, float(previous["Q"]))
        source = f"measured {previous['run']} (rung {previous['rung']:g}, Q {previous['Q']:.2f})"
    predicted = abs(dynamics["g_j"]) + dynamics["m_jj"] * a * q_used
    fraction = predicted / effort
    refused = fraction > fraction_max
    return {"Q_used": q_used, "Q_source": source, "predicted_peak_nm": predicted, "effort_nm": effort,
            "predicted_fraction": fraction, "torque_fraction_max": fraction_max, "refused": bool(refused),
            "reason": (f"predicted |tau| {predicted:.1f} N*m = {fraction:.2f} x effort > {fraction_max:g}"
                       if refused else None)}


# ---------------------------------------------------------------------------
# Segments
# ---------------------------------------------------------------------------


def _segment_id(joint: str, pose_name: str, repeat: int) -> str:
    return f"modal_{joint}_{pose_name}_{repeat}"


def build_modal_segments(config: LabConfig, world, asset, joint_names: Sequence[str], time_step: float,
                         margin: float, options: ModalOptions, *, approach, segment_type):
    """The rung's chirps, grouped by pose: approach -> chirps -> ... -> return home.

    ``approach(start, end, sid)`` returns a validated approach segment and
    ``segment_type`` is ``planning.PlanSegment``. Returns ``(segments, report)``;
    the report is ``plan/modal_report.json``."""
    from .planning import PlanError, check_caps_on_analytic

    arrays = config.limits.as_arrays(config.joint_order)
    order = list(config.joint_order)
    previous = measured_q(options.previous) if options.previous else {}
    a = options.rung * config.modal.a_max
    targets = chirp_targets(config, options)
    report: dict[str, Any] = {
        "schema": "erd.modal_report/1", "rung": options.rung, "a": a, "a_max": config.modal.a_max,
        "law": config.modal.law, "cap_margin": CAP_MARGIN, "Q_default": Q_DEFAULT, "Q_floor": Q_FLOOR,
        "torque_model": f"|g_j| + M_jj a Q, URDF ({asset.name}), no payload",
        "previous_runs": [str(p) for p in options.previous],
        "chirps": [], "refused": [], "segments": {},
    }
    by_pose: dict[int, list[dict[str, Any]]] = {}
    for target in targets:
        by_pose.setdefault(target["pose_index"], []).append(target)

    segments = []
    previous_end = np.asarray(config.poses.home, dtype=float)
    for pose_index, pose_targets in by_pose.items():
        pose_q = np.asarray(config.poses.standstill[pose_index], dtype=float)
        pose_segments = []
        for target in pose_targets:
            joint = target["joint"]
            j = order.index(joint)
            band = config.modal.joints[joint]
            dynamics = pose_dynamics(asset, pose_q, j)
            rule = torque_rule(dynamics, a, float(arrays["effort"][j]), config.modal.torque_fraction_max,
                               previous.get((joint, target["pose_name"])))
            trajectory = chirp_trajectory(config, joint_names, joint, pose_index, a, time_step)
            evaluator = trajectory.analytic
            entry = {**{k: target[k] for k in ("joint", "pose_index", "pose_name", "extra_pose", "repeats", "kind", "split")},
                     "pose_q": pose_q.tolist(), "f0": band.f0, "f1": band.f1, "T": band.T, "rung": options.rung, "a": a,
                     "M_jj": dynamics["m_jj"], "g_j": dynamics["g_j"], "dg_dq_jj": dynamics["dg_dq_jj"],
                     **rule, "binds": bind_ranges(evaluator), "peaks": chirp_peaks(evaluator, time_step),
                     "duration_s": trajectory.duration, "segment_ids": []}
            report["chirps"].append(entry)
            if rule["refused"]:
                report["refused"].append({"joint": joint, "pose_name": target["pose_name"], "reason": rule["reason"]})
                continue
            if np.any(trajectory.position < arrays["position_lower"] - 1e-9) or \
                    np.any(trajectory.position > arrays["position_upper"] + 1e-9):
                raise PlanError(f"modal {joint} at {target['pose_name']}: outside limits.position")
            check_caps_on_analytic(trajectory, arrays["velocity"], arrays["acceleration"])  # jerk: grid_jerk
            if entry["peaks"]["jerk"] > arrays["jerk"][j] * (1.0 + 1e-6):
                raise PlanError(f"modal {joint} at {target['pose_name']}: jerk {entry['peaks']['jerk']:.1f} on the "
                                f"plan grid exceeds the cap {arrays['jerk'][j]:g}")
            collision = world.validate_path(trajectory.position, margin=margin, max_joint_step=0.01)
            if not collision.valid:
                raise PlanError(f"modal {joint} at {target['pose_name']}: collision between {collision.closest_pair}")
            info = {k: entry[k] for k in ("joint", "pose_name", "pose_q", "f0", "f1", "T", "rung", "dg_dq_jj",
                                          "Q_used", "kind", "split")}
            info.update(a_max=a, M_jj=entry["M_jj"])
            for repeat in range(target["repeats"]):
                sid = _segment_id(joint, target["pose_name"], repeat)
                meta = {**trajectory.metadata, "rung": options.rung, "repeat": repeat, "segment_kind": target["kind"]}
                frozen = MaterializedTrajectory(time=trajectory.time, position=trajectory.position,
                                                velocity=trajectory.velocity, acceleration=trajectory.acceleration,
                                                joint_names=tuple(joint_names), metadata=meta, analytic=evaluator)
                pose_segments.append(segment_type(sid, target["kind"], frozen, frozen.digest()))
                entry["segment_ids"].append(sid)
                report["segments"][sid] = {**info, "repeat": repeat}
        if pose_segments:
            segments.append(approach(previous_end, pose_q, f"modal_approach_{config.modal.pose_name(pose_index)}",
                                     "modal_approach"))
            segments += pose_segments
            previous_end = pose_q
    if not segments:
        raise ModalError(f"rung {options.rung:g}: every chirp was refused by the torque rule: {report['refused']}")
    segments.append(approach(previous_end, np.asarray(config.poses.home, dtype=float), "modal_return", "modal_return"))
    return segments, report


def sweep_plateau(span: float, speed: float, *, acceleration: float, jerk: float, plateau_s: float,
                  dt: float) -> tuple[float, float]:
    """``(ramp, plateau)`` of one sweep: ``velocity_sweep``'s ramp, and the
    plateau ``min(plateau_s, what fits in span)`` on the ``dt`` grid."""
    ramp = max(np.pi * speed / (2 * acceleration), np.pi * np.sqrt(speed / (2 * jerk)), 2 * dt)
    ramp = float(np.ceil(ramp / dt) * dt)
    fits = span / speed - ramp                    # displacement = speed * (ramp + plateau)
    plateau = float(np.floor(min(plateau_s, fits) / dt + 1e-9) * dt)
    return ramp, plateau


def build_te_sweep_segments(config: LabConfig, world, joint_names: Sequence[str], time_step: float, margin: float,
                            *, approach, segment_type):
    """RR_23 C5: per joint and speed, one sweep each way, centred in the range,
    plateau ``min(plateau_s, what fits)``; shorter than ``min_plateau_s`` is
    skipped and listed. Returns ``(segments, report)`` (``plan/te_sweep_report.json``)."""
    from .identification_plan import velocity_sweep
    from .planning import PlanError

    spec = config.te_sweeps
    if spec is None:
        raise ModalError(f"{config.source_path}: no `te_sweeps:` block")
    arrays = config.limits.as_arrays(config.joint_order)
    order = list(config.joint_order)
    pose = np.asarray(config.poses.standstill[spec.pose], dtype=float)
    report: dict[str, Any] = {"schema": "erd.te_sweep_report/1", "pose": spec.pose, "pose_q": pose.tolist(),
                              "plateau_s": spec.plateau_s, "min_plateau_s": spec.min_plateau_s,
                              "planned": [], "skipped": [], "segments": {}}
    segments = []
    previous = np.asarray(config.poses.home, dtype=float)
    for joint, (lo, hi) in spec.joints.items():
        j = order.index(joint)
        lo, hi = max(lo, float(arrays["position_lower"][j])), min(hi, float(arrays["position_upper"][j]))
        acceleration, jerk = float(arrays["acceleration"][j]), float(arrays["jerk"][j])
        for speed in spec.speeds:
            if speed > float(arrays["velocity"][j]) + 1e-12:
                report["skipped"].append({"joint": joint, "speed": speed, "reason": "above limits.velocity"})
                continue
            ramp, plateau = sweep_plateau(hi - lo, speed, acceleration=acceleration, jerk=jerk,
                                          plateau_s=spec.plateau_s, dt=time_step)
            if plateau < spec.min_plateau_s - 1e-9:
                report["skipped"].append({"joint": joint, "speed": speed, "plateau_fits_s": max(plateau, 0.0),
                                          "reason": f"plateau {max(plateau, 0.0):.2f} s < min_plateau_s"})
                continue
            kwargs = dict(acceleration=acceleration, jerk=jerk, plateau_s=plateau, dt=time_step, joint_names=joint_names)
            displacement = float(velocity_sweep(pose, j, speed, **kwargs).position[-1, j] - pose[j])
            start = pose.copy()
            start[j] = (lo + hi) / 2 - displacement / 2
            label = f"te_{joint}_{speed:g}".replace(".", "p")
            segments.append(approach(previous, start, f"{label}_approach", "te_approach"))
            forward = velocity_sweep(start, j, speed, **kwargs)
            reverse = velocity_sweep(forward.position[-1], j, -speed, **kwargs)
            for direction, trajectory in (("positive", forward), ("negative", reverse)):
                sid = f"{label}_{direction}"
                q = trajectory.position
                if np.any(q < arrays["position_lower"] - 1e-9) or np.any(q > arrays["position_upper"] + 1e-9):
                    raise PlanError(f"{sid}: outside limits.position")
                if np.any(np.abs(trajectory.velocity) > arrays["velocity"] + 1e-8) or \
                        np.any(np.abs(trajectory.acceleration) > arrays["acceleration"] + 1e-8):
                    raise PlanError(f"{sid}: velocity/acceleration limit")
                if np.any(np.abs(np.gradient(trajectory.acceleration, trajectory.time, axis=0)) > arrays["jerk"] + 1e-6):
                    raise PlanError(f"{sid}: jerk limit")
                collision = world.validate_path(q, margin=margin, max_joint_step=0.01)
                if not collision.valid:
                    raise PlanError(f"{sid}: collision between {collision.closest_pair}")
                meta = trajectory.metadata
                segments.append(segment_type(sid, KIND_SWEEP, trajectory, trajectory.digest()))
                report["segments"][sid] = {"joint": joint, "speed": float(np.sign(meta["speed"]) * speed),
                                           "direction": direction, "plateau_start_s": float(meta["plateau_start_s"]),
                                           "plateau_end_s": float(meta["plateau_end_s"]), "kind": KIND_SWEEP,
                                           "split": SPLIT_CALIBRATION}
            report["planned"].append({"joint": joint, "speed": speed, "ramp_s": ramp, "plateau_s": plateau,
                                      "range": [lo, hi], "start": float(start[j]),
                                      "end": float(forward.position[-1, j])})
            previous = reverse.position[-1]
    if not segments:
        raise ModalError(f"te_sweeps: nothing fits: {report['skipped']}")
    segments.append(approach(previous, np.asarray(config.poses.home, dtype=float), "te_return", "te_return"))
    return segments, report


# ---------------------------------------------------------------------------
# Printouts
# ---------------------------------------------------------------------------


def print_modal_report(report: Mapping[str, Any]) -> None:
    print(f"modal chirps: rung {report['rung']:g} (a = {report['a']:g} rad/s^2; law {report['law']}, "
          f"v/jerk clips at {report['cap_margin']:g} x the lab limits)")
    print(f"  {'joint':22s} {'pose':11s} {'band Hz':>11s} {'rep':>3s} {'|dq|':>6s} {'|ddq|':>6s} {'jerk':>6s} "
          f"{'M_jj':>6s} {'|g|':>6s} {'Q':>5s} {'|tau|/eff':>9s}  kind / clips")
    for c in report["chirps"]:
        p = c["peaks"]
        clips = "; ".join(f"{name} {'+'.join(f'{lo:g}-{hi:g}' for lo, hi in spans)}"
                          for name, spans in c["binds"].items() if spans)
        status = "REFUSED" if c["refused"] else c["kind"]
        print(f"  {c['joint']:22s} {c['pose_name']:11s} {c['f0']:4g}-{c['f1']:<5g} {c['repeats']:3d} "
              f"{p['velocity']:6.3f} {p['acceleration']:6.3f} {p['jerk']:6.1f} {c['M_jj']:6.2f} {abs(c['g_j']):6.1f} "
              f"{c['Q_used']:5.2f} {c['predicted_fraction']:9.3f}  {status} / {clips}")
        if c["Q_source"] != "default":
            print(f"      Q from {c['Q_source']}")
    if report["refused"]:
        print("refused chirps (left out of the plan; measure their lower rung first):")
        for r in report["refused"]:
            print(f"  {r['joint']} at {r['pose_name']}: {r['reason']}")
    else:
        print("refused chirps: none")


def print_te_sweep_report(report: Mapping[str, Any]) -> None:
    print(f"TE sweeps at pose {report['pose']} (plateau target {report['plateau_s']:g} s, min {report['min_plateau_s']:g} s):")
    for p in report["planned"]:
        print(f"  planned {p['joint']:22s} {p['speed']:5.2f} rad/s  plateau {p['plateau_s']:5.2f} s  "
              f"ramp {p['ramp_s']:5.3f} s  {p['start']:+.3f} -> {p['end']:+.3f} rad, both directions")
    for s in report["skipped"]:
        print(f"  skipped {s['joint']:22s} {s['speed']:5.2f} rad/s  {s['reason']}")
