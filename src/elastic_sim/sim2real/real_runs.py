"""Real erd runs as replay bags (S2R_01 Phase 1).

A run folder (``erd.lab/1``) holds the converted dataset (``dataset/``), the
frozen plan (``plan/``), the robot's raw stream (``raw.parquet``) and the
identification recording (``identification/``). :func:`load_real_run` turns
it into :class:`ReplayBag` s:

* **excitation** bags come from the dataset: row k = plan sample k = FRI
  cycle k of the window (``convert_cli`` aligned and de-duplicated them);
* the **sidecar** attaches what only ``raw.parquet`` carries (``ipo_position``,
  the command echo, the external torque, ``fri_cycle``). The window start
  isn't stored, but the dataset copies ``q`` and ``-ft`` bit for bit from the
  raw stream, so the 10,001 consecutive raw rows that match exactly identify it;
* **identify_excitation** and **hold** bags come from
  ``identification/raw.parquet``, windowed from the events brackets, the plan
  length and ``fri_cycle``.

The sidecar is cached in ``data/sim2real/cache/<robot>/<run>/`` (digest-keyed,
idempotent). Nothing here imports ``erd_recording``: the rules it needs are
re-implemented and cited.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from scipy.signal import savgol_filter

from ..materialized import MaterializedTrajectory
from .config import REPO_ROOT

BagKind = Literal["excitation", "identify_excitation", "hold"]

SIDECAR_SCHEMA = "elastic_sim.sim2real.sidecar/1"
DEFAULT_CACHE_ROOT = REPO_ROOT / "data" / "sim2real" / "cache"
#: ``convert_cli.KUKA_TORQUE_SIGN_REAL``: raw.parquet keeps KUKA's torque
#: sign; the dataset (and every bag here) is in the model convention.
KUKA_TORQUE_SIGN_REAL = -1.0
#: ``convert_cli._SEGMENT_PAD_S``: padding around the events bracket for the
#: cross-correlation refinement.
SEGMENT_PAD_S = 0.5
#: FR_05 §6 / ``convert_cli.standstill_holds``: holds drop their first 0.5 s.
HOLD_SKIP_S = 0.5
PLAN_ANALYTIC_TOLERANCE = 1e-9
_SIDECAR_CHANNELS = ("ipo_position", "commanded_position", "external_effort")


class RealRunError(ValueError):
    """The run folder can't be used for replay; the message says why."""


@dataclass(frozen=True)
class ReplayBag:
    """One recorded segment on its FRI cycle grid, model sign convention."""

    run_id: str
    bag: str
    kind: BagKind
    t: np.ndarray                      # (N,) s, (cycle - cycle0) / rate
    q: np.ndarray                      # (N, n) motor side
    dq: np.ndarray                     # (N, n) SG derivative of q
    tau_cmd: np.ndarray                # (N, n) KUKA commanded torque
    ft: np.ndarray                     # (N, n) joint torque sensor
    ipo: np.ndarray | None = None      # (N, n) KUKA's interpolated setpoint
    cmd_echo: np.ndarray | None = None  # (N, n) FRI command echo
    external: np.ndarray | None = None  # (N, n) external torque, model sign
    fri_cycle: np.ndarray | None = None  # (N,)
    plan: MaterializedTrajectory | None = None
    digests: Mapping[str, str] = field(default_factory=dict)

    @property
    def n_rows(self) -> int:
        return len(self.t)


@dataclass(frozen=True)
class RealRun:
    run_id: str
    path: Path
    robot: str
    hardware: str
    sim_asset: str
    joint_names: tuple[str, ...]
    rate_hz: float
    dataset_sha256: str
    contract: Mapping[str, Any]
    dataset_manifest: Mapping[str, Any]
    lab_config: Mapping[str, Any]
    manifest: Mapping[str, Any]
    plan_manifest: Mapping[str, Any]
    bags: tuple[ReplayBag, ...]
    sidecar: Mapping[str, Any] | None = None  # windows.json of the cache

    def bag(self, name: str) -> ReplayBag:
        for bag in self.bags:
            if bag.bag == name:
                return bag
        raise KeyError(f"{self.run_id}: no bag {name!r}")

    @property
    def n_dof(self) -> int:
        return len(self.joint_names)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_real_run(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    segments: Sequence[BagKind] = ("excitation",),
    sidecar: bool = True,
    run_id: str | None = None,
    cache_root: str | Path | None = None,
) -> RealRun:
    """Load a real run for replay (S2R_01 T-1.1).

    Refuses a run whose ``manifest.yaml`` status isn't ``valid``, a dataset
    whose sha256 differs from ``expected_sha256``, and any plan segment whose
    digest doesn't match ``plan.manifest.json``. ``sidecar`` attaches the raw
    channels (building the cache on first use); ``identify_excitation`` and
    ``hold`` bags need it.
    """
    base = _load_base(Path(path), expected_sha256=expected_sha256, run_id=run_id)
    unknown = set(segments) - {"excitation", "identify_excitation", "hold"}
    if unknown:
        raise ValueError(f"unknown segment kinds: {sorted(unknown)}")
    if not sidecar:
        if set(segments) != {"excitation"}:
            raise RealRunError(f"{base.path}: identify_excitation/hold bags come from the raw recording: "
                               "load with sidecar=True")
        return base
    cache_dir, windows, _ = build_replay_cache(base, cache_root=cache_root)
    cached = pd.read_parquet(cache_dir / "replay_bags.parquet")
    bags = []
    for bag in base.bags:
        rows = cached[cached["bag"] == bag.bag]
        bags.append(dataclasses.replace(
            bag, ipo=_block(rows, "ipo", base.n_dof), cmd_echo=_block(rows, "cmd", base.n_dof),
            external=_block(rows, "ext", base.n_dof), fri_cycle=rows["fri_cycle"].to_numpy(dtype=np.int64)))
    plans = _plan_entries(base.plan_manifest)
    for name, entry in windows["bags"].items():
        if entry["kind"] == "excitation" or entry["kind"] not in segments:
            continue
        rows = cached[cached["bag"] == name]
        plan = _load_plan_segment(base.path, plans, name, base)
        bags.append(_bag_from_frame(rows, base, name, entry["kind"], plan, plans[name]["digest"]))
    order = {"excitation": 0, "identify_excitation": 1, "hold": 2}
    bags = [b for b in bags if b.kind in segments]
    bags.sort(key=lambda b: order[b.kind])
    return dataclasses.replace(base, bags=tuple(bags), sidecar=windows)


def _load_base(path: Path, *, expected_sha256: str | None, run_id: str | None) -> RealRun:
    path = path.expanduser().resolve()
    manifest = _read_yaml(path / "manifest.yaml")
    status = manifest.get("status")
    if status != "valid":
        raise RealRunError(f"{path}: manifest.yaml status is {status!r}, not 'valid'")
    lab = _read_yaml(path / "config.yaml")
    robot = str(lab.get("robot", ""))
    hardware = str(lab.get("hardware", ""))
    sim_asset = str((lab.get("description") or {}).get("sim_asset", robot))
    dataset = path / "dataset" / f"{robot}.parquet"
    if not dataset.is_file():
        raise RealRunError(f"{path}: no dataset at {dataset.relative_to(path)}")
    sha = file_sha256(dataset)
    if expected_sha256 is not None and sha != expected_sha256:
        raise RealRunError(f"{dataset}: sha256 {sha} differs from the expected {expected_sha256}")
    contract = _read_json(dataset.with_suffix(".contract.json"))
    dataset_manifest = _read_json(dataset.with_suffix(".manifest.json"))
    plan_manifest = _read_json(path / "plan" / "plan.manifest.json")
    joint_names = tuple(dataset_manifest.get("joint_names") or plan_manifest["joint_names"])
    if tuple(plan_manifest["joint_names"]) != joint_names:
        raise RealRunError(f"{path}: plan joint names {plan_manifest['joint_names']} differ from the dataset's")
    rate_hz = 1.0 / float(dataset_manifest["sample_time_step"])
    base = RealRun(run_id=run_id or path.name, path=path, robot=robot, hardware=hardware, sim_asset=sim_asset,
                   joint_names=joint_names, rate_hz=rate_hz, dataset_sha256=sha, contract=contract,
                   dataset_manifest=dataset_manifest, lab_config=lab, manifest=manifest,
                   plan_manifest=plan_manifest, bags=())
    _check_event_digests(path / "events.json", _plan_entries(plan_manifest))
    frame = pd.read_parquet(dataset)
    plans = _plan_entries(plan_manifest)
    bags = []
    for name in pd.unique(frame["bag"]):
        rows = frame[frame["bag"] == name]
        if name not in plans or plans[name]["kind"] != "excitation":
            raise RealRunError(f"{path}: dataset bag {name!r} has no excitation segment in plan.manifest.json")
        plan = _load_plan_segment(path, plans, name, base)
        if len(rows) != len(plan.time):
            raise RealRunError(f"{path}: bag {name} has {len(rows)} rows, its plan {len(plan.time)} samples")
        n = len(joint_names)
        bags.append(ReplayBag(
            run_id=base.run_id, bag=str(name), kind="excitation", t=rows["t"].to_numpy(dtype=float),
            q=_block(rows, "q", n), dq=_block(rows, "dq", n), tau_cmd=_block(rows, "tau", n), ft=_block(rows, "ft", n),
            plan=plan, digests={"plan": plans[name]["digest"], "dataset_sha256": sha}))
    return dataclasses.replace(base, bags=tuple(bags))


def _plan_entries(plan_manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    entries = {}
    for key in ("segments", "standstill_segments", "identify_segments"):
        for entry in plan_manifest.get(key, []):
            entries[entry["segment_id"]] = dict(entry)
    return entries


def _load_plan_segment(path: Path, plans: Mapping[str, Mapping[str, Any]], name: str,
                       run: RealRun) -> MaterializedTrajectory:
    """The frozen plan of a segment, digest-checked, with the analytic
    evaluator attached when it is a Fourier excitation."""
    plan = MaterializedTrajectory.load(path / "plan" / f"{name}.json")
    if plan.digest() != plans[name]["digest"]:
        raise RealRunError(f"{path}: plan/{name}.json digest {plan.digest()} doesn't match plan.manifest.json "
                           f"({plans[name]['digest']})")
    if tuple(plan.joint_names) != run.joint_names:
        raise RealRunError(f"{path}: plan/{name}.json joint names differ from the dataset's")
    if plan.metadata.get("generator") != "fourier_excitation":
        return plan
    from ..assets import AssetRegistry
    from ..excitation import trajectory_from_metadata

    asset = AssetRegistry.for_repository().load(run.sim_asset)
    step = float(plan.time[1] - plan.time[0])
    rebuilt = trajectory_from_metadata(asset, dict(plan.metadata), time_step=step)
    if rebuilt.position.shape != plan.position.shape:
        raise RealRunError(f"{path}: plan/{name}.json has {plan.position.shape} samples, its coefficients "
                           f"rebuild {rebuilt.position.shape}")
    error = float(np.max(np.abs(rebuilt.position - plan.position)))
    if error > PLAN_ANALYTIC_TOLERANCE:
        raise RealRunError(f"{path}: plan/{name}.json analytic evaluator differs from its samples by "
                           f"{error:.3g} rad (> {PLAN_ANALYTIC_TOLERANCE:g})")
    return dataclasses.replace(plan, analytic=rebuilt.analytic)


def _check_event_digests(events_path: Path, plans: Mapping[str, Mapping[str, Any]]) -> None:
    if not events_path.is_file():
        return
    for event in json.loads(events_path.read_text(encoding="utf-8")):
        entry = plans.get(event["segment_id"])
        if entry is not None and event.get("plan_digest") not in (None, entry["digest"]):
            raise RealRunError(f"{events_path}: {event['segment_id']} ran plan digest {event['plan_digest']}, "
                               f"plan.manifest.json has {entry['digest']}")


# ---------------------------------------------------------------------------
# Sidecar (T-1.2, T-1.3) and cache (T-1.4)
# ---------------------------------------------------------------------------


def cache_dir_for(run: RealRun, cache_root: str | Path | None = None) -> Path:
    root = Path(cache_root) if cache_root is not None else DEFAULT_CACHE_ROOT
    return root / run.robot / run.path.name


def build_replay_cache(
    run: RealRun | str | Path, *, cache_root: str | Path | None = None, force: bool = False,
    expected_sha256: str | None = None,
) -> tuple[Path, dict[str, Any], bool]:
    """Build (or reuse) ``replay_bags.parquet`` + ``windows.json`` for a run.

    Returns ``(cache_dir, windows, built)``; ``built`` is False when a cache
    with the same input key already exists (idempotent no-op).
    """
    if not isinstance(run, RealRun):
        run = _load_base(Path(run), expected_sha256=expected_sha256, run_id=None)
    target = cache_dir_for(run, cache_root)
    key, inputs = _cache_key(run)
    windows_path, bags_path = target / "windows.json", target / "replay_bags.parquet"
    if not force and windows_path.is_file() and bags_path.is_file():
        existing = json.loads(windows_path.read_text(encoding="utf-8"))
        if existing.get("key") == key:
            return target, existing, False
    frames, windows = build_sidecar(run)
    windows = {"schema": SIDECAR_SCHEMA, "key": key, "inputs": inputs, **windows}
    target.mkdir(parents=True, exist_ok=True)
    tmp_bags, tmp_windows = bags_path.with_suffix(".parquet.tmp"), windows_path.with_suffix(".json.tmp")
    pd.concat(frames, ignore_index=True).to_parquet(tmp_bags, index=False)
    tmp_windows.write_text(json.dumps(_jsonable(windows), indent=2), encoding="utf-8")
    tmp_bags.replace(bags_path)
    tmp_windows.replace(windows_path)
    return target, json.loads(windows_path.read_text(encoding="utf-8")), True


def _cache_key(run: RealRun) -> tuple[str, dict[str, Any]]:
    inputs: dict[str, Any] = {"dataset_sha256": run.dataset_sha256}
    for name in ("raw.parquet", "events.json", "plan/plan.manifest.json", "identification/raw.parquet",
                 "identification/events.json"):
        source = run.path / name
        inputs[name] = file_sha256(source) if source.is_file() else None
    payload = json.dumps({"schema": SIDECAR_SCHEMA, **inputs}, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest(), inputs


def build_sidecar(run: RealRun) -> tuple[list[pd.DataFrame], dict[str, Any]]:
    """The cache content: one frame per bag (every replay column) and the
    per-bag windows/checks plus the run summary."""
    if run.hardware != "real":
        raise RealRunError(f"{run.path}: the sidecar needs hardware 'real' (got {run.hardware!r}); a mock run's "
                           "dataset is resampled, so its rows can't be traced back to raw.parquet")
    raw_path = run.path / "raw.parquet"
    if not raw_path.is_file():
        raise RealRunError(f"{run.path}: no raw.parquet (the sidecar needs the raw stream)")
    n = run.n_dof
    sign = KUKA_TORQUE_SIGN_REAL
    raw = pd.read_parquet(raw_path)
    unique = raw.drop_duplicates("fri_cycle")
    q_raw = _block(unique, "q", n)
    ft_raw = sign * _block(unique, "ft", n)
    haystack = np.hstack([q_raw, ft_raw])
    stamps = raw[_stamp_column(raw)].to_numpy()
    events = _read_json_list(run.path / "events.json")

    frames, entries = [], {}
    for bag in run.bags:
        needle = np.hstack([bag.q, bag.ft])
        starts = match_block(haystack, needle)
        if len(starts) != 1:
            raise RealRunError(f"{run.path}: bag {bag.bag}: {len(starts)} exact matches of its "
                               f"{len(needle)} q/-ft rows in raw.parquet (need exactly 1)")
        window = unique.iloc[starts[0]:starts[0] + bag.n_rows]
        cycles = window["fri_cycle"].to_numpy(dtype=np.int64)
        consecutive = bool(np.all(_cycle_steps(cycles) == 1))
        t_error = float(np.max(np.abs(bag.t - (cycles - cycles[0]) / run.rate_hz)))
        if not consecutive or t_error > 1e-9:
            raise RealRunError(f"{run.path}: bag {bag.bag}: matched rows aren't consecutive FRI cycles "
                               f"(consecutive={consecutive}, max |t - cycle/rate| = {t_error:.3g} s)")
        ipo = _block(window, "ipo_position", n)
        entry = {"kind": "excitation", "source": "raw.parquet",
                 "raw_index": [int(window.index[0]), int(window.index[-1])],
                 "fri_cycle": [int(cycles[0]), int(cycles[-1])],
                 "rows": int(bag.n_rows), "matched_rows": int(bag.n_rows), "consecutive_cycles": consecutive,
                 "tracking": _tracking(bag.q, ipo)}
        latency = _event_latency(events, bag.bag, stamps, raw["fri_cycle"].to_numpy(dtype=np.int64), cycles[0])
        if latency is not None:
            entry["latency_cycles"] = latency
        entries[bag.bag] = entry
        frames.append(_bag_frame(bag.bag, "excitation", bag.t, bag.q, bag.dq, bag.tau_cmd, bag.ft, window, n, sign))

    identification = _identification_bags(run)
    for name, (kind, frame, entry) in identification.items():
        frames.append(frame)
        entries[name] = entry

    summary = {"tracking": _run_tracking(frames, entries, n)}
    latencies = [e["latency_cycles"] for e in entries.values() if e["kind"] == "excitation" and "latency_cycles" in e]
    if latencies:
        summary["excitation_latency_cycles"] = {"median": float(np.median(latencies)), "min": int(min(latencies)),
                                                "max": int(max(latencies))}
        for name, entry in entries.items():
            if entry["kind"] == "identify_excitation" and "latency_cycles" in entry:
                delta = entry["latency_cycles"] - float(np.median(latencies))
                entry["latency_vs_excitation_cycles"] = delta
                entry["latency_ok"] = bool(abs(delta) <= 1.0)
    holds = [f for f in frames if f["kind"].iat[0] == "hold"]
    if holds:
        summary["holds"] = hold_checks(holds, run)
    return frames, {"run": {"path": str(run.path), "robot": run.robot, "joint_names": list(run.joint_names),
                            "rate_hz": run.rate_hz},
                    "bags": entries, "summary": summary}


def match_block(haystack: np.ndarray, needle: np.ndarray) -> list[int]:
    """Every start row where ``needle`` (m x c) occurs as consecutive rows of
    ``haystack`` (N x c), compared exactly (no tolerance)."""
    haystack = np.asarray(haystack)
    needle = np.asarray(needle)
    m = len(needle)
    candidates = np.flatnonzero(np.all(haystack == needle[0], axis=1))
    candidates = candidates[candidates + m <= len(haystack)]
    return [int(s) for s in candidates if np.array_equal(haystack[s:s + m], needle)]


def _identification_bags(run: RealRun) -> dict[str, tuple[str, pd.DataFrame, dict[str, Any]]]:
    """identify_excitation_* and identify_hold_* from identification/raw.parquet (T-1.3)."""
    recording = run.path / "identification"
    plans = _plan_entries(run.plan_manifest)
    wanted = [name for name, entry in plans.items() if entry["kind"] in ("identify_hold", "identify_excitation")]
    if not wanted or not (recording / "raw.parquet").is_file():
        return {}
    n = run.n_dof
    sign = KUKA_TORQUE_SIGN_REAL
    raw = pd.read_parquet(recording / "raw.parquet")
    events = _read_json_list(recording / "events.json")
    stamps = raw[_stamp_column(raw)].to_numpy()
    all_cycles = raw["fri_cycle"].to_numpy(dtype=np.int64)
    unique = raw.drop_duplicates("fri_cycle")
    pad = int(round(SEGMENT_PAD_S * run.rate_hz))
    sg_window, sg_poly = _sg(run)
    result = {}
    for name in wanted:
        entry_plan = plans[name]
        kind = "hold" if entry_plan["kind"] == "identify_hold" else "identify_excitation"
        bracket = _event_bracket(events, name)
        if bracket is None:
            raise RealRunError(f"{recording}: no start/end events for {name}")
        if bracket[0].get("plan_digest") != entry_plan["digest"]:
            raise RealRunError(f"{recording}: {name} ran plan digest {bracket[0].get('plan_digest')}, "
                               f"plan.manifest.json has {entry_plan['digest']}")
        lo, hi = (int(i) for i in np.searchsorted(stamps, [bracket[0]["stamp_ns"], bracket[1]["stamp_ns"]]))
        plan = _load_plan_segment(run.path, plans, name, run)
        n_plan = len(plan.time)
        entry: dict[str, Any] = {"kind": kind, "source": "identification/raw.parquet",
                                 "events_raw_index": [lo, hi]}
        if kind == "identify_excitation":
            start = refine_window(raw, (lo, hi), plan.position, n, pad_samples=pad)
            entry["latency_cycles"] = int(all_cycles[start] - all_cycles[lo])
            first = int(np.searchsorted(unique.index.to_numpy(), start))
            window = unique.iloc[first:first + n_plan]
        else:
            first = int(np.searchsorted(unique.index.to_numpy(), lo))
            last = int(np.searchsorted(unique.index.to_numpy(), hi))
            window = unique.iloc[first:min(last, first + n_plan)]
            skip = int(round(HOLD_SKIP_S * run.rate_hz))
            entry["dropped_first_rows"] = skip
            window = window.iloc[skip:]
        cycles = window["fri_cycle"].to_numpy(dtype=np.int64)
        steps = _cycle_steps(cycles)
        entry.update(raw_index=[int(window.index[0]), int(window.index[-1])],
                     fri_cycle=[int(cycles[0]), int(cycles[-1])], rows=int(len(window)),
                     plan_samples=n_plan, consecutive_cycles=bool(np.all(steps == 1)),
                     lost_cycles=int(np.maximum(steps - 1, 0).sum()))
        if kind == "identify_excitation" and len(window) != n_plan:
            raise RealRunError(f"{recording}: {name}: {len(window)} unique cycles after the refined start, "
                               f"plan has {n_plan}")
        t = (cycles - cycles[0]) / run.rate_hz
        q = _block(window, "q", n)
        dq = savgol_filter(q, sg_window, sg_poly, deriv=1, delta=1.0 / run.rate_hz, axis=0, mode="interp")
        entry["tracking"] = _tracking(q, _block(window, "ipo_position", n))
        frame = _bag_frame(name, kind, t, q, dq, sign * _block(window, "tau", n), sign * _block(window, "ft", n),
                           window, n, sign)
        result[name] = (kind, frame, entry)
    return result


def refine_window(raw: pd.DataFrame, window: tuple[int, int], plan_position: np.ndarray, n_dof: int, *,
                  pad_samples: int) -> int:
    """Start row of a segment, by cross-correlating its plan with the
    ``commanded_position`` echo (sum over joints, de-meaned) over the events
    bracket padded by ``pad_samples``: ``convert_cli._refine_window`` +
    ``bagio.find_segment_start``."""
    start, end = window
    padded_start = max(0, start - pad_samples)
    padded_end = min(len(raw), end + pad_samples)
    reference = _block(raw.iloc[padded_start:padded_end], "commanded_position", n_dof).sum(axis=1)
    plan_signal = np.asarray(plan_position, dtype=float).sum(axis=1)
    if len(reference) < len(plan_signal):
        return start
    correlation = np.correlate(reference - reference.mean(), plan_signal - plan_signal.mean(), mode="valid")
    return padded_start + int(np.argmax(correlation))


def hold_checks(holds: Sequence[pd.DataFrame], run: RealRun) -> dict[str, Any]:
    """FR_05 §6 standstill checks on the hold bags: sigma(ft) (linear-detrended
    per hold, mean over holds, as ``validate.noise_table``) and the per-joint
    slope of ft on the asset's Pinocchio gravity (``fit_linear_per_joint``)."""
    from .. import identification as idn
    from ..assets import AssetRegistry

    n = run.n_dof
    sigmas = []
    for frame in holds:
        values = _block(frame, "ft", n)
        t = np.arange(len(values), dtype=float)
        design = np.column_stack([t, np.ones_like(t)])
        coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
        sigmas.append(np.std(values - design @ coefficients, axis=0))
    pin, model, data = idn.build_model(AssetRegistry.for_repository().load(run.sim_asset))
    measured = np.vstack([_block(f, "ft", n) for f in holds])
    gravity = np.vstack([np.asarray([pin.computeGeneralizedGravity(model, data, row).copy()
                                     for row in _block(f, "q", n)]) for f in holds])
    slope, intercept, r2 = _linear_fit(measured, gravity)
    return {"n_holds": len(holds), "rows": [int(len(f)) for f in holds],
            "sigma_ft": np.mean(sigmas, axis=0).tolist(),
            "gravity_slope": slope.tolist(), "gravity_intercept": intercept.tolist(), "gravity_r2": r2.tolist(),
            "statistic": "sigma: linear-detrended per hold, mean over holds; slope: ft = a * g(q) + b per joint, "
                         "all hold samples stacked (first 0.5 s of each hold dropped)"}


def _linear_fit(target: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = target.shape[1]
    slope, intercept, r2 = np.zeros(n), np.zeros(n), np.zeros(n)
    for j in range(n):
        design = np.column_stack([candidate[:, j], np.ones(len(candidate))])
        (a, b), *_ = np.linalg.lstsq(design, target[:, j], rcond=None)
        residual = target[:, j] - (a * candidate[:, j] + b)
        ss_tot = float(np.sum((target[:, j] - target[:, j].mean()) ** 2))
        slope[j], intercept[j] = a, b
        r2[j] = 1.0 - float(np.sum(residual ** 2)) / ss_tot if ss_tot > 0.0 else 1.0
    return slope, intercept, r2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bag_frame(name: str, kind: str, t: np.ndarray, q: np.ndarray, dq: np.ndarray, tau: np.ndarray,
               ft: np.ndarray, window: pd.DataFrame, n: int, sign: float) -> pd.DataFrame:
    columns: dict[str, Any] = {"bag": name, "kind": kind, "t": t}
    for prefix, values in (("q", q), ("dq", dq), ("tau", tau), ("ft", ft),
                           ("ipo", _block(window, "ipo_position", n)),
                           ("cmd", _block(window, "commanded_position", n)),
                           ("ext", sign * _block(window, "external_effort", n))):
        for j in range(n):
            columns[f"{prefix}{j}"] = values[:, j]
    columns["fri_cycle"] = window["fri_cycle"].to_numpy(dtype=np.int64)
    columns["raw_index"] = window.index.to_numpy(dtype=np.int64)
    return pd.DataFrame(columns)


def _bag_from_frame(rows: pd.DataFrame, run: RealRun, name: str, kind: BagKind, plan: MaterializedTrajectory,
                    digest: str) -> ReplayBag:
    n = run.n_dof
    return ReplayBag(run_id=run.run_id, bag=name, kind=kind, t=rows["t"].to_numpy(dtype=float),
                     q=_block(rows, "q", n), dq=_block(rows, "dq", n), tau_cmd=_block(rows, "tau", n),
                     ft=_block(rows, "ft", n), ipo=_block(rows, "ipo", n), cmd_echo=_block(rows, "cmd", n),
                     external=_block(rows, "ext", n), fri_cycle=rows["fri_cycle"].to_numpy(dtype=np.int64),
                     plan=plan, digests={"plan": digest, "dataset_sha256": run.dataset_sha256})


def _tracking(q: np.ndarray, ipo: np.ndarray) -> dict[str, list[float]]:
    error = q - ipo
    return {"rms_q_minus_ipo_rad": np.sqrt(np.mean(error ** 2, axis=0)).tolist(),
            "max_abs_q_minus_ipo_rad": np.max(np.abs(error), axis=0).tolist()}


def _run_tracking(frames: Iterable[pd.DataFrame], entries: Mapping[str, Any], n: int) -> dict[str, Any]:
    excitation = [f for f in frames if f["kind"].iat[0] == "excitation"]
    error = np.vstack([_block(f, "q", n) - _block(f, "ipo", n) for f in excitation])
    worst = max(entries.items(), key=lambda item: max(item[1]["tracking"]["max_abs_q_minus_ipo_rad"])
                if item[1]["kind"] == "excitation" else -1.0)
    return {"excitation_bags": len(excitation),
            "rms_q_minus_ipo_rad": np.sqrt(np.mean(error ** 2, axis=0)).tolist(),
            "max_abs_q_minus_ipo_rad": np.max(np.abs(error), axis=0).tolist(),
            "worst_bag": worst[0]}


def _event_bracket(events: Sequence[Mapping[str, Any]], segment_id: str):
    """First ``*_start`` and last ``*_end`` event (``bagio.event_window``)."""
    starts = [e for e in events if e["segment_id"] == segment_id and e["kind"].endswith("_start")]
    ends = [e for e in events if e["segment_id"] == segment_id and e["kind"].endswith("_end")]
    return (starts[0], ends[-1]) if starts and ends else None


def _event_latency(events, segment_id, stamps, cycles, start_cycle) -> int | None:
    bracket = _event_bracket(events, segment_id)
    if bracket is None:
        return None
    lo = int(np.searchsorted(stamps, bracket[0]["stamp_ns"]))
    return int(start_cycle - cycles[min(lo, len(cycles) - 1)])


def _cycle_steps(cycles: np.ndarray) -> np.ndarray:
    steps = np.diff(np.asarray(cycles, dtype=np.int64))
    steps[steps < -(2 ** 31)] += 2 ** 32  # FRI's uint32 sequence wraps
    return steps


def _stamp_column(raw: pd.DataFrame) -> str:
    """``identify_cli.stamp_column``: publish-side stamps, ``bag_stamp_ns`` on old files."""
    return "stamp_ns" if "stamp_ns" in raw else "bag_stamp_ns"


def _sg(run: RealRun) -> tuple[int, int]:
    differentiation = run.dataset_manifest.get("differentiation") or run.contract.get("differentiation") or {}
    return int(differentiation.get("sg_window", 5)), int(differentiation.get("sg_poly", 3))


def _block(frame: pd.DataFrame, prefix: str, n: int) -> np.ndarray:
    return frame[[f"{prefix}{j}" for j in range(n)]].to_numpy(dtype=float)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RealRunError(f"missing {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RealRunError(f"missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value
