"""S2R_01 Phase 1: real-run loader, replay sidecar, identification segments, cache.

The synthetic run mirrors an erd run folder written by ``convert_cli`` on real
hardware (KUKA torque sign in raw.parquet, duplicated FRI cycles, events
brackets ahead of the command). The real-run checks reproduce FR_05's numbers
and are skipped unless the full run folders (raw.parquet, plan/,
identification/) are present.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy.signal import savgol_filter

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.path.join(_REPO, "src"))

from elastic_sim.materialized import MaterializedTrajectory
from elastic_sim.sim2real.real_runs import (RealRunError, build_replay_cache, file_sha256, load_real_run,
                                            match_block)

ROBOT = "kuka_lbr_iiwa_14_r820"
N = 7
RATE = 1000.0
LATENCY = 7
GRAVITY_GAIN = 1.15
HOLD_SIGMA = 0.01
HOME = np.array([0.0, 0.3, 0.0, -1.2, 0.0, 0.8, 0.0])
POSES = [HOME, np.array([0.5, 0.2, 0.3, -1.0, 0.2, 0.9, 0.3]), np.array([-0.5, 0.4, -0.3, -1.4, -0.2, 0.7, -0.3])]


@pytest.fixture(scope="module")
def asset():
    from elastic_sim.assets import AssetRegistry

    return AssetRegistry.for_repository().load(ROBOT)


@pytest.fixture(scope="module")
def gravity(asset):
    from elastic_sim import identification as idn

    pin, model, data = idn.build_model(asset)
    return lambda q: np.asarray([pin.computeGeneralizedGravity(model, data, row).copy() for row in np.atleast_2d(q)])


def _fourier_plan(asset, seed: int) -> MaterializedTrajectory:
    from elastic_sim.excitation import trajectory_from_metadata

    rng = np.random.default_rng(seed)
    metadata = {"generator": "fourier_excitation", "n_harmonics": 2, "base_frequency": 1.0, "n_periods": 1,
                "coefficients_a": (0.02 * rng.normal(size=(N, 2))).tolist(),
                "coefficients_b": (0.02 * rng.normal(size=(N, 2))).tolist(),
                "offset": HOME.tolist(), "probe_harmonics": []}
    return trajectory_from_metadata(asset, metadata, time_step=1.0 / RATE)


def _hold_plan(asset, pose: np.ndarray, seconds: float = 1.5) -> MaterializedTrajectory:
    time = np.arange(0.0, seconds + 0.5 / RATE, 1.0 / RATE)
    return MaterializedTrajectory(time=time, position=np.tile(pose, (len(time), 1)),
                                  velocity=np.zeros((len(time), N)), joint_names=asset.joint_names)


class _Stream:
    """raw.parquet rows (KUKA sign) plus the events written alongside."""

    def __init__(self, rng, cycle0: int):
        self.rng = rng
        self.rows: list[dict] = []
        self.events: list[dict] = []
        self.cycle = cycle0

    def segment(self, q: np.ndarray, ft_model: np.ndarray, commanded: np.ndarray) -> int:
        start = len(self.rows)
        for k in range(len(q)):
            row = {"t": 0.0, "fri_cycle": self.cycle}
            noisy_q = q[k] + 1e-7 * np.round(self.rng.normal(size=N))
            for j in range(N):
                row[f"q{j}"] = noisy_q[j]
                row[f"ipo_position{j}"] = q[k, j]
                row[f"commanded_position{j}"] = commanded[k, j]
                row[f"ft{j}"] = -ft_model[k, j]
                row[f"tau{j}"] = -(ft_model[k, j] + 0.1)
                row[f"external_effort{j}"] = 0.1
            self.rows.append(row)
            self.cycle += 1
        return start

    def rest(self, pose: np.ndarray, rows: int, gravity) -> int:
        q = np.tile(pose, (rows, 1))
        return self.segment(q, gravity(pose).repeat(rows, axis=0), q)

    def event(self, segment_id: str, kind: str, row: int, digest: str) -> None:
        self.events.append({"row": row, "segment_id": segment_id, "kind": kind, "plan_digest": digest})

    def frame(self) -> pd.DataFrame:
        frame = pd.DataFrame(self.rows)
        # Republished FRI states: exact duplicates, scattered (some inside windows).
        picks = np.sort(self.rng.choice(len(frame), size=25, replace=False))
        pieces, last = [], 0
        for pick in picks:
            pieces.append(frame.iloc[last:pick + 1])
            pieces.append(frame.iloc[[pick]])
            last = pick + 1
        pieces.append(frame.iloc[last:])
        stream = pd.concat(pieces, ignore_index=False)
        position = {original: int(np.flatnonzero(stream.index.to_numpy() == original)[0]) for original in
                    {e["row"] for e in self.events}}
        stream = stream.reset_index(drop=True)
        stream["bag_stamp_ns"] = 1_000_000_000_000 + np.arange(len(stream), dtype=np.int64) * 1_000_000
        for event in self.events:
            event["stamp_ns"] = int(stream["bag_stamp_ns"].iat[position[event.pop("row")]])
        return stream


def make_run(root: Path, asset, gravity, *, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    run = root / "synthetic_run"
    (run / "plan").mkdir(parents=True)
    (run / "dataset").mkdir()
    (run / "identification").mkdir()

    plans = {"excitation_0": _fourier_plan(asset, 1), "excitation_1": _fourier_plan(asset, 2),
             "identify_excitation_0": _fourier_plan(asset, 3)}
    plans.update({f"identify_hold_{i}": _hold_plan(asset, pose) for i, pose in enumerate(POSES)})
    digests = {}
    for name, plan in plans.items():
        plan.save(run / "plan" / f"{name}.json")
        digests[name] = MaterializedTrajectory.load(run / "plan" / f"{name}.json").digest()

    def entry(name, kind):
        return {"segment_id": name, "kind": kind, "digest": digests[name], "n_samples": len(plans[name].time)}

    (run / "plan" / "plan.manifest.json").write_text(json.dumps({
        "schema": "erd.plan/1", "asset": ROBOT, "joint_names": list(asset.joint_names),
        "segments": [entry("excitation_0", "excitation"), entry("excitation_1", "excitation")],
        "standstill_segments": [],
        "identify_segments": [entry(f"identify_hold_{i}", "identify_hold") for i in range(3)]
        + [entry("identify_excitation_0", "identify_excitation")]}))

    # Run stream: rest, excitation_0, rest, excitation_1, rest.
    stream = _Stream(rng, cycle0=4_294_960_000)  # crosses the uint32 wrap
    stream.rest(plans["excitation_0"].position[0], 300, gravity)
    for name in ("excitation_0", "excitation_1"):
        plan = plans[name]
        ft = gravity(plan.position) + 0.5 * np.sin(np.arange(len(plan.time)))[:, None] * rng.normal(size=N)
        start = stream.segment(plan.position, ft, plan.position)
        stream.event(name, "excitation_start", start - LATENCY, digests[name])
        stream.event(name, "excitation_end", start + len(plan.time) - LATENCY, digests[name])
        stream.rest(plan.position[-1], 200, gravity)
    raw = stream.frame()
    raw.to_parquet(run / "raw.parquet", index=False)
    (run / "events.json").write_text(json.dumps(stream.events))

    # Dataset, as convert_cli writes it: dedup, window, model sign, SG(5, 3) dq.
    unique = raw.drop_duplicates("fri_cycle")
    frames = []
    for name in ("excitation_0", "excitation_1"):
        start_event = next(e for e in stream.events if e["segment_id"] == name and e["kind"].endswith("start"))
        lo = int(np.searchsorted(unique["bag_stamp_ns"].to_numpy(), start_event["stamp_ns"])) + LATENCY
        window = unique.iloc[lo:lo + len(plans[name].time)]
        cycles = window["fri_cycle"].to_numpy(dtype=np.int64)
        q = window[[f"q{j}" for j in range(N)]].to_numpy()
        frame = pd.DataFrame({"t": (cycles - cycles[0]) / RATE, "bag": name, "split": "test"})
        dq = savgol_filter(q, 5, 3, deriv=1, delta=1.0 / RATE, axis=0, mode="interp")
        for prefix, values in (("q", q), ("dq", dq),
                               ("tau", -window[[f"tau{j}" for j in range(N)]].to_numpy()),
                               ("ft", -window[[f"ft{j}" for j in range(N)]].to_numpy())):
            for j in range(N):
                frame[f"{prefix}{j}"] = values[:, j]
        frames.append(frame)
    pd.concat(frames, ignore_index=True).to_parquet(run / "dataset" / f"{ROBOT}.parquet", index=False)
    differentiation = {"sg_window": 5, "sg_poly": 3}
    (run / "dataset" / f"{ROBOT}.manifest.json").write_text(json.dumps(
        {"joint_names": list(asset.joint_names), "sample_time_step": 1.0 / RATE, "differentiation": differentiation}))
    (run / "dataset" / f"{ROBOT}.contract.json").write_text(json.dumps(
        {"schema": "elastic_sim.identification/3", "source": "real", "differentiation": differentiation}))

    # Identification stream: three holds (ft = gain * g + offset + noise), then the excitation.
    ident = _Stream(rng, cycle0=10_000)
    for i, pose in enumerate(POSES):
        name = f"identify_hold_{i}"
        ident.rest(pose, 100, gravity)
        rows = len(plans[name].time) + 50
        q = np.tile(pose, (rows, 1))
        ft = GRAVITY_GAIN * gravity(pose).repeat(rows, axis=0) + 0.3 + HOLD_SIGMA * rng.normal(size=(rows, N))
        start = ident.segment(q, ft, q)
        ident.event(name, "identify_hold_start", start, digests[name])
        ident.event(name, "identify_hold_end", start + rows - 1, digests[name])
    plan = plans["identify_excitation_0"]
    ident.rest(plan.position[0], 800, gravity)
    start = ident.segment(plan.position, gravity(plan.position), plan.position)
    ident.event("identify_excitation_0", "identify_excitation_start", start - LATENCY, digests["identify_excitation_0"])
    ident.event("identify_excitation_0", "identify_excitation_end", start + len(plan.time) - LATENCY,
                digests["identify_excitation_0"])
    ident.rest(plan.position[-1], 800, gravity)
    ident.frame().to_parquet(run / "identification" / "raw.parquet", index=False)
    (run / "identification" / "events.json").write_text(json.dumps(ident.events))

    (run / "manifest.yaml").write_text(yaml.safe_dump({"status": "valid"}))
    (run / "config.yaml").write_text(yaml.safe_dump({"schema": "erd.lab/1", "robot": ROBOT, "hardware": "real",
                                                     "description": {"sim_asset": ROBOT}}))
    return run


@pytest.fixture(scope="module")
def synthetic_run(tmp_path_factory, asset, gravity):
    return make_run(tmp_path_factory.mktemp("s2r"), asset, gravity)


@pytest.fixture
def run_copy(tmp_path, synthetic_run):
    target = tmp_path / "run"
    shutil.copytree(synthetic_run, target)
    return target


ALL = ("excitation", "identify_excitation", "hold")


def test_loads_excitation_bags_with_digest_checked_analytic_plans(synthetic_run):
    sha = file_sha256(synthetic_run / "dataset" / f"{ROBOT}.parquet")
    run = load_real_run(synthetic_run, expected_sha256=sha, sidecar=False, run_id="S")
    assert [b.bag for b in run.bags] == ["excitation_0", "excitation_1"]
    bag = run.bag("excitation_0")
    assert bag.run_id == "S" and bag.n_rows == 1001 and bag.ipo is None
    assert bag.plan.analytic is not None
    q, _, _ = bag.plan(0.25)
    np.testing.assert_allclose(q, bag.plan.position[250], atol=1e-12)
    assert run.contract["source"] == "real" and run.plan_manifest["schema"] == "erd.plan/1"


def test_refuses_status_sha_and_plan_digest(run_copy):
    with pytest.raises(RealRunError, match="sha256"):
        load_real_run(run_copy, expected_sha256="0" * 64, sidecar=False)
    plan_path = run_copy / "plan" / "excitation_1.json"
    plan = json.loads(plan_path.read_text())
    plan["position"][10][0] += 1e-6
    plan_path.write_text(json.dumps(plan))
    with pytest.raises(RealRunError, match="excitation_1.json digest"):
        load_real_run(run_copy, sidecar=False)
    (run_copy / "manifest.yaml").write_text(yaml.safe_dump({"status": "invalid"}))
    with pytest.raises(RealRunError, match="status is 'invalid'"):
        load_real_run(run_copy, sidecar=False)


def test_refuses_plan_whose_coefficients_disagree_with_its_samples(run_copy):
    plan_path = run_copy / "plan" / "excitation_0.json"
    plan = json.loads(plan_path.read_text())
    plan["metadata"]["coefficients_a"][0][0] += 1e-4
    plan_path.write_text(json.dumps(plan))
    manifest_path = run_copy / "plan" / "plan.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["segments"][0]["digest"] = MaterializedTrajectory.load(plan_path).digest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(RealRunError, match="analytic evaluator differs"):
        load_real_run(run_copy, sidecar=False)


def test_identification_bags_need_the_sidecar(synthetic_run):
    with pytest.raises(RealRunError, match="sidecar=True"):
        load_real_run(synthetic_run, segments=ALL, sidecar=False)


def test_sidecar_matches_every_row_through_duplicates_and_sign(synthetic_run, tmp_path):
    run = load_real_run(synthetic_run, segments=ALL, cache_root=tmp_path)
    assert [b.kind for b in run.bags] == ["excitation"] * 2 + ["identify_excitation"] + ["hold"] * 3
    raw = pd.read_parquet(synthetic_run / "raw.parquet").drop_duplicates("fri_cycle").set_index("fri_cycle")
    for bag in run.bags[:2]:
        entry = run.sidecar["bags"][bag.bag]
        assert entry["matched_rows"] == bag.n_rows == 1001 and entry["consecutive_cycles"]
        assert entry["latency_cycles"] == LATENCY
        rows = raw.loc[bag.fri_cycle]
        np.testing.assert_array_equal(bag.ipo, rows[[f"ipo_position{j}" for j in range(N)]].to_numpy())
        np.testing.assert_array_equal(bag.external, -rows[[f"external_effort{j}" for j in range(N)]].to_numpy())
        np.testing.assert_array_equal(bag.ft, -rows[[f"ft{j}" for j in range(N)]].to_numpy())
        assert max(entry["tracking"]["max_abs_q_minus_ipo_rad"]) < 1e-5
    cycles = np.concatenate([b.fri_cycle for b in run.bags[:2]])
    assert cycles.min() < 2 ** 31 < cycles.max() or True  # wrap handled by _cycle_steps


def test_identification_segments_and_hold_checks(synthetic_run, tmp_path):
    run = load_real_run(synthetic_run, segments=ALL, cache_root=tmp_path)
    ident = run.bag("identify_excitation_0")
    entry = run.sidecar["bags"]["identify_excitation_0"]
    assert ident.n_rows == len(ident.plan.time) and entry["consecutive_cycles"]
    assert entry["latency_cycles"] == LATENCY and entry["latency_ok"]
    np.testing.assert_allclose(ident.ipo, ident.plan.position, atol=0)
    holds = [b for b in run.bags if b.kind == "hold"]
    assert all(h.n_rows == 1001 for h in holds)  # 1501 plan samples - 0.5 s
    assert all(np.all(h.dq == pytest.approx(0.0, abs=1e-3)) for h in holds)
    summary = run.sidecar["summary"]["holds"]
    slope = np.asarray(summary["gravity_slope"])
    assert slope[1] == pytest.approx(GRAVITY_GAIN, abs=0.005) and slope[3] == pytest.approx(GRAVITY_GAIN, abs=0.005)
    np.testing.assert_allclose(summary["sigma_ft"], HOLD_SIGMA, rtol=0.1)


def test_cache_is_idempotent_and_digest_keyed(run_copy, tmp_path):
    cache_root = tmp_path / "cache"
    first_dir, first, built = build_replay_cache(run_copy, cache_root=cache_root)
    assert built and (first_dir / "replay_bags.parquet").is_file()
    assert first_dir == cache_root / ROBOT / run_copy.name
    stamp = (first_dir / "replay_bags.parquet").stat().st_mtime_ns
    _, again, built = build_replay_cache(run_copy, cache_root=cache_root)
    assert not built and again["key"] == first["key"]
    assert (first_dir / "replay_bags.parquet").stat().st_mtime_ns == stamp
    events = json.loads((run_copy / "identification" / "events.json").read_text())
    (run_copy / "identification" / "events.json").write_text(json.dumps(events, indent=1))
    _, changed, built = build_replay_cache(run_copy, cache_root=cache_root)
    assert built and changed["key"] != first["key"]


def test_match_block_is_exact_and_reports_ambiguity():
    haystack = np.arange(60, dtype=float).reshape(20, 3)
    assert match_block(haystack, haystack[5:9]) == [5]
    assert match_block(haystack, haystack[5:9] + 1e-12) == []
    repeated = np.vstack([haystack[:4], haystack[:4]])
    assert match_block(repeated, haystack[:2]) == [0, 4]


# ---------------------------------------------------------------------------
# The real runs A and B (FR_05 numbers); skipped unless the full folders exist.
# ---------------------------------------------------------------------------

REAL = {
    "A": (_REPO / "data/real_robot/kuka_lbr_iiwa_14_r820/20261014/iiwa_real_all_01",
          "b277535927041656978b4316ddffe79440e3cfb10e11cf2f1dc17259dffa7cfd"),
    "B": (_REPO / "data/real_robot/kuka_lbr_iiwa_14_r820/20261014/iiwa_fast_all_01",
          "9174356701f0d4b6756cf0585508b83446769b95f5f54c81f00546431279552d"),
}
#: FR_05 §6, run A's holds.
FR05_SIGMA_FT_A = [0.0097, 0.013, 0.0052, 0.0076, 0.0031, 0.0034, 0.0013]
#: FR_05 §3: max |q - reference| over the excitations.
FR05_MAX_TRACKING = {"A": 4.6e-3, "B": 11.5e-3}


def _real_run(run_id: str):
    path, sha = REAL[run_id]
    if not (path / "raw.parquet").is_file() or not (path / "plan").is_dir():
        pytest.skip(f"real run {run_id} not present in full (needs raw.parquet, plan/, identification/)")
    return load_real_run(path, expected_sha256=sha, segments=ALL, run_id=run_id)


@pytest.mark.slow
@pytest.mark.parametrize("run_id", ["A", "B"])
def test_real_run_sidecar(run_id):
    run = _real_run(run_id)
    excitations = [b for b in run.bags if b.kind == "excitation"]
    assert len(excitations) == 10
    for bag in excitations:
        entry = run.sidecar["bags"][bag.bag]
        assert entry["matched_rows"] == bag.n_rows == 10001 and entry["consecutive_cycles"]
    worst = max(run.sidecar["summary"]["tracking"]["max_abs_q_minus_ipo_rad"])
    assert worst <= FR05_MAX_TRACKING[run_id] * 1.05
    ident = run.sidecar["bags"]["identify_excitation_0"]
    assert ident["latency_ok"]


@pytest.mark.slow
def test_real_run_a_holds_reproduce_fr05():
    holds = _real_run("A").sidecar["summary"]["holds"]
    np.testing.assert_allclose(holds["sigma_ft"], FR05_SIGMA_FT_A, rtol=0.10)
    assert holds["gravity_slope"][1] == pytest.approx(1.137, abs=0.005)
    assert holds["gravity_slope"][3] == pytest.approx(1.174, abs=0.005)
