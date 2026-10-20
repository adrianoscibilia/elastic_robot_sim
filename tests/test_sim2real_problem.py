"""S2R_01 T-4.2/T-4.3/T-4.5, S2R_04 P3-5/P3-7, S2R_06 P4a-1, S2R_08 P4a-1':
problem, history + resume, identifiability, the calibrated label (A-12') and
freeze-then-refit (A-12), the session zero (A-13), the run folder and
--report-only, on a synthetic real run."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.path.join(_REPO, "src"))
sys.path.insert(0, os.path.join(_REPO, "tests"))

from elastic_sim.sim2real.calibration import Interrupted, calibrate, report_only
from elastic_sim.sim2real.config import load_config
from elastic_sim.sim2real.parameters import Registry
from elastic_sim.sim2real.plant import NONPHYSICAL_LOSS, PlantBase
from elastic_sim.sim2real.problem import CalibrationProblem
from elastic_sim.sim2real.real_runs import load_real_run

BASE_CONFIG = _REPO / "config" / "sim2real" / "kuka_lbr_iiwa_14_r820_base.yaml"
ALL = ("excitation", "identify_excitation", "hold")


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    from test_sim2real_real_runs import make_run

    from elastic_sim import identification as idn
    from elastic_sim.assets import AssetRegistry

    root = tmp_path_factory.mktemp("s2r_problem")
    asset = AssetRegistry.for_repository().load("kuka_lbr_iiwa_14_r820")
    pin, model, data = idn.build_model(asset)

    def gravity(q):
        return np.asarray([pin.computeGeneralizedGravity(model, data, row).copy() for row in np.atleast_2d(q)])

    path = make_run(root, asset, gravity)
    run = load_real_run(path, segments=ALL, run_id="S", cache_root=root / "cache")
    return root, path, run


def _config(root: Path, path: Path, **overrides):
    raw = {"extends": str(BASE_CONFIG), "runs": {"S": {"path": str(path)}},
           "split": {"train": ["S:excitation_0", "S:identify_hold_*"], "val": ["S:excitation_1"],
                     "test": ["S:identify_excitation_0"]},
           "replay": {"mode": "R0", "r0": {"decimate": 10, "hold_decimate": 20}},
           "parameters": {"free": ["inertial.*.mass", "sensor.ft.*.offset@S", "friction.link.*.coulomb"]},
           "loss": {"prior_weight_factor": 0.01},
           "optimizer": {"method": "least_squares", "budget": 4}, "output": str(root / "out")}
    raw.update(overrides)
    target = root / f"stage_{abs(hash(json.dumps(raw, sort_keys=True, default=str))) % 10 ** 8}.yaml"
    target.write_text(yaml.safe_dump(raw))
    return load_config(target)


@pytest.fixture(scope="module")
def base():
    return PlantBase.from_config(load_config(BASE_CONFIG))


def test_calibrate_writes_the_run_folder_and_report_only_rebuilds_it(synthetic, base):
    """Label mode (S2R_08 A-12', the default): the stage's full set in
    best.yaml, labelled once on the full free set, partners for the
    not-calibrated ones, and no value changed against the fit."""
    from elastic_sim.sim2real import identifiability as ida

    root, path, run = synthetic
    config = _config(root, path, optimizer={"method": "least_squares", "budget": 30})
    assert config.identifiability.mode == "label"
    folder = calibrate(config, runs={"S": run}, base=base, run_id="e2e", figures=False)
    for name in ("config.yaml", "inputs.json", "registry.yaml", "identifiability_theta0.json", "identifiability.json",
                 "identifiability.md", "history.jsonl", "working.yaml", "best.yaml", "metrics.json",
                 "replay_train.parquet", "replay_val.parquet", "replay_test.parquet", "report.md"):
        assert (folder / name).is_file(), name
    metrics = json.loads((folder / "metrics.json").read_text())
    assert not (folder / "history_refit1.jsonl").exists() and len(metrics["summary"]["phases"]) == 1
    selected = next(json.loads(x) for x in (folder / "history.jsonl").read_text().splitlines()[1:]
                    if json.loads(x)["i"] == metrics["summary"]["best_index"])
    assert metrics["best_theta"] == selected["theta"]  # no value changes vs the fit
    overlay = yaml.safe_load((folder / "best.yaml").read_text())
    assert overlay["identifiability_mode"] == "label" and "partners" in overlay["note"]
    entries = overlay["parameters"]
    ident = json.loads((folder / "identifiability.json").read_text())
    d_post = {r["name"]: r["data_share_posterior"] for r in ident["parameters"] if r["free"]}
    assert set(entries) == set(d_post) == set(Registry.from_config(config, base).free_names)
    assert ident["a12"]["mode"] == "label" and not ident["a12"]["frozen"]
    labelled = ida.labels(ident)
    assert any(e["calibrated"] for e in labelled.values()) and not all(e["calibrated"] for e in labelled.values())
    for name, entry in entries.items():
        assert entry["calibrated"] == (d_post[name] >= 0.9), name
        if not entry["calibrated"]:
            assert "not separated by these data" in entry["not_calibrated_reason"]
            assert [p["name"] for p in entry["partners"]] == [p["name"] for p in labelled[name]["partners"]]
            assert all(abs(p["corr"]) >= 0.5 and p["name"] in entries for p in entry["partners"])
    report = (folder / "report.md").read_text()
    assert "mode **label**" in report and "partners" in report
    assert "Calibrated label (S2R_08 A-12')" in (folder / "identifiability.md").read_text()
    report_md = report
    (folder / "report.md").unlink()
    report_only(folder, figures=False)
    assert (folder / "report.md").read_text() == report_md


def test_freeze_refit_mode_reproduces_s2r07(synthetic, base):
    """S2R_08 P4a-1' (2): ``mode: freeze_refit`` is S2R_07's A-12, unchanged."""
    root, path, run = synthetic
    config = _config(root, path, optimizer={"method": "least_squares", "budget": 30},
                     identifiability={"mode": "freeze_refit"})
    folder = calibrate(config, runs={"S": run}, base=base, run_id="e2e_freeze_refit", figures=False)
    for name in ("config.yaml", "inputs.json", "registry.yaml", "identifiability_theta0.json", "identifiability.json",
                 "identifiability.md", "history.jsonl", "working.yaml", "best.yaml", "metrics.json",
                 "replay_train.parquet", "replay_val.parquet", "replay_test.parquet", "report.md"):
        assert (folder / name).is_file(), name
    inputs = json.loads((folder / "inputs.json").read_text())
    assert inputs["runs"]["S"]["dataset_sha256"] == run.dataset_sha256 and inputs["code"]["digest"]
    metrics = json.loads((folder / "metrics.json").read_text())
    best = Registry.from_overlay(folder / "best.yaml")
    assert {n: s.value for n, s in best.specs.items()} == metrics["best_theta"]  # bit for bit
    # A-12: the stage's full set, each marked; the calibrated ones carry
    # d_post >= 0.9 (F-2), the frozen ones the d_post that froze them.
    entries = yaml.safe_load((folder / "best.yaml").read_text())["parameters"]
    assert set(entries) == set(Registry.from_config(config, base).free_names)
    ident = json.loads((folder / "identifiability.json").read_text())
    frozen = {e["name"] for e in ident["a12"]["frozen"]}
    assert frozen and len(frozen) < len(entries)
    for name, entry in entries.items():
        assert entry["calibrated"] == (name not in frozen), name
        if entry["calibrated"]:
            assert 0.0 <= entry["data_share_posterior"] <= 1.0 and "data_share" in entry  # A-11 / F-2
        else:
            assert entry["frozen_d_post"] < 0.9 and entry["value"] == Registry.from_config(config, base).specs[
                name].mean
    assert all(r["data_share_posterior"] >= 0.9 for r in ident["parameters"] if r["free"]) or \
        ident["a12"]["below_threshold_after_refit"]
    assert (folder / "history_refit1.jsonl").is_file() and "not calibrated" in (folder / "report.md").read_text()
    # (No test-improvement assertion: the synthetic ft has 4 N m of noise and
    # no link friction, so A-12 freezing Coulomb at its round-6 nominal can
    # cost more than the fit gains. The fit itself must improve on theta0.)
    history = [json.loads(x) for x in (folder / "history.jsonl").read_text().splitlines()[1:]]
    assert metrics["summary"]["phases"][0]["train_loss"] < history[0]["loss"]
    report = (folder / "report.md").read_text()
    (folder / "report.md").unlink()
    report_only(folder, figures=False)
    assert (folder / "report.md").read_text() == report


def _problem(config, run, base, history=None, **kwargs):
    registry = Registry.from_config(config, base)
    return CalibrationProblem.from_config(config, registry, base, {"S": run}, history_path=history, **kwargs)


@pytest.mark.parametrize("method", ["random", "cma", "least_squares"])
def test_resume_reproduces_the_best_theta(synthetic, base, tmp_path, method):
    """T-4.2 done-when: kill and resume == uninterrupted (fixed seed, R0)."""
    from elastic_sim.sim2real.optimizers import CMAOptimizer

    if method == "cma" and not CMAOptimizer.is_available():
        pytest.skip("cma not installed")
    root, path, run = synthetic
    budget, stop = (8, 3) if method == "least_squares" else (14, 5)
    config = _config(root, path, optimizer={"method": method, "budget": budget, "seed": 3})
    full = calibrate(config, runs={"S": run}, base=base, run_id=f"full_{method}", figures=False)
    with pytest.raises(Interrupted):
        calibrate(config, runs={"S": run}, base=base, run_id=f"killed_{method}", figures=False, stop_after=stop)
    resumed = calibrate(config, runs={"S": run}, base=base, run_id=f"killed_{method}", figures=False, resume=True)
    a = json.loads((full / "metrics.json").read_text())
    b = json.loads((resumed / "metrics.json").read_text())
    assert a["best_theta"] == b["best_theta"] and a["summary"]["best_index"] == b["summary"]["best_index"]
    assert a["summary"]["phases"][-1]["phase"] == b["summary"]["phases"][-1]["phase"]
    total = 0
    for history in sorted(resumed.glob("history*.jsonl")):
        lines = [json.loads(x) for x in history.read_text().splitlines()][1:]
        assert [r["i"] for r in lines] == list(range(len(lines)))  # nothing re-logged
        total += len(lines)
    assert total == a["summary"]["evaluations"]


def test_identifiability_flags_what_the_data_cannot_see(synthetic, base):
    from elastic_sim.sim2real import identifiability as ida

    root, path, run = synthetic
    config = _config(root, path, parameters={"free": ["inertial.*.mass", "inertial.*.com_*",
                                                       "sensor.ft.*.offset@*"]})
    problem = _problem(config, run, base)
    # A-13: offsets of runs without a training bag are not free at all.
    assert "sensor.ft.iiwa_A3.offset@A" not in problem.registry.free_names
    assert problem.registry.specs["sensor.ft.iiwa_A3.offset@A"].frozen
    theta = problem.registry.values()
    names = list(problem.registry.free_names)
    block = ida.analyse(problem.data_jacobian_u(theta, names), names, prior_weight=problem.loss.prior_weight)
    rows = {r["name"]: r for r in block["parameters"]}
    # Link 1's COM along the A1 axis: nothing to see.
    assert rows["inertial.iiwa_link_1.com_z"]["sensitivity"] == 0.0
    assert rows["inertial.iiwa_link_1.com_z"]["data_share_posterior"] < 1e-9
    assert rows["inertial.iiwa_link_1.com_z"]["prior_determined"]  # F-2: d_post < 0.9
    assert rows["sensor.ft.iiwa_A3.offset@S"]["data_share_posterior"] > 0.9
    frozen = {r["name"] for r in block["recommended_frozen"]}
    assert "inertial.iiwa_link_1.com_z" in frozen
    assert block["collinearity_index_after_freeze"] <= 30.0
    # d_post <= d: correlations only take information away.
    for r in block["parameters"]:
        assert r["data_share_posterior"] <= r["data_share"] + 1e-9
    assert "| parameter |" in ida.to_markdown(block)


def test_a_nonphysical_theta_gets_a_logged_finite_loss(synthetic, base, tmp_path):
    import dataclasses

    root, path, run = synthetic
    problem = _problem(_config(root, path), run, base, history=tmp_path / "history.jsonl")
    good = problem.plant

    def broken(theta):
        plant = good(theta)
        link = plant.base.link_names[2]
        inertials = dict(plant.inertials)
        inertials[link] = dataclasses.replace(inertials[link], inertia=np.diag([1.0, 0.1, 0.1]), urdf_scale=None)
        return dataclasses.replace(plant, inertials=inertials)

    problem.plant = broken
    z = problem.registry.z0()
    evaluation = problem.evaluate(z)
    assert evaluation.loss == NONPHYSICAL_LOSS and evaluation.error and "triangle" in " ".join(evaluation.reasons)
    residual = problem.residuals(z)
    assert 0.5 * float(residual @ residual) == pytest.approx(NONPHYSICAL_LOSS)
    record = json.loads((tmp_path / "history.jsonl").read_text().splitlines()[-1])
    assert record["error"] and record["loss"] == NONPHYSICAL_LOSS
    assert problem.best is None  # never selected


def test_a12_freezes_exactly_one_member_of_a_collinear_pair():
    """S2R_06 P4a-1: two nearly collinear columns (only their sum is seen)
    and an independent one: A-12 freezes one member of the pair, at its
    freeze value, and keeps the other two."""
    from elastic_sim.sim2real import identifiability as ida

    rng = np.random.default_rng(0)
    a = rng.standard_normal(400)
    c = rng.standard_normal(400)
    b = a + 1e-4 * rng.standard_normal(400)
    columns = {"a": a, "b": b, "c": c}
    calls = []

    def jacobian_at(theta, names):
        calls.append((dict(theta), list(names)))
        return np.column_stack([columns[n] for n in names])

    frozen, theta = ida.freeze_plan(jacobian_at, {"a": 0.3, "b": -0.2, "c": 0.1}, ["a", "b", "c"],
                                    {"a": 0.0, "b": 0.5, "c": 0.0}, prior_weight=1e-2, d_post_min=0.9)
    assert len(frozen) == 1 and frozen[0]["name"] in ("a", "b") and frozen[0]["d_post"] < 0.9
    name = frozen[0]["name"]
    assert theta[name] == {"a": 0.0, "b": 0.5}[name] and theta["c"] == 0.1
    assert calls[-1][1] == [n for n in ("a", "b", "c") if n != name]  # d_post recomputed without it
    remaining = ida.posterior_share(np.column_stack([columns[n] for n in calls[-1][1]]), 1e-2)
    assert np.all(remaining >= 0.9)


def test_label_marks_both_members_of_a_collinear_pair_as_partners():
    """S2R_08 P4a-1' (1): a, b ~ a + eps and an independent c. One d_post on
    the full set labels both a and b not calibrated, each the other's
    partner; c is calibrated. Nothing is recomputed, so nothing moves."""
    from elastic_sim.sim2real import identifiability as ida

    rng = np.random.default_rng(0)
    a = rng.standard_normal(400)
    c = rng.standard_normal(400)
    b = a + 1e-4 * rng.standard_normal(400)
    block = ida.analyse(np.column_stack([a, b, c]), ["a", "b", "c"], prior_weight=1e-2)
    labelled = ida.labels(block, d_post_min=0.9)
    assert not labelled["a"]["calibrated"] and not labelled["b"]["calibrated"] and labelled["c"]["calibrated"]
    assert [p["name"] for p in labelled["a"]["partners"]] == ["b"]
    assert [p["name"] for p in labelled["b"]["partners"]] == ["a"]
    assert labelled["a"]["partners"][0]["corr"] < -0.9  # only a + b is seen: the errors anti-correlate
    assert "partners" not in labelled["c"]
    # The sequential A-12 would have called the survivor calibrated.
    frozen, _ = ida.freeze_plan(lambda t, n: np.column_stack([{"a": a, "b": b, "c": c}[k] for k in n]),
                                {"a": 0.0, "b": 0.0, "c": 0.0}, ["a", "b", "c"], {"a": 0.0, "b": 0.0, "c": 0.0},
                                prior_weight=1e-2)
    assert len(frozen) == 1


def test_session_zero_fits_only_the_test_run_offsets(synthetic, base):
    """S2R_06 P4a-1 / A-13: a second session T = the same data with its
    sensor shifted. T trains nothing; after the stage fit only offset@T
    moves (fitted on T's holds), to the training offsets + the shift."""
    import dataclasses

    root, path, run = synthetic
    shift = np.array([0.3, -0.4, 0.2, 0.25, -0.1, 0.05, -0.08])
    twin = dataclasses.replace(run, run_id="T", bags=tuple(dataclasses.replace(b, run_id="T", ft=b.ft + shift)
                                                             for b in run.bags))
    config = _config(root, path, runs={"S": {"path": str(path)}, "T": {"path": str(path)}},
                     split={"train": ["S:excitation_0", "S:identify_hold_*"], "val": ["S:excitation_1"],
                            "zero": ["T:identify_hold_*"], "test": ["T:identify_excitation_0", "T:excitation_*"]},
                     parameters={"free": ["inertial.*.mass", "sensor.ft.*.offset@*"]},
                     identifiability={"mode": "label"})
    registry = Registry.from_config(config, base)
    assert not any(n.endswith("@T") for n in registry.free_names)  # excluded automatically
    folder = calibrate(config, runs={"S": run, "T": twin}, base=base, run_id="zero", figures=False)
    metrics = json.loads((folder / "metrics.json").read_text())
    zero = metrics["session_zero"]
    assert sorted(zero["names"]) == sorted(f"sensor.ft.{j}.offset@T" for j in base.joint_names)
    selected = next(json.loads(x) for x in (folder / "history.jsonl").read_text().splitlines()[1:]
                    if json.loads(x)["i"] == metrics["summary"]["best_index"])
    best = metrics["best_theta"]
    for name, value in selected["theta"].items():
        assert best[name] == value, name  # nothing but offset@T changed
    # The same zero on S's own holds gives S's session offsets; T's = those + the shift.
    from elastic_sim.sim2real.calibration import session_zero
    from elastic_sim.sim2real.replay_r0 import R0Engine, prepare_bags

    problem = CalibrationProblem.from_config(config, registry, base, {"S": run, "T": twin})
    holds = R0Engine(base, prepare_bags([b for b in run.bags if b.kind == "hold"], config.replay.r0, time_step=1e-3))
    own, _ = session_zero(problem, holds, best, [f"sensor.ft.{j}.offset@S" for j in base.joint_names])
    checked = 0
    for j, joint in enumerate(base.joint_names):
        name = f"sensor.ft.{joint}.offset@T"
        spec = registry.specs[name]
        expected = own[f"sensor.ft.{joint}.offset@S"] + shift[j]
        if not spec.lower + 0.01 < expected < spec.upper - 0.01 or abs(own[f"sensor.ft.{joint}.offset@S"]) >= \
                spec.upper - spec.mean - 0.01:
            assert name in zero["at_bound"] or spec.lower < best[name] < spec.upper  # reported, not compared
            continue
        assert best[name] == pytest.approx(expected, abs=1e-4), joint
        checked += 1
    assert checked >= 4
    entries = yaml.safe_load((folder / "best.yaml").read_text())["parameters"]
    assert entries["sensor.ft.iiwa_A2.offset@T"]["calibrated"] and "session zero" in entries[
        "sensor.ft.iiwa_A2.offset@T"]["calibrated_by"]
    assert not entries["sensor.ft.iiwa_A2.offset@A"]["calibrated"]  # no data, no zero: not calibrated
    # The strict column replays T's test bags with S's offsets.
    strict = metrics["best_strict"]["test"]["torque"]["per_joint_rmse"]
    zeroed = metrics["best"]["test"]["torque"]["per_joint_rmse"]
    assert metrics["best_strict"]["offsets_from"] == "S" and strict != zeroed


def test_reduced_set_sensitivity_and_candidate(synthetic, base):
    """S2R_06 P4a-3/4/5 tooling: a comparator on a run's A-12 reduced set,
    the lambda-sensitivity table and the candidate overlay."""
    from elastic_sim.sim2real.report import sensitivity, write_candidate

    root, path, run = synthetic
    config = _config(root, path, optimizer={"method": "least_squares", "budget": 20})
    stage = calibrate(config, runs={"S": run}, base=base, run_id="stage", figures=False)
    entries = yaml.safe_load((stage / "best.yaml").read_text())["parameters"]
    calibrated = sorted(n for n, e in entries.items() if e["calibrated"])
    reduced = calibrate(config, method="random", seed=1, budget=6, runs={"S": run}, base=base, run_id="reduced",
                        figures=False, reduced_from=stage)
    registry = Registry.from_yaml((reduced / "registry.yaml").read_text())
    assert sorted(registry.free_names) == calibrated
    for name in set(entries) - set(calibrated):  # frozen at the stage run's values
        assert registry.specs[name].frozen and registry.specs[name].value == entries[name]["value"]
    metrics = json.loads((reduced / "metrics.json").read_text())
    assert len(metrics["summary"]["phases"]) == 1  # A-12 off on a fixed set
    reduced_entries = yaml.safe_load((reduced / "best.yaml").read_text())["parameters"]
    assert set(reduced_entries) == set(entries)
    assert sorted(n for n, e in reduced_entries.items() if e["calibrated"]) == calibrated

    import dataclasses

    other = dataclasses.replace(config, loss=dataclasses.replace(config.loss, prior_weight_factor=0.1),
                                resolved={**config.resolved, "loss": {"prior_weight_factor": 0.1}})
    other_folder = calibrate(other, runs={"S": run}, base=base, run_id="stage_x0.1", figures=False)
    table, payload = sensitivity(stage, {0.1: other_folder})
    assert payload["factors"] == [0.01, 0.1] and "| test RMSE (zeroed) |" in table
    assert all(isinstance(f, str) for f in payload["flags"])

    candidate = write_candidate(stage)
    text = yaml.safe_load(candidate.read_text())
    assert text["kind"] == "candidate" and text["n_parameters"] == len(entries)
    assert text["n_calibrated"] == len(calibrated) and text["parameters"] == entries
