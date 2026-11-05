"""identified_filter/02 round 2: the r8 campaign's orchestration (no training, no data)."""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "r8_campaign.py"


@pytest.fixture()
def campaign(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("r8_campaign_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "RESULTS", tmp_path / "results")
    run = tmp_path / "run"
    run.mkdir()
    monkeypatch.setattr(module, "run_dir", lambda settings: run)
    yield module
    sys.modules.pop(spec.name, None)


def _stage(job, priority, hours, ok=True, platform="ur10", stage="P"):
    return {"job": job, "priority": priority, "platform": platform, "stage": stage, "wall_hours": hours,
            "smoke_ok": ok}


def test_schedule_drops_smoke_failures_with_reason_then_the_tail(campaign):
    start = dt.datetime(2026, 10, 30, 18, 0)
    stages = [_stage("J1", 1, 30.0), _stage("J2", 1, 10.0, ok=False), _stage("J5", 2, 10.0),
              _stage("J8", 3, 10.0), _stage("J9", 3, float("nan"))]
    dropped, reasons, total = campaign.schedule(stages, start, start + dt.timedelta(hours=42), "r8")
    assert dropped[0] == "J2:ur10" and "smoke test failed on stage(s) P" in reasons["J2:ur10"]
    # 30 + 10 + 10 = 50 h > 42 h: drop from the tail (J9, J8: P3), stop once it fits (J5 would fit at 40 h).
    assert set(dropped) == {"J2:ur10", "J9:ur10", "J8:ur10"}
    assert reasons["J8:ur10"].startswith("past the deadline")
    assert total == pytest.approx(40.0)
    assert stages[2]["cumulative_hours"] == pytest.approx(40.0)


def test_schedule_never_drops_p1_for_time(campaign):
    start = dt.datetime(2026, 10, 30, 18, 0)
    stages = [_stage("J1", 1, 100.0), _stage("J5", 2, 1.0)]
    dropped, _, _ = campaign.schedule(stages, start, start + dt.timedelta(hours=10), "r8")
    assert dropped == ["J5:ur10"]


def test_all_goes_on_after_gate_exit_and_plan_failure(campaign, monkeypatch, tmp_path):
    calls = []
    contract = tmp_path / "prod.contract.json"
    contract.write_text("{}")
    monkeypatch.setattr(campaign, "cmd_data", lambda s, a: calls.append("data") or 1)
    monkeypatch.setattr(campaign, "cmd_plan", lambda s, a: calls.append("plan") or 1)
    monkeypatch.setattr(campaign, "cmd_run", lambda s, a: calls.append("run") or 0)
    monkeypatch.setattr(campaign, "platform_paths", lambda s, p: {"production": tmp_path / "prod.parquet"})
    settings = {"platforms": {"ur10": {}}, "run_id": "t"}
    assert campaign.cmd_all(settings, types.SimpleNamespace()) == 0
    assert calls == ["data", "plan", "run"]
    progress = (campaign.RESULTS / "progress.md").read_text()
    assert "all: data" in progress and "all: plan" in progress and "all: run" in progress


def test_all_stops_without_a_contract(campaign, monkeypatch, tmp_path):
    monkeypatch.setattr(campaign, "cmd_data", lambda s, a: 1)
    monkeypatch.setattr(campaign, "cmd_plan", lambda s, a: pytest.fail("plan must not run"))
    monkeypatch.setattr(campaign, "platform_paths", lambda s, p: {"production": tmp_path / "missing.parquet"})
    assert campaign.cmd_all({"platforms": {"ur10": {}}, "run_id": "t"}, types.SimpleNamespace()) == 1


def test_all_trains_the_platforms_that_built_and_skips_the_others(campaign, monkeypatch, tmp_path):
    """03 F5: a failed build costs that platform, not the campaign."""
    (tmp_path / "ur10.contract.json").write_text("{}")
    seen = []
    monkeypatch.setattr(campaign, "cmd_data", lambda s, a: 1)
    monkeypatch.setattr(campaign, "cmd_plan", lambda s, a: seen.append(sorted(s["platforms"])) or 0)
    monkeypatch.setattr(campaign, "cmd_run", lambda s, a: seen.append(sorted(s["platforms"])) or 0)
    monkeypatch.setattr(campaign, "platform_paths", lambda s, p: {"production": tmp_path / f"{p}.parquet"})
    settings = {"platforms": {"ur10": {}, "iiwa_drive": {}}, "run_id": "t"}
    assert campaign.cmd_all(settings, types.SimpleNamespace()) == 0
    assert seen == [["ur10"], ["ur10"]]
    log = (campaign.run_dir(settings) / "campaign.log").read_text()
    assert "SKIP  all: no r8 production contract for ['iiwa_drive']; training on ['ur10']" in log


def _fake_quiet(inhibit_ok, masked):
    def quiet(command):
        if command[0] == "systemd-inhibit":
            return subprocess.CompletedProcess(command, 0 if inhibit_ok else 1, "", "Failed to inhibit: Access denied")
        if command[0] == "systemctl":
            return subprocess.CompletedProcess(command, 0, "masked\n" if masked else "loaded\n", "")
        return subprocess.CompletedProcess(command, 0, "Linger=no\n", "")
    return quiet


# systemd-inhibit is never used: polkit denies it inside the --user unit even when the terminal allows it.
@pytest.mark.parametrize("inhibit_ok,masked,expect", [(True, False, "refuse"), (True, True, "masked"),
                                                       (False, True, "masked"), (False, False, "refuse")])
def test_sleep_guard(campaign, monkeypatch, inhibit_ok, masked, expect):
    monkeypatch.setattr(campaign.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(campaign, "_quiet", _fake_quiet(inhibit_ok, masked))
    prefix, message = campaign.sleep_guard()
    if expect == "masked":
        assert prefix == [] and "masked: OK" in message
    else:
        assert prefix is None and "sudo systemctl mask sleep.target" in message
    assert campaign.linger_check().startswith("WARNING")


def test_launch_runs_all_by_default_and_refuses_without_sleep_safety(campaign, monkeypatch, capsys):
    monkeypatch.setattr(campaign.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(campaign, "_quiet", _fake_quiet(False, False))
    launched = []
    monkeypatch.setattr(campaign.subprocess, "call", lambda command: launched.append(command) or 0)
    settings = {"unit": "r8-campaign", "run_id": "t"}
    args = types.SimpleNamespace(only_run=False, dry_run=False, config=str(campaign.CONFIG))
    assert campaign.cmd_launch(settings, args) == 1 and not launched
    monkeypatch.setattr(campaign, "_quiet", _fake_quiet(False, True))
    assert campaign.cmd_launch(settings, args) == 0
    assert launched[-1][launched[-1].index(str(SCRIPT)) + 1] == "all"
    args.only_run = True
    campaign.cmd_launch(settings, args)
    assert launched[-1][launched[-1].index(str(SCRIPT)) + 1] == "run"
    assert "journalctl --user -u r8-campaign -f" in capsys.readouterr().out


def test_run_skips_p2_stages_after_the_deadline_but_not_p1(campaign, monkeypatch):
    started = []

    def fake_stage(settings, run, job_name, job, platform, seed, stage, dataset, timeout_s, epochs=None):
        started.append(job_name)
        return {"rmse": "1.0", "checkpoint_path": "x"}

    monkeypatch.setattr(campaign, "run_stage", fake_stage)
    monkeypatch.setattr(campaign, "stage_runs", lambda s, j, p: [{"name": "R"}])
    monkeypatch.setattr(campaign, "evaluate_job", lambda *a: None)
    monkeypatch.setattr(campaign, "nvidia_logger", lambda run, stop: None)
    monkeypatch.setattr(campaign, "free_gb", lambda path: 1e6)
    monkeypatch.setattr(campaign, "platform_paths", lambda s, p: {"production": Path("x.parquet")})
    settings = {"run_id": "t", "deadline": "2000-01-01T00:00", "parallel_seeds": 2, "seeds": [0], "min_free_gb": 50,
                "platforms": {"ur10": {}}, "jobs": {"J4": {"priority": 1}, "J5": {"priority": 2}}}
    assert campaign.cmd_run(settings, types.SimpleNamespace(job=["J4", "J5"])) == 0
    assert started == ["J4"]
    log = (campaign.run_dir(settings) / "campaign.log").read_text()
    assert "SKIP  J5 ur10 s0 R: would start after the deadline" in log


def test_config_budget_is_r8_sized(campaign):
    settings = campaign.load_settings()
    assert (settings["epochs"], settings["patience"]) == (150, 20)
    assert (settings["student_epochs"], settings["student_patience"]) == (40, 8)
    assert settings["privileged_source"] == "effective"
    json.dumps(settings)


def test_stage_timeout_scales_with_parallel_seeds_and_has_a_floor(campaign):
    # 2026-10-31: 1.5 x a one-seed estimate killed three J1 seeds sharing the GPU.
    assert campaign.stage_timeout_s({"parallel_seeds": 3}, 1.0) == 12 * 3600.0
    assert campaign.stage_timeout_s({"parallel_seeds": 3}, 4.0) == 1.5 * 4.0 * 3 * 3600.0
    assert campaign.stage_timeout_s({"parallel_seeds": 1, "min_stage_timeout_h": 2}, 1.0) == 2 * 3600.0


def _done(campaign, run, job, stage, **row):
    done = campaign.stage_done(run, job, "ur10", 0, stage)
    done.parent.mkdir(parents=True, exist_ok=True)
    done.write_text(json.dumps(row))


def test_student_skips_a_teacher_at_the_zero_predictor(campaign):
    run = campaign.run_dir({})
    student = {"name": "S", "init": "J1"}
    assert "no result" in campaign.teacher_failure(run, "J3", "ur10", 0, student)
    _done(campaign, run, "J1", "T", best_val_rmse="25.3", val_zero_rmse="25.4")
    assert "zero predictor" in campaign.teacher_failure(run, "J3", "ur10", 0, student)
    _done(campaign, run, "J1", "T", best_val_rmse="5.9", val_zero_rmse="25.4")
    assert campaign.teacher_failure(run, "J3", "ur10", 0, student) is None
    _done(campaign, run, "J1", "T", best_val_rmse="25.3")   # a run predating val_zero_rmse is not judged
    assert campaign.teacher_failure(run, "J3", "ur10", 0, student) is None
    # An own-teacher job reads its own T; a stage continuing its student, or with no init, has no teacher.
    assert "J9" in campaign.teacher_failure(run, "J9", "ur10", 0, {"name": "S", "init": "T"})
    assert campaign.teacher_failure(run, "J3", "ur10", 0, {"name": "J", "init": "S"}) is None
    assert campaign.teacher_failure(run, "J2", "ur10", 0, {"name": "P"}) is None


def test_failed_teacher_means_no_student_stage(campaign, monkeypatch):
    started = []

    def fake_stage(settings, run, job_name, job, platform, seed, stage, dataset, timeout_s, epochs=None):
        started.append(stage["name"])
        return {"rmse": "1.0", "checkpoint_path": "x"}

    monkeypatch.setattr(campaign, "run_stage", fake_stage)
    monkeypatch.setattr(campaign, "stage_runs", lambda s, j, p: [{"name": "S", "init": "J1"}, {"name": "J", "init": "S"}])
    monkeypatch.setattr(campaign, "evaluate_job", lambda *a: None)
    monkeypatch.setattr(campaign, "nvidia_logger", lambda run, stop: None)
    monkeypatch.setattr(campaign, "free_gb", lambda path: 1e6)
    monkeypatch.setattr(campaign, "platform_paths", lambda s, p: {"production": Path("x.parquet")})
    settings = {"run_id": "t", "deadline": "2999-01-01T00:00", "parallel_seeds": 1, "seeds": [0], "min_free_gb": 50,
                "platforms": {"ur10": {}}, "jobs": {"J3": {"priority": 1}}}
    assert campaign.cmd_run(settings, types.SimpleNamespace(job=["J3"])) == 0
    assert started == []
    assert "SKIP  J3 ur10 s0 S: teacher J1 s0 left no result" in (campaign.run_dir(settings) / "campaign.log").read_text()


def test_an_aborted_stage_is_not_retried(campaign, monkeypatch, tmp_path):
    calls = []

    class Finished:
        pid = 1

        def __init__(self, command, **kwargs):
            calls.append(command)

        def wait(self, timeout=None):
            return campaign.STAGE_ABORT_EXIT

    run = campaign.run_dir({})
    monkeypatch.setattr(campaign.subprocess, "Popen", Finished)
    monkeypatch.setattr(campaign, "write_chain", lambda *a: tmp_path / "chain.yaml")
    row = campaign.run_stage({}, run, "J1", {}, "ur10", 0, {"name": "T"}, Path("x"), 10.0)
    assert row is None and len(calls) == 1
    assert campaign.stage_done(run, "J1", "ur10", 0, "T").with_suffix(".failed").exists()
    assert "zero predictor, not retried" in (run / "campaign.log").read_text()


def test_identified_learning_rate_is_halved_on_the_ur10_only(campaign):
    settings = campaign.load_settings()
    job = {"kind": "phys"}
    ur10 = campaign.stage_runs(settings, job, "ur10")[0]["overrides"]
    iiwa = campaign.stage_runs(settings, job, "iiwa_drive")[0]["overrides"]
    r7, _ = campaign.r7_runs(settings, "ur10")
    assert ur10["learning_rate"] == pytest.approx(0.5 * r7["kalnn_twomass"]["training_overrides"]["learning_rate"])
    r7, _ = campaign.r7_runs(settings, "iiwa_drive")
    assert iiwa["learning_rate"] == pytest.approx(r7["kalnn_twomass"]["training_overrides"]["learning_rate"])
    for overrides in (ur10, iiwa):
        assert (overrides["grad_clip_norm"], overrides["fail_fast_epochs"]) == (1.0, 10)
    residual = campaign.stage_runs(settings, {"kind": "r7_residual"}, "ur10")[0]["overrides"]
    assert "grad_clip_norm" not in residual and "fail_fast_epochs" not in residual


def test_modal_file_gets_no_privileged_evaluation(campaign, monkeypatch, tmp_path):
    names = {k: tmp_path / f"{k}.parquet" for k in ("production", "gainshift", "oop", "modal")}
    for path in names.values():
        path.touch()
    commands = []
    monkeypatch.setattr(campaign, "platform_paths", lambda s, p: names)
    monkeypatch.setattr(campaign, "contract_kwargs", lambda s, p: {"burn_in_s": 1.0})
    monkeypatch.setattr(campaign.subprocess, "call", lambda command, **kw: commands.append(command) or 0)
    campaign.evaluate_job({}, campaign.run_dir({}), "J1", "ur10", {0: {"checkpoint_path": "c", "base_name": "J1_T"}})
    privileged = [c[c.index("--dataset") + 1] for c in commands if "--privileged" in c]
    assert sorted(Path(p).stem for p in privileged) == ["gainshift", "oop", "production"]
    assert any(Path(c[c.index("--dataset") + 1]).stem == "modal" for c in commands)


def test_r9_step2_kwargs_carry_the_plant_curve_the_warm_cutoff_and_the_urdf(campaign):
    """identified_filter/07 Sec 2: the step-2 filter's kwargs from the r9 config; r8 has none."""
    root = SCRIPT.parents[1] / "config" / "campaigns"
    r9 = campaign.load_settings(root / "r9_campaign.yaml")
    r8 = campaign.load_settings(root / "r8_campaign.yaml")
    assert campaign.step2_kwargs(r8, "ur10") == {}
    kwargs = campaign.step2_kwargs(r9, "ur10")
    assert (kwargs["spring"], kwargs["inertia"], kwargs["damping"]) == ("tangent", "urdf", "zeta")
    assert kwargs["spring_knees"][1] == [29.0, 108.0] and kwargs["spring_knees"][2] == [14.0, 48.0]   # size 32 / 25
    assert kwargs["spring_factors"][1] == [0.69, 1.0, 1.44]
    assert abs(kwargs["spring_warm_cutoff_hz"] - 0.5 * 4.18) < 0.01   # 0.5 x the pan's f_n,min (band table)
    assert len(kwargs["urdf_chain"]["joints"]) == 6 and kwargs["lambda_inertia"] is None
    assert kwargs["stiffness_widening"] == 1.0 and kwargs["alpha_range"] == [0.7, 1.4]
    assert set(r9["jobs"]) == {"J1", "J2", "J3", "J11", "J5"} and r9["privileged_source"] == "sampled"
    assert r9["results_dir"] != r8.get("results_dir")


def test_nothing_starts_after_stop_at_whatever_the_priority(campaign, monkeypatch):
    """09 Sec 2: past the hard stop no stage starts, P1 included, and the final sweep still runs."""
    started, sweep = [], []

    def fake_stage(settings, run, job_name, job, platform, seed, stage, dataset, timeout_s, epochs=None):
        started.append(job_name)
        return {"rmse": "1.0", "checkpoint_path": "x"}

    monkeypatch.setattr(campaign, "run_stage", fake_stage)
    monkeypatch.setattr(campaign, "stage_runs", lambda s, j, p: [{"name": "T"}])
    monkeypatch.setattr(campaign, "evaluate_job", lambda *a: None)
    monkeypatch.setattr(campaign, "nvidia_logger", lambda run, stop: None)
    monkeypatch.setattr(campaign, "free_gb", lambda path: 1e6)
    monkeypatch.setattr(campaign, "platform_paths", lambda s, p: {k: Path(f"missing-{k}.parquet") for k in
                                                                  ("production", "gainshift")})
    monkeypatch.setattr(campaign.subprocess, "call", lambda command, **kw: sweep.append(command) or 0)
    settings = {"run_id": "t", "deadline": "2999-01-01T00:00", "stop_at": "2000-01-01T00:00", "parallel_seeds": 1,
                "seeds": [0], "min_free_gb": 50, "platforms": {"ur10": {}}, "jobs": {"J1": {"priority": 1}}}
    assert campaign.cmd_run(settings, types.SimpleNamespace(job=None)) == 0
    assert started == []
    log = (campaign.run_dir(settings) / "campaign.log").read_text()
    assert "SKIP  J1 ur10 s0 T: would start after stop_at 2000-01-01 00:00:00" in log
    assert any("run_real_eval.sh" in " ".join(map(str, c)) for c in sweep)
    assert any("r8_summary.py" in " ".join(map(str, c)) for c in sweep)


def test_a_running_stage_gets_stop_at_and_a_capped_wait_and_no_retry_after_it(campaign, monkeypatch, tmp_path):
    seen = []

    class Killed:
        pid = 1

        def __init__(self, command, **kwargs):
            seen.append(kwargs["env"][campaign.STOP_AT_ENV])

        def wait(self, timeout=None):
            seen.append(timeout)
            return 1

    run = campaign.run_dir({})
    monkeypatch.setattr(campaign.subprocess, "Popen", Killed)
    monkeypatch.setattr(campaign, "write_chain", lambda *a: tmp_path / "chain.yaml")
    soon = (dt.datetime.now() + dt.timedelta(hours=2)).replace(microsecond=0)
    settings = {"stop_at": soon.isoformat(), "stop_grace_h": 0.5}
    assert campaign.run_stage(settings, run, "J1", {}, "ur10", 0, {"name": "T"}, Path("x"), 10 * 3600.0) is None
    assert seen[0] == soon.isoformat()
    assert 2.4 * 3600 < seen[1] <= 2.5 * 3600        # stop_at + the grace, not the 10 h timeout
    assert len(seen) == 4                             # before stop_at a failed stage is still retried once
    past = {"stop_at": "2000-01-01T00:00"}
    seen.clear()
    campaign.run_stage(past, run, "J1", {}, "ur10", 0, {"name": "T"}, Path("x"), 10 * 3600.0)
    assert seen[1] == 60.0 and len(seen) == 2       # the floor, and no second attempt
    assert "no retry after stop_at" in (run / "campaign.log").read_text()
    assert campaign.capped_timeout_s({}, 123.0) == 123.0


def test_r9_config_has_the_09_deadline_and_hard_stop(campaign):
    settings = campaign.load_settings(SCRIPT.parents[1] / "config" / "campaigns" / "r9_campaign.yaml")
    assert campaign.stop_at_of(settings) == dt.datetime(2026, 11, 4, 11, 0)
    assert dt.datetime.fromisoformat(settings["deadline"]) == dt.datetime(2026, 11, 4, 6, 0)
