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


def _fake_quiet(inhibit_ok, masked):
    def quiet(command):
        if command[0] == "systemd-inhibit":
            return subprocess.CompletedProcess(command, 0 if inhibit_ok else 1, "", "Failed to inhibit: Access denied")
        if command[0] == "systemctl":
            return subprocess.CompletedProcess(command, 0, "masked\n" if masked else "loaded\n", "")
        return subprocess.CompletedProcess(command, 0, "Linger=no\n", "")
    return quiet


@pytest.mark.parametrize("inhibit_ok,masked,expect", [(True, False, "inhibit"), (False, True, "masked"),
                                                       (False, False, "refuse")])
def test_sleep_guard(campaign, monkeypatch, inhibit_ok, masked, expect):
    monkeypatch.setattr(campaign.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(campaign, "_quiet", _fake_quiet(inhibit_ok, masked))
    prefix, message = campaign.sleep_guard()
    if expect == "inhibit":
        assert prefix[0] == "systemd-inhibit" and "sleep:idle:handle-lid-switch" in prefix[1]
    elif expect == "masked":
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
