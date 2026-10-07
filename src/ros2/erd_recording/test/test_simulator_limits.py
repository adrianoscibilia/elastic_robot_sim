"""RR_14 P-2b/c: the simulator-limits list marks, but doesn't count, the
failures a simulator can't avoid, and is never consulted on real hardware;
D-4's clock ratio is 1.0 everywhere but URSim."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from erd_recording.validate import SIMULATOR_LIMITS, finalize_checks, simulator_limit

ROBOT = "ur10_cb3"


def _checks(**failing: bool) -> list[dict]:
    names = ("uniform grid", "tau != ft", "sign convention", "session health", "noise")
    return [{"check": name, "ok": not failing.get(name.replace(" ", "_").replace("!=", "ne"), False)}
            for name in names]


@pytest.mark.parametrize("hardware, failing, listed", [
    ("mock", {"tau_ne_ft": True, "sign_convention": True}, ["tau != ft", "sign convention"]),
    ("ursim", {"tau_ne_ft": True}, ["tau != ft"]),
])
def test_listed_failures_are_marked_and_do_not_fail(hardware, failing, listed):
    result = finalize_checks(_checks(**failing), hardware=hardware, robot=ROBOT)
    assert result["ok"] and result["errors"] == []
    assert result["simulator_limits"] == listed
    for check in result["checks"]:
        if check["check"] in listed:
            assert check["ok"] is False
            assert check["simulator_limit"] == SIMULATOR_LIMITS[(hardware, check["check"])]


def test_unlisted_failures_still_fail_on_a_simulator():
    result = finalize_checks(_checks(sign_convention=True), hardware="ursim", robot=ROBOT)
    assert not result["ok"] and result["errors"] == ["sign convention"]
    result = finalize_checks(_checks(tau_ne_ft=True), hardware="emulator", robot="kuka_lbr_iiwa_14_r820")
    assert not result["ok"] and result["errors"] == ["tau != ft"]  # the emulator has no list


def test_the_list_is_ignored_on_real_hardware():
    for hardware, check in SIMULATOR_LIMITS:
        assert simulator_limit(check, "real") is None
    result = finalize_checks(_checks(tau_ne_ft=True, sign_convention=True), hardware="real", robot=ROBOT)
    assert not result["ok"]
    assert result["errors"] == ["tau != ft", "sign convention"]
    assert result["simulator_limits"] == []
    assert all("simulator_limit" not in check for check in result["checks"])


def test_passing_checks_are_never_marked():
    result = finalize_checks(_checks(), hardware="mock", robot=ROBOT)
    assert result["ok"] and result["simulator_limits"] == []


@pytest.mark.parametrize("hardware, expected", [("real", 1.0), ("mock", 1.0), ("emulator", 1.0), ("ursim", 0.937)])
def test_clock_ratio_only_on_ursim(tmp_path, hardware, expected):
    from erd_recording.identify_cli import robot_clock_ratio

    (tmp_path / "rtde_alignment.json").write_text(json.dumps({"wall_seconds_per_robot_second": 1.0 / 0.937}))
    assert robot_clock_ratio(SimpleNamespace(hardware=hardware), tmp_path) == pytest.approx(expected)
