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


# ---------------------------------------------------------------------------
# RR_16 Q-3 (F-6, D-7): fail closed on `hardware`
# ---------------------------------------------------------------------------


def test_validate_dataset_requires_hardware():
    import inspect

    from erd_recording.validate import validate_dataset

    assert inspect.signature(validate_dataset).parameters["hardware"].default is inspect.Parameter.empty
    with pytest.raises(TypeError, match="hardware"):
        validate_dataset(None, n_dof=1, nominal_dt=0.001, position_lower=None, position_upper=None,  # type: ignore[arg-type]
                         quantization=1e-7, tau_source="a", ft_source="b", effort_limit=None)


@pytest.mark.parametrize("hardware", ["Real", "sim", "", None, "urSim"])
def test_unknown_hardware_is_refused(hardware):
    with pytest.raises(ValueError, match="hardware must be one of"):
        finalize_checks(_checks(), hardware=hardware, robot=ROBOT)
    with pytest.raises(ValueError, match="hardware must be one of"):
        simulator_limit("tau != ft", hardware)


def test_unknown_hardware_is_refused_by_validate_dataset_before_any_check():
    from erd_recording.validate import validate_dataset

    with pytest.raises(ValueError, match="hardware must be one of"):
        validate_dataset(None, n_dof=1, nominal_dt=0.001, position_lower=None, position_upper=None,  # type: ignore[arg-type]
                         quantization=1e-7, tau_source="a", ft_source="b", effort_limit=None, hardware="sim")


def _drifting_window(local=0.926, n=581):
    import numpy as np
    import pandas as pd

    return pd.DataFrame({"timestamp": np.arange(n) * 0.008,
                         "stamp_ns": np.rint(np.arange(n) * 0.008 / local * 1e9).astype(np.int64)})


def test_window_ratio_used_on_ursim():
    from erd_recording.identify_cli import window_clock_ratio

    assert window_clock_ratio(_drifting_window(), 0.939, hardware="ursim") == pytest.approx(0.926)


@pytest.mark.parametrize("hardware", ["real", "mock", "emulator"])
def test_window_ratio_is_one_elsewhere_even_with_an_alignment_file(tmp_path, hardware):
    from erd_recording.identify_cli import robot_clock_ratio, window_clock_ratio

    (tmp_path / "rtde_alignment.json").write_text(json.dumps({"wall_seconds_per_robot_second": 1.0 / 0.937}))
    config = SimpleNamespace(hardware=hardware)
    assert robot_clock_ratio(config, tmp_path) == 1.0
    # even if a caller passed a non-1 run ratio, the gate is the hardware
    assert window_clock_ratio(_drifting_window(), 0.937, hardware=hardware) == 1.0


def test_window_ratio_refuses_unknown_hardware():
    from erd_recording.identify_cli import window_clock_ratio

    with pytest.raises(ValueError):
        window_clock_ratio(_drifting_window(), 0.937, hardware="sim")
