"""Unit tests for the speed-override math (INV-3)."""

import numpy as np
from elastic_sim.trajectory import resolve_global_factor


def test_min_rule_user_stricter():
    # user wants 50%, limits allow 80% -> min = 0.5
    f = resolve_global_factor(
        0.625, {"x": 0.625, "y": 0, "z": 0},
        speed_override=50.0, v_lim_cart=0.5,
        v_lim_axis={"x": None, "y": None, "z": None},
    )
    assert abs(f - 0.5) < 1e-9


def test_min_rule_limit_stricter():
    # nominal peak 0.8, limit 0.5 -> auto 0.625; user 100% -> min = 0.625
    f = resolve_global_factor(
        0.8, {"x": 0.8, "y": 0, "z": 0},
        speed_override=100.0, v_lim_cart=0.5,
        v_lim_axis={"x": None, "y": None, "z": None},
    )
    assert abs(f - 0.625) < 1e-9


def test_never_speeds_up():
    # nominal peak below limit -> auto capped at 1.0
    f = resolve_global_factor(
        0.2, {"x": 0.2, "y": 0, "z": 0},
        speed_override=100.0, v_lim_cart=0.5,
        v_lim_axis={"x": None, "y": None, "z": None},
    )
    assert abs(f - 1.0) < 1e-9


def test_zero_velocity_safe():
    f = resolve_global_factor(
        0.0, {"x": 0, "y": 0, "z": 0},
        speed_override=100.0, v_lim_cart=0.5,
        v_lim_axis={"x": None, "y": None, "z": None},
    )
    assert f == 1.0  # no division by zero


def test_per_axis_limit_applies():
    f = resolve_global_factor(
        0.4, {"x": 0.4, "y": 0.1, "z": 0.1},
        speed_override=100.0, v_lim_cart=0.5,
        v_lim_axis={"x": 0.2, "y": None, "z": None},
    )
    assert abs(f - 0.5) < 1e-9  # x: 0.2/0.4 = 0.5 dominates
