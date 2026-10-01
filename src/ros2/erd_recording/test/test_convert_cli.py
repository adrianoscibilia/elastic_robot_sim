"""RR_04 A-8: the pure window-refinement piece of convert_cli, unit-tested
without a bag or ROS (the bag-reading wrappers themselves need rosbag2_py
and are only exercised live, not in this suite -- see convert_cli.py's
module docstring)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from erd_recording.convert_cli import _refine_window

N_DOF = 3
TIME_STEP = 0.001


def _synthetic_raw(n_samples: int) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    t = np.arange(n_samples) * TIME_STEP
    columns = {"t": t}
    for i in range(N_DOF):
        # A settle-then-move signal: near zero, then a plan-shaped excerpt,
        # then settle again -- like a real excitation window inside a longer
        # continuous recording.
        columns[f"commanded_position{i}"] = 0.001 * rng.normal(size=n_samples)
    return pd.DataFrame(columns)


def test_refine_window_recovers_the_exact_planned_window():
    raw = _synthetic_raw(400)
    plan_length = 100
    true_start = 150
    # A handful of full cycles across the window (genuinely zero-mean, like a
    # real Fourier-series excitation segment) -- a near-DC slice of a slow
    # sine, as an earlier version of this test used, defeats the
    # demeaning cross-correlation the same way it would defeat a real
    # analyst reading only the tangent of a curve at one point.
    plan_position = np.zeros((plan_length, N_DOF))
    for i in range(N_DOF):
        plan_position[:, i] = np.sin(2 * np.pi * (5 + i) * np.arange(plan_length) / plan_length)
    for i in range(N_DOF):
        raw.loc[true_start:true_start + plan_length - 1, f"commanded_position{i}"] = plan_position[:, i]

    # A deliberately imprecise event-based window (RR_04 A-8: events only
    # bracket approximately; refine_window's cross-correlation must recover
    # the exact one).
    approximate_window = (true_start - 5, true_start + plan_length + 7)
    refined = _refine_window(raw, approximate_window, plan_position, N_DOF, pad_samples=20)
    assert refined == (true_start, true_start + plan_length)


def test_refine_window_falls_back_when_padding_is_insufficient():
    raw = _synthetic_raw(50)
    plan_position = np.zeros((100, N_DOF))  # longer than the whole available frame
    window = (0, 50)
    refined = _refine_window(raw, window, plan_position, N_DOF, pad_samples=5)
    assert refined == window  # can't refine: falls back to the event-based window unchanged
