"""Parquet output must contain all required columns and non-empty data (INV-5 / O3)."""

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_REPO, "src"))

from elastic_sim.rollout import RolloutResult

REQUIRED_COLUMNS = [
    "t",
    "ref_x", "ref_y", "ref_z",
    "vel_x", "vel_y", "vel_z",
    "q_motor_x", "q_link_x",
    "q_motor_y", "q_link_y",
    "q_motor_z", "q_link_z",
    "dq_motor_x", "dq_link_x",
    "dq_motor_y", "dq_link_y",
    "dq_motor_z", "dq_link_z",
    "tau_motor_x", "tau_link_x",
    "tau_motor_y", "tau_link_y",
    "tau_motor_z", "tau_link_z",
]

N = 100


def _make_rollout(n=N):
    t = np.linspace(0, 1.0, n)
    arr3 = lambda: np.random.randn(n, 3)
    return RolloutResult(
        time=t,
        ref_pos=arr3(), ref_vel=arr3(),
        q_motor=arr3(), q_link=arr3(),
        dq_motor=arr3(), dq_link=arr3(),
        tau_motor=arr3(), tau_link=arr3(),
    )


def test_schema_all_columns_present():
    rollout = _make_rollout()
    df = rollout.to_dataframe()
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    assert not missing, f"Missing columns: {missing}"


def test_schema_non_empty():
    rollout = _make_rollout()
    df = rollout.to_dataframe()
    for col in ("t", "q_motor_x", "dq_motor_x", "tau_motor_x"):
        assert len(df[col]) > 0


def test_parquet_roundtrip():
    pytest.importorskip("pyarrow", reason="pyarrow not installed — skipping parquet roundtrip")
    rollout = _make_rollout()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "test.parquet")
        rollout.to_dataframe().to_parquet(path, index=False)
        df = pd.read_parquet(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    assert not missing, f"Missing columns after roundtrip: {missing}"
    assert len(df) == N
