"""Schema checks for local KUKA/Baxter ground-truth adapters."""

import os
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(_REPO / "src"))

from elastic_sim.asset_dataset import load_asset_rollouts, rollout_to_dataframe


def test_baxter_csv_loader_builds_timestamped_rollout():
    source = _REPO / "assets" / "datasets" / "baxter" / "raw" / "left_circle_p-15_t105.csv"
    if not source.is_file():
        pytest.skip("local Baxter dataset is not installed")
    rollout = load_asset_rollouts(source, dataset_format="baxter_csv", sample_time_s=0.002)[0]
    assert rollout.q.shape == rollout.dq.shape == rollout.tau.shape
    assert rollout.q.shape[1] == 7
    frame = rollout_to_dataframe(rollout, bag="sample")
    assert {"time", "q0", "dq0", "tau0", "bag"}.issubset(frame)


def test_kuka_mat_loader_preserves_motor_and_link_encoders():
    pytest.importorskip("scipy")
    source = _REPO / "assets" / "datasets" / "kuka_kr300" / "raw" / "raw_data" / "recording_2021_12_15_20H_29M.mat"
    if not source.is_file():
        pytest.skip("local KUKA dataset is not installed")
    rollout = load_asset_rollouts(source, dataset_format="kuka_raw_mat")[0]
    assert rollout.q.shape == rollout.motor_q.shape == (90881, 6)
    assert rollout.link_q is not None and rollout.link_dq is not None
    assert rollout.time[1] - rollout.time[0] == pytest.approx(0.004)
