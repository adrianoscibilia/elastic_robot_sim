"""T1.7: converter, contract, baselines and validation on synthetic fixtures."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.signal import savgol_filter

from elastic_sim import diagnostics as diag
from elastic_sim import identification as idn
from elastic_sim import round6_checks
from elastic_sim.assets import AssetRegistry

from erd_recording.contract import differentiation_from_reference, real_baselines, real_contract, write_real_contract
from erd_recording.convert import iiwa_dataset_frame, ur10_dataset_frame, write_real_dataset
from erd_recording.identification import MotorSideFit
from erd_recording.validate import check_saturation, check_tau_ne_ft, check_uniform_grid

SG_WINDOW = 11
SG_POLY = 3
TIME_STEP = 0.001
N_DOF = 7
CONSUMER_REPO = Path(os.environ["ERD_CONSUMER_REPO"])      # defaults set in conftest.py
CONSUMER_PYTHON = Path(os.environ["ERD_CONSUMER_PYTHON"])


def _synthetic_iiwa_raw(n_samples: int = 400, n_bags: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for bag_index in range(n_bags):
        t = np.arange(n_samples) * TIME_STEP
        phase = rng.uniform(0, 2 * np.pi, size=N_DOF)
        freq = rng.uniform(0.5, 1.5, size=N_DOF)
        q = 0.3 * np.sin(2 * np.pi * freq[None, :] * t[:, None] + phase[None, :])
        tau = rng.normal(scale=5.0, size=(n_samples, N_DOF))
        ft = tau + rng.normal(scale=0.5, size=(n_samples, N_DOF))  # correlated but non-identical
        for i in range(n_samples):
            row = {"t": t[i], "bag": f"bag_{bag_index}", "split": "test"}
            for j in range(N_DOF):
                row[f"q{j}"] = q[i, j]
                row[f"tau{j}"] = tau[i, j]
                row[f"ft{j}"] = ft[i, j]
            rows.append(row)
    return pd.DataFrame(rows)


def test_iiwa_dq_is_exact_sg_derivative():
    raw = _synthetic_iiwa_raw()
    frame = iiwa_dataset_frame(raw, n_dof=N_DOF, sg_window=SG_WINDOW, sg_poly=SG_POLY, time_step=TIME_STEP)
    for bag, group in frame.groupby("bag", sort=False):
        q = group[[f"q{i}" for i in range(N_DOF)]].to_numpy()
        dq = group[[f"dq{i}" for i in range(N_DOF)]].to_numpy()
        expected = savgol_filter(q, SG_WINDOW, SG_POLY, deriv=1, delta=TIME_STEP, axis=0, mode="interp")
        assert np.allclose(dq, expected, atol=0, rtol=0)


def test_iiwa_dq_passes_round6_check_position_derivative():
    raw = _synthetic_iiwa_raw()
    frame = iiwa_dataset_frame(raw, n_dof=N_DOF, sg_window=SG_WINDOW, sg_poly=SG_POLY, time_step=TIME_STEP)
    manifest = {
        "n_dof": N_DOF,
        "sample_time_step": TIME_STEP,
        "differentiation": {"sg_window": SG_WINDOW, "sg_poly": SG_POLY},
        "noise": {"dq": {"source": "position_derivative"}},
    }
    result = round6_checks.check_position_derivative(frame, manifest)
    assert result is not None and result["ok"], result


def test_tau_equals_ft_is_refused():
    raw = _synthetic_iiwa_raw(n_samples=100, n_bags=1)
    frame = iiwa_dataset_frame(raw, n_dof=N_DOF, sg_window=SG_WINDOW, sg_poly=SG_POLY, time_step=TIME_STEP)
    for i in range(N_DOF):
        frame[f"ft{i}"] = frame[f"tau{i}"]  # craft the exact I-9 defect pattern
    result = check_tau_ne_ft(frame, n_dof=N_DOF, quantization=1e-4,
                             tau_source="fri_commanded_torque", ft_source="iiwa_joint_torque_sensor_raw")
    assert not result["ok"]


def test_tau_ne_ft_passes_on_genuinely_different_signals():
    raw = _synthetic_iiwa_raw(n_samples=200, n_bags=1)
    frame = iiwa_dataset_frame(raw, n_dof=N_DOF, sg_window=SG_WINDOW, sg_poly=SG_POLY, time_step=TIME_STEP)
    result = check_tau_ne_ft(frame, n_dof=N_DOF, quantization=1e-4,
                             tau_source="fri_commanded_torque", ft_source="iiwa_joint_torque_sensor_raw")
    assert result["ok"]


def test_uniform_grid_detects_a_gap():
    raw = _synthetic_iiwa_raw(n_samples=50, n_bags=1)
    frame = iiwa_dataset_frame(raw, n_dof=N_DOF, sg_window=SG_WINDOW, sg_poly=SG_POLY, time_step=TIME_STEP)
    ok = check_uniform_grid(frame, nominal_dt=TIME_STEP)
    assert ok["ok"]
    frame.loc[frame.index[25], "t"] += 0.05  # a dropped-sample gap
    bad = check_uniform_grid(frame, nominal_dt=TIME_STEP)
    assert not bad["ok"]


def test_saturation_flags_over_limit():
    raw = _synthetic_iiwa_raw(n_samples=50, n_bags=1)
    frame = iiwa_dataset_frame(raw, n_dof=N_DOF, sg_window=SG_WINDOW, sg_poly=SG_POLY, time_step=TIME_STEP)
    tight_limit = np.full(N_DOF, 1.0)  # the synthetic tau has std=5, guaranteed to trip a 1 Nm limit
    result = check_saturation(frame, n_dof=N_DOF, effort_limit=tight_limit)
    assert not result["ok"]
    generous_limit = np.full(N_DOF, 1000.0)
    assert check_saturation(frame, n_dof=N_DOF, effort_limit=generous_limit)["ok"]


# ---------------------------------------------------------------------------
# real_baselines vs elastic_sim.diagnostics.bag_diagnostics equivalence
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def iiwa_asset():
    return AssetRegistry.for_repository().load("kuka_lbr_iiwa_14_r820_table")


def _synthetic_sim_like_frame(asset, n_samples=300):
    """A frame that looks like a round-6 sim bag: q_link==q_motor==q, so
    ``bag_diagnostics`` and :func:`real_baselines` should agree on the T-2
    baselines to numerical precision (T1.7 acceptance)."""
    pin, model, data = idn.build_model(asset)
    n_dof = len(asset.joint_names)
    rng = np.random.default_rng(1)
    t = np.arange(n_samples) * TIME_STEP
    q = 0.2 * np.sin(2 * np.pi * 0.5 * t[:, None] + rng.uniform(0, 1, n_dof)[None, :])
    dq = np.gradient(q, TIME_STEP, axis=0)
    ddq = savgol_filter(dq, SG_WINDOW, SG_POLY, deriv=1, delta=TIME_STEP, axis=0)
    rigid = np.asarray([idn.inverse_dynamics(pin, model, data, q[i], dq[i], ddq[i]) for i in range(n_samples)])
    tau = rigid + rng.normal(scale=0.1, size=(n_samples, n_dof))
    ft = rigid + rng.normal(scale=0.2, size=(n_samples, n_dof))

    columns = {"t": t, "bag": ["bag_0"] * n_samples, "tier": ["elastic"] * n_samples, "backend": ["mujoco"] * n_samples}
    for j in range(n_dof):
        columns[f"q{j}"] = q[:, j]
        columns[f"dq{j}"] = dq[:, j]
        columns[f"q_motor{j}"] = q[:, j]
        columns[f"q_link{j}"] = q[:, j]
        columns[f"dq_link{j}"] = dq[:, j]
        columns[f"tau{j}"] = tau[:, j]
        columns[f"ft{j}"] = ft[:, j]
    return pd.DataFrame(columns)


def test_real_baselines_matches_bag_diagnostics_on_simlike_frame(iiwa_asset):
    # RR_04 A-9: real_baselines is now computed per bag then medianed, not
    # fit once over the concatenated frame (the SG derivative must not run
    # across a bag boundary). Build 3 independent sim-like bags, take the
    # median of the *reference* bag_diagnostics() call per bag by hand, and
    # check real_baselines()'s per-split median agrees to 1e-9 -- the
    # identity test now exercised on a genuinely multi-bag frame.
    n_dof = len(iiwa_asset.joint_names)
    bags = [_synthetic_sim_like_frame(iiwa_asset, n_samples=300) for _ in range(3)]
    for index, bag in enumerate(bags):
        bag["bag"] = f"bag_{index}"
        bag["split"] = "test"
    frame = pd.concat(bags, ignore_index=True)

    reference_rows = [
        diag.bag_diagnostics(bag, asset=iiwa_asset, n_dof=n_dof, time_step=TIME_STEP, sg_window=SG_WINDOW)
        for bag in bags
    ]
    reference_median = {
        column: float(np.median([getattr(row, column) for row in reference_rows]))
        for column in ("baseline_rms_tau", "baseline_rms_mean", "baseline_rms_rigid",
                       "baseline_rms_state", "baseline_rms_state_tau")
    }

    mine = real_baselines(frame, asset=iiwa_asset, n_dof=n_dof, time_step=TIME_STEP, sg_window=SG_WINDOW, sg_poly=SG_POLY)
    test_split = mine["splits"]["test"]
    assert test_split["n_bags"] == 3

    for column, expected in reference_median.items():
        assert test_split[column] == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------
# write_real_dataset atomicity (RR_04 C-2)
# ---------------------------------------------------------------------------


def test_write_real_dataset_leaves_no_partial_file_if_the_contract_write_fails(tmp_path, monkeypatch):
    raw = _synthetic_iiwa_raw(n_samples=50, n_bags=1)
    frame = iiwa_dataset_frame(raw, n_dof=N_DOF, sg_window=SG_WINDOW, sg_poly=SG_POLY, time_step=TIME_STEP)
    manifest = {"schema_version": 2, "n_dof": N_DOF, "split": "test"}
    output = tmp_path / "run.parquet"

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated crash between write_dataset and write_real_contract")

    monkeypatch.setattr("erd_recording.contract.write_real_contract", _boom)
    with pytest.raises(RuntimeError):
        from erd_recording.convert import write_real_dataset

        write_real_dataset(frame, manifest, {}, output)

    assert not output.exists()
    assert not output.with_suffix(".manifest.json").exists()
    assert not output.with_suffix(".contract.json").exists()


def test_write_real_dataset_succeeds_atomically(tmp_path):
    from erd_recording.convert import write_real_dataset

    raw = _synthetic_iiwa_raw(n_samples=50, n_bags=1)
    frame = iiwa_dataset_frame(raw, n_dof=N_DOF, sg_window=SG_WINDOW, sg_poly=SG_POLY, time_step=TIME_STEP)
    manifest = {"schema_version": 2, "n_dof": N_DOF, "split": "test"}
    output = tmp_path / "run.parquet"

    csv_path, manifest_path, contract_path = write_real_dataset(
        frame, manifest, {"schema": "elastic_sim.identification/3", "splits": {}}, output,
    )
    assert csv_path.is_file() and manifest_path.is_file() and contract_path.is_file()
    assert not (tmp_path / ".tmp").exists()


# ---------------------------------------------------------------------------
# differentiation_from_reference (RR_01 S8.2, RR_04 A-8)
# ---------------------------------------------------------------------------


def test_differentiation_same_rate_copies_the_window(tmp_path):
    reference_path = tmp_path / "iiwa_drive.contract.json"
    reference_path.write_text('{"stub": "reference contract bytes to hash"}')
    reference_contract = {"differentiation": {"rate": 1000.0, "sg_window": 11, "sg_poly": 3,
                                              "sg_cutoff_hz": 0.45 * 1000.0 / 11}}
    result = differentiation_from_reference(reference_contract, rate_real=1000.0, probe_top_real=0.0,
                                            reference_path=reference_path)
    assert result["sg_window"] == 11
    assert result["sg_poly"] == 3
    assert result["sample_rate_hz"] == 1000.0
    assert result["cutoff_matched"] is True
    assert str(reference_path) in result["derived_from"]


def test_differentiation_different_rate_matches_the_cutoff():
    # 500 Hz reference -> 125 Hz real (the UR10 CB3 case, RR_01 S8.2). A
    # window of 45 at 500 Hz (a realistic production-scale probe-resolving
    # window, unlike the 11-sample commissioning default) keeps the solved
    # 125 Hz window comfortably above the sg_poly + 2 floor.
    ref_rate, ref_window, ref_poly = 500.0, 45, 3
    ref_cutoff = 0.45 * ref_rate / ref_window
    reference_contract = {"differentiation": {"rate": ref_rate, "sg_window": ref_window, "sg_poly": ref_poly,
                                              "sg_cutoff_hz": ref_cutoff}}
    result = differentiation_from_reference(reference_contract, rate_real=125.0, probe_top_real=5.0,
                                            reference_path=__file__)
    assert result["sg_poly"] == ref_poly
    assert result["sg_window"] % 2 == 1
    achieved_cutoff = 0.45 * 125.0 / result["sg_window"]
    # "matches the cutoff within one window step" (RR_01 T1.7 acceptance):
    # the nearest odd window can only get the cutoff within one step's worth
    # of relative error, not bit-exact.
    one_step_cutoff = 0.45 * 125.0 / (result["sg_window"] - 2)
    assert abs(achieved_cutoff - ref_cutoff) <= abs(one_step_cutoff - ref_cutoff) + 1e-9


# ---------------------------------------------------------------------------
# Consumer load test: dynamic_model_nn.dataset.CustomDataset in its own env
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not CONSUMER_PYTHON.exists(), reason="dynamic_model_nn .venv not present on this machine")
def test_consumer_loads_real_dataset(iiwa_asset, tmp_path):
    n_dof = len(iiwa_asset.joint_names)
    raw = _synthetic_iiwa_raw(n_samples=400, n_bags=2)
    frame = iiwa_dataset_frame(raw, n_dof=n_dof, sg_window=SG_WINDOW, sg_poly=SG_POLY, time_step=TIME_STEP)

    manifest = {
        "schema_version": 2, "n_dof": n_dof, "joint_names": list(iiwa_asset.joint_names),
        "signals": {"target": "link_torque", "target_source": "measured", "target_kind": "joint_torque_sensor"},
        "differentiation": {"sg_window": SG_WINDOW, "sg_poly": SG_POLY},
        "split": "test",
    }
    output = tmp_path / "synthetic_mock.parquet"

    contract = real_contract(
        manifest, hardware="mock", n_dof=n_dof, target_instrument="iiwa_joint_torque_sensor_raw",
        target_semantics="joint torque sensor, raw, gravity included [Nm]",
        q_side="motor", dq_side="motor", dq_source="position_derivative",
        tau_instrument="fri_commanded_torque",
        controller={"location": "drive", "vendor": "kuka_position_control", "model_free": False,
                   "reference": "jtc_quintic"},
        differentiation={"rate": 1.0 / TIME_STEP, "sg_window": SG_WINDOW, "sg_poly": SG_POLY,
                         "sg_cutoff_hz": 0.45 / TIME_STEP / SG_WINDOW, "probe_top_hz": 0.0},
        baselines={"statistic": "synthetic fixture, not a real median", "splits": {"test": {
            "n_bags": 2, "target_rms": 1.0, "baseline_rms_tau": 1.0, "baseline_rms_mean": 1.0,
            "baseline_rms_state": 1.0, "baseline_rms_state_tau": 1.0, "baseline_rms_rigid": 1.0,
            "link_friction_share": None, "noise_share": None, "differentiation_share": None,
        }}},
        real_block={"robot_id": "synthetic_mock"},
    )
    assert contract["source"] == "synthetic_mock" and contract["real"]["hardware"] == "mock"
    csv_path, manifest_path, contract_path = write_real_dataset(frame, manifest, contract, output)

    script = (
        f"import sys; sys.path.insert(0, r'{CONSUMER_REPO}'); "
        "from dataset import CustomDataset; "
        f"ds = CustomDataset(r'{csv_path}'); "
        "print('LOADED', len(ds), ds.dof, ds.has_joint_torque)"
    )
    result = subprocess.run([str(CONSUMER_PYTHON), "-c", script], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert "LOADED" in result.stdout


def test_written_iiwa_file_preserves_position_derivative(tmp_path):
    """Check the file after the shared writer's dtype conversion, not RAM."""
    from erd_recording.convert import write_real_dataset
    raw = _synthetic_iiwa_raw()
    raw['q0'] += 1.23456789  # sub-float32 encoder changes at a nonzero pose
    frame = iiwa_dataset_frame(raw, n_dof=N_DOF, sg_window=SG_WINDOW,
                               sg_poly=SG_POLY, time_step=TIME_STEP)
    manifest = {'schema_version': 2, 'n_dof': N_DOF, 'sample_time_step': TIME_STEP,
                'differentiation': {'sg_window': SG_WINDOW, 'sg_poly': SG_POLY},
                'noise': {'dq': {'source': 'position_derivative'}}}
    path, _, _ = write_real_dataset(frame, manifest, {'reference_synthetic': True}, tmp_path / 'test.parquet')
    restored = pd.read_parquet(path)
    result = round6_checks.check_position_derivative(restored, manifest)
    assert result['ok'], result
