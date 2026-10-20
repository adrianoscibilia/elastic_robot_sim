"""S2R_01 T-4.1 / S2R_04 P3-4: the loss and the physical metric suite on synthetic residuals."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.path.join(_REPO, "src"))

from elastic_sim.sim2real.config import LossSettings, load_config
from elastic_sim.sim2real.loss import Loss
from elastic_sim.sim2real.metrics import (friction_signature, hold_slopes, metrics, plausibility, residual_spectrum,
                                          torque_errors)
from elastic_sim.sim2real.result import ReplayResult, read_replay, write_replay

N = 7
RATE = 200.0


def _result(ft, ft_sim, *, kind="excitation", split="train", run="B", bag="excitation_0", dq=None, tau=None,
            gain=None, offset=None):
    rows = len(ft)
    t = np.arange(rows) / RATE
    zeros = np.zeros((rows, N))
    return ReplayResult(run_id=run, bag=bag, kind=kind, split=split, mode="R0", rows=np.arange(rows) * 5, t=t,
                        q=zeros, dq=zeros if dq is None else dq, ddq=zeros, q_link=zeros, ft=ft,
                        tau_s=ft_sim if tau is None else tau, ft_sim=ft_sim,
                        gain=np.ones(N) if gain is None else gain, offset=np.zeros(N) if offset is None else offset)


def test_zero_residual_gives_lambda_times_the_prior_penalty():
    rng = np.random.default_rng(0)
    ft = rng.normal(size=(400, N)) * np.linspace(10, 0.1, N)
    train = [_result(ft, ft.copy())]
    loss = Loss.from_training(train, np.full(N, 0.01), LossSettings())
    assert loss.prior_weight == 1.0 / (400 * N)
    value = loss.evaluate(train, prior_penalty=3.0)
    assert value.data == 0.0 and value.total == pytest.approx(loss.prior_weight * 3.0)
    scaled = Loss.from_training(train, np.full(N, 0.01), LossSettings(prior_weight_factor=0.01))
    assert scaled.prior_weight == pytest.approx(0.01 / (400 * N))


def test_scale_floors_stop_quiet_joints_from_dominating():
    rng = np.random.default_rng(1)
    ft = rng.normal(size=(1000, N))
    ft[:, 0] *= 50.0     # a gravity joint: its own RMS
    ft[:, 6] *= 1e-4     # a near-zero channel
    noise = np.full(N, 0.001)
    noise[5] = 0.5       # a noisy channel: 10 sigma floor binds
    loss = Loss.from_training([_result(ft, ft)], noise, LossSettings())
    scale = loss.scales["ft"]
    assert scale.scale[0] == pytest.approx(scale.rms_train[0])
    assert scale.scale[6] == pytest.approx(0.05 * scale.rms_train[0])     # share floor
    assert scale.scale[5] == pytest.approx(5.0)                            # 10 x noise
    assert scale.as_dict()["binding"][0] == "rms_train" and scale.as_dict()["binding"][6] == "share"


def test_least_squares_residual_matches_the_squared_loss():
    rng = np.random.default_rng(2)
    a = rng.normal(size=(300, N)) * 5
    b = rng.normal(size=(120, N)) * 5
    results = [_result(a, a + rng.normal(size=a.shape) * 0.3), _result(b, b + 0.2, bag="excitation_1")]
    loss = Loss.from_training(results, np.full(N, 0.01), LossSettings())
    prior_rows = rng.normal(size=5)
    rho = loss.residual_vector(results, prior_rows)
    value = loss.evaluate(results, prior_penalty=0.5 * float(prior_rows @ prior_rows))
    assert 0.5 * float(rho @ rho) == pytest.approx(value.squared, rel=1e-12)
    e = loss.errors(results)
    assert value.data == pytest.approx(float(np.mean(e)))


def test_torque_errors_on_a_known_residual():
    ft = np.ones((100, N)) * 2.0
    sim = ft + np.tile(np.arange(N, dtype=float) * 0.1, (100, 1))
    block = torque_errors([_result(ft, sim)])
    np.testing.assert_allclose(block["per_joint_rmse"], np.arange(N) * 0.1, atol=1e-12)
    np.testing.assert_allclose(block["per_joint_nrmse"], np.arange(N) * 0.05, atol=1e-12)
    assert block["rmse_median_over_bags"] == pytest.approx(np.sqrt(np.mean((np.arange(N) * 0.1) ** 2)))


def test_hold_slopes_use_gain_times_tau_and_the_gravity_span_rule():
    rng = np.random.default_rng(3)
    model = np.zeros((600, N))
    model[:, 1] = rng.uniform(-60, 40, 600)   # spans >= 2 N m
    model[:, 3] = rng.uniform(-20, 10, 600)
    model[:, 5] = rng.uniform(-0.5, 0.5, 600)  # spans < 2 N m: not reported
    offset = np.full(N, 0.4)
    real = 1.12 * model + 0.3 + 0.001 * rng.normal(size=model.shape)
    hold = _result(real, model + offset, kind="hold", tau=model, offset=offset, bag="identify_hold_0")
    block = hold_slopes([hold])
    assert block["B"]["joints"] == [1, 3]
    assert block["B"]["slope"][1] == pytest.approx(1.12, abs=1e-3)
    assert block["B"]["intercept"][3] == pytest.approx(0.3, abs=1e-2)


def test_friction_signature_finds_a_coulomb_leftover():
    rng = np.random.default_rng(4)
    dq = rng.uniform(-0.6, 0.6, (5000, N))
    residual = 0.7 * np.sign(dq) * (np.arange(N) == 2)
    block = friction_signature([_result(np.zeros_like(dq), residual, dq=dq)])
    assert block["odd_component"][2] == pytest.approx(0.7, abs=1e-12)
    assert block["odd_component"][0] == pytest.approx(0.0, abs=1e-12)
    assert block["max_abs_bin_mean"][2] == pytest.approx(0.7)


def test_residual_spectrum_bands_sum_to_the_mean_square():
    t = np.arange(4000) / RATE
    residual = np.zeros((4000, N))
    residual[:, 0] = 1.0 * np.sin(2 * np.pi * 0.25 * t)
    residual[:, 1] = 0.5 * np.sin(2 * np.pi * 1.0 * t)
    residual[:, 2] = 0.2 * np.sin(2 * np.pi * 5.0 * t)
    residual[:, 3] = 0.3  # DC
    block = residual_spectrum([_result(np.zeros_like(residual), residual)])
    power = np.asarray(block["power_nm2"])
    np.testing.assert_allclose(power.sum(axis=0), np.mean(residual ** 2, axis=0), rtol=1e-9, atol=1e-15)
    assert power[0, 0] == pytest.approx(0.5) and power[1, 1] == pytest.approx(0.125)
    assert power[2, 2] == pytest.approx(0.02) and power[0, 3] == pytest.approx(0.09)


def test_metrics_split_selection_and_parquet_round_trip(tmp_path):
    rng = np.random.default_rng(5)
    ft = rng.normal(size=(50, N))
    results = [_result(ft, ft + 0.1, split="train"), _result(ft, ft + 0.2, split="test", run="A", bag="excitation_3")]
    block = metrics(results, "test", contracts={"A": {"baselines": {"splits": {"all": {"baseline_rms_rigid": 3.5}}}}})
    assert block["bags"] == ["A:excitation_3"] and block["torque"]["rmse_median_over_bags"] == pytest.approx(0.2)
    assert block["baselines"]["A"]["baseline_rms_rigid"] == 3.5
    back = read_replay(write_replay(results, tmp_path / "replay.parquet"))
    assert [r.key for r in back] == [r.key for r in results]
    np.testing.assert_array_equal(back[1].ft_sim, results[1].ft_sim)
    np.testing.assert_array_equal(back[1].offset, results[1].offset)


def test_plausibility_reports_prior_distance_data_share_and_moves():
    from elastic_sim.sim2real.parameters import Registry
    from elastic_sim.sim2real.plant import PlantBase, PlantParameters

    config = load_config(_REPO / "config" / "sim2real" / "kuka_lbr_iiwa_14_r820_base.yaml")
    base = PlantBase.from_config(config)
    registry = Registry.from_config(config, base)
    theta = registry.values()
    theta["inertial.iiwa_link_2.mass"] *= 1.1
    plant = PlantParameters.from_theta(base, theta)
    share = {name: 0.95 for name in registry.free_names}
    share["inertial.iiwa_link_1.com_z"] = 0.0
    # S2R_06 F-2: d_post decides (the diagonal d is carried as a diagnostic).
    block = plausibility(registry, theta, plant, data_share={n: 1.0 for n in share}, data_share_posterior=share,
                         not_calibrated={"inertial.iiwa_link_1.com_y": "A-12: not seen"})
    rows = {r["name"]: r for r in block["parameters"]}
    assert rows["inertial.iiwa_link_1.com_z"]["prior_determined"] and block["n_prior_determined"] == 1
    assert not rows["inertial.iiwa_link_1.com_y"]["calibrated"] and block["n_not_calibrated"] == 1
    assert rows["inertial.iiwa_link_2.mass"]["calibrated"] and not rows["inertial.iiwa_link_2.mass"]["prior_determined"]
    assert block["physical"]["consistent"]
    assert block["inertial_moves_vs_asset"]["iiwa_link_2"]["delta_mass_pct"] == pytest.approx(10.0)
