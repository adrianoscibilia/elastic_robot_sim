"""S2R_01 T-3.2 / S2R_04 P3-2: the R0 replay engine, its Jacobian and gate G-1."""

from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.path.join(_REPO, "src"))

from elastic_sim.sim2real.config import R0Settings, load_config
from elastic_sim.sim2real.parameters import Registry
from elastic_sim.sim2real.plant import IntermediateBodies, PlantBase, PlantParameters
from elastic_sim.sim2real.real_runs import ReplayBag
from elastic_sim.sim2real.replay_r0 import R0Engine, gate_g1, prepare_bags

BASE_CONFIG = _REPO / "config" / "sim2real" / "kuka_lbr_iiwa_14_r820_base.yaml"
HOME = np.array([0.0, 0.4, 0.0, -1.2, 0.0, 0.8, 0.0])
DT = 1e-3


@pytest.fixture(scope="module")
def config():
    return load_config(BASE_CONFIG)


@pytest.fixture(scope="module")
def base(config):
    return PlantBase.from_config(config)


@pytest.fixture(scope="module")
def registry(config, base):
    return Registry.from_config(config, base)


@pytest.fixture(scope="module")
def intermediate(base):
    return IntermediateBodies.from_elastic_build(base)


def _bag(base, rng, name, run_id="A", kind="excitation", rows=2001):
    from elastic_sim import identification as idn

    t = np.arange(rows) * DT
    if kind == "hold":
        q = np.tile(HOME + rng.uniform(-0.3, 0.3, 7), (rows, 1)) + 1e-7 * rng.standard_normal((rows, 7))
    else:
        freq = rng.uniform(0.2, 1.0, (3, 7))
        amp = rng.uniform(0.05, 0.25, (3, 7))
        phase = rng.uniform(0, 2 * np.pi, (3, 7))
        q = HOME + np.sum(amp[:, None, :] * np.sin(2 * np.pi * freq[:, None, :] * t[None, :, None] + phase[:, None, :]),
                          axis=0)
    pin, model, data = idn.build_model(base.asset)
    ft = np.asarray([pin.computeGeneralizedGravity(model, data, row) for row in q]) + 0.3
    dq = np.gradient(q, DT, axis=0)
    return ReplayBag(run_id=run_id, bag=name, kind=kind, t=t, q=q, dq=dq, tau_cmd=np.zeros_like(q), ft=ft)


def _theta(registry, base, rng):
    theta = registry.values()
    for name in registry.free_names:
        spec = registry.specs[name]
        theta[name] = spec.from_unit(rng.uniform(-0.4, 0.4))
    theta["transmission.iiwa_A2.stiffness"] = base.stiffness[1]
    return theta


def test_contract_recipe_is_the_contracts_rigid_baseline_bit_for_bit(base):
    """G-1's engine path vs erd_recording.contract._per_bag_baselines's rigid
    term (idn.inverse_dynamics per row, SG(5, 3) of the dataset dq)."""
    from scipy.signal import savgol_filter

    from elastic_sim import identification as idn

    bag = _bag(base, np.random.default_rng(0), "excitation_0")
    prepared = prepare_bags([bag], R0Settings(), time_step=DT, recipe="contract")
    engine = R0Engine(base, prepared, deflection=False, intermediate=None, friction=False)
    tau = engine.tau_s(PlantParameters.nominal(base))
    pin, model, data = idn.build_model(base.asset)
    ddq = savgol_filter(bag.dq, 5, 3, deriv=1, delta=DT, axis=0, mode="interp")
    reference = np.asarray([idn.inverse_dynamics(pin, model, data, bag.q[i], bag.dq[i], ddq[i])
                            for i in range(bag.n_rows)])
    np.testing.assert_array_equal(tau, reference)


def test_spring_torque_is_the_regressor_times_the_standard_parameters(base, registry, intermediate):
    """tau_s = Y(q_link) (pi_theta + pi_intermediate) + friction: the linear
    structure the analytic Jacobian relies on (A-9 bodies included)."""
    rng = np.random.default_rng(1)
    prepared = prepare_bags([_bag(base, rng, "excitation_0")], R0Settings(decimate=50), time_step=DT)
    engine = R0Engine(base, prepared, intermediate=intermediate)
    plant = PlantParameters.from_theta(base, _theta(registry, base, rng), runs=("A", "B"))
    q_link, tau, _ = engine.simulate(plant)
    pin = engine.pin
    pi = np.concatenate([pin.Inertia(i.mass, i.com, i.inertia).toDynamicParameters()
                         for i in plant.inertials.values()]) + intermediate.dynamic_parameters(7)
    for k in range(engine.n_rows):
        y = pin.computeJointTorqueRegressor(engine.model, engine.data, q_link[k], engine.dq[k], engine.ddq[k])
        expected = y @ pi + plant.link_friction.torque(engine.dq[k])
        np.testing.assert_allclose(tau[k], expected, rtol=0, atol=1e-10)


def test_analytic_jacobian_matches_central_differences(base, registry, intermediate):
    """T-3.2 done-when: every free group (inertials, link friction, offsets,
    gain) and K, to 1e-5 relative per column."""
    rng = np.random.default_rng(2)
    bags = [_bag(base, rng, "excitation_0", run_id="B"), _bag(base, rng, "identify_hold_0", run_id="A", kind="hold")]
    prepared = prepare_bags(bags, R0Settings(decimate=40, hold_decimate=200), time_step=DT)
    engine = R0Engine(base, prepared, intermediate=intermediate)
    theta = _theta(registry, base, rng)
    names = list(registry.free_names) + [f"transmission.{j}.stiffness" for j in base.joint_names]
    plant = PlantParameters.from_theta(base, theta, runs=("A", "B"))
    jacobian = engine.jacobian(plant, names)
    worst = 0.0
    for i, name in enumerate(names):
        # K moves ft_sim by ~1e-9 N m per 1e-6 relative step: take a larger one.
        numeric = engine._central_difference(theta, name, relative_step=1e-3 if "stiffness" in name else 1e-6)
        scale = np.linalg.norm(numeric)
        if scale == 0.0:
            assert np.linalg.norm(jacobian[:, i]) < 1e-9, name  # structurally zero (e.g. link 1 com_z)
            continue
        error = np.linalg.norm(jacobian[:, i] - numeric) / scale
        worst = max(worst, error)
        assert error <= 1e-5, f"{name}: {error:.2e}"
    print(f"\nR0 Jacobian vs central differences: worst relative column error {worst:.2e}")


def test_prepare_bags_decimates_holds_rest_and_filters_quasi_static_rows(base):
    rng = np.random.default_rng(3)
    excitation = _bag(base, rng, "excitation_0", rows=4001)
    hold = _bag(base, rng, "identify_hold_0", kind="hold", rows=4001)
    settings = R0Settings(decimate=5, hold_decimate=50)
    prepared = {b.bag: b for b in prepare_bags([excitation, hold], settings, time_step=DT,
                                               split_of=lambda run, bag: "train" if bag.startswith("exc") else None)}
    assert len(prepared["excitation_0"].rows) == 801 and prepared["excitation_0"].split == "train"
    assert len(prepared["identify_hold_0"].rows) == 81 and prepared["identify_hold_0"].split == ""
    assert np.all(prepared["identify_hold_0"].dq == 0.0) and np.all(prepared["identify_hold_0"].ddq == 0.0)
    # First half a slow ramp (0.01 rad/s), second half the fast excitation.
    t = excitation.t
    ramp = HOME + 0.01 * t[:, None] * np.ones(7)
    q = np.where((t < 2.0)[:, None], ramp, excitation.q - excitation.q[2000] + ramp[2000])
    mixed = dataclasses.replace(excitation, bag="excitation_1", q=q)
    calm = prepare_bags([mixed], dataclasses.replace(settings, rows="quasi_static"), time_step=DT)[0]
    assert 300 < len(calm.rows) < 801
    assert np.all(np.abs(calm.dq) < 0.05) and np.all(np.abs(calm.ddq) < 0.2)
    assert prepare_bags([excitation], settings, time_step=DT, split_of=lambda r, b: None, keep_unsplit=False) == []


def test_intermediate_bodies_change_r0_by_milli_newton_metres(base, intermediate):
    """S2R_04 N-8 measured: 1 g bodies at the joint origins move the gravity
    torque by mN m (below the 0.01 N m ft noise of the shoulder, not 1e-5)."""
    assert len(intermediate.inertias) == 7
    assert all(abs(i.mass - 1e-3) < 1e-15 for i in intermediate.inertias.values())
    prepared = prepare_bags([_bag(base, np.random.default_rng(4), "excitation_0")], R0Settings(decimate=20),
                            time_step=DT)
    plant = PlantParameters.nominal(base)
    with_bodies = R0Engine(base, prepared, intermediate=intermediate).tau_s(plant)
    without = R0Engine(base, prepared, intermediate=None).tau_s(plant)
    change = np.max(np.abs(with_bodies - without), axis=0)
    print("\nintermediate bodies, max |delta tau| per joint (N m): " + " ".join(f"{v:.4f}" for v in change))
    assert 1e-4 < change.max() < 0.05


REAL = {"A": _REPO / "data/real_robot/kuka_lbr_iiwa_14_r820/20261014/iiwa_real_all_01",
        "B": _REPO / "data/real_robot/kuka_lbr_iiwa_14_r820/20261014/iiwa_fast_all_01"}


@pytest.mark.slow
@pytest.mark.parametrize("run_id", ["A", "B"])
def test_gate_g1_reproduces_the_contract_rigid_baseline(config, base, run_id):
    """Gate G-1 (S2R_01 T-3.2): <= 1e-6 relative to the contract's baseline_rms_rigid."""
    from elastic_sim.sim2real.real_runs import load_real_run

    if not (REAL[run_id] / "dataset").is_dir():
        pytest.skip(f"real run {run_id} not present")
    run = load_real_run(REAL[run_id], expected_sha256=config.runs[run_id].sha256, sidecar=False, run_id=run_id)
    gate = gate_g1(run, base)
    print(f"\nG-1 {run_id}: {gate['baseline_rms_rigid']:.12g} vs contract {gate['contract']:.12g} "
          f"(relative {gate['relative_error']:.1e})")
    assert gate["relative_error"] <= 1e-6
