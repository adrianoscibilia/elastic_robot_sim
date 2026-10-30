"""Zollo et al. (2015) model-based baseline: gate G1 and the building blocks.

G1 (`REFACTOR_SPECS/model_based_baseline/01_zollo_plan.md` Sec 6): on a
Spong-model plant that is exactly the identification's model, noise-free
and with exact accelerations, both variants recover the base parameters to
numerical precision -- Zollo's 2-DoF planar example unmarked, the iiwa
``slow``.  Plus the pieces the evaluation relies on: the incremental QR,
the delay alignment, the save/load round trip and the forward model.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.path.join(_REPO, "src"))

from elastic_sim.zollo import synthetic as syn
from elastic_sim.zollo.basis import Accumulator, select_base, solve
from elastic_sim.zollo.data import _delayed
from elastic_sim.zollo.identify import IdentificationConfig, identify, load_identification, recovery_table
from elastic_sim.zollo.consistent import body_eigenvalues, solve_log_cholesky
from elastic_sim.zollo.predictors import ElasticModel, LinkModel, consumer_inputs, identified_transmission, is_positive_definite
from elastic_sim.zollo.regressor import Layout
from elastic_sim.zollo.signals import Preprocessing

EXACT = Preprocessing(cutoff_hz=None, parallel_filter=False, decimate=1, edge_s=0.0)


def _planar_arm(rigid_model):
    return syn.SpongArm(rigid_model, K=np.array([3000.0, 1800.0]), D=np.array([4.0, 2.0]), B=np.array([2.0, 1.0]),
                        motor_viscous=np.array([5.0, 3.0]), motor_coulomb=np.array([2.0, 1.0]),
                        link_viscous=np.array([1.0, 0.5]), link_coulomb=np.array([3.0, 1.5]))


@pytest.fixture(scope="module")
def planar():
    rigid_model = syn.planar_two_link()
    arm = _planar_arm(rigid_model)
    bags = [syn.as_bag(syn.simulate(arm, duration=3.0, step=2e-4, sample_every=5, seed=s), f"b{s}") for s in range(3)]
    return rigid_model, arm, bags


def _max_base_error(result, truth, groups=None) -> float:
    rows = recovery_table(result, truth)
    scale = max(abs(r["true"]) for r in rows)
    return max(abs(r["identified"] - r["true"]) / max(abs(r["true"]), 1e-3 * scale) for r in rows
               if groups is None or r["group"] in groups)


@pytest.mark.parametrize("variant", ["A", "B"])
def test_g1_planar_exact_recovery(planar, variant):
    rigid_model, arm, bags = planar
    config = IdentificationConfig(variant=variant, preprocessing=EXACT, include_rigid_tier=False)
    result = identify(bags, config, rigid_model, truth={"e00": arm.truth()})
    assert result.solution.residual_rms < 1e-9
    assert _max_base_error(result, {"e00": arm.truth()}) < 1e-6
    # Zollo's planar arm: both masses' first moments along the link and the second link's
    # inertia are identifiable alone or in combination; the first mass never acts on a torque.
    names = result.layout.names
    assert result.base.categories[names.index("mass.m[1]")] == "unidentifiable"
    for group in ("stiffness", "damping", "link_viscous", "link_coulomb"):
        assert all(result.base.categories[names.index(f"{group}[{j}]")] == "identifiable" for j in (1, 2))
    if variant == "A":
        assert result.value("rotor", 0) == pytest.approx(2.0, rel=1e-6)
        assert result.value("motor_coulomb", 1) == pytest.approx(1.0, rel=1e-6)


def test_g1_planar_per_robot_transmission(planar):
    rigid_model, arm, bags = planar
    softer = syn.SpongArm(rigid_model, K=arm.K * 0.5, D=arm.D, B=arm.B * 1.5, motor_viscous=arm.motor_viscous,
                          motor_coulomb=arm.motor_coulomb, link_viscous=arm.link_viscous,
                          link_coulomb=arm.link_coulomb)
    extra = [syn.as_bag(syn.simulate(softer, duration=3.0, step=2e-4, sample_every=5, seed=10 + s), f"c{s}", "e01")
             for s in range(2)]
    truth = {"e00": arm.truth(), "e01": softer.truth()}
    config = IdentificationConfig(variant="A", transmission="per_robot", preprocessing=EXACT, include_rigid_tier=False)
    result = identify(bags + extra, config, rigid_model, truth=truth)
    assert result.value("stiffness", 0, "e01") == pytest.approx(1500.0, rel=1e-6)
    assert result.value("rotor", 1, "e00") == pytest.approx(1.0, rel=1e-6)
    assert _max_base_error(result, truth) < 1e-6


def test_g1_pr_per_robot_motor_recovers_what_pooling_biases(planar):
    """G1-pr (04_variant_A_elastic.md Sec 1): robots that differ only in rotor inertia and motor friction.

    A-pr (motor groups per robot, the rest pooled) recovers the pooled link and
    spring base parameters and each robot's motor parameters to rounding;
    pooled A, forced to one rotor and one motor friction, biases the link and
    spring through the coupled rows -- the test discriminates.
    """
    rigid_model, arm, bags = planar
    motors = {"e01": (1.6, 0.6, 1.8), "e02": (0.6, 1.7, 0.5)}          # rotor, viscous, Coulomb factors
    arms = {"e00": arm}
    for robot, (b, fv, fc) in motors.items():
        arms[robot] = syn.SpongArm(rigid_model, K=arm.K, D=arm.D, B=arm.B * b, motor_viscous=arm.motor_viscous * fv,
                                   motor_coulomb=arm.motor_coulomb * fc, link_viscous=arm.link_viscous,
                                   link_coulomb=arm.link_coulomb)
    extra = [syn.as_bag(syn.simulate(arms[robot], duration=3.0, step=2e-4, sample_every=5, seed=20 + 3 * i + s),
                        f"{robot}_{s}", robot) for i, robot in enumerate(motors) for s in range(2)]
    data = bags + extra
    truth = {robot: a.truth() for robot, a in arms.items()}
    shared = ("mass", "first_moment", "inertia", "link_viscous", "link_coulomb", "stiffness", "damping")
    config = IdentificationConfig(variant="A", transmission="per_robot_motor", preprocessing=EXACT,
                                  include_rigid_tier=False)
    result = identify(data, config, rigid_model, truth=truth)
    assert config.label == "Apr_pooled"
    assert result.layout.robots == ("e00", "e01", "e02")
    assert not any(c.robot for c in result.layout.columns if c.group in shared)
    assert result.solution.residual_rms < 1e-9
    assert _max_base_error(result, truth) < 1e-6
    for robot, a in arms.items():
        for group, value in (("rotor", a.B), ("motor_viscous", a.motor_viscous), ("motor_coulomb", a.motor_coulomb)):
            for j in range(2):
                assert result.value(group, j, robot) == pytest.approx(value[j], rel=1e-6), (group, j, robot)
    # Same rows, one motor parameter set: the link and spring base parameters move.
    pooled = identify(data, IdentificationConfig(variant="A", preprocessing=EXACT, include_rigid_tier=False),
                      rigid_model, truth=truth)
    assert pooled.solution.residual_rms > 1e-3
    assert _max_base_error(pooled, truth, shared) > 1e-3
    assert _max_base_error(result, truth, shared) < 1e-6


def test_per_robot_motor_needs_variant_a():
    with pytest.raises(ValueError, match="variant A"):
        IdentificationConfig(variant="B", transmission="per_robot_motor")


def _arm_like(arm, rigid_model, K=1.0, D=1.0, B=1.0, fv=1.0, fc=1.0):
    return syn.SpongArm(rigid_model, K=arm.K * K, D=arm.D * D, B=arm.B * B, motor_viscous=arm.motor_viscous * fv,
                        motor_coulomb=arm.motor_coulomb * fc, link_viscous=arm.link_viscous,
                        link_coulomb=arm.link_coulomb)


def test_g1_pc_per_robot_transmission_consistent(planar):
    """G1-pc (06_variant_A_consistent.md Sec 2): robots differing in K, D, rotor inertia and motor friction.

    The truth is physically consistent, so the constrained optimum is the
    unconstrained one: A-pc recovers every link base parameter and every
    robot's K, D, B, f_v, f_c to rounding, and the log-Cholesky LM agrees.
    """
    rigid_model, arm, bags = planar
    factors = {"e01": (0.6, 1.5, 1.6, 0.6, 1.8), "e02": (1.4, 0.7, 0.6, 1.7, 0.5)}          # K D B f_v f_c
    arms = {"e00": arm, **{r: _arm_like(arm, rigid_model, *f) for r, f in factors.items()}}
    extra = [syn.as_bag(syn.simulate(arms[r], duration=3.0, step=2e-4, sample_every=5, seed=30 + 3 * i + s),
                        f"{r}_{s}", r) for i, r in enumerate(factors) for s in range(2)]
    truth = {r: a.truth() for r, a in arms.items()}
    config = IdentificationConfig(variant="A", transmission="per_robot_transmission", preprocessing=EXACT,
                                  include_rigid_tier=False)
    assert config.is_consistent and config.label == "Apc_pooled"
    result = identify(bags + extra, config, rigid_model, truth=truth)
    report = result.meta["consistency"]
    assert report["status"] == "optimal"
    assert report["cost"] <= report["cost_unconstrained"] + 1e-9
    assert _max_base_error(result, truth) < 1e-6
    for robot, a in arms.items():
        for group, value in (("stiffness", a.K), ("damping", a.D), ("rotor", a.B), ("motor_viscous", a.motor_viscous),
                             ("motor_coulomb", a.motor_coulomb)):
            for j in range(2):
                assert result.value(group, j, robot) == pytest.approx(value[j], rel=1e-6), (group, j, robot)
    assert body_eigenvalues(result.full[result.layout.inertial_slice()]).min() >= 1e-6 * (1 - 1e-6)
    K, D = identified_transmission(result)                                     # C3: the median robot
    np.testing.assert_allclose(K, np.median([a.K for a in arms.values()], axis=0), rtol=1e-6)
    np.testing.assert_allclose(D, np.median([a.D for a in arms.values()], axis=0), rtol=1e-6)
    lm = solve_log_cholesky(result.accumulator, result.base, result.layout,
                            np.concatenate([rigid_model[1].inertias[i].toDynamicParameters() for i in (1, 2)]))
    np.testing.assert_allclose(result.base.base_of(lm.full), result.base.base_of(result.full), rtol=1e-6, atol=1e-9)


def test_g1_pc_constraint_binds_on_an_unanchored_wrist():
    """G1-pc, the non-PD case: a light wrist whose link reading carries an error ``c ddq`` (an unmodelled lag).

    The spring term absorbs it as ``-K c`` of link inertia: the unconstrained
    LS puts the wrist's Izz below zero, a base no consistent body reaches.
    The constrained fit returns every J_i positive definite, at a cost >= the
    unconstrained one, with the wrist's constraint active.  The first body's
    mass is unidentifiable and relaxes its J_1 >= eps I for free, so the
    infimum is approached only as it grows: the solver may stop inaccurate,
    a hair inside eps, which is why J_i is checked > 0, not >= eps.
    """
    rigid_model = syn.planar_two_link(masses=(20.0, 0.2))
    arm = syn.SpongArm(rigid_model, K=np.array([3000.0, 300.0]), D=np.array([4.0, 0.2]), B=np.array([2.0, 0.1]),
                       motor_viscous=np.array([5.0, 0.3]), motor_coulomb=np.array([2.0, 0.1]),
                       link_viscous=np.array([1.0, 0.05]), link_coulomb=np.array([3.0, 0.15]))
    bags = []
    for s in range(2):
        run = syn.simulate(arm, duration=3.0, step=2e-4, sample_every=5, seed=s, kp=np.array([4e3, 300.0]),
                           kd=np.array([150.0, 5.0]))
        bag = syn.as_bag(run, f"w{s}")
        bag.exact = {k: v.copy() for k, v in bag.exact.items()}
        bag.exact["q"][:, 1] += 2e-4 * run["ddq"][:, 1]
        bags.append(bag)
    plain = identify(bags, IdentificationConfig(variant="A", transmission="per_robot_transmission", consistent=False,
                                                preprocessing=EXACT, include_rigid_tier=False), rigid_model)
    names = plain.layout.names
    izz2 = plain.solution.base[list(plain.base.independent).index(names.index("inertia.Izz[2]"))]
    assert izz2 < 0.0
    result = identify(bags, IdentificationConfig(variant="A", transmission="per_robot_transmission",
                                                 preprocessing=EXACT, include_rigid_tier=False), rigid_model)
    report = result.meta["consistency"]
    assert report["cost"] >= plain.solution.rss * (1 - 1e-12)
    assert report["cost_increase_relative"] > 1e-3
    assert 2 in report["active_bodies"]
    assert body_eigenvalues(result.full[result.layout.inertial_slice()]).min() > 0.0
    link = LinkModel.from_identification(result, rigid_model)
    assert all(is_positive_definite(link.mass(np.array([0.0, q2]))) for q2 in np.linspace(-np.pi, np.pi, 13))


def test_per_robot_transmission_needs_variant_a():
    with pytest.raises(ValueError, match="variant A"):
        IdentificationConfig(variant="B", transmission="per_robot_transmission")
    assert IdentificationConfig(variant="A", transmission="per_robot_transmission", consistent=False).label == \
        "Apc_pooled_noC2"
    assert IdentificationConfig(variant="A", transmission="per_robot_motor", consistent=True).label == "Apr_pooled_C2"


def test_g1_noise_degrades_gracefully(planar):
    """With noise the filtered path still lands near the truth; the error grows with the noise (Zollo Table 2)."""
    rigid_model, arm, _ = planar
    runs = [syn.simulate(arm, duration=4.0, step=2e-4, sample_every=5, seed=s) for s in range(3)]
    pre = Preprocessing(cutoff_hz=20.0, decimate=2, edge_s=0.1)
    errors = []
    for sigma in (1e-5, 3e-3):
        bags = [syn.as_bag(run, f"n{i}", noise={"theta": sigma, "q": sigma, "u": 100 * sigma, "tau_J": 100 * sigma},
                           seed=i) for i, run in enumerate(runs)]
        result = identify(bags, IdentificationConfig(variant="A", preprocessing=pre, include_rigid_tier=False),
                          rigid_model)
        errors.append([abs(result.value("stiffness", 0) - 3000.0) / 3000.0, abs(result.value("rotor", 0) - 2.0) / 2.0])
    quiet, loud = np.asarray(errors)
    assert (quiet < 1e-3).all()
    assert (loud > 10 * quiet).all()
    assert loud[1] > loud[0]                   # the rotor inertia goes first, as in Zollo's Table 6


@pytest.mark.slow
def test_g1_iiwa_exact_recovery():
    from elastic_sim import identification as idn
    from elastic_sim.assets import AssetRegistry

    rigid_model = idn.build_model(AssetRegistry.for_repository(_REPO).load("kuka_lbr_iiwa_14_r820_table"))
    arm = syn.iiwa_arm(rigid_model, [2.4e4, 2.4e4, 1.6e4, 1.6e4, 9.0e3, 5.5e3, 5.5e3],
                       [1.0, 1.0, 0.5, 0.5, 0.25, 0.15, 0.15])
    q0 = np.array([0.0, 0.6, 0.0, -1.2, 0.0, 0.8, 0.0])
    bags = [syn.as_bag(syn.simulate(arm, duration=2.0, step=5e-5, sample_every=20, seed=s, q0=q0, amplitude=0.4,
                                    kp=np.array([3e4, 3e4, 1.5e4, 1.5e4, 5e3, 2e3, 2e3]),
                                    kd=np.array([600, 600, 300, 300, 100, 40, 40.0])), f"b{s}") for s in range(4)]
    for variant in ("A", "B"):
        config = IdentificationConfig(variant=variant, preprocessing=EXACT, include_rigid_tier=False)
        result = identify(bags, config, rigid_model, truth={"e00": arm.truth()})
        assert _max_base_error(result, {"e00": arm.truth()}) < 1e-6, variant


def test_incremental_qr_matches_direct_least_squares():
    rng = np.random.default_rng(3)
    W = rng.normal(size=(500, 6))
    W[:, 4] = 2.0 * W[:, 1] - W[:, 2]          # dependent on earlier columns
    W[:, 5] = 0.0                              # never acts
    truth = np.array([1.0, -2.0, 0.5, 3.0, 0.0, 0.0])
    y = W @ truth
    acc = Accumulator(6)
    for block in np.array_split(np.arange(500), 7):
        acc.add(W[block], y[block])
    base = select_base(acc)
    assert base.independent.tolist() == [0, 1, 2, 3]
    assert base.zero.tolist() == [5]
    assert base.categories == ["identifiable", "combination", "combination", "identifiable", "combination",
                               "unidentifiable"]
    solution = solve(acc, base)
    np.testing.assert_allclose(solution.base, base.base_of(truth), atol=1e-10)
    assert solution.rss < 1e-18


def test_delay_alignment_puts_privileged_channels_on_the_recorded_grid():
    x = np.arange(10.0)[:, None]
    recorded = _delayed(x, 1)                   # what the recorder does to q, ft
    assert recorded[1:, 0].tolist() == x[:-1, 0].tolist()


def test_layout_per_robot_columns():
    layout = Layout.build(2, "A", robots=["e00", "e01"])
    assert "stiffness[1]@e01" in layout.names and "link_viscous[1]" in layout.names
    shared = [layout.columns[i] for i in layout.active("e00")]
    assert all(c.robot in (None, "e00") for c in shared)


def test_save_load_round_trip_and_forward_model(planar, tmp_path):
    rigid_model, arm, bags = planar
    config = IdentificationConfig(variant="B", preprocessing=EXACT, include_rigid_tier=False)
    result = identify(bags, config, rigid_model, truth={"e00": arm.truth()})
    path = tmp_path / "identification.json"
    result.save(path)
    loaded = load_identification(path)
    np.testing.assert_allclose(loaded.full, result.full)
    bag = bags[0]
    # ID-inverse at the true link state reproduces tau_J exactly on the identified model's own plant.
    link = LinkModel.from_identification(loaded, rigid_model)
    np.testing.assert_allclose(link.torque(bag.exact["q"], bag.exact["dq"], bag.exact["ddq"]), bag.tau_J, atol=1e-8)
    # ID-elastic driven by the true motor motion tracks the plant's tau_J.
    model = ElasticModel.from_identification(loaded, rigid_model)
    run = model.run(bag.theta, bag.dtheta, float(bag.t[1] - bag.t[0]))
    assert run.diverged_at is None
    error = run.tau[200:] - bag.tau_J[200:]
    assert np.sqrt(np.mean(error ** 2)) < 0.02 * np.sqrt(np.mean(bag.tau_J ** 2))


def test_consumer_inputs_use_the_contract_window(planar):
    _, _, bags = planar
    bag = bags[0]
    theta, dtheta, ddtheta = consumer_inputs(bag, {"differentiation": {"sg_window": 5, "sg_poly": 3}})
    assert theta is bag.theta and dtheta is bag.dtheta
    assert np.sqrt(np.mean((ddtheta[5:-5] - bag.exact["ddtheta"][5:-5]) ** 2)) < 0.05 * np.sqrt(
        np.mean(bag.exact["ddtheta"] ** 2))
