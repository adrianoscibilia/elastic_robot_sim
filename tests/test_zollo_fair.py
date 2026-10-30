"""The fair classical baselines (`REFACTOR_SPECS/model_based_baseline/03_fair_comparison_plan.md`).

FG2 (no privileged reads) on a subset of the UR10 r7 file, and the pieces the
fit relies on: the windows rule of the LNNs' ``dataset.bag_windows``, the
log-Cholesky parametrisation, the batched filter against the producer's
``filter_sequence`` and the output-error Jacobian against finite differences.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.path.join(_REPO, "src"))

from elastic_sim.zollo import fair
from elastic_sim.zollo import synthetic as syn

UR10 = _REPO / "data/identification/r7-nopayload/ur10/ur10_table_round7_nopayload_val4.parquet"


def _priors(n: int, step: float, burn_in: float) -> fair.FairPriors:
    return fair.FairPriors(step, burn_in, 5, 3, np.full(n, 500.0), np.full(n, 5000.0), np.full(n, 0.05),
                           np.full(n, 1.0), np.full(n, 0.3))


def test_sequence_windows_follow_the_lnn_rule():
    """UR10: 5001 samples at 2 ms, burn-in 2.7 s -> 4 windows of 1850 samples, stride 925, 500 loss rows each."""
    t = np.arange(5001) * 0.002
    priors = _priors(6, 0.002, 2.7)
    windows = fair.sequence_windows(t, priors)
    assert windows == [(0, 1850), (925, 2775), (1850, 3700), (2775, 4625)]
    assert all(fair.head_mask(t[a:b], 2.7).sum() == 500 for a, b in windows)


def test_log_cholesky_round_trip_and_consistency():
    rng = np.random.default_rng(1)
    for _ in range(20):
        theta = rng.normal(scale=0.5, size=10)
        pi = fair.inertial_from_log_cholesky(theta)
        assert fair.physically_consistent(pi)
        np.testing.assert_allclose(fair.log_cholesky_from_inertial(pi), theta, atol=1e-9)


def test_batch_filter_is_filter_sequence():
    import torch

    from elastic_sim.two_mass import filter_sequence

    rng = np.random.default_rng(0)
    n, L, h = 3, 400, 1e-3
    A = rng.normal(size=(2, L, n, n))
    M = np.einsum("blij,blkj->blik", A, A) + np.eye(n)
    tau = rng.normal(size=(2, L, n)).cumsum(axis=1)
    K, D = rng.uniform(1e3, 1e4, n), rng.uniform(1.0, 10.0, (2, n))
    delta, velocity = fair.BatchFilter(torch.tensor(M), torch.tensor(K), torch.tensor(D), h).run(
        torch.tensor(tau)[..., None])
    for b in range(2):
        reference = filter_sequence(tau[b], M[b], K, D[b], h, coupling="full")
        np.testing.assert_allclose(delta[b, :, :, 0].numpy(), reference["delta"], rtol=1e-10, atol=1e-14)
        np.testing.assert_allclose(velocity[b, :, :, 0].numpy(), reference["delta_dot"], rtol=1e-10, atol=1e-12)


def test_output_error_jacobian_is_exact():
    """Every block of the gradient (inertials through tau_r, M_hat and M_bar; friction; K; zeta) vs differences."""
    rigid_model = syn.planar_two_link()
    arm = syn.SpongArm(rigid_model, K=np.array([3000.0, 1800.0]), D=np.array([4.0, 2.0]), B=np.array([2.0, 1.0]),
                       motor_viscous=np.array([5.0, 3.0]), motor_coulomb=np.array([2.0, 1.0]),
                       link_viscous=np.array([1.0, 0.5]), link_coulomb=np.array([3.0, 1.5]))
    bags = [syn.as_bag(syn.simulate(arm, duration=2.0, step=2e-4, sample_every=5, seed=s), f"b{s}") for s in range(3)]
    step = float(bags[0].t[1] - bags[0].t[0])
    priors = _priors(2, step, 0.3)
    rows = [fair.BagRows.build(b, rigid_model, 5, 3) for b in bags]
    problem = fair.TwoMassProblem(rows[:2], rows[2:], rigid_model, priors, np.array([0.5, 2.0]), epsilon=1e-2,
                                  weighted=True, device="cpu", chunk=3)
    pi = fair.nominal_full(rigid_model, 2)[fair.FairLayout(2).inertial_slice()]
    theta = np.concatenate([fair.log_cholesky_from_inertial(b) for b in pi.reshape(-1, 10)])
    x = np.concatenate([theta, [1.0, 0.5, 3.0, 1.5], *priors.logits(np.array([2000.0, 1500.0]), np.array([0.2, 0.3]))])
    _, _, g = problem.evaluate(x, jacobian="all")
    for k in range(len(x)):
        e = 1e-6 * max(1.0, abs(x[k]))
        plus, minus = x.copy(), x.copy()
        plus[k] += e
        minus[k] -= e
        fd = (problem.evaluate(plus) - problem.evaluate(minus)) / (2 * e)
        assert 2 * g[k] == pytest.approx(fd, rel=1e-4, abs=1e-9 * max(1.0, np.abs(2 * g).max())), k


@pytest.mark.skipif(not UR10.exists(), reason="needs the r7 UR10 file")
def test_fg2_no_privileged_reads():
    """FG2: noise in every column but t, bag, split, q, dq, ft leaves the fit and the predictions bit-identical."""
    import pyarrow.parquet as pq

    from elastic_sim import identification as idn
    from elastic_sim.assets import AssetRegistry
    from elastic_sim.zollo.data import sidecar

    names = ["t0_e01_f0_mujoco", "t1_e02_f0_mujoco", "t0_e00_f0_mujoco", "t0_e07_f0_mujoco"]
    frame = pq.read_table(UR10, filters=[("bag", "in", names)]).to_pandas()
    noisy = frame.copy()
    rng = np.random.default_rng(7)
    keep = set(fair.fair_columns(6))
    for column in noisy.columns:
        if column not in keep:
            noisy[column] = rng.normal(size=len(noisy)) if noisy[column].dtype.kind in "fiu" else "noise"
    privileged = [c for c in frame.columns if c.startswith(("q_link", "dq_link", "tau", "defl", "stiffness__"))
                  or "_clean" in c]
    assert privileged and not (noisy[privileged].to_numpy() == frame[privileged].to_numpy()).all()
    rigid_model = idn.build_model(AssetRegistry.for_repository(_REPO).load("ur10_table"))
    priors = fair.FairPriors.from_contract(sidecar(UR10, "contract"), 0.002)
    settings = fair.OutputErrorSettings(stage1_iterations=3, stage2_iterations=1, stage3_iterations=1, device="cpu")
    results = []
    for data in (frame, noisy):
        train = [fair.BagRows.build(b, rigid_model, 5, 3) for b in fair.bags_from_frame(data, 6, ("train",))]
        val = [fair.BagRows.build(b, rigid_model, 5, 3) for b in fair.bags_from_frame(data, 6, ("val",))]
        test = fair.bags_from_frame(data, 6, ("test",))
        scale = fair.target_scale([r.bag for r in train])
        rigid = fair.fit_rigid(train, val, rigid_model, epsilon=1e-2, weighted=True, scale=scale)
        two_mass = fair.fit_two_mass(rigid, train, val, rigid_model, priors, scale, settings=settings, log=None)
        rows = fair.BagRows.build(test[0], rigid_model, 5, 3)
        prediction = two_mass.model.predict(rigid_model, rows.theta, rows.dtheta, rows.ddtheta, 0.002)["tau"]
        results.append((rigid.model.full, two_mass.model.full, two_mass.model.stiffness, two_mass.model.zeta,
                        prediction))
    for clean, perturbed in zip(*results):
        assert np.array_equal(clean, perturbed)
