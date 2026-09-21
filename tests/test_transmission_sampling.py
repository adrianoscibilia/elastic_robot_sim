"""Parameter-sampling invariants for the identification dataset (R3_03).

Pure numpy/dataclass logic -- no simulator backend needed, so this file runs
in every environment, including one without Pinocchio/MuJoCo installed.
"""

import os
import sys

import numpy as np
import pytest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_REPO, "src"))

from elastic_sim.dataset import TransmissionSampling, sample_robots

INTERVALS = ((1.5e4, 3.5e4), (1.5e4, 3.5e4), (1.0e4, 2.5e4), (1.0e4, 2.5e4),
             (5.0e3, 1.5e4), (3.0e3, 1.0e4), (3.0e3, 1.0e4))


def _coverage(robots, intervals):
    k = np.array([r.stiffness for r in robots])
    lo, hi = np.array(intervals).T
    realized = np.log(k.max(axis=0) / k.min(axis=0))
    declared = np.log(hi / lo)
    return realized / declared


def test_iid_reproduces_the_shipped_robots():
    """Regression guard: the documented seed/index promise must not move."""
    sampling = TransmissionSampling(robots=6, stiffness=INTERVALS, sampling="iid")
    k0 = np.array(sample_robots(sampling, 20260917)[0].stiffness)
    expected = [32170.0, 25873.0, 23972.0, 23342.0, 10875.0, 4363.0, 5676.0]
    assert np.allclose(k0, expected, rtol=2e-4)


def test_iid_undercovers_and_stratified_does_not():
    iid = sample_robots(TransmissionSampling(robots=6, stiffness=INTERVALS, sampling="iid"), 20260917)
    strat = sample_robots(TransmissionSampling(robots=20, stiffness=INTERVALS, sampling="stratified"), 20260917)
    assert _coverage(iid, INTERVALS).min() < 0.6  # documents the finding
    assert _coverage(strat, INTERVALS).min() > 0.90  # documents the fix


def test_stratified_does_not_correlate_joints():
    robots = sample_robots(TransmissionSampling(robots=64, stiffness=INTERVALS, sampling="stratified"), 7)
    k = np.log(np.array([r.stiffness for r in robots]))
    corr = np.corrcoef(k.T)
    off = corr[~np.eye(len(INTERVALS), dtype=bool)]
    assert np.abs(off).max() < 0.35, "strata must be permuted independently per joint"


def test_sobol_also_covers_well_and_is_prefix_stable():
    pytest.importorskip("scipy.stats.qmc")
    sobol_20 = sample_robots(TransmissionSampling(robots=20, stiffness=INTERVALS, sampling="sobol"), 20260917)
    assert _coverage(sobol_20, INTERVALS).min() > 0.75
    sobol_6 = sample_robots(TransmissionSampling(robots=6, stiffness=INTERVALS, sampling="sobol"), 20260917)
    for a, b in zip(sobol_6, sobol_20[:6]):
        assert np.allclose(a.stiffness, b.stiffness), "sobol must be prefix-stable like iid"


def test_legacy_interval_form_still_loads_and_matches_iid_directly():
    """The historical [[min, max], ...] form keeps working without conversion."""
    sampling = TransmissionSampling(robots=3, stiffness=INTERVALS)  # default sampling="iid"
    a = sample_robots(sampling, 11)
    b = sample_robots(sampling, 11)
    assert all(np.array_equal(x.stiffness, y.stiffness) for x, y in zip(a, b))


def test_nominal_and_factor_form_is_validated():
    with pytest.raises(ValueError, match="both"):
        TransmissionSampling(robots=1, stiffness=INTERVALS, stiffness_nominal=(1.0,) * 7)
    with pytest.raises(ValueError, match="stiffness_nominal"):
        TransmissionSampling(robots=1)  # neither form given
    sampling = TransmissionSampling(
        robots=8, stiffness_nominal=(2.4e4,) * 7, stiffness_factor=(0.5, 2.0), sampling="stratified",
    )
    robots = sample_robots(sampling, 5)
    k = np.array([r.stiffness for r in robots])
    assert (k >= 2.4e4 * 0.5).all() and (k <= 2.4e4 * 2.0).all()


def test_common_fraction_controls_joint_correlation():
    independent = sample_robots(TransmissionSampling(
        robots=64, stiffness_nominal=(2.4e4,) * 7, stiffness_factor=(0.25, 4.0),
        stiffness_common_fraction=0.0, sampling="stratified",
    ), 3)
    correlated = sample_robots(TransmissionSampling(
        robots=64, stiffness_nominal=(2.4e4,) * 7, stiffness_factor=(0.25, 4.0),
        stiffness_common_fraction=1.0, sampling="stratified",
    ), 3)
    k_ind = np.log(np.array([r.stiffness for r in independent]))
    k_cor = np.log(np.array([r.stiffness for r in correlated]))
    off_ind = np.corrcoef(k_ind.T)[~np.eye(7, dtype=bool)]
    off_cor = np.corrcoef(k_cor.T)[~np.eye(7, dtype=bool)]
    assert np.abs(off_ind).max() < 0.35
    assert off_cor.min() > 0.9, "common_fraction=1 must give a one-dimensional family"


def test_rotor_inertia_is_fixed_unless_nominal_is_given():
    fixed = sample_robots(TransmissionSampling(robots=5, stiffness=INTERVALS, rotor_inertia=(0.25,) * 7), 3)
    assert all(np.allclose(r.rotor_inertia, 0.25) for r in fixed)

    sampled = sample_robots(TransmissionSampling(
        robots=5, stiffness=INTERVALS, rotor_inertia_nominal=(1.0,) * 7, rotor_inertia_factor=(0.5, 2.0),
    ), 3)
    rotor = np.array([r.rotor_inertia for r in sampled])
    assert np.std(rotor, axis=0).min() > 0.0, "rotor inertia must vary across robots when sampled"
    assert (rotor >= 0.5).all() and (rotor <= 2.0).all()


def test_rotor_inertia_stream_does_not_perturb_the_stiffness_stream():
    without = sample_robots(TransmissionSampling(robots=6, stiffness=INTERVALS), 20260917)
    with_rotor = sample_robots(TransmissionSampling(
        robots=6, stiffness=INTERVALS, rotor_inertia_nominal=(1.0,) * 7, rotor_inertia_factor=(0.5, 2.0),
    ), 20260917)
    assert all(np.array_equal(a.stiffness, b.stiffness) for a, b in zip(without, with_rotor))


def test_unknown_sampling_method_is_rejected():
    with pytest.raises(ValueError, match="sampling"):
        TransmissionSampling(robots=1, stiffness=INTERVALS, sampling="bogus")


def test_unknown_provenance_class_is_rejected():
    with pytest.raises(ValueError, match="provenance"):
        TransmissionSampling(robots=1, stiffness=INTERVALS, stiffness_provenance=("P", "X", "C", "C", "C", "E", "E"))
