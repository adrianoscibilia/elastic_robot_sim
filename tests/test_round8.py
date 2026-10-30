"""Round 8 (`identified_filter/01` Sec 1): regimes, band, bags per split, out-of-prior robots, chirps, the oracle fit."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from elastic_sim.dataset import load_config
from elastic_sim.dataset_worklist import (bag_excitation, iter_conditions, regime_excitation, split_labels_of,
                                          trajectory_seed, trajectories_for)
from elastic_sim.round8 import (ChirpEvaluator, OopSpec, fit_two_mass_multistart, modal_trajectory,
                                sample_oop_robots)
from elastic_sim.two_mass import filter_sequence

CONFIGS = Path(__file__).resolve().parents[1] / "config" / "identification"


@pytest.fixture(scope="module")
def ur10():
    return load_config(CONFIGS / "ur10_table_round8.yaml")


def test_r7_config_has_no_round8_and_keeps_its_excitation():
    config = load_config(CONFIGS / "ur10_table_round7_nopayload.yaml")
    assert config.round8 is None
    tier = [t for t in config.tiers if not t.is_rigid][0]
    regime, excitation = bag_excitation(config, tier, 1)
    assert regime == "r7"
    assert excitation == regime_excitation(config.excitation, config.regime, config.seed,
                                           trajectory_seed(config, tier, 1))


def test_band_and_comb(ur10):
    spec = ur10.round8
    f_sg = 0.09 / ur10.sample_time_step
    assert spec.comb_hz[1] <= f_sg + 1e-9
    for row in spec.band:
        assert row["resolvable"] == (row["f_n_min_hz"] <= f_sg)
    tier = [t for t in ur10.tiers if not t.is_rigid][0]
    regime, excitation = bag_excitation(ur10, tier, 0)
    assert regime == "full"
    lines = np.asarray(excitation.probe_harmonics) * excitation.base_frequency
    assert lines.min() >= spec.comb_hz[0] - excitation.base_frequency and lines.max() <= spec.comb_hz[1] + 1e-9
    assert excitation.probe_budget == "additive"


def test_regime_draw_is_independent_of_the_robot_k(ur10):
    """The comb must not encode K: the same trajectory seed gives the same regime whatever the stiffness."""
    tier = [t for t in ur10.tiers if not t.is_rigid][3]
    stiffer = replace(tier, stiffness=tuple(3.0 * np.asarray(tier.stiffness)))
    for index in range(3):
        assert bag_excitation(ur10, tier, index, "test") == bag_excitation(ur10, stiffer, index, "test")


def test_bags_per_split_and_regimes(ur10):
    labels = split_labels_of(ur10)
    counts = Counter(labels.values())
    assert counts["test"] == 30 and counts["val"] == 30
    regimes = Counter()
    for index, tier, _, _ in iter_conditions(ur10):
        regimes[(labels[tier.name], bag_excitation(ur10, tier, index)[0])] += 1
    for regime in ("full", "real_like", "off"):
        assert regimes[("test", regime)] == 30
    assert regimes[("val", "full")] == 30 and regimes[("val", "real_like")] + regimes[("val", "off")] == 30
    for tier in ur10.tiers:
        assert trajectories_for(ur10, tier, labels) == (3 if labels[tier.name] == "test" else 2)


def test_real_like_is_the_p1_comb(ur10):
    tier = [t for t in ur10.tiers if split_labels_of(ur10)[t.name] == "test"][0]
    regime, excitation = bag_excitation(ur10, tier, 1)
    assert regime == "real_like"
    assert excitation.base_frequency == 0.2 and excitation.n_periods == 2
    assert excitation.probe_harmonics[0] * 0.2 == pytest.approx(1.6)
    assert excitation.probe_harmonics[-1] * 0.2 == pytest.approx(9.6)
    assert excitation.probe_budget == "split" and excitation.probe_weighting == "acceleration"


def test_reseed_moves_only_that_bag(ur10):
    tier = [t for t in ur10.tiers if not t.is_rigid][0]
    other = [t for t in ur10.tiers if not t.is_rigid][1]
    moved = replace(ur10, reseed=((tier.name, 0, 1),))
    assert trajectory_seed(moved, tier, 0) != trajectory_seed(ur10, tier, 0)
    assert trajectory_seed(moved, tier, 1) == trajectory_seed(ur10, tier, 1)
    assert trajectory_seed(moved, other, 0) == trajectory_seed(ur10, other, 0)


def test_oop_robots_lie_outside_the_prior(ur10):
    sampling = ur10.transmission
    robots = sample_oop_robots(sampling, ur10.seed, OopSpec(robots=10))
    low = np.asarray(sampling.stiffness_nominal) * sampling.stiffness_factor[0]
    high = np.asarray(sampling.stiffness_nominal) * sampling.stiffness_factor[1]
    for robot in robots:
        k = np.asarray(robot.stiffness)
        assert np.all((k <= low * 0.67 + 1e-6) | (k >= high * 1.5 - 1e-6))


def test_chirp_law_and_rest(ur10):
    spec = ur10.round8.modal
    evaluator = ChirpEvaluator(spec.poses[0][1], 1, 1.5, 10.0, 30.0, spec)
    f = np.geomspace(1.5, 10.0, 50)
    amplitude = evaluator.amplitude(f)
    assert np.all(amplitude <= spec.position_amplitude_max + 1e-15)
    assert np.all(amplitude * (2 * np.pi * f) ** 2 <= spec.a_max + 1e-12)
    assert np.all(amplitude * (2 * np.pi * f) ** 3 <= spec.jerk_max + 1e-9)
    from elastic_sim.assets import AssetRegistry

    asset = AssetRegistry.for_repository().load(ur10.asset)
    trajectory = modal_trajectory(spec, asset, 0, 0.002)
    assert len(spec.segments) == len(spec.joints) * len(spec.poses) * spec.repeats
    assert np.allclose(trajectory.position[0], spec.poses[0][1]) and np.allclose(trajectory.position[-1], spec.poses[0][1])
    assert np.abs(trajectory.velocity[0]).max() < 1e-9
    moving = np.flatnonzero(np.abs(trajectory.position - trajectory.position[0]).max(axis=0) > 0)
    assert list(moving) == [asset.joint_names.index(spec.segments[0][0])]


def test_coupled_fit_recovers_k_and_d():
    rng = np.random.default_rng(0)
    steps, n, h = 3000, 3, 0.002
    t = np.arange(steps) * h
    tau = np.column_stack([sum(np.sin(2 * np.pi * f * t + rng.uniform(0, 6)) for f in np.geomspace(2, 40, 25))
                           for _ in range(n)])
    inertia = np.tile(np.array([[3.0, 0.4, 0.1], [0.4, 2.0, 0.2], [0.1, 0.2, 1.0]]), (steps, 1, 1))
    K, D = np.array([2e4, 1.5e4, 8e3]), np.array([20.0, 15.0, 6.0])
    target = filter_sequence(tau, inertia, K, D, h, coupling="full")["tau_s"]
    k, d = fit_two_mass_multistart(tau, inertia, target, np.ones(steps, bool), h, K, D, np.ones(n, bool))
    assert np.allclose(k / K, 1.0, atol=1e-3) and np.allclose(d / D, 1.0, atol=1e-2)
