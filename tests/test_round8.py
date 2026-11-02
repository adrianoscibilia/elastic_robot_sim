"""Round 8 (`identified_filter/01` Sec 1): regimes, band, bags per split, out-of-prior robots, chirps, the oracle fit."""

from __future__ import annotations

import importlib.util
import math
import types
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from elastic_sim.dataset import load_config
from elastic_sim.dataset_worklist import (FEASIBILITY_RESEEDS, bag_excitation, bag_wanted, collision_payload,
                                          iter_conditions, regime_excitation, reseed_attempt, settle_trajectories,
                                          split_labels_of, trajectory_key, trajectory_seed, trajectories_for)
from elastic_sim.excitation import _cap_probe_lines, project_coefficients
from elastic_sim.payload import Payload
from elastic_sim.round6_checks import gate_gainshift
from elastic_sim.round8 import (ChirpEvaluator, OopSpec, coverage_report, fit_two_mass_multistart, fit_usable, modal_trajectory,
                                regime_draw, sample_oop_robots)
from elastic_sim.two_mass import filter_sequence

CONFIGS = Path(__file__).resolve().parents[1] / "config" / "identification"
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


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


# ------------------------------------------------------- identified_filter/03 ---

def _build_round6():
    spec = importlib.util.spec_from_file_location("build_round6_under_test", SCRIPTS / "build_round6.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ASSET = types.SimpleNamespace(name="arm", urdf_path="arm.urdf")


def _work(config, *indices):
    tiers = [t for t in config.tiers if not t.is_rigid][:len(indices)]
    return [(tier, index, Payload()) for tier, index in zip(tiers, indices)]


def _map_failing(failing):
    """A stand-in for the pool: ``None`` (no feasible trajectory) for every (tier, attempt) in ``failing``."""
    def fake(_, jobs):
        return [None if failing(tier.name, reseed_attempt(config, tier.name, index)) else object()
                for config, _, tier, index, _ in jobs]
    return fake


def test_infeasible_bag_is_reseeded_until_one_seed_fits(ur10):
    work = _work(ur10, 0, 1)
    name = work[0][0].name
    config, dropped = settle_trajectories(ur10, _ASSET, work, {}, _map_failing(lambda t, a: t == name and a < 2))
    assert dropped == [] and config.excluded_bags == ()
    assert reseed_attempt(config, name, 0) == 2
    assert reseed_attempt(config, work[1][0].name, 1) == 0


def test_bag_infeasible_at_every_seed_is_dropped_and_recorded_not_raised(ur10):
    work = _work(ur10, 0, 0)
    name, kept = work[0][0].name, work[1][0].name
    tried = []

    def fake(function, jobs):
        jobs = list(jobs)
        tried.extend(job[2].name for job in jobs if job[2].name == name)
        return _map_failing(lambda t, a: t == name)(function, jobs)

    config, dropped = settle_trajectories(ur10, _ASSET, work, {}, fake)
    assert len(tried) == 1 + FEASIBILITY_RESEEDS
    assert dropped == [{"tier": name, "trajectory": 0, "tries": 1 + FEASIBILITY_RESEEDS,
                        "reason": "no feasible excitation trajectory"}]
    assert config.excluded_bags == ((name, 0),) and reseed_attempt(config, name, 0) == 0
    tiers = {t.name: t for t in config.tiers}
    assert not bag_wanted(config, tiers[name], 0) and bag_wanted(config, tiers[kept], 0)
    assert not any(tier.name == name and index == 0 for index, tier, _, _ in iter_conditions(config))


def test_gainshift_runs_the_production_files_final_seeds(ur10):
    build = _build_round6()
    test_tier = next(t for t in ur10.tiers if split_labels_of(ur10)[t.name] == "test")
    dropped_tier = [t for t in ur10.tiers if split_labels_of(ur10)[t.name] == "test"][1]
    production = {"reseed": [[test_tier.name, 0, 2]], "excluded_bags": [[dropped_tier.name, 0]]}
    plain = build.variant_config(ur10, "gainshift")
    paired = build.paired_with_production(plain, production)
    built = replace(ur10, reseed=((test_tier.name, 0, 2),))
    assert trajectory_seed(paired, test_tier, 0) == trajectory_seed(built, test_tier, 0)
    assert trajectory_seed(plain, test_tier, 0) != trajectory_seed(built, test_tier, 0)
    # One trajectory, so gate 4 compares the same digests; the production-dropped bag is not in the twin at all.
    assert trajectory_key(paired, _ASSET, test_tier, 0, Payload()) == trajectory_key(built, _ASSET, test_tier, 0, Payload())
    assert not bag_wanted(paired, dropped_tier, 0)


def test_gate_4_accepts_the_paired_file_and_refuses_an_unpaired_one():
    production = {"split": {"test": ["e1"]}, "records": [{"bag": "t0_e1", "tier": "e1", "trajectory_digest": "reseeded"}],
                  "gates": {}}
    twin = {"records": [{"bag": "t0_e1", "tier": "e1", "trajectory_digest": "reseeded", "control_natural_frequency": 1.0,
                         "omega_fraction": 0.4}]}
    assert gate_gainshift(production, twin)["same_trajectory_digests"]
    twin["records"][0]["trajectory_digest"] = "first seed"
    assert not gate_gainshift(production, twin)["same_trajectory_digests"]


def test_failing_bags_split_into_excitation_and_fit_limited(ur10):
    joint = ur10.round8.verification.main_joints[0]
    others = [row["joint"] for row in ur10.round8.band if row["joint"] != joint]
    row = lambda bag, observable, snr: {  # noqa: E731
        "bag": bag, f"resolvable__{joint}": True, f"observable__{joint}": observable,
        f"ident_K_relerr__{joint}": 0.0 if observable else 0.9, f"elastic_snr__{joint}": snr,
        **{f"{key}__{o}": value for o in others for key, value in
           (("resolvable", False), ("observable", False), ("ident_K_relerr", np.nan), ("elastic_snr", np.nan))}}
    table = pd.DataFrame([row("good", True, 9.0), row("fit", False, 9.0), row("excitation", False, 1.0)])
    report = coverage_report(table, {"good": "full", "fit": "full", "excitation": "full"}, ur10.round8)
    assert report["failing_full_bags"] == ["excitation", "fit"]
    assert report["excitation_limited_bags"] == ["excitation"] and report["fit_limited_bags"] == ["fit"]
    assert report["mode"] == "report"


def test_full_regime_caps_and_their_record(ur10):
    spec = replace(ur10.round8, comb_hz=(0.5, 45.0))
    _, excitation = regime_draw(spec, ur10.excitation, "train", 0, ur10.seed, 7)
    assert min(excitation.probe_harmonics) * excitation.base_frequency >= 1.5
    assert excitation.probe_line_amplitude_max == 0.005
    recorded = spec.as_dict()["full"]
    assert recorded["lowest_line_hz"] == 1.5 and recorded["line_amplitude_max"] == 0.005
    assert regime_draw(spec, ur10.excitation, "test", 2, ur10.seed, 7)[1].probe_line_amplitude_max == 0.0


def test_line_cap_scales_lines_down_and_keeps_rest_at_the_start():
    omega = 2.0 * math.pi * 0.1
    indices = np.arange(15, 60, dtype=float)
    rng = np.random.default_rng(0)
    a, b = project_coefficients(rng.normal(size=(3, 45)), rng.normal(size=(3, 45)), indices=indices)
    cap = 0.005
    # Scaled up so the low lines are far over the cap, as a position-weighted comb with a large budget is.
    before = np.hypot(a, b) / (indices * omega)
    a, b = a * (5.0 * cap / before.max()), b * (5.0 * cap / before.max())
    a_cap, b_cap = _cap_probe_lines(a, b, omega, indices, cap)
    amplitude = np.hypot(a_cap, b_cap) / (indices * omega)
    assert amplitude.max() <= cap * (1.0 + 1e-9)
    assert (amplitude > 0.01 * cap).all()  # scaled down, none dropped
    assert np.abs(a_cap.sum(axis=1)).max() < 1e-9 and np.abs(b_cap @ indices).max() < 1e-9


def test_round8_checks_collisions_with_what_the_plant_carries(ur10):
    assert ur10.payload.mode == "none"
    tier = next(t for t in ur10.tiers if not t.is_rigid)
    box = Payload(mass=3.0, offset=(0.0, 0.0, 0.07), size=0.24)
    assert collision_payload(ur10, box, tier).is_empty
    assert trajectory_key(ur10, _ASSET, tier, 0, box) == trajectory_key(ur10, _ASSET, tier, 0, Payload())
    r7 = load_config(CONFIGS / "ur10_table_round7_nopayload.yaml")
    assert collision_payload(r7, box, tier) is box


def test_sag_failure_is_reported_and_keeps_the_file():
    build = _build_round6()
    sag = {"gate": "2 sag", "ok": False, "failing_bags": ["t0_e1"]}
    report = {"ok": False, "hard_checks": [{"ok": True}], "gates": [sag, {"gate": "3", "ok": True}]}
    build.accept_sag_failures(report)
    assert report["ok"] and report["sag_failing_bags"] == ["t0_e1"] and not sag["ok"]
    report = {"ok": False, "hard_checks": [{"ok": False}], "gates": [dict(sag)]}
    build.accept_sag_failures(report)
    assert not report["ok"]


def test_stiffness_ranges_of_round_3(ur10):
    iiwa = load_config(CONFIGS / "kuka_lbr_iiwa_14_r820_table_drive_round8.yaml")
    r7 = load_config(CONFIGS / "ur10_table_round7_nopayload.yaml")
    assert ur10.transmission.stiffness_factor == r7.transmission.stiffness_factor
    assert ur10.gates.sag_mode == "gate"
    assert iiwa.transmission.stiffness_factor == (0.25, 4.0) and iiwa.gates.sag_mode == "report"


def test_fit_usable_rejects_run_away_and_out_of_contract_fits():
    """03 F3: a fit counts only inside factor 3 of the sampled K, the contract's K and D ranges, and below f_SG."""
    names = list("abcde")
    fit = {f"f_n_filter__{x}": 10.0 for x in names} | {"f_n_filter__e": 200.0}
    fit |= {f"ident_D__{x}": 5.0 for x in names} | {"ident_D__d": 500.0}
    fit |= {"ident_K__a": 1100.0, "ident_K__b": 1e12, "ident_K__c": 2500.0, "ident_K__d": 1000.0, "ident_K__e": 1000.0}
    group = pd.DataFrame({f"stiffness__{x}": [1000.0] for x in names})
    joint = {"stiffness_range": [500.0, 2000.0], "damping_range": [1.0, 50.0]}
    usable = fit_usable(fit, group, names, {"joints": [joint] * 5}, 100.0)
    assert usable.tolist() == [True, False, False, False, False]


def test_b3_comb_over_the_main_joints_capped_and_r8_unchanged():
    """identified_filter/04 B3 (r9): the comb over the main joints' [0.5 f_n,min, 2 f_n,max], capped; r8 as before."""
    from elastic_sim.dataset import load_config
    from elastic_sim.round8 import band_comb

    root = Path(__file__).resolve().parents[1] / "config" / "identification"
    for stem, cap in (("ur10_table", 25.0), ("kuka_lbr_iiwa_14_r820_table_drive", 40.0)):
        r8, r9 = (load_config(root / f"{stem}_round{r}.yaml") for r in (8, 9))
        assert r8.round8.full.band_joints == () and "band_joints" not in r8.round8.as_dict()["full"]
        rows = [row for row in r9.round8.band if row["joint"] in r9.round8.full.band_joints]
        assert r9.round8.comb_hz == band_comb(r9.round8.band, r9.round8.full)
        assert r9.round8.comb_hz[0] == min(0.5 * row["f_n_min_hz"] for row in rows)
        assert r9.round8.comb_hz[1] == min(cap, max(2.0 * row["f_n_max_hz"] for row in rows)) == cap
        assert r9.round8.full.acceleration_share == 0.5 and r8.round8.full.acceleration_share == 0.7
        assert r9.plant_extras.transmission_error_tier == "te_randomised"
        assert r9.round8.band == r8.round8.band                       # same robots' modes: only the comb moved
