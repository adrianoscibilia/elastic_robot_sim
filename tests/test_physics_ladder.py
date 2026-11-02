"""identified_filter/04 A1, A2, C1 and 05 step 1: the ladder's rungs, gate, gating and alignment; the te_randomised
tier; the time-varying filter; the TE tracker; the summary's seed."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]


def _load(relative: str, name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ladder():
    return _load("tools/r8_physics_ladder.py", "r8_physics_ladder_under_test")


@pytest.fixture(scope="module")
def te_band():
    return _load("tools/r8_te_band.py", "r8_te_band_under_test")


@pytest.fixture(scope="module")
def summary():
    return _load("scripts/r8_summary.py", "r8_summary_under_test")


@pytest.fixture(scope="module")
def ur10(ladder):
    from elastic_sim.dataset import load_config

    return load_config(REPO / "config/identification/ur10_table_round8.yaml")


def test_rungs_switch_on_one_feature_each(ladder, ur10):
    rung = {name: ladder.rung_config(ur10, name) for name in ladder.RUNGS}
    assert rung["L4"] is ur10
    assert not rung["L0"].noise.enabled and rung["L1"].noise.enabled
    assert not rung["L1"].plant_extras.has_nonlinear_spring and rung["L2"].plant_extras.has_nonlinear_spring
    assert not rung["L2"].plant_extras.has_transmission_error and rung["L3"].plant_extras.has_transmission_error
    assert rung["L3"].plant_extras.ripple_amplitude == 0.0
    # The r8 configs set no ripple, which is why the report reads L4 as L3.
    assert ur10.plant_extras.ripple_amplitude == 0.0


def test_l3p_is_l2_plus_the_tier(ladder, ur10):
    l3p = ladder.rung_config(ur10, "L3p").plant_extras
    assert l3p.transmission_error_tier == "te_randomised" and l3p.has_nonlinear_spring and l3p.ripple_amplitude == 0.0
    assert l3p.transmission_error_amplitude == ur10.plant_extras.transmission_error_amplitude
    assert ladder.rung_config(ur10, "L3p").noise.enabled
    # The tier never reaches r8's own rungs.
    for rung in ("L0", "L1", "L2", "L3", "L4"):
        assert ladder.rung_config(ur10, rung).plant_extras.transmission_error_tier == "fixed"


def _bag_extras(ladder, config, robot: str, seed: int = 11, stiffness=None):
    from elastic_sim.dataset import plant_extras_for_bag

    stiffness = np.full(6, 2.0e4) if stiffness is None else stiffness
    return plant_extras_for_bag(ladder.rung_config(config, "L3p").plant_extras, 6, config.seed, seed, robot=robot,
                                stiffness=stiffness, effort=np.full(6, 300.0)).transmission_error


def test_tier_draws_within_the_ranges_per_robot_with_order_n_half_of_2n(ladder, ur10):
    bound = 2.0e-4
    a, b = _bag_extras(ladder, ur10, "e001"), _bag_extras(ladder, ur10, "e002")
    assert a.amplitude != b.amplitude and a.amplitude_n != b.amplitude_n
    draws = [_bag_extras(ladder, ur10, f"e{i:03d}") for i in range(200)]
    two_n, one_n = (np.array([getattr(d, name) for d in draws]) / bound for name in ("amplitude", "amplitude_n"))
    assert two_n.min() >= 0.1 and two_n.max() <= 1.0 and one_n.min() >= 0.05 and one_n.max() <= 0.5
    assert np.median(two_n) == pytest.approx(np.sqrt(0.1), rel=0.1)
    assert a.order == (200.0,) * 6 and a.order_n == (100.0,) * 6


def test_tier_amplitude_is_per_robot_and_the_phase_per_bag(ladder, ur10):
    one, other_bag = _bag_extras(ladder, ur10, "e001", seed=11), _bag_extras(ladder, ur10, "e001", seed=12)
    assert one.amplitude == other_bag.amplitude and one.amplitude_n == other_bag.amplitude_n
    assert one.phase != other_bag.phase and one.phase_n != other_bag.phase_n and one.phase != one.phase_n


def test_tier_draws_are_independent_of_k(ladder, ur10):
    # By construction: a robot's amplitudes do not read its stiffness at all...
    soft, stiff = (_bag_extras(ladder, ur10, "e001", stiffness=np.full(6, k)) for k in (1.0e3, 1.0e5))
    assert soft.amplitude == stiff.amplitude and soft.amplitude_n == stiff.amplitude_n
    # ... and across the build's robots, whose stiffness comes from another stream, no correlation shows.
    robots = [t for t in ur10.tiers if not t.is_rigid]
    assert len(robots) == 180
    k = np.log([t.stiffness for t in robots])
    for name in ("amplitude", "amplitude_n"):
        amplitude = np.log([getattr(_bag_extras(ladder, ur10, t.name), name) for t in robots])
        for j in range(6):
            assert abs(np.corrcoef(k[:, j], amplitude[:, j])[0, 1]) < 3.0 / np.sqrt(len(robots))


def test_off_tier_draws_nothing_new_and_keeps_the_single_line(ur10):
    from elastic_sim.dataset import plant_extras_for_bag
    from elastic_sim.plant_extras import TransmissionError

    assert ur10.plant_extras.transmission_error_tier == "fixed"
    extras = plant_extras_for_bag(ur10.plant_extras, 6, ur10.seed, 11, robot="e001", stiffness=np.full(6, 2.0e4),
                                  effort=np.full(6, 300.0))
    error = extras.transmission_error
    assert error.amplitude == (2.0e-4,) * 6 and error.amplitude_n == () and error.phase_n == ()
    # Stream 12's first six draws, as r8 made them.
    phase = np.random.default_rng((ur10.seed, 12, 11)).uniform(0.0, 2.0 * np.pi, size=6)
    assert error.phase == tuple(phase)
    theta = np.linspace(-1.0, 1.0, 6)
    assert np.array_equal(error.error(theta), 2.0e-4 * np.sin(200.0 * theta + phase))
    assert np.array_equal(error.slope(theta), 2.0e-4 * 200.0 * np.cos(200.0 * theta + phase))
    assert TransmissionError().is_empty


def test_two_line_error_is_the_sum_and_slope_its_derivative():
    from elastic_sim.plant_extras import TransmissionError

    error = TransmissionError(amplitude=(2e-4,), order=(320.0,), phase=(0.3,), amplitude_n=(1e-4,), order_n=(160.0,),
                              phase_n=(1.1,))
    theta = np.linspace(0.0, 0.2, 2001).reshape(-1, 1)
    expected = 2e-4 * np.sin(320.0 * theta + 0.3) + 1e-4 * np.sin(160.0 * theta + 1.1)
    assert np.array_equal(error.error(theta), expected)
    numeric = np.gradient(expected[:, 0], theta[:, 0])
    assert np.max(np.abs(error.slope(theta)[1:-1, 0] - numeric[1:-1])) < 1e-3 * 0.08
    assert error.max_slope() == pytest.approx(2e-4 * 320.0 + 1e-4 * 160.0)


def test_tier_reaches_the_explicit_bound_and_the_contract(ladder, ur10):
    from elastic_sim.dataset_bounds import explicit_term_table
    from elastic_sim.dataset_io import te_tier_contract
    from dataclasses import asdict
    from diagnose_controller_modes import _load_asset

    asset = _load_asset(ur10.asset)
    damping = lambda c: {r.joint: r.value for r in explicit_term_table(c, asset)  # noqa: E731
                         if r.term == "spring damping (explicit part)"}
    tier, fixed = damping(ladder.rung_config(ur10, "L3p")), damping(ur10)
    # Top of both ranges: slope A (2n + 0.5 n) = 1.25 x A 2n, and the term goes with its square.
    assert tier["elbow_joint"] == pytest.approx(1.25 ** 2 * fixed["elbow_joint"])
    assert te_tier_contract(asdict(ur10.plant_extras)) is None
    contract = te_tier_contract(asdict(ladder.rung_config(ur10, "L3p").plant_extras))
    assert contract["tier"] == "te_randomised" and contract["amplitude_2n_factor"] == (0.1, 1.0)
    assert "not calibrated" in contract["justification"]


def test_config_rejects_a_bad_tier(ur10):
    from dataclasses import replace

    with pytest.raises(ValueError, match="tier must be one of"):
        replace(ur10.plant_extras, transmission_error_tier="calibrated")
    with pytest.raises(ValueError, match="amplitude_n_factor"):
        replace(ur10.plant_extras, transmission_error_tier="te_randomised")


def test_knee_factor_is_the_catalogue_outside_the_blend_and_smooth_across():
    from elastic_sim.two_mass import knee_factor

    factors, knees = np.array([[0.69, 1.0, 1.44]]), np.array([[29.0, 108.0]])
    at = lambda tau: knee_factor(np.array([[tau]]), factors, knees)[0, 0]  # noqa: E731
    assert at(0.0) == 0.69 and at(29.0 * 0.9) == 0.69
    assert at(29.0 * 1.1) == pytest.approx(1.0) and at(108.0 * 0.9) == pytest.approx(1.0)
    assert at(108.0 * 1.1) == pytest.approx(1.44) and at(-500.0) == pytest.approx(1.44)
    assert at(29.0) == pytest.approx(0.5 * (0.69 + 1.0))
    sweep = np.linspace(0.0, 200.0, 20001).reshape(-1, 1)
    curve = knee_factor(sweep, factors, knees)[:, 0]
    assert np.all(np.diff(curve) >= -1e-12)
    assert np.max(np.abs(np.diff(curve))) < 1.5 * 0.31 * (sweep[1, 0] / (2 * 0.1 * 29.0))   # smoothstep's peak slope


def test_constant_stiffness_given_per_sample_changes_nothing():
    from elastic_sim.two_mass import filter_sequence

    t = np.arange(0.0, 2.0, 1e-3)
    tau = np.column_stack([5.0 * np.sin(2 * np.pi * 3.0 * t), 2.0 * np.sin(2 * np.pi * 7.0 * t)])
    inertia, k, d = np.tile([2.0, 1.0], (len(t), 1)), np.array([2e4, 1e4]), np.array([20.0, 10.0])
    for integrator in ("trapezoidal", "backward_euler"):
        one = filter_sequence(tau, inertia, k, d, 1e-3, integrator=integrator)["tau_s"]
        other = filter_sequence(tau, inertia, np.tile(k, (len(t), 1)), d, 1e-3, integrator=integrator)["tau_s"]
        assert np.array_equal(one, other)


def test_tangent_spring_equals_the_total_one_at_constant_stiffness():
    from elastic_sim.two_mass import filter_sequence

    t = np.arange(0.0, 2.0, 1e-3)
    tau = np.column_stack([5.0 * np.sin(2 * np.pi * 3.0 * t), 2.0 * np.sin(2 * np.pi * 7.0 * t)])
    inertia, k, d = np.tile([2.0, 1.0], (len(t), 1)), np.array([2e4, 1e4]), np.array([20.0, 10.0])
    total = filter_sequence(tau, inertia, k, d, 1e-3)["tau_s"]
    tangent = filter_sequence(tau, inertia, k, d, 1e-3, spring="tangent")["tau_s"]
    assert tangent == pytest.approx(total, rel=1e-8, abs=1e-8)


def test_tangent_spring_has_no_step_at_a_knee_and_follows_the_curve():
    from elastic_sim.plant_extras import StiffnessNonlinearity
    from elastic_sim.two_mass import filter_sequence, knee_factor

    # A slow ramp through a knee: the true spring's deflection keeps the torque continuous.
    h, steps, k0 = 1e-3, 4000, 1e4
    factors, knees = np.array([[0.7, 1.0, 1.4]]), np.array([[20.0, 60.0]])
    tau = np.linspace(0.0, 100.0, steps).reshape(-1, 1)
    k = k0 * knee_factor(tau, factors, knees)
    args = (np.full((steps, 1), 1.0), k, np.array([5.0]), h)
    total, tangent = (filter_sequence(tau, *args, spring=spring) for spring in ("total", "tangent"))
    # Quasi-static: both carry the load.
    assert tangent["tau_s"][-1, 0] == pytest.approx(100.0, abs=0.5)
    # Only the tangent form's deflection is the catalogue curve's.
    spring = StiffnessNonlinearity.from_torque_knees(factors, knees, np.array([k0]))
    assert spring.torque(tangent["delta"][-1:], np.array([k0]))[0, 0] == pytest.approx(100.0, rel=0.02)
    assert abs(spring.torque(total["delta"][-1:], np.array([k0]))[0, 0] - 100.0) > 5.0
    # And the step: crossing the knee within about a period of the mode, from rest above the first knee.
    tau = np.interp(np.arange(steps) * h, [0.0, 1.0, 1.05, 4.0], [50.0, 50.0, 70.0, 70.0]).reshape(-1, 1)
    k = k0 * knee_factor(tau, factors, knees)
    args = (np.full((steps, 1), 1.0), k, np.array([5.0]), h)
    total, tangent = (filter_sequence(tau, *args, spring=spring)["tau_s"] for spring in ("total", "tangent"))
    assert np.abs(tangent - tau).max() < 0.5 * np.abs(total - tau).max()


def test_catalogue_curve_is_the_plants_spring_and_its_inverse():
    from elastic_sim.plant_extras import StiffnessNonlinearity
    from elastic_sim.two_mass import catalogue_deflection, catalogue_torque

    factors, knees = np.array([[0.69, 1.0, 1.44], [0.74, 1.0, 1.68]]), np.array([[29.0, 108.0], [14.0, 48.0]])
    k0 = np.array([2.6e4, 1.23e4])
    deflection = np.random.default_rng(0).normal(0.0, 0.01, (2000, 2))
    plant = StiffnessNonlinearity.from_torque_knees(factors, knees, k0).torque(deflection, k0)
    assert np.abs(catalogue_torque(deflection, k0, factors, knees) - plant).max() < 1e-9
    assert np.abs(catalogue_deflection(plant, k0, factors, knees) - deflection).max() < 1e-15


def test_tangent_step_and_its_reversal_cancel_inside_a_blend():
    """06 Sec 1.2: the step's mean slope, so an oscillation inside a knee's blend does not ratchet S up the curve."""
    from elastic_sim.two_mass import filter_sequence, knee_factor

    h, steps, k0 = 1e-3, 3000, 1e4
    factors, knees = np.array([[1.0, 1.5]]), np.array([[50.0]])
    t = np.arange(steps) * h
    # A ramp into the blend, then a lightly damped ring there, then the ramp back out to the start.
    tau = (np.interp(t, [0.0, 0.5, 2.5, 3.0], [40.0, 50.0, 50.0, 40.0])
           + 2.0 * np.sin(2 * np.pi * 15.0 * t) * ((t > 0.5) & (t < 2.5))).reshape(-1, 1)
    # The region follows the filter's own S (iterated to its fixed point), so what is left is the discretisation's.
    selector = tau
    for _ in range(12):
        k = k0 * knee_factor(selector, factors, knees, 0.1)
        out = filter_sequence(tau, np.full((steps, 1), 1.0), k, np.array([1.0]), h, spring="tangent",
                              initial_deflection=np.array([0.004]))
        selector = out["tau_s"] - 1.0 * out["delta_dot"]
    spring = selector
    # Back below the blend, the spring sits on the same straight segment it started on: S - K_0 delta at both ends.
    start, end = spring[0, 0] - k0 * out["delta"][0, 0], spring[-1, 0] - k0 * out["delta"][-1, 0]
    assert abs(end - start) < 0.02
    with pytest.raises(ValueError, match="initial_deflection"):
        filter_sequence(tau, np.full((steps, 1), 1.0), k, np.array([1.0]), h, initial_deflection=np.array([0.0]))


def test_ltv_recursion_follows_a_step_in_stiffness_at_the_new_frequency():
    from elastic_sim.two_mass import filter_sequence

    # A free oscillation after a stiffness change rings at sqrt(K / M) of the new K.
    h, m, steps = 1e-4, 1.0, 40000
    k = np.full((steps, 1), 1000.0)
    k[steps // 2:] = 4000.0
    tau = np.zeros((steps, 1))
    tau[:50] = 100.0
    out = filter_sequence(tau, np.full((steps, 1), m), k, np.zeros(1), h)["delta"][:, 0]
    def frequency(x):
        return (np.sum(np.diff(np.sign(x)) != 0) / 2.0) / (len(x) * h)
    assert frequency(out[1000:steps // 2]) == pytest.approx(np.sqrt(1000.0) / (2 * np.pi), rel=0.02)
    assert frequency(out[steps // 2:]) == pytest.approx(np.sqrt(4000.0) / (2 * np.pi), rel=0.02)


def _gating_rows(joint, content, floor, content_l3p=None):
    rows = []
    for rung, candidate, value in (("L0", "oracle_effective", floor), ("L0", "rigid", content), ("L2", "rigid", content),
                                   ("L3p", "rigid", content_l3p if content_l3p else content)):
        for bag in ("a", "b"):
            rows.append({"bag": bag, "regime": "full", "rung": rung, "variant": "clean", "candidate": candidate,
                         "joint": joint, "mse": 1.0, "mse_band": value ** 2})
    return rows


def test_d3_gates_on_the_te_free_content_and_reports_the_other_reading(ladder):
    rows = (_gating_rows("elbow_joint", 3.0, 1.0) + _gating_rows("shoulder_lift_joint", 2.9, 1.0)
            + _gating_rows("shoulder_pan_joint", 2.0, 1.0, content_l3p=4.0))
    gating = ladder.content_gating(pd.DataFrame(rows), "ur10").set_index("joint")
    assert gating.loc["elbow_joint", "gated"] and not gating.loc["shoulder_lift_joint", "gated"]
    assert not gating.loc["shoulder_pan_joint", "gated"] and gating.loc["shoulder_pan_joint", "gated_on_L3p_bags"]


def test_step_gate_needs_ltv_on_the_required_joint(ladder):
    def verdict(joint, oracle, passes):
        return {"regime": "full", "rung": "L3p", "joint": joint, "oracle": oracle, "band_vs_rigid": 0.5,
                "band_vs_rigid_low": 0.4, "band_vs_rigid_high": 0.6, "overall_vs_nominal": 0.9, "passes": passes}

    rows = [verdict("shoulder_lift_joint", "oracle_ltv", False), verdict("shoulder_lift_joint", "oracle_effective", True),
            verdict("shoulder_lift_joint", "oracle_ltv_tangent", True),
            verdict("elbow_joint", "oracle_ltv", False), verdict("elbow_joint", "oracle_effective", True),
            verdict("elbow_joint", "oracle_ltv_tangent", False)]
    gating = pd.DataFrame([{"regime": "full", "joint": j, "gated": True, "gated_on_L3p_bags": True}
                           for j in ("shoulder_lift_joint", "elbow_joint")])
    step = ladder.step1_gate(pd.DataFrame(rows), gating, "ur10").set_index("joint")
    assert not step.loc["shoulder_lift_joint", "passes"] and step.loc["shoulder_lift_joint", "ltv_required"]
    assert step.loc["elbow_joint", "passes"]
    # The tangent form is read on its own column: it never turns the spec's verdict.
    assert step.loc["shoulder_lift_joint", "passes_tangent"] and step.loc["elbow_joint", "passes_tangent"]


def test_lag_is_measured_and_the_target_aligned(ladder):
    x = np.sin(np.arange(200) * 0.3)
    assert ladder.delayed(x, 0) is x
    assert np.array_equal(ladder.delayed(x, 2)[2:], x[:-2]) and ladder.delayed(x, 2)[0] == x[0]
    group = pd.DataFrame({"q_motor_clean0": x, "q0": ladder.delayed(x, 1)})
    assert ladder.recorded_lag(group) == 1
    assert ladder.recorded_lag(pd.DataFrame({"q_motor_clean0": x, "q0": x})) == 0


def _scores(platform_joint, *, oracle_band, oracle_all, rigid_band=1.0, nominal_all=1.0, rung="L3"):
    rows = []
    for bag in ("a", "b"):
        for candidate, band, overall in (("rigid", rigid_band, 9.0), ("nominal", 9.0, nominal_all),
                                         ("oracle_effective", oracle_band, oracle_all),
                                         ("oracle_sampled", oracle_band, oracle_all)):
            rows.append({"bag": bag, "regime": "full", "rung": rung, "variant": "consumer", "candidate": candidate,
                         "joint": platform_joint, "mse": overall ** 2, "mse_band": band ** 2, "f_n": 10.0})
    return rows


@pytest.mark.parametrize("band, overall, passes", [(0.89, 0.99, True), (0.91, 0.99, False), (0.5, 1.01, False)])
def test_gate_is_the_spec_s(ladder, band, overall, passes):
    table = pd.DataFrame(_scores("shoulder_lift_joint", oracle_band=band, oracle_all=overall))
    verdict = ladder.gate(table, "ur10")
    assert set(verdict["passes"]) == {passes}
    assert verdict["band_vs_rigid"].iloc[0] == pytest.approx(band)
    assert verdict["overall_vs_nominal"].iloc[0] == pytest.approx(overall)


def test_first_failure_names_the_lowest_failing_rung(ladder):
    rows = (_scores("elbow_joint", oracle_band=0.5, oracle_all=0.5, rung="L1")
            + _scores("elbow_joint", oracle_band=2.0, oracle_all=2.0, rung="L3")
            + _scores("elbow_joint", oracle_band=3.0, oracle_all=3.0, rung="L4"))
    first = ladder.first_failures(ladder.gate(pd.DataFrame(rows), "ur10"))
    assert set(first["first_fail"]) == {"L3"}
    assert first["band_vs_rigid"].iloc[0] == pytest.approx(2.0)


def test_tracker_recovers_a_known_line_and_rejects_other_orders(te_band):
    step, order, amplitude = 1e-3, 320.0, 3.0
    t = np.arange(0.0, 8.0, step)
    speed = 0.5                                 # f_TE = 25.5 Hz
    q = speed * t
    rng = np.random.default_rng(0)
    x = amplitude * np.sin(order * q + 0.7) + 5.0 * np.sin(2 * np.pi * 0.5 * t) + 0.05 * rng.normal(size=len(t))
    window = int(te_band.WINDOW_S / step)
    inner = slice(window, -window)
    assert np.median(te_band.tracked_amplitude(x, order * q, window)[inner]) == pytest.approx(amplitude, rel=0.03)
    assert np.median(te_band.tracked_amplitude(x, 0.7 * order * q, window)[inner]) < 0.1 * amplitude


def test_ratio_uses_the_te_above_the_floor_and_marks_the_floor_bins(te_band):
    def line(source, b, te, floor):
        return {"source": source, "joint": "A2", "signal": "ft", "bin": b, "rows": 2000, "seconds": 2.0,
                "power": 2000 * (te ** 2 + floor ** 2), "floor_power": 2000 * floor ** 2}

    table = pd.DataFrame([line("real:x", 3, 0.5, 0.1), line("sim:off", 3, 5.0, 0.1),
                          line("real:x", 4, 0.0, 0.2), line("sim:off", 4, 4.0, 0.1)])
    ratio = te_band.ratios(te_band.te_amplitude(table)).iloc[0]
    assert ratio["bins"] == 2 and ratio["bins_at_floor"] == 1
    # Bin 3: 0.5 / 5.  Bin 4: nothing above the floor, so 0.2 is an upper bound on the real TE: 0.2 / 4.
    assert ratio["ratio"] == pytest.approx(np.median([0.1, 0.05]), rel=1e-6)


def test_deployed_seed_is_picked_on_validation_not_on_test(summary, tmp_path):
    def done(job, seed, stage, val, test):
        path = tmp_path / "jobs" / job / "ur10" / f"s{seed}_{stage}.done.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"best_val_rmse": str(val), "rmse": str(test),
                                    "checkpoint_path": f"{job}_s{seed}_{stage}"}))

    done("J1", 0, "T", 25.3, 1.0)        # the failed teacher, with the luckiest test RMSE
    done("J1", 2, "T", 5.0, 5.9)
    done("J3", 0, "S", 0.1, 0.1)         # the encoder stage's NLL: not the job's final stage
    done("J3", 0, "J", 6.0, 6.0)
    done("J3", 1, "J", 5.0, 6.5)
    assert {e["job"]: e["seed"] for e in summary.deployed_checkpoints(tmp_path).values()} == {"J1": "s2", "J3": "s1"}


def test_summary_table_reports_the_deployed_seed(summary, tmp_path):
    (tmp_path / "jobs" / "J1" / "ur10").mkdir(parents=True)
    for seed, val in ((0, 25.3), (2, 5.0)):
        (tmp_path / "jobs" / "J1" / "ur10" / f"s{seed}_T.done.json").write_text(
            json.dumps({"best_val_rmse": str(val), "checkpoint_path": f"c{seed}"}))
    (tmp_path / "eval" / "J1").mkdir(parents=True)
    pd.DataFrame({"checkpoint": ["c0", "c2"], "rmse": [28.0, 5.9]}).to_csv(tmp_path / "eval" / "J1" / "ur10-production.csv",
                                                                           index=False)
    table = summary.simulated_table(tmp_path)
    assert table["deployed_seed"].iloc[0] == "s2"
    assert table["rmse_deployed"].iloc[0] == 5.9
    assert table["rmse_mean"].iloc[0] == pytest.approx(16.95)
    assert "| job |" in summary.markdown(table) or "job" in summary.markdown(table)


def test_low_passed_selector_is_critically_damped_causal_and_starts_at_rest(ladder):
    """07 S1 (iii): two identical one-pole stages, unit DC gain, started on the first sample, no overshoot."""
    step, cutoff = 0.002, 2.0
    x = np.concatenate([np.full(50, 3.0), np.full(4000, 1.0)])[:, None] * np.ones((1, 2))
    y = ladder.low_passed(x, cutoff, step)
    assert np.allclose(y[:50], 3.0)                                   # at rest on x[0]: no start transient
    assert np.all(np.diff(y[50:, 0]) <= 1e-12)                        # monotone: critically damped, no ringing
    assert abs(y[-1, 0] - 1.0) < 1e-9                                 # unit DC gain
    impulse = ladder.low_passed(np.r_[np.zeros(10), 1.0, np.zeros(2000)][:, None], cutoff, step)[:, 0]
    assert np.all(impulse[:10] == 0.0)                                # causal
    a = np.exp(-2 * np.pi * cutoff * step)
    k = np.arange(2001)
    assert np.allclose(impulse[10:], (1 - a) ** 2 * (k + 1) * a ** k, atol=1e-15)   # (1 - a)^2 (k + 1) a^k


def test_fixed_point_warm_start_reaches_the_same_selection(ladder):
    """07 S1.2: started from the low-passed selector or from |tau_r|, the fixed point is the same filter output."""
    from elastic_sim.two_mass import catalogue_deflection

    rng = np.random.default_rng(0)
    t = np.arange(1500) * 0.002
    tau = np.column_stack([40.0 + 30.0 * np.sin(2 * np.pi * 0.7 * t) + 8.0 * np.sin(2 * np.pi * 9.0 * t),
                           12.0 * np.sin(2 * np.pi * 1.1 * t + 0.3)]) + 0.1 * rng.normal(size=(1500, 2))
    inertia = np.broadcast_to(np.array([[0.8, 0.05], [0.05, 0.3]]), (1500, 2, 2))
    stiffness, damping = np.array([2.0e4, 8.0e3]), np.array([30.0, 8.0])
    curve = (np.array([[0.69, 1.0, 1.44]] * 2), np.array([[29.0, 108.0], [14.0, 48.0]]), 0.1)
    initial = catalogue_deflection(tau[0], stiffness, curve[0], curve[1])
    cold, _ = ladder.fixed_point_pass(tau, inertia, stiffness, damping, curve, 0.002, initial)
    warm, passes = ladder.fixed_point_pass(tau, inertia, stiffness, damping, curve, 0.002, initial,
                                           start=ladder.selection(tau, "lowpass", 2.0, 0.002))
    assert 1 < passes <= ladder.FIXED_POINT_PASSES
    assert np.abs(cold["tau_s"] - warm["tau_s"]).max() < 1e-4 * np.abs(cold["tau_s"]).max()
