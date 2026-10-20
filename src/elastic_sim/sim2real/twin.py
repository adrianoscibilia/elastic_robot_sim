"""Twin experiment, R0 part: gate G-2a (S2R_01 T-4.6, S2R_04 P3-6).

Synthetic "real" bags from R0 itself, on a real run's recorded states, at a
known theta*:

1. The true state is the recorded one: ``q`` at 1 kHz, ``dq``/``ddq`` its
   SG derivatives (the replay settings), holds at rest.
2. ``ft*`` solves the R0 observation self-consistently (the deflection uses
   the torque it produces): ``ft = g tau_s(q - (ft - o)/(g K)) + o``, by
   fixed-point iteration (contraction ~ |d tau / d q| / K ~ 1e-2).
3. FR_05 §6 noise, measured on the run's own holds: ``sigma(ft)`` on ``ft``
   and ``sigma(q)`` on ``q`` (the replay re-derives dq/ddq from the noisy q).

The calibration then runs on the synthetic run exactly as on a real one
(fit, the A-12' label), and :func:`recovery` scores it.

**G-2a' (S2R_08 §2.2)**, five seeds (20261019, 7, 11, 3, 5), per seed:

(i) identifiable combinations within 2 %; (ii) the run's offsets **with
d_post >= 0.9** (full free set) within 0.05 N m; (iii) held-out ft RMSE
<= 1.25 x the residual at theta* (noise plus the differentiation floor,
F-4) on every joint, the S2R_06 1.2 still printed; (iv)-b on the full-set
posterior over all free parameters, >= 90 % of the errors within 2
posterior sd and none beyond 3 sd (the calibrated subset's fraction is
reported). S2R_06's (iv)-a (no d_post < 0.9 after A-12) is dropped: under
A-12' it holds by the label's definition. The diagonal-d and d_post flag
checks are reported, not gated.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .config import R0Settings, Sim2RealConfig
from .parameters import Registry
from .plant import IntermediateBodies, PlantBase, PlantParameters
from .real_runs import RealRun
from .replay_r0 import R0Engine, prepare_bags

FIXED_POINT_ITERATIONS = 6
#: G-2a thresholds (S2R_04 P3-6; S2R_01 T-4.6).
COMBINATION_TOLERANCE = 0.02
OFFSET_TOLERANCE_NM = 0.05
#: S2R_08 G-2a' (iii); back to NOISE_FACTOR_S2R06 once the F-3 hold poses are in a dataset.
NOISE_FACTOR = 1.25
NOISE_FACTOR_S2R06 = 1.2
FLAG_MIN_OFFSET_SIGMA = 0.2
#: G-2a' (iv)-b: >= WITHIN_FRACTION of the errors within POSTERIOR_SD_FACTOR
#: posterior sd, none beyond POSTERIOR_SD_MAX.
POSTERIOR_SD_FACTOR = 2.0
WITHIN_FRACTION = 0.9
POSTERIOR_SD_MAX = 3.0
#: S2R_08 §2.2: the S2R_06 three plus 3 and 5 (added before the 1.25 is used).
GATE_SEEDS = (20261019, 7, 11, 3, 5)
TEST_SEEDS = GATE_SEEDS[:3]


def perturbed_theta(registry: Registry, base: PlantBase, rng: np.random.Generator, *, run: str) -> dict[str, float]:
    """theta* of S2R_01 T-4.6: masses x U(0.85, 1.15), COM + U(+-2 cm), link
    friction x log-U(0.5, 2), the run's offsets U(+-0.5 N m); everything else
    at its prior mean (gain 1, inertia scale 1)."""
    theta = registry.values()
    for link, nominal in base.inertials.items():
        theta[f"inertial.{link}.mass"] = nominal.mass * rng.uniform(0.85, 1.15)
        for i, axis in enumerate("xyz"):
            theta[f"inertial.{link}.com_{axis}"] = nominal.com[i] + rng.uniform(-0.02, 0.02)
    for j, joint in enumerate(base.joint_names):
        for field, values in (("viscous", base.link_friction.viscous), ("coulomb", base.link_friction.coulomb)):
            theta[f"friction.link.{joint}.{field}"] = float(values[j]) * float(np.exp(rng.uniform(np.log(0.5),
                                                                                                    np.log(2.0))))
        theta[f"sensor.ft.{joint}.offset@{run}"] = rng.uniform(-0.5, 0.5)
    for name, value in theta.items():
        spec = registry.specs[name]
        if not spec.lower <= value <= spec.upper:
            raise ValueError(f"theta* {name} = {value} outside its bounds")
    return theta


def make_twin(run: RealRun, base: PlantBase, theta_star: Mapping[str, float], settings: R0Settings,
              rng: np.random.Generator, *, intermediate: IntermediateBodies | None) -> tuple[RealRun, dict[str, Any]]:
    """The synthetic copy of ``run`` (same bags, sidecar and states; ``q``
    and ``ft`` replaced) and a record of what was injected."""
    holds = run.sidecar["summary"]["holds"]
    sigma_ft = np.asarray(holds["sigma_ft"], dtype=float)
    sigma_q = np.asarray(holds["sigma_q"], dtype=float)
    full = dataclasses.replace(settings, decimate=1, hold_decimate=1, rows="all")
    prepared = prepare_bags(run.bags, full, time_step=1.0 / run.rate_hz)
    engine = R0Engine(base, prepared, deflection=True, intermediate=intermediate)
    plant = PlantParameters.from_theta(base, theta_star, runs=(run.run_id,))
    offsets = np.vstack([plant.sensor.offset(r) for r in engine.run_ids])[engine.row_run]
    ft = engine.simulate(plant, deflection_ft=None, deflection=False)[2]
    changes = []
    for _ in range(FIXED_POINT_ITERATIONS):
        new = engine.simulate(plant, deflection_ft=ft)[2]
        changes.append(float(np.max(np.abs(new - ft))))
        ft = new
    bags = []
    for bag, rows in zip(run.bags, engine.slices):
        clean = ft[rows]
        bags.append(dataclasses.replace(bag, q=bag.q + sigma_q * rng.standard_normal(bag.q.shape),
                                        ft=clean + sigma_ft * rng.standard_normal(clean.shape)))
    twin = dataclasses.replace(run, bags=tuple(bags))
    return twin, {"sigma_ft": sigma_ft.tolist(), "sigma_q": sigma_q.tolist(), "fixed_point_max_change": changes,
                  "offsets": offsets[0].tolist(), "rows": int(engine.n_rows)}


def recovery(registry: Registry, theta_hat: Mapping[str, float], theta_star: Mapping[str, float],
             identifiability: Mapping[str, Any], *, run: str, names: Sequence[str] | None = None,
             calibrated: Mapping[str, bool] | None = None) -> dict[str, Any]:
    """Per parameter of the stage set (``names``, default the free ones):
    error in prior sigmas, prior offset of theta*, d, d_post, posterior sd,
    calibrated (A-12); offsets in N m."""
    rows = []
    by_name = {r["name"]: r for r in identifiability["parameters"]}
    for name in (registry.free_names if names is None else names):
        spec = registry.specs[name]
        error = spec.prior_distance(theta_hat[name]) - spec.prior_distance(theta_star[name])
        info = by_name.get(name, {})
        rows.append({"name": name, "true": float(theta_star[name]), "recovered": float(theta_hat[name]),
                     "error_sigma": float(error), "true_offset_sigma": float(spec.prior_distance(theta_star[name])),
                     "data_share": info.get("data_share"), "data_share_posterior": info.get("data_share_posterior"),
                     "posterior_sd_sigma": info.get("posterior_sd_sigma"),
                     "calibrated": True if calibrated is None else bool(calibrated.get(name, True))})
    offsets = {r["name"]: r["recovered"] - r["true"] for r in rows
               if ".offset@" in r["name"] and r["name"].endswith("@" + run)}
    return {"parameters": rows, "offset_error_nm": offsets,
            "offset_error_max_nm": max((abs(v) for v in offsets.values()), default=0.0)}


def flag_check(rows: Sequence[Mapping[str, Any]], *, share_key: str,
               min_offset_sigma: float = FLAG_MIN_OFFSET_SIGMA) -> dict[str, Any]:
    """Does the data share flag what the twin could not recover?

    For every free parameter that theta* moved by >= ``min_offset_sigma``
    prior sigmas, the recovered fraction is ``1 - |error| / |move|`` (1: the
    data found theta*; 0: it stayed at the prior). A parameter is **falsely
    data-determined** when its share says data (>= 0.5) but less than half
    of its move was recovered: the case S2R_04 R-2 guards against (a
    prior-set number presented as calibrated). The reverse (share < 0.5 but
    recovered) is reported, not failed."""
    considered, false_determined, false_prior = 0, [], []
    for r in rows:
        share = r.get(share_key)
        if share is None or abs(r["true_offset_sigma"]) < min_offset_sigma:
            continue
        considered += 1
        fraction = 1.0 - abs(r["error_sigma"]) / abs(r["true_offset_sigma"])
        entry = {"name": r["name"], "share": share, "recovered_fraction": fraction,
                 "error_sigma": r["error_sigma"], "true_offset_sigma": r["true_offset_sigma"]}
        if share >= 0.5 and fraction < 0.5:
            false_determined.append(entry)
        elif share < 0.5 and fraction >= 0.5:
            false_prior.append(entry)
    return {"share": share_key, "min_offset_sigma": min_offset_sigma, "considered": considered,
            "false_data_determined": false_determined, "false_prior_determined": false_prior,
            "pass": not false_determined}


def run_twin_gate(config: Sim2RealConfig, *, run: str = "B", seed: int, budget: int | None = None,
                  run_id: str | None = None, resume: bool = False, real: RealRun | None = None,
                  base: PlantBase | None = None, figures: bool = True) -> tuple[Path, dict[str, Any]]:
    """Make the twin of ``run``, calibrate it with ``config`` (a normal run
    folder: fit, the label) and score G-2a'; writes ``twin_gate.{json,md}`` there."""
    from .calibration import calibrate
    from .problem import build_engines
    from .real_runs import load_real_run
    from .report import write_json

    base = PlantBase.from_config(config) if base is None else base
    registry = Registry.from_config(config, base)
    if real is None:
        spec = config.runs[run]
        real = load_real_run(spec.path, expected_sha256=spec.sha256, segments=config.segments, run_id=run)
    rng = np.random.default_rng(seed)
    theta_star = perturbed_theta(registry, base, rng, run=run)
    intermediate = IntermediateBodies.from_elastic_build(base)
    twin, injected = make_twin(real, base, theta_star, config.replay.r0, rng, intermediate=intermediate)
    folder = calibrate(config, runs={run: twin}, base=base, budget=budget, run_id=run_id, resume=resume,
                       figures=figures, command=f"twin gate, run {run}, seed {seed}")
    metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    ident = json.loads((folder / "identifiability.json").read_text(encoding="utf-8"))
    d_post_min = float((ident.get("a12") or {}).get("d_post_min", 0.9))
    theta_hat = registry.values()
    theta_hat.update(metrics["best_theta"])
    stage_names = [n for n in metrics["best_theta"] if n in registry.free_names]
    calibrated = {n: n not in metrics.get("not_calibrated", {}) for n in stage_names}
    rows = recovery(registry, theta_hat, theta_star, ident, run=run, names=stage_names, calibrated=calibrated)

    engine = build_engines(config, base, {run: twin}, splits=("val",))["val"]
    model = {}
    for label, theta in (("hat", theta_hat), ("star", theta_star)):
        plant = PlantParameters.from_theta(base, theta, runs=(run,))
        _, tau, _ = engine.simulate(plant)
        model[label] = plant.sensor.gain * tau
    difference = np.sqrt(np.mean((model["hat"] - model["star"]) ** 2, axis=0))
    reference = np.sqrt(np.mean(model["star"] ** 2, axis=0))
    combination = (difference / reference).tolist()
    sigma = np.asarray(injected["sigma_ft"])
    held_out = np.asarray(metrics["best"]["val"]["torque"]["per_joint_rmse"])
    # F-4: the floor is the val residual at theta* itself (the injected noise
    # plus R0's own floor: dq/ddq differentiated from the noisy q).
    star = engine.replay(PlantParameters.from_theta(base, theta_star, runs=(run,)))
    star_dynamic = [r for r in star if r.kind != "hold"]
    floor = np.sqrt(np.mean(np.vstack([r.residual for r in star_dynamic]) ** 2, axis=0))
    stiffness = {r["name"]: r["data_share"] for r in ident["parameters"] if r["name"].startswith("transmission.")}
    free = [r for r in rows["parameters"] if r["calibrated"]]
    offsets = {r["name"]: r["recovered"] - r["true"] for r in rows["parameters"] if r["name"].endswith("@" + run)
               and ".offset@" in r["name"] and (r["data_share_posterior"] or 0.0) >= d_post_min}
    mode = (ident.get("a12") or {}).get("mode", "freeze_refit")
    # (iv)-b on the posterior of every parameter the fit had free (under
    # A-12' the full set; in freeze_refit mode the frozen ones have none).
    with_sd = [r for r in rows["parameters"] if r["posterior_sd_sigma"]]

    def ratio(r):
        return abs(r["error_sigma"]) / r["posterior_sd_sigma"]

    def fraction(subset, k):
        return float(np.mean([ratio(r) <= k for r in subset])) if subset else None

    outside = [{"name": r["name"], "error_sigma": r["error_sigma"], "posterior_sd_sigma": r["posterior_sd_sigma"],
                "ratio": ratio(r), "calibrated": r["calibrated"]} for r in with_sd if ratio(r) > POSTERIOR_SD_FACTOR]
    beyond = [e for e in outside if e["ratio"] > POSTERIOR_SD_MAX]
    within = fraction(with_sd, POSTERIOR_SD_FACTOR)
    checks = {
        "combinations": {"per_joint_relative": combination, "tolerance": COMBINATION_TOLERANCE,
                         "pass": bool(max(combination) <= COMBINATION_TOLERANCE)},
        "offsets": {"error_nm": offsets, "max_nm": max((abs(v) for v in offsets.values()), default=0.0),
                    "tolerance_nm": OFFSET_TOLERANCE_NM, "d_post_min": d_post_min,
                    "not_gated": sorted(n for n in rows["offset_error_nm"] if n not in offsets),
                    "pass": bool(all(abs(v) <= OFFSET_TOLERANCE_NM for v in offsets.values()))},
        "held_out_noise": {"per_joint_rmse": held_out.tolist(), "sigma_ft": sigma.tolist(),
                           "floor_at_theta_star": floor.tolist(), "floor_over_sigma": (floor / sigma).tolist(),
                           "ratio": (held_out / floor).tolist(), "factor": NOISE_FACTOR,
                           "pass_s2r06": bool(np.all(held_out <= NOISE_FACTOR_S2R06 * floor)),
                           "factor_s2r06": NOISE_FACTOR_S2R06,
                           "pass": bool(np.all(held_out <= NOISE_FACTOR * floor))},
        "posterior": {"mode": mode, "n_parameters": len(with_sd), "n_calibrated": len(free),
                      "n_not_calibrated": len(rows["parameters"]) - len(free), "outside_2_sd": outside,
                      "beyond_3_sd": beyond, "within_2_sd": within,
                      "within_2_sd_calibrated": fraction([r for r in with_sd if r["calibrated"]], POSTERIOR_SD_FACTOR),
                      "within_fraction_min": WITHIN_FRACTION, "factor": POSTERIOR_SD_FACTOR,
                      "max_factor": POSTERIOR_SD_MAX,
                      "pass": bool(within is not None and within >= WITHIN_FRACTION and not beyond)},
        "flags_d": flag_check(rows["parameters"], share_key="data_share"),
        "flags_d_post": flag_check(free, share_key="data_share_posterior"),
    }
    checks["posterior_calibration"] = {"within_2_sd": within, "within_3_sd": fraction(with_sd, 3.0)}
    excluded = [n for n in registry.specs if ".offset@" in n and not n.endswith("@" + run)
                and registry.specs[n].frozen and registry.specs[n].sigma]
    checks["unobserved_excluded"] = {"names": excluded, "note": "A-13: offsets of runs without training bags are "
                                     "not free (reported, not gated)"}
    frozen_rows = [r for r in rows["parameters"] if not r["calibrated"]]
    checks["not_calibrated"] = {"names": [r["name"] for r in frozen_rows],
                                "max_abs_error_sigma": max((abs(r["error_sigma"]) for r in frozen_rows), default=0.0)}
    gate_pass = all(checks[k]["pass"] for k in ("combinations", "offsets", "held_out_noise", "posterior"))
    gate = {"gate": "G-2a'", "definition": "S2R_08 §2.2", "pass": bool(gate_pass), "seed": seed, "run": run,
            "injected": injected, "theta_star": {n: theta_star[n] for n in stage_names}, "checks": checks,
            "stiffness_data_share": stiffness, "recovery": rows["parameters"]}
    write_json(folder / "twin_gate.json", gate)
    (folder / "twin_gate.md").write_text(_gate_markdown(gate, base.joint_names), encoding="utf-8")
    return folder, gate


def gate_summary(gates: Sequence[Mapping[str, Any]]) -> tuple[str, dict[str, Any]]:
    """The verdict over seeds (S2R_08 §2.2): pass = every seed passes."""
    ok = {True: "pass", False: "**FAIL**"}
    lines = [f"# Gate G-2a' (S2R_08 §2.2): {'PASS' if all(g['pass'] for g in gates) else 'FAIL'} over "
             f"{len(gates)} seeds", "",
             "| seed | (i) combinations max | (ii) offsets (d_post >= 0.9) max | (iii) held-out / floor max "
             f"(<= {NOISE_FACTOR}; S2R_06 {NOISE_FACTOR_S2R06}) | (iv)-b within 2 sd, all (calibrated) | "
             "beyond 3 sd | calibrated / not | verdict |",
             "|---|---|---|---|---|---|---|---|"]
    for g in gates:
        c = g["checks"]
        post = c["posterior"]
        held = c["held_out_noise"]
        lines.append(f"| {g['seed']} | {100 * max(c['combinations']['per_joint_relative']):.3f} % "
                     f"{ok[c['combinations']['pass']]} | {c['offsets']['max_nm']:.4f} N m "
                     f"({len(c['offsets']['error_nm'])} gated) {ok[c['offsets']['pass']]} | "
                     f"{max(held['ratio']):.3f} {ok[held['pass']]} (1.2: {ok[held['pass_s2r06']]}) | "
                     f"{_fmt(post['within_2_sd'])} ({_fmt(post['within_2_sd_calibrated'])}) "
                     f"{ok[post['within_2_sd'] is not None and post['within_2_sd'] >= post['within_fraction_min']]} | "
                     f"{len(post['beyond_3_sd'])} {ok[not post['beyond_3_sd']]} | {post['n_calibrated']} / "
                     f"{post['n_not_calibrated']} | {'PASS' if g['pass'] else 'FAIL'} |")
    return "\n".join(lines) + "\n", {"gate": "G-2a'", "definition": "S2R_08 §2.2",
                                     "pass": bool(all(g["pass"] for g in gates)), "seeds": [g["seed"] for g in gates],
                                     "per_seed": {str(g["seed"]): g["pass"] for g in gates}}


def _gate_markdown(gate: Mapping[str, Any], joints: Sequence[str]) -> str:
    c = gate["checks"]
    post = c["posterior"]
    held = c["held_out_noise"]
    ok = {True: "pass", False: "**FAIL**"}
    lines = [f"# Gate G-2a' (twin, R0, S2R_08 §2.2): {'PASS' if gate['pass'] else 'FAIL'}", "",
             f"Run {gate['run']}'s states, seed {gate['seed']}; label mode `{post['mode']}`; injected sigma(ft) "
             + ", ".join(f"{v:.4f}" for v in gate["injected"]["sigma_ft"]) + " N m; fixed-point max change per "
             "iteration " + ", ".join(f"{v:.1e}" for v in gate["injected"]["fixed_point_max_change"]) + ".", "",
             "| check | value | tolerance | verdict |", "|---|---|---|---|",
             f"| (i) identifiable combinations (noise-free g tau on val, per joint) | max "
             f"{max(c['combinations']['per_joint_relative']) * 100:.3f} % | {c['combinations']['tolerance'] * 100:g} % | "
             f"{ok[c['combinations']['pass']]} |",
             f"| (ii) offsets@{gate['run']} with d_post >= {c['offsets']['d_post_min']:g} "
             f"({len(c['offsets']['error_nm'])} of them) | max {c['offsets']['max_nm']:.4f} N m | "
             f"{c['offsets']['tolerance_nm']} N m | {ok[c['offsets']['pass']]} |",
             f"| (iii) held-out ft RMSE / floor at theta* (noise + differentiation), every joint | max "
             f"{max(held['ratio']):.3f} | {held['factor']} | {ok[held['pass']]} |",
             f"| (iii) at S2R_06's factor (printed, not gated) | max {max(held['ratio']):.3f} | {held['factor_s2r06']} | "
             f"{ok[held['pass_s2r06']]} |",
             f"| (iv)-b errors within {post['factor']:g} posterior sd (full set, all {post['n_parameters']} free) | "
             f"{_fmt(post['within_2_sd'])} | >= {post['within_fraction_min']:g} | "
             f"{ok[post['within_2_sd'] is not None and post['within_2_sd'] >= post['within_fraction_min']]} |",
             f"| (iv)-b errors beyond {post['max_factor']:g} posterior sd | {len(post['beyond_3_sd'])} | 0 | "
             f"{ok[not post['beyond_3_sd']]} |",
             f"| calibrated subset ({post['n_calibrated']}): within {post['factor']:g} sd (reported) | "
             f"{_fmt(post['within_2_sd_calibrated'])} | – | – |",
             f"| posterior calibration (reported): within 2 / 3 posterior sd | "
             f"{_fmt(c['posterior_calibration']['within_2_sd'])} / {_fmt(c['posterior_calibration']['within_3_sd'])} "
             f"| ~0.95 / ~1 | – |",
             f"| d_post flags among the calibrated (reported) | {len(c['flags_d_post']['false_data_determined'])} of "
             f"{c['flags_d_post']['considered']} moved | 0 | – |",
             f"| diagonal d flags over the stage set (reported, F-2: diagnostic) | "
             f"{len(c['flags_d']['false_data_determined'])} of {c['flags_d']['considered']} moved | – | – |",
             f"| not calibrated | {len(c['not_calibrated']['names'])}, max |error| "
             f"{c['not_calibrated']['max_abs_error_sigma']:.2f} sigma | – | – |",
             f"| other runs' offsets excluded (A-13) | {len(c['unobserved_excluded']['names'])} | – | – |", "",
             "Per joint: | " + " | ".join(joints) + " |", "|---|" + "---|" * len(joints),
             "| combination error, % | " + " | ".join(f"{100 * v:.3f}" for v in c["combinations"]["per_joint_relative"])
             + " |",
             "| held-out RMSE / floor | " + " | ".join(f"{v:.3f}" for v in held["ratio"]) + " |",
             "| floor / sigma(ft) | " + " | ".join(f"{v:.2f}" for v in held["floor_over_sigma"]) + " |",
             ""]
    if post["outside_2_sd"]:
        lines += [f"Parameters outside {post['factor']:g} posterior sd:", "",
                  "| parameter | calibrated | error (sigma) | posterior sd (sigma) | ratio |", "|---|---|---|---|---|"]
        lines += [f"| `{e['name']}` | {'yes' if e['calibrated'] else 'no'} | {e['error_sigma']:+.3f} | "
                  f"{e['posterior_sd_sigma']:.3f} | {e['ratio']:.2f} |" for e in post["outside_2_sd"]]
        lines.append("")
    if c["offsets"]["not_gated"]:
        lines += ["Offsets not gated (d_post < 0.9): " + ", ".join(f"`{n}`" for n in c["offsets"]["not_gated"]), ""]
    lines += ["Stiffness data share (frozen, analysed): "
              + ", ".join(f"{k.split('.')[1]} {v:.2e}" for k, v in gate["stiffness_data_share"].items()), "",
              "| parameter | calibrated | true | recovered | error (sigma) | true offset (sigma) | d_post | post. sd | d |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in gate["recovery"]:
        lines.append(f"| `{r['name']}` | {'yes' if r['calibrated'] else 'no'} | {r['true']:.6g} | {r['recovered']:.6g} | "
                     f"{r['error_sigma']:+.3f} | {r['true_offset_sigma']:+.3f} | {_fmt(r['data_share_posterior'])} | "
                     f"{_fmt(r['posterior_sd_sigma'])} | {_fmt(r['data_share'])} |")
    return "\n".join(lines) + "\n"


def _fmt(value: Any, spec: str = ".3f") -> str:
    return "–" if value is None else format(value, spec)


__all__ = ["GATE_SEEDS", "TEST_SEEDS", "flag_check", "gate_summary", "make_twin", "perturbed_theta", "recovery", "run_twin_gate"]
