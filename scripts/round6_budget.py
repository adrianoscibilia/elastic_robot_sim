#!/usr/bin/env python3
"""The round-6 contribution budget, before any production build (`R6_00` Sec 8).

Two robots per platform, every Sec 5 effect toggled once, same robots,
trajectories and seeds in every build.  For each bag it splits the target
``tau_s`` (clean) into

* **rigid dynamics**: ``rnea(theta, thetadot, thetaddot)`` of the robot's own
  rigid model (payload included) on the *motor-side* state -- what a rigid
  model fed the motor-side inputs already explains;
* **elastic content**: what is left of ``tau_s - rigid`` with every effect off
  *and the instrument off*.  With the instrument on, the loop's own reading
  noise shakes the rotor (on the iiwa the velocity is a causal derivative of a
  noisy position) and that jitter, filtered by the spring, would be counted as
  elasticity; it is reported separately as **loop-noise content**;
* **each effect**: the change of ``tau_s - rigid`` when that one effect is
  switched off.  Differencing the non-rigid parts, not the targets, keeps the
  closed loop's own change of trajectory (which the rigid model explains) out
  of the effect;
* **all extras together**: ``(tau_s - rigid)`` all-on minus all-off;
* **noise**: ``ft - ft_clean`` of the all-on build.

Variances are pooled over every sample of the platform's bags and over its
*elastically observable* joints (`R6_02` P2-2: the joints whose link-side
mode band lies inside the resolvable band, :mod:`elastic_sim.link_modes`),
and reported as shares of the target's variance on those joints.  Gates
(`R6_02` P2-2, replacing `R6_00` Sec 8):

1. elastic content >= 5 x target noise and >= 5 x loop-noise content
   (observable above the instrument);
2. elastic content >= 10 % of the non-rigid part (a learning signal, not a
   rounding error);
3. every other share is reported, not gated against the elastic content.

Per-joint shares are reported for every joint, observable or not.  If a
platform fails, the script exits 1 and the owner decides.

    python scripts/round6_budget.py --config config/identification/ur10_table_round6.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(_REPO / "src"))
# One BLAS/OpenMP thread per worker: the build already runs one process per
# core, and threaded BLAS on top drove the load past 400 on 32 cores.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
sys.path.insert(0, os.fspath(_REPO / "scripts"))

import numpy as np
import pandas as pd

from diagnose_controller_modes import _load_asset, _shrink
from elastic_sim import identification as idn
from elastic_sim.dataset import PlantExtrasSampling, default_jobs, generate, load_config
from elastic_sim.link_modes import link_mode_table
from elastic_sim.payload import Payload, payload_asset


def toggles(config) -> dict[str, object]:
    """Each Sec 5 effect's off-switch, for the effects this platform has."""
    extras = config.plant_extras
    out: dict[str, object] = {}
    if extras.has_link_friction:
        out["link friction"] = replace(config, plant_extras=replace(extras, link_friction_tier="off"))
    if extras.has_nonlinear_spring:
        out["nonlinear spring"] = replace(config, plant_extras=replace(
            extras, stiffness_factors=(), stiffness_breakpoints=(), stiffness_breakpoints_torque=(),
            stiffness_breakpoints_effort_fraction=()))
    if extras.has_transmission_error:
        out["transmission error"] = replace(config, plant_extras=replace(extras, transmission_error_amplitude=()))
    if extras.ripple_amplitude:
        out["torque ripple"] = replace(config, plant_extras=replace(extras, ripple_amplitude=0.0))
    out["motor friction"] = replace(config, motor_friction=replace(
        config.motor_friction, viscous_fraction=0.0, coulomb_fraction=0.0))
    out["all off"] = replace(config, plant_extras=PlantExtrasSampling(), motor_friction=replace(
        config.motor_friction, viscous_fraction=0.0, coulomb_fraction=0.0))
    out["all off, instrument off"] = replace(out["all off"], noise=replace(config.noise, enabled=False))
    return out


def _block(frame: pd.DataFrame, prefix: str, n: int) -> np.ndarray:
    return frame[[f"{prefix}{i}" for i in range(n)]].to_numpy(dtype=float)


def non_rigid_parts(frame: pd.DataFrame, manifest: dict, asset, step: float) -> dict[str, dict[str, np.ndarray]]:
    """Per bag: ``tau_s``, the motor-side rigid prediction, noise and ``tau_cmd - tau_s``."""
    n = int(manifest["n_dof"])
    payloads = {r["bag"]: r.get("payload") for r in manifest["records"]}
    parts: dict[str, dict[str, np.ndarray]] = {}
    models: dict[tuple, tuple] = {}
    for bag, group in frame.groupby("bag", sort=False):
        payload_record = payloads.get(bag)
        payload = Payload() if not payload_record else Payload(
            mass=payload_record["mass"], offset=tuple(payload_record["offset"]), size=payload_record["size"])
        key = () if payload.is_empty else (payload.mass, payload.offset, payload.size)
        if key not in models:
            with payload_asset(asset, payload) as asset_p:
                models[key] = idn.build_model(asset_p)
        pin, model, data = models[key]
        theta = _block(group, "q_motor_clean", n)
        theta_dot = _block(group, "dq_motor_clean", n)
        theta_ddot = np.gradient(theta_dot, step, axis=0)
        rigid = np.asarray([np.asarray(pin.rnea(model, data, theta[i], theta_dot[i], theta_ddot[i]))
                            for i in range(len(theta))])
        target = _block(group, "ft_clean", n)
        parts[str(bag)] = {"target": target, "rigid": rigid, "non_rigid": target - rigid,
                           "noise": _block(group, "ft", n) - target,
                           "input_gap": _block(group, "tau", n) - target}
    return parts


def _build_variant(variant, asset, step: float, keep_inputs: bool, *, jobs: int = 1, verbose: bool = False,
                   cache: dict | None = None):
    frame, manifest, _ = generate(variant, asset, verbose=verbose, jobs=jobs,
                                  trajectory_cache={} if cache is None else cache)
    parts = non_rigid_parts(frame, manifest, asset, step)
    if not keep_inputs:
        return parts, None
    n = len(asset.joint_names)
    return parts, {"q": _block(frame, "q", n), "q_clean": _block(frame, "q_motor_clean", n),
                   "dq": _block(frame, "dq", n), "dq_clean": _block(frame, "dq_motor_clean", n)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--robots", type=int, default=2)
    parser.add_argument("--trajectories", type=int, default=1)
    parser.add_argument("--jobs", type=int, default=None)
    parser.add_argument("--variant-jobs", type=int, default=1,
                        help="build the toggled variants in this many parallel processes (bags serial in each)")
    parser.add_argument("--out", default=None, help="output prefix (default reports/round6/budget_<stem>)")
    parser.add_argument("--probe-fraction", type=float, default=None,
                        help="override excitation.probe_acceleration_fraction")
    parser.add_argument("--no-save", action="store_true", help="print only; write no report files")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.probe_fraction is not None:
        config = replace(config, excitation=replace(config.excitation,
                                                    probe_acceleration_fraction=float(args.probe_fraction)))
    asset = _load_asset(config.asset)
    config = _shrink(config, trajectories=args.trajectories, robots=args.robots, backends=tuple(config.backends))
    # Elastic robots only: the rigid tier has no elastic content to budget.
    config = replace(config, only_tiers=tuple(t.name for t in config.tiers if not t.is_rigid))
    jobs = default_jobs(config.backends) if args.jobs is None else args.jobs
    step = float(config.sample_time_step)
    cache: dict = {}
    verbose = not args.quiet

    variants = [("all on", config), *toggles(config).items()]
    builds: dict[str, dict] = {}
    if args.variant_jobs > 1:
        # One process per variant, each building its bags serially: the
        # variants are independent builds of the same robots and seeds.
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=args.variant_jobs) as pool:
            futures = {name: pool.submit(_build_variant, variant, asset, step, name == "all on")
                       for name, variant in variants}
            for name, future in futures.items():
                builds[name], extra = future.result()
                if name == "all on":
                    inputs = {"all on": extra}
                if verbose:
                    print(f"=== {name}: done")
    else:
        for name, variant in variants:
            if verbose:
                print(f"\n=== {name} ===")
            builds[name], extra = _build_variant(variant, asset, step, name == "all on", jobs=jobs,
                                                 verbose=verbose, cache=cache)
            if name == "all on":
                inputs = {"all on": extra}

    names = list(asset.joint_names)
    modes = link_mode_table(config, asset)
    observable = [j for j, mode in enumerate(modes) if mode.observable]
    mostly = [j for j, mode in enumerate(modes) if mode.mostly_observable]
    base = builds["all on"]
    bags = sorted(base)
    stack = lambda key, source=base: np.vstack([source[b][key] for b in bags])  # noqa: E731
    target = stack("target")
    components = {
        "rigid dynamics": stack("rigid"),
        "non-rigid part": stack("non_rigid"),
        "elastic content": stack("non_rigid", builds["all off, instrument off"]),
        "loop-noise content": stack("non_rigid", builds["all off"]) - stack("non_rigid", builds["all off, instrument off"]),
        "all extras together": stack("non_rigid") - stack("non_rigid", builds["all off"]),
        "noise": stack("noise"),
    }
    effects = [name for name in builds if name not in ("all on", "all off", "all off, instrument off")]
    for effect in effects:
        components[effect] = stack("non_rigid") - stack("non_rigid", builds[effect])
    if "motor friction" in builds:
        components["motor friction, in tau_cmd - tau_s"] = (stack("input_gap")
                                                            - stack("input_gap", builds["motor friction"]))

    def pooled_over(columns: list[int]) -> dict[str, float]:
        total = float(np.sum((target[:, columns] - target[:, columns].mean(axis=0)) ** 2))
        return {key: float(np.sum((value[:, columns] - value[:, columns].mean(axis=0)) ** 2)) / total
                for key, value in components.items()}

    pooled = pooled_over(observable) if observable else {key: float("nan") for key in components}
    pooled_all = pooled_over(list(range(len(names))))
    per_joint = pd.DataFrame({
        key: np.var(value, axis=0) / np.var(target, axis=0) for key, value in components.items()
    }, index=names)
    per_joint.insert(0, "observable", [mode.observable for mode in modes])
    per_joint.insert(1, "f_link_low_hz", [mode.f_low for mode in modes])
    per_joint.insert(2, "f_link_high_hz", [mode.f_high for mode in modes])
    per_joint.insert(3, "resolvable_share", [mode.resolvable_share for mode in modes])

    def gate_rows(shares: dict[str, float]) -> dict[str, dict[str, float | bool]]:
        elastic = shares["elastic content"]
        return {
            "elastic content >= 5 x target noise": {
                "elastic": elastic, "noise": shares["noise"], "ok": bool(elastic >= 5.0 * shares["noise"])},
            "elastic content >= 5 x loop-noise content": {
                "elastic": elastic, "loop_noise": shares["loop-noise content"],
                "ok": bool(elastic >= 5.0 * shares["loop-noise content"])},
            "elastic content >= 10 % of the non-rigid part": {
                "ratio": elastic / shares["non-rigid part"],
                "ok": bool(elastic >= 0.1 * shares["non-rigid part"])},
        }

    # R6_04 A-3: the gate set is the joints with >= 95 % of their prior
    # resolvable; the strict whole-band set is reported beside it.  Gates 1a
    # and 1b block (R6_04 Sec 4 step 6); gate 2 is reported.
    pooled_mostly = pooled_over(mostly) if mostly else {key: float("nan") for key in components}
    gates_mostly = gate_rows(pooled_mostly)
    blocking = ("elastic content >= 5 x target noise", "elastic content >= 5 x loop-noise content")
    ok_mostly = bool(mostly) and all(gates_mostly[g]["ok"] for g in blocking)
    gates = gate_rows(pooled)
    ok = bool(observable) and all(gates[g]["ok"] for g in blocking)

    # The loop's measured inputs, against the clean ones (what the controller and a model read).
    frame_inputs = inputs["all on"]
    input_noise = {channel: float(np.sqrt(np.mean((frame_inputs[channel] - frame_inputs[channel + "_clean"]) ** 2))
                                  / np.sqrt(np.mean(frame_inputs[channel + "_clean"] ** 2)))
                   for channel in ("q", "dq")}
    print(f"\nrecorded input noise / clean RMS (pooled): q {input_noise['q']:.3g}, dq {input_noise['dq']:.3g}")
    print(f"\nContribution budget, {config.asset}: {len(bags)} bags, variance shares of tau_s, pooled over the "
          f"observable joints {[names[j] for j in observable]} (all joints alongside)")
    for key in pooled:
        print(f"  {key:40s} {pooled[key]:10.4%}   all joints {pooled_all[key]:10.4%}")
    formatters = {column: (lambda v: f"{v:.3g}") if column.startswith("f_link") else (lambda v: f"{v:.3%}")
                  for column in per_joint.columns if column != "observable"}
    print("\nper joint (shares of each joint's own tau_s variance; f_link in Hz):\n"
          + per_joint.to_string(formatters=formatters))
    print("\ngates (R6_02 P2-2, observable joints):")
    for gate, verdict in gates.items():
        detail = ", ".join(f"{k} {v:.4g}" for k, v in verdict.items() if k != "ok")
        print(f"  [{'x' if verdict['ok'] else ' '}] {gate}: {detail}")
    probe = config.excitation
    print(f"\nprobe: {len(probe.probe_harmonics)} lines {probe.probe_harmonics[0] * probe.base_frequency:g}-"
          f"{probe.probe_harmonics[-1] * probe.base_frequency:g} Hz, acceleration fraction "
          f"{probe.probe_acceleration_fraction:g}")
    print(f"\nsame gates on the mostly-observable joints {[names[j] for j in mostly]} (>= 95 % of the prior "
          "resolvable):")
    for gate, verdict in gates_mostly.items():
        detail = ", ".join(f"{k} {v:.4g}" for k, v in verdict.items() if k != "ok")
        print(f"  [{'x' if verdict['ok'] else ' '}] {gate}: {detail}")
    print(f"\nbudget {'PASSES' if ok_mostly else 'FAILS'} on {config.asset} (gates 1a/1b on the >= 95 % set; "
          f"gate 2 reported); strict-band set: {'passes' if ok else 'fails'}")
    if args.no_save:
        return 0 if ok_mostly else 1

    out = Path(args.out) if args.out else _REPO / "reports" / "round6" / f"budget_{Path(config.output).stem}"
    out.parent.mkdir(parents=True, exist_ok=True)
    per_joint.to_csv(out.with_name(out.name + "_per_joint.csv"))
    out.with_name(out.name + ".json").write_text(json.dumps(
        {"config": args.config, "asset": config.asset, "bags": bags,
         "observable_joints": [names[j] for j in observable], "link_modes": [m.as_dict() for m in modes],
         "probe_acceleration_fraction": float(config.excitation.probe_acceleration_fraction),
         "probe_frequencies_hz": [h * config.excitation.base_frequency for h in config.excitation.probe_harmonics],
         "pooled_shares_observable": pooled, "pooled_shares_all_joints": pooled_all,
         "input_noise_over_rms": input_noise, "gates": gates, "ok": ok,
         "mostly_observable_joints": [names[j] for j in mostly], "pooled_shares_mostly_observable": pooled_mostly,
         "gates_mostly_observable": gates_mostly, "ok_mostly_observable": ok_mostly}, indent=2, default=float), encoding="utf-8")
    print(f"wrote {out.with_name(out.name + '.json')}")
    return 0 if ok_mostly else 1


if __name__ == "__main__":
    raise SystemExit(main())
