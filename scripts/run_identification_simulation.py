#!/usr/bin/env python3
"""Run and inspect ONE identification rollout, without writing a dataset.

This is the debugging counterpart to ``generate_identification_dataset.py``.
It shares the same YAML config, runs a single trajectory under a single tier
and backend, prints whether the numbers make sense, and saves nothing unless
``--output`` is given.

Examples
--------
Watch the arm execute an excitation trajectory::

    python scripts/run_identification_simulation.py --visualize

Check a trajectory without starting a simulator at all::

    python scripts/run_identification_simulation.py --trajectory-only

Inspect sampled elastic robot e03 in Newton and keep the samples::

    python scripts/run_identification_simulation.py \
        --tier e03 --backend newton --output /tmp/one_rollout.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(_REPO / "src"))

from elastic_sim import excitation as exc
from elastic_sim import identification as idn
from elastic_sim.assets import AssetRegistry, load_asset_spec
from elastic_sim.controllers import CONTROLLER_MODES, ControllerDraw
from elastic_sim.dataset import (
    DEFAULT_CONFIG, RIGID_TIER, elastic_time_step, load_config, payload_for, resolve_bag, rollout_frame,
    run_condition, sample_all_payloads,
)
from elastic_sim.payload import Payload, payload_asset
from elastic_sim.torque_runners import control_separation_ratio, link_inertia_envelope, link_inertia_max


def _load_asset(reference: str):
    candidate = Path(reference)
    if candidate.is_file():
        return load_asset_spec(candidate)
    return AssetRegistry.for_repository(_REPO).load(reference)


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else _REPO / candidate


def _check_manifest_agrees(parser, config, manifest_path: Path) -> None:
    """Fail loudly when this config could not have produced ``manifest_path``.

    Stratified/Sobol sampling is defined by the whole batch (``robots``), so
    ``--tier eNN`` against a dataset built with a different ``robots``,
    ``sampling`` or ``seed`` silently reproduces the wrong robot instead of
    erroring -- this is the mismatch class R3_03 introduces.
    """
    import json

    if not manifest_path.is_file():
        parser.error(f"--manifest {manifest_path} does not exist")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded = manifest.get("transmission_sampling", {})
    checks = (
        ("robots", recorded.get("robots"), config.transmission.robots),
        ("sampling", recorded.get("sampling"), config.transmission.sampling),
        ("seed", manifest.get("seed"), config.seed),
    )
    mismatches = [f"{name}: manifest={recorded_value!r} config={config_value!r}"
                  for name, recorded_value, config_value in checks if recorded_value != config_value]
    if mismatches:
        parser.error(
            f"--manifest {manifest_path} disagrees with the current config, so --tier would not "
            f"reproduce its robots: {'; '.join(mismatches)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="YAML defaults (see config/identification/)")
    parser.add_argument("--asset", default=None)
    parser.add_argument("--backend", default=None, choices=("mujoco", "newton"))
    parser.add_argument("--tier", default=None, help=f"Tier name: {RIGID_TIER} or a sampled robot e00, e01, ...")
    parser.add_argument("--manifest", default=None,
                        help="A written dataset's .manifest.json; with --tier, errors if this config's "
                             "robots/sampling/seed disagree with the ones that produced it")
    parser.add_argument("--seed", type=int, default=None, help="Trajectory seed")
    parser.add_argument("--trajectory", type=int, default=0,
                        help="Trajectory index within the tier (0 by default); reproduces the dataset's "
                             "regime for that index when excitation.regime is enabled")
    parser.add_argument("--base-frequency", type=float, default=None)
    parser.add_argument("--max-acceleration", type=float, default=None)
    parser.add_argument("--candidates", type=int, default=None)
    parser.add_argument("--control-frequency", type=float, default=None)
    parser.add_argument("--controller-mode", default=None, choices=list(CONTROLLER_MODES),
                        help="Override simulation.controller.mode for this one rollout (R5_00 Sec 6.1)")
    parser.add_argument("--visualize", action="store_true", help="Open the native viewer")
    parser.add_argument("--realtime-scale", type=float, default=None, help="1.0 is real time")
    parser.add_argument("--trajectory-only", action="store_true",
                        help="Design and check the trajectory; do not simulate")
    parser.add_argument("--save-trajectory", default=None, help="Optional trajectory JSON path")
    parser.add_argument("--output", default=None, help="Optional CSV; nothing is written without it")
    args = parser.parse_args()

    config = load_config(_resolve(args.config))
    asset_name = args.asset or config.asset
    asset = _load_asset(asset_name)
    asset.resolve_active_joints()
    asset.validate_resources()

    if args.manifest and args.tier:
        _check_manifest_agrees(parser, config, _resolve(args.manifest))

    from dataclasses import replace

    tier_name = args.tier or config.tiers[0].name
    matches = [tier for tier in config.tiers if tier.name == tier_name]
    if not matches:
        parser.error(f"unknown tier {tier_name!r}; config has {[t.name for t in config.tiers]}")
    tier = matches[0]

    # Same rule as generate(): a tier's payload is only known in advance (and
    # so only scoreable against payload-fitted collision geometry) when every
    # tier gets its own trajectory; a shared trajectory (trajectories_per_
    # robot: false) is payload-free by construction.  Previously this script
    # always used the bare asset and never passed a payload to run_condition
    # at all -- for the shipped config (trajectories_per_robot: true,
    # payload.enabled), that reproduced the wrong trajectory for any robot
    # whose payload changed a collision rejection, and always ran a
    # payload-free rollout regardless (R3_14 Sec 1.1).
    elastic_tiers = [t for t in config.tiers if not t.is_rigid]
    if args.controller_mode is not None:
        config = replace(config, controller=replace(config.controller, mode=args.controller_mode))
    payload_by_tier, payload_by_key = sample_all_payloads(config, elastic_tiers)
    payload = (
        payload_for(config, payload_by_tier, payload_by_key, tier, args.trajectory)
        if config.trajectories_per_robot else Payload()
    )

    print(f"asset      : {asset.name} ({len(asset.joint_names)} joints)")
    print(f"config     : {_resolve(args.config)}")
    if payload is not None and not payload.is_empty:
        print(f"payload    : {payload.as_dict()}")

    if args.seed is not None or args.base_frequency is not None or args.max_acceleration is not None:
        # Ad-hoc overrides: --seed replaces the derived trajectory seed
        # outright and --base-frequency/--max-acceleration replace the
        # regime's derived excitation, so this path cannot go through
        # resolve_bag (which always derives both from the config) and
        # reproduces nothing by design -- it is for poking at one-off
        # variations, not for --tier/--trajectory reproduction.
        from elastic_sim.dataset import regime_excitation, sample_control_gains, trajectory_seed
        from elastic_sim.kinematics import PortableKinematics

        seed = trajectory_seed(config, tier, args.trajectory) if args.seed is None else args.seed
        excitation = regime_excitation(config.excitation, config.regime, config.seed, seed)
        if args.base_frequency is not None or args.max_acceleration is not None:
            excitation = replace(
                excitation,
                base_frequency=args.base_frequency or excitation.base_frequency,
                max_acceleration=args.max_acceleration or excitation.max_acceleration,
            )
        candidates = args.candidates or config.candidates
        with payload_asset(asset, payload) as asset_p:
            kinematics = PortableKinematics(asset_p)
            trajectory = exc.optimize_excitation(
                asset, excitation, seed=seed, n_candidates=candidates, kinematics=kinematics
            )
        natural_frequency, damping_ratio = sample_control_gains(
            config.control_gains, config.control_frequency, config.control_damping_ratio, config.seed, seed,
        )
        from elastic_sim.controllers import sample_velocity_loop
        from elastic_sim.dataset import plant_extras_for_bag

        position_gain, velocity_bandwidth, integral_time = sample_velocity_loop(
            config.controller, config.seed, seed,
        )
        draw = ControllerDraw(natural_frequency, damping_ratio, position_gain, velocity_bandwidth, integral_time)
        extras = plant_extras_for_bag(config.plant_extras, len(asset.joint_names), config.seed, seed)
        with payload_asset(asset, payload) as asset_p:
            report_kinematics = PortableKinematics(asset_p)
            _report_trajectory(asset, trajectory, report_kinematics)
    else:
        from elastic_sim.kinematics import PortableKinematics

        bag_config = replace(config, candidates=args.candidates or config.candidates)
        resolved = resolve_bag(bag_config, asset, tier, args.trajectory, payload)
        trajectory = resolved.trajectory
        natural_frequency, damping_ratio = resolved.natural_frequency, resolved.damping_ratio
        draw, extras = resolved.draw, resolved.extras
        # PortableKinematics is re-derived here (not reused from resolve_bag,
        # which tears its own down before returning) purely to report the
        # collision margin below; cheap relative to the trajectory search.
        with payload_asset(asset, payload) as asset_p:
            report_kinematics = PortableKinematics(asset_p)
            _report_trajectory(asset, trajectory, report_kinematics)

    if args.save_trajectory:
        trajectory.save(args.save_trajectory)
        print(f"  saved trajectory to {args.save_trajectory}")
    if args.trajectory_only:
        return

    backend = args.backend or config.backends[0]

    run_config = replace(
        config,
        visualize=bool(args.visualize) or config.visualize,
        realtime_scale=args.realtime_scale if args.realtime_scale is not None else config.realtime_scale,
        control_frequency=args.control_frequency or config.control_frequency,
    )
    if args.control_frequency is not None:
        natural_frequency = args.control_frequency
        draw = replace(draw, natural_frequency=natural_frequency)
    friction = idn.FrictionModel.from_asset(asset)
    units = _units(asset)
    print(f"\ntier {tier.name!r} on {backend}" + (" with viewer" if run_config.visualize else ""))
    print(f"  controller   : {config.controller.mode}"
          + (" (model-free)" if config.controller.is_model_free else "")
          + ("" if config.controller.nominal.knows_payload else ", payload-unaware"))
    if config.controller.mode == "velocity_pi":
        print(f"  velocity loop: kp_pos={draw.position_gain:.2f} 1/s  omega_v={draw.velocity_bandwidth:.1f} rad/s"
              f"  Ti={draw.integral_time:.3f} s")
    if extras is not None and not extras.is_empty:
        print(f"  plant extras : {extras.describe()}")
    if config.control_gains.enabled:
        print(f"  control gains: natural_frequency={natural_frequency:.2f} rad/s  damping_ratio={damping_ratio:.3f}")
    link_inertia = None
    if not tier.is_rigid:
        bounds = None
        if config.excitation.position_window:
            window_lower, window_upper = exc.effective_position_window(asset, config.excitation)
            bounds = tuple(zip(window_lower.tolist(), window_upper.tolist()))
        with payload_asset(asset, payload) as asset_p:
            link_inertia = link_inertia_envelope(asset_p, n_samples=config.transmission.inertia_samples, bounds=bounds)
            worst_link_inertia = link_inertia_max(asset_p, n_samples=config.transmission.inertia_samples, bounds=bounds)
        transmission = tier.transmission(len(asset.joint_names), link_inertia)
        _report_transmission(transmission, run_config, units)
        from elastic_sim.controllers import effective_bandwidth

        ratios = control_separation_ratio(
            transmission.stiffness, transmission.rotor_inertia, worst_link_inertia,
            effective_bandwidth(config.controller, draw),
        )
        # The bound guards SeaMotorController's feedback linearization, which a
        # model-free loop does not do; there it is reported, not wanted
        # (R5_02 Sec 2.4).
        wanted = not config.controller.is_model_free
        suffix = (f"(>= {config.control_separation.min_ratio:g} wanted)" if wanted
                  else f"(informational: {config.controller.mode} does no feedback linearization)")
        print(f"  control/transmission separation ratio {suffix}:")
        for name, ratio in zip(asset.joint_names, ratios):
            flag = "  <-- below min_ratio" if wanted and ratio < config.control_separation.min_ratio else ""
            print(f"    {name:20s} {ratio:6.2f}{flag}")
    result = run_condition(asset, trajectory, tier, backend, friction, run_config, link_inertia=link_inertia,
                           payload=payload, natural_frequency=natural_frequency, damping_ratio=damping_ratio,
                           draw=draw, extras=None if tier.is_rigid else extras)
    _report_rollout(asset, result, friction, units)

    if not args.output:
        print("\nNothing written (pass --output to save this rollout).")
        return
    frame = rollout_frame(
        asset, trajectory, result, bag=f"{tier.name}_{backend}", tier=tier,
        backend=backend, friction=friction, resample_step=config.sample_time_step,
        signals=config.signals, measurement=config.measurement, measurement_seed=config.seed,
    )
    target = Path(args.output).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    print(f"\nWrote {len(frame)} samples to {target}")


def _report_trajectory(asset, trajectory, kinematics) -> None:
    metadata = trajectory.metadata
    lower, upper, velocity = exc.joint_bounds(
        asset,
        exc.FourierExcitationConfig(
            n_harmonics=metadata["n_harmonics"], base_frequency=metadata["base_frequency"],
            n_periods=metadata["n_periods"], limit_margin=metadata["limit_margin"],
            max_acceleration=metadata["max_acceleration"],
            velocity_fraction=metadata["velocity_fraction"],
        ),
    )
    print(f"\ntrajectory : {trajectory.duration:.2f}s, {len(trajectory.time)} samples, "
          f"seed {metadata['seed']}")
    print(f"  regressor condition number : {metadata['condition_number']:.0f}"
          "   (lower excites the dynamics better)")
    print(f"  peak |dq| / limit          : "
          f"{np.round(np.abs(trajectory.velocity).max(axis=0) / velocity, 2)}")
    print(f"  peak |ddq|                 : {np.abs(trajectory.acceleration).max():.3f} "
          f"/ {metadata['max_acceleration']:.3f} {_units(asset)['ddq']}")
    print(f"  starts and ends at rest    : "
          f"{bool(np.abs(trajectory.velocity[[0, -1]]).max() < 1e-9)}")
    inside = bool((trajectory.position >= lower - 1e-9).all() and (trajectory.position <= upper + 1e-9).all())
    print(f"  inside joint-limit band    : {inside}")
    margin = float(asset.metadata.get("collision", {}).get("margin", 0.0))
    report = kinematics.validate_path(
        trajectory.position, margin=margin,
        max_joint_step=float(asset.metadata.get("collision", {}).get("max_joint_step", 0.05)),
    )
    print(f"  collision-free (margin {margin:g}) : {report.valid} "
          f"(closest {report.minimum_distance:.4f} m, {report.closest_pair})")


def _units(asset) -> dict[str, str]:
    """Unit labels for this asset's joint type (R5_02 H-7, R5_03 Sec 2.4).

    Every printed label was hard-coded to a revolute arm's rad/Nm, which is
    wrong for the FMRR gantry's three prismatic axes: there the same numbers are
    metres and newtons.  The numbers were always right; only the labels lied.
    """
    joints = asset.resolve_active_joints()
    if all(joint.joint_type == "prismatic" for joint in joints):
        return {"q": "m", "dq": "m/s", "ddq": "m/s^2", "tau": "N",
                "stiffness": "N/m", "damping": "N s/m", "rotor": "kg"}
    return {"q": "rad", "dq": "rad/s", "ddq": "rad/s^2", "tau": "Nm",
            "stiffness": "Nm/rad", "damping": "Nm s/rad", "rotor": "kg m^2"}


def _row(label: str, value: str) -> str:
    """One report line, with the label column a fixed width."""
    return f"  {label:<27}: {value}"


def _report_transmission(transmission, config, units) -> None:
    print(_row(f"stiffness [{units['stiffness']}]", f"{np.round(transmission.stiffness, 0)}"))
    print(_row("damping ratio", f"{np.round(transmission.damping_ratio, 3)}"))
    print(_row(f"damping [{units['damping']}]", f"{np.round(transmission.damping, 2)}"))
    print(_row(f"rotor inertia [{units['rotor']}]", f"{np.round(transmission.rotor_inertia, 4)}"))
    print(_row("highest mode per joint [Hz]", f"{np.round(transmission.natural_frequency(), 0)}"))
    print(_row("integration step", f"{elastic_time_step(transmission, config):.2e} s"))


def _report_rollout(asset, result, friction, units) -> None:
    q, dq, ddq = result["q_link"], result["dq_link"], result["ddq_link"]
    tau, q_ref = result["tau_motor"], result["q_ref"]
    print(f"  wall time                  : {result['wall_time']:.1f}s"
          + (f" ({result['solver']})" if "solver" in result else ""))
    print(_row("tracking RMS |q - q_ref|", f"{np.sqrt(np.mean((q - q_ref) ** 2)):.3e} {units['q']}"))
    print(_row("max |q - q_ref|", f"{np.abs(q - q_ref).max():.3e} {units['q']}"))
    ratio = np.mean(np.abs(result["tau_feedback"])) / max(np.mean(np.abs(result["tau_feedforward"])), 1e-12)
    print(f"  feedback / feedforward     : {ratio:.4f}   (small means the data is dynamics, not control)")
    limits = np.asarray([j.effort or np.inf for j in asset.resolve_active_joints()])
    print(_row(f"effort RMS per joint [{units['tau']}]",
               f"{np.round(np.sqrt(np.mean(tau ** 2, axis=0)), 2)}"))
    print(f"  peak |tau| / effort limit  : {np.round(np.abs(tau).max(axis=0) / limits, 3)}")
    if result.get("mode") == "elastic":
        deflection = np.abs(result["q_link"] - result["q_motor"])
        print(_row("max transmission deflection", f"{deflection.max():.3e} {units['q']}"))
        print(_row("RMS |tau_link - tau_motor|",
                   f"{np.sqrt(np.mean((result['tau_link'] - tau) ** 2)):.3f} {units['tau']}"
                   "  (target vs input)"))
        if not result.get("independent_of_mujoco", True):
            print("  note: this Newton run uses SolverMuJoCo and is not independent of MuJoCo")
    else:
        pin, model, data = idn.build_model(asset)
        stride = slice(None, None, max(1, len(q) // 400))
        predicted = np.asarray([
            idn.inverse_dynamics(pin, model, data, qi, dqi, ddqi, friction=friction)
            for qi, dqi, ddqi in zip(q[stride], dq[stride], ddq[stride])
        ])
        residual = tau[stride] - predicted
        print(_row("|tau - inverse dynamics|",
                   f"{np.sqrt(np.mean(residual ** 2)):.3e} {units['tau']} "
                   "(should be ~0: the label must match the model)"))


if __name__ == "__main__":
    main()
