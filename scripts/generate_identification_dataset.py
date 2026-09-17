#!/usr/bin/env python3
"""Generate a dynamic-model-identification dataset from simulation only.

Excitation trajectories are optimized for regressor conditioning, executed as
torque-driven rollouts in MuJoCo and/or Newton across a rigid tier and a
transmission-stiffness ladder, and written as one flat CSV in the schema that
``dynamic_model_nn`` consumes.

Settings come from a YAML file under ``config/identification/``; every one of
them can be overridden on the command line.  See
``scripts/run_identification_simulation.py`` to inspect or watch a single
rollout without writing a dataset.

Examples
--------
    python scripts/generate_identification_dataset.py
    python scripts/generate_identification_dataset.py --backends mujoco --trajectories 8
    python scripts/generate_identification_dataset.py --visualize --no-save
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(_REPO / "src"))

from elastic_sim.assets import AssetRegistry, load_asset_spec
from elastic_sim.dataset import DEFAULT_CONFIG, Tier, default_tiers, generate, load_config, write_dataset
from elastic_sim.excitation import FourierExcitationConfig


def _load_asset(reference: str):
    candidate = Path(reference)
    if candidate.is_file():
        return load_asset_spec(candidate)
    return AssetRegistry.for_repository(_REPO).load(reference)


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else _REPO / candidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="YAML defaults (see config/identification/)")
    parser.add_argument("--asset", default=None)
    parser.add_argument("--backends", nargs="+", choices=("mujoco", "newton"), default=None)
    parser.add_argument("--stiffness", nargs="*", type=float, default=None,
                        help="Transmission stiffness ladder; pass with no values for rigid only")
    parser.add_argument("--no-rigid", action="store_true", help="Omit the rigid reference tier")
    parser.add_argument("--trajectories", type=int, default=None)
    parser.add_argument("--friction-samples", type=int, default=None)
    parser.add_argument("--friction-scale", nargs=2, type=float, default=None)
    parser.add_argument("--harmonics", type=int, default=None)
    parser.add_argument("--base-frequency", type=float, default=None)
    parser.add_argument("--periods", type=int, default=None)
    parser.add_argument("--max-acceleration", type=float, default=None)
    parser.add_argument("--candidates", type=int, default=None)
    parser.add_argument("--control-frequency", type=float, default=None)
    parser.add_argument("--sample-step", type=float, default=None, help="Output sampling step [s]")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-save", action="store_true",
                        help="Run everything but write nothing, for debugging")
    parser.add_argument("--visualize", action="store_true", help="Open the native viewer for each rollout")
    parser.add_argument("--realtime-scale", type=float, default=None, help="1.0 is real time")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    config = load_config(_resolve(args.config))

    tiers = config.tiers
    if args.stiffness is not None:
        tiers = default_tiers(args.stiffness)
        template = next((tier for tier in config.tiers if not tier.is_rigid), None)
        if template is not None:
            tiers = tuple(
                tier if tier.is_rigid else Tier(
                    tier.name, tier.stiffness,
                    transmission_damping_ratio=template.transmission_damping_ratio,
                    rotor_inertia=template.rotor_inertia,
                )
                for tier in tiers
            )
    if args.no_rigid:
        tiers = tuple(tier for tier in tiers if not tier.is_rigid)
    if not tiers:
        parser.error("no tiers selected: keep the rigid tier or pass --stiffness values")

    excitation = config.excitation
    sample_step = args.sample_step or config.sample_time_step
    excitation = FourierExcitationConfig(
        n_harmonics=args.harmonics or excitation.n_harmonics,
        base_frequency=args.base_frequency or excitation.base_frequency,
        n_periods=args.periods or excitation.n_periods,
        time_step=sample_step,
        limit_margin=excitation.limit_margin,
        max_acceleration=args.max_acceleration or excitation.max_acceleration,
        velocity_fraction=excitation.velocity_fraction,
    )
    config = replace(
        config,
        asset=args.asset or config.asset,
        backends=tuple(args.backends) if args.backends else config.backends,
        tiers=tiers,
        n_trajectories=args.trajectories or config.n_trajectories,
        n_friction_samples=args.friction_samples or config.n_friction_samples,
        friction_scale_range=(tuple(args.friction_scale) if args.friction_scale
                              else config.friction_scale_range),
        seed=config.seed if args.seed is None else args.seed,
        excitation=excitation,
        candidates=args.candidates or config.candidates,
        control_frequency=args.control_frequency or config.control_frequency,
        sample_time_step=sample_step,
        output=args.output or config.output,
        visualize=bool(args.visualize) or config.visualize,
        realtime_scale=(args.realtime_scale if args.realtime_scale is not None
                        else config.realtime_scale),
    )
    if config.n_trajectories < 1 or config.n_friction_samples < 1:
        parser.error("trajectories and friction-samples must be positive")

    asset = _load_asset(config.asset)
    asset.resolve_active_joints()
    asset.validate_resources()
    if not args.quiet:
        print(f"asset {asset.name}: tiers {[t.name for t in config.tiers]} "
              f"x backends {list(config.backends)} x {config.n_trajectories} trajectories "
              f"x {config.n_friction_samples} friction samples")

    frame, manifest = generate(config, asset, verbose=not args.quiet)
    if args.no_save:
        print(f"\n{len(frame)} samples in {manifest['n_bags']} bags; nothing written (--no-save).")
        return
    csv_path, manifest_path = write_dataset(frame, manifest, _resolve(config.output))
    print(f"\nWrote {len(frame)} samples in {manifest['n_bags']} bags to {csv_path}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
